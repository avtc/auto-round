# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Algorithm Fusion Pipeline abstraction.

This module defines the core data structures and utilities for composing
pre-processing algorithms (AWQ smooth, SmoothQuant, Rotation…) with a
block-quantization algorithm (RTN, SignRound/AutoRound…) into a single
shared-calibration bundle.

Design invariants (see AWQ_REFACTOR_PLAN.md §0.0 and §3.0):
- ``AlgorithmComposer`` is the *first-class abstraction*; it is NOT just
  AWQ's helper.
- All block-wise scheduling in ``Compressor`` operates against
  ``AlgorithmComposer``, never against a concrete ``AWQTransform``.
- Single-algorithm use is expressed as
  ``AlgorithmComposer(preprocessors=[], block_quantizer=q)``, which is
  semantically identical to the current direct-quantizer path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import threading

import torch
import transformers  # noqa: I001  (bias-correction target types)

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.config_resolver import (
    get_algorithm_class,
    resolve_shared_config_values,
    split_quantization_configs,
)
from auto_round.algorithms.utils import _has_nvfp4_layer
from auto_round.logger import logger
from auto_round.utils import clear_memory
from auto_round.utils.device_manager import device_manager


def _stream_out_device(block):
    """Park collected calibration rows on the producing block's home.

    BlockForwardRunner parks returned rows on its fixed cache_device (the
    primary GPU); under streaming with round-robin homes every non-primary
    block would then pay a cross-device toll per tuning iteration. When the
    streaming loop has stamped a home device, collect there instead. Returns
    None for non-streaming runs (runner default, unchanged behavior).
    """
    home = getattr(block, "_stream_home_device", None)
    return torch.device(home) if home is not None else None


def _as_hidden_tensor(out):
    """Normalize a block output (tensor | dict | tuple) to its hidden-states tensor."""
    import torch as _torch

    if isinstance(out, dict):
        for key in ("hidden_states", "last_hidden_state"):
            if key in out:
                return out[key]
        for v in out.values():
            if isinstance(v, _torch.Tensor):
                return v
        raise ValueError("bias correction: block output dict has no tensor")
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _iter_hidden_tensors(out):
    """Yield per-sample hidden tensors from a collection output.

    The collection contract returns a per-sample LIST (text) or a dict/tuple
    wrapping one (diffusion / multi-output). Every yielded tensor keeps its
    leading sample/batch dims; consumers reduce over them.
    """
    import torch as _torch

    if isinstance(out, (tuple, list)) and out and isinstance(out[0], _torch.Tensor):
        yield from out
    else:
        yield _as_hidden_tensor(out)


def _apply_block_bias_correction(block, y_fp, y_q) -> bool:
    """Absorb b = mean(y_fp - y_q) into the block's residual-feeding projection.

    The correction restores the block's output mean exactly at its boundary
    (the residual stream), where downstream layers and the export stack read
    it. The sink is the LAST Linear/Conv1D whose output width matches the
    hidden size (typically ``mlp.down_proj`` / shared-expert down projection);
    modules on routed-expert paths are deprioritized since they execute on a
    token subset only. Bias is created as zeros when absent (native in all
    export formats).
    """
    import torch.nn as nn

    # b = mean over ALL calibration rows of (y_fp - y_q): stream the per-sample
    # list (fp32 accumulators, no full cat -- VRAM stays one sample's output).
    # Under a DDP tune the pool may be scattered across replica devices (the
    # tune distributes it in place), so per-pair parts can land on different
    # devices: pin the accumulator to the block's device (also where the bias
    # will be applied) and move each part before adding.
    _acc_dev = next(block.parameters(), torch.empty(0)).device
    b = None
    rows = 0
    for y_fp_t, y_q_t in zip(_iter_hidden_tensors(y_fp), _iter_hidden_tensors(y_q)):
        y_fp_t = y_fp_t.detach()
        y_q_t = y_q_t.detach()
        y_q_t = y_q_t.to(y_fp_t.device)  # distributed pool: pair may straddle devices
        d = y_fp_t.to(torch.float32) - y_q_t.to(torch.float32)
        part = d.sum(dim=tuple(range(d.dim() - 1))).to(_acc_dev)
        b = part if b is None else b + part
        rows += d.numel() // d.shape[-1]
    if b is None:
        logger.warning("[bias_correct] no rows collected; correction skipped")
        return False
    b = b / rows

    def _out_features(mod):
        # nn.Linear weight is [out, in]; Conv1D weight is [in, out]
        return mod.weight.shape[1] if isinstance(mod, transformers.pytorch_utils.Conv1D) else mod.weight.shape[0]

    targets = [
        (name, mod)
        for name, mod in block.named_modules()
        if isinstance(mod, (nn.Linear, transformers.pytorch_utils.Conv1D)) and _out_features(mod) == b.numel()
    ]
    non_expert = [(n, m) for n, m in targets if ".experts." not in n and "experts." not in n]
    pool = non_expert or targets
    if not pool:
        logger.warning("[bias_correct] no projection with out_features==hidden found; correction skipped")
        return False
    name, mod = pool[-1]
    dtype = mod.weight.dtype
    if getattr(mod, "bias", None) is not None:
        mod.bias = nn.Parameter((mod.bias.data.to(torch.float32) + b).to(dtype))
    else:
        mod.bias = nn.Parameter(b.to(dtype))
    logger.info("[bias_correct] absorbed block output drift (mean |b| %.3e) into %s", b.abs().mean().item(), name)
    return True


def _collect_qoff_noise_stats_from_outputs(y_fp, y_q, path):
    """Per-channel stats of the quantization noise (y_fp - y_q) for one block.

    Consumes block outputs the pipeline already computed (no extra forward).
    Writes ``{mean, var}`` CPU tensors [hidden] to ``path`` (created with
    parents). Returns (mean, var).
    """
    import os as _os

    # stream all per-sample rows; fp64 accumulators keep the variance stable
    # over the full 262k-row pool (E[x^2] - E[x]^2 cancellation)
    sum_d = None
    sumsq_d = None
    rows = 0
    for y_fp_t, y_q_t in zip(_iter_hidden_tensors(y_fp), _iter_hidden_tensors(y_q)):
        y_q_t = y_q_t.detach().to(y_fp_t.device)  # distributed pool pair alignment
        d = (y_fp_t.detach().to(torch.float64) - y_q_t.detach().to(torch.float64)).reshape(-1, y_fp_t.shape[-1])
        part = d.sum(dim=0)
        partsq = d.pow(2).sum(dim=0)
        sum_d = part if sum_d is None else sum_d + part
        sumsq_d = partsq if sumsq_d is None else sumsq_d + partsq
        rows += d.shape[0]
    if sum_d is None:
        raise ValueError("qoff noise stats: no rows collected")
    mean = (sum_d / rows).float().cpu()
    var = (sumsq_d / rows - (sum_d / rows).pow(2)).float().cpu()
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    torch.save({"mean": mean, "var": var}, path)
    return mean, var


if TYPE_CHECKING:  # avoid circular imports at runtime
    from auto_round.algorithms.quantization.base import BaseQuantizer
    from auto_round.algorithms.quantization.config import QuantizationConfig
    from auto_round.algorithms.transforms.base import BasePreprocessor
    from auto_round.compressors import BaseOrchestrator


# ---------------------------------------------------------------------------
# Context dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BlockContext:
    """Per-block context threaded through the lifecycle hooks.

    Passed to lifecycle methods like ``block_forward_hooks``,
    ``pre_quantize_block``, ``post_quantize_block``, etc.

    ``block_names`` preserves the *scheduling group* (which may contain more
    than one block when ``nblocks > 1``).  Pre-processing algorithms that
    only support single-block operation (e.g. AWQ Phase-1) must check
    ``len(block_names) == 1`` in ``prepare_block_group`` and raise
    ``ValueError`` with a user-readable message.
    """

    model: "torch.nn.Module"
    block_names: list[str]  # scheduling group; len > 1 when nblocks > 1
    block_name: str  # = block_names[0] for single-block; descriptive label for multi
    block_index: int  # 0-based index within the current all_blocks group
    bs: int = 1
    is_mllm: bool = False  # fail-fast gate for algorithms that don't support MLLM
    is_diffusion: bool = False  # fail-fast gate for algorithms that don't support diffusion
    pbar: Any = None
    block_cnt: int = 0  # total number of blocks being quantized in this run


# ---------------------------------------------------------------------------
# AlgorithmComposer
# ---------------------------------------------------------------------------
def _can_compile_block_forward(block_quantizer, rotation_configs, user_enabled: bool) -> bool:
    """Return whether every component participating in block replay supports compilation."""
    if not user_enabled or not block_quantizer.can_compile_block_forward():
        return False
    return all(getattr(config, "can_compile_block_forward", lambda: True)() for config in rotation_configs)


class AlgorithmComposer:
    """An ordered composition of pre-processors + one block quantizer, built from
    a list of algorithm config objects and an optional compressor.

    The ``preprocessors`` list is order-sensitive: algorithms are applied in
    the listed order (e.g. ``[Rotation, AWQ]``).  There must be **exactly one**
    ``block_quantizer`` (the terminal weight-compression step).

    Usage::

        composer = AlgorithmComposer(configs, compressor=self)
    """

    def __init__(self, configs: list, orchestrator: "BaseOrchestrator" = None) -> None:
        """Build the pipeline from a list of algorithm config instances.

        Resolution rules:

        1. If no ``QuantizationConfig`` with a ``BaseQuantizer`` is found in
           *configs*, a default :class:`RTNConfig` is appended automatically.
        2. ``BasePreprocessor`` instances go into ``preprocessors`` (in order).
        3. Exactly one ``BaseQuantizer`` becomes ``block_quantizer``.
        4. Multiple block-quantization configs raise ``ValueError``.
        5. If *compressor* is provided, every member is bound to it and
           ``block_forward`` / quantization metadata are extracted from it.

        Args:
            configs:    List of algorithm config objects (``QuantizationConfig``
                        subclasses such as :class:`RTNConfig`, :class:`SignRoundConfig`,
                        :class:`AWQConfig`, …).
            compressor: The :class:`~auto_round.compressors.base.BaseCompressor`
                        instance driving this pipeline.  When supplied, members are
                        bound to it and ``block_forward`` is taken from it.
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer
        from auto_round.algorithms.quantization.config import QuantizationConfig
        from auto_round.algorithms.transforms.base import BasePreprocessor, BaseRotationConfig

        configs = list(configs)

        # Rotation configs travel in the same config list but are not pipeline
        # members (they are ``BaseRotationConfig``, not ``QuantizationConfig``).
        # Capture them here so the composer owns the full rotation lifecycle
        # (see ``apply_model_transforms`` / ``finalize_run``) and the orchestrator
        # stays rotation-agnostic.
        self._rotation_configs = [c for c in configs if isinstance(c, BaseRotationConfig)]
        self._layerwise_rotation = bool(getattr(orchestrator, "layerwise_rotation", False))
        self._orchestrator_ref = orchestrator

        _algos = {getattr(c, "algorithm", None) for c in self._rotation_configs}
        if "presinq" in _algos and "spinquant" in _algos:
            # Both transforms modify norm weights; their composition is untested
            # and easy to get silently wrong — require an explicit choice.
            raise ValueError(
                "PreSINQ and SpinQuant transforms are mutually exclusive: both "
                "modify RMSNorm weights. Pass only one of them in alg_configs."
            )

        _, block_quantizer_configs = split_quantization_configs(configs)
        if not block_quantizer_configs:
            from auto_round.algorithms.quantization.rtn.config import RTNConfig

            configs = configs + [RTNConfig()]

        configs = resolve_shared_config_values(configs)

        preprocessors: list = []
        block_quantizers: list = []

        for cfg in configs:
            if not isinstance(cfg, QuantizationConfig):
                continue
            from auto_round.algorithms.registry import normalize_algorithm_config

            cfg = normalize_algorithm_config(cfg)
            alg_cls = get_algorithm_class(cfg)
            if alg_cls is None:
                raise ValueError(f"Unknown algorithm config type {type(cfg).__name__!r}.")
            q = alg_cls(cfg)
            if orchestrator is not None:
                q.bind(orchestrator)
            if isinstance(q, BasePreprocessor):
                preprocessors.append(q)
            elif isinstance(q, BaseQuantizer):
                block_quantizers.append(q)
            else:
                raise TypeError(
                    f"Algorithm class {type(q).__name__} must inherit either " "BasePreprocessor or BaseQuantizer."
                )

        if len(block_quantizers) > 1:
            raise ValueError(
                f"AlgorithmComposer allows exactly one block-quantization config, "
                f"but got {len(block_quantizers)}: "
                f"{[type(q).__name__ for q in block_quantizers]}. "
                "Ensure only one of RTNConfig / SignRoundConfig / etc. is in the pipeline."
            )
        if len(block_quantizers) == 0:
            raise ValueError("No block quantizer found in configs.")

        seen = set()
        for pre in preprocessors:
            name = type(pre).__name__
            if name in seen:
                raise ValueError(
                    f"Duplicate preprocessor {name} in AlgorithmComposer. "
                    "Repeated instances of the same preprocessor are not supported yet."
                )
            seen.add(name)

        self.preprocessors = preprocessors

        # TODO wenhuach support multi quantizers
        can_compile_block_forward = False
        self.block_quantizer = block_quantizers[0]
        if orchestrator is not None:
            user_torch_compile = bool(
                getattr(getattr(orchestrator, "compress_context", None), "enable_torch_compile", False)
            )
            rotation_configs = getattr(orchestrator, "rotation_configs", ())
            can_compile_block_forward = _can_compile_block_forward(
                self.block_quantizer, rotation_configs, user_torch_compile
            )
            if (
                user_torch_compile
                and not can_compile_block_forward
                and any(not getattr(config, "can_compile_block_forward", lambda: True)() for config in rotation_configs)
            ):
                logger.info("Block-forward torch.compile is disabled because an enabled rotation is incompatible.")

            if _has_nvfp4_layer(orchestrator):
                can_compile_block_forward = False
                logger.info("Block-forward torch.compile is disabled because at least one quantized layer uses NVFP4.")

            # Bind compressor-level infrastructure (set before _build_quantizer is called).
            self.block_forward = (
                BlockForwardRunner.from_orchestrator(orchestrator, enable_torch_compile=can_compile_block_forward)
                if orchestrator is not None
                else None
            )
            # A little tricky
            if self.block_quantizer is not None:
                self.block_quantizer.bind_block_forward_runner(self.block_forward)
        self.scheme = getattr(orchestrator, "scheme_context", None)

        # Rotation lifecycle state (populated by apply_model_transforms)
        self._rotation_transforms: list = []
        self._rotation_prepared: bool = False
        # Serializes layer-wise rotate_layer calls: transforms keep per-weight
        # registries (e.g. PreSINQ's shadow scales) that are cleared at the end
        # of each layer, so the background early-transform thread and the
        # in-loop ready stage must never run concurrently on the same object.
        self._ready_transform_lock = threading.Lock()

    # ── Internal hook helpers (act_max calibration) ───────────────────────────

    def _register_act_max_hooks(self, block: "torch.nn.Module") -> list:
        """Register per-module act_max tracking hooks for static activation quantization.

        Returns a list of hook handles that the caller must remove when done.
        """
        from auto_round.compressors.utils import is_nv_fp
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size

        is_act_nv_fp = getattr(self.block_quantizer.config, "is_act_nv_fp", False)

        def collect_act_max(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            if input.numel() == 0:
                return
            module_act_data_type = getattr(module, "act_data_type", None) or getattr(module, "data_type", None)
            is_module_act_nv_fp = is_nv_fp(module_act_data_type) if module_act_data_type else is_act_nv_fp
            input, _, _ = reshape_pad_tensor_by_group_size(input, module.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if is_module_act_nv_fp:
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                return
            act_max = act_max.to(module.act_max.device)
            if is_module_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

        def should_collect(name, module):
            from auto_round.compressors.utils import check_need_act_calibration
            from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized

            if isinstance(module, tuple(SUPPORTED_LAYER_TYPES)):
                return (
                    hasattr(module, "act_dynamic")
                    and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
                    and check_to_quantized(module)
                )
            if hasattr(module, "bits"):
                act_dynamic = getattr(module, "act_dynamic", True)
                act_data_type = getattr(module, "act_data_type", None)
                act_bits = getattr(module, "act_bits", 16)
                return (
                    module.bits <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(module)
                )
            return False

        handles = []
        if should_collect("", block):
            handles.append(block.register_forward_hook(collect_act_max))
            return handles
        for name, module in block.named_modules():
            if name and should_collect(name, module):
                handles.append(module.register_forward_hook(collect_act_max))
        return handles

    def _get_fp_act_hooks(self, block: "torch.nn.Module") -> list:
        """Register FP-input act_max + quantizer forward hooks."""
        if not self.need_quanted_input():
            # If having q_input, act_max will be collected in q_input forward hook,
            # no need to collect in fp_input forward hook
            handles = self._register_act_max_hooks(block)
        else:
            handles = []
        handles.extend(self.block_quantizer.register_fp_input_forward_hooks(block))
        return handles

    def _get_q_act_hooks(self, block: "torch.nn.Module") -> list:
        """Register Q-input act_max + quantizer forward hooks."""
        handles = self._register_act_max_hooks(block)
        handles.extend(self.block_quantizer.register_qinput_forward_hooks(block))
        return handles

    def _attach_act_max_for_outside_layer(self, layer: "torch.nn.Module", fp_inputs, q_inputs) -> None:
        """Compute and attach act_max for an outside-block layer from cached inputs.

        Mirrors the hook-based act_max collection done for in-block layers, but
        iterates over already-cached tensors directly instead of running a forward pass.

        Args:
            layer: The layer module to attach act_max to.
            fp_inputs: List of FP input tensors collected during calibration.
            q_inputs: Optional list of quantized input tensors; used instead of
                ``fp_inputs`` when provided.
        """
        from auto_round.compressors.utils import is_nv_fp
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size

        target_input = q_inputs or fp_inputs
        act_group_size = getattr(layer, "act_group_size")
        if act_group_size is None:
            act_group_size = layer.group_size
        act_data_type = getattr(layer, "act_data_type")
        if act_data_type is None:
            act_data_type = layer.data_type
        is_act_nv_fp_flag = is_nv_fp(act_data_type) if act_data_type else False

        for inp in target_input:
            if isinstance(inp, (tuple, list)):
                inp = inp[0]
            if inp.numel() == 0:
                continue
            inp, _, _ = reshape_pad_tensor_by_group_size(inp, act_group_size)
            act_max = torch.max(torch.abs(inp), dim=-1).values

            if not hasattr(layer, "act_max") or layer.act_max.numel() == 0:
                layer.act_max = act_max
                if is_act_nv_fp_flag:
                    max_val = act_max.max()
                    layer.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                continue

            act_max = act_max.to(layer.act_max.device)
            if is_act_nv_fp_flag:
                max_val = torch.max(act_max.max(), layer.act_max.max())
                layer.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                layer.act_max = torch.max(act_max, layer.act_max)

    def need_quanted_input(self):
        for preprocessor in self.preprocessors:
            if getattr(preprocessor, "enable_quanted_input", False):
                return True
        if getattr(self.block_quantizer, "enable_quanted_input", False):
            return True
        return False

    def compress_embedding_layer(self):
        return self.block_quantizer.quantize_embedding_layer()

    def _ddp_collection_devices(self, block, fp_inputs):
        """Mirror devices for sharding the no-grad collection forwards.

        Engages only when AR_TUNE_DDP_WORLD is active for this block's home
        (CUDA, stamped streaming home, pool divisible into equal shards) and
        the runner uses the plain tensor output layout. Returns None to keep
        the serial single-GPU collection.
        """
        from auto_round import envs as _envs

        world = int(getattr(_envs, "AR_TUNE_DDP_WORLD", 1) or 1)
        if world <= 1 or not isinstance(fp_inputs, list) or not fp_inputs:
            return None
        home = getattr(block, "_stream_home_device", None)
        if home is None or home.type != "cuda":
            return None
        runner_cfg = getattr(self.block_forward, "output_config", None)
        if runner_cfg and list(runner_cfg) != ["hidden_states"]:
            return None  # multi-output layouts need the serial last_output_dict path
        n = len(fp_inputs)
        if n < world or n % world != 0:
            return None
        try:
            from auto_round.algorithms.quantization.sign_round.data_parallel import (
                effective_ddp_groups,
                resolve_ddp_plan,
            )

            plan = resolve_ddp_plan(
                world,
                home,
                n,
                visible_cuda_devices=list(range(torch.cuda.device_count())),
                groups=effective_ddp_groups(),
            )
        except Exception:  # pragma: no cover - CUDA reachability on odd hosts
            return None
        return plan.devices if plan.world > 1 else None

    @staticmethod
    def _merge_stats_on() -> bool:
        from auto_round import envs as _envs

        return bool(getattr(_envs, "AR_TUNE_DDP_MERGE_STATS", False))

    def _collect_forward(self, block, inputs, input_others, out_dev, allow_shard: bool = True):
        """Collection forward: sharded across DDP mirrors when eligible.

        ``allow_shard=False`` forces the serial path: passes that carry
        forward hooks (act_max / fp-input / q-input stats) must not shard --
        hook writes land on the ephemeral mirror copies and are freed with
        them, silently dropping that shard's statistics. Exception:
        AR_TUNE_DDP_MERGE_STATS=1 folds the mergeable stats (imatrix, act_max)
        from the mirrors back into the home, so hook passes may shard.
        """
        if self._coll_devs is None or not allow_shard:
            return self.block_forward(block, inputs, input_others, cache_device=out_dev)
        from auto_round import envs as _envs
        from auto_round.algorithms.quantization.sign_round.data_parallel import sharded_nograd_forward

        # hook-carrying passes may shard ONLY under the merge opt-in: mirror-side
        # hook stats are folded back into the home by _merge_mirror_stats
        _merge = bool(getattr(_envs, "AR_TUNE_DDP_MERGE_STATS", False))
        return sharded_nograd_forward(
            self.block_forward, block, inputs, input_others, out_dev, self._coll_devs, merge_stats=_merge
        )

    # ── Per-block pipeline orchestration ─────────────────────────────────────

    def compress_block(
        self,
        block,
        fp_inputs,
        input_others,
        block_ctx: BlockContext,
        q_inputs=None,
        input_ids=None,
        **kwargs,
    ) -> tuple:
        """Run the full per-block algorithm pipeline: calibration → quantization → collection.

        Orchestrates preprocessors, quantizer calibration, and block quantization in
        the canonical order.  Infrastructure concerns (device management, memory cleanup,
        q_input variable reassignment) remain the caller's responsibility.

        The interface mirrors :meth:`~auto_round.algorithms.quantization.base.BaseQuantizer.quantize_block`
        but covers the entire multi-step pipeline including preprocessor calibration and
        output collection.

        Args:
            block: The transformer block to process.
            input_ids: Full-precision (FP) cached inputs.
            input_others: Auxiliary kwargs (attention_mask, position_ids, …).
            ctx: Per-block lifecycle context (:class:`BlockContext`).
            q_inputs: Quantized-input tensors from the previous block, or ``None``.
            valid_token_mask: Optional mask for per-token loss weighting.

        Returns:
            ``(new_q_input, reference_output)``:

            - *new_q_input*: block output after quantization (``None`` when
              ``enable_quanted_input`` is ``False``).
            - *reference_output*: FP reference output collected before optimization.
        """
        block_forward_fn = self.block_forward
        _out_dev = _stream_out_device(block)
        self._coll_devs = self._ddp_collection_devices(block, fp_inputs)
        if self._coll_devs:
            # distributed calibration pool: each DDP device owns its sample
            # shard (matching the tune shards), so shard-local reads never
            # cross devices; serial consumers use device-safe cats
            from auto_round.algorithms.quantization.sign_round.data_parallel import distribute_pool

            if isinstance(fp_inputs, list) and fp_inputs:
                distribute_pool(fp_inputs, self._coll_devs)
            if isinstance(q_inputs, list) and q_inputs:
                distribute_pool(q_inputs, self._coll_devs)
        import time as _time

        from auto_round.utils.tune_profile import make_tune_profiler
        from auto_round.utils.tune_profile import stage as _prof_stage

        _prof_dev = _out_dev
        if _prof_dev is None:
            _p = next(block.parameters(), None)
            _prof_dev = _p.device if _p is not None else torch.device("cpu")
        _prof = make_tune_profiler(_prof_dev)
        _prof_t0 = _time.perf_counter() if _prof is not None else None

        # ── Step 0: Layer-wise rotation (before any reference/calibration) ────
        # Rotates this block's weights and installs online hooks so all downstream
        # calibration and reference collection operate on the rotated block. No-op
        # unless layer-wise rotation is active.
        # staged-copy adoption gate: prefetch replicas hold the RAW checkpoint
        # weights; any in-place weight mutation (AWQ clip/scale, layer-wise
        # rotation) makes them stale -> mirrors must deep-copy from home instead
        block._stream_weights_pristine = not self.preprocessors and not getattr(self, "_rotation_transforms", None)
        with _prof_stage(_prof, "ready"):
            if getattr(block, "_bg_ready_done", False):
                # the streaming orchestrator's background thread already loaded
                # and transformed this block on its home device
                logger.debug("[ready] %s: transforms pre-applied by the background thread", block_ctx.block_name)
            else:
                with self._ready_transform_lock:
                    self._run_block_ready_transforms(block, block_ctx)

        # ── Step 1: Preprocessor calibration (e.g. AWQ activation stats) ──────
        with _prof_stage(_prof, "pre_calib"), torch.no_grad():
            pre_hooks = []
            for pre in self.preprocessors:
                pre_hooks.extend(pre.register_fp_input_forward_hooks(block))
            if pre_hooks:
                block_forward_fn(block, fp_inputs, input_others, cache_device=_out_dev)
            for h in pre_hooks:
                h.remove()

            pre_q_hooks = []
            for pre in self.preprocessors:
                if hasattr(pre, "register_qinput_forward_hooks"):
                    pre_q_hooks.extend(pre.register_qinput_forward_hooks(block))
            if pre_q_hooks:
                block_forward_fn(
                    block, q_inputs if q_inputs is not None else fp_inputs, input_others, cache_device=_out_dev
                )
            for h in pre_q_hooks:
                h.remove()

        # ── Step 2: pre_quantize_block (stats consolidation + weight transforms) ──
        with _prof_stage(_prof, "pre_quantize"):
            for pre in self.preprocessors:
                pre.pre_quantize_block(block_ctx)

        reference_output = None
        reference_next_input = None
        # ── Step 3: Quantizer calibration (act_max, imatrix, etc.) ─────────────
        if fp_inputs is not None:
            with _prof_stage(_prof, "ref_collect"), torch.no_grad():
                quant_hooks = self._get_fp_act_hooks(block)
                reference_output = self._collect_forward(
                    block, fp_inputs, input_others, _out_dev, allow_shard=not quant_hooks or self._merge_stats_on()
                )
                reference_next_input = getattr(block_forward_fn, "last_output_dict", None) or reference_output
                for h in quant_hooks:
                    h.remove()

                if self.block_quantizer.enable_quanted_input:
                    q_hooks = self._get_q_act_hooks(block)
                    if q_hooks:
                        self._collect_forward(
                            block,
                            q_inputs if q_inputs is not None else fp_inputs,
                            input_others,
                            _out_dev,
                            allow_shard=not q_hooks or self._merge_stats_on(),
                        )
                        for h in q_hooks:
                            h.remove()

            # ── Step 3.5: MoE scale alignment + global scale update ─────────────────
            # Must run after calibration hooks (act_max collected) and before quantize_block.
            act_dynamic = self.scheme.act_dynamic if (self.scheme and self.scheme.act_dynamic is not None) else True
            data_type = self.scheme.data_type if self.scheme else "int"
            group_size = self.scheme.group_size if self.scheme else -1
            act_data_type = self.scheme.act_data_type if self.scheme else data_type
            if act_data_type is not None or not act_dynamic:
                from auto_round.compressors.utils import is_nv_fp
                from auto_round.data_type.utils import update_block_global_scale_if_needed
                from auto_round.utils import set_amax_for_all_moe_layers

                if is_nv_fp(act_data_type) or not act_dynamic:
                    set_amax_for_all_moe_layers(block, attr_name="act_max")
                update_block_global_scale_if_needed(block, data_type, group_size)

        if q_inputs is not None and fp_inputs is not q_inputs:
            clear_memory(fp_inputs)
        else:
            clear_memory()
        # ── Step 4: quantize_block ──────────────────────────────────────────────
        # When quantized input is available from the previous block, use it;
        # otherwise fall back to the FP input.
        effective_input = q_inputs if q_inputs is not None else fp_inputs
        with _prof_stage(_prof, "tune"):
            self.block_quantizer.quantize_block(
                block,
                effective_input,
                input_others,
                reference_output,
                q_inputs,
                block_ctx,
                input_ids=input_ids,
                _block_prof=_prof,
            )

        # ── Steps 5+6: post_quantize_block + collect quantized-block outputs ──
        with _prof_stage(_prof, "post"):
            for pre in self.preprocessors:
                pre.post_quantize_block(block_ctx)
        # ── Step 6: Collect quantized-block outputs for the next block ──────────
        with _prof_stage(_prof, "post_collect"):
            from auto_round import envs as _envs

            if _envs.AR_QOFF_NOISE_STATS and len(block_ctx.block_names) > 1:
                raise ValueError(
                    "AR_QOFF_NOISE_STATS collects one stats file per compress_block call; with "
                    f"nblocks={len(block_ctx.block_names)} a call covers several blocks and the "
                    "per-block file contract breaks. Rerun with nblocks=1."
                )
            if _envs.AR_BIAS_CORRECT and _envs.AR_BLOCK_PARALLEL_WORKER:
                raise ValueError(
                    "AR_BIAS_CORRECT mutates module biases, which are not part of the worker result files -- "
                    "the correction would be silently lost. It is a serial/streaming-only pass."
                )
            if self.block_quantizer.enable_quanted_input:
                with torch.no_grad():
                    new_q_input = self._collect_forward(block, effective_input, input_others, _out_dev)
                    new_q_input = getattr(block_forward_fn, "last_output_dict", None) or new_q_input
                if reference_next_input is not None:
                    # stats BEFORE bias correction: the correction absorbs the
                    # mean by construction, so post-correction stats would be ~0
                    if _envs.AR_QOFF_NOISE_STATS:
                        with _prof_stage(_prof, "noise"):
                            _collect_qoff_noise_stats_from_outputs(
                                reference_next_input,
                                new_q_input,
                                f"{_envs.AR_QOFF_NOISE_STATS}/block_{block_ctx.block_index:04d}.pt",
                            )
                    if _envs.AR_BIAS_CORRECT:
                        with _prof_stage(_prof, "bias"):
                            _apply_block_bias_correction(block, reference_next_input, new_q_input)
            else:
                new_q_input = None
                _stats_y_q = None
                if reference_next_input is not None:
                    if _envs.AR_QOFF_NOISE_STATS or _envs.AR_BIAS_CORRECT:
                        with torch.no_grad():
                            _stats_y_q = block_forward_fn(block, effective_input, input_others, cache_device=_out_dev)
                    if _envs.AR_QOFF_NOISE_STATS:
                        _collect_qoff_noise_stats_from_outputs(
                            reference_next_input,
                            _stats_y_q,
                            f"{_envs.AR_QOFF_NOISE_STATS}/block_{block_ctx.block_index:04d}.pt",
                        )
                    if _envs.AR_BIAS_CORRECT:
                        _apply_block_bias_correction(block, reference_next_input, _stats_y_q)

        if _prof is not None:
            _prof.log_summary(
                block_name=getattr(block_ctx, "block_name", ""),
                iters_done=1,
                wall=_time.perf_counter() - _prof_t0,
                prefix="compress-profile",
            )
        return new_q_input, reference_next_input

    def compress_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_inputs=None,
        q_inputs=None,
        disable_opt_rtn=None,  # TODO wenhuach rename this to search_init_scale
        input_ids=None,
    ) -> None:
        """Quantize a single layer that lives outside transformer blocks.

        Mirrors :meth:`compress_block` for the outside-block case: attaches
        act_max calibration when static activation quantization is required,
        then delegates to the block quantizer.

        Args:
            layer: The layer module to quantize. Must have a ``global_name``
                attribute.
            fp_inputs: Per-sample FP activations for calibration, or ``None``
                to fall back to zero-shot RTN.
            q_inputs: Optional quantized activations from a previous stage.
            valid_token_mask: Per-sample token masks for loss weighting.
            disable_opt_rtn: Override optimized-RTN; ``None`` defers to quantizer config.
        """
        # Attach act_max for static activation quantization when inputs are available.
        if fp_inputs is not None:
            from auto_round.compressors.utils import is_nv_fp

            act_data_type = getattr(layer, "act_data_type")
            if act_data_type is None:
                act_data_type = "fp"
            act_dynamic = getattr(layer, "act_dynamic", True)
            if is_nv_fp(act_data_type) or not act_dynamic:
                self._attach_act_max_for_outside_layer(layer, fp_inputs, q_inputs)

        # Infrastructure: move layer to the tuning device before handing off to the quantizer.
        device = getattr(layer, "tuning_device", device_manager.device)  # TODO this should be handled by compressor
        layer = layer.to(device)

        return self.block_quantizer.quantize_layer_outside_block(
            layer,
            fp_inputs=fp_inputs,
            q_inputs=q_inputs,
            disable_opt_rtn=disable_opt_rtn,
            input_ids=input_ids,
        )

    # ── Convenience act-calib helpers ────────────────────────────────────────

    def members(self) -> list:
        """Return all algorithm members: preprocessors followed by the block quantizer."""
        return list(self.preprocessors) + [self.block_quantizer]

    def dispatch_block(self, block: "torch.nn.Module", input_ids, input_others: dict):
        """Dispatch block to device(s) via the pipeline's algorithms.

        Iterates all members; if exactly one overrides the default dispatch_block,
        it is called. If multiple override, warns and uses the first one only.
        If none override, uses the block_quantizer's default (simple .to(device)).
        """
        from auto_round.algorithms.quantization.base import BaseQuantizer

        overriders = []
        for member in self.members():
            if not hasattr(member, "dispatch_block"):
                continue
            # Check if the member overrides the base default
            if type(member).dispatch_block is not BaseQuantizer.dispatch_block:
                overriders.append(member)

        if len(overriders) > 1:
            names = [type(m).__name__ for m in overriders]
            logger.warning(
                f"Multiple pipeline members override dispatch_block: {names}. "
                f"Only {names[0]} will be used; others are ignored."
            )

        if overriders:
            return overriders[0].dispatch_block(block, input_ids, input_others)
        return self.block_quantizer.dispatch_block(block, input_ids, input_others)

    def prepare_run(self, composer: "AlgorithmComposer" = None):
        for alg in self.members():
            alg.prepare_run(composer=self)

    def finalize_run(self):
        for alg in self.members():
            alg.finalize_run()
        # Rotation teardown is part of the model-level finalize stage.
        self._finalize_rotation(self._owning_model())

    # ------------------------------------------------------------------
    # Rotation lifecycle (owned entirely by the composer)
    # ------------------------------------------------------------------
    #
    # Rotation is a model-level pre-quantisation transform. Full-model rotation
    # must run *before* calibration data is cached, which is earlier than the
    # per-member ``prepare_run`` stage; layer-wise rotation instead prepares its
    # matrices here and rotates each block from within ``compress_block``. Both
    # are driven internally so the orchestrator only calls the single generic
    # entry point :meth:`apply_model_transforms`.

    _STREAMED_ROTATION_MSG = (
        "Rotation transforms on a streamed model (stream_quantization / "
        "AR_DISK_STREAM_MODEL meta skeleton) can only run layer-wise: leave "
        "layerwise_rotation unset (it auto-enables under both streaming modes) "
        "or pass --layerwise_rotation, so each block is folded inside the "
        "block loop instead of materializing the whole model."
    )

    def _model_has_meta(self, model) -> bool:
        return any(p.device.type == "meta" for p in model.parameters()) or any(
            b.device.type == "meta" for b in model.buffers()
        )

    def _assert_model_foldable_in_place(self, model) -> None:
        """Refuse whole-model rotation on streamed skeletons with an actionable error.

        Lazy-MoE replacement modules legitimately hold expert weights on meta
        until materialized from their original module -- resolve those first
        (skipping the call entirely when none exist, so a streamed skeleton does
        not log thousands of leftover warnings). Any meta parameters or buffers
        REMAINING afterwards belong to a streamed skeleton (no original to
        materialize from) and must not be folded.
        """
        from auto_round import envs

        if getattr(self._orchestrator_ref, "stream_quantization", False) or bool(envs.AR_DISK_STREAM_MODEL):
            raise ValueError(self._STREAMED_ROTATION_MSG)
        from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase, materialize_model_

        if any(isinstance(m, ReplacementModuleBase) for m in model.modules()):
            materialize_model_(model)
        if self._model_has_meta(model):
            raise ValueError(self._STREAMED_ROTATION_MSG)

    def _resolve_rotation_data_type(self) -> str:
        """Best-effort resolution of the quantization data_type for rotation dispatch."""
        if self.scheme is not None and getattr(self.scheme, "data_type", None):
            return self.scheme.data_type
        if self.block_quantizer is not None:
            return getattr(self.block_quantizer.config, "data_type", "mx_fp")
        return "mx_fp"

    def _owning_model(self) -> "torch.nn.Module | None":
        """Return the live model driven by this pipeline (via the block quantizer binding)."""
        if self.block_quantizer is not None:
            return getattr(self.block_quantizer, "model", None)
        return None

    def apply_model_transforms(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """Apply model-level pre-quantisation transforms (rotation) to *model*.

        Generic entry point invoked once by the orchestrator before calibration
        caching / the block loop. For full-model rotation the model is rotated
        immediately and returned; for layer-wise rotation only the rotation
        matrices are initialised and the per-block work is deferred to
        :meth:`compress_block`. Idempotent — repeated calls are a no-op.

        Returns:
            The (possibly mutated) model.
        """
        if self._rotation_prepared:
            return model

        self._rotation_transforms = []
        if not self._rotation_configs:
            self._rotation_prepared = True
            return model

        from auto_round.algorithms.transforms import apply_rotation, normalize_rotation_config
        from auto_round.algorithms.transforms.base import BaseRotation

        # Weight transforms read and rewrite every module's parameters, so any
        # lazy materialization (e.g. fused-MoE replacement modules holding
        # expert weights on the meta device until first use) must be resolved
        # before the transform pass, not at first block touch. Layer-wise mode
        # is the exception: transforms apply per block after the block loop
        # materializes it, so the model must NOT be fully materialized here
        # (it may be intentionally larger than available memory).
        if not self._layerwise_rotation:
            self._assert_model_foldable_in_place(model)

        data_type = self._resolve_rotation_data_type()
        logger.info("Applying Hadamard transform to the model.")
        for rotation_cfg in self._rotation_configs:
            if self._layerwise_rotation:
                normalised = normalize_rotation_config(rotation_cfg)
                if normalised is None:
                    continue
                rotation = BaseRotation.from_config(normalised)
                if rotation.supports_layerwise:
                    logger.info(
                        "[Rotation] Layer-wise mode: preparing R matrices only "
                        "(rotation deferred to per-block hook)."
                    )
                    rotation.prepare_layerwise(model, data_type=data_type)
                    self._rotation_transforms.append(rotation)
                    continue
                logger.warning(
                    f"[Rotation] {rotation.__class__.__name__} does not support "
                    f"layer-wise mode. Falling back to full-model rotation."
                )
                # The fallback is whole-model rotation: the streamed-model guard
                # skipped earlier because layerwise mode was requested, but this
                # transform cannot honor it -- re-check before folding.
                self._assert_model_foldable_in_place(model)
            model = apply_rotation(model, rotation_cfg, data_type=data_type)

        self._rotation_prepared = True
        return model

    def _run_block_ready_transforms(self, block: "torch.nn.Module", block_ctx: "BlockContext") -> None:
        """Apply layer-wise rotation to a block before reference collection.

        Called as the first step of :meth:`compress_block`. No-op when no
        layer-wise rotation transforms are active. Uses the block's global
        index (``block_ctx.block_index``) as the rotation layer index.
        """
        if not self._rotation_transforms:
            return

        block_idx = block_ctx.block_index
        block_names = block_ctx.block_names
        if isinstance(block_names, (list, tuple)) and len(block_names) > 1:
            sub_modules = list(block.layers) if hasattr(block, "layers") else [block]
            for j, sub_mod in enumerate(sub_modules):
                for t in self._rotation_transforms:
                    t.rotate_layer(sub_mod, layer_idx=block_idx + j)
        else:
            for t in self._rotation_transforms:
                t.rotate_layer(block, layer_idx=block_idx)

    def run_ready_transforms(self, block: "torch.nn.Module", block_ctx: "BlockContext") -> None:
        """Apply the block's weight-only ready transforms OUTSIDE compress_block.

        Used by the streaming orchestrator's background early-transform thread
        (default ON; AR_DISABLE_BG_READY_TRANSFORMS opts out): while block N tunes on its ping-pong
        group, the next block is loaded onto the idle group's home device and
        its layer-wise transforms run there, taking the transform off the
        critical path. Serialized against the in-loop ready stage via
        ``_ready_transform_lock``; activation-dependent preprocessors are NOT
        run here (they need the calibration chain and stay in-loop).
        """
        if not self._rotation_transforms:
            return
        with self._ready_transform_lock:
            self._run_block_ready_transforms(block, block_ctx)

    def _finalize_rotation(self, model: "torch.nn.Module") -> None:
        """Finalize layer-wise rotation after all blocks are processed (no-op when inactive)."""
        for t in self._rotation_transforms:
            t.finalize_layerwise(model)

    @property
    def has_layerwise_rotation(self) -> bool:
        """Whether layer-wise rotation transforms are active."""
        return bool(self._rotation_transforms)
