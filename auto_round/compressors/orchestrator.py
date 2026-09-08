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
import copy
import gc
import logging
import os
import re
import time as _time
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model
from tqdm import tqdm

from auto_round import envs
from auto_round.calibration import CalibrationContext
from auto_round.calibration.utils import (
    _update_inputs,
)
from auto_round.compressors.base import BaseOrchestrator
from auto_round.compressors.utils import (
    _get_quantized_layer_names_outside_blocks,
    immediate_pack,
    is_nv_fp,
    rehome_block_,
    strip_stale_device_hooks_,
)
from auto_round.data_type.utils import update_block_global_scale_if_needed
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    compress_layer_names,
    convert_module_to_hp_if_necessary,
    flatten_list,
    get_block_names,
    get_lm_head_name,
    get_module,
    is_auto_device_mapping,
    memory_monitor,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
    set_module,
    to_device,
)
from auto_round.utils.device import (
    _force_trim_malloc,
)
from auto_round.utils.device_manager import device_manager
from auto_round.utils.model import is_moe_model_via_config
from auto_round.utils.peak_watch import PeakWatcher
from auto_round.wrapper import WrapperMultiblock

if TYPE_CHECKING:
    from auto_round.utils.resume import ResumeState


# TODO wenhuach align all the API args


def _mark_load_seg(parts: dict, key: str, t0: float) -> float:
    """Fold one load sub-phase into the perf breakdown; returns a fresh t0."""
    if parts is not None:
        parts[key] = _time.perf_counter() - t0
    return _time.perf_counter()


def _format_load_breakdown(parts: dict, min_s: float = 0.05) -> str:
    """Render the load sub-phase breakdown for the [perf] line.

    Segments below ``min_s`` fold away so a fast load stays a single number;
    empty string keeps the line unchanged for runs without the counters.
    """
    if not parts:
        return ""
    shown = [(k, v) for k, v in parts.items() if v >= min_s]
    if not shown:
        return ""
    return " (" + ", ".join(f"{k} {v:.1f}s" for k, v in shown) + ")"


def _fmt_mem_regions(maps, cap=4):
    """Compact ``rss/size path`` list of the top host RSS mappings.

    ``[anon]`` covers CUDA pinned pools / torch host caches, ``[heap]`` the
    allocator, file-backed paths are checkpoint mmaps (basename only).
    """
    import os

    parts = []
    for m in maps[:cap]:
        name = os.path.basename(m.path or "") or "[anon]"
        parts.append(f"{m.rss / 2**30:.2f}G/{m.size / 2**30:.2f}G {name}")
    if len(maps) > cap:
        parts.append(f"(+{len(maps) - cap} more)")
    return "; ".join(parts)


def _fmt_mem_top(big, dev, cap=3):
    """Compact ``size name`` list of the largest tensors on one device."""
    entries = sorted((b for b in big if b[1].startswith(f"{dev}:")), reverse=True)
    # "cuda:1:name" -> "name" (cpu names carry no index slot)
    parts = [f"{n / 2**30:.2f}G {name.split(':', 2)[-1]}" for n, name in entries[:cap]]
    if len(entries) > cap:
        parts.append(f"(+{len(entries) - cap} more)")
    return ", ".join(parts)


def _format_host_buckets(buckets: dict) -> str:
    """Render host inventory buckets compactly for the [stream-mem] log line.

    Buckets below the 0.01G render resolution (e.g. placeholder per-block
    entries) would print as ``block:N=0.00G`` and drown the real residents,
    so they collapse into one ``[N negligible buckets]`` marker. The
    tracked-total sum still counts every bucket.
    """
    shown = {k: v for k, v in buckets.items() if v >= 0.005 * 2**30}
    parts = [f"{k}={v / 2**30:.2f}G" for k, v in sorted(shown.items())]
    skipped = len(buckets) - len(shown)
    if skipped:
        parts.append(f"[{skipped} negligible buckets]")
    return ", ".join(parts)


_FUSED_EXPERT_PROJECTIONS = frozenset({"gate_up_proj", "gate_proj", "up_proj", "down_proj"})

#: per-expert unfused projection -> its fused on-disk stack, e.g.
#: ``mlp.experts.7.gate_proj.weight`` lives inside ``mlp.experts.gate_up_proj``
_FUSED_STACK_RE = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _first_chain_row(fp_inputs):
    """The first calibration row from chain inputs.

    Plain blocks forward a list of rows; block classes registered in
    ``_BLOCK_OUTPUT_REGISTRY`` forward a dict of per-key row lists (e.g.
    ``{"hidden_states": [...], "prev_topk_indices": ...}``) - prefer its
    ``hidden_states``."""
    rows = fp_inputs.get("hidden_states") if isinstance(fp_inputs, dict) else fp_inputs
    if isinstance(rows, dict):
        rows = next(iter(rows.values()))
    return rows[0]


def _canonical_group_leaf(name: str) -> str:
    """Fold a relative tensor/param name onto a layout-neutral spelling so
    module-side (unfused, per-expert) and checkpoint-side (fused 3D) trees
    compare equal for sibling matching: digit segments collapse and the
    per-expert split projections map onto their fused stack names."""
    m = _FUSED_STACK_RE.match(name)
    if m:
        base, _, proj = m.groups()
        fused = "gate_up_proj" if proj in ("gate_proj", "up_proj") else "down_proj"
        return f"{base}.{fused}.weight"
    return re.sub(r"\.\d+\.", ".", name)


class CheckpointOnlyRMSNorm(torch.nn.Module):
    """Generic RMSNorm for checkpoint-only predictor groups (e.g. the prologue
    norms of a multi-token-prediction block the modeling code never builds).

    The checkpoint supplies only the weight vector; the forward follows the
    transformers numerics convention (fp32 compute, one downcast) so a
    materialized group tree can run like any decoder block later on."""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        with torch.device("meta"):
            self.weight = torch.nn.Parameter(torch.empty(hidden), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hs = hidden_states.float()
        hs = hs * torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (hs * self.weight.float()).to(hidden_states.dtype)


def _ensure_module_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Walk/create intermediate shells for *path* and return the parent that
    a leaf module should be attached to."""
    segments = path.split(".")
    parent = model
    for seg in segments[:-1]:
        child = getattr(parent, seg, None)
        if child is None:
            child = torch.nn.Module()
            parent.add_module(seg, child)
        parent = child
    return parent


def synthesize_predictor_e(input_rows: torch.Tensor, embed=None) -> torch.Tensor:
    """Build the embedding-side predictor input (e-first concat).

    ``input_rows`` is either the already-embedded chain rows (the first
    block's input, the cheapest source at tuning time) or raw token ids
    together with ``embed``. Position ``t`` consumes token ``t+1``'s
    embedding; the final position repeats the last row - the uniform
    convention of the verified predictor families (the concat is
    embedding-side first everywhere)."""
    if input_rows.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        if embed is None:
            raise ValueError("token-id rows need the embedding module to build the predictor input")
        shifted = torch.cat([input_rows[:, 1:], input_rows[:, -1:]], dim=1)
        return embed(shifted)
    return torch.cat([input_rows[:, 1:], input_rows[:, -1:]], dim=1)


def forward_checkpoint_only_predictor(shell: torch.nn.Module, hidden_states: torch.Tensor, **input_others):
    """Run a materialized predictor tree like a decoder block.

    norm_e(e) concat norm_h(h) -> concat mixer -> decoder layer -> final
    norm. The embedding-side input arrives as the ``_predictor_e`` batch
    input (per-row tensors the forward runner batches like any other
    auxiliary input) or falls back to the value bound on the shell; every
    other keyword input passes through to the layer unchanged."""
    refs = getattr(shell, "_predictor_refs", None)
    if refs is None:
        raise RuntimeError("predictor forward called on a shell without bound role refs")
    e = input_others.pop("_predictor_e", None)
    if e is None:
        e = shell._predictor_e
    if e is None:
        raise RuntimeError(
            "predictor embedding-side input is not bound yet; synthesize it from the chain " "state before tuning"
        )
    if isinstance(e, (list, tuple)):
        e = torch.cat(list(e), dim=0)
    if e.device != hidden_states.device:
        e = e.to(hidden_states.device)
    x = torch.cat([refs["norm_e"](e), refs["norm_h"](hidden_states)], dim=-1)
    x = refs["fc"](x)
    x = refs["layer"](x, **input_others)
    x = x[0] if isinstance(x, (tuple, list)) else x
    final_norm = refs.get("final_norm")
    return final_norm(x) if final_norm is not None else x


def bind_checkpoint_only_predictor(shell: torch.nn.Module, refs: dict, e: torch.Tensor = None) -> None:
    """Attach the predictor forward and role refs to the group shell.

    The ref dict keeps module handles out of ``named_modules`` (the real tree
    already registers them once under their checkpoint paths), and the bound
    forward lets block-level machinery call the group like any decoder
    block. ``e`` may be bound later, right before the first forward."""
    shell._predictor_refs = refs
    shell._predictor_e = e
    shell.forward = lambda hidden_states, **input_others: forward_checkpoint_only_predictor(
        shell, hidden_states, **input_others
    )


def _is_fused_expert_weight_name(tensor_name: str) -> bool:
    """True for stacked per-expert weight tensors: the name under ``.weight``
    is a fused MoE projection and the checkpoint stores one dim-0 stacked
    tensor for all experts (the layout the family module replacements
    unfuse for the main body)."""
    leaf = tensor_name.removesuffix(".weight")
    return leaf.rsplit(".", 1)[-1] in _FUSED_EXPERT_PROJECTIONS


def _apply_pin_attrs(module: torch.nn.Module, entry: dict) -> None:
    """Copy a layer_config pin's quantization attributes onto *module*.

    Single source for every placeholder/pin materialization site (scattered
    placeholders, fused-expert slices, tree Linears)."""
    for key in ("bits", "group_size", "data_type", "sym", "scale_dtype"):
        if entry.get(key) is not None:
            setattr(module, key, entry[key])
    module.act_bits = entry.get("act_bits", 16)
    module.act_sym = entry.get("act_sym", True)
    module.act_data_type = entry.get("act_data_type", None)


def materialize_placeholder_linear_from_tensor(
    model: torch.nn.Module, path: str, tensor: torch.Tensor, entry: dict
) -> None:
    """Attach a CPU-weight ``nn.Linear`` placeholder at *path* from a real
    tensor slice.

    Used for checkpoint-only fused expert stacks: the per-expert module path
    differs from the single 3D checkpoint tensor, so the outside-block loader
    (which loads by name) cannot fetch these weights; they arrive real at
    materialization time instead.
    """
    lin = torch.nn.Linear(int(tensor.shape[1]), int(tensor.shape[0]), bias=False)
    with torch.no_grad():
        lin.weight.copy_(tensor)
    _apply_pin_attrs(lin, entry)
    lin.global_name = path
    _ensure_module_path(model, path).add_module(path.rsplit(".", 1)[-1], lin)


def materialize_placeholder_linear(
    model: torch.nn.Module, path: str, shape: tuple, entry: dict, has_bias: bool
) -> None:
    """Attach a meta-weight placeholder ``nn.Linear`` at *path* for a pinned
    layer that exists only in the checkpoint (checkpoint-only blocks such as
    an MTP layer the modeling code never instantiates).

    The placeholder carries the pin's quantization attributes and the
    checkpoint tensor path as ``global_name``; the outside-block pass then
    loads its real weights and quantizes, packs and shard-writes it like any
    other pinned layer. Weights start on the meta device so materialization
    costs no host RAM at checkpoint scale.
    """
    parent = _ensure_module_path(model, path)
    with torch.device("meta"):
        lin = torch.nn.Linear(int(shape[1]), int(shape[0]), bias=has_bias)
    _apply_pin_attrs(lin, entry)
    lin.global_name = path
    parent.add_module(path.rsplit(".", 1)[-1], lin)


# values that disable block staging; shared by the loop-start depth check and
# the staging-device resolver - the CLI parser mirrors it via the same tuple
STREAM_PREFETCH_OFF = ("", "off", "0", "false")


class CompressionOrchestrator(BaseOrchestrator):

    def __init__(
        self,
        config: Union[object, list[object]],  # TODO rename this to alg_config wenhuach
        model: Union[torch.nn.Module, str],
        tokenizer: Any = None,
        platform: str = "hf",
        format: Union[str, list, None] = None,
        dataset: Optional[Union[str, list, tuple, torch.utils.data.DataLoader]] = None,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: Optional[bool] = None,
        seed: int = 42,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            platform=platform,
            format=format,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            low_cpu_mem_usage=low_cpu_mem_usage,
            dataset=dataset,
            **kwargs,
        )

    def post_init(self) -> None:
        """Run base post-init then attach the registered calibrator strategy.

        Model-type mixins select the calibrator kind; the calibrator owns
        ``try_cache_inter_data_gpucpu`` / ``cache_inter_data`` orchestration
        plus the model-specific ``calib`` body.
        """
        if self._post_init_done:
            return
        super().post_init()
        if self.need_calib and self.calibration is None:
            from auto_round.calibration import get_calibrator

            kind = self._get_calibrator_kind()
            self.calibration = get_calibrator(kind)(self)

    def _get_calibrator_kind(self) -> str:
        """Return the registry name of the calibrator to use.

        Default ``"llm"``.  ``MLLMMixin`` / ``DiffusionMixin`` override this
        to select ``"mllm"`` / ``"diffusion"``.
        """
        return "llm"

    @torch.no_grad()
    def cache_data(
        self,
        block_names: list,
        nsamples: int,
        layer_names: Optional[list] = None,
        last_cache_name: Optional[str] = None,
    ) -> Any:
        """Thin wrapper around ``self.calibration.collect``.

        Public API kept for backward compatibility (entry.py and
        LLM-Compressor integration).
        """
        if self.calibration is None:
            self.post_init()

        res = self.calibration(block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name)
        # Sync batch_size back in case calibration clamped it due to insufficient samples
        # Tricky setting
        self.calibration_context.batch_size = self.calibration.batch_size
        self.alg_composer.block_forward.batch_size = self.calibration_context.batch_size
        self.calibration_context.seqlen = self.calibration.seqlen
        self.calibration_context.batch_dim = self.calibration.batch_dim
        self.calibration_context.dataset = self.calibration.dataset
        self.calibration_context.is_only_supported_bs1 = self.calibration.is_only_supported_bs1
        # Reset gradient_accumulate_steps in case batch_size was clamped to 1 for some models
        if self.calibration_context.is_only_supported_bs1:
            compressors = self.alg_composer.block_quantizer
            if not isinstance(compressors, (list, tuple)):
                compressors = [compressors]
            else:
                compressors = list(compressors)
            compressors.extend(self.alg_composer.preprocessors)
            for compressor in compressors:
                if hasattr(compressor, "gradient_accumulate_steps"):
                    compressor.gradient_accumulate_steps = (
                        compressor.gradient_accumulate_steps * self.calibration_context.orig_batch_size
                    )

        return res

    @staticmethod
    def _assert_block_materialized(block: torch.nn.Module, block_name: str) -> None:
        """Fail loudly when a streamed block still carries meta parameters.

        After direct streaming and the replacement-module materialization ran,
        every block PARAMETER must have real storage. A leftover meta parameter
        means the checkpoint (even after the conversion-name aliases) had no
        tensor for it -- silently continuing would crash later inside tuning
        or packing with an opaque ``Cannot copy out of meta tensor``. Buffers
        are exempt: computed tables (e.g. rotary) are rebuilt elsewhere.
        """
        still_meta = [name for name, p in block.named_parameters(recurse=True) if p.device.type == "meta"]
        if still_meta:
            shown = ", ".join(still_meta[:8]) + (" ..." if len(still_meta) > 8 else "")
            raise ValueError(
                f"[stream] {len(still_meta)} parameter(s) of {block_name!r} stayed on meta after streaming "
                f"(no checkpoint tensor matched, even via conversion name aliases): {shown}. The checkpoint "
                "may spell these differently from the modeling code; if transformers' conversion "
                "registry lacks the family's renames, extend it there."
            )

    @staticmethod
    def _should_offload_after_pack(compress_context) -> bool:
        """Whether a just-processed block still needs an offloader state write.

        With immediate saving the block is already flushed to the output
        shards; writing its state_dict afterwards would leave a ~block-sized
        dead file per block until process exit.
        """
        return compress_context.low_cpu_mem_usage and not compress_context.is_immediate_saving

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        # Thin wrapper around auto_round.calibration.inputs.preprocess_block_inputs.
        from auto_round.calibration.inputs import preprocess_block_inputs

        return preprocess_block_inputs(
            inputs,
            model_context=self.model_context,
            compress_context=self.compress_context,
            first_input_name=first_input_name,
        )

    def _quantize_blocks(
        self,
        model: torch.nn.Module,
        inputs: dict,
        block_names: list,
        q_input: torch.Tensor | None = None,
        nblocks: int = 1,
        pbar: tqdm | None = None,
        input_others_extra_blocks: dict | None = None,
        token_ids: list[torch.Tensor] | None = None,
        resume_state: Optional["ResumeState"] = None,
        resume_input_ids=None,
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        nblocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.
        resume_state: when set and already partway through this block group
            (`resume_state.resume_index > 0`), the caller has already
            substituted `inputs`/`q_input` for the first not-yet-done block;
            this method just needs to start its loop there instead of at
            index 0, and record each block as done afterward. See
            auto_round/utils/resume.py.
        resume_input_ids: the exact `input_ids` value the interrupted run had
            live for the first not-yet-done block (cached by
            `ResumeState.mark_block_done`). `inputs` still supplies
            `input_others` (legitimately re-sourced from the same pre-cache
            every iteration regardless of resuming), but the chained
            hidden-state tensor itself must come from here, not be re-derived
            from `inputs` -- see auto_round/utils/resume.py's module
            docstring for why those two aren't interchangeable.

        Returns:
        None
        """
        clear_memory()
        for n, m in model.named_parameters():
            m.requires_grad_(False)

        input_ids, input_others = self._preprocess_block_inputs(inputs)
        if resume_input_ids is not None:
            input_ids = resume_input_ids

        if pbar is None:
            pbar = tqdm(range(0, len(block_names), nblocks))

        start_index = resume_state.resume_index if resume_state is not None and nblocks == 1 else 0
        for i in range(start_index, len(block_names), nblocks):
            if input_others_extra_blocks and block_names[i] in input_others_extra_blocks:
                input_others = input_others_extra_blocks[block_names[i]]
                _, input_others = self._preprocess_block_inputs(input_others)
                input_others_extra_blocks.pop(block_names[i])
            if i != 0:
                pbar.update(1)
            if nblocks == 1:
                n = block_names[i]
                pbar.set_description(f"Quantizing {n}")
                m = get_module(model, n)
            else:
                names = block_names[i : min(i + nblocks, len(block_names))]
                pbar.set_description(f"Quantizing [{i + 1}-{min(i + nblocks, len(block_names))}]/{len(block_names)}")
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            # Also reload when disk streaming is active even if `low_cpu_mem_usage`
            # has been forced False (e.g. GGUF export -- see base.py's
            # `_adjust_immediate_packing_and_saving`, which disables `low_cpu_mem_usage` for
            # gguf formats for reasons unrelated to disk streaming). Disk streaming
            # can be turned on explicitly via `AR_DISK_STREAM_MODEL=1` *or* chosen
            # automatically for fused-MoE checkpoints (see ModelContext's
            # `_should_use_meta_skeleton`); either way the model was built as a meta
            # skeleton and `_disk_stream_index` is set. Under streaming, a block
            # starts on the meta device regardless of `low_cpu_mem_usage`, which only
            # ever controlled whether to *free* it again after use -- without this,
            # the block below is never materialized at all and `m.to(device)` crashes
            # with "Cannot copy out of meta tensor". The block intentionally stays
            # real afterward (no matching post-tune offload runs when
            # `low_cpu_mem_usage` is False -- see the `is_immediate_saving`-adjacent
            # offload call further down), matching upstream's own choice not to cycle
            # blocks for these formats.
            disk_streaming = getattr(self.model_context, "_disk_stream_index", None) is not None
            if self._should_offload_after_pack(self.compress_context) or envs.AR_DISK_STREAM_MODEL or disk_streaming:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            block_name_or_names = n if nblocks == 1 else names

            # ── Infrastructure: materialize, dtype convert, device placement ──
            materialize_model_(m)
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device_manager.device)

            m = self.alg_composer.dispatch_block(m, input_ids, input_others)
            if self._peak_watch is not None:
                self._peak_watch.set_phase("tune")

            if envs.AR_MEM_COUNTERS:
                self._log_device_inventory(
                    {"input_others": input_others},
                    "",
                    extra_buckets={
                        "cache-block": input_ids,
                        "cache-remaining": input_others_extra_blocks or {},
                    },
                )

            # ── Pipeline lifecycle: per-block setup ───────────────────────────
            from auto_round.algorithms.composer import BlockContext

            current_block_names = (
                block_name_or_names if isinstance(block_name_or_names, list) else [block_name_or_names]
            )
            current_block_name = current_block_names[0] if len(current_block_names) == 1 else str(block_name_or_names)
            # bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff #TODO recover infer_bs_coeff
            bs = self.calibration_context.batch_size

            ctx = BlockContext(
                model=model,
                block_names=current_block_names,
                block_name=current_block_name,
                block_index=i,
                bs=bs,
                is_mllm=self.model_context.is_mllm,
                is_diffusion=self.model_context.is_diffusion,
                pbar=pbar,
                block_cnt=(len(block_names) + nblocks - 1) // nblocks,
            )

            # ── Run block pipeline (calibration → quantization → collection) ──
            new_q_input, reference_output = self.alg_composer.compress_block(
                m,
                input_ids,
                input_others,
                block_ctx=ctx,
                q_inputs=q_input,
                input_ids=token_ids,
            )

            # ── Infrastructure: memory management ─────────────────────────────
            # Mirrors the original q_input-swap + end-of-loop clear_memory semantics:
            # clear the FP input when a quantized input was used, then clear the old
            # q_input (effective_input) before advancing to the next block.
            if q_input is not None:
                if input_ids is not q_input:
                    clear_memory(input_ids)
                else:
                    clear_memory()
                next_input_ids = reference_output
                clear_memory(q_input if q_input is not next_input_ids else None)
            else:
                next_input_ids = reference_output
                clear_memory(input_ids if input_ids is not next_input_ids else None)

            q_input = new_q_input

            # ── Infrastructure: hook removal, device cleanup, logging ─────────
            if len(device_manager.device_list) > 1 and not self.model_context.is_diffusion:
                accelerate.hooks.remove_hook_from_submodules(m)
            mv_module_from_gpu(m)
            clear_memory(device_list=device_manager.device_list)
            memory_monitor.log_summary()

            # ── Infrastructure: immediate_pack / shard write ──────────────────
            if self.compress_context.is_immediate_packing:
                for _n, _mod in m.named_modules():
                    if hasattr(_mod, "bits") and check_to_quantized(_mod):
                        from auto_round.compressors.utils import immediate_pack as _immediate_pack

                        module_name = getattr(_mod, "global_name", None)
                        if module_name is None and nblocks == 1 and _n:
                            module_name = f"{n}.{_n}"
                        if module_name is None:
                            continue
                        _immediate_pack(module_name, self.layer_config)

            input_ids = next_input_ids

            if self.compress_context.is_immediate_saving:
                self.shard_writer.write(m, is_finalize=False)
                # ShardWriter only actually flushes to disk once its
                # shard-size budget is reached (`_flush_shard`, private but
                # there's no public equivalent) -- `write()` above may just
                # buffer this block's tensors in memory. Force a flush here
                # whenever resumability is active, since marking a block
                # "done" in the resume manifest is a lie if a crash before
                # the next natural flush would lose its tensors entirely.
                # Only pay this extra small-shard-fragmentation cost when
                # AR_RESUME_DIR is actually set.
                if resume_state is not None:
                    self.shard_writer._flush_shard()

            if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                if nblocks == 1:
                    self._offloader(model, n, overwrite=True)
                else:
                    for name in names:
                        self._offloader(model, name, overwrite=True)

            # Record this block as durably done (its quantized weights are
            # either flushed to a shard on disk via ShardWriter, or saved to
            # the offloader's temp dir) only now, after that write has
            # happened -- so a crash before this point correctly re-does the
            # block on resume instead of skipping it with incomplete/missing
            # output. See auto_round/utils/resume.py.
            if envs.AR_MEM_COUNTERS:
                # post-tune sample: the VmHWM delta vs the pre-tune line
                # measures this block's during-tuning transient peak
                self._log_device_inventory(
                    None,
                    "post",
                    extra_buckets={"cache-remaining": input_others_extra_blocks or {}},
                )
            if self._peak_watch is not None:
                self._peak_watch.set_phase("write")
                self._peak_watch.log("")
                self._peak_watch.reset_run_max()
            if resume_state is not None and nblocks == 1:
                # `input_ids` was already reassigned to `next_input_ids`
                # above -- it now holds the value the *next* block should use
                # as its chained hidden-state input, which is exactly what
                # needs to be persisted here.
                resume_state.mark_block_done(n, q_input, input_ids)
        if pbar is not None:
            pbar.update(1)

        if self._peak_watch is not None:
            self._peak_watch.stop()
        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        for n, m in self.model.named_modules():
            if hasattr(m, "name"):
                delattr(m, "name")

        del q_input
        del input_ids
        del input_others
        del inputs

        clear_memory()

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.post_init()

        # mirror the normal path's model.to(amp_dtype): every tensor the
        # streamer materializes must follow the same dtype policy or the
        # streamed run quantizes raw checkpoint precision while a fully
        # loaded run quantizes the converted dtype (amp state resolves here,
        # after the context finished loading). Applies at amp=False too:
        # amp_dtype resolves to fp32 there, matching _set_amp_dtype's full
        # upcast of a non-streamed run
        streamer = getattr(self.model_context, "checkpoint_streamer", None)
        if streamer is not None:
            streamer.load_dtype = self.amp_dtype

        if not self.need_calib:
            try:
                return self._quantize_zero_shot()
            except BaseException:
                # exception-path teardown: without this, a failed run leaves
                # the daemon prefetch reader alive - it keeps staging until it
                # holds depth+1 block-sized tensors on the staging devices plus
                # pooled shard handles, and an in-process retry (catch + rerun,
                # the flow the context-reset design anticipates) starts with a
                # zombie thread pinning VRAM it can never release
                bg_t = getattr(self, "_bg_pack_thread", None)
                if bg_t is not None and bg_t.is_alive():
                    # join (bounded: one block's pack+write) so no orphan
                    # keeps writing shards after the exception escapes; a
                    # worker-side error is NOT re-raised here - the original
                    # failure wins
                    try:
                        bg_t.join()
                    except Exception as join_err:  # noqa: BLE001 - never mask the original
                        logger.warning("[stream] bg-pack join during teardown hit an error: %s", join_err)
                self._bg_pack_thread = None
                streamer = getattr(self.model_context, "checkpoint_streamer", None)
                if streamer is not None:
                    try:
                        streamer.stop_prefetch()
                        streamer.close()
                    except Exception as teardown_err:  # noqa: BLE001 - never mask the original
                        logger.warning("[stream] teardown after failure hit an error: %s", teardown_err)
                raise

        return self._quantize_data_driven()

    def _stream_resume_jump_chain(self, calib_state, resume_states) -> None:
        """Jump the streaming calibration chain to the deepest saved frontier entry.

        The per-group manifests hold the successor chain entry (FP hidden
        states) exactly as the serial path persists it, so a run
        interrupted in ANY mode can hand its frontier to the streaming loop:
        the deepest group with progress provides the entry its next pending
        block would consume. ``input_others``/``token_ids`` are static per row
        and rebuilt deterministically by ``prepare_streaming_calibration``.
        """
        if calib_state is None:
            return
        for rs in resume_states:
            if rs is None or rs.resume_index <= 0:
                continue
            fully_done = rs.resume_index >= len(rs.block_names)
            entry = rs.load_input_ids()
            if entry is None:
                if fully_done:
                    continue  # nothing left to consume the entry in this group
                # A group with pending blocks but no loadable successor entry:
                # either the crash-window guard rejected an entry written past
                # the manifest, or the chain file is missing/corrupt. Continuing
                # would skip manifest-done blocks while the chain stays at the
                # raw embedding outputs - silent wrong-input tuning, the exact
                # corruption the guard exists to prevent. The shards written by
                # such a run were computed on the wrong chain, so salvage is not
                # safe: fail loud and tell the user what to delete.
                raise RuntimeError(
                    f"[stream] resume state for group {rs.block_names[0]}..{rs.block_names[-1]} is "
                    "inconsistent: {}/{} blocks marked done but the successor chain entry is missing "
                    "or was rejected (crash between the tensor save and the manifest write, or a "
                    "deleted/corrupt chain file). Automatic salvage would tune the frontier block "
                    "on the wrong inputs; delete the resume directory AND the output directory "
                    "and rerun.".format(rs.resume_index, len(rs.block_names))
                )
            calib_state["fp_inputs"] = entry
            if isinstance(entry, torch.Tensor):
                _rows_probe = [entry]  # a plain tensor is a valid simple-chain entry
            else:
                _rows_probe = entry.get("hidden_states") if isinstance(entry, dict) else entry
            if isinstance(_rows_probe, dict):
                _rows_probe = next(iter(_rows_probe.values()), None)
            if (
                not isinstance(_rows_probe, (list, tuple))
                or not _rows_probe
                or not isinstance(_rows_probe[0], torch.Tensor)
            ):
                _desc = (
                    f"{type(entry).__name__}"
                    if not isinstance(_rows_probe, (list, tuple))
                    else f"rows[{len(_rows_probe)}] first={type(_rows_probe[0]).__name__ if _rows_probe else 'empty'}"
                )
                raise RuntimeError(
                    f"[stream] resume: the saved successor chain entry for group "
                    f"{rs.block_names[0]}..{rs.block_names[-1]} is not a usable row set "
                    f"({_desc}); the first resumed block would tune on wrong/no inputs. "
                    "The bg-finish snapshot may have captured a partially-filled chain - "
                    "rerun the frontier block with AR_STREAM_BG_PACK=0 or from a fresh "
                    "resume dir."
                )
            q_input = rs.load_q_input()
            if self.alg_composer.need_quanted_input():
                # the quantized-input chain is required (SignRound default):
                # a missing or crash-window-rejected q_input must fail loud
                # like input_ids above, not silently tune the frontier block
                # on FP inputs
                if q_input is None:
                    raise RuntimeError(
                        f"[stream] resume state for group {rs.block_names[0]}..{rs.block_names[-1]} has a "
                        "usable FP chain entry but no quantized-input entry, while the quantizer requires "
                        "one (enable_quanted_input). The run that wrote this state crashed between the "
                        "chain tensor saves, or the file was removed; delete the resume directory AND the "
                        "output directory and rerun."
                    )
                calib_state["q_inputs"] = q_input

    @staticmethod
    def _stream_resume_pending_offset(all_blocks, resume_states):
        """Flat index of the first block not yet done in the resume manifests.

        Returns None when every block is already done. Groups are visited in
        order; a fully-done group's frontier equals its length, so the offset
        lands on the next group's first pending block.
        """
        seen = 0
        for gi, blocks in enumerate(all_blocks):
            rs = resume_states[gi] if resume_states is not None and gi < len(resume_states) else None
            frontier = rs.resume_index if rs is not None else 0
            if frontier < len(blocks):
                return seen + frontier
            seen += len(blocks)
        return None

    def _gguf_blob_mode(self) -> bool:
        """True when a streaming GGUF run packs per-block ggml payloads to blob shards.

        In this mode the GGUF container cannot be written progressively (it is
        a single file), so packed payloads spill to ``gguf-blobs/`` shards via
        :class:`GgufBlobStore` and the final container is assembled from them
        at save time. Compressed-tensors shard writes are skipped for such
        runs: the blobs are the durable output.
        """
        formats = getattr(self, "formats", None)
        if not isinstance(formats, list) or len(formats) != 1 or not formats[0].is_gguf():
            return False
        return bool(getattr(self.model_context, "stream_quantization", False))

    def _ensure_gguf_blob_conversion_(self, streamer) -> None:
        """Create the GGUF conversion instance up front for streaming blob runs.

        Two reasons this cannot stay lazy (first block's pack): (a) MTP
        detection must see checkpoint-only predictor trees, which only
        materialize at group tail - hours after the first block - or nextn
        blocks are silently excluded; (b) resume adoption happens before the
        loop, and the blob store must exist by then.
        """
        if not self._gguf_blob_mode():
            return
        from auto_round.compressors.utils import _get_save_folder_name
        from auto_round.export.export_to_gguf.blob_store import GgufBlobStore
        from auto_round.export.export_to_gguf.config import ModelType
        from auto_round.export.export_to_gguf.export import create_model_class

        save_folder = _get_save_folder_name(self.formats[0])
        store = GgufBlobStore.get_or_create(save_folder)
        mtp_names = None
        if streamer is not None:
            mtp_names = [n for n in streamer.weight_map if n.startswith(("mtp.", "model.mtp."))]
        instance = create_model_class(
            save_folder,
            self.model,
            self._layer_config_with_regex_pins_(),
            self.formats[0].get_backend_name(),
            low_cpu_mem_usage=True,
            model_type=ModelType.TEXT,
            device=str(self.device),
            quant_nontext_module=getattr(self.model_context, "quant_nontext_module", False),
            is_auto_scheme=getattr(self.formats[0], "is_auto_scheme", False),
            blob_store=store,
            mtp_checkpoint_names=mtp_names,
        )
        # register as the pack-time global: pack_gguf_layer's lazy creation
        # would otherwise build a SECOND instance without the MTP hint and
        # discard this one - and the hint's detection window (before the
        # checkpoint-only tree materializes) never comes back
        import auto_round.export.export_to_gguf.export as _gguf_export

        if getattr(_gguf_export, "gguf_model_instance_global", None) is None:
            _gguf_export.gguf_model_instance_global = [instance]

    def _layer_config_with_regex_pins_(self):
        """layer_config plus the resolver-retained regex pin entries.

        Pins like ``'.*mtp.*'`` match nothing at resolution time on streaming
        runs - the checkpoint-only predictor tree materializes hours later -
        so the resolver parks them in ``regex_config`` as literal regex keys.
        Merging them here lets the gguf dtype walk honor them (exact concrete
        entries still take precedence: they are looked up first)."""
        merged = dict(self.layer_config)
        try:
            regex_pins = self.regex_config or {}
        except AttributeError:
            regex_pins = {}
        for key, val in regex_pins.items():
            merged.setdefault(key, dict(val) if isinstance(val, dict) else val)
        return merged

    def _adopt_blob_store_(self) -> None:
        """Adopt a crashed run's blob shards when resuming a streaming GGUF export.

        Mirrors ``ShardWriter.adopt_existing_shards``: without this, a fresh
        store restarts the shard counter at 1 and overwrites the crashed run's
        blobs while the resume manifest skips their already-done blocks.
        """
        if not self._gguf_blob_mode():
            return
        from auto_round.compressors.utils import _get_save_folder_name
        from auto_round.export.export_to_gguf.blob_store import GgufBlobStore

        store = GgufBlobStore.get_or_create(_get_save_folder_name(self.formats[0]))
        adopted = store.adopt_existing()
        if adopted:
            logger.info("[stream] gguf blob resume: adopted %d tensor(s) from prior shards", adopted)

    @staticmethod
    def _park_rows_cpu_(rows):
        """Move a row list (or single tensor) to host RAM in place-safe form."""
        if isinstance(rows, list):
            return [r.to("cpu") if torch.is_tensor(r) and r.device.type != "cpu" else r for r in rows]
        if torch.is_tensor(rows) and rows.device.type != "cpu":
            return rows.to("cpu")
        return rows

    @staticmethod
    def _snapshot_chain_rows_(state):
        """Detached deep copy of chain rows for a resume snapshot.

        The bg-finish worker serializes the snapshot seconds later, while the
        main loop has already reset the shared chain containers for the next
        block (27B evidence: the saved entry read back as rows[128] with a
        None first row). Copying here - on the main thread, at capture time -
        makes the snapshot immune to that mutation. None slots are preserved:
        the resume-entry validation rejects them loudly instead of letting a
        partially-filled chain masquerade as a usable frontier."""

        def _cp(v):
            return v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else v

        if isinstance(state, dict):
            return {k: CompressionOrchestrator._snapshot_chain_rows_(v) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(CompressionOrchestrator._snapshot_chain_rows_(v) for v in state)
        return _cp(state)

    def _write_finished_block_(
        self, block, block_name: str, tied_weights_layers: set, rs, q_snap, fp_snap, is_model_last: bool
    ) -> None:
        """Immediate-saving tail of a finished block, shared by the serial
        path and the background pack worker: save non-quantized leaf modules,
        write the block scope, park to meta, then - only after a durable
        flush - let the manifest claim the block done (crash-durability
        contract: done implies tensors are in a shard file)."""
        gguf_blob = self._gguf_blob_mode()
        if not gguf_blob:
            for _n, m in block.named_modules():
                if (
                    not any(m.children())
                    and len(m.state_dict()) > 0
                    and hasattr(m, "global_name")
                    and m.global_name not in tied_weights_layers
                    and not check_to_quantized(m)
                ):
                    set_module(self.model, m.global_name, copy.deepcopy(m))
                    self.shard_writer.write(name=m.global_name)
                    get_module(self.model, m.global_name).to("meta")
                    m.to("meta")
            self.shard_writer.write(name=block_name)
        block.to("meta")
        if rs is not None:
            if not gguf_blob:
                self.shard_writer._flush_shard()
            _t0 = _time.perf_counter()
            rs.mark_block_done(block_name, q_snap, None if is_model_last else fp_snap)
            return _time.perf_counter() - _t0
        return 0.0

    def _start_bg_pack_block(
        self,
        block,
        block_name: str,
        load_device,
        layer_config: dict,
        nblocks: int,
        tied_weights_layers,
        rs,
        q_snap,
        fp_snap,
        is_model_last: bool,
    ):
        """Pack + shard-write the FINISHED block in a background thread.

        Runs while the main loop advances to the next block (which tunes on
        the other ping-pong group): :func:`immediate_pack_block` on the
        block's home device, then the leaf saves, block-scope write, flush,
        resume-manifest update and meta-release -- exactly the serial tail,
        on the now-idle group. ``q_snap``/``fp_snap`` are CAPTURED references
        to the next block's inputs (the main loop mutates ``calib_state``
        the moment it advances); ``mark_block_done`` must receive those, not
        a live dict read. Exactly one pipeline thread runs at a time (the
        loop joins the previous one first): shard writes stay ordered and
        the lock-free ShardWriter has a single writer. A worker failure is
        re-raised at join time -- a silently skipped pack would corrupt the
        checkpoint.
        """
        import threading as _threading

        holder = {"exc": None, "pack": 0.0, "write": 0.0, "snap": 0.0}

        def _worker():
            import time as _wtime

            try:
                _t0 = _time.perf_counter()
                from auto_round.compressors.utils import immediate_pack_block as _immediate_pack_block

                _immediate_pack_block(block, block_name, layer_config, nblocks=nblocks, device=load_device)
                holder["pack"] = _time.perf_counter() - _t0
                if self.compress_context.is_immediate_saving:
                    _t0 = _time.perf_counter()
                    holder["snap"] = (
                        self._write_finished_block_(
                            block, block_name, tied_weights_layers, rs, q_snap, fp_snap, is_model_last
                        )
                        or 0.0
                    )
                    holder["write"] = _time.perf_counter() - _t0
                    if envs.AR_PERF_COUNTERS:
                        logger.info(
                            "[stream] bg pack+write %s: pack %.1fs write %.1fs snapshot %.1fs",
                            block_name,
                            holder["pack"],
                            holder["write"],
                            holder["snap"],
                        )
                else:
                    mv_module_from_gpu(block)
                    if envs.AR_PERF_COUNTERS:
                        logger.info("[stream] bg pack %s: pack %.1fs", block_name, holder["pack"])
            except BaseException as e:  # noqa: BLE001 - re-raised at join
                holder["exc"] = e
            # NOTE: deliberately NO clear_memory()/gc here: empty_cache +
            # gc.collect() are process-wide and firing them from this thread
            # while the main loop's CUDA kernels are in flight corrupted
            # in-flight accesses on the server (async illegal-memory-access
            # surfacing in an unrelated allocation). The join point clears.

        t = _threading.Thread(target=_worker, daemon=True, name=f"bg-pack-{block_name}")
        t.autoround_state = holder
        t.start()
        # the exception-path teardown must be able to reach this worker: an
        # orphaned pack thread keeps writing to the (lock-free) shard writer
        # and a same-object catch-and-rerun would race it
        self._bg_pack_thread = t
        return t

    @staticmethod
    def _resolve_bg_finish_mode(blob_mode: bool, mode: str) -> bool:
        """Whether the blob-mode finish worker (meta-park + resume snapshot)
        overlaps on a background thread.

        Blob-mode PACKING must stay serial (the conversion instance carries
        shared mutable state), but the finish tail touches no instance state
        and does no GPU math - only D2H copies and file writes - so it needs
        no second staging device. Only an explicit "off" disables it.
        """
        if mode == "off":
            return False
        return blob_mode

    def _start_bg_finish_block(
        self, block, block_name: str, tied_weights_layers: set, rs, q_snap, fp_snap, is_model_last: bool
    ):
        """Finish a packed blob block on a background thread.

        Sibling of :func:`_start_bg_pack_block` for the blob path, where the
        pack (conversion-instance prepare_tensors) has ALREADY run serially in
        the main loop: the worker only runs the finish tail - meta-park plus
        the crash-resume snapshot (``mark_block_done`` persists the q/fp chain
        frontier, ~10 GB D2H + disk per block on a 27B) so that cost overlaps
        with the next block's tune. ``q_snap``/``fp_snap`` are captured refs to
        the successor chain rows (the loop replaces dict entries on advance,
        never mutating the tensors in place - the same contract the pack
        pipeline relies on). Crash window safety: the blob flush already
        happened at pack tail, so ``done`` still implies durable shards; a
        crash before the worker's mark-done simply re-quantizes one block and
        the blob adoption dedupes its shards.
        """
        import threading as _threading

        holder = {"exc": None, "write": 0.0, "snap": 0.0}

        def _worker():
            import time as _wtime

            try:
                holder["write"] = (
                    self._write_finished_block_(
                        block, block_name, tied_weights_layers, rs, q_snap, fp_snap, is_model_last
                    )
                    or 0.0
                )
                if envs.AR_PERF_COUNTERS:
                    logger.info("[stream] bg finish %s: write %.1fs (snapshot)", block_name, holder["write"])
            except BaseException as e:  # noqa: BLE001 - re-raised at join
                holder["exc"] = e
                # same rule as the pack pipeline: no process-wide cache clears
                # from this thread while the main loop's kernels are in flight

        t = _threading.Thread(target=_worker, daemon=True, name=f"bg-finish-{block_name}")
        t.autoround_state = holder
        t.start()
        # exception-path teardown reaches the worker through the same attr:
        # at most one background block-pipeline thread exists at any moment
        self._bg_pack_thread = t
        return t

    @staticmethod
    def _resolve_bg_pack_mode(mode: str, stage_device_count: int, immediate_packing: bool) -> bool:
        """Resolve AR_STREAM_BG_PACK (auto|1|0) against pipeline support.

        "auto" runs the pipeline whenever supported (streaming with >=2
        staging devices and immediate packing); "1" requires it and fails
        loudly when unsupported rather than silently serializing; "0"
        serializes the pack into the main loop.
        """
        supported = bool(stage_device_count >= 2 and immediate_packing)
        if mode == "off":
            return False
        if mode == "on" and not supported:
            raise ValueError(
                "AR_STREAM_BG_PACK=1 requires the background pack pipeline to be supported: "
                f"--stream_quantization with >=2 staging devices and immediate packing "
                f"(stage_devices={stage_device_count}, immediate_packing={immediate_packing})"
            )
        return supported

    @staticmethod
    def _main_loop_may_move_block_off_gpu(is_immediate_saving: bool) -> bool:
        """Whether the streaming loop itself may move the finished block off the GPU.

        Only the non-immediate-saving path does: with immediate saving the
        block's lifecycle is owned elsewhere (serial write inline, or the
        background pack worker when the pipeline is active). Moving it from
        the main loop while the worker compresses races the pack - weights
        dragged toward cpu while the search's scale/zero-point attributes
        stay on the home device.
        """
        return not is_immediate_saving

    @staticmethod
    def _join_bg_pack(thread) -> None:
        """Join a background pack pipeline, surfacing any worker failure."""
        thread.join()
        clear_memory()  # single-threaded again: safe to release cached blocks
        holder = getattr(thread, "autoround_state", None) or {}
        exc = holder.get("exc")
        if exc is not None:
            raise RuntimeError(
                f"background pack pipeline for a block failed ({exc!r}); refusing to continue -- "
                "the checkpoint would silently miss packed tensors. Fix the underlying failure "
                "or set AR_STREAM_BG_PACK=0."
            ) from exc

    @staticmethod
    def _release_cuda_cache(reason: str = "") -> None:
        """Release reserved-but-unallocated accelerator segments (resume-rebuild debris).

        The resume rebuild (chain jump, shard adoption, hydration) frees
        transient accelerator allocations whose segments stay cached by the
        allocator; the first post-resume block then OOMs on fragmentation
        even though its live set fits comfortably (measured: ~6.8G reserved
        but unallocated right before an OOM on a block the fresh run had
        quantized in the same co-located placement). Backend-agnostic via
        clear_memory (cuda/xpu/hpu/mps); a failure here must surface rather
        than defer the symptom to a confusing later OOM.
        """
        clear_memory()  # no device_list: resolves to the run's configured devices
        if reason and logger.isEnabledFor(logging.DEBUG):
            from auto_round.utils.device_manager import get_current_device_manager

            parts = []
            dev_mgr = get_current_device_manager()
            for idx in range(dev_mgr.device_count()):
                alloc = dev_mgr.memory_allocated(idx) / 2**30
                reserved = dev_mgr.memory_reserved(idx) / 2**30
                parts.append(f"{dev_mgr.type}:{idx} alloc {alloc:.2f}G reserved {reserved:.2f}G")
            logger.debug("[stream] accelerator state after %s: %s", reason, "; ".join(parts))

    def _release_cached_segments_if_fragmented(self, device, min_gap_bytes: int = 2 * 2**30) -> bool:
        """Return provably-free cached segments on ``device`` at a phase boundary.

        A block-sized transient on the staging path leaves its segments in
        the allocator's cache; the next block's tuning then OOMs on memory
        that is free but reserved. Checked once per block right after load
        (before any tuning allocation): when the free-in-reserve gap exceeds
        ``min_gap_bytes`` the cached segments are released. Pure allocator
        hygiene - no live tensor is touched.
        """
        try:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type == "cpu":
                return False
            from auto_round.utils.device_manager import get_ar_device

            dev_mgr = get_ar_device(dev.type)
            if not dev_mgr.is_available():
                return False
            idx = dev.index if dev.index is not None else dev_mgr.current_device()
            gap = dev_mgr.memory_reserved(idx) - dev_mgr.memory_allocated(idx)
            if gap < min_gap_bytes:
                return False
            dev_mgr.synchronize(idx)
            dev_mgr.empty_cache()
            return True
        except Exception:  # noqa: BLE001  diagnostics only; never break the run
            return False

    @staticmethod
    def _trim_host_heap() -> bool:
        """Best-effort glibc malloc_trim(0): hand freed host-heap pages back
        to the OS.

        The streaming reader allocates many small per-tensor buffers while
        the long-lived calibration chain stays resident; freed buffers end up
        scattered below live allocations, so the allocator keeps the pages
        mapped and RSS (and the VmHWM peak) grows monotonically even though
        nothing references the memory. A trim after each block collapses
        that growth. Returns True when a trim actually ran.
        """
        try:
            import ctypes

            return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
        except Exception:  # noqa: BLE001  diagnostics only; never break the run
            return False

    @staticmethod
    def _peak_rss_gb():
        """Kernel high-water mark RSS (VmHWM) when available, else None.

        Sampled inventories fire between tuning phases and miss the
        during-tuning transient peak; the kernel counter catches it for free.
        """
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 2**20  # kB -> GiB
        except OSError:
            pass
        return None

    @staticmethod
    def _mem_bucket(name: str) -> str:
        """Bucket a module-path name for the device-memory inventory."""
        if ".layers." in name:
            return "block:" + name.split(".layers.")[1].split(".")[0]
        if "embed" in name.lower():
            return "embeddings"
        return "nonblock:" + name.split(".")[0]

    def _build_resume_states(self, all_blocks: list) -> list:
        """Build one ResumeState per block group under AR_RESUME_DIR.

        Shared by the serial tuning path and the streaming zero-shot loop so
        the per-group manifests (signature strings, group_{idx} layout) are
        byte-compatible: a manifest advanced by one execution mode is
        consumed by the other and vice versa.
        """
        from auto_round.utils.resume import ResumeState, compute_run_signature, layer_config_fingerprint

        model_dir = getattr(self.model_context, "disk_stream_model_dir", None) or getattr(
            getattr(self.model_context.model, "config", None), "_name_or_path", None
        )
        dataset_desc = str(getattr(self, "dataset", None))
        # str(self.scheme) alone is bits-blind for AutoScheme runs: two runs
        # with different avg_bits share it, so include the resolved
        # per-layer allocation (see layer_config_fingerprint docstring). The
        # qi= component keeps enable_quanted_input (chain semantics) from
        # silently sharing manifests across different modes.
        scheme_desc = (
            str(self.scheme)
            + "|"
            + layer_config_fingerprint(getattr(self, "layer_config", None))
            + "|qi="
            + str(bool(self.alg_composer.need_quanted_input()))
        )
        states = []
        for group_idx, block_names in enumerate(all_blocks):
            sig = compute_run_signature(
                model_dir,
                scheme_desc,
                dataset_desc,
                self.calibration_context.nsamples,
                self.calibration_context.seqlen,
                block_names,
            )
            states.append(ResumeState(os.path.join(envs.AR_RESUME_DIR, f"group_{group_idx}"), sig, block_names))
        return states

    def _largest_block_bytes(self) -> float:
        """Largest block's checkpoint-tensor footprint from the meta skeleton (bytes)."""
        try:
            block_names = [n for sub in get_block_names(self.model, quant_vision=True) for n in sub]
        except Exception:  # noqa: BLE001  sizing must never break staging
            return float("inf")
        per_block: dict = {}
        for name, t in self.model.named_parameters():
            for b in block_names:
                if name == b or name.startswith(b + "."):
                    per_block[b] = per_block.get(b, 0) + t.numel() * t.element_size()
                    break
        if not per_block:
            return float("inf")
        return float(max(per_block.values()))

    def _primary_fits_largest_block(self, quant_dev: torch.device):
        """(largest_block_GiB, free_GiB) when the primary GPU can join the auto
        rotation, else None.

        Fit rule: free VRAM (queried live) - 3 GiB headroom >= largest block.
        The headroom covers tuning transients on top of the block (batch
        slices, in-place qdq, pack buffers). Conservative on purpose: the
        largest block gates EVERY block's home, and sizes vary by layer type.
        """
        if quant_dev.type != "cuda" or quant_dev.index is None:
            return None
        try:
            free_b, _total_b = torch.cuda.mem_get_info(quant_dev.index)
        except Exception:  # noqa: BLE001
            return None
        largest = self._largest_block_bytes()
        free_gb = free_b / 2**30
        if not largest < float("inf") or free_gb - 3.0 < largest / 2**30:
            return None
        return largest / 2**30, free_gb

    def _log_device_inventory(self, calib_state: dict, tag: str, extra_buckets: dict = None) -> None:
        """DEBUG-gated diagnostic: per-GPU breakdown of the streaming parent's memory.

        Walks model tensors (meta skipped, deduped), the calibration chain
        state, and compares against the allocator's view; the residual
        ("other") captures temporaries, packing buffers and optimizer state.
        """
        import collections

        seen: set = set()
        _tag = f"{tag} " if tag else ""
        per_dev: dict = collections.defaultdict(lambda: collections.defaultdict(int))
        top_the = 0.1 * 2**30  # list tensors >= 0.1G alongside the buckets
        big: list = []  # (nbytes, "dev:name") tuples when the inventory threshold is set

        def _add(dev: str, bucket: str, t, name: str = None) -> None:
            nbytes = t.numel() * t.element_size()
            per_dev[dev][bucket] += nbytes
            if top_the and nbytes >= top_the:
                big.append((nbytes, f"{dev}:{name or bucket}"))

        for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            if t.device.type not in ("cuda", "cpu") or id(t) in seen:
                continue
            seen.add(id(t))
            _add(str(t.device), self._mem_bucket(name), t, name)

        def _walk(v, bucket="chain", prefix=None, depth=0):
            if isinstance(v, torch.Tensor):
                if v.device.type in ("cuda", "cpu") and id(v) not in seen:
                    seen.add(id(v))
                    _add(str(v.device), bucket, v, prefix)
            elif isinstance(v, dict):
                # snapshot: the bg pack worker mutates composer-held dicts
                # concurrently; iterating them live raises mid-walk
                for k, x in list(v.items()):
                    _walk(x, bucket, f"{prefix}.{k}" if prefix else str(k), depth)
            elif isinstance(v, (list, tuple)):
                for j, x in enumerate(list(v)):
                    _walk(x, bucket, f"{prefix}[{j}]" if prefix else f"[{j}]", depth)
            elif (
                depth < 4
                and v is not self.model
                and not isinstance(v, torch.nn.Module)
                and hasattr(v, "__dict__")
                and not isinstance(v, (str, bytes, type(None), int, float, bool))
            ):
                # plain holder objects (quantizer configs, collectors): walk
                # their fields so live-but-untracked tensors get a bucket
                for k, x in list(vars(v).items()):
                    _walk(x, bucket, f"{prefix}.{k}" if prefix else str(k), depth + 1)

        if calib_state is not None:
            for key in ("fp_inputs", "q_inputs"):
                _walk(calib_state.get(key))
            # masks / position ids / rope tables: persistent small residents
            _walk(calib_state.get("input_others"), bucket="chain-kwargs")
        _walk(getattr(self, "alg_composer", None), bucket="quantizer")
        for bucket_name, payload in (extra_buckets or {}).items():
            _walk(payload, bucket=bucket_name)
        if self.shard_writer is not None:
            pending = getattr(self.shard_writer, "current_shard_tensors", None) or {}
            # the bg pack worker inserts into this dict concurrently; iterate
            # a stable copy (same hazard the composer walk snapshots against)
            for _t in list(pending.values()):
                if isinstance(_t, torch.Tensor) and _t.device.type in ("cuda", "cpu") and id(_t) not in seen:
                    seen.add(id(_t))
                    _add(str(_t.device), "shard-pending", _t)

        # host-side breakdown: tracked CPU tensors vs process RSS, so the
        # residual (allocator cache / fragmentation / untracked holders) is
        # visible next to the deliberate residents (chain, masks, rope)
        try:
            import psutil

            rss_gb = psutil.Process().memory_info().rss / 2**30
            peak_gb = self._peak_rss_gb()
            host_buckets = per_dev.get("cpu", {})
            host_parts = _format_host_buckets(host_buckets)
            tracked_gb = sum(v for v in host_buckets.values()) / 2**30
            # drill-downs ride the summary line (no separate per-drill lines):
            # top host tensors name the big tracked residents; regions classify
            # the RESIDUAL (dead memory holds no tensor objects): [heap] =>
            # allocator fragmentation; anonymous mappings => CUDA pinned pools
            # / torch host caches; file-backed => checkpoint mmaps
            trailing = ""
            if top_the:
                top_parts = _fmt_mem_top(big, "cpu")
                if top_parts:
                    trailing += f" | top: {top_parts}"
                regions = [m for m in psutil.Process().memory_maps(grouped=False) if m.rss > 0]
                regions.sort(key=lambda m: -m.rss)
                region_parts = _fmt_mem_regions(regions)
                if region_parts:
                    trailing += f" | regions: {region_parts}"
            logger.info(
                "[stream-mem] %shost: rss %.2fG (peak %.2fG) | %s | residual(rss-tracked) %.2fG%s",
                _tag,
                rss_gb,
                peak_gb,
                host_parts or "no tracked cpu tensors",
                max(0.0, rss_gb - tracked_gb),
                trailing,
            )
        except Exception:  # noqa: BLE001  diagnostics must never break the run
            pass
        for idx in range(torch.cuda.device_count()):
            dev = f"cuda:{idx}"
            alloc = torch.cuda.memory_allocated(idx) / 2**30
            reserved = torch.cuda.memory_reserved(idx) / 2**30
            buckets = sorted(per_dev.get(dev, {}).items(), key=lambda kv: -kv[1])
            parts = ", ".join(f"{k}={v / 2**30:.2f}G" for k, v in buckets if v > 0)
            tracked = sum(v for _, v in buckets)
            if tracked == 0 and alloc == 0.0:
                # idle GPU (not mapped / nothing staged): all-zero lines are
                # noise; any nonzero usage still shows below
                continue
            top_parts = _fmt_mem_top(big, dev) if top_the else ""
            other = max(0.0, alloc - tracked / 2**30)
            logger.info(
                "[stream-mem] %s%s: alloc %.2fG / reserved %.2fG | %s | other(alloc-tracked) %.2fG%s",
                _tag,
                dev,
                alloc,
                reserved,
                parts or "no tracked tensors",
                other,
                f" | top: {top_parts}" if top_parts else "",
            )

    def _tuning_headroom_profile(self):
        """(iters, moe_routing_bytes) sizing the staging search headroom.

        iters: max across the block quantizers (a config without the field is
        zero-shot by construction). moe_routing_bytes: one chunked forward
        keeps a [tokens, top_k, hidden] fp32 router-affinity buffer. Returns
        (None, None) for a MoE model whose shape cannot be derived so the
        streamer keeps the conservative default rather than under-reserving.
        """
        iters = self._max_tune_iters()
        routing = None
        cfg = self.model_context.config
        if cfg is not None and is_moe_model_via_config(cfg):
            # VL/MoE families nest the text backbone's fields on text_config;
            # num_experts_per_tok is the transformers-wide convention (family
            # attribute_maps remap local spellings onto it), moe_top_k is dbrx
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                cfg = text_cfg
            hidden = getattr(cfg, "hidden_size", None)
            topk = None
            for attr in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "n_experts_per_tok"):
                topk = getattr(cfg, attr, None)
                if topk:
                    break
            batch = getattr(self.calibration_context, "batch_size", None)
            seqlen = getattr(self.calibration_context, "seqlen", None)
            if hidden and topk and batch and seqlen:
                routing = int(batch) * int(seqlen) * int(topk) * int(hidden) * 4
            else:
                return None, None
        return iters, routing

    def _stream_mapped_enabled(self) -> bool:
        """Mapped placement engages when the device map is a module template
        or an alternate prefetch map is given. Plain device lists keep the
        prefetch-rotation semantics while prefetch is on; with prefetch OFF
        there is no rotation to keep, and single-homing a large block on the
        primary would OOM - so a plain multi-device list means mapped
        placement. The alternate map is only legal with stream_prefetch
        enabled - enforced at startup."""
        if getattr(self, "stream_prefetch_device_map", None):
            return True
        from auto_round.utils.stream_placement import is_placement_template

        base_map = getattr(device_manager, "device_map", None)
        if is_placement_template(base_map):
            return True
        if str(getattr(self, "stream_prefetch", "off") or "off").strip().lower() not in STREAM_PREFETCH_OFF:
            return False
        devices = [d for d in str(base_map or "").split(",") if d.strip()]
        return len(devices) > 1

    @staticmethod
    def _pin_stream_mapped_(block, placement: dict) -> None:
        """Bind a mapped streamed block to its per-leaf devices.

        Sets ``tuning_device`` from the placement (wrappers prefer it:
        ``wrapper.py self.device = orig_layer.tuning_device or device``) and
        attaches per-leaf ``AlignDevicesHook(io_same_device=True)`` so chain
        rows, mask kwargs and activations hop to each module's device during
        forwards - without this, any block forward (mask-form probe, chain
        pass, tuning) mixes devices and fails on the first cross-device op.
        """
        from auto_round.compressors.utils import attach_stream_align_

        multi = len(device_manager.device_list) > 1
        for name, mod in block.named_modules():
            if list(mod.children()):
                continue
            if not any(True for _ in mod.parameters(recurse=False)):
                continue
            dev = placement.get(name)
            if dev is None:
                continue
            mod.tuning_device = torch.device(dev)
            if multi:
                # own lightweight align hook: no attach-time tensor moves
                # (accelerate's init_hook re-moves weights and re-wraps
                # parameters - a second move set racing the rehome), and
                # idempotent so the restage re-pin replaces instead of
                # chaining stale execution devices
                attach_stream_align_(mod, mod.tuning_device)

    def _mapped_shared_groups_(self) -> list:
        """Modules kept together on one device while mapping a block.

        Two sources, same placement semantics: comma-key layer_config
        entries (shared-quantization groups - scales/zeros are searched on
        the merged tensor, so the group MUST sit on one device) and
        ``shared_layers`` groups (same-quantization-scheme declarations -
        keeping them whole is harmless and matches the fused-kernel
        intent). Matching is by leaf basename per parent, block-wise."""
        groups = [
            [part.strip() for part in key.split(",") if part.strip()]
            for key in (self.layer_config or {})
            if isinstance(key, str) and "," in key
        ]
        for group in getattr(self, "shared_layers", None) or []:
            members = [str(m).strip() for m in (group or []) if str(m).strip()]
            if len(members) >= 2:
                groups.append(members)
        return groups

    def _make_mapped_resolver(self, flat_block_names: list) -> dict:
        """Lazy, thread-safe per-block placement resolution for mapped runs.

        Staged parity alternates templates: even blocks use the base map,
        odd blocks the ``stream_prefetch_device_map`` map (default: same
        map). Device-list maps additionally derive a placement from the
        block's reference forward (the first full pass through the real
        module structure) via a one-shot probe, cache it under the block's
        structure signature (digit-stripped leaf layout), and re-place the
        block BEFORE tuning state exists - so identical layouts reuse the
        flow-ordered placement and even the deriving block tunes under it.
        Registration order is only the pre-derivation fallback. Both the
        main loop and the prefetch reader thread resolve, hence the lock.
        """
        import threading

        from auto_round.utils.stream_placement import (
            FlowProbe,
            _atomic_groups,
            _leaf_param_bytes,
            _normalize_device,
            _shared_atoms,
            block_signature,
            complete_container_params,
            is_placement_template,
            partition_flow_order,
            resolve_block_placement,
        )

        base_map = getattr(device_manager, "device_map", None)
        next_map = getattr(self, "stream_prefetch_device_map", None) or base_map
        fallback = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        shared_groups = self._mapped_shared_groups_()
        state = {"lock": threading.RLock(), "placements": {}, "flow_templates": {}, "pending_restage": None}
        logged = {"engaged": False}

        def _template_for(flat_idx: int):
            return next_map if flat_idx % 2 else base_map

        def _template_devices(template) -> list:
            if is_placement_template(template):
                return []  # hand-written template: placement is the user's
            devices = [d.strip() for d in str(template).split(",") if d.strip()]
            return devices or []

        def _resolve(block_name: str) -> dict:
            with state["lock"]:
                cached = state["placements"].get(block_name)
            if cached is not None:
                return cached
            flat_idx = flat_block_names.index(block_name)
            template = _template_for(flat_idx)
            block = get_module(self.model, block_name)
            leaf_names = [
                n
                for n, m in block.named_modules()
                if not list(m.children()) and any(True for _ in m.parameters(recurse=False))
            ]
            devices = _template_devices(template)
            sig = block_signature(block)
            key = (sig, tuple(str(d) for d in devices)) if devices else None
            flow = state["flow_templates"].get(key) if key else None
            if flow is not None:
                placement = {leaf: flow[leaf] for leaf in leaf_names if leaf in flow}
                for leaf in leaf_names:
                    placement.setdefault(leaf, fallback)
                # the leaf filter above drops the template's container keys
                # (co-located container-direct params, e.g. GDN A_log) -
                # re-derive them for THIS block's module tree
                placement = complete_container_params(placement, block)
                if not logged.get("reused"):
                    logged["reused"] = True
                    logger.info(
                        "[stream-mapped] reusing flow-derived placement (layout %s..., %d devices)",
                        sig[:24],
                        len(devices),
                    )
            else:
                placement = resolve_block_placement(block, template, fallback, shared_leaf_groups=shared_groups)
            if not logged.get("engaged"):
                logged["engaged"] = True
                logger.info(
                    "[stream-mapped] placement engaged: %d block(s); device-list maps derive the "
                    "placement from each layout's first reference forward; rows park on host RAM",
                    len(flat_block_names),
                )
            with state["lock"]:
                state["placements"][block_name] = placement
            return placement

        def _derive(block, records, devices, dev_objs):
            leaf_names = [
                n
                for n, m in block.named_modules()
                if not list(m.children()) and any(True for _ in m.parameters(recurse=False))
            ]
            get_mod = dict(block.named_modules())
            atom_of = {}
            for gkey, leaves in _atomic_groups(leaf_names):
                for leaf in leaves:
                    atom_of[leaf] = ("c", gkey)
            for atom in _shared_atoms(leaf_names, shared_groups):
                for leaf in atom:
                    atom_of[leaf] = ("s", tuple(sorted(atom)))
            units = []
            seen_atoms = set()
            fired = set()
            for rel, in_bytes in records:
                if rel in fired:
                    continue
                fired.add(rel)
                atom = atom_of.get(rel)
                if atom is not None:
                    if atom in seen_atoms:
                        continue
                    seen_atoms.add(atom)
                    members = [n for n in leaf_names if atom_of.get(n) == atom]
                    units.append((members, sum(_leaf_param_bytes(get_mod[n]) for n in members), in_bytes, True))
                else:
                    units.append(([rel], _leaf_param_bytes(get_mod[rel]), in_bytes, False))
            for leaf in leaf_names:  # never executed: place for balance
                if leaf not in fired:
                    units.append(([leaf], _leaf_param_bytes(get_mod[leaf]), 0, atom_of.get(leaf) is not None))
            placement = partition_flow_order(units, dev_objs)
            placement = complete_container_params(placement, block)
            key = (block_signature(block), tuple(str(d) for d in devices))
            with state["lock"]:
                state["flow_templates"][key] = placement
            logger.info(
                "[stream-mapped] derived placement from the reference forward (layout %s..., %d modules, "
                "%d devices); identical layouts reuse it",
                key[0][:24],
                len(placement),
                len(dev_objs),
            )
            return placement

        def _maybe_probe(block, block_name: str, has_forward: bool):
            """Install the reference-forward probe when derivation applies."""
            if not has_forward:
                return None
            flat_idx = flat_block_names.index(block_name)
            devices = _template_devices(_template_for(flat_idx))
            if not devices:
                return None
            dev_objs = [_normalize_device(d) for d in devices]
            key = (block_signature(block), tuple(str(d) for d in devices))
            with state["lock"]:
                if key in state["flow_templates"]:
                    return None

            def _on_complete(records):
                placement = _derive(block, records, devices, dev_objs)
                with state["lock"]:
                    state["pending_restage"] = {"block_id": id(block), "placement": placement}

            return FlowProbe(block, _on_complete)

        def _make_restage(block, block_name: str):
            """Build the fp->wrapper boundary callback that applies the derived
            placement to the deriving block itself (weights only - no tuning
            params or gradients exist at that point)."""

            def _restage():
                pending = state.get("pending_restage")
                if pending is None or pending["block_id"] != id(block):
                    return
                placement = pending["placement"]
                from auto_round.compressors.utils import rehome_block_mapped_

                # release the reference forward's activation pools first: the
                # moves allocate new blocks on the target devices while the
                # source copies free only after each rebind, so every byte of
                # reclaimable cache directly widens the transient headroom.
                # clear_memory is backend-agnostic (cuda/xpu/hpu/mps; cpu
                # entries are skipped). No swallow: a failure here (e.g. an
                # invalid device in the placement) must surface, not defer to
                # a confusing later OOM
                clear_memory(device_list=sorted({str(d) for d in placement.values()} | {str(fallback)}))
                rehome_block_mapped_(block, placement, fallback)
                # drain the moves before anything touches CUDA again: the
                # rehome enqueues async cross-device copies, and a fault in
                # them would otherwise surface at an unrelated later call
                # (the hook attach) with a misleading stack
                if torch.cuda.is_available():
                    for _dv in {str(d) for d in placement.values()} | {str(fallback)}:
                        if str(_dv).startswith("cuda"):
                            torch.cuda.synchronize(torch.device(str(_dv)))
                CompressionOrchestrator._pin_stream_mapped_(block, placement)
                block._stream_mapped = placement
                with state["lock"]:
                    state["placements"][block_name] = placement
                    state["pending_restage"] = None
                logger.debug("[stream-mapped] restaged %s onto its flow-derived placement before tuning", block_name)

            return _restage

        state["resolve"] = _resolve
        state["maybe_probe"] = _maybe_probe
        state["make_restage"] = _make_restage
        return state

    def _resolve_stream_stage_devices(self):
        """Resolve the ``stream_prefetch`` mode into the staging-device list.

        Returns None for host-RAM staging. "auto"/"on" pick ONE other CUDA
        device from the device map (all visible GPUs when no map is set); the
        quant device joins the rotation when its free VRAM fits the largest
        block, and the sole-GPU case stages on the quant device itself under
        the same fit rule -- host RAM is the last resort ("on" never silently
        disables). An explicit device string stages on that device. At most
        two homes ever rotate: lookahead is one block, so more devices would
        only spread block VRAM around.
        """
        mode = str(getattr(self, "stream_prefetch", "off") or "off").strip().lower()
        from auto_round.utils.stream_placement import is_placement_template

        _mapped = getattr(self, "stream_prefetch_device_map", None) is not None or is_placement_template(
            getattr(device_manager, "device_map", None)
        )
        if _mapped:
            # mapped placement owns staging homes per module; the rotation
            # list would contradict the template
            if mode not in STREAM_PREFETCH_OFF and mode not in ("auto", "cpu"):
                raise ValueError(
                    f"stream_prefetch={mode!r} conflicts with mapped placement: the device map "
                    "already fixes per-module staging targets (use off/auto/cpu)"
                )
            if mode not in STREAM_PREFETCH_OFF:
                logger.info("[stream-mapped] stream_prefetch=%s superseded by the placement map", mode)
            return None
        if mode in STREAM_PREFETCH_OFF:
            return None
        if mode == "cpu":
            return None
        quant_dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        forced = mode == "on"
        if mode in ("auto", "on"):
            ram_last_resort = (
                "[stream] prefetch fallback: staging in host RAM (RAM->GPU hop still beats an on-demand "
                "NVMe re-read)"
            )
            if quant_dev.type != "cuda":
                if forced:
                    logger.info("%s: quant device is %s, not CUDA", ram_last_resort, quant_dev)
                    return None  # host-RAM staging (devices None)
                logger.warning("[stream] stream_prefetch='auto' ignored: quant device is %s, not CUDA", quant_dev)
                return None
            n_gpu = torch.cuda.device_count()
            if n_gpu == 0:
                if forced:
                    logger.info("%s: no CUDA devices visible", ram_last_resort)
                    return None
                logger.warning("[stream] stream_prefetch='auto' ignored: no CUDA devices visible")
                return None
            # auto staging never reaches outside the user's device map: an
            # explicit --device_map is a sandbox declaration (other GPUs may
            # belong to other jobs), while the default (no --device_map)
            # resolves to every visible GPU, so nothing changes there
            allowed = [torch.device(str(d)) for d in device_manager.device_list if str(d).startswith("cuda")]
            pool = allowed or [torch.device("cuda", i) for i in range(n_gpu)]
            others = [d for d in pool if d != quant_dev]
            if others:
                devices = [others[0]]
                # The primary joins the rotation when its free VRAM comfortably
                # holds the LARGEST block (block sizes vary by layer type):
                # parent working set + block + tuning transients must coexist
                # there for one block at a time.
                if self._primary_fits_largest_block(quant_dev) is not None:
                    devices = [quant_dev, others[0]]
            elif self._primary_fits_largest_block(quant_dev) is not None:
                # sole GPU: it may stage on itself when the largest block fits
                # with tuning headroom
                devices = [quant_dev]
            else:
                logger.info("%s: largest block does not fit free VRAM on the sole GPU", ram_last_resort)
                return None
        else:
            try:
                dev = torch.device(mode)
            except RuntimeError as e:
                raise ValueError(
                    f"stream_prefetch {mode!r} is not a valid device; use off/auto/on/cpu or a device "
                    "string like 'cuda:1'"
                ) from e
            if dev.type == "meta":
                raise ValueError(f"invalid staging device {dev}: meta tensors hold no data")
            if quant_dev.type != "cuda" and dev.type == "cuda":
                logger.warning(
                    "[stream] GPU staging device ignored with CPU quant device %s; using host RAM", quant_dev
                )
                return None
            devices = [dev]
        logger.info(
            "[stream] staging one block ahead on %s; staged blocks quantize in place",
            [str(d) for d in devices],
        )
        return devices or None

    def _materialize_fused_expert_stack_(self, streamer, tensor_name: str, shape: tuple, entry: dict) -> int:
        """Unfuse one pinned 3D expert stack into per-expert placeholder Linears.

        Mirrors the family module replacement's materialization: a stacked
        ``[E, 2I, H]`` ``gate_up_proj`` splits into per-expert ``gate_proj`` /
        ``up_proj``, ``[E, H, I]`` ``down_proj`` becomes per-expert
        ``down_proj``. The per-expert paths inherit the pin's quantization
        entry (re-resolved per expert so precise pins keep working). Returns
        the number of materialized experts.
        """
        base, proj = tensor_name[: -len(".weight")].rsplit(".", 1)
        num_experts = int(shape[0])
        stack = streamer.fetch(tensor_name)
        made = 0
        for e in range(num_experts):
            if proj == "gate_up_proj":
                intermediate = int(shape[1]) // 2
                gate = stack[e, :intermediate, :].contiguous()
                up = stack[e, intermediate:, :].contiguous()
                for sub, t in (("gate_proj", gate), ("up_proj", up)):
                    path = f"{base}.{e}.{sub}"
                    sub_entry = self._pin_entry_for(path) or entry
                    materialize_placeholder_linear_from_tensor(self.model, path, t, sub_entry)
                made += 1
            else:  # down_proj / gate_proj / up_proj stacked plain
                path = f"{base}.{e}.{proj}"
                sub_entry = self._pin_entry_for(path) or entry
                materialize_placeholder_linear_from_tensor(self.model, path, stack[e].contiguous(), sub_entry)
                made += 1
        return made

    def _max_tune_iters(self) -> int:
        """Highest iters across the block quantizers (0 when closed-form)."""
        quantizers = self.alg_composer.block_quantizer
        if not isinstance(quantizers, (list, tuple)):
            quantizers = [quantizers]
        return max(int(getattr(q, "iters", 0) or 0) for q in quantizers) if quantizers else 0

    def _text_config(self):
        """The text-backbone config (VL composites nest it under text_config)."""
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        return text_cfg if text_cfg is not None else cfg

    def _pin_entry_for(self, layer_path: str) -> Optional[dict]:
        """Quantization entry pinning *layer_path*, or None.

        The resolver expands pins only over module-side names (checkpoint-only
        layers cannot receive plan entries), so pins for those live solely in
        regex_config: match the path against every pattern (exact pins are
        normalized to regexes too) and return the first hit. Each pattern is
        also tried against the layer's tensor spelling (``path + ".weight"``):
        a pin written with the tensor dot (``.*mtp.fc.``) matches the tensor
        but never the bare module path, and losing it to a broader module-side
        pattern would silently quantize a layer the user pinned float."""
        from auto_round.utils.common import to_standard_regex

        entry = self.layer_config.get(layer_path)
        if isinstance(entry, dict):
            return entry
        candidates = (layer_path, f"{layer_path}.weight")
        for pattern, val in (getattr(self, "regex_config", None) or {}).items():
            try:
                rx = to_standard_regex(pattern)
                if any(re.search(rx, c) for c in candidates):
                    return val
            except re.error:  # pragma: no cover - malformed user pattern
                continue
        return None

    def _checkpoint_only_groups_(self, streamer) -> list:
        """Maximal checkpoint tensor subtrees with no module counterpart.

        Families ship multi-token-prediction weights in different topologies:
        a top-level ``mtp.*`` subtree the modeling class strips at load, a
        ``model.mtp.*`` subtree one level down, or an extra digit-indexed
        decoder block beyond the model's block list. All reduce to one derived
        rule: for every tensor nothing claimed, the first path prefix absent
        from the module tree anchors a checkpoint-only group. Conversion-
        registry renames (e.g. a shared expert stored under a legacy prefix)
        are claimed through their rewrite target and never form groups.
        """
        claimed = set()
        for n, m in self.model.named_modules():
            for leaf in list(m._parameters) + list(m._buffers):
                resolved = streamer.resolve_checkpoint_name(f"{n}.{leaf}" if n else leaf)
                if resolved is not None:
                    claimed.add(resolved)
        module_names = {n for n, _ in self.model.named_modules()}
        groups = set()
        for t in streamer.tensor_names:
            if t in claimed:
                continue
            parts = t.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                if prefix not in module_names:
                    groups.add(prefix)
                    break
        return sorted(groups)

    def _analyze_checkpoint_only_group_(self, streamer, group) -> Optional[dict]:
        """Classify a checkpoint-only group into predictor roles by shape and
        name keywords alone (no family registry): the group-level 2D
        ``[hidden, 2*hidden]`` weight is the concat mixer; 1D ``[hidden]``
        vectors with an embedding/hidden keyword are the prologue norms; any
        remaining 1D vector is the final norm; the decoder-layer subtree root
        is found by descending while every deep tensor shares one head
        component. Returns None when the group does not match the uniform
        predictor pattern (families all verified: e-norm cat h-norm -> mixer
        -> one decoder layer -> final norm -> shared head)."""
        tensors = streamer.names_under(group)
        direct, deep = {}, []
        for n in tensors:
            rel = n[len(group) + 1 :]
            if rel.count(".") == 1 and rel.endswith(".weight"):
                direct[rel[: -len(".weight")]] = n
            elif rel.count(".") >= 2:
                deep.append(n)
        if not deep:
            return None
        text_cfg = self._text_config()
        cfg = text_cfg
        hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size", None)
        if not isinstance(hidden, int):
            return None
        fc = norm_e = norm_h = final_norm = None
        for leaf, full in direct.items():
            meta = streamer.tensor_meta(full)
            shape = meta[0] if meta else None
            low = leaf.lower()
            if shape is not None and len(shape) == 2 and fc is None and shape[0] == hidden and shape[1] == 2 * hidden:
                fc = full
            elif shape is not None and len(shape) == 1 and shape[0] == hidden:
                if norm_e is None and ("embed" in low or low.endswith("enorm") or low.startswith("e_")):
                    norm_e = full
                elif norm_h is None and ("hidden" in low or low.endswith("hnorm") or low.startswith("h_")):
                    norm_h = full
                elif final_norm is None:
                    final_norm = full
        if fc is None or norm_e is None or norm_h is None:
            return None
        rels = [n[len(group) + 1 :] for n in deep]
        root = ""
        while True:
            heads = {r[len(root) + 1 :].split(".")[0] if root else r.split(".")[0] for r in rels}
            if len(heads) != 1:
                break
            nxt = f"{root}.{next(iter(heads))}" if root else next(iter(heads))
            if any(r == nxt for r in rels) or not all(r == nxt or r.startswith(nxt + ".") for r in rels):
                break
            root = nxt
        return {
            "prefix": group,
            "fc": fc,
            "norm_e": norm_e,
            "norm_h": norm_h,
            "final_norm": final_norm,
            "layer_root": f"{group}.{root}" if root else group,
        }

    def _resolve_group_param_source_(self, streamer, sibling_name, layer_root, param_rel):
        """Map one sibling param (relative name) to its checkpoint source
        under the group's layer root: the registry-resolved spelling of the
        sibling's own tensor first (``resolve_checkpoint_name``), then the
        identity spelling when the group side already uses it, then the
        fused expert-stack bridge. Returns ``("direct", ckpt_name)`` /
        ``("fused", fused_name, expert_idx, projection)`` or None."""
        rel = None
        resolved = streamer.resolve_checkpoint_name(f"{sibling_name}.{param_rel}")
        if resolved is not None and resolved.startswith(sibling_name + "."):
            rel = resolved[len(sibling_name) + 1 :]
        elif f"{layer_root}.{param_rel}" in streamer.weight_map:
            # the group side already spells the param exactly like the module
            # tree (identity families, or the sibling's own tensors are absent
            # from this checkpoint view)
            rel = param_rel
        if rel is not None:
            cand = f"{layer_root}.{rel}"
            if cand in streamer.weight_map:
                return ("direct", cand)
        m = _FUSED_STACK_RE.match(param_rel)
        if m:
            base, idx, proj = m.group(1), int(m.group(2)), m.group(3)
            fused = f"{base}.gate_up_proj.weight" if proj in ("gate_proj", "up_proj") else f"{base}.down_proj.weight"
            cand = f"{layer_root}.{fused}"
            if cand in streamer.weight_map:
                return ("fused", cand, idx, proj)
        return None

    def _pick_sibling_layer_(self, streamer, info, all_blocks, snapshots=None):
        """Pick the decoder block whose parameter set best matches the
        group's layer tensors and covers it completely (every sibling param
        resolves to a checkpoint source). Returns ``(score, block_name,
        module)`` or None; families whose blocks differ (e.g. full-attention
        vs gated-delta-net) auto-select the right sibling by name overlap.
        ``snapshots`` holds pristine pre-quantization structure copies (the
        live tree is packed QuantLinear by the time this runs)."""
        from auto_round.utils.model import get_module

        layer_root = info["layer_root"]
        ckpt_set = {_canonical_group_leaf(n[len(layer_root) + 1 :]) for n in streamer.names_under(layer_root)}
        best = None
        for block in all_blocks:
            for bname in block:
                if bname == layer_root or bname.startswith(layer_root + ".") or layer_root.startswith(bname + "."):
                    continue  # never sibling against the group itself
                mod = (snapshots or {}).get(bname)
                if mod is None:
                    mod = get_module(self.model, bname)
                if mod is None or not any(True for _ in mod.children()) or not any(True for _ in mod.parameters()):
                    continue
                params = [rel for rel, _ in mod.named_parameters()]
                sources = [self._resolve_group_param_source_(streamer, bname, layer_root, rel) for rel in params]
                if any(s is None for s in sources):
                    continue  # incomplete coverage would leave a meta param
                score = len({_canonical_group_leaf(rel) for rel in params} & ckpt_set)
                if best is None or score > best[0]:
                    best = (score, bname, mod)
        return best

    def _attach_checkpoint_only_group_tree_(self, streamer, info, sibling_name, sibling) -> set:
        """Build a REAL module tree for the group: deep-copy the sibling
        block's (meta) structure as the predictor layer, attach the prologue
        modules (mixer Linear + RMSNorms), and load only what no later pass
        can load (fused expert slices; everything else stays meta for the
        outside-block/root pass-through streams). Pinned Linears receive the
        pin's quantization attributes so they quantize like any pinned layer.
        Returns the checkpoint tensor names the tree claimed."""
        group, layer_root = info["prefix"], info["layer_root"]
        claimed = set()
        layer_mod = copy.deepcopy(sibling)
        # the snapshot copies instance-level forwards too (positional adapters,
        # replacement wrappers) whose closures bind the ORIGINAL module -
        # calling them re-enters the source block instead of the copy. Restore
        # each module's class forward; modern block signatures take the same
        # keyword inputs the forward runner already passes.
        for m in layer_mod.modules():
            if "forward" in m.__dict__:
                cls_fwd = getattr(type(m), "forward", None)
                if cls_fwd is not None:
                    m.forward = cls_fwd.__get__(m, type(m))
        parent = _ensure_module_path(self.model, layer_root)
        parent.add_module(layer_root.rsplit(".", 1)[-1], layer_mod)
        for n, m in layer_mod.named_modules():
            m.global_name = f"{layer_root}{('.' + n) if n else ''}"

        # fused expert stacks have no per-expert checkpoint name: every
        # expert slice is assigned its real data now (one fetch per stack);
        # pinned experts quantize through the outside-block pass, unpinned
        # ones need the real weights so the tuning forward does not hit a
        # popped/meta parameter. A stack is only claimed when ALL of its
        # slices are pinned - a partially pinned stack stays the verbatim
        # checkpoint copy (it is the durable home of the unpinned experts).
        # Direct-name params stay meta - the outside-block pass loads them
        # right before quantizing, and the root pass-through streams the rest,
        # keeping peak host RAM at one stack.
        fused_cache = {}
        claimed_fused = set()
        partial_fused = set()

        def _fused_slice(fused_name, idx, proj, stack):
            if proj == "down_proj":
                return stack[idx].contiguous()
            inter = int(stack.shape[1]) // 2
            return stack[idx, :inter, :].contiguous() if proj == "gate_proj" else stack[idx, inter:, :].contiguous()

        for rel, p in list(layer_mod.named_parameters()):
            if p.device.type != "meta":
                continue
            src = self._resolve_group_param_source_(streamer, sibling_name, layer_root, rel)
            if src is None:
                raise RuntimeError(  # defensive: _pick_sibling_layer_ guarantees coverage
                    f"[stream] sibling param {rel!r} lost its checkpoint source while building {group!r}"
                )
            if src[0] != "fused":
                continue
            _, fused_name, idx, proj = src
            if fused_name not in fused_cache:
                fused_cache[fused_name] = streamer.fetch(fused_name)
            stack = fused_cache[fused_name]
            t = _fused_slice(fused_name, idx, proj, stack)
            owner_path = f"{layer_root}.{rel[: -len('.weight')]}"
            entry = self._pin_entry_for(owner_path)
            bits = entry.get("bits") if isinstance(entry, dict) else None
            pinned = isinstance(bits, int) and bits < 16
            streamer._assign_leaf_(layer_mod, rel, t)
            if pinned:
                claimed_fused.add(fused_name)
            else:
                partial_fused.add(fused_name)
        fused_cache.clear()
        claimed_fused -= partial_fused

        text_cfg = self._text_config()
        cfg = text_cfg
        hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size", None)
        eps = float(getattr(text_cfg, "rms_norm_eps", 1e-6) or 1e-6)

        def _pin_linear(path, module):
            entry = self._pin_entry_for(path)
            bits = entry.get("bits") if isinstance(entry, dict) else None
            if isinstance(module, torch.nn.Linear) and isinstance(bits, int) and bits < 16:
                _apply_pin_attrs(module, entry)

        for n, m in layer_mod.named_modules():
            if any(m.children()) or not isinstance(m, torch.nn.Linear):
                continue
            path = f"{layer_root}.{n}" if n else layer_root
            m.global_name = path
            _pin_linear(path, m)
            src = self._resolve_group_param_source_(streamer, sibling_name, layer_root, n + ".weight")
            if src is not None and src[0] == "direct":
                claimed.add(src[1])
        claimed |= claimed_fused

        # prologue: concat mixer + norms under their checkpoint paths
        fc_meta = streamer.tensor_meta(info["fc"])
        if fc_meta is None:
            raise RuntimeError(
                f"[stream] cannot read metadata for the predictor concat mixer {info['fc']!r} "
                "(unknown name or unreadable shard)"
            )
        fc_path = info["fc"][: -len(".weight")]
        with torch.device("meta"):
            fc = torch.nn.Linear(int(fc_meta[0][1]), int(fc_meta[0][0]), bias=False)
        fc.global_name = fc_path
        _ensure_module_path(self.model, fc_path).add_module(fc_path.rsplit(".", 1)[-1], fc)
        _pin_linear(fc_path, fc)
        claimed.add(info["fc"])
        for role in ("norm_e", "norm_h", "final_norm"):
            name = info.get(role)
            if name is None:
                continue
            path = name[: -len(".weight")]
            mod = CheckpointOnlyRMSNorm(hidden, eps)
            mod.global_name = path
            _ensure_module_path(self.model, path).add_module(path.rsplit(".", 1)[-1], mod)
            claimed.add(name)
        # bind the predictor forward onto the group shell so tuning machinery
        # can run the tree like a decoder block (embedding-side input bound
        # later, from the calibration chain). When the group IS the layer
        # (extra digit-block topology), the shell is the layer module itself:
        # binding the predictor forward overwrites the layer forward, so the
        # layer must be called through its restored class forward instead of
        # the module (calling the module would re-enter the predictor).
        shell = self.model.get_submodule(group)
        layer_call = layer_mod
        if shell is layer_mod:
            layer_call = type(layer_mod).forward.__get__(layer_mod, type(layer_mod))
        bind_checkpoint_only_predictor(
            shell,
            {
                "norm_e": self.model.get_submodule(info["norm_e"][: -len(".weight")]),
                "norm_h": self.model.get_submodule(info["norm_h"][: -len(".weight")]),
                "final_norm": (
                    self.model.get_submodule(info["final_norm"][: -len(".weight")]) if info.get("final_norm") else None
                ),
                "fc": fc,
                "layer": layer_call,
            },
        )
        return claimed

    def _tune_checkpoint_only_groups_(self, streamer, tree_groups, calib_state, block_count) -> set:
        """Tune materialized predictor trees with the run's tuning config.

        The tree joins the chain's tail as one extra block: hidden states are
        the final FP/quantized chain output, the embedding-side input is
        synthesized from the chain's token ids (shifted by one), positional
        inputs are reused from the chain's auxiliary inputs, and the SAME
        quantizer configuration tunes the tree's pinned Linears. Afterwards
        pack + shard-write mirror the block loop's tail. Returns the groups
        that tuned; a group that cannot (missing chain state) falls back to
        the closed-form outside-block search."""
        from auto_round.algorithms.composer import BlockContext
        from auto_round.compressors.utils import immediate_pack_block as _immediate_pack_block
        from auto_round.utils.model import check_to_quantized as _ctq
        from auto_round.utils.streaming_calibration import materialize_residual_meta

        tuned = set()
        fp_inputs = (calib_state or {}).get("fp_inputs")
        token_ids = (calib_state or {}).get("token_ids")
        if not fp_inputs or not token_ids:
            if tree_groups:
                logger.warning(
                    "[stream] checkpoint-only groups %s fall back to the closed-form search: the calibration "
                    "chain kept no token ids to synthesize predictor inputs from",
                    ", ".join(tree_groups),
                )
            return tuned
        embed = self.model.get_input_embeddings()
        embed_name = next((n for n, m in self.model.named_modules() if m is embed), None)
        if embed is None or embed_name is None:
            logger.warning(
                "[stream] checkpoint-only groups %s fall back to the closed-form search: no input embeddings",
                ", ".join(tree_groups),
            )
            return tuned
        if any(p.is_meta for p in embed.parameters()):
            # embedding lookup only - keep the (possibly vocabulary-sized)
            # table on host RAM, the synthesized rows move to the tune device
            streamer.load_module_(embed, embed_name, device="cpu")
        e_rows = [synthesize_predictor_e(ids, embed=embed) for ids in token_ids]
        cfg = self.model_context.model.config
        for group in tree_groups:
            shell = self.model.get_submodule(group)
            streamer.load_module_(shell, group, device=str(self.device))
            materialize_residual_meta(shell, cfg, self.device)
            # single-row fallback so direct calls (mask probes, spot checks)
            # work without the batched input plumbing
            shell._predictor_e = e_rows[0]
            io = dict(calib_state["input_others"])
            if calib_state.get("keymask_2d"):
                # resolve the predictor layer's attention-mask convention by
                # probe, exactly like the block loop (a full-attention
                # predictor behind gated-delta-net blocks must flip forms)
                from auto_round.utils.streaming_calibration import materialize_mask_form, resolve_chain_mask_form

                # the probe runs two rows: match the shell's single-row
                # fallback e-input to that batch so the predictor forward's
                # cat does not reject every form on shape
                _e_single = shell._predictor_e
                if isinstance(_e_single, torch.Tensor) and _e_single.dim() >= 1 and _e_single.shape[0] == 1:
                    shell._predictor_e = torch.cat([_e_single, _e_single], dim=0)
                try:
                    form = resolve_chain_mask_form(
                        shell,
                        _first_chain_row(fp_inputs),
                        calib_state["keymask_2d"][0],
                        io,
                        preferred=calib_state.get("_mask_form"),
                        amp=self.amp,
                        amp_dtype=self.amp_dtype,
                    )
                finally:
                    shell._predictor_e = _e_single
                io["attention_mask"] = [materialize_mask_form(m, form) for m in calib_state["keymask_2d"]]
            io["_predictor_e"] = e_rows
            ctx = BlockContext(
                model=self.model,
                block_names=[group],
                block_name=group,
                block_index=block_count,
                block_cnt=block_count + 1,
            )
            logger.info("[stream] tuning checkpoint-only group %s with the run's tuning config", group)
            try:
                self.alg_composer.compress_block(
                    shell,
                    fp_inputs,
                    io,
                    block_ctx=ctx,
                    q_inputs=calib_state.get("q_inputs"),
                    input_ids=token_ids,
                )
            finally:
                io.pop("_predictor_e", None)
            _immediate_pack_block(shell, group, self.layer_config, nblocks=self.nblocks, device=str(self.device))
            if self._gguf_blob_mode():
                # blob mode: no ggml pack happens here (the group is not a
                # block-last layer), and CT shards must not leak into the
                # GGUF output dir. Keep the tuned tree live: the save-time
                # prepare_tensors pass packs it from the in-memory qdq
                # weights straight into blob shards.
                pass
            else:
                self.shard_writer.write(name=group)
                self._write_unpacked_group_tensors_(streamer, [group])
                shell.to("meta")
            tuned.add(group)
            clear_memory()
        return tuned

    def _park_untuned_tree_shells_(self, tree_groups, mtp_tuned) -> None:
        """Park checkpoint-only group trees the tune path did not.

        Groups the tune path handled park themselves after packing. For the
        rest (zero-shot runs, or a tune that fell back early) the
        outside-block pass packed the pinned experts and the verbatim stack
        written by ``_write_unpacked_group_tensors_`` is the durable home of
        the unpinned ones - real per-expert slices left in the tree would be
        written a SECOND time by the finalize capture loop, under module
        names no checkpoint carries (so no dedup stops them)."""
        for group in tree_groups:
            if group not in mtp_tuned:
                self.model.get_submodule(group).to("meta")

    def _write_unpacked_group_tensors_(self, streamer, tree_groups) -> None:
        """Write every tree-group tensor that did not end up packed.

        Attaching a tree dissolves the group out of the checkpoint-only
        verbatim pass (its prefixes now exist in the module tree), so the
        tensors no path claimed would be silently dropped: tensors whose
        owner module quantized are replaced by their packed forms; a fused
        expert stack is dropped only when EVERY per-expert slice under it
        packed (a partially pinned stack stays the verbatim checkpoint copy
        - it is the durable home of the unpinned experts); everything else
        (norms, unpinned weights, extras) is copied through byte-for-byte
        from the checkpoint."""
        from auto_round.utils.model import check_to_quantized, get_module

        saved = set(getattr(self.shard_writer, "_all_saved", None) or [])
        for group in tree_groups:
            for n in streamer.names_under(group):
                if n in saved:
                    continue
                owner = get_module(self.model, n.rsplit(".", 1)[0])
                if owner is not None and check_to_quantized(owner):
                    continue  # its packed form was written by the outside-block pass
                meta = streamer.tensor_meta(n)
                if (
                    meta is not None
                    and len(meta[0]) == 3
                    and _is_fused_expert_weight_name(n)
                    and self._tree_fused_stack_fully_packed_(n, meta[0])
                ):
                    continue  # every per-expert slice packed; the stack is superseded
                self.shard_writer.save_tensor(n, streamer.fetch(n, raw=True))

    def _tree_fused_stack_fully_packed_(self, tensor_name: str, shape) -> bool:
        """True when every per-expert slice of a fused expert stack quantized.

        The tree spells per-expert modules (``experts.N.gate_proj`` ...) while
        the checkpoint stores the fused 3D stack, so the stack's own module
        lookup cannot answer this - walk the slices instead."""
        from auto_round.utils.model import check_to_quantized, get_module

        base, proj = tensor_name[: -len(".weight")].rsplit(".", 1)
        sub_projs = ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
        for e in range(int(shape[0])):
            for sub in sub_projs:
                m = get_module(self.model, f"{base}.{e}.{sub}")
                if m is None or not check_to_quantized(m):
                    return False
        return True

    def _materialize_pinned_checkpoint_only_blocks_(self, streamer, all_blocks, snapshots=None) -> tuple[set, list]:
        """Materialize pinned checkpoint-only blocks, preferring a REAL module
        tree (sibling structure snapshot + predictor prologue) over scattered
        placeholder Linears.

        Blocks whose tensors exist only in the checkpoint (an MTP layer the
        modeling code never instantiates) normally pass through verbatim.
        When the layer_config pins their layers for quantization: a group
        matching the uniform predictor pattern with a fully-covering decoder
        sibling becomes a real tree (pinned Linears quantize+pack through the
        outside-block pass; the tree is forward-capable for tuning runs);
        otherwise scattered placeholder Linears keep the pinned tensors
        quantizable. Tuning runs tune real trees with the run's own config;
        groups without a covering sibling cannot join the tuning forward, so
        their pinned layers quantize through the closed-form search instead.
        Returns ``(claimed tensor names, tree group prefixes)``.
        """
        claimed = set()
        tree_groups = []
        groups = self._checkpoint_only_groups_(streamer)
        skipped_non_2d = 0
        for blk in groups:
            names = sorted(streamer.names_under(blk))
            pinned_quantizable = []
            for n in names:
                if not n.endswith(".weight"):
                    continue
                entry = self._pin_entry_for(n[: -len(".weight")])
                bits = entry.get("bits") if isinstance(entry, dict) else None
                if not isinstance(bits, int) or bits >= 16:
                    continue
                meta = streamer.tensor_meta(n)
                if meta is not None and (len(meta[0]) == 2 or (len(meta[0]) == 3 and _is_fused_expert_weight_name(n))):
                    pinned_quantizable.append(n)
            if not pinned_quantizable:
                if any(
                    isinstance((e := self._pin_entry_for(n[: -len(".weight")])), dict)
                    and isinstance(e.get("bits"), int)
                    and e["bits"] < 16
                    for n in names
                    if n.endswith(".weight")
                ):
                    # a pin targeted this group but nothing quantizable matched
                    # (per-expert spellings the checkpoint never stores, 1D
                    # norms, non-weight tensors): the group ships verbatim
                    logger.warning(
                        "[stream] checkpoint-only group %s stays verbatim: its pin matched no quantizable "
                        "tensor (pins must target 2D weights or fused expert stacks by their checkpoint names)",
                        blk,
                    )
                continue  # unpinned or kept floating: verbatim path
            info = self._analyze_checkpoint_only_group_(streamer, blk)
            sibling = self._pick_sibling_layer_(streamer, info, all_blocks, snapshots) if info is not None else None
            if sibling is not None:
                # a real tree can join the chain's tail on tuning runs - the
                # tune step quantizes it with the run's own config
                tree_groups.append(blk)
                claimed |= self._attach_checkpoint_only_group_tree_(streamer, info, sibling[1], sibling[2])
                continue
            if info is not None:
                logger.warning(
                    "[stream] checkpoint-only group %s resembles a predictor block but no decoder sibling "
                    "covers its tensors; quantizing its pinned layers as scattered placeholders",
                    blk,
                )
            for n in names:
                if not n.endswith(".weight"):
                    continue
                layer_path = n[: -len(".weight")]
                entry = self._pin_entry_for(layer_path)
                bits = entry.get("bits") if isinstance(entry, dict) else None
                if not isinstance(bits, int) or bits >= 16:
                    continue  # unpinned or kept floating: verbatim path
                meta = streamer.tensor_meta(n)
                if meta is None:
                    continue
                shape = meta[0]
                if len(shape) == 3 and _is_fused_expert_weight_name(n):
                    # fused expert stack (e.g. [E, 2I, H] gate_up / [E, H, I]
                    # down): slice into per-expert placeholder Linears, the
                    # same split the family's module replacement applies to
                    # the main body - per-expert modules load by name, so the
                    # slices arrive as real weights instead of meta
                    self._materialize_fused_expert_stack_(streamer, n, shape, entry)
                    claimed.add(n)
                    continue
                if len(shape) != 2:
                    # norms and other buffers stay verbatim
                    skipped_non_2d += 1
                    continue
                bias = layer_path + ".bias"
                materialize_placeholder_linear(
                    self.model, layer_path, meta[0], entry, has_bias=bias in streamer.tensor_names
                )
                claimed.add(n)
                if bias in streamer.tensor_names:
                    claimed.add(bias)
        if tree_groups:
            logger.info(
                "[stream] built real module tree(s) for checkpoint-only group(s) %s; pinned layers quantize "
                "through the outside-block pass",
                ", ".join(tree_groups),
            )
        if claimed:
            logger.info(
                "[stream] materialized %d pinned layer(s) from checkpoint-only blocks; " "they will be quantized",
                len(claimed),
            )
        if skipped_non_2d:
            logger.warning(
                "[stream] %d pinned tensor(s) in checkpoint-only groups are not 2D weights (e.g. 1D norms or "
                "3D tensors whose names do not match the fused-expert patterns); leaving them unquantized",
                skipped_non_2d,
            )
        return claimed, tree_groups

    def _outside_block_quant_device(self) -> torch.device:
        """Device for the outside-block pass. The block loop has finished by
        then, so accelerators are idle: quantize there when one is available
        (materialized checkpoint-only groups can hold hundreds of expert
        Linears of compute-bound search); CPU otherwise."""
        try:
            d = torch.device(getattr(self.model_context, "device", "cpu"))
            if d.type != "cpu":
                return d
        except (RuntimeError, ValueError):
            pass
        return torch.device("cpu")

    @staticmethod
    def _chain_hidden_rows(chain_state):
        """A chain input/output as a plain list of per-sample row tensors.

        The chain keeps rows as a list, or a dict of per-key row lists for
        block classes registered in ``_BLOCK_OUTPUT_REGISTRY`` (e.g.
        gated-delta-net): take its ``hidden_states`` rows."""
        rows = chain_state.get("hidden_states") if isinstance(chain_state, dict) else chain_state
        if isinstance(rows, dict):
            rows = next(iter(rows.values()))
        return rows

    def _final_norm_module_(self, lm_head_name):
        """``(name, module)`` of the final norm feeding lm_head, or ``(None, None)``.

        lm_head consumes the POST-norm hidden states (``model.norm`` in
        Llama-family models); the chain tail holds the raw last-block output.
        Discovery is name-agnostic: a leaf module in the backbone that parents
        the blocks, after the last block and before lm_head, whose only
        parameters are a 1D weight (plus an optional 1D bias) - the
        LayerNorm/RMSNorm shape signature. The backbone restriction matters on
        wrapper models: vision towers and predictor placeholders also sit
        between the last text block and lm_head and carry same-signature (even
        same-width) norms."""
        blocks = get_block_names(self.model) or []
        # get_block_names returns groups of full block module names
        block_prefixes = [b for group in blocks for b in group]
        mods = list(self.model.named_modules())
        last_block_pos = -1
        positions = {}
        for idx, (n, _m) in enumerate(mods):
            positions[n] = idx
            if any(n == b or n.startswith(b + ".") for b in block_prefixes):
                last_block_pos = max(last_block_pos, idx)
        lm_pos = positions.get(lm_head_name, len(mods))
        backbone = self._block_backbone_prefix_(block_prefixes)
        best = None
        for idx, (n, m) in enumerate(mods):
            if idx <= last_block_pos or idx >= lm_pos:
                continue
            if backbone is not None and not (n == backbone or n.startswith(backbone + ".")):
                continue
            if list(m.children()):
                continue
            params = dict(m.named_parameters(recurse=False))
            weight = params.get("weight")
            if weight is None or weight.dim() != 1:
                continue
            if any(k != "weight" and (k != "bias" or params[k].dim() != 1) for k in params):
                continue
            best = (n, m)  # last match: the one adjacent to lm_head
        return best if best is not None else (None, None)

    @staticmethod
    def _block_backbone_prefix_(block_prefixes):
        """Module prefix of the container that parents all blocks (e.g.
        ``model.language_model`` for ``model.language_model.layers.*``), or
        ``None`` when it cannot be derived."""
        if not block_prefixes:
            return None
        splits = [b.split(".") for b in block_prefixes]
        depth = 0
        while True:
            if any(len(s) <= depth + 1 for s in splits) or any(s[depth] != splits[0][depth] for s in splits):
                break
            depth += 1
        # ``depth`` counts the shared components INCLUDING the block-list
        # container (``layers``); the backbone is its parent
        return ".".join(splits[0][: max(depth - 1, 0)]) if depth >= 2 else None

    def _resolve_lm_head_name_(self, remain_layer_names):
        """lm_head's module name from the quantization plan, or ``None``.

        Plan-derived names are the anchor (the data-driven path matches
        ``"lm_head" in layer_name`` the same way). Module order is
        deliberately NOT consulted: checkpoint-only placeholder trees attach
        after lm_head and steal the last-leaf position."""
        candidates = [n for n in remain_layer_names if n == "lm_head" or n.rsplit(".", 1)[-1] == "lm_head"]
        if not candidates:
            candidates = [n for n in remain_layer_names if "lm_head" in n]
        if not candidates:
            logger.debug("[stream] no lm_head in the outside-block plan; lm_head is not quantized this run")
            return None
        if len(candidates) > 1:
            logger.warning("[stream] multiple lm_head candidates in the plan %s; tuning %s", candidates, candidates[0])
        return candidates[0]

    def _lm_head_tune_inputs_(
        self, calib_state, remain_layer_names, streamer=None, lm_head_name=None, pre_captured=None
    ):
        """Per-sample ``(fp_rows, q_rows, token_ids)`` for tuning lm_head from
        the calibration chain's tail, or ``None`` to keep the closed-form search.

        At iters>0 the same SignRound loop that tunes blocks - and that the
        data-driven path already uses for outside-block layers - tunes lm_head
        here. The chain tail is the RAW last-block output; lm_head consumes
        POST-final-norm states, so the final norm is applied to the rows (its
        weights stream in when still meta). ``lm_head_name`` may be passed by
        the caller (already resolved from the plan); left unset it is resolved
        here the same way."""
        if self._max_tune_iters() <= 0:
            return None
        if lm_head_name is None:
            lm_head_name = self._resolve_lm_head_name_(remain_layer_names)
        if lm_head_name is None:
            return None
        if pre_captured is not None:
            # the checkpoint-only tree tune consumes/mutates the shared chain
            # tail (a 27B run read it back as placeholder rows); the caller
            # captured an immutable copy BEFORE that tune
            fp_inputs, token_ids = pre_captured
        else:
            fp_inputs = (calib_state or {}).get("fp_inputs")
            token_ids = (calib_state or {}).get("token_ids")
        # len() not truthiness: a raw tensor chain tail must not hit ambiguous
        # bool evaluation on the way to the format rejection below
        if fp_inputs is None or token_ids is None or len(fp_inputs) == 0 or len(token_ids) == 0:
            logger.warning(
                "[stream] lm_head falls back to the closed-form search: the calibration chain kept no "
                "final hidden states or token ids to tune from"
            )
            return None
        fp_rows = self._chain_hidden_rows(fp_inputs)
        if (
            not isinstance(fp_rows, (list, tuple))
            or len(fp_rows) == 0
            or len(token_ids) != len(fp_rows)
            or not all(isinstance(r, torch.Tensor) for r in fp_rows)
        ):
            _shape_desc = (
                f"{type(fp_rows).__name__}"
                if not isinstance(fp_rows, (list, tuple))
                else f"list[{len(fp_rows)}] of {type(fp_rows[0]).__name__ if fp_rows else 'empty'}"
                + (f" dim{tuple(fp_rows[0].shape)}" if fp_rows and isinstance(fp_rows[0], torch.Tensor) else "")
            )
            logger.warning(
                "[stream] lm_head falls back to the closed-form search: unexpected chain-tail row format "
                f"(fp_rows={_shape_desc}, token_ids={len(token_ids)}, "
                f"fp_inputs={type(fp_inputs).__name__}"
                + (f" keys={sorted(fp_inputs.keys())[:6]}" if isinstance(fp_inputs, dict) else "")
                + ")"
            )
            return None
        q_inputs = (calib_state or {}).get("q_inputs")
        q_rows = None
        if q_inputs is not None:
            q_rows = self._chain_hidden_rows(q_inputs)
            if not isinstance(q_rows, (list, tuple)) or len(q_rows) != len(fp_rows):
                logger.warning(
                    "[stream] lm_head tuning uses FP chain inputs (enable_quanted_input cannot be honored): "
                    "quantized chain rows are missing or malformed"
                )
                q_rows = None
        # lm_head consumes POST-final-norm hidden states; the chain tail holds
        # the raw last-block output. Apply the final norm before tuning - a
        # tune on the wrong scale fits clip params to the wrong distribution.
        norm_name, norm_mod = self._final_norm_module_(lm_head_name)
        if norm_mod is None:
            logger.warning(
                "[stream] lm_head falls back to the closed-form search: cannot locate the final norm that "
                "feeds it (needed to turn chain rows into lm_head inputs)"
            )
            return None
        # a wrongly picked leaf (learned gate, per-head norm) is detectable: the
        # real final norm scales the hidden dim lm_head consumes
        in_features = getattr(get_module(self.model, lm_head_name), "in_features", None)
        if in_features is not None and norm_mod.weight.numel() != in_features:
            logger.warning(
                "[stream] lm_head falls back to the closed-form search: candidate final norm %s does not "
                "match lm_head's input width (%d vs %d)",
                norm_name,
                norm_mod.weight.numel(),
                in_features,
            )
            return None
        if any(p.is_meta for p in norm_mod.parameters()):
            if streamer is None:
                logger.warning(
                    "[stream] lm_head falls back to the closed-form search: the final norm is still meta and "
                    "no checkpoint streamer is available to load it"
                )
                return None
            if not streamer.names_under(norm_name):
                logger.warning(
                    "[stream] lm_head falls back to the closed-form search: the checkpoint stores no "
                    "tensors under the final norm path %s",
                    norm_name,
                )
                return None
            streamer.load_module_(norm_mod, norm_name, device=str(fp_rows[0].device))
        with torch.no_grad():
            dev, dt = fp_rows[0].device, norm_mod.weight.dtype
            fp_rows = [norm_mod(r.to(dev).to(dt)).to(dev) for r in fp_rows]
            if q_rows is not None:
                q_rows = [norm_mod(r.to(dev).to(dt)).to(dev) for r in q_rows]
        # fresh lists: the tune loop reassigns entries (dtype/device casts) and
        # must not mutate the chain state it shares tensors with
        return list(fp_rows), (list(q_rows) if q_rows is not None else None), token_ids

    def _quantize_zero_shot(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Zero-shot (RTN) quantization path — no calibration data needed.

        This replaces the standalone ``ZeroShotCompressor.quantize()`` method.
        Block-wise RTN quantization without any input data.
        """
        from auto_round.algorithms.composer import BlockContext

        formats = self.formats if isinstance(self.formats, list) else []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self.alg_composer.compress_embedding_layer()  # leave to gguf itself to handle

        # Release memory
        clear_memory()

        # In RTN mode (iters == 0), force blockwise quantization to avoid
        # full-model materialization and linear CPU RAM growth.
        logger.info("Zero-shot mode (no calibration data needed): using blockwise quantization.")

        tied_weights_keys = getattr(self.model, "_tied_weights_keys", [])
        if tied_weights_keys is None:
            tied_weights_keys = []
        if isinstance(tied_weights_keys, dict):
            tied_weights_values = list(tied_weights_keys.values())
        else:
            tied_weights_values = list(tied_weights_keys)
        tied_weights_layers = [".".join(val.split(".")[:-1]) for val in tied_weights_values]  # rm weight/bias
        # In fact, we should detect whether it is is_separate_lm_head, to simplify, we don't do it
        if getattr(self, "formats", None) and self.formats[0].is_gguf():
            lm_head_name = get_lm_head_name(self.model)
            if lm_head_name is not None:
                tied_weights_layers.append(lm_head_name)

        # -- stream_quantization: per-block tensor streaming from the checkpoint --
        streamer = getattr(self.model_context, "checkpoint_streamer", None)
        if streamer is not None and not self.compress_context.is_immediate_saving:
            raise ValueError(
                "stream_quantization=True requires immediate saving "
                "(enable low_cpu_mem_usage=True and keep inplace packing; int data types only)."
            )
        if streamer is not None and isinstance(self.formats, list) and len(self.formats) > 1:
            if any(fmt.is_gguf() for fmt in self.formats):
                # the GGUF blob path is single-artifact by design; a mixed
                # format list would emit stray compressed-tensors shards into
                # the GGUF output dir (and the blob branches key on formats[0])
                raise ValueError(
                    "stream_quantization with a gguf format must be single-format "
                    f"(got {len(self.formats)} formats); run the GGUF export separately."
                )

        all_blocks = self.quant_block_list or get_block_names(self.model)
        flat_block_names = [name for group in all_blocks for name in group]

        # Pristine structure snapshots for checkpoint-only groups: by the
        # time the materializer runs (after the block loop) every block has
        # been packed to QuantLinear and lost its plain parameter names - the
        # sibling structure snapshot must be taken while the meta skeleton is
        # still untouched. Snapshots are structure-only (meta tensors, no
        # data) and only captured when the checkpoint actually carries
        # groups no module claims.
        block_snapshots = None
        if streamer is not None and flat_block_names:
            ckpt_only = self._checkpoint_only_groups_(streamer)
            if ckpt_only:
                block_snapshots = {
                    name: copy.deepcopy(get_module(self.model, name))
                    for name in flat_block_names
                    if get_module(self.model, name) is not None
                }

        # Optional streaming calibration: the rows are embedded once and each
        # block's compress_block reference output (replay through the
        # transformed block, activation hooks firing) feeds the next block -
        # the data-driven chain semantics without a full model load.
        calib_state = None
        if streamer is not None and getattr(self, "stream_calibration", False):
            from auto_round.utils.streaming_calibration import prepare_streaming_calibration

            fp_inputs, input_others, summary = prepare_streaming_calibration(
                self.model,
                streamer,
                dataset=self.dataset,
                device=str(self.device),
                seqlen=self.calibration_context.seqlen,
                tokenizer=self.tokenizer,
                first_block=get_module(self.model, flat_block_names[0]) if flat_block_names else None,
                nsamples=int(getattr(self.calibration_context, "nsamples", 128) or 128),
            )
            calib_state = {
                "fp_inputs": fp_inputs,
                "input_others": input_others,
                "token_ids": summary.get("token_ids"),
                "keymask_2d": summary.get("keymask_2d"),
            }
            # drop the local aliases: the block loop replaces
            # calib_state["fp_inputs"] with each new generation, but a live
            # local reference would pin the FIRST generation (a generation-
            # sized region) for the whole run
            fp_inputs = None
            input_others = None

        # -- AR_RESUME_DIR: resume support (byte-compatible manifests with the
        # serial and streaming paths -- either execution mode continues where the
        # stopped). Requires immediate saving: "done" in the manifest is the
        # contract that the block's tensors are durably in a shard file. --
        resume_states = None
        if envs.AR_RESUME_DIR and self.compress_context.is_immediate_saving:
            resume_states = self._build_resume_states(all_blocks)
            # quantize_and_save clears these after a successful export; without
            # the assignment a rerun of the same command would skip every block
            # and export nothing
            self._resume_states = resume_states
            if any(rs is not None and rs.resume_index > 0 for rs in resume_states):
                self._stream_resume_jump_chain(calib_state, resume_states)
                if self.shard_writer is not None:
                    self.shard_writer.adopt_existing_shards()  # never overwrite the crashed run's shards
                self._adopt_blob_store_()
                # the replay leaves granular-read debris that would raise every
                # later block's peak (VmHWM keeps the high-water mark)
                self._trim_host_heap()
                self._release_cuda_cache("resume rebuild")

        # GGUF blob runs create their conversion instance up front: MTP
        # detection needs the checkpoint-side mtp.* names before any
        # checkpoint-only predictor tree materializes at group tail
        self._ensure_gguf_blob_conversion_(streamer)

        if streamer is not None:
            # startup reads (embeddings / chain init) touch shards the block
            # loop never revisits - on fresh runs too, not only resume; close
            # them before the prefetch pipeline starts so their mappings stop
            # counting in RSS
            streamer.release_startup_handles_()

        # Prefetch pipeline: a background reader stages upcoming blocks ahead
        # of the quantize loop. With a staging device the blocks land directly
        # on it and are quantized in place there (the quant device joins the
        # rotation under 'auto' when its free VRAM fits); otherwise they wait
        # in host RAM. Lookahead is ONE block, always: quantize time per block
        # dwarfs its load time, so deeper staging would only hold extra
        # block-sized VRAM.
        _prefetch_mode = str(getattr(self, "stream_prefetch", "off") or "off").strip().lower()
        prefetch_depth = 0 if _prefetch_mode in STREAM_PREFETCH_OFF else 1
        _mapped_state = (
            self._make_mapped_resolver(flat_block_names)
            if (streamer is not None and self._stream_mapped_enabled())
            else None
        )
        stage_devices = self._resolve_stream_stage_devices() if (streamer is not None and prefetch_depth > 0) else None
        _stage_device_of = None
        if _mapped_state is not None:
            _fallback = str(self.device)

            def _stage_device_of(idx, name, prefix, _st=_mapped_state, _fb=_fallback):
                placement = _st["resolve"](prefix)
                if not placement:
                    return None
                rel = name[len(prefix) + 1 :] if prefix and name.startswith(prefix + ".") else name
                from auto_round.utils.stream_placement import first_placement_device, placement_device_of

                # tensors that match no entry default INSIDE the block's map
                # (front of the block), never the global primary
                _dfb = str(first_placement_device(placement, _fb))
                return placement_device_of(placement, rel, _dfb)

        prefetch_names = flat_block_names
        if resume_states is not None:
            _pending_offset = self._stream_resume_pending_offset(all_blocks, resume_states)
            if _pending_offset is None:
                prefetch_names = []
            elif _pending_offset > 0:
                logger.info(
                    "[stream] resume: %d block(s) already done; continuing from %s",
                    _pending_offset,
                    flat_block_names[_pending_offset],
                )
                prefetch_names = flat_block_names[_pending_offset:]
        if streamer is not None and prefetch_depth > 0 and prefetch_names:
            tuning_iters, moe_routing_bytes = self._tuning_headroom_profile()
            streamer.start_prefetch(
                prefetch_names,
                depth=prefetch_depth,
                stage_devices=stage_devices,
                tuning_iters=tuning_iters,
                moe_routing_bytes=moe_routing_bytes,
                stage_device_of=_stage_device_of,
            )

        # -- Background pack pipeline (AR_STREAM_BG_PACK=auto|1|0)
        # The finished block's immediate-pack + shard-write tail runs in a
        # background thread on its (now idle) ping-pong home while the loop
        # advances to the next block's tune on the other group. Supported only
        # with >=2 staging groups (the finished block must stay GPU-resident
        # on a group nobody else needs) and immediate packing; exactly one
        # pipeline thread runs at a time (the loop joins the previous one
        # before spawning the next, ordering shard writes and serializing the
        # lock-free ShardWriter behind a single writer at any moment).
        _bg_pack_eligible = self._resolve_bg_pack_mode(
            envs.AR_STREAM_BG_PACK,
            len(stage_devices) if stage_devices else 0,
            self.compress_context.is_immediate_packing,
        )
        if _bg_pack_eligible and _mapped_state is not None:
            logger.info("[stream-mapped] background packing is not yet wired for mapped placement; packing serially")
            _bg_pack_eligible = False
        if _bg_pack_eligible and self._gguf_blob_mode():
            # GGUF blob packing runs the conversion instance's prepare_tensors
            # with shared mutable state (current_packing_block); it is not
            # safe to overlap with the next block's tune on a worker thread
            logger.info("[stream] gguf blob mode: packing runs serially in the main loop")
            _bg_pack_eligible = False
        _bg_pack = None
        _bg_finish = None
        _bg_finish_eligible = False
        if self._gguf_blob_mode():
            # GGUF blob packing runs the conversion instance's prepare_tensors
            # with shared mutable state (current_packing_block); it is not
            # safe to overlap with the next block's tune on a worker thread -
            # in any configuration. The finish tail (meta-park + resume
            # snapshot) has no such state and does no GPU math, so it still
            # overlaps on a worker regardless of staging-device count; only
            # an explicit AR_STREAM_BG_PACK=off serializes everything.
            _bg_finish_eligible = self._resolve_bg_finish_mode(True, envs.AR_STREAM_BG_PACK)
            logger.info(
                "[stream] gguf blob mode: packing runs serially in the main loop%s",
                "; block finish (resume snapshot) overlaps on a worker" if _bg_finish_eligible else "",
            )
            _bg_pack_eligible = False

        # Model-level algorithm lifecycle before the block loop, mirroring the
        # data-driven path: SignRoundV2Quantizer.prepare_run binds the optimized
        # wrapper (imatrix-weighted init); skipping it left V2 tuning silently
        # on the plain min/max wrapper. Inert for iters=0 RTN runs (no member
        # overrides prepare_run there).
        self.alg_composer.prepare_run()

        total_block_cnt = sum(len(block) for block in all_blocks)
        pbar = tqdm(range(total_block_cnt))
        stream_block_idx = 0
        blocks_before = 0
        _peak_watch = PeakWatcher() if envs.AR_MEM_COUNTERS else None
        if _peak_watch is not None:
            _peak_watch.start()
        for g_idx, block_names in enumerate(all_blocks):
            rs = resume_states[g_idx] if resume_states is not None and g_idx < len(resume_states) else None
            for k_idx, block_name in enumerate(block_names):
                pbar.set_description(f"Quantizing {block_name}")
                if rs is not None and k_idx < rs.resume_index:
                    # durably done in a previous run: its tensors are already
                    # in an adopted output shard -- just skip (no staging slot
                    # is consumed, the block is never materialized)
                    pbar.set_description(f"Skipping {block_name} (done)")
                    pbar.update(1)
                    continue
                block = get_module(self.model, block_name)

                # ── Infrastructure: reload from disk when streaming ───────
                # Fused-MoE checkpoints (and explicit `AR_DISK_STREAM_MODEL=1`)
                # build an all-meta skeleton to reduce RAM: each decoder block
                # starts on the meta device and its real weights must be read
                # back from the checkpoint before quantization. The data-driven
                # path does this same reload; without it here the zero-shot
                # (RTN) path leaves the block on meta and `layer.to(device)`
                # crashes with "Cannot copy out of meta tensor".
                disk_streaming = getattr(self.model_context, "_disk_stream_index", None) is not None
                if self.compress_context.low_cpu_mem_usage or envs.AR_DISK_STREAM_MODEL or disk_streaming:
                    self._offloader.reload(self.model, block_name)

                # ── Infrastructure: materialize ───────────────────────────
                _t_load = _time.perf_counter()
                _load_sub = {} if envs.AR_PERF_COUNTERS else None
                _t_seg = _time.perf_counter()
                if _peak_watch is not None:
                    _peak_watch.set_phase("load")
                _placement = _mapped_state["resolve"](block_name) if _mapped_state is not None else None
                if streamer is not None:
                    if _placement is not None:
                        # mapped placement: fetch every tensor straight to its
                        # module's device; the primary stays the fallback home
                        from auto_round.utils.stream_placement import first_placement_device

                        # unmatched tensors load INSIDE the block's device set,
                        # never onto the global primary (it may host prefetch
                        # for other blocks)
                        load_device = str(first_placement_device(_placement, torch.device(str(self.device))))

                        def _dev_of(name, _p=_placement, _pre=block_name):
                            rel = name[len(_pre) + 1 :] if _pre and name.startswith(_pre + ".") else name
                            from auto_round.utils.stream_placement import placement_device_of

                            return placement_device_of(_p, rel, load_device)

                        streamer.load_module_(block, block_name, device=load_device, device_of=_dev_of)
                    else:
                        if stage_devices:
                            # the reader records where each block was ACTUALLY staged
                            # (first-fit under asymmetric pressure may diverge from
                            # the rotation); quantize in place on that home
                            load_device = str(
                                streamer._prefetch_stage_dev.get(
                                    block_name, stage_devices[stream_block_idx % len(stage_devices)]
                                )
                            )
                        else:
                            load_device = str(self.device)
                        streamer.load_module_(block, block_name, device=load_device)
                    _t_seg = _mark_load_seg(_load_sub, "io", _t_seg)
                    streamer.close_shards_not_serving_(flat_block_names[flat_block_names.index(block_name) + 1 :])
                    streamer.close_main_pool_()
                    _t_seg = _mark_load_seg(_load_sub, "close", _t_seg)
                materialize_model_(block)
                _t_seg = _mark_load_seg(_load_sub, "mat", _t_seg)
                if streamer is not None:
                    self._assert_block_materialized(block, block_name)
                    strip_stale_device_hooks_(block)
                    if _placement is not None:
                        # keep the block distributed: mapped leaves stay on
                        # their template devices, everything else on the
                        # primary. No single-home pin - the leaves already
                        # carry tuning_device from the placement resolver.
                        from auto_round.compressors.utils import rehome_block_mapped_

                        _n_moved = rehome_block_mapped_(block, _placement, load_device)
                        block._stream_mapped = _placement
                        self._pin_stream_mapped_(block, _placement)
                        # derivation: probe the reference forward (first full
                        # pass) and restage onto the derived placement at the
                        # fp->wrapper boundary, before any tuning state exists
                        _has_fwd = calib_state is not None and calib_state.get("fp_inputs") is not None
                        if _mapped_state["maybe_probe"](block, block_name, _has_fwd) is not None:
                            block._stream_restage_after_fp_ = _mapped_state["make_restage"](block, block_name)
                    else:
                        _n_moved = rehome_block_(block, load_device)
                        block._stream_home_device = torch.device(load_device)
                        # pin leaf tuning_device to the home: WrapperLinear prefers it
                        # (wrapper.py self.device = orig_layer.tuning_device or device),
                        # and quantize_block's local device otherwise defaults to the
                        # global primary - the wrapper would drag the wrapped layers
                        # back to cuda:0 while unwrapped siblings (conv1d) stay home.
                        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

                        SignRoundQuantizer._pin_stream_home(block, block._stream_home_device)
                    if stream_block_idx == 0:
                        logger.debug(
                            "[stream] device hygiene for %s: stale accelerate hooks stripped; %d tensor(s) "
                            "re-homed to %s (shared/setup modules start on the primary)",
                            block_name,
                            _n_moved,
                            load_device,
                        )
                    _t_seg = _mark_load_seg(_load_sub, "rehome", _t_seg)
                    if envs.AR_MEM_COUNTERS and logger.isEnabledFor(logging.DEBUG):
                        # diagnostics: accounted separately so the io figure
                        # stays honest about the actual load cost
                        self._log_device_inventory(calib_state, f"block {stream_block_idx}")
                        _t_seg = _mark_load_seg(_load_sub, "inv", _t_seg)
                    if _placement is not None:
                        for _dev in {str(d) for d in _placement.values()} | {load_device}:
                            self._release_cached_segments_if_fragmented(_dev)
                    else:
                        self._release_cached_segments_if_fragmented(load_device)
                _t_load = _time.perf_counter() - _t_load

                # ── Pure algorithm ────────────────────────────────────────
                # global block index/count so consumers keyed on "last block
                # of the run" (SignRound's LFQ loss gate, layerwise rotation
                # indices) behave exactly as on the data-driven path. This is
                # the position in the FULL block list -- resumed-done blocks
                # that were skipped above still count (unlike the staging
                # rotation index).
                ctx = BlockContext(
                    model=self.model,
                    block_names=[block_name],
                    block_name=block_name,
                    block_index=blocks_before + k_idx,
                    block_cnt=total_block_cnt,
                )
                # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
                if is_nv_fp(self.act_data_type) or not self.act_dynamic:
                    set_amax_for_all_moe_layers(block, attr_name="act_max")

                update_block_global_scale_if_needed(block, self.data_type, self.group_size)
                if streamer is not None and calib_state is not None and calib_state.get("keymask_2d"):
                    # resolve this block's attention-mask convention by probe
                    # (models can mix forms per block: GDN linear-attention
                    # blocks take the 2D padding mask, full-attention blocks
                    # the 4D form); the previous block's form is tried first
                    from auto_round.utils.streaming_calibration import (
                        materialize_mask_form,
                        resolve_chain_mask_form,
                    )

                    form = resolve_chain_mask_form(
                        block,
                        _first_chain_row(calib_state["fp_inputs"]),
                        calib_state["keymask_2d"][0],
                        calib_state["input_others"],
                        preferred=calib_state.get("_mask_form"),
                        amp=self.amp,
                        amp_dtype=self.amp_dtype,
                    )
                    if form != calib_state.get("_mask_form"):
                        # block types come in runs; the transition is the signal
                        logger.debug("[stream_calibration] attention-mask form change at %s: %s", block_name, form)
                    calib_state["input_others"]["attention_mask"] = [
                        materialize_mask_form(m, form) for m in calib_state["keymask_2d"]
                    ]
                    calib_state["_mask_form"] = form
                if calib_state is not None and calib_state["fp_inputs"] is not None:
                    _t_tune = _time.perf_counter()
                    if _peak_watch is not None:
                        _peak_watch.set_phase("tune")
                    new_q_input, reference_output = self.alg_composer.compress_block(
                        block,
                        calib_state["fp_inputs"],
                        calib_state["input_others"],
                        block_ctx=ctx,
                        q_inputs=calib_state.get("q_inputs"),
                        input_ids=calib_state.get("token_ids"),
                    )
                    if _mapped_state is not None:
                        # mapped contract: rows live on host RAM (the tune
                        # parks them there; keep the advanced chain there too
                        # so the map's GPUs hold only block state)
                        reference_output = self._park_rows_cpu_(reference_output)
                        new_q_input = self._park_rows_cpu_(new_q_input)
                    calib_state["fp_inputs"] = reference_output
                    if self.alg_composer.need_quanted_input():
                        # qon: the next block tunes against this block's
                        # quantized outputs, mirroring the data-driven loop.
                        calib_state["q_inputs"] = new_q_input
                    else:
                        calib_state.pop("q_inputs", None)
                    block._stream_tune_seconds = _time.perf_counter() - _t_tune
                else:
                    self.alg_composer.compress_block(block, fp_inputs=None, input_others={}, block_ctx=ctx)
                if _bg_pack is not None:
                    # the previous block's pipeline must finish before this
                    # block's spawn: shard-write order follows quantization
                    # order and the ShardWriter has no internal locking
                    self._join_bg_pack(_bg_pack)
                    _bg_pack = None
                    self._bg_pack_thread = None
                if _bg_finish is not None:
                    # same ordering contract for the blob finish worker: its
                    # mark_block_done must land before the next one spawns
                    # (ResumeState asserts in-order completion)
                    self._join_bg_pack(_bg_finish)
                    _bg_finish = None
                    self._bg_pack_thread = None
                if _bg_pack_eligible:
                    # pack + write of the FINISHED block move to a background
                    # pipeline thread: they run on this block's (now idle)
                    # ping-pong home while the loop advances to the next
                    # block's tune on the other group. Snapshots of the next
                    # block's inputs are captured NOW -- the main loop mutates
                    # calib_state as soon as it advances.
                    _q_snap = self._snapshot_chain_rows_(
                        calib_state.get("q_inputs") if calib_state is not None else None
                    )
                    _fp_snap = self._snapshot_chain_rows_(calib_state["fp_inputs"] if calib_state is not None else None)
                    _is_last = g_idx == len(all_blocks) - 1 and k_idx == len(block_names) - 1
                    _bg_pack = self._start_bg_pack_block(
                        block,
                        block_name,
                        load_device,
                        self.layer_config,
                        self.nblocks,
                        tied_weights_layers,
                        rs,
                        _q_snap,
                        _fp_snap,
                        _is_last,
                    )
                elif self.compress_context.is_immediate_packing:
                    _t_pack = _time.perf_counter()
                    from auto_round.compressors.utils import immediate_pack_block as _immediate_pack_block

                    _immediate_pack_block(
                        block,
                        block_name,
                        self.layer_config,
                        nblocks=self.nblocks,
                        device=load_device if streamer is not None else None,
                    )
                    _t_pack = _time.perf_counter() - _t_pack
                else:
                    _t_pack = 0.0

                # ── Infrastructure: shard write / device cleanup ──────────
                if _bg_finish_eligible and self.compress_context.is_immediate_saving:
                    # blob mode: the pack already ran serially above; the
                    # finish tail (meta-park + resume snapshot) moves to a
                    # worker and overlaps with the next block's tune. The
                    # snap refs are captured NOW, before the loop advances.
                    _is_last = g_idx == len(all_blocks) - 1 and k_idx == len(block_names) - 1
                    _bg_finish = self._start_bg_finish_block(
                        block,
                        block_name,
                        tied_weights_layers,
                        rs,
                        self._snapshot_chain_rows_(calib_state.get("q_inputs") if calib_state is not None else None),
                        self._snapshot_chain_rows_(
                            (None if _is_last else calib_state["fp_inputs"]) if calib_state is not None else None
                        ),
                        _is_last,
                    )
                    _t_write = 0.0
                    _t_snap = 0.0
                    if envs.AR_PERF_COUNTERS:
                        logger.info(
                            "[perf] block %s: load %.1fs%s tune %.1fs pack %.1fs write %.1fs snap %.1fs",
                            block_name,
                            _t_load,
                            _format_load_breakdown(_load_sub),
                            getattr(block, "_stream_tune_seconds", 0.0),
                            _t_pack,
                            _t_write,
                            _t_snap,
                        )
                elif not _bg_pack_eligible and self.compress_context.is_immediate_saving:
                    _t_write = _time.perf_counter()
                    is_model_last = g_idx == len(all_blocks) - 1 and k_idx == len(block_names) - 1
                    self._write_finished_block_(
                        block,
                        block_name,
                        tied_weights_layers,
                        rs,
                        self._snapshot_chain_rows_(calib_state.get("q_inputs") if calib_state is not None else None),
                        self._snapshot_chain_rows_(
                            (None if is_model_last else calib_state["fp_inputs"]) if calib_state is not None else None
                        ),
                        is_model_last,
                    )
                    _t_write = _time.perf_counter() - _t_write
                    _t_snap = 0.0
                    if envs.AR_PERF_COUNTERS:
                        logger.info(
                            "[perf] block %s: load %.1fs%s tune %.1fs pack %.1fs write %.1fs snap %.1fs",
                            block_name,
                            _t_load,
                            _format_load_breakdown(_load_sub),
                            getattr(block, "_stream_tune_seconds", 0.0),
                            _t_pack,
                            _t_write,
                            _t_snap,
                        )
                else:
                    if self._main_loop_may_move_block_off_gpu(self.compress_context.is_immediate_saving):
                        mv_module_from_gpu(block)
                        if self.compress_context.low_cpu_mem_usage and streamer is None:
                            self._offloader(self.model, block_name)

                if envs.AR_MEM_COUNTERS and logger.isEnabledFor(logging.DEBUG):
                    # post-tune sample: catches the during-tuning transient
                    # peak via VmHWM that the pre-tune snapshot misses
                    self._log_device_inventory(calib_state, f"block {stream_block_idx} post")
                if _peak_watch is not None:
                    _peak_watch.set_phase("write")
                    _peak_watch.log(f"block {stream_block_idx}")
                    _peak_watch.reset_run_max()
                if _bg_pack is not None:
                    # the worker's pack/write kernels are in flight: the
                    # process-wide gc/empty_cache of clear_memory() from this
                    # thread is the exact hazard the worker's own NOTE forbids
                    # (mirrored) - it corrupted in-flight accesses on the
                    # server. Host-side hygiene only; the join at the next
                    # block boundary runs the full clear single-threaded.
                    self._trim_host_heap()
                else:
                    clear_memory()
                    self._trim_host_heap()
                # upstream-pattern per-block summary: stays unconditional,
                # matching the data-driven loop. AR_MEM_COUNTERS gates only
                # the branch-added [stream-mem] drill-down lines, never
                # upstream calls.
                memory_monitor.log_summary()
                stream_block_idx += 1  # consumed a staging slot: rotate the round-robin home
                pbar.update(1)
            # group tail: advance the global-index base by THIS group's block
            # count once (an update inside the k_idx loop would scale it by
            # the block count of every block)
            blocks_before += len(block_names)
        if _peak_watch is not None:
            # stop once after ALL groups: a per-group stop would kill the
            # sampler after the first group and later groups would lose peak
            # attribution entirely
            _peak_watch.stop()

        # Pipeline lifecycle: model-level teardown (also finalizes rotation)
        if _bg_pack is not None:
            # final save/index write below reads the ShardWriter state; the
            # last block's pack pipeline must be complete first
            self._join_bg_pack(_bg_pack)
            _bg_pack = None
            self._bg_pack_thread = None
        if _bg_finish is not None:
            # the last block's finish worker owns the resume frontier; it
            # must land before checkpoint-only groups extend the chain tail
            self._join_bg_pack(_bg_finish)
            _bg_finish = None
            self._bg_pack_thread = None
        # Checkpoint-only blocks with a layer_config pin (e.g. an MTP layer
        # the modeling code never instantiates): materialize BEFORE the run
        # finalizes - tuning runs join these groups to the chain's tail with
        # the run's own tuning config, and the composer must stay live for
        # that. Pinned groups without a covering sibling quantize through the
        # closed-form search in both regimes instead.
        materialized_tensors = set()
        tree_groups = []
        mtp_tuned = set()
        if streamer is not None:
            materialized_tensors, tree_groups = self._materialize_pinned_checkpoint_only_blocks_(
                streamer, all_blocks, block_snapshots
            )
            block_snapshots = None
            _lm_pre_capture = None
            if tree_groups and calib_state is not None and self._max_tune_iters() > 0:
                _fp = calib_state.get("fp_inputs")
                _tok = calib_state.get("token_ids")
                if _fp is not None and _tok is not None:
                    _lm_pre_capture = (self._snapshot_chain_rows_(_fp), _tok)
            if tree_groups:
                _tune_iters = self._max_tune_iters()
                if _tune_iters > 0:
                    mtp_tuned = self._tune_checkpoint_only_groups_(streamer, tree_groups, calib_state, total_block_cnt)
        if streamer is not None and prefetch_depth > 0:
            streamer.stop_prefetch()
        if streamer is not None:
            streamer.close()
        self.alg_composer.finalize_run()

        remain_layer_names = []
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in self.model.named_modules():
            if not check_to_quantized(m):
                continue
            # Skip if this layer is part of any block (by prefix match)
            if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                continue
            # Tuned predictor groups are already packed + written by the tune
            # step; re-running the closed-form search on packed modules would
            # corrupt them
            if any(n == t or n.startswith(f"{t}.") for t in mtp_tuned):
                continue
            remain_layer_names.append(n)
        outside_qdev = self._outside_block_quant_device()
        # resolve from the plan: placeholder trees attach after lm_head and
        # break module-order detection (last-leaf heuristics)
        lm_head_name = self._resolve_lm_head_name_(remain_layer_names)
        # iters>0: tune lm_head with the chain's final hidden states, exactly
        # like the data-driven path tunes outside-block layers (the
        # per-sample tune loop lives in quantize_layer_outside_block);
        # iters=0 keeps the closed-form search on the same device
        lm_tune = (
            self._lm_head_tune_inputs_(
                calib_state,
                remain_layer_names,
                streamer=streamer,
                lm_head_name=lm_head_name,
                pre_captured=_lm_pre_capture if streamer is not None else None,
            )
            if streamer is not None
            else None
        )
        for name in remain_layer_names:
            module = get_module(self.model, name)
            logger.info(f"Quantizing remaining layer {name} on {outside_qdev}.")
            from auto_round.utils.device import log_cuda_memory_census

            # phase boundary: the block loop just freed its large tuning
            # buffers; returning them to the driver keeps the allocator pool
            # compact before the (potentially huge) outside-block wrappers are
            # built, instead of reserving fragmented segments nobody can use
            # (backend-agnostic; cpu entries are skipped by clear_memory)
            clear_memory(device_list=[str(outside_qdev)])
            log_cuda_memory_census(f"outside-block loop entry {name}", outside_qdev)
            if streamer is not None:
                # load the layer itself; streaming its parent prefix would
                # materialize the parent's whole subtree (every block weight).
                # Placeholders whose weights arrived real (unfused expert
                # slices) have no checkpoint tensors under their path.
                if streamer.names_under(name) or any(p.is_meta for p in module.parameters()):
                    streamer.load_module_(module, name, device=str(outside_qdev))
                elif outside_qdev.type != "cpu":
                    module.to(outside_qdev)
            tune_kwargs = {}
            if lm_tune is not None and name == lm_head_name:
                _fp_rows, _q_rows, _token_ids = lm_tune
                tune_kwargs = {"fp_inputs": _fp_rows, "q_inputs": _q_rows, "input_ids": _token_ids}
                logger.info("[stream] tuning lm_head with the run's tuning config on %s", outside_qdev)
            self.alg_composer.compress_layer_outside_block(get_module(self.model, name), **tune_kwargs)
            if streamer is not None and self.compress_context.is_immediate_saving:
                # pack + write now: the export pack loop is skipped under the
                # streaming meta skeleton (mixed meta/real), so shards are the
                # only place the layer's packed state can live - the export
                # restore re-derives its scheme from there
                from auto_round.compressors.utils import immediate_pack as _immediate_pack

                _immediate_pack(name, self.layer_config, device=str(outside_qdev))
                if not self._gguf_blob_mode():
                    self.shard_writer.write(name=name)
                # gguf blob mode keeps the module live here on purpose: the
                # save-time prepare_tensors pass packs outside-block tensors
                # (embeddings, lm_head, norms) from the in-memory qdq weights
            # Outside-block layers (embed_tokens/lm_head/etc.) are typically few so just
            # log a summary after each one.
            clear_memory()
            memory_monitor.log_summary()

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.compress_context.low_cpu_mem_usage and streamer is None:
            self._offloader.reload(self.model)
        if streamer is not None and self.compress_context.is_immediate_saving and self._gguf_blob_mode():
            # GGUF blob mode: still-meta root tensors (norms, vision tower on
            # MLLM, ...) get hydrated from the checkpoint at save time by
            # _hydrate_meta_from_checkpoint, and the save-time
            # prepare_tensors pass packs them straight into blob shards - the
            # compressed-tensors root pass-through below would only duplicate
            # them into shards this run never reads.
            logger.info("[stream] gguf blob mode: root tensors deferred to the save-time gguf pass")
        elif streamer is not None and self.compress_context.is_immediate_saving:
            # Root pass-through tensors (embeddings, final norm, lm_head, ...) are
            # still meta; stream them in so ShardWriter.finalize() sees real data
            # (finalize silently skips meta tensors).
            from auto_round.compressors.utils import check_to_quantized as _ctq

            saved = set(self.shard_writer._all_saved)
            quantized_prefixes = tuple(n for n, m in self.model.named_modules() if _ctq(m) and not any(m.children()))
            targets = dict(self.model.named_parameters())
            targets.update(dict(self.model.named_buffers()))
            non_meta_unsaved = [
                pname
                for pname, tensor in self.model.state_dict().items()
                if pname not in saved and tensor.device.type != "meta"
            ]
            n_streamed = 0
            for pname, tensor in self.model.state_dict().items():
                if pname in saved or tensor.device.type != "meta":
                    continue
                if any(pname == q or pname.startswith(q + ".") for q in quantized_prefixes):
                    continue  # quantized layer's original weight - packed name was written
                tgt = targets.get(pname, None)
                if tgt is None:
                    logger.debug(f"[stream] root tensor {pname} has no live parameter/buffer; skipped")
                    continue
                # transformers module names may differ from the checkpoint's
                # (conversion-registry aliases, e.g. MoE router weights)
                ckpt_name = streamer.resolve_checkpoint_name(pname)
                if ckpt_name is None:
                    logger.warning(f"[stream] root tensor {pname} missing from checkpoint; skipped")
                    continue
                streamer._assign_leaf_(self.model, pname, streamer.fetch(ckpt_name))
                # match the non-streaming path's dtype policy: a fully loaded
                # model is converted via model.to(amp_dtype) during context
                # setup, but that cast never touches meta tensors - without
                # this, the streamed export would keep raw checkpoint fp32
                # while the normal export carries the converted dtype
                fetched = dict(self.model.named_parameters())
                fetched.update(dict(self.model.named_buffers()))
                t = fetched.get(pname, None)
                if t is not None and t.is_floating_point() and t.dtype != self.amp_dtype:
                    with torch.no_grad():
                        t.data = t.data.to(self.amp_dtype)
                n_streamed += 1
            logger.debug(
                "[stream] root pass-through: %d tensor(s) streamed from checkpoint, %d already materialized "
                "in memory (%s)",
                n_streamed,
                len(non_meta_unsaved),
                ", ".join(non_meta_unsaved[:6]) + ("..." if len(non_meta_unsaved) > 6 else ""),
            )

            # export reads the in-memory model. Tensors already written to
            # adopted shards (e.g. by an earlier crashed run) are "saved" and
            # therefore skipped above, which would leave them meta in memory;
            # materialize everything the checkpoint can still supply so the
            # model is not mixed meta/real at export time
            for pname, tensor in self.model.state_dict().items():
                if tensor.device.type != "meta":
                    continue
                if any(pname == q or pname.startswith(q + ".") for q in quantized_prefixes):
                    continue
                ckpt_name = streamer.resolve_checkpoint_name(pname)
                if ckpt_name is not None:
                    streamer._assign_leaf_(self.model, pname, streamer.fetch(ckpt_name))

            # computed buffers (rotary inv_freq & friends) never appear in a
            # checkpoint; rebuild them so the model is not mixed meta/real at
            # export time (mixed meta makes packing silently skip everything)
            from auto_round.utils.streaming_calibration import materialize_residual_meta

            materialize_residual_meta(self.model, self.model_context.model.config, torch.device("cpu"))

            # Tree-materialized groups dissolved out of the checkpoint-only
            # verbatim pass below (their prefixes now live in the module
            # tree); write whatever their trees did not pack byte-for-byte
            # (tuned groups already wrote; the write pass is idempotent)
            if tree_groups:
                self._write_unpacked_group_tensors_(streamer, tree_groups)
                self._park_untuned_tree_shells_(tree_groups, mtp_tuned)

            # Checkpoint-only groups (an MTP layer kept as an extra digit block,
            # or a whole top-level subtree transformers strips at load) have no
            # module to quantize or write; pass their tensors through verbatim
            # so the export stays complete.
            for blk in self._checkpoint_only_groups_(streamer):
                names = streamer.names_under(blk)
                if not names:
                    continue
                logger.info(
                    f"[stream] {blk} has no module counterpart; " f"writing {len(names)} checkpoint tensors verbatim"
                )
                for n2 in names:
                    if n2 in materialized_tensors:
                        continue  # claimed by a materialized placeholder (quantized + packed)
                    self.shard_writer.save_tensor(n2, streamer.fetch(n2, raw=True))
            # Auxiliary safetensors files the checkpoint index never references
            # (a family shipping its multi-token-prediction weights as their own
            # file) are invisible to every index-based scan; route their tensors
            # through the shard writer so the export index covers them (loaders
            # discover weights through the index, not by globbing the folder).
            from safetensors import safe_open

            referenced = set(streamer.weight_map.values())
            for fname in sorted(os.listdir(streamer.model_path)):
                if not fname.endswith(".safetensors") or fname in referenced:
                    continue
                path = os.path.join(streamer.model_path, fname)
                if not os.path.isfile(path):
                    continue
                with safe_open(path, framework="pt") as f:
                    keys = list(f.keys())
                    if not keys:
                        continue
                    logger.info(
                        f"[stream] writing {len(keys)} tensor(s) from unreferenced checkpoint file {fname} verbatim"
                    )
                    # one tensor resident at a time: the file can be a whole
                    # predictor block (GB-scale) on memory-tight hosts
                    for k2 in keys:
                        self.shard_writer.save_tensor(k2, f.get_tensor(k2))
        if self.compress_context.is_immediate_saving and not self._gguf_blob_mode():
            self.shard_writer.write(is_finalize=True)
        elif self.compress_context.is_immediate_saving:
            logger.info("[stream] gguf blob mode: final output is assembled by the gguf save path")

        self.model_context.quantized = True
        return self.model, self.layer_config

    def _assert_no_cpu_offload(self) -> None:
        """Fail fast when accelerate had to CPU-offload part of the model.

        The data-driven loop stages each block on the tuning device and feeds
        it the cached block inputs; that invariant assumes every block weight
        is GPU-resident under a contiguous accelerate split. When the visible
        VRAM pool is smaller than the model (observed: 27B bf16 on 2x24GB),
        accelerate silently offloads weights to CPU, block/input device
        co-residency breaks, and the run dies mid-loop with a confusing
        device-mismatch inside a block forward. Streaming mode
        (--stream_quantization) is the sanctioned path for tight VRAM.
        """
        hf_map = getattr(self.model_context.model, "hf_device_map", None)
        if not hf_map:
            return
        offloaded = [name for name, dev in hf_map.items() if str(dev) in ("cpu", "disk")]
        if offloaded:
            raise RuntimeError(
                f"data-driven quantization requires full GPU residency, but accelerate offloaded "
                f"{len(offloaded)} module(s) to CPU/disk (first: {offloaded[0]}). The visible VRAM "
                "pool is smaller than the model -- use more GPUs (no CUDA_VISIBLE_DEVICES "
                "restriction) or --stream_quantization."
            )

    def _quantize_data_driven(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Data-driven quantization path — uses calibration data for optimization."""

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        self._check_compatibility()
        self._assert_no_cpu_offload()

        if bool(self.quant_block_list):
            all_blocks = self.quant_block_list
        else:
            all_blocks = get_block_names(self.model_context.model)

        if len(all_blocks) == 0:
            logger.warning("could not find blocks, exit with original model")
            return self.model_context.model, self.layer_config

        has_gguf = (
            hasattr(self, "formats")
            and self.formats is not None
            and any(fmt.is_gguf() for fmt in (self.formats if isinstance(self.formats, list) else []))
        )
        if has_gguf or self.super_group_size is not None:
            layer_names = []
        else:
            layer_names = _get_quantized_layer_names_outside_blocks(
                model=self.model_context.model,
                layer_config=self.layer_config,
                supported_types=SUPPORTED_LAYER_TYPES,
                quant_block_list=self.quant_block_list,
            )
        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        _last_cache_name = to_cache_block_names[-1] if len(to_cache_block_names) > 1 else None
        to_cache_layer_names = layer_names
        if self.super_group_size is not None:
            to_cache_layer_names = []
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        all_inputs = self.cache_data(
            to_cache_block_names,
            self.calibration_context.nsamples,
            to_cache_layer_names,
            last_cache_name=_last_cache_name,
        )
        # Raw token IDs from the tokenizer, cached during calibration for use in quantize_block.
        input_ids_cache = all_inputs.pop("input_ids", None)
        self.inputs = all_inputs

        all_q_inputs = None
        # Leave it to gguf itself to handle
        if has_gguf and self.alg_composer.need_quanted_input():  # pylint: disable=E1101
            is_quantized_embedding = self.alg_composer.compress_embedding_layer()  #
            clear_memory()
            if is_quantized_embedding:
                all_inputs = copy.deepcopy(self.inputs)
                clear_memory(self.inputs)
                all_q_inputs = self.cache_data(
                    to_cache_block_names,
                    self.calibration_context.nsamples,
                    to_cache_layer_names,
                    last_cache_name=_last_cache_name,
                )
        # Remove accelerate dispatch hooks before moving parameters.
        # hf_device_map is kept for reference but hooks are no longer needed.
        if hasattr(self.model_context.model, "hf_device_map") and len(self.model_context.model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(self.model_context.model)
        self.model_context.model = mv_module_from_gpu(self.model_context.model)
        clear_memory(device_list=device_manager.device_list)
        memory_monitor.log_summary()
        logger.info("caching done")
        if self.compress_context.low_cpu_mem_usage:
            if self.model_context.is_model_patched and not self.compress_context.is_immediate_saving:
                self._offloader(
                    self.model_context.model,
                    all_blocks,
                    clear_memory=True,
                    device_list=device_manager.device_list,
                )
                if not self._offloader.enabled:
                    self.compress_context.low_cpu_mem_usage = False
            elif self.model_context._disk_stream_index is not None:
                # Dense (non-MoE-patched) models normally get low_cpu_mem_usage
                # disabled here because the per-block offload/reload dance is
                # pointless when the whole model is already CPU-resident from
                # the initial full load -- there's no memory to save. That
                # assumption doesn't hold when the model started as a meta
                # skeleton (AR_DISK_STREAM_MODEL=1): blocks are still on meta
                # and must go through the same reload()-before/offload()-after
                # cycle to get materialized from disk one at a time and freed
                # again, so keep it enabled here.
                pass
            else:
                self.compress_context.low_cpu_mem_usage = False
        if len(all_blocks) > 1:
            pbar = tqdm(range(0, sum([len(i) for i in all_blocks]), self.nblocks))
        else:
            pbar = tqdm(range(0, len(all_blocks[0]), self.nblocks))  # move the alg warning outside pbar

        start_time = _time.time()

        self.alg_composer.prepare_run()

        # Build one ResumeState per block group (almost always just one group
        # for text-only dense models) when AR_RESUME_DIR is set, so a
        # crash/kill mid-tuning can resume from the first not-yet-quantized
        # block instead of restarting from block 0. See auto_round/utils/resume.py.
        resume_states = None
        if envs.AR_RESUME_DIR:
            if not self.compress_context.is_immediate_saving and not self.compress_context.low_cpu_mem_usage:
                logger.warning(
                    "AR_RESUME_DIR is set but neither immediate saving nor "
                    "low_cpu_mem_usage is active. Without low_cpu_mem_usage, "
                    "already-quantized blocks are never offloaded anywhere a "
                    "resumed process could find them (see OffloadManager's "
                    "deterministic resume directory in offload.py), so a "
                    "resumed run's in-memory model will have meta/empty "
                    "weights for blocks completed in a PRIOR process. Pass "
                    "low_cpu_mem_usage=True (or a format= to quantize_and_save) "
                    "for resumability to be meaningful here."
                )
            resume_states = self._build_resume_states(all_blocks)
            # a resumed run must never overwrite shards the crashed run wrote:
            # adopt them so the writer continues at the next shard index
            if any(rs is not None and rs.resume_index > 0 for rs in resume_states):
                if self.shard_writer is not None:
                    self.shard_writer.adopt_existing_shards()
                self._adopt_blob_store_()

        _mem_inv = envs.AR_MEM_COUNTERS and logger.isEnabledFor(logging.DEBUG)
        _peak_watch = PeakWatcher() if envs.AR_MEM_COUNTERS else None
        self._peak_watch = _peak_watch
        if _peak_watch is not None:
            _peak_watch.start()
        if _mem_inv:
            self._log_device_inventory(
                None, "cache-built", extra_buckets={"cache-remaining": all_inputs, "cache-q": all_q_inputs or {}}
            )
        for group_idx, block_names in enumerate(all_blocks):
            inputs = all_inputs[block_names[0]]
            all_inputs.pop(block_names[0])
            q_inputs = None
            if all_q_inputs is not None:
                q_inputs = all_q_inputs[block_names[0]]
                all_q_inputs.pop(block_names[0])

            inputs, q_inputs = _update_inputs(inputs, q_inputs)

            clear_memory(self.inputs)

            resume_state = resume_states[group_idx] if resume_states is not None else None
            resume_input_ids = None
            if resume_state is not None and resume_state.resume_index > 0:
                if self.nblocks != 1:
                    logger.warning(
                        "AR_RESUME_DIR is set but nblocks != 1; resuming mid-group is only "
                        "supported for nblocks=1 -- restarting this group from block 0."
                    )
                    resume_state = None
                else:
                    resume_name = block_names[resume_state.resume_index]
                    # Only used here for `input_others` (position/mask info,
                    # which is legitimately re-sourced from this same cache
                    # every iteration regardless of resuming); the actual
                    # chained `input_ids` comes from `resume_input_ids`
                    # below, not this cache -- see
                    # auto_round/utils/resume.py's module docstring for why
                    # the two aren't interchangeable.
                    if resume_name in all_inputs:
                        inputs = all_inputs.pop(resume_name)
                    q_inputs = resume_state.load_q_input()
                    resume_input_ids = resume_state.load_input_ids()
                    if resume_input_ids is None:
                        logger.warning(
                            "AR_RESUME_DIR manifest is missing its cached input_ids tensor; "
                            "restarting this group from block 0 instead of resuming with a "
                            "possibly-inconsistent chain value."
                        )
                        resume_state = None
                    else:
                        pbar.update(resume_state.resume_index)

            self._quantize_blocks(
                self.model_context.model,
                inputs,
                block_names,
                q_input=q_inputs if q_inputs is not None else None,
                nblocks=self.nblocks,
                pbar=pbar,
                input_others_extra_blocks=all_inputs,
                token_ids=input_ids_cache,
                resume_state=resume_state,
                resume_input_ids=resume_input_ids,
            )
            if self.compress_context.is_immediate_packing and len(self.formats) != 1:
                raise ValueError(
                    f"Expected exactly one packing format when 'immediate_packing' is True, "
                    f"but got {len(self.formats)} formats."
                )

        if resume_states is not None:
            if self.compress_context.is_immediate_saving:
                # Don't clear resume state yet when exporting to shards --
                # a crash in the save/export step that follows this method
                # returning (config writing, tokenizer copy, format-specific
                # global packing pass) would otherwise force a full
                # re-tune from block 0 on the next attempt, even though
                # every block's weights are already correctly flushed to
                # disk. quantize_and_save() clears these once
                # save_quantized() actually succeeds.
                self._resume_states = resume_states
            else:
                for rs in resume_states:
                    rs.clear()

        pbar.set_description("Quantizing done")
        pbar.close()
        if self.compress_context.low_cpu_mem_usage:
            if envs.AR_RESUME_DIR and not self.compress_context.is_immediate_saving:
                # `reload(names=None)` only reloads names in
                # `self._offloader._saved` -- populated by THIS process's own
                # offload() calls. A resumed process never touches blocks it
                # skipped via ResumeState (they're left exactly as the meta
                # skeleton started), so they'd never be in `_saved` and would
                # stay meta in the returned model. Request every block
                # explicitly so _reload()'s discovery check (see offload.py)
                # gets a chance to pull each skipped block's real quantized
                # weights back from a prior crashed process's offload dir.
                #
                # Skipped entirely under is_immediate_saving: ShardWriter has
                # already flushed every block's packed weights to disk (both
                # this run's and, via its own discovery, a prior crashed run's),
                # and the `shard_writer.write(is_finalize=True)` call right
                # below would treat any block reloaded back to real memory here
                # as newly-dirty and re-emit its raw, unpacked weight tensor
                # alongside the already-packed one.
                self._offloader.reload(self.model_context.model, flatten_list(all_blocks))
            elif not self.compress_context.is_immediate_saving:
                self._offloader.reload(self.model_context.model)
        self._quantize_layers_outside_blocks(layer_names, all_inputs, token_ids=input_ids_cache)

        convert_module_to_hp_if_necessary(
            self.model_context.model, self.model_context.amp_dtype, device_manager.device, to_cpu=True
        )
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        end_time = _time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
        # mirror the zero-shot path's lifecycle: free member-owned caches
        # (SignRound LFQ/lm_head refs, AWQ/SVDQuant stats) and finalize
        # layer-wise rotation once the block loop is done
        self.alg_composer.finalize_run()
        for n, m in self.model_context.model.named_modules():
            if isinstance(m, tuple(SUPPORTED_LAYER_TYPES)):
                if check_to_quantized(m):
                    quantized_layers.append(n)
                else:
                    unquantized_layers.append(n)
            elif hasattr(m, "scales") or hasattr(m, "scale"):  # packing_immediately
                quantized_layers.append(n)
        summary_info = (
            f"Summary: quantized {len(quantized_layers)}/{len(quantized_layers) + len(unquantized_layers)} in the model"
        )
        if len(unquantized_layers) > 0:
            compressed_unquantized_layers = compress_layer_names(unquantized_layers)
            summary_info += f", unquantized layers: {compressed_unquantized_layers}"
        logger.info(summary_info)

        self.model_context.quantized = True
        return self.model_context.model, self.layer_config

    def _quantize_layers_outside_blocks(
        self,
        layer_names: list,
        layer_inputs: dict,
        token_ids: list[torch.Tensor] | None = None,
    ) -> None:
        """Quantizes specified layers based on inputs and configuration.

        Args:
            layer_names (list): list of layer names to quantize.
            layer_inputs (dict): Dictionary mapping layer names to input data.

        Returns:
            None
        """
        # TODO currently we take all the layers outside blocks as post block layers which is not optimal
        # if there is no input for layer, we use rtn

        for layer_name in copy.deepcopy(layer_names):
            if layer_name not in layer_inputs:
                if self.act_bits < 16 and not self.act_dynamic:
                    if "lm_head" in layer_name:
                        logger.warning_once(
                            "Static activation quantization for lm_head is not fully supported yet. "
                            "If lm_head calibration inputs are missing, activation scale may fall back to unit scale "
                            "or quantization may be skipped."
                        )
                    # Activation quantization requires collected inputs
                    msg_prefix = (
                        f"Activation max hook for layer '{layer_name}' is unavailable due to "
                        f"insufficient collected inputs. "
                    )
                    if "fp8_e5m2" in self.act_data_type:
                        logger.warning(msg_prefix + "Please notes that unit scale is used for this layer.")
                    else:
                        logger.warning(
                            msg_prefix + "Static activation quantization is not supported or ineffective, "
                            "Skipping quantization for this layer."
                        )
                        layer_names.remove(layer_name)
                        continue
                self.alg_composer.compress_layer_outside_block(
                    get_module(self.model, layer_name),
                    disable_opt_rtn=getattr(self, "disable_opt_rtn", False),
                    input_ids=token_ids,
                )
                layer_names.remove(layer_name)
                if self.compress_context.is_immediate_packing:
                    immediate_pack(layer_name, self.layer_config)

                if self.compress_context.is_immediate_saving:
                    m = get_module(self.model, layer_name)
                    self.shard_writer.write(m, name=layer_name, is_finalize=False)
        if len(layer_names) == 0:
            memory_monitor.update()
            memory_monitor.log_summary()
            return
        q_layer_inputs = None
        enable_quanted_input = self.alg_composer.need_quanted_input()
        has_gguf = False

        if hasattr(self, "formats") and self.formats is not None:
            has_gguf = any(format_.is_gguf() for format_ in self.formats)
        if has_gguf and self.compress_context.is_immediate_packing:
            enable_quanted_input = False

        if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1 and enable_quanted_input:
            dispatch_model(self.model, self.model.hf_device_map)

        if enable_quanted_input:
            logger.info("starting to cache layer inputs for %s, this may be quite slow ", layer_names)
            q_layer_inputs = self.cache_data([], self.calibration_context.nsamples, layer_names=layer_names)
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  # self.model.hf_device_map has not been changed
        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        clear_memory()
        for layer_name in layer_names:
            layer_input = layer_inputs[layer_name]
            layer_input = to_device(layer_input, self.compress_context.cache_device)
            q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
            q_layer_input = to_device(q_layer_input, self.compress_context.cache_device)
            self.alg_composer.compress_layer_outside_block(
                get_module(self.model, layer_name),
                fp_inputs=layer_input,
                q_inputs=q_layer_input,
                input_ids=token_ids,
            )
            if self.compress_context.is_immediate_packing:
                immediate_pack(layer_name, self.layer_config)

            if self.compress_context.is_immediate_saving:
                m = get_module(self.model, layer_name)
                self.shard_writer.write(m, name=layer_name, is_finalize=False)
            del layer_input
            clear_memory(q_layer_input)
            memory_monitor.log_summary()

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        # ``seqlen`` clamping is owned by ``CalibrationState``.
        self.calibration_context.clamp_seqlen(self.model_context)

        if self.group_size == 0 and "fp8" not in self.data_type:
            logger.warning("`group_size==0` is not supported for data_type other than fp8 ")

    # This is also for llmc
    def normalize_decoding_layer_inputs_(self, decoding_layer_inputs: list[tuple[tuple[Any, dict[str, Any]]]]) -> None:
        """Replay captured decoding-layer calls to populate ``self.inputs``.

        Converts the raw ``(args, kwargs)`` tuples captured by LLM-Compressor's
        input hook into the ``self.inputs`` dict format expected by
        :meth:`quantize_block`.  The logic mirrors the old-arch implementation in
        ``compressors/base.py``.

        Args:
            decoding_layer_inputs:
                A list of entries captured by a forward hook on the decoding layer.
                Each element is a tuple whose first item is ``(args, kwargs)``.
        """
        first_block_name = self.quant_block_list[0][0]

        class _FakeDecodingLayer(torch.nn.Module):

            def forward(self, *args, **kwargs):
                return args, kwargs

        fake_layer = _FakeDecodingLayer()
        fake_layer.orig_forward = fake_layer.forward
        fake_layer._true_orig_forward = lambda *a, **kw: (a, kw)
        fake_layer.forward = partial(self.calibration._get_block_forward_func(first_block_name), fake_layer)

        self.calibration.inputs = {}
        self.calibration.last_cache_name = None
        for step_input in decoding_layer_inputs:
            args, kwargs = step_input[0]
            fake_layer(*args, **kwargs)

    # This is the API for llm-compressor, not used in AutoRound
    def quantize_block(
        self,
        block: torch.nn.Module,
        inputs: Any,
        q_input: Union[torch.Tensor, dict, None] = None,
        device: Union[str, torch.device] = "cpu",
        auto_offload: bool = True,
        reference_output=None,
    ) -> Any:
        """Quantize a single decoded block of the model (public API for LLM-Compressor).

        This method handles both data-driven and zero-shot (RTN) quantization.
        When calibration data is not needed, ``inputs`` and ``q_input`` are accepted
        for interface compatibility but not used for algorithm purposes.

        Args:
            block: The transformer block (decoder layer) to quantize.
            inputs: Either:

                - the raw decoding-layer inputs captured by
                  LLM-Compressor's hook (list of ``((args, kwargs),)`` tuples),
                  in which case they are normalized via
                  :meth:`normalize_decoding_layer_inputs_`; **or**
                - a :class:`~auto_round.calibration.state.CalibrationState`
                  instance produced by a :class:`~auto_round.calibration.base.Calibrator`,
                  which is bound directly without re-normalization.
            q_input: Optional quantized input from the previous block.  ``None`` on
                the first block.
            device: Target device for quantization (e.g. ``"cuda:0"``).
            auto_offload: When *True*, use the device-map-aware offloading path;
                otherwise move ``block`` directly to ``device``.
            reference_output: Optional pre-computed FP16 reference outputs (list of
                tensors, one per calibration sample). When provided, the internal
                collect_reference forward pass is skipped, saving significant peak
                CPU RAM on large models. Supplied by LLM-Compressor's SequentialPipeline
                via ``AutoRoundModifier.set_fp_ref_outputs()``. Requires auto-round ≥ 0.14.2.

        Returns:
            tuple: ``(q_outputs, reference_output)`` where *q_outputs* is the
            block's output after quantization (or ``None`` when
            ``enable_quanted_input`` is ``False``), and *reference_output* is the
            full-precision reference output collected before optimization.
        """

        if self.diffusion:
            raise NotImplementedError(
                f"Currently, {self.__class__.__name__} does not support quantize_block for diffusion models."
            )

        # Ensure post_init has been called (sets up model_context, compress_context,
        # quantizer, layer_config, etc.).
        if not self._post_init_done:
            self.post_init()

        # ── Zero-shot (RTN) path: no calibration data needed ──────────────────
        if not self.need_calib:
            from auto_round.algorithms.composer import BlockContext

            materialize_model_(block)
            convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)
            block = block.to(device)

            ctx = BlockContext(
                model=self.model,
                block_names=[getattr(block, "global_name", "")],
                block_name=getattr(block, "global_name", ""),
                block_index=0,
            )
            self.alg_composer.compress_block(block, None, {}, block_ctx=ctx, q_inputs=None)

            mv_module_from_gpu(block)
            return None, None

        if len(self.quant_block_list) != 1 or len(self.quant_block_list[0]) != 1:
            raise ValueError(
                f"{self.__class__.__name__}.quantize_block supports exactly one target block, "
                f"but quant_block_list is {self.quant_block_list!r}. "
                "Use to_quant_block_names to select a single block."
            )
        expected_block_name = self.quant_block_list[0][0]
        actual_block_name = getattr(block, "global_name", None)
        if actual_block_name is not None and actual_block_name != expected_block_name:
            raise ValueError(
                f"quantize_block received block {actual_block_name!r}, but cached inputs are for "
                f"{expected_block_name!r}. Pass the matching block or update to_quant_block_names."
            )

        # When called from LLM-Compressor, `wrapped_model` is a single decoder layer
        # (not the full VL model), so it must not be treated as an MLLM regardless of
        # whether the original model had multimodal assets.  Force is_mllm=False for
        # the duration of this call to stay on the standard LLM quantize_block path.
        orig_is_mllm = self.model_context.is_mllm
        self.model_context.is_mllm = False

        if isinstance(inputs, CalibrationContext):
            # Caller already produced a CalibrationState (typically via
            # ``Calibrator.collect``).  Bind it as the authoritative store so
            # the quantizer reads the same ``inputs`` / ``attention_mask`` /
            # ``batch_dim``.
            self.calibration_context = inputs
        else:
            self.normalize_decoding_layer_inputs_(inputs)
        block_inputs = self.calibration.inputs[self.quant_block_list[0][0]]
        input_ids, input_others = self._preprocess_block_inputs(block_inputs, "hidden_states")

        # ── Infrastructure: materialize, dtype convert, device placement ──────
        materialize_model_(block)
        convert_module_to_hp_if_necessary(block, self.model_context.amp_dtype, device)

        if auto_offload:
            if (
                is_auto_device_mapping(device_manager.device_map)
                and len(device_manager.device_list) > 1
                and not self.model_context.is_diffusion
            ):
                from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

                card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                    block,
                    device_manager.device_list,
                    input_ids,
                    self.compress_context.low_gpu_mem_usage,
                    self.calibration_context.batch_size,
                    device,
                )
            else:
                block = block.to(device)
                card_0_in_high_risk, loss_device = False, device
        else:
            card_0_in_high_risk, loss_device = False, device

        if len(device_manager.device_list) > 1 and auto_offload:
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module

            for n, m in block.named_modules():
                if len(list(m.children())) != 0 or not hasattr(m, "tuning_device"):
                    continue
                add_hook_to_module(m, AlignDevicesHook(m.tuning_device, io_same_device=True), True)

        blk_name = self.quant_block_list[0][0]

        bs = self.calibration_context.batch_size

        from auto_round.algorithms.composer import BlockContext

        ctx = BlockContext(
            model=self.model,
            block_names=[blk_name],
            block_name=blk_name,
            block_index=0,
            bs=bs,
            is_mllm=False,
            is_diffusion=False,
        )

        # ── Run block pipeline (calibration → quantization → collection) ──────
        new_q_input, reference_output = self.alg_composer.compress_block(
            block,
            input_ids,
            input_others,
            block_ctx=ctx,
            q_inputs=q_input,
            reference_output=reference_output,
        )

        # ── Cleanup ───────────────────────────────────────────────────────────
        if q_input is not None:
            if input_ids is not q_input:
                clear_memory(input_ids)
            else:
                clear_memory()

        if len(device_manager.device_list) > 1:
            accelerate.hooks.remove_hook_from_submodules(block)
        mv_module_from_gpu(block)
        self.model_context.is_mllm = orig_is_mllm
        return new_q_input, reference_output
