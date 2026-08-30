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
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import autocast

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.data_parallel import (
    accumulate_serial_loss,
    run_paced_replica_steps,
    run_proactive_paced_steps,
    use_one_shot,
)
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    collect_best_params,
    shard_samplers,
)
from auto_round.logger import logger
from auto_round.utils import (
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager


def _rehome_calibration_state(
    active_inputs,
    fp_outputs,
    device,
    margin_bytes: int = 2 * 1024**3,
) -> int:
    """Move a block's calibration tensors to its tuning device, once per block.

    Under streaming with round-robin homes, collected rows are parked on the
    GPU that PRODUCED them (previous block's home / the primary); a block
    tuning on a different device would then pay a cross-device toll on every
    iteration (batch staging inside forward, the reference cat+to, snapshot
    copies). One guarded bulk move replaces those per-iteration tolls.

    Lists are mutated in place. VRAM-guarded: skips (returns 0) when the
    target device cannot hold the off-device bytes with margin. Only CUDA
    devices are considered; CPU targets and CPU-resident inputs are no-ops.
    """
    if device is None or getattr(device, "type", "") != "cuda":
        return 0
    lists = [lst for lst in (active_inputs, fp_outputs) if isinstance(lst, list)]
    off_device = [t for lst in lists for t in lst if torch.is_tensor(t) and t.device != device]
    needed = sum(t.nbytes for t in off_device)
    if needed == 0:
        return 0
    free, _total = torch.cuda.mem_get_info(device)
    if needed + margin_bytes > free:
        logger.info(
            "[tune-locality] keeping calibration state off %s: need %.1fGiB + margin, free %.1fGiB",
            device,
            needed / 2**30,
            free / 2**30,
        )
        return 0
    t0 = time.perf_counter()
    moved = 0
    for lst in lists:
        for i, t in enumerate(lst):
            if torch.is_tensor(t) and t.device != device:
                lst[i] = t.to(device)
                moved += 1
    logger.info(
        "[tune-locality] re-homed %d calibration tensors (%.1fGiB) to %s in %.1fs",
        moved,
        needed / 2**30,
        device,
        time.perf_counter() - t0,
    )
    return moved


def _shard_recipe_anchor(replica_group):
    """Anchor the deferred AR_TUNE_RECIPE grids, sharded across the replicas.

    Every wrapped module's init-search is deterministic (weight + imatrix
    only), so replica r anchoring its round-robin share of the modules
    produces grids bit-identical to a serial anchor. The anchored
    (weight_min, weight_max) tensors and the recipe flags are then broadcast
    to the same-named module on every other replica; frozen recipes pin the
    min/max margins to constants on all copies exactly like a non-deferred
    init would.
    """
    replicas = replica_group.replicas
    home = replica_group.home
    names = [n for n, m in home.named_modules() if getattr(m, "_recipe_anchor_deferred", False)]
    if not names:
        return
    world = len(replicas)
    owner = {n: i % world for i, n in enumerate(names)}

    def _anchor(r):
        rep = replicas[r]
        mods = dict(rep.named_modules())
        targets = [n for n in names if owner[n] == r]
        if not targets:
            return
        dev_r = next(rep.parameters()).device
        if dev_r.type == "cuda":
            with torch.cuda.device(dev_r):
                for n in targets:
                    mods[n].anchor_recipe_grid()
        else:
            for n in targets:
                mods[n].anchor_recipe_grid()

    replica_group.run_threaded([lambda r=r: _anchor(r) for r in range(world)])

    mods_by_rep = [dict(rep.named_modules()) for rep in replicas]
    for n in names:
        src = mods_by_rep[owner[n]][n]
        for r in range(world):
            if r == owner[n]:
                continue
            dst = mods_by_rep[r][n]
            # address-stable broadcast: template-cached mirrors' graphs baked
            # the existing weight_min/max storages -- copy in place when the
            # shapes match (value-identical for fresh replicas too)
            _wm = src.weight_min.to(device=dst.device, dtype=dst.weight_min.dtype)
            _wx = src.weight_max.to(device=dst.device, dtype=dst.weight_max.dtype)
            if isinstance(dst.weight_min, torch.Tensor) and dst.weight_min.shape == _wm.shape:
                dst.weight_min.copy_(_wm)
                dst.weight_max.copy_(_wx)
            else:
                dst.weight_min = _wm
                dst.weight_max = _wx
            dst._tune_recipe = src._tune_recipe
            dst._tune_recipe_frozen_margins = src._tune_recipe_frozen_margins
            dst._recipe_anchor_deferred = False  # grid arrived via broadcast
            if src._tune_recipe_frozen_margins:
                dst._pin_margins_frozen()


def _maybe_qoff_noise(active_inputs, block_index: int, enable_quanted_input: bool):
    """Inject per-channel quantization noise into FP chain inputs (AR_QOFF_NOISE).

    The qoff tuning path optimizes against FP-reference inputs that the
    deployed quantized chain never produces; injecting the measured noise
    statistics of the previous quantized block closes that gap. Deterministic:
    the noise draw is seeded by the block index. Returns a NEW list; the
    cached FP inputs are never modified in place.
    """
    import os

    from auto_round import envs

    if not envs.AR_QOFF_NOISE:
        return active_inputs
    if enable_quanted_input:
        raise ValueError(
            "AR_QOFF_NOISE=1 is a qoff-path aid; with quantized-input chaining (qon) the tuning "
            "already sees real quantized inputs. Unset AR_QOFF_NOISE."
        )
    if not isinstance(active_inputs, list):
        raise ValueError("AR_QOFF_NOISE=1 supports list-of-tensor calibration inputs only.")
    stats_dir = envs.AR_QOFF_NOISE_STATS
    if not stats_dir:
        raise ValueError(
            "AR_QOFF_NOISE=1 requires AR_QOFF_NOISE_STATS=<dir> with the collected per-block noise "
            "stats. Generate them first with a zero-shot pass (e.g. --iters 0) using the SAME "
            "AR_QOFF_NOISE_STATS directory, then run the tuning pass."
        )
    if block_index == 0:
        logger.info("[qoff_noise] block 0 inputs are embeddings; no previous-block stats to inject")
        return active_inputs
    path = os.path.join(stats_dir, f"block_{block_index - 1:04d}.pt")
    if not os.path.exists(path):
        raise ValueError(
            f"AR_QOFF_NOISE stats file {path} is missing. Run a zero-shot collection pass "
            "(--iters 0) with AR_QOFF_NOISE_STATS={stats_dir!r} first."
        )
    stats = torch.load(path, map_location="cpu", weights_only=True)
    mean = stats["mean"].to(torch.float32)
    std = stats["var"].to(torch.float32).clamp_min(0).sqrt()
    gen = torch.Generator(device="cpu").manual_seed(10_000 + block_index)
    out = []
    for t in active_inputs:
        if t.shape[-1] != mean.numel():
            raise ValueError(
                f"AR_QOFF_NOISE stats width ({mean.numel()}) does not match block inputs "
                f"({t.shape[-1]}); the stats dir likely belongs to another model."
            )
        shift = mean + std * torch.randn(t.shape, generator=gen, dtype=torch.float32)
        out.append(t + shift.to(dtype=t.dtype, device=t.device))
    return out


from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.composer import BlockContext


@register_pipeline_member(SignRoundConfig)
def _addr_fingerprint(module):
    """Addresses + checksums of the graph-read state (grids, imatrix, v params)."""
    fp = {}
    for name, m in module.named_modules():
        wm = getattr(m, "weight_min", None)
        if isinstance(wm, torch.Tensor):
            fp[f"{name}.wmin"] = (wm.data_ptr(), float(wm.to(torch.float32).sum()))
        im = getattr(m, "imatrix", None) or getattr(getattr(m, "orig_layer", None), "imatrix", None)
        if isinstance(im, torch.Tensor):
            fp[f"{name}.imat"] = (im.data_ptr(), float(im.to(torch.float32).sum()))
        v = getattr(getattr(m, "params", {}), "get", lambda _k: None)("v")
        if v is not None:
            fp[f"{name}.v"] = (v.data_ptr(), float(v.to(torch.float32).sum()))
    return fp


class SignRoundQuantizer(BaseQuantizer):

    def __init__(self, config: SignRoundConfig) -> None:
        super().__init__(config)
        self.iters = config.iters
        self.lr = config.lr
        self.minmax_lr = config.minmax_lr
        self.lr_scheduler = config.lr_scheduler
        self.momentum = config.momentum
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.enable_norm_bias_tuning = config.enable_norm_bias_tuning
        self.gradient_accumulate_steps = config.gradient_accumulate_steps

        self.enable_alg_ext = config.enable_alg_ext
        self.not_use_best_mse = config.not_use_best_mse
        self.enable_quanted_input = config.enable_quanted_input
        self.dynamic_max_gap = config.dynamic_max_gap
        self.enable_lfq = config.enable_lfq

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block
        # Kept for per-layer (mixed-bit) lr resolution during tuning.
        self._config = config
        self.lr_is_auto = getattr(config, "lr_is_auto", False)
        self.minmax_lr_is_auto = getattr(config, "minmax_lr_is_auto", False)
        # Emit the low-bit lr notice at most once across all blocks/layers.
        self._logged_low_bit_lr = False

    def _maybe_log_low_bit_lr(self, bits) -> None:
        """Log once when low-bit (<=3) layers get the higher 2.0/iters lr."""
        if self._logged_low_bit_lr or not self.lr_is_auto:
            return
        if self.iters >= 1000 and bits is not None and bits <= 3:
            logger.info("using higher lr (2.0/iters) for <=3 bit layers to improve accuracy")
            self._logged_low_bit_lr = True

    @staticmethod
    def _pin_stream_home(block, home):
        """Pin every leaf with parameters to the streamed home device.

        Sets ``tuning_device`` (the designed per-layer override that wrappers
        and outside-block quantization honor) so SignRound tuning, the loss,
        and packing all stay on the block's home instead of being re-sharded
        across the full device_map or dragged to the global primary.
        """
        home = torch.device(home)
        for _n, _mod in block.named_modules():
            if list(_mod.children()):
                continue
            if any(True for _ in _mod.parameters(recurse=False)):
                _mod.tuning_device = home
        return block

    def dispatch_block(self, block, input_ids, input_others):
        """Multi-GPU aware block dispatch for SignRound tuning.

        Stores _card_0_in_high_risk and _loss_device on self for use in quantize_block.
        Returns the block after device placement.
        """
        from auto_round.utils import is_auto_device_mapping

        stream_home = getattr(block, "_stream_home_device", None)
        if stream_home is not None:
            # streaming loop: the block was streamed onto this device and
            # rehomed there - it is the single tuning device, never sharded
            return self._pin_stream_home(block, stream_home)
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
                device_manager.device,
            )
            if len(device_manager.device_list) > 1:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in block.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
        else:
            block = block.to(device_manager.device)
            card_0_in_high_risk, loss_device = False, device_manager.device

        self._card_0_in_high_risk = card_0_in_high_risk
        self._loss_device = loss_device
        return block

    def _get_non_zero_cnt(self, tensor: list[torch.Tensor], indices: list[int]) -> int:
        current_tensors = [tensor[i] for i in indices]
        non_zero_cnt = 0
        for t in current_tensors:
            non_zero_cnt += torch.count_nonzero(t).item()
        return non_zero_cnt

    def _get_loss(
        self,
        pred_output: torch.Tensor,
        ref_output: torch.Tensor,
        indices: torch.Tensor,
        loss_func: Callable,
        device: Union[str, torch.device] = "cpu",
        valid_token_mask: Optional[torch.Tensor] = None,
        input_ids=None,
    ):
        autocast_ctx = (
            nullcontext()
            if self.model_context.amp
            else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
        )
        if valid_token_mask:
            tmp_attention_mask = [valid_token_mask[i].to(device) for i in indices]
            tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0)
            tmp_attention_mask.unsqueeze_(-1)

            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    (pred_output * tmp_attention_mask).to(torch.float32),
                    (ref_output * tmp_attention_mask).to(torch.float32),
                )
        else:
            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    pred_output.to(torch.float32), ref_output.to(torch.float32)
                )

        return loss

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.calibration_context.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.calibration_context.batch_dim]

    def _init_lm_components(self) -> None:
        """Lazily locate and cache ``_post_block_modules`` and ``_lm_head`` for LFQ loss.

        ``_post_block_modules`` is an ordered list of modules that must be applied
        between the last transformer-block output and the lm_head projection.
        Most architectures have a single final norm; OPT additionally has an
        optional ``project_out`` projection.

        Supported without manual configuration:
            LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3 — ``model.model.norm``
            OPT — ``model.model.decoder.{final_layer_norm, project_out}`` (both optional)
            GPT-2 / Falcon / Bloom — ``model.transformer.ln_f``
            GPT-NeoX / Pythia — ``model.gpt_neox.final_layer_norm``
            Phi / Phi-2 — ``model.model.final_layernorm``
            MPT — ``model.transformer.norm_f``
            ChatGLM — ``model.transformer.encoder.final_layernorm``
            RWKV — ``model.rwkv.ln_out``

        Raises ``AttributeError`` if no lm_head equivalent can be found.
        """
        if hasattr(self, "_lm_head"):
            return

        model = self.model

        # ── lm_head ──────────────────────────────────────────────────────────
        for name in ("lm_head", "embed_out", "output", "head"):
            if hasattr(model, name):
                self._lm_head = getattr(model, name)
                break
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. " "Checked: lm_head, embed_out, output, head."
            )

        # ── post-block processing (ordered list applied before lm_head) ───────
        # OPT: decoder has both an optional final_layer_norm *and* an optional
        # project_out that maps ffn_dim → word_embed_proj_dim.  Detect by probing
        # for the characteristic project_out attribute (may be None).
        try:
            decoder = model.model.decoder
            _ = decoder.project_out  # raises AttributeError if not an OPT decoder
            self._post_block_modules = [m for m in (decoder.final_layer_norm, decoder.project_out) if m is not None]
            return
        except AttributeError:
            pass

        # All other architectures: single optional final norm.
        norm_getters = [
            lambda: model.model.norm,  # LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3
            lambda: model.transformer.ln_f,  # GPT-2 / Falcon / Bloom
            lambda: model.gpt_neox.final_layer_norm,  # GPT-NeoX / Pythia
            lambda: model.model.final_layernorm,  # Phi / Phi-2
            lambda: model.transformer.norm_f,  # MPT
            lambda: model.transformer.encoder.final_layernorm,  # ChatGLM
            lambda: model.rwkv.ln_out,  # RWKV
        ]
        self._post_block_modules = []
        for getter in norm_getters:
            try:
                norm = getter()
                if norm is not None:
                    self._post_block_modules = [norm]
                    break
            except AttributeError:
                continue

    # Keywords that identify non-text (visual / audio / multimodal) blocks.
    # LFQ loss is only meaningful for pure language-model decoder blocks.
    _NON_TEXT_BLOCK_KEYWORDS = frozenset(
        {
            "vis",
            "vision",
            "visual",
            "image",
            "img",
            "audio",
            "video",
            "patch",
            "pixel",
            "clip",
            "vit",
            "perceiver",
            "resampler",
            "connector",
            "projector",
        }
    )

    def _is_text_decoder_block(self, block_name: str) -> bool:
        """Return ``True`` if *block_name* refers to a text-decoder block.

        Blocks whose names contain any of the non-text keywords (vision, audio,
        image, …) are considered multimodal and excluded from LFQ loss.
        """
        name_lower = block_name.lower()
        return not any(kw in name_lower for kw in self._NON_TEXT_BLOCK_KEYWORDS)

    def lfq_loss(self, hidden_state: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute LM cross-entropy loss from the last block's hidden states.

        Applies every post-block module (final norm, optional projection, …) in
        order, then runs lm_head and computes next-token prediction loss.
        Positions marked with ``-100`` in *input_ids* are excluded from the loss.

        Args:
            hidden_state: Last block output, shape ``[batch, seq_len, hidden]``.
            input_ids:    Token-ID labels with ``-100`` for ignored positions,
                          shape ``[batch, seq_len]``.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        self._init_lm_components()
        device = hidden_state.device

        for module in self._post_block_modules:
            module.to(device)
            hidden_state = module(hidden_state)

        self._lm_head.to(device)
        logits = self._lm_head(hidden_state)

        if hasattr(self.model, "loss_function"):
            loss = self.model.loss_function(
                logits=logits,
                labels=input_ids.to(device),
                vocab_size=self.model.config.vocab_size,
            )
        else:
            import torch.nn.functional as F

            # Standard causal-LM shift: predict token t+1 from hidden state t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous().to(device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        input_ids=None,
        **kwargs,
    ) -> dict:
        """Apply the AutoRound optimization algorithm to a block.

        This is the pure-algorithm entry point.  All infrastructure concerns
        (device placement, act-max hook collection, DDP setup, memory cleanup,
        logging) are handled by the Compressor before and after this call.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as the optimization
                target for the sign-gradient descent loss (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample). Used to derive the valid-token loss mask once (result
                cached on ``self._cached_valid_token_mask`` for reuse across
                all blocks). ``None`` disables loss masking.
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Best quantization parameters found during optimization, or an
                empty dict if no trainable parameters were found.
        """
        from auto_round.utils.tune_profile import stage as _bp_stage

        _prof_bp = kwargs.pop("_block_prof", None)

        device = getattr(block, "_stream_home_device", None) or device_manager.device
        loss_device = getattr(self, "_loss_device", device)
        card_0_in_high_risk = getattr(self, "_card_0_in_high_risk", False)
        mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

        valid_token_mask = None
        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Result is cached on self so it is computed only once across all blocks.
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        # Use quantized inputs if available and enabled
        active_inputs = q_inputs if (q_inputs is not None and self.enable_quanted_input) else fp_inputs
        active_inputs = _maybe_qoff_noise(active_inputs, block_ctx.block_index, self.enable_quanted_input)
        _home = torch.device(device) if not isinstance(device, torch.device) else device
        from auto_round.algorithms.quantization.sign_round.data_parallel import expect_pool_local

        # calibration-state placement is DEFERRED: under an engaged DDP plan the
        # pools are scattered across the plan devices (shard-local reads); the
        # gather-to-home re-home runs only on the serial path
        _rehome_deferred = (active_inputs, fp_outputs, _home)
        nsamples = len(active_inputs) if isinstance(active_inputs, list) else self._count_samples(active_inputs)

        batch_size = self.calibration_context.batch_size
        global_batch_size = batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)

        # ── Optional single-process data parallelism (AR_TUNE_DDP_WORLD) ─────
        # eligibility resolved BEFORE wrapping: when DDP engages, the recipe
        # init-search anchor is DEFERRED and runs sharded across the replica
        # devices (deterministic search on local weights) right after the
        # mirrors exist -- and before tuning-parameter collection, so frozen
        # recipes remove the margin parameters everywhere consistently.
        replica_group = None
        _tpl_entry = None
        mirror_optimizers = []
        mirror_schedules = []
        params_per_replica = []
        from auto_round import envs as _envs

        _dp_world = int(getattr(_envs, "AR_TUNE_DDP_WORLD", 1) or 1)
        _dp_placed = False  # True once pools are distributed across the engaged plan
        _scaler_early = self._get_scaler()  # pylint: disable=assignment-from-none
        _dp_eligible = (
            _dp_world > 1
            and self.iters > 0
            and _home.type == "cuda"
            and _scaler_early is None
            and self.gradient_accumulate_steps == 1
            and valid_token_mask is None
            and not self.enable_lfq
            and isinstance(active_inputs, list)
            and isinstance(fp_outputs, list)
        )
        with _bp_stage(_prof_bp, "wrap_search"):
            quantized_layer_names, unquantized_layer_names = self.wrapper_block(
                block,
                self.enable_minmax_tuning,
                self.enable_norm_bias_tuning,
                enable_torch_compile=self.compress_context.enable_torch_compile,
                device=device,
                defer_recipe_anchor=_dp_eligible,
            )
        _plan = None
        if _dp_eligible:
            from auto_round.algorithms.quantization.sign_round.data_parallel import ReplicaGroup, resolve_ddp_plan

            _explicit = [d.strip() for d in str(_envs.AR_TUNE_DDP_DEVICES or "").split(",") if d.strip()]
            _mirror_bytes = sum(p.numel() * p.element_size() for p in block.parameters()) + sum(
                p.numel() * p.element_size()
                for _m in block.modules()
                if hasattr(_m, "orig_layer")
                for p in _m.params.values()
            )
            _free = None
            try:
                _free = {
                    torch.device("cuda", idx): torch.cuda.mem_get_info(idx)[0]
                    for idx in range(torch.cuda.device_count())
                }
            except Exception:  # pragma: no cover - non-CUDA reachability
                _free = None
            from auto_round.algorithms.quantization.sign_round.data_parallel import effective_ddp_groups

            _plan = resolve_ddp_plan(
                _dp_world,
                _home,
                global_batch_size,
                groups=effective_ddp_groups(),
                visible_cuda_devices=list(range(torch.cuda.device_count())) if _free else None,
                explicit_devices=_explicit or None,
                vram_free_bytes=_free,
                mirror_footprint_bytes=_mirror_bytes,
            )
            if _plan.enabled and _plan.world & (_plan.world - 1) == 0:
                from auto_round.algorithms.quantization.sign_round.data_parallel import distribute_pool

                # distributed calibration pool: shard-local tune reads; replaces
                # the serial gather-to-home re-home (each device takes 1/world)
                with _bp_stage(_prof_bp, "distribute"):
                    distribute_pool(active_inputs, _plan.devices)
                    distribute_pool(fp_outputs, _plan.devices)
                _dp_placed = True
                _staged_src = getattr(block, "_stream_prefetch_source", None)
                if _staged_src is not None:
                    _graphs_streamer = getattr(_staged_src, "streamer", None)
                    _staged_src = _staged_src.unpack()
                from auto_round.algorithms.quantization.sign_round.data_parallel import (
                    _GRAPH_TEMPLATE_CACHE,
                    _template_signature,
                    sync_mirror_from_home,
                    template_cache_size,
                )

                _tpl_entry = None
                _tpl_key = None
                _tpl_home = None  # persistent HOME staging replica (template cache)
                if template_cache_size() > 0:
                    _tpl_key = _template_signature(
                        block,
                        _plan.world,
                        _plan.devices,
                        extra=(
                            str(_envs.AR_TUNE_RECIPE),
                            getattr(self, "asym_search", None),
                            self.enable_minmax_tuning,
                            nsamples,
                            global_batch_size,
                            tuple(fp_outputs[0].shape) if isinstance(fp_outputs, list) and fp_outputs else None,
                        ),  # per-module bits/group/data_type/sym travel via named_modules
                    )
                    _tpl_entry = _GRAPH_TEMPLATE_CACHE.pop(_tpl_key, None)
                    if _tpl_entry is not None and len(_tpl_entry["replicas"]) != _plan.world:
                        _tpl_entry = None  # stale layout -- rebuild
                if _tpl_entry is not None:
                    if "ptrs" in _tpl_entry:
                        for _r, _rep in enumerate(_tpl_entry["replicas"]):
                            _old_fp, _new_fp = _tpl_entry["ptrs"][_r], _addr_fingerprint(_rep)
                            for _k, _old in _old_fp.items():
                                _new = _new_fp.get(_k)
                                if _new is None or _old is None:
                                    continue
                                if _new[0] != _old[0]:
                                    logger.warning(
                                        "[tune-ddp] template cache: replica %d %s REBOUND between blocks "
                                        "(ptr %s -> %s) -- the captured graph still reads the OLD address",
                                        _r,
                                        _k,
                                        _old[0],
                                        _new[0],
                                    )
                    _tpl_home = _tpl_entry["replicas"][0]
                    for _rep in _tpl_entry["replicas"]:
                        sync_mirror_from_home(block, _rep)
                    replica_group = ReplicaGroup(
                        block,
                        _plan,
                        grad_transport=_envs.AR_TUNE_DDP_GRAD_TRANSPORT,
                        staged_source=None,
                        mirrors=_tpl_entry["replicas"][1:],
                        home_replica=_tpl_home,
                    )
                    logger.info(
                        "[tune-ddp] graph template cache hit: %d replica(s) adopted incl. home staging "
                        "(captured graphs kept)",
                        len(_tpl_entry["replicas"]),
                    )
                else:
                    # miss: a DEDICATED home staging copy (the real block is
                    # packed/meta'd by the orchestrator after the block -- it
                    # must never enter the cache). Uniform flow: the loop
                    # always tunes the staging home and copies the final
                    # iterate back at loop end.
                    if template_cache_size() > 0:
                        _tpl_home = copy.deepcopy(block)
                    replica_group = ReplicaGroup(
                        block,
                        _plan,
                        grad_transport=_envs.AR_TUNE_DDP_GRAD_TRANSPORT,
                        staged_source=_staged_src,
                        home_replica=_tpl_home,
                    )
                for note in _plan.notes:
                    logger.info("[tune-ddp] %s", note)
                with _bp_stage(_prof_bp, "sharded_anchor"):
                    _shard_recipe_anchor(replica_group)
                if _tpl_entry is not None:
                    _mods_h = dict(block.named_modules())
                    for _r, _mir in enumerate(replica_group.replicas[1:], start=1):
                        _dh, _dm = 0.0, 0.0
                        for _n, _hm in _mods_h.items():
                            _mm = dict(_mir.named_modules()).get(_n)
                            if _mm is None:
                                continue
                            for _attr in ("weight_min", "weight_max"):
                                _a, _b = getattr(_hm, _attr, None), getattr(_mm, _attr, None)
                                if isinstance(_a, torch.Tensor) and isinstance(_b, torch.Tensor):
                                    _dh += float(_a.to(torch.float32).sum())
                                    _dm += float(_b.to(torch.float32).sum())
                        if abs(_dh - _dm) > 1e-3 * max(1.0, abs(_dh)):
                            logger.warning(
                                "[tune-ddp] template cache: mirror %d grids diverge from home "
                                "(home sum %.4f vs mirror %.4f) -- stale anchor path",
                                _r,
                                _dh,
                                _dm,
                            )
                logger.info(
                    "[tune-ddp] engaged: world=%d shard=%d devices=%s grad_transport=%s",
                    _plan.world,
                    _plan.shard_size,
                    [str(d) for d in _plan.devices],
                    _envs.AR_TUNE_DDP_GRAD_TRANSPORT,
                )
            elif _plan.enabled:
                logger.info("[tune-ddp] disabled: resolved world %d is not a power of two", _plan.world)

        def _collect_tuning_params(mod):
            """Collect (round, minmax) params + per-lr groups from a wrapped block."""
            r_params, m_params = [], []
            r_groups = {}
            m_groups = {}
            for _n, m in mod.named_modules():
                if hasattr(m, "orig_layer"):
                    layer_bits = getattr(m.orig_layer, "bits", None)
                    layer_lr = self._config.compute_lr(layer_bits)
                    if layer_lr is None:
                        layer_lr = self.lr
                    self._maybe_log_low_bit_lr(layer_lr)
                    layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
                    if layer_minmax_lr is None:
                        layer_minmax_lr = self.minmax_lr
                    for key in m.params.keys():
                        if "min" in key or "max" in key:
                            m_params.append(m.params[key])
                            m_groups.setdefault(float(layer_minmax_lr), []).append(m.params[key])
                        else:
                            r_params.append(m.params[key])
                            r_groups.setdefault(float(layer_lr), []).append(m.params[key])
            return r_params, m_params, r_groups, m_groups

        # Group parameters by their effective lr so that mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) use a per-layer lr
        # derived from each layer's own bit-width.
        round_params, minmax_params, round_lr_groups, minmax_lr_groups = _collect_tuning_params(block)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            return {}

        # Build optimizer param groups with a per-layer lr for the rounding
        # parameters (and min-max parameters when enabled). Under the template
        # cache the loop tunes the STAGING home -- the optimizer must step the
        # staging params (the real block's copies never receive grads).
        if replica_group is not None and replica_group.tune_home is not block:
            round_params, minmax_params, round_lr_groups, minmax_lr_groups = _collect_tuning_params(
                replica_group.tune_home
            )
        params = [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in round_lr_groups.items()]
        if self.enable_minmax_tuning:
            params += [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in minmax_lr_groups.items()]

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        if self.gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
        total_loss = 0
        # Compute num_elm once before the loop (used to normalise the accumulated loss).
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not valid_token_mask:
            whole_indices = torch.arange(global_batch_size)
            if isinstance(active_inputs, list):  # dict for diffusion, tricky setting, not sure whether it's correct
                num_elm = sum(active_inputs[i.item()].numel() for i in whole_indices)

        block, sync_gradients = setup_ddp_if_needed_(self, block, device_manager.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        # shard-constrained DDP sampling: with the distributed pool, device r
        # owns samples [r*shard, (r+1)*shard); drawing each replica's
        # sub-batch from its own shard keeps every per-iteration pool read
        # device-local (a global shuffled batch scatters pieces across
        # devices and costs cross-device cats every iteration)
        _dp_samplers = None
        if (
            replica_group is not None
            and _dp_placed
            and global_batch_size % replica_group.world == 0
            and _envs.AR_TUNE_DDP_SHARD_SAMPLER
        ):
            _dp_samplers = shard_samplers(nsamples, replica_group.world, global_batch_size // replica_group.world)
        block_fwd = self.block_forward

        # CUDA graphs for the DDP tune loop: the per-replica step is split
        # into an eager per-iteration PREPARE (gather this iteration's shard
        # into stable device buffers) and a captured COMPUTE (fwd + loss +
        # bwd on those buffers). Rotating shard contents stay correct --
        # only the addresses must repeat. Engagement is environment-gated
        # only (list inputs / not diffusion / no valid-token mask); anything
        # failing during prepare/capture/replay raises and halts the run.
        _graphs_active = False
        _graphs_gate = None
        _graphed_steps = None
        _graphs_warmups_left = 0
        _graphs_capture_pending = False
        _graphs_defer_logged = False
        _graphs_streamer = None
        from auto_round.algorithms.quantization.sign_round.data_parallel import cuda_graphs_engage

        # background pack / ready-transform threads that may issue CUDA
        # concurrently with this block's tune loop (streaming pipeline):
        # capture DEFERS until they finish -- global-mode capture needs the
        # process CUDA-quiet, and joining the packer would serialize ~half
        # the block wall
        _bg_attr = getattr(block, "_bg_cuda_threads", None)
        _bg_cuda_threads = [t for t in (_bg_attr.value if _bg_attr is not None else ()) if t is not None]
        if (
            _envs.AR_TUNE_CUDA_GRAPHS
            and replica_group is not None
            and cuda_graphs_engage(
                replica_group,
                inputs_are_list=isinstance(active_inputs, list),
                is_diffusion=getattr(block_fwd, "is_diffusion", False),
                has_valid_token_mask=valid_token_mask is not None,
            )
        ):
            _graphs_active = True
            _ga = getattr(block, "_graphs_capture_gate", None)
            _graphs_gate = _ga.value if _ga is not None else None
            if _graphs_gate is not None:
                # capture-first: hold bg pack/ready work until THIS block's
                # graphs are captured (released right after; the gate OWNER
                # also releases on every exit, so workers can wait unbounded)
                _graphs_gate.clear()

        def _make_graphed_step(r, step=None):
            # Static buffers: the FIRST gather's tensors become the
            # graph's baked inputs (their addresses must repeat);
            # every later prepare refreshes them in place. Shared
            # leaves (rotary tables) keep their identity -- no copy.
            # A template-cache hit passes the PERSISTENT st (and the step
            # object holding the captured graph): the fresh closures bind to
            # THIS block's pools while the baked addresses stay alive.
            rep = replica_group.replicas[r]
            dev_r = next(rep.parameters()).device
            st = step._ar_st if step is not None else None  # persistent statics of the cached graph
            if st is None:
                st = {"inputs": None, "others": None, "ref": None}
            from auto_round.algorithms.quantization.sign_round.data_parallel import HostLeafPinner, _copy_into_static

            pinner = HostLeafPinner()
            _pinned_once = False

            def prepare(shard):
                nonlocal _pinned_once
                with torch.cuda.device(dev_r):
                    expect_pool_local([fp_outputs[j] for j in shard], dev_r, "ddp-ref")
                    ref = torch.cat([fp_outputs[j].to(dev_r) for j in shard], dim=0)
                    bins, bother = [], []
                    if _fence_free:
                        # host-int batches: no device indices tensor (pageable
                        # H2D) and no per-element .item() fences in select_batch
                        for i in range(0, len(shard), block_fwd.batch_size):
                            bi = shard[i : i + block_fwd.batch_size]
                            _in, _oth = block_fwd.select_batch(active_inputs, input_others, bi)
                            bins.append(_in)
                            bother.append(pinner.wrap(_oth))
                        if not _pinned_once:
                            _pinned_once = True
                            pinner.pin_repeats()
                    else:
                        # legacy gather, verbatim (A/B isolation for the gate)
                        indices = torch.tensor(shard, dtype=torch.long, device=dev_r)
                        for i in range(0, len(indices), block_fwd.batch_size):
                            bi = indices[i : i + block_fwd.batch_size]
                            _in, _oth = block_fwd.select_batch(active_inputs, input_others, bi)
                            bins.append(_in)
                            bother.append(_oth)

                    def _to_dev(x):
                        if isinstance(x, torch.Tensor) and x.device != dev_r:
                            return x.to(dev_r)
                        if isinstance(x, dict):
                            return {k: _to_dev(v) for k, v in x.items()}
                        if isinstance(x, (tuple, list)):
                            return type(x)(_to_dev(v) for v in x)
                        return x

                    if st["ref"] is None:
                        # shallow-copy the dict nodes: block_forward MUTATES
                        # the kwargs it receives (pops keys); the static
                        # structure must stay pristine for later compares.
                        # Stage every leaf on dev_r: to_device() inside
                        # the captured region must no-op (unpinned H2D
                        # during capture is illegal); later prepares
                        # refresh cross-device eagerly via copy_.
                        st["ref"], st["inputs"] = _to_dev(ref), [_to_dev(b) for b in bins]
                        st["others"] = [_to_dev(dict(o)) for o in bother]
                        return
                    _copy_into_static(st["ref"], ref)
                    _copy_into_static(st["inputs"], bins)
                    _copy_into_static(st["others"], bother)

            def compute():
                with torch.cuda.device(dev_r):
                    outs = []
                    for _in, _oth in zip(st["inputs"], st["others"]):
                        # dict() per call: the runner pops keys from the
                        # kwargs it receives (same tensor leaves -- the
                        # graph-baked addresses are untouched)
                        raw = block_fwd._forward_one_batch(rep, _in, dict(_oth))
                        out = block_fwd._normalize_output(raw, rep)
                        outs.append(out.to(dev_r) if out.device != dev_r else out)
                    pred_r = outs[0] if len(outs) == 1 else torch.cat(outs, dim=block_fwd.batch_dim)
                    # valid_token_mask is None on this path, so the
                    # indices argument of _get_loss is never consumed
                    loss_r = self._get_loss(pred_r, st["ref"], None, mse_loss, dev_r, None)
                    loss_r.backward()
                    return loss_r

            from auto_round.algorithms.quantization.sign_round.data_parallel import GraphedReplicaStep

            if step is not None:
                step.prepare = prepare
                step.compute = compute
                step._ar_st = st  # template cache: the static dict travels with the step
                return step
            _gs = GraphedReplicaStep(prepare, compute, name=f"replica-{r}", device=dev_r)
            _gs._ar_st = st
            return _gs

        _graphed_worker_ms = [[] for _ in range(replica_group.world) if True] if replica_group is not None else []

        def _dp_run(r):
            _t0 = time.perf_counter()
            _losses[r] = _graphed_steps[r](_shards[r])
            _graphed_worker_ms[r].append((time.perf_counter() - _t0) * 1000.0)

        if _graphs_active and _graphed_steps is None:
            # built BEFORE the ddp warm-up pass: the warm-up backward must run
            # on the steps' capture streams (see the warm-up site)
            _graphed_steps = [
                _make_graphed_step(r, step=(_tpl_entry["steps"][r] if _tpl_entry is not None else None))
                for r in range(replica_group.world)
            ]
            _graphs_warmups_left = (
                0 if getattr(_graphed_steps[0], "_graph", None) is not None else _graphed_steps[0].warmup_iters
            )
            if all(getattr(_gs, "_graph", None) is not None for _gs in _graphed_steps):
                # full template hit: nothing will be captured -- do not hold
                # the bg gate or wait for a quiet process (replays need neither)
                _graphs_capture_pending = False
            logger.info(
                "[tune-ddp] cuda graphs enabled: %d replica step(s), capture after %d warmup iteration(s)",
                len(_graphed_steps),
                _graphs_warmups_left,
            )

        _bg_busy = [t for t in _bg_cuda_threads if t.is_alive()]
        _prefetch_busy = _graphs_streamer is not None and _graphs_streamer.prefetch_pending()

        # When low_gpu_mem_usage is enabled, active_inputs / fp_outputs are intentionally
        # kept on CPU to limit GPU memory.  However block_fwd normally routes pred_output
        # through CPU (cache_device="cpu") and the very next line moves it back to
        # loss_device — a wasteful GPU→CPU→GPU roundtrip on every batch × iteration.
        # pred_output is a transient single-batch tensor consumed immediately for the
        # loss and then freed, so keeping it on the compute device costs no persistent
        # extra memory.  Pass it as a per-call override so self.cache_device is unchanged.
        _fwd_cache_device = (
            device
            if getattr(self.compress_context, "low_gpu_mem_usage", False) and not str(device).startswith("cpu")
            else None
        )

        from auto_round import envs as _envs
        from auto_round.utils.tune_profile import make_tune_profiler
        from auto_round.utils.tune_profile import stage as tune_stage

        # ── Optional single-process data parallelism (AR_TUNE_DDP_WORLD) ─────
        # (eligibility, plan, mirrors, the sharded recipe anchor and the pool
        # distribution ran BEFORE parameter collection -- see the early block
        # after wrapper_block; only mirror optimizers + warm-up remain here)
        if _dp_eligible and replica_group is not None:
            # mirror-side optimizers replicate the home group structure
            for mirror in replica_group.mirrors:
                r_ps, m_ps, r_gr, m_gr = _collect_tuning_params(mirror)
                m_params = [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in r_gr.items()]
                if self.enable_minmax_tuning:
                    m_params += [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in m_gr.items()]
                m_opt = self.optimizer(m_params, lr=lr, weight_decay=0, **extra_kwargs)
                m_sched = (
                    torch.optim.lr_scheduler.LinearLR(m_opt, start_factor=1.0, end_factor=0.0, total_iters=self.iters)
                    if self.lr_scheduler is None
                    else copy.deepcopy(self.lr_scheduler)
                )
                mirror_optimizers.append(m_opt)
                mirror_schedules.append(m_sched)
            params_per_replica = [
                r_ps + m_ps if self.enable_minmax_tuning else r_ps
                for r_ps, m_ps, _r, _m in (_collect_tuning_params(rep) for rep in replica_group.replicas)
            ]
            # Warm up every replica SERIALLY in the main thread: torch.compile
            # materializes its per-device kernels lazily at first call, and
            # compiling from several worker threads at once races in dynamo.
            # The warm-up also validates each mirror end-to-end; a failure
            # ABORTS the run (DDP was explicitly requested -- continuing
            # serially would silently invalidate the configuration and any
            # measurement against it). Grads are discarded.
            try:
                with _bp_stage(_prof_bp, "ddp_setup"):
                    for r, rep in enumerate(replica_group.replicas):
                        _warm = list(range(r * _plan.shard_size, (r + 1) * _plan.shard_size))
                        _dev_r = next(rep.parameters()).device
                        _warm_step = _graphed_steps[r] if _graphed_steps is not None else None
                        _warm_stream = _warm_step._capture_stream if _warm_step is not None else None
                        with torch.cuda.device(_dev_r):
                            # graphs on: the FIRST backward per replica must run
                            # on the capture stream -- AccumulateGrad nodes are
                            # created once per parameter and keep their birth
                            # stream; default-stream birth dooms every later
                            # capture (legacy-stream dependency)
                            if _warm_stream is not None:
                                _warm_stream.wait_stream(torch.cuda.current_stream(_dev_r))
                            import contextlib

                            _sctx = (
                                torch.cuda.stream(_warm_stream)
                                if _warm_stream is not None
                                else contextlib.nullcontext()
                            )
                            with _sctx:
                                expect_pool_local([fp_outputs[j] for j in _warm], _dev_r, "warmup-ref")
                                _ref_w = torch.cat([fp_outputs[j].to(_dev_r) for j in _warm], dim=0)
                                _pred_w = block_fwd.forward(rep, active_inputs, input_others, _warm, _dev_r)
                                _loss_w = self._get_loss(_pred_w, _ref_w, _warm, mse_loss, _dev_r, None)
                                _loss_w.backward()
                            if _warm_stream is not None:
                                torch.cuda.current_stream(_dev_r).wait_stream(_warm_stream)
                    for _opt in [optimizer] + mirror_optimizers:
                        # graphs replay backward into the captured grad
                        # addresses: the warm-up backward above allocated
                        # them, freeing here would make replay write into
                        # recycled memory
                        _opt.zero_grad(set_to_none=not _graphs_active)
            except Exception as _warm_err:  # noqa: BLE001 - re-raised with context
                replica_group.teardown()
                raise RuntimeError(
                    "AR_TUNE_DDP_WORLD was requested but the replica warm-up failed -- "
                    "DDP was explicitly engaged, so refusing to continue on the serial "
                    "path (that would silently invalidate the requested configuration "
                    "and any measurement against it). Fix the underlying failure or "
                    "unset AR_TUNE_DDP_WORLD."
                ) from _warm_err

        if not _dp_placed:  # serial path (or DDP ineligible): gather-to-home as before
            _rehome_calibration_state(*_rehome_deferred)
        tune_prof = make_tune_profiler(device)
        from auto_round.algorithms.quantization.sign_round.data_parallel import _debug_flat_leak_probe

        _debug_flat_leak_probe(f"{getattr(block_ctx, 'block_name', 'block')}-start")
        _tune_wall_t0 = time.perf_counter() if tune_prof is not None else None
        if tune_prof is not None and tune_prof.debug:
            _ai = active_inputs if isinstance(active_inputs, list) else []
            tune_prof.log_placement(
                device=device,
                loss_device=loss_device,
                cache_device=self.compress_context.cache_device,
                inputs=[(str(t.device), tuple(t.shape)) for t in _ai[:2]],
                fp_outputs=[(str(t.device), tuple(t.shape)) for t in fp_outputs[:2]],
                nsamples=nsamples,
            )

        # delayed-loss mode (DDP only): read loss(i).item() at iteration i+1 so
        # the drain overlaps the next iteration's already-enqueued GPU work
        # instead of stalling the pipeline between iterations; best-params
        # selection semantics are unchanged (same (loss, pre-step params)
        # pairs, same strict-less rule -- resolved one iteration later)
        _pacing = bool(_envs.AR_TUNE_REPLICA_PACING)
        _fence_free = bool(_envs.AR_TUNE_PREPARE_FENCE_FREE)
        _proactive = bool(_envs.AR_TUNE_PROACTIVE_PREPARE)
        _serial_delayed = bool(_envs.AR_TUNE_SERIAL_DELAYED_LOSS)
        _serial_sched_on = bool(_envs.AR_TUNE_SERIAL_DEVICE_SCHEDULE)
        _proactive_schedule = None
        _graphs_prepared_iter = -2
        _async = replica_group is not None and _envs.AR_TUNE_ASYNC_LOSS and self.dynamic_max_gap <= 0
        _sign_ok = self.momentum is None or float(self.momentum) == 0.0
        _exch_session = None
        if (
            replica_group is not None
            and _envs.AR_TUNE_EXCH_OVERLAP
            and _sign_ok
            and _envs.AR_TUNE_DDP_SIGN_EXCHANGE
            and replica_group.world >= 2
            and (replica_group.world & (replica_group.world - 1)) == 0
            # eager backward only: post-accumulate-grad hooks (the completion
            # signal) do not fire on CUDA-graph replay -- with graphs active
            # the buckets would never complete and the join would stall
            and not _graphs_active
            # int8 transport scales per SEGMENT amax -- bucketing changes the
            # segments, so restrict to transports where bucketing is exact
            and replica_group.grad_transport in ("fp32", "bf16")
            # the one-shot arm is a different algorithm; do not mix per phase
            and not use_one_shot(replica_group.world, replica_group.grad_transport)
            # frozen sentinels cannot register hooks; the monolithic path
            # tolerates them, the bucketed one does not
            and all(p.requires_grad for p in params_per_replica[0])
        ):
            from auto_round.algorithms.quantization.sign_round.data_parallel import BucketExchangeSession

            _exch_session = BucketExchangeSession(params_per_replica, replica_group.grad_transport)
            _exch_session.arm()
        if replica_group is not None and _envs.AR_TUNE_ASYNC_LOSS and 0 < self.dynamic_max_gap:
            logger.warning(
                "[tune-ddp] dynamic_max_gap is set: keeping the immediate loss read "
                "(the early-stop decision needs the current loss in-loop)"
            )
        _atracker = None
        _snap_dev = None
        if _async:
            from auto_round.algorithms.quantization.sign_round.data_parallel import _AsyncBestTracker

            _loss_dev = (
                torch.device(loss_device)
                if loss_device is not None
                else (fp_outputs[0].device if fp_outputs else torch.device("cpu"))
            )
            _atracker = _AsyncBestTracker(self.iters, _loss_dev, world=replica_group.world)
            # same parking policy as the legacy best-params snapshot below
            _snap_dev = (
                _home
                if getattr(self.compress_context.cache_device, "type", "") == "cuda" and _home.type == "cuda"
                else self.compress_context.cache_device
            )

        _graphs_t0 = time.perf_counter()
        _graphs_capture_t = None
        _graphs_capture_wall = 0.0
        _graphs_capture_iter = -1
        _serial_rows = None
        _sched_host = None
        _serial_idx_dev = None
        if _serial_sched_on and replica_group is None:
            # T8a: consume the whole shuffle schedule upfront (identical RNG
            # stream to the lazy per-iteration next_batch sequence -- tested)
            # and stage each iteration's indices through one pinned host
            # tensor + one device buffer: no host RNG state machine and no
            # pageable H2D on the hot path; graph-capture-ready (static
            # address after allocator warmup)
            # capture the global-RNG state after each draw so an early break
            # (dynamic_max_gap) can restore the exact state the LAZY path
            # would have left -- subsequent blocks' shuffles stay bit-identical
            _serial_rng_states = [random.getstate()]
            _serial_rows = []
            for _ in range(self.iters):
                _serial_rows.append(index_sampler.next_batch())
                _serial_rng_states.append(random.getstate())
            try:
                _first = active_inputs if isinstance(active_inputs, list) else next(iter(active_inputs.values()))
                if isinstance(_first, list):
                    _first = _first[0]
                if torch.cuda.is_available() and torch.is_tensor(_first) and _first.device.type == "cuda":
                    _sched_host = torch.tensor(_serial_rows, dtype=torch.long).pin_memory()
                    _serial_idx_dev = torch.empty(global_batch_size, dtype=torch.long, device=_first.device)
            except (RuntimeError, OSError):
                # pinning / staging is best-effort: host rows alone still
                # remove the per-iteration RNG work (bit-identical draws)
                logger.info("[tune-serial] schedule staging fell back to host rows (pin/alloc failed)")
                _sched_host = None
                _serial_idx_dev = None

        _serial_step = None
        _serial_warmups_left = 0
        _serial_graphs_defer_logged = False
        _lfq_here = (
            block_ctx.block_index == block_ctx.block_cnt - 1
            and self.enable_lfq
            and input_ids is not None
            and self._is_text_decoder_block(block_ctx.block_name)
        )
        _serial_streamer = getattr(getattr(block, "_stream_prefetch_source", None), "streamer", None)
        # snapshots collect from the module the loop actually tunes (the
        # persistent HOME staging replica on a template-cache hit; the real
        # block otherwise)
        _tune_src = replica_group.tune_home if replica_group is not None else block
        from auto_round.algorithms.quantization.sign_round.data_parallel import (
            _sign_step_foreach,
            serial_graphs_engage,
            sign_step_foreach_ok,
        )

        if bool(_envs.AR_TUNE_CUDA_GRAPHS) and serial_graphs_engage(
            replica_group,
            device_is_cuda=str(device).startswith("cuda"),
            inputs_are_list=isinstance(active_inputs, list),
            is_diffusion=bool(block_fwd.is_diffusion),
            has_valid_token_mask=valid_token_mask is not None,
            is_lfq_block=_lfq_here,
            loss_on_tune_device=(loss_device is None or torch.device(loss_device) == torch.device(device)),
            fence_free=_fence_free,
            accumulate_one=self.gradient_accumulate_steps == 1,
        ):
            from auto_round.algorithms.quantization.sign_round.data_parallel import make_serial_graphed_step

            _serial_step = make_serial_graphed_step(
                block,
                block_fwd,
                active_inputs,
                input_others,
                fp_outputs,
                lambda pred, ref: self._get_loss(pred, ref, None, mse_loss, device, None),
                torch.device(device),
            )
            logger.info(
                "[tune-serial] cuda graphs enabled: capture after %d warmup iteration(s)", _serial_step.warmup_iters
            )

        _serial_step_fn = None
        _serial_pre_by_id = None
        if _serial_step is not None:
            # T8c escalation: capture the OPTIMIZER STEP inside the region too
            # (whole-iteration graph). Requires the pure-sign configuration
            # (momentum/wd zero -- the captured update mirrors exactly that
            # SignSGD path) and no dynamic early-stop (conservative: the
            # immediate-mode loss read stays on the T8b path).
            _serial_step_ok = (
                scaler is None
                and (self.momentum is None or float(self.momentum) == 0.0)
                and self.dynamic_max_gap <= 0
                and all(float(g.get("weight_decay", 0.0) or 0.0) == 0.0 for g in optimizer.param_groups)
            )
            if _serial_step_ok:
                try:
                    from auto_round.algorithms.quantization.sign_round.data_parallel import (
                        make_serial_step_fn,
                        serial_step_vram_ok,
                    )

                    _needed = sum(
                        p.numel() * p.element_size()
                        for g in optimizer.param_groups
                        for p in g["params"]
                        if p.requires_grad
                    )
                    _free_fn = (lambda: torch.cuda.mem_get_info(device)) if str(device).startswith("cuda") else None
                    if serial_step_vram_ok(_needed, _free_fn):
                        _serial_step_fn, _serial_pre_by_id = make_serial_step_fn(optimizer, torch.device(device))
                        _serial_step = make_serial_graphed_step(
                            block,
                            block_fwd,
                            active_inputs,
                            input_others,
                            fp_outputs,
                            lambda pred, ref: self._get_loss(pred, ref, None, mse_loss, device, None),
                            torch.device(device),
                            step_fn=_serial_step_fn,
                        )
                        logger.info(
                            "[tune-serial] whole-iteration capture: optimizer step inside the graph "
                            "(staging %.2f GiB)",
                            _needed / 2**30,
                        )
                    else:
                        logger.info(
                            "[tune-serial] whole-iteration capture skipped: staging needs %.2f GiB free",
                            _needed / 2**30,
                        )
                except (RuntimeError, ValueError) as _se:
                    logger.info(
                        "[tune-serial] whole-iteration capture unavailable (%s); eager step between replays", _se
                    )
                    _serial_step_fn, _serial_pre_by_id = None, None
                    # R1-4: drop partially-injected device-lr keys (a later
                    # state_dict() would carry them as opaque group options)
                    for _g in optimizer.param_groups:
                        _g.pop("_ar_neg_lr_dev", None)
            # R1-3: warmups must reflect whichever step object is live
            _serial_warmups_left = _serial_step.warmup_iters if _serial_step is not None else 0

        if _proactive and _pacing and _graphs_active and _dp_samplers is not None and replica_group is not None:
            # consume the whole shuffle schedule upfront (same RNG stream as
            # the lazy per-iteration next_batch sequence -- deterministic)
            _proactive_schedule = [[s.next_batch() for s in _dp_samplers] for _ in range(self.iters)]
        _tprof_n = int(_envs.AR_TUNE_TORCH_PROF or 0)
        _tprof_start, _tprof_end = 5, 5 + _tprof_n
        _tprof_ctx = None
        _tprof_block_logged = False
        for i in range(self.iters):
            if _tprof_n > 0 and i == _tprof_start:
                from torch.profiler import ProfilerActivity, profile

                _tprof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                _tprof_ctx.__enter__()
                if not _tprof_block_logged:
                    _tprof_block_logged = True
                    logger.info(
                        "[tune-ddp] torch profiler capturing iterations %d-%d (chrome trace on exit)",
                        i,
                        min(self.iters, _tprof_end) - 1,
                    )
            elif _tprof_ctx is not None and (i == _tprof_end or i == self.iters - 1):
                _tprof_ctx.__exit__(None, None, None)
                try:
                    _tprof_ctx.export_chrome_trace("tune_torch_prof.json")
                    logger.info("[tune-ddp] torch profiler trace saved to tune_torch_prof.json")
                except Exception as _pe:  # noqa: BLE001 - profiling must never break tuning
                    logger.warning("[tune-ddp] torch profiler trace export failed: %s", _pe)
                _tprof_ctx = None
            if self.enable_alg_ext and self.scheme.data_type.endswith("dq"):
                for _src in {_tune_src, block}:
                    for n, m in _src.named_modules():
                        m.cur_iter = i
            total_loss = 0
            _loss_acc = None  # per-iteration reset (device fp64 accumulator)
            with tune_stage(tune_prof, "sampler"):
                if _proactive_schedule is not None:
                    _shards = _proactive_schedule[i]
                    global_indices = [j for sh in _shards for j in sh]
                elif _dp_samplers is not None:
                    _shards = [s.next_batch() for s in _dp_samplers]
                    global_indices = [j for sh in _shards for j in sh]
                elif _serial_rows is not None:
                    global_indices = _serial_rows[i]
                    if _serial_idx_dev is not None and _serial_step is None:
                        # one async pinned H2D per iteration (same stream as
                        # the consumers -- FIFO-ordered, no event needed);
                        # the graphed step gathers from host rows instead
                        _serial_idx_dev.copy_(_sched_host[i], non_blocking=True)
                else:
                    global_indices = index_sampler.next_batch()
                if valid_token_mask:
                    num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)

            if replica_group is not None:
                _world = replica_group.world
                if _dp_samplers is None:
                    _shard = len(global_indices) // _world
                    _shards = [global_indices[r * _shard : (r + 1) * _shard] for r in range(_world)]
                _losses = [None] * _world

                def _dp_replica_step(r, shard):
                    rep = replica_group.replicas[r]
                    dev_r = next(rep.parameters()).device
                    with torch.cuda.device(dev_r):
                        expect_pool_local([fp_outputs[j] for j in shard], dev_r, "ddp-ref")
                        ref_r = torch.cat([fp_outputs[j].to(dev_r) for j in shard], dim=0)
                        # always device-local: the runner's default cache_device
                        # is the PRIMARY GPU -- parking a mirror's output there
                        # would both mismatch the loss and cost a cross-device
                        # copy every iteration
                        pred_r = block_fwd.forward(rep, active_inputs, input_others, shard, dev_r)
                        loss_r = self._get_loss(pred_r, ref_r, shard, mse_loss, dev_r, None)
                        _losses[r] = loss_r.detach()
                        loss_r.backward()

                # overlapped exchange only on steady iterations: hooks must not
                # record CUDA events inside a capture (event record during
                # stream capture is illegal)
                _use_exch_overlap = _exch_session is not None and (
                    _graphed_steps is None or all(getattr(_g, "_graph", None) is not None for _g in _graphed_steps)
                )
                if _use_exch_overlap:
                    _exch_session.begin_iter()
                    _exch_session.start()
                if _graphs_capture_pending and not _bg_busy and not _prefetch_busy:
                    # barrier point: pool workers are parked and no CUDA work
                    # is in flight process-wide -- torch.cuda.graph forbids
                    # concurrent CUDA during capture, so all captures run
                    # sequentially here on the main thread; the immediate
                    # replay inside capture_now executes this iteration's work
                    _cap_t0 = time.perf_counter()
                    with tune_stage(tune_prof, "fwd"):
                        for r in range(_world):
                            _graphed_steps[r].prepare(_shards[r])
                        try:
                            for r in range(_world):
                                if getattr(_graphed_steps[r], "_graph", None) is None:
                                    _losses[r] = _graphed_steps[r].capture_now()
                                else:
                                    # template-cached mirror: keep replaying
                                    _losses[r] = _graphed_steps[r].replay_only()
                        except BaseException:
                            # halt semantics: open the gate before the run
                            # aborts so held bg workers never outlive the
                            # failure (the owner's finally is the backstop)
                            if _graphs_gate is not None:
                                _graphs_gate.set()
                            raise
                        _graphs_capture_pending = False
                        _graphs_capture_t = time.perf_counter()
                        _graphs_capture_wall = _graphs_capture_t - _cap_t0
                        _graphs_capture_iter = i + 1
                        if _graphs_gate is not None:
                            _graphs_gate.set()  # release held bg pack/ready work
                        _cached = sum(1 for _g in _graphed_steps if getattr(_g, "_graph", None) is not None) - (
                            1 if getattr(_graphed_steps[0], "_graph", None) is not None else 0
                        )
                        logger.info(
                            "[tune-ddp] cuda graphs captured %d replica step(s)%s",
                            _world - max(0, _cached),
                            f" (+{max(0, _cached)} template-cached replays)" if _cached > 0 else "",
                        )
                elif _graphed_steps is not None:
                    # warm-up AND deferred iterations both dispatch through
                    # the pool: the uncaptured step path runs prepare+compute
                    # eagerly, so a deferral never skips an iteration
                    if _graphs_capture_pending and not _graphs_defer_logged:
                        _graphs_defer_logged = True
                        logger.warning(
                            "[tune-ddp] cuda graph capture deferred: %d background thread(s), "
                            "prefetch_pending=%s -- capturing at the first quiet iteration",
                            len(_bg_busy),
                            _prefetch_busy,
                        )
                    _all_captured = all(getattr(_g, "_graph", None) is not None for _g in _graphed_steps)
                    with tune_stage(tune_prof, "fwd"):
                        if _pacing and _all_captured:
                            # two rounds: every prepare completes (pool latch)
                            # before any replay launches -- replays start in
                            # lockstep instead of trailing prepare jitter;
                            # proactively, iteration i+1's prepare is enqueued
                            # right after this replay round (stream FIFO orders
                            # the static-buffer write-after-read)
                            _was_prepared = _graphs_prepared_iter == i
                            _next_sh = (
                                _proactive_schedule[i + 1]
                                if _proactive_schedule is not None and i + 1 < self.iters
                                else None
                            )
                            run_proactive_paced_steps(
                                replica_group,
                                _graphed_steps,
                                _shards,
                                _next_sh,
                                _losses,
                                _graphed_worker_ms,
                                prepared=_was_prepared,
                            )
                            _graphs_prepared_iter = (i + 1) if _next_sh is not None else i
                        else:
                            replica_group.run_threaded([lambda r=r: _dp_run(r) for r in range(_world)])
                    if _graphs_warmups_left > 0:
                        _graphs_warmups_left -= 1
                        if _graphs_warmups_left == 0:
                            _graphs_capture_pending = True
                else:
                    with tune_stage(tune_prof, "fwd"):
                        replica_group.run_threaded([lambda r=r: _dp_replica_step(r, _shards[r]) for r in range(_world)])
                if _use_exch_overlap:
                    # bucket exchange ran concurrently with the backward tail;
                    # every bucket has shipped -- drain the thread and check
                    with tune_stage(tune_prof, "exchange"):
                        _exch_session.join_and_check()
                else:
                    # sync_grads stages itself (bufprep / exchange / writeback).
                    # sign_exchange: the update consumes only sign(mean-grad)
                    # and weight_decay is 0, so exchanging int8 signs after
                    # the reduce-scatter is bitwise-identical across ranks
                    # and at least as faithful as a lossy transport
                    # all-gather -- but a momentum buffer would mix
                    # magnitudes back in, so gate on it.
                    replica_group.sync_grads(
                        params_per_replica,
                        prof=tune_prof,
                        sign_exchange=self.momentum is None or float(self.momentum) == 0.0,
                    )
                # report the global-batch mean: mean of equal-size shard means.
                # .item() blocks the host until each replica's fwd+loss+bwd has
                # drained; with P2P transport the allreduce enqueue above is fully
                # asynchronous, so that drain-wait surfaces HERE, not in the
                # allreduce stage -- keep it in its own bucket so "other" stays
                # honest.
                if _async:
                    # poll: resolve READY flags only -- the host never waits;
                    # per-iter host loss values are gone entirely (values live
                    # on device, reads happen once at the loop-end drain)
                    with tune_stage(tune_prof, "loss_sync"):
                        _atracker.poll()
                    with tune_stage(tune_prof, "snapshot"):
                        _atracker.stage(
                            None if self.not_use_best_mse else collect_best_params(_tune_src, _snap_dev), _losses, i
                        )
                else:
                    with tune_stage(tune_prof, "loss_sync"):
                        if _tpl_entry is not None and i in (0, 5) and _losses and all(l is not None for l in _losses):
                            # template-cache diagnostic: per-replica losses -- a mirror
                            # diverging from the home here pins stale replay state
                            logger.info(
                                "[tune-ddp] template-cache probe blk iter=%d replica losses: %s",
                                i,
                                ["%.3e" % float(l) for l in _losses],
                            )
                        total_loss = sum(l.item() for l in _losses if l is not None) / _world
            else:
                if _serial_step is not None:
                    if _serial_step_fn is not None:
                        # whole-iteration mode: refresh the DEVICE lr the
                        # captured step reads (host float cast; the schedule
                        # advances below -- ordering matches _step's tail)
                        for _g in optimizer.param_groups:
                            if "_ar_neg_lr_dev" in _g:
                                _g["_ar_neg_lr_dev"].fill_(-float(_g["lr"]))
                    with tune_stage(tune_prof, "fwd"):
                        if _serial_warmups_left > 0:
                            # warm-up: prepare + EAGER compute on the capture
                            # stream (AccumulateGrad birth-stream correctness)
                            _loss = _serial_step(global_indices)
                            _serial_warmups_left -= 1
                        elif (
                            getattr(_serial_step, "_graph", None) is None
                            and not any(t.is_alive() for t in _bg_cuda_threads)
                            # the ckpt-prefetch thread issues CUDA (staging
                            # copies / VRAM polling) regardless of DDP --
                            # torch.cuda.graph needs the process quiet
                            and not (_serial_streamer is not None and _serial_streamer.prefetch_pending())
                        ):
                            # capture barrier: single-threaded serial loop, but
                            # bg pack/transform threads from a previous block
                            # may still issue CUDA -- torch.cuda.graph needs the
                            # process quiet; capture records AND replays this
                            # iteration's work
                            _serial_step.prepare_only(global_indices)
                            _loss = _serial_step.capture_now()
                        elif getattr(_serial_step, "_graph", None) is None and not _serial_graphs_defer_logged:
                            _serial_graphs_defer_logged = True
                            logger.info(
                                "[tune-serial] cuda graph capture deferred: background CUDA threads still active"
                            )
                        else:
                            # steady state: prepare refresh + single replay
                            _loss = _serial_step(global_indices)
                    num_elm = 1 if num_elm <= 0 else num_elm
                    with tune_stage(tune_prof, "sync"):
                        if _serial_delayed:
                            _loss_acc = accumulate_serial_loss(_loss_acc, _loss, num_elm)
                        else:
                            total_loss += _loss.item() / num_elm
                    if mid_iter_mem_check:
                        # between iterations only: the legacy in-loop checks
                        # sit between loss and backward -- INSIDE the captured
                        # region here
                        clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)
                else:
                    for batch_start in range(0, len(global_indices), batch_size):
                        indices = global_indices[batch_start : batch_start + batch_size]
                        with tune_stage(tune_prof, "ref_h2d"):
                            _loss_dev = torch.device(loss_device) if loss_device is not None else fp_outputs[0].device
                            expect_pool_local([fp_outputs[i] for i in indices], _loss_dev, "serial-ref")
                            ref_output = torch.cat([fp_outputs[i].to(_loss_dev) for i in indices], dim=0)
                        if _serial_idx_dev is not None:
                            _idx_arg = _serial_idx_dev[batch_start : batch_start + batch_size]
                            _host_arg = indices
                        else:
                            _idx_arg, _host_arg = indices, None
                        with tune_stage(tune_prof, "fwd"):
                            pred_output = block_fwd.forward(
                                block, active_inputs, input_others, _idx_arg, _fwd_cache_device, host_indices=_host_arg
                            )
                            if loss_device is not None:
                                pred_output = pred_output.to(loss_device)
                        if (
                            block_ctx.block_index == block_ctx.block_cnt - 1
                            and self.enable_lfq
                            and input_ids is not None
                            and self._is_text_decoder_block(block_ctx.block_name)
                        ):
                            with tune_stage(tune_prof, "loss"):
                                loss = self.lfq_loss(pred_output, torch.cat([input_ids[i] for i in indices], dim=0))
                        else:
                            with tune_stage(tune_prof, "loss"):
                                loss = self._get_loss(
                                    pred_output, ref_output, indices, mse_loss, device, valid_token_mask
                                )
                        num_elm = 1 if num_elm <= 0 else num_elm
                        with tune_stage(tune_prof, "sync"):
                            if _serial_delayed:
                                # accumulate on device: no host fence between loss
                                # and backward -- the single read happens after the
                                # batch loop (all backwards already enqueued)
                                _loss_acc = accumulate_serial_loss(_loss_acc, loss, num_elm)
                            else:
                                total_loss += loss.item() / num_elm

                        if mid_iter_mem_check:
                            # clear memory to avoid OOM due to memory fragmentation
                            clear_memory_if_reached_threshold(threshold=0.5, device_list=device_manager.device_list)

                        with tune_stage(tune_prof, "bwd"):
                            self._scale_loss_and_backward(scaler, loss)

                        if mid_iter_mem_check:
                            # clear memory to avoid OOM due to memory fragmentation
                            clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)

            if _serial_delayed and _loss_acc is not None:
                total_loss = float(_loss_acc)
            if not _async:
                if i == 0:
                    init_loss = total_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        # park the snapshot where it was produced when the pipeline
                        # already caches on GPU: cross-device D2D snapshots cost
                        # ~0.1s/iter on non-primary homes
                        _snap_dev = (
                            _home
                            if getattr(self.compress_context.cache_device, "type", "") == "cuda"
                            and _home.type == "cuda"
                            else self.compress_context.cache_device
                        )
                        with tune_stage(tune_prof, "snapshot"):
                            if _serial_step_fn is not None:
                                from auto_round.algorithms.quantization.sign_round.data_parallel import (
                                    collect_best_params_pre,
                                )

                                best_params = collect_best_params_pre(block, _snap_dev, _serial_pre_by_id)
                            else:
                                best_params = collect_best_params(_tune_src, _snap_dev)
                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    if _serial_step_fn is not None:
                        from auto_round.algorithms.quantization.sign_round.data_parallel import (
                            collect_best_params_pre,
                        )

                        best_params = collect_best_params_pre(
                            block, self.compress_context.cache_device, _serial_pre_by_id
                        )
                    else:
                        best_params = collect_best_params(_tune_src, self.compress_context.cache_device)

                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter:
                        if _serial_rows is not None:
                            # the precompute consumed ALL iters draws; the lazy
                            # path would have stopped after iteration i's draw --
                            # restore that exact global-RNG state so the NEXT
                            # block's IndexSampler shuffle stays bit-identical
                            random.setstate(_serial_rng_states[i + 1])
                        break
            elif _async and self.not_use_best_mse and i == self.iters - 1:
                # loss-delay-only mode keeps the single last-iter collection
                best_params = collect_best_params(_tune_src, self.compress_context.cache_device)
            if replica_group is not None:
                with tune_stage(tune_prof, "step"):
                    self._step(scaler, optimizer, lr_schedule, keep_grads=_graphs_active)

                    def _mirror_step(opt, sched):
                        if sign_step_foreach_ok(opt):
                            _sign_step_foreach(opt, free_grads=not _graphs_active)
                        else:
                            opt.step()
                            opt.zero_grad(set_to_none=not _graphs_active)
                        sched.step()

                    # mirror optimizers exclude the HOME replica (its step ran
                    # serially above) -- pad with a no-op home slot so the
                    # fns width matches the replica worker pool (a narrower
                    # list would silently fall back to spawn-per-call)
                    replica_group.run_threaded(
                        [lambda: None]
                        + [
                            lambda oo=opt, ss=sch: _mirror_step(oo, ss)
                            for opt, sch in zip(mirror_optimizers, mirror_schedules)
                        ]
                    )
            else:
                sync_gradients()
                with tune_stage(tune_prof, "step"):
                    if _serial_step_fn is not None:
                        # the optimizer's host-side step() never runs in this
                        # mode, so torch's 'skipping the first value' hint is
                        # a false positive -- the lr sequence is identical
                        # (verified); suppress just that warning
                        import warnings as _warnings

                        with _warnings.catch_warnings():
                            _warnings.filterwarnings("ignore", message=".*seems to be deprecated.*")
                            _warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`.*")
                            lr_schedule.step()  # step ran inside the (re)play
                    else:
                        self._step(scaler, optimizer, lr_schedule, keep_grads=_serial_step is not None)

        try:
            if _graphed_steps is not None and len(_graphed_worker_ms) and all(len(w) for w in _graphed_worker_ms):
                _means = [sum(w) / len(w) for w in _graphed_worker_ms]
                logger.info(
                    "[tune-ddp] replica worker ms/iter (unpaced: prepare+replay; paced: replay round): %s (spread %.1f)",
                    ["%.1f" % m for m in _means],
                    max(_means) - min(_means),
                )
        except NameError:
            pass
        if _exch_session is not None:
            _exch_session.close()
        if _graphs_capture_t is not None:
            _pre = _graphs_capture_iter
            _post = self.iters - _graphs_capture_iter
            _pre_ms = (_graphs_capture_t - _graphs_t0) * 1000 / max(_pre, 1)
            _post_ms = (time.perf_counter() - _graphs_capture_t) * 1000 / max(_post, 1)
            logger.info(
                "[tune-ddp] cuda graphs timing: eager %d iters %.1f ms/iter -> replay %d iters %.1f ms/iter "
                "(capture %.2fs)",
                _pre,
                _pre_ms,
                _post,
                _post_ms,
                _graphs_capture_wall,
            )
        if _graphs_gate is not None:
            _graphs_gate.set()  # never hold bg work past the tune loop
        if _graphs_active and _graphed_steps is not None and any(st._graph is None for st in _graphed_steps):
            logger.warning(
                "[tune-ddp] cuda graphs were never captured for this block "
                "(background pack/transform threads stayed busy through the tune window); it ran eager"
            )
        if replica_group is not None:
            if _tpl_home is not None and _tpl_home is not block:
                # copy the TUNED end state back: downstream readers (unwrapper/
                # pack/refit) walk the real block -- grids, frozen-pin layout,
                # flags and all (a plain value sync leaves unanchored grids)
                from auto_round.algorithms.quantization.sign_round.data_parallel import sync_tuned_state_back

                sync_tuned_state_back(_tpl_home, block)
            try:
                from auto_round.algorithms.quantization.sign_round.data_parallel import (
                    _GRAPH_TEMPLATE_CACHE,
                    template_cache_size,
                )

                if (
                    template_cache_size() > 0
                    and _tpl_key is not None
                    and _graphed_steps is not None
                    and all(getattr(_st, "_graph", None) is not None for _st in _graphed_steps)
                ):
                    _steps = list(_graphed_steps)
                    # neutralize the closures: they still close over THIS
                    # block's pools -- a cache hit rebinds before any use, and
                    # replay_only never calls them
                    for _stp in _steps:
                        _stp.prepare = lambda *a, **k: None
                        _stp.compute = lambda: None
                    _GRAPH_TEMPLATE_CACHE[_tpl_key] = {
                        "replicas": list(replica_group.replicas),
                        "steps": _steps,
                        # address fingerprint: every tensor the captured graphs read
                        # must still live at these addresses on a hit -- drift means
                        # something REBOUND instead of refreshing in place
                        "ptrs": [_addr_fingerprint(rep) for rep in replica_group.replicas],
                    }
                    while len(_GRAPH_TEMPLATE_CACHE) > template_cache_size():
                        _GRAPH_TEMPLATE_CACHE.popitem(last=False)
            except Exception as _cache_err:  # noqa: BLE001 - caching is best-effort
                logger.debug("[tune-ddp] graph template cache insert skipped (%r)", _cache_err)
            replica_group.teardown()
        if _async:
            # the one legal fence in the whole loop: bounded wait, then the
            # bulk value reads (init/best/last) and the final promotions
            with tune_stage(tune_prof, "loss_sync"):
                _atracker.drain()
            if _atracker.init_loss is not None:
                init_loss = _atracker.init_loss / _world
            total_loss = float(_atracker.all_losses[self.iters - 1]) / _world if self.iters > 0 else init_loss
            if _atracker.best_loss is not None:
                best_loss = _atracker.best_loss / _world  # tracker stores the raw sum
                last_best_iter = _atracker.best_iter
            if not self.not_use_best_mse:
                best_params = _atracker.best_params or {}
        last_loss = total_loss
        if tune_prof is not None:
            tune_prof.log_summary(
                block_name=getattr(block_ctx, "block_name", ""),
                iters_done=(i + 1) if self.iters > 0 else 0,
                wall=time.perf_counter() - _tune_wall_t0,
            )
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0 and init_loss is not None:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
            )
        elif self.iters > 0:  # no iter-0 loss recorded (e.g. loss read skipped)
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, best iter {best_iter}: {last_loss:.6f}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        with _bp_stage(_prof_bp, "clear_mem"):
            self.compress_context.clear_memory()  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"Unquantized layers: {unquantized_layer_names}")
        with torch.no_grad(), _bp_stage(_prof_bp, "refit"):
            unwrapper_block(block, best_params)  # includes AR_POST_SCALE_REFIT

        if self.config.is_act_nv_fp:
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        from auto_round.algorithms.quantization.sign_round.data_parallel import _debug_flat_leak_probe

        _debug_flat_leak_probe(f"{getattr(block_ctx, 'block_name', 'block')}-end")
        logger.infoclean(dump_info)
        return best_params

    def quantize_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_inputs: Optional[list[torch.Tensor]] = None,
        q_inputs: Optional[list[torch.Tensor]] = None,
        disable_opt_rtn: Optional[bool] = None,
        input_ids: Optional[list[torch.Tensor]] = None,
    ):
        """Quantize a single layer that lives outside a transformer block.

        When ``fp_inputs`` is provided the layer is tuned with the sign-gradient
        descent optimizer (same loss loop as block-level quantization).  When
        ``fp_inputs`` is ``None`` the method falls back to zero-shot RTN.

        Args:
            layer: The layer module to quantize.  Must have a ``global_name``
                attribute for model re-insertion and logging.
            fp_inputs: Per-sample FP activations fed into this layer, used as
                calibration inputs during optimization. ``None`` triggers RTN
                fallback.
            q_inputs: Per-sample quantized activations from the previous stage,
                used instead of ``fp_inputs`` during the forward pass when
                cascaded quantized-input is enabled. ``None`` means use
                ``fp_inputs`` for both reference and tuning forward.
            disable_opt_rtn: Override optimized-RTN; ``None`` defers to quantizer config.
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample); used to derive the valid-token loss mask via
                ``_compute_valid_token_mask``. ``None`` disables loss masking.
        """

        layer_name = layer.global_name
        if fp_inputs is None:
            logger.info(f"using rtn to quantize {layer_name}")
            self._quantize_layer_via_rtn(
                layer,
                disable_opt_rtn=(
                    disable_opt_rtn if disable_opt_rtn is not None else getattr(self.config, "disable_opt_rtn", True)
                ),
            )
            return

        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Reuse the cached mask if already computed by a previous block.
        valid_token_mask = None
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        logger.info(f"quantizing layer {layer_name}")
        # Layer is already on the correct device (placed by the caller / AlgorithmComposer).
        device = layer.weight.device if hasattr(layer, "weight") else device_manager.device
        from auto_round import envs as _layer_envs

        _serial_delayed = bool(_layer_envs.AR_TUNE_SERIAL_DELAYED_LOSS)
        for i in range(len(fp_inputs)):
            fp_inputs[i] = fp_inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        ).to(device)
        round_params = []
        minmax_params = []
        for key in wrapper_linear.params.keys():
            if "min" in key or "max" in key:
                minmax_params.append(wrapper_linear.params[key])
            else:
                round_params.append(wrapper_linear.value)
        if len(round_params) + len(minmax_params) <= 0:
            dump_info = f"quantized {layer_name}"
            logger.info(dump_info)
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name, {})
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        # Use a lr derived from this layer's own bit-width so mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) tune each layer correctly.
        layer_bits = getattr(layer, "bits", None)
        layer_lr = self._config.compute_lr(layer_bits)
        if layer_lr is not None:
            lr = torch.tensor(layer_lr)
        self._maybe_log_low_bit_lr(layer_bits)
        layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
        if layer_minmax_lr is not None:
            minmax_lr = torch.tensor(layer_minmax_lr)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(fp_inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        best_params = None
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None

        gradient_accumulate_steps = (
            self.calibration_context.batch_size * self.gradient_accumulate_steps
        )  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # Compute num_elm once before the loop.
        if gradient_accumulate_steps != 1:
            whole_indices = list(range(global_batch_size))
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, whole_indices)
            elif q_inputs is not None:
                num_elm = self._count_layer_input_elements(q_inputs, whole_indices)
            else:
                num_elm = self._count_layer_input_elements(fp_inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()

            for batch_start in range(0, len(global_indices), batch_size):
                indices = global_indices[batch_start : batch_start + batch_size]
                if q_inputs is not None:
                    current_input = [q_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = [fp_inputs[i] for i in indices]
                    org_input = torch.cat(org_input, dim=0).to(device)
                else:
                    current_input = [fp_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = current_input
                with torch.no_grad():
                    current_output = layer(org_input)
                autocast_ctx = (
                    nullcontext()
                    if not self.model_context.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
                )
                if valid_token_mask:
                    tmp_valid_mask = [valid_token_mask[i] for i in indices]
                    tmp_valid_mask = torch.cat(tmp_valid_mask, dim=0).to(device)
                    tmp_valid_mask.unsqueeze_(-1)

                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            (output_q * tmp_valid_mask).to(torch.float32),
                            (current_output * tmp_valid_mask).to(torch.float32),
                        )

                else:
                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            output_q.to(torch.float32),
                            current_output.to(torch.float32),  # mul 1.0 will copy the output
                        )

                num_elm = 1 if num_elm <= 0 else num_elm
                if _serial_delayed:
                    _term = loss.detach().double() / num_elm
                else:
                    total_loss += loss.item() / num_elm

                self._scale_loss_and_backward(scaler, loss)
                if _serial_delayed:
                    # cumulative across iterations, exactly like the legacy sum
                    total_loss += float(_term)
            if i == 0:
                init_loss = total_loss

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.6f} -> iter {best_iter}: {last_loss:.6f}"
        logger.info(dump_info)

    def finalize_run(self) -> None:
        """Clear per-run caches (``_cached_valid_token_mask``, LFQ components)."""
        for attr in ("_cached_valid_token_mask", "_lm_head", "_post_block_modules"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_optimizer(self, optimizer: Any):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        if optimizer is not None:
            logger.warning_once(
                "The optimizer setting in config will be ignored in AutoRound, using SignSGD as default."
            )
        return SignSGD

    def _count_layer_input_elements(self, input_ids, indices: list) -> int:
        return sum(input_ids[i].numel() for i in indices)

    def _get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return None

    def _scale_loss_and_backward(self, scaler: Any, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        scale_loss.backward()
        if is_hpex_available():
            htcore.mark_step()
        return scale_loss

    def _step(self, scaler: Any, optimizer: Any, lr_schedule: Any, keep_grads: bool = False):
        """Performs a step in the optimization process.

        Pure-sign SignSGD (no scaler, HPU apart, momentum/weight-decay/maximize
        off everywhere) takes the bit-identical foreach fast path; the grad
        release matches ``keep_grads`` (parked for graph replays, freed
        otherwise). Any other configuration runs the original
        step/zero_grad/schedule sequence.

        Args:
        scaler: The scaler to be used.
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.
        keep_grads: grads keep their addresses (CUDA-graph replay loops).

        Returns:
        None
        """
        from auto_round.algorithms.quantization.sign_round.data_parallel import _sign_step_foreach, sign_step_foreach_ok

        if scaler is None and not is_hpex_available() and sign_step_foreach_ok(optimizer):
            # pure-sign SignSGD: the single-tensor loop is launch-bound
            # (~1195 groups x sign+mul+add+zero); the foreach form is
            # bit-identical and ~3 kernels per param group. The zero is
            # included (parked when grads must keep their addresses).
            _sign_step_foreach(optimizer, free_grads=not keep_grads)
            lr_schedule.step()
            return
        optimizer.step()
        # for hpu
        if is_hpex_available():
            htcore.mark_step()
        # keep_grads=True (CUDA-graph tune loop): grads must stay parked at
        # their captured addresses for the next replay
        optimizer.zero_grad(set_to_none=not keep_grads)
        lr_schedule.step()
