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
import os
import sys
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Union

import accelerate
import torch
from accelerate.big_modeling import dispatch_model
from tqdm import tqdm

from auto_round import envs
from auto_round.compressors.block_parallel import (
    has_block_results as _bp_has_block_results,
    load_chain_state as _bp_load_chain_state,
    save_block_results as _bp_save_block_results,
    save_chain_state as _bp_save_chain_state,
)
from auto_round.calibration import CalibrationContext
from auto_round.calibration.utils import (
    _update_inputs,
)
from auto_round.compressors.base import BaseOrchestrator
from auto_round.compressors.utils import (
    _get_quantized_layer_names_outside_blocks,
    immediate_pack,
    is_nv_fp,
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
from auto_round.wrapper import WrapperMultiblock

if TYPE_CHECKING:
    from auto_round.utils.resume import ResumeState


# TODO wenhuach align all the API args
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

        Subclasses (MLLM/Diffusion) override ``calib`` directly on the
        CompressionOrchestrator; the calibrator owns ``try_cache_inter_data_gpucpu`` /
        ``cache_inter_data`` orchestration plus the LLM ``calib`` body.
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

    def _preprocess_block_inputs(self, inputs, first_input_name="input_ids"):
        # Thin wrapper around auto_round.calibration.inputs.preprocess_block_inputs.
        from auto_round.calibration.inputs import preprocess_block_inputs

        return preprocess_block_inputs(
            inputs,
            model_context=self.model_context,
            compress_context=self.compress_context,
            first_input_name=first_input_name,
        )

    def _ff_one_block(self, model, block_name, entry, input_others, group_idx, next_idx, bp_results_dir):
        """One original-weights forward of ``block_name`` chaining ``entry``.

        Mirrors the serial reference chain exactly (same block_forward call,
        original weights, no calibration/tuning) and optionally checkpoints the
        resulting entry for the next block. The block is parked back on meta --
        it was never tuned, its weights are unchanged from the checkpoint, and
        keeping materialized copies would accumulate host memory.
        """
        m_ff = get_module(model, block_name)
        if self.compress_context.low_cpu_mem_usage or envs.AR_DISK_STREAM_MODEL:
            self._offloader.reload(model, block_name)
        materialize_model_(m_ff)
        m_ff = self.alg_composer.dispatch_block(m_ff, entry, input_others)
        with torch.no_grad():
            ff_output = self.alg_composer.block_forward(m_ff, entry, input_others)
        if bp_results_dir:
            _bp_save_chain_state(bp_results_dir, group_idx, next_idx, ff_output)
        if len(device_manager.device_list) > 1 and not self.model_context.is_diffusion:
            accelerate.hooks.remove_hook_from_submodules(m_ff)
        m_ff.to("meta")
        clear_memory(device_list=device_manager.device_list)
        return ff_output

    def _manifest_frontier(self, group_idx: int):
        """(prefix, entry) from the serial manifest, or (0, None).

        The manifest is the shared cross-mode resume contract: written by the
        serial loop and by the parallel sequencer alike, holding the chained
        entry at the contiguous done-prefix. Used by workers to obtain a replay
        starting point without any parallel-specific state.
        """
        import json as _json

        resume_dir = envs.AR_RESUME_DIR
        if not resume_dir:
            return 0, None
        gdir = os.path.join(resume_dir, f"group_{group_idx}")
        manifest = os.path.join(gdir, "resume_manifest.json")
        input_ids = os.path.join(gdir, "resume_input_ids.pt")
        if not (os.path.isfile(manifest) and os.path.isfile(input_ids)):
            return 0, None
        try:
            with open(manifest, encoding="utf-8") as f:
                data = _json.load(f)
            completed = data.get("completed_blocks", [])
            entry = torch.load(input_ids, map_location="cpu", weights_only=True)
            if entry is None:
                # prefix without a usable entry (e.g. absorption consumed a
                # missing handoff): replay from the group entry instead of
                # chaining from a wrong starting state
                return 0, None
            from auto_round.compressors.block_parallel import _to_device_recursive

            return len(completed), _to_device_recursive(entry, self.compress_context.cache_device)
        except Exception:  # noqa: BLE001
            return 0, None

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
        span: Optional[tuple] = None,
        group_idx: int = 0,
        bp_results_dir: Optional[str] = None,
        start_block_idx: Optional[int] = None,
        produce_only: Optional[tuple] = None,
    ):
        """Quantize and dequantize the weights of the specified blocks in the model.

        Args:
        model: The PyTorch model to be quantized.
        inputs: The input data for quantization.
        block_names: The names of the blocks to be quantized and dequantized.
        nblocks: The number of blocks to quantize and dequantize.
        device: The device for quantization and dequantization.
        span: Optional ``(start, end)`` restriction for block-parallel workers:
            blocks before ``start`` are fast-forwarded with original-weights
            no-grad forwards (to chain this span's entry input, mirroring the
            serial reference chain), blocks from ``end`` are skipped, and only
            blocks inside the span are tuned.
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
        if start_block_idx is not None:
            # queue mode: the caller restored this block's entry input from a
            # chain checkpoint -- start here directly, no fast-forward replay
            start_index = start_block_idx
        if produce_only is not None:
            # queue mode producer job: fast-forward-only pass publishing chain
            # checkpoints; nothing is tuned and no results are written
            start_index = produce_only[0]
        for i in range(start_index, len(block_names), nblocks):
            if produce_only is not None and i >= produce_only[1]:
                break
            if span is not None and i >= span[1]:
                break  # span restriction: remaining blocks belong to another worker
            if (
                span is not None
                and bp_results_dir
                and i >= span[0]
                and _bp_has_block_results(bp_results_dir, block_names[i])
            ):
                # resume: this block was tuned in a previous run. With a chain
                # checkpoint for the next boundary the skip costs nothing;
                # without one, fall through to the fast-forward replay below.
                restored = _bp_load_chain_state(
                    bp_results_dir,
                    group_idx,
                    i + 1,
                    device=input_ids.device if isinstance(input_ids, torch.Tensor) else "cpu",
                )
                if restored is not None:
                    input_ids = restored
                    pbar.update(1)
                    continue
            if input_others_extra_blocks and block_names[i] in input_others_extra_blocks:
                input_others = input_others_extra_blocks[block_names[i]]
                _, input_others = self._preprocess_block_inputs(input_others)
                input_others_extra_blocks.pop(block_names[i])
            if i != 0:
                pbar.update(1)
            if produce_only is not None or span is not None and (
                i < span[0]
                or (
                    bp_results_dir
                    and i >= span[0]
                    and _bp_has_block_results(bp_results_dir, block_names[i])
                    and _bp_load_chain_state(bp_results_dir, group_idx, i + 1, device="cpu") is None
                )
            ):
                input_ids = self._ff_one_block(
                    model, block_names[i], input_ids, input_others, group_idx, i + 1, bp_results_dir
                )
                continue
            if nblocks == 1:
                n = block_names[i]
                pbar.set_description(f"Quantizing {n}")
                m = get_module(model, n)
            else:
                names = block_names[i : min(i + nblocks, len(block_names))]
                pbar.set_description(f"Quantizing [{i + 1}-{min(i + nblocks, len(block_names))}]/{len(block_names)}")
                modules = [get_module(model, n) for n in names]
                m = WrapperMultiblock(modules)

            # Also reload when `AR_DISK_STREAM_MODEL` is set even if
            # `low_cpu_mem_usage` has been forced False (e.g. GGUF export --
            # see base.py's `_finalize_compress_context`, which disables
            # `low_cpu_mem_usage` for gguf formats for reasons unrelated to disk
            # streaming). Under streaming, a block starts on the meta device
            # regardless of `low_cpu_mem_usage`, which only ever controlled whether
            # to *free* it again after use -- without this, the block below is never
            # materialized at all and `m.to(device)` crashes with "Cannot copy out
            # of meta tensor". The block intentionally stays real afterward (no
            # matching post-tune offload runs when `low_cpu_mem_usage` is False --
            # see the `is_immediate_saving`-adjacent offload call further down),
            # matching upstream's own choice not to cycle blocks for these formats.
            if self.compress_context.low_cpu_mem_usage or envs.AR_DISK_STREAM_MODEL:
                if nblocks == 1:
                    self._offloader.reload(model, n)
                else:
                    self._offloader.reload(model, names)

            block_name_or_names = n if nblocks == 1 else names

            # ── Infrastructure: materialize, dtype convert, device placement ──
            materialize_model_(m)
            convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device_manager.device)

            m = self.alg_composer.dispatch_block(m, input_ids, input_others)

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
            is_bp_worker = bool(envs.AR_BLOCK_PARALLEL_WORKER)
            if self.compress_context.is_immediate_packing and not is_bp_worker:
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

            if span is not None and bp_results_dir and i >= span[0]:
                # resumable unit complete: this block's tuned scale/zp plus the
                # chain state entering the next block
                _bp_save_block_results(
                    bp_results_dir, block_names[i], self._collect_tuned_layers(block_names[i])
                )
                _bp_save_chain_state(bp_results_dir, group_idx, i + 1, input_ids)

            if self.compress_context.is_immediate_saving and not is_bp_worker:
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

            if is_bp_worker:
                # Worker: the block's tuned scale/zp are already persisted to
                # the results dir; packing/shards/offload-write all belong to
                # the parent's apply pass. Park the block back on meta so a
                # long queue-serving lifetime never accumulates host memory.
                get_module(model, block_name_or_names).to("meta")
            elif self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
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
            if resume_state is not None and nblocks == 1:
                # `input_ids` was already reassigned to `next_input_ids`
                # above -- it now holds the value the *next* block should use
                # as its chained hidden-state input, which is exactly what
                # needs to be persisted here.
                resume_state.mark_block_done(n, q_input, input_ids)
        if pbar is not None:
            pbar.update(1)

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

    def _build_resume_states(self, all_blocks: list) -> list:
        """Build one ResumeState per block group under AR_RESUME_DIR.

        Shared by the serial tuning path and the block-parallel parent so the
        per-group manifests (signature strings, group_{idx} layout) are
        byte-compatible: a manifest advanced by the parallel sequencer is
        consumed by an unmodified serial run and vice versa.
        """
        from auto_round.utils.resume import ResumeState, compute_run_signature, layer_config_fingerprint

        model_dir = getattr(self.model_context, "disk_stream_model_dir", None) or getattr(
            getattr(self.model_context.model, "config", None), "_name_or_path", None
        )
        dataset_desc = str(getattr(self, "dataset", None))
        # str(self.scheme) alone is bits-blind for AutoScheme runs: two runs
        # with different avg_bits share it, so include the resolved
        # per-layer allocation (see layer_config_fingerprint docstring).
        scheme_desc = (
            str(self.scheme)
            + "|"
            + layer_config_fingerprint(getattr(getattr(self, "quantizer", None), "layer_config", None))
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
            states.append(
                ResumeState(os.path.join(envs.AR_RESUME_DIR, f"group_{group_idx}"), sig, block_names)
            )
        return states

    def _parent_shared_inputs_path(self) -> Optional[str]:
        """Deterministic shared-inputs location for a parent parallel run (resume)."""
        from auto_round.compressors import block_parallel as bp

        results_dir = bp.parallel_results_dir()
        if results_dir is None:
            return None
        return os.path.join(bp.shared_dir(results_dir), "calib_inputs.pt")

    def _bp_compressor_list(self) -> list:
        compressors = self.alg_composer.block_quantizer
        if not isinstance(compressors, (list, tuple)):
            compressors = [compressors]
        else:
            compressors = list(compressors)
        compressors.extend(self.alg_composer.preprocessors)
        return compressors

    def _snapshot_calib_state(self, all_inputs: dict, input_ids_cache) -> dict:
        """Capture the post-cache context so workers can skip their own pass."""
        cctx = self.calibration_context
        return {
            "inputs": all_inputs,
            "input_ids": input_ids_cache,
            "batch_size": cctx.batch_size,
            "seqlen": cctx.seqlen,
            "batch_dim": cctx.batch_dim,
            "is_only_supported_bs1": getattr(cctx, "is_only_supported_bs1", False),
            "orig_batch_size": getattr(cctx, "orig_batch_size", None),
            "block_forward_batch_size": self.alg_composer.block_forward.batch_size,
            "ga_steps": [getattr(c, "gradient_accumulate_steps", None) for c in self._bp_compressor_list()],
        }

    def _restore_calib_state(self, state: dict) -> None:
        cctx = self.calibration_context
        cctx.batch_size = state["batch_size"]
        cctx.seqlen = state["seqlen"]
        cctx.batch_dim = state["batch_dim"]
        cctx.is_only_supported_bs1 = state["is_only_supported_bs1"]
        if state.get("orig_batch_size") is not None:
            cctx.orig_batch_size = state["orig_batch_size"]
        self.alg_composer.block_forward.batch_size = state["block_forward_batch_size"]
        for comp, ga in zip(self._bp_compressor_list(), state.get("ga_steps", [])):
            if ga is not None and hasattr(comp, "gradient_accumulate_steps"):
                comp.gradient_accumulate_steps = ga

    def _maybe_block_parallel_tune(
        self, all_blocks: list, is_worker: bool, all_inputs: dict = None, input_ids_cache=None, pbar=None
    ) -> bool:
        """Run block-parallel tuning when enabled; returns True if results were applied.

        Parent-side: spawns one worker process per eligible GPU (each pinned via
        CUDA_VISIBLE_DEVICES and restricted to a contiguous span of blocks), waits,
        merges the workers' tuned scale/zp dumps, and applies them through the
        ordinary immediate-pack/save flow. Workers themselves (``is_worker``)
        take the span-restricted serial loop instead.
        """
        if is_worker:
            return False
        from auto_round.compressors import block_parallel as bp

        def _fail_if_requested(reason: str) -> bool:
            # Explicitly requested but cannot run: fail loudly instead of
            # silently burning hours in the serial loop. Want serial? Don't set
            # the parallel flag.
            raise RuntimeError(
                f"AR_ENABLE_BLOCK_PARALLEL_TUNING=1 was requested but block-parallel tuning "
                f"cannot run: {reason}. Fix the condition, or unset the flag for serial tuning."
            )

        reason = bp.block_parallel_tuning_enabled(
            need_quanted_input=bool(self.alg_composer.need_quanted_input()),
            super_group_size=self.super_group_size,
            nblocks=self.nblocks,
            n_block_groups=len(all_blocks),
            is_immediate_packing=self.compress_context.is_immediate_packing,
            is_immediate_saving=self.compress_context.is_immediate_saving,
            n_blocks_total=sum(len(group) for group in all_blocks),
            argv=sys.argv,
        )
        if reason == "env":
            return False
        if reason is not None:
            logger.info("block-parallel tuning disabled: %s", reason)
            return _fail_if_requested(reason)
        # Worker pool: the user's --device_map is the request; eligibility is
        # derived from the model (VRAM) and from measured host RAM -- no knobs.
        allowed = []
        for dev in device_manager.device_list:
            try:
                allowed.append(int(str(dev).split(":")[-1]))
            except ValueError:
                continue
        # per-GPU VRAM requirement: largest block's weight bytes with a
        # working-set multiplier (original + QDQ copy + workspace) plus an
        # activation margin. Shapes are known even on the meta skeleton.
        largest_block_bytes = 0
        for group in all_blocks:
            for block_name in group:
                block = get_module(self.model_context.model, block_name)
                if block is None:
                    continue
                nbytes = sum(p.numel() * p.element_size() for p in block.parameters())
                largest_block_bytes = max(largest_block_bytes, nbytes)
        required_vram_bytes = int(largest_block_bytes * 3) + (4 * 1024**3)
        gpu_ids = bp.eligible_gpus(required_vram_bytes, allowed_indices=allowed or None)

        # Resume storage reuses the serial flag: AR_RESUME_DIR set -> per-block
        # result files + chain checkpoints under <AR_RESUME_DIR>/block_parallel,
        # validated by run signature; unset -> fresh scratch dir (no resume).
        results_dir = bp.parallel_results_dir()
        resumable = results_dir is not None
        import shutil

        if resumable:
            from auto_round.utils.resume import compute_run_signature, layer_config_fingerprint

            model_dir = getattr(self.model_context, "disk_stream_model_dir", None) or getattr(
                getattr(self.model_context.model, "config", None), "_name_or_path", None
            )
            signature = compute_run_signature(
                model_dir,
                str(self.scheme)
                + "|"
                + layer_config_fingerprint(getattr(getattr(self, "quantizer", None), "layer_config", None))
                + "|qi="
                + str(bool(self.alg_composer.need_quanted_input())),
                str(getattr(self, "dataset", None)),
                self.calibration_context.nsamples,
                self.calibration_context.seqlen,
                flatten_list(all_blocks),
            )
            if os.path.isdir(results_dir) and not bp.signature_matches(results_dir, signature):
                logger.warning(
                    "block-parallel tuning: results in %s were produced under a different run "
                    "signature; discarding them and starting fresh",
                    results_dir,
                )
                shutil.rmtree(results_dir, ignore_errors=True)
            bp.save_signature(results_dir, signature)
        else:
            output_root = getattr(self.compress_context, "output_dir", None) or "tmp_autoround"
            results_dir = os.path.join(output_root, "block_parallel_results")
            shutil.rmtree(results_dir, ignore_errors=True)  # no resume: never reuse stale results
        sdir = bp.shared_dir(results_dir)
        nonblocks_path = os.path.join(sdir, "nonblocks.pt")
        inputs_path = os.path.join(sdir, "calib_inputs.pt")
        os.makedirs(sdir, exist_ok=True)
        if not os.path.isfile(nonblocks_path):
            block_prefixes = flatten_list(
                get_block_names(self.model_context.model, quant_vision=self.model_context.quant_nontext_module)
            )
            n_saved = bp.save_shared_nonblocks(self.model_context.model, block_prefixes, nonblocks_path)
            logger.info("block-parallel tuning: shared %d non-block tensors to %s", n_saved, nonblocks_path)
        if not os.path.isfile(inputs_path) and all_inputs is not None:
            bp.save_shared_inputs(self._snapshot_calib_state(all_inputs, input_ids_cache), inputs_path)
            logger.info("block-parallel tuning: shared calibration inputs to %s", inputs_path)

        # Canonical done-state: serial manifests (contiguous prefix per group),
        # advanced by this sequencer as blocks complete in order. An unmodified
        # serial run can resume from them later; out-of-order (fringe)
        # completions live only in per-block result files until absorbed.
        resume_states = self._build_resume_states(all_blocks) if resumable else [None] * len(all_blocks)

        # done sets: manifest prefix + result files from a prior run (works
        # after a serial run too -- manifests carry the prefix, no result files
        # needed -- and regardless of how many workers produced them)
        done = {}
        for g, blocks in enumerate(all_blocks):
            prefix = resume_states[g].resume_index if resume_states[g] is not None else 0
            done[g] = set(range(prefix))
            done[g] |= {k for k, b in enumerate(blocks) if bp.has_block_results(results_dir, b)}
        total_blocks = sum(len(blocks) for blocks in all_blocks)
        n_todo = total_blocks - sum(len(d) for d in done.values())

        logger.info(
            "block-parallel tuning: %d workers on GPUs %s, %d block(s) to tune, results %s%s",
            len(gpu_ids), gpu_ids, n_todo, results_dir,
            " (resumable: rerun the same command to finish missing blocks)" if resumable else "",
        )
        env_extra = {
            "AR_BLOCK_PARALLEL_RESULTS": results_dir,
            "AR_BLOCK_PARALLEL_SHARED_NONBLOCKS": nonblocks_path,
            "AR_BLOCK_PARALLEL_SHARED_INPUTS": inputs_path,
        }
        import time as _t0

        _spawn_t0 = _t0.time()
        procs = bp.spawn_workers(sys.argv, gpu_ids, log_dir=results_dir, extra_env=env_extra)
        logger.info(
            "[bp-trace] spawn complete in %.1fs (pids %s); pushing initial assignments",
            _t0.time() - _spawn_t0, [proc.pid for proc in procs],
        )

        def _ordered_undone():
            return [(g, k) for g, blocks in enumerate(all_blocks) for k, b in enumerate(blocks)
                    if k not in done[g] and (g, k) not in assigned.values()]

        assigned = {}  # rank -> (group, index); absent == worker idle/finished
        stopped = set()
        seen_results = set()
        sticky_pref = {}  # rank -> (g, k+1) the worker holds the entry for
        finished_ranks = set()  # ranks that reported a result since last pass

        def _send(rank: int, message: str) -> None:
            proc = procs[rank]
            if proc.poll() is not None:
                return
            try:
                proc.stdin.write(message + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass  # worker death is handled by the liveness check

        def _assign_next(rank: int, prefer=None) -> None:
            """Push the next assignment to ``rank``: the sticky preference when
            still undone and unassigned, else the lowest undone block. Worker
            count never matters -- this works for any pool size."""
            if prefer is not None and prefer in _ordered_undone():
                target = prefer  # sticky: the worker already holds this entry
            else:
                candidates = _ordered_undone()
                if not candidates:
                    stopped.add(rank)
                    _send(rank, "stop")
                    return
                target = candidates[0]
            assigned[rank] = target
            _send(rank, f"tune {target[0]} {target[1]}")
            logger.info("block-parallel tuning: assigned g%d k%d to worker %d", target[0], target[1], rank)

        for rank in range(len(procs)):
            _assign_next(rank)

        import time as _time

        shown_done = sum(len(d) for d in done.values())
        if pbar is not None:
            pbar.set_description("Tuning blocks (parallel)")
            pbar.refresh()
        rss_tick = 0
        while True:
            undone_left = total_blocks - sum(len(d) for d in done.values())
            if undone_left == 0 and not assigned:
                break
            _time.sleep(0.5)
            # fatal-fast liveness: any worker exit while work remains aborts
            for widx, proc in enumerate(procs):
                code = proc.poll()
                if code is None:
                    continue
                for sibling in procs:
                    if sibling.poll() is None:
                        sibling.terminate()
                raise RuntimeError(
                    f"block-parallel tuning: worker {widx} exited with code {code} while blocks "
                    f"remained; terminated siblings. Completed blocks are persisted -- rerun the "
                    f"same command to resume. Worker logs: {results_dir}/worker_*.log"
                )
            # new results -> absorb, then push the next assignment to that worker
            for g, blocks in enumerate(all_blocks):
                for k, b in enumerate(blocks):
                    if (g, k) in seen_results or not bp.has_block_results(results_dir, b):
                        continue
                    seen_results.add((g, k))
                    done[g].add(k)
                    rank = bp.read_result_rank(results_dir, b)
                    if rank >= 0:
                        assigned.pop(rank, None)
                        sticky_pref[rank] = (g, k + 1)
                        finished_ranks.add(rank)
            # ordered commit into the serial manifests, consuming the entry
            # handoff files the workers published (then deleting them)
            if resumable:
                for g, blocks in enumerate(all_blocks):
                    rs = resume_states[g]
                    while rs is not None and rs.resume_index < len(blocks) and rs.resume_index in done[g]:
                        k = rs.resume_index
                        entry_next = _bp_load_chain_state(results_dir, g, k + 1, device="cpu")
                        rs.mark_block_done(blocks[k], None, entry_next)
                        try:
                            os.remove(bp._chain_state_path(results_dir, g, k + 1))
                        except OSError:
                            pass
            # re-assign ranks that reported a result since the last pass
            for rank in list(finished_ranks):
                finished_ranks.discard(rank)
                if rank not in assigned and rank not in stopped:
                    _assign_next(rank, prefer=sticky_pref.pop(rank, None))
            done_now = sum(len(d) for d in done.values())
            if pbar is not None and done_now != shown_done:
                pbar.update(done_now - shown_done)
                shown_done = done_now
            rss_tick += 1
            if rss_tick % 60 == 0:
                bp.log_worker_rss(procs)
            if rss_tick % 10 == 0:
                logger.info(
                    "[bp-trace] tick %d: done=%d/%d assigned=%s idle=%d",
                    rss_tick, done_now, total_blocks,
                    dict(assigned), sum(1 for r in range(len(procs)) if r not in assigned and r not in stopped),
                )

        for rank in range(len(procs)):
            _send(rank, "stop")
        codes = bp.wait_workers(procs)
        if resumable:
            for g, blocks in enumerate(all_blocks):
                rs = resume_states[g]
                while rs is not None and rs.resume_index < len(blocks) and rs.resume_index in done[g]:
                    k = rs.resume_index
                    entry_next = _bp_load_chain_state(results_dir, g, k + 1, device="cpu")
                    rs.mark_block_done(blocks[k], None, entry_next)
                    try:
                        os.remove(bp._chain_state_path(results_dir, g, k + 1))
                    except OSError:
                        pass

        tuned = bp.load_all_block_results(results_dir)
        missing = bp.missing_result_blocks(results_dir, all_blocks)
        if missing or not tuned:
            raise RuntimeError(
                f"block-parallel tuning: no results for {len(missing)} block(s) "
                f"(first: {missing[0] if missing else 'n/a'}); logs in {results_dir}"
            )
        logger.info(
            "block-parallel tuning: merged tuned results for %d layers (worker exit codes %s)",
            len(tuned), codes,
        )
        self._apply_tuned_results(all_blocks, tuned)
        if resumable:
            # quantize_and_save clears these after a successful export (same as serial)
            self._resume_states = resume_states
        return True

    _IMAGE_INPUT_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "videos", "images")

    @classmethod
    def _inputs_contain_images(cls, inputs) -> bool:
        """True when cached calibration inputs carry image/video tensors."""
        if not isinstance(inputs, dict):
            return False
        for key, value in inputs.items():
            if isinstance(key, str) and any(token in key for token in cls._IMAGE_INPUT_KEYS):
                if value is not None and (not isinstance(value, (list, tuple)) or len(value) > 0):
                    return True
        return False

    def _park_unused_modules_for_worker(self, all_inputs: dict) -> None:
        """Meta-park modules that block-wise tuning never invokes.

        lm_head: the calibration forward early-stops at the last cached block
        and block-wise tuning computes per-block losses only, so the head is
        structurally unreachable -- parked unconditionally.

        Vision encoders: unreachable for text-only calibration; parked only
        when the cached calibration inputs contain no image/video data (the
        runtime check replaces a configuration knob).

        Embeddings stay materialized (the first block's cached input depends
        on them). Parking cuts the duplicated non-block params from each
        worker's host RAM.
        """
        import gc as _gc

        model = self.model_context.model
        candidates = []
        lm_head = get_lm_head_name(model)
        if lm_head:
            candidates.append(lm_head)
        has_images = any(self._inputs_contain_images(v) for v in all_inputs.values())
        if not has_images:
            vision_tokens = ("visual", "vision", "multi_modal", "multimodal")
            for name, _module in model.named_modules():
                if not name or name.count(".") > 1:
                    continue  # encoder containers only, not their internals
                if any(tok in name.lower().split(".")[-1] for tok in vision_tokens):
                    candidates.append(name)
        freed = 0
        for name in dict.fromkeys(candidates):
            module = get_module(model, name)
            if module is None:
                continue
            params = list(module.parameters())
            if not params or all(param.device.type == "meta" for param in params):
                continue
            freed += sum(param.numel() * param.element_size() for param in params)
            module.to("meta")
        if freed:
            _gc.collect()
            logger.info(
                "block-parallel worker: parked unused modules on meta (~%.2f GiB host RAM saved)",
                freed / (1024**3),
            )

    def _serve_block_queue(self, all_blocks: list, all_inputs: dict, token_ids, pbar) -> None:
        """Worker-side assignment loop (push protocol).

        Assignments arrive as stdin lines (``tune G K`` / ``stop``). For each
        block: obtain its entry -- from the held previous chain step when the
        assignment is sticky (zero cost), else by replaying from the serial
        manifest frontier (bounded by the in-flight gap, ~one block per worker)
        -- tune it, extend the chain by one forward to hold the next entry, and
        persist the result (block file; rank rides along as advisory metadata).
        All durable state is files; the pipe carries ephemeral assignments only.
        """
        from auto_round.compressors import block_parallel as bp

        results_dir = envs.AR_BLOCK_PARALLEL_RESULTS
        if not results_dir:
            raise RuntimeError("block-parallel worker is missing its results environment")
        rank = envs.AR_BLOCK_PARALLEL_RANK

        def _group_entry(blocks):
            entry = dict(all_inputs[blocks[0]])
            entry, _q = _update_inputs(entry, None)
            return entry

        held = None  # (group, next_index, entry)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "stop":
                break
            _cmd, g, k = line.split()
            g, k = int(g), int(k)
            logger.info("[bp-worker %d] assignment: %s", rank, line)
            blocks = all_blocks[g]
            if k >= len(blocks):
                raise RuntimeError(f"assignment out of range: {line}")

            entry = None
            if held is not None and held[0] == g and held[1] == k:
                entry = held[2]  # sticky: previous assignment's chain tail
            else:
                prefix, frontier_entry = self._manifest_frontier(g)
                if prefix > k:
                    prefix = 0  # stale/odd manifest: replay from the group entry
                    frontier_entry = None
                if prefix < k:
                    # replay prefix..k-1 (publishes entries the parent can
                    # consume for manifest absorption on the way)
                    self._quantize_blocks(
                        self.model_context.model,
                        _group_entry(blocks),
                        blocks,
                        nblocks=1,
                        pbar=pbar,
                        input_others_extra_blocks=dict(all_inputs),
                        token_ids=token_ids,
                        group_idx=g,
                        bp_results_dir=results_dir,
                        start_block_idx=prefix,
                        produce_only=(prefix, k),
                        resume_input_ids=frontier_entry,
                    )
                # a sibling worker replaying the same prefix may still be mid-
                # publish of this checkpoint -- retry briefly before failing
                import time as _t

                entry = None
                for _attempt in range(60):
                    entry = _bp_load_chain_state(
                        results_dir, g, k, device=self.compress_context.cache_device
                    )
                    if entry is not None:
                        break
                    _t.sleep(0.5)
                if entry is None:
                    raise RuntimeError(f"chain entry for block {k} missing after replay")
            if isinstance(entry, list) and any(t is None for t in entry):
                bad = [i for i, t in enumerate(entry) if t is None][:4]
                raise RuntimeError(
                    f"chain entry for block {k} contains None samples at indices {bad} -- "
                    "the checkpoint file is corrupt; delete it and rerun"
                )

            # extend the chain BEFORE tuning: the ff consumes the pristine
            # entry; whatever the tuning pipeline does to the list afterward
            # no longer matters (nothing reads it again). The block's original
            # weights are streamed from the checkpoint either way.
            if k + 1 < len(blocks):
                _others = self._preprocess_block_inputs(_group_entry(blocks))[1]
                next_entry = self._ff_one_block(
                    self.model_context.model,
                    blocks[k],
                    entry,
                    _others,
                    g,
                    k + 1,
                    results_dir,
                )
                held = (g, k + 1, next_entry)
            else:
                held = None

            self._quantize_blocks(
                self.model_context.model,
                _group_entry(blocks),
                blocks,
                nblocks=1,
                pbar=pbar,
                input_others_extra_blocks=dict(all_inputs),
                token_ids=token_ids,
                span=(k, k + 1),
                group_idx=g,
                bp_results_dir=results_dir,
                start_block_idx=k,
                resume_input_ids=entry,
            )
            logger.info("[bp-worker %d] block %s done (rank holds %s)", rank, blocks[k], held[1] if held else "-")

    def _collect_tuned_layers(self, block_name: str) -> dict:
        """Extract this block's tuned {layer_name: {scale, zp}} from live modules.

        Keying rule matches ``_apply_tuned_results``: prefer ``global_name``,
        fall back to the module path (identical construction on both sides).
        """
        results = {}
        block = get_module(self.model_context.model, block_name)
        for sub_name, sub in block.named_modules():
            if not hasattr(sub, "scale"):
                continue
            layer_name = getattr(sub, "global_name", None) or (f"{block_name}.{sub_name}" if sub_name else block_name)
            results[layer_name] = {
                "scale": sub.scale.detach().cpu() if isinstance(sub.scale, torch.Tensor) else sub.scale,
                "zp": sub.zp.detach().cpu() if isinstance(sub.zp, torch.Tensor) else sub.zp,
            }
        return results

    def _dump_parallel_worker_results(self, all_blocks: list) -> None:
        """Worker-side safety net: persist any tuned blocks not yet saved.

        Blocks are normally saved right after tuning inside ``_quantize_blocks``
        (the resumable unit); this end-of-run pass only covers blocks whose
        in-loop save was somehow skipped. Blocks this worker never touched have
        no tuned attributes and contribute nothing.
        """
        out_dir = envs.AR_BLOCK_PARALLEL_RESULTS
        if not out_dir:
            raise RuntimeError("AR_BLOCK_PARALLEL_WORKER is set but AR_BLOCK_PARALLEL_RESULTS is missing")
        saved = 0
        for blocks in all_blocks:
            for block_name in blocks:
                if _bp_has_block_results(out_dir, block_name):
                    continue
                results = self._collect_tuned_layers(block_name)
                if results:
                    _bp_save_block_results(out_dir, block_name, results)
                    saved += 1
        logger.info(
            "block-parallel worker: results dir %s (%d blocks saved by the end-of-run net)", out_dir, saved
        )

    def quantize(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Quantize the model and return the quantized model along with layer configurations.The entry of AutoRound.
        Returns:
        The quantized model and layer configurations.
        """
        self.post_init()

        if not self.need_calib:
            return self._quantize_zero_shot()

        return self._quantize_data_driven()

    @torch.no_grad()
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

        all_blocks = self.quant_block_list or get_block_names(self.model)
        pbar = tqdm(range(sum(len(block) for block in all_blocks)))
        for block_names in all_blocks:
            for block_name in block_names:
                pbar.set_description(f"Quantizing {block_name}")
                block = get_module(self.model, block_name)

                # ── Infrastructure: materialize ───────────────────────────
                materialize_model_(block)

                # ── Pure algorithm ────────────────────────────────────────
                ctx = BlockContext(
                    model=self.model,
                    block_names=[block_name],
                    block_name=block_name,
                    block_index=0,
                )
                # ── MoE scale alignment for FP8 dispatch efficiency ────────────────
                if is_nv_fp(self.act_data_type) or not self.act_dynamic:
                    set_amax_for_all_moe_layers(block, attr_name="act_max")

                update_block_global_scale_if_needed(block, self.data_type, self.group_size)
                self.alg_composer.compress_block(block, fp_inputs=None, input_others={}, block_ctx=ctx)
                if self.compress_context.is_immediate_packing:
                    for _n, _mod in block.named_modules():
                        if hasattr(_mod, "bits") and check_to_quantized(_mod):
                            from auto_round.compressors.utils import immediate_pack as _immediate_pack

                            module_name = getattr(_mod, "global_name", None)
                            if module_name is None and self.nblocks == 1 and _n:
                                module_name = f"{block.global_name}.{_n}"
                            if module_name is None:
                                continue
                            _immediate_pack(module_name, self.layer_config)

                # ── Infrastructure: shard write / device cleanup ──────────
                if self.compress_context.is_immediate_saving:
                    # Save non-quantized leaf modules (e.g. norms, embeddings in block).
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
                    # Write at block scope for any remaining params/buffers.
                    self.shard_writer.write(name=block_name)
                    block.to("meta")
                else:
                    mv_module_from_gpu(block)
                    if self.compress_context.low_cpu_mem_usage:
                        self._offloader(self.model, block_name)

                clear_memory()
                memory_monitor.log_summary()
                pbar.update(1)

        remain_layer_names = []
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in self.model.named_modules():
            if not check_to_quantized(m):
                continue
            # Skip if this layer is part of any block (by prefix match)
            if any(n == block_name or n.startswith(f"{block_name}.") for block_name in block_name_set):
                continue
            remain_layer_names.append(n)
        for name in remain_layer_names:
            logger.info(f"Quantizing remaining layer {name} on CPU.")
            self.alg_composer.compress_layer_outside_block(get_module(self.model, name))
            # Outside-block layers (embed_tokens/lm_head/etc.) are typically few so just
            # log a summary after each one.
            clear_memory()
            memory_monitor.log_summary()

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model)
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

        self.model_context.quantized = True
        return self.model, self.layer_config

    def _quantize_data_driven(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        """Data-driven quantization path — uses calibration data for optimization."""

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        self._check_compatibility()

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
        _shared_inputs_path = (
            envs.AR_BLOCK_PARALLEL_SHARED_INPUTS
            if envs.AR_BLOCK_PARALLEL_WORKER
            else self._parent_shared_inputs_path()
        )
        if _shared_inputs_path and os.path.isfile(_shared_inputs_path):
            # block-parallel worker (or parent resuming a prior parallel run):
            # reuse the parent-saved calibration inputs + synced context
            # snapshot via mmap -- skips the dataset pass and cache forward
            from auto_round.compressors import block_parallel as _bp_shared

            payload = _bp_shared.load_shared_inputs(_shared_inputs_path)
            self._restore_calib_state(payload)
            all_inputs = payload["inputs"]
            input_ids_cache = payload["input_ids"]
            logger.info(
                "calibration inputs restored from shared file %s (cache pass skipped)",
                _shared_inputs_path,
            )
        else:
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

        start_time = time.time()

        self.alg_composer.prepare_run()

        # Build one ResumeState per block group (almost always just one group
        # for text-only dense models) when AR_RESUME_DIR is set, so a
        # crash/kill mid-tuning can resume from the first not-yet-quantized
        # block instead of restarting from block 0. See auto_round/utils/resume.py.
        # ── Block-parallel tuning (AR_ENABLE_BLOCK_PARALLEL_TUNING, experimental) ──
        # Worker processes re-exec the same CLI pinned to one GPU each, tune a
        # contiguous span of blocks against the original-weights input chain
        # (blocks are independent: the serial loop chains reference outputs,
        # not quantized outputs), and dump tuned scale/zp. The parent merges
        # the dumps and packs/saves through the ordinary immediate flow. The
        # serial loop below remains the default and the fallback.
        parallel_applied = self._maybe_block_parallel_tune(
            all_blocks,
            bool(envs.AR_BLOCK_PARALLEL_WORKER),
            all_inputs=all_inputs,
            input_ids_cache=input_ids_cache,
            pbar=pbar,
        )

        resume_states = None
        if envs.AR_RESUME_DIR and not parallel_applied and not envs.AR_BLOCK_PARALLEL_WORKER:
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


        try:
            if envs.AR_BLOCK_PARALLEL_WORKER:
                # queue-serve mode: consume dispatch jobs (tune / chain-produce)
                # until the parent signals STOP and the queue is drained
                self._park_unused_modules_for_worker(all_inputs)
                self._serve_block_queue(all_blocks, all_inputs, input_ids_cache, pbar)
            for group_idx, block_names in enumerate(
                all_blocks if not parallel_applied and not envs.AR_BLOCK_PARALLEL_WORKER else []
            ):
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
        finally:
            # ── Pipeline lifecycle: finalize_quantization (model-level teardown)
            self.alg_composer.finalize_run()
        pbar.set_description("Quantizing done")
        pbar.close()
        if envs.AR_BLOCK_PARALLEL_WORKER:
            # Worker process: its job ends with serving queue jobs. Packing,
            # saving, outside-block layers and model reloads belong to the parent.
            self._dump_parallel_worker_results(all_blocks)
            return self.model_context.model, self.layer_config
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

        end_time = time.time()
        cost_time = end_time - start_time
        logger.info(f"quantization tuning time {cost_time}")

        # Dump a summary
        quantized_layers = []
        unquantized_layers = []
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

    # def _immediate_pack_and_save_module(self, module_name):
    #     from auto_round.compressors.shard_writer import ShardWriter
    #
    #     shard_writer = ShardWriter.get_shard_writer()
    #     to_cpu = self.compress_context.low_gpu_mem_usage
    #     module = get_module(self.model, module_name)
    #     if self.compress_context.is_immediate_packing:
    #         immediate_pack(module_name, self.layer_config)
    #         if to_cpu:
    #             module = module.to("cpu")
    #             packed_module = get_module(self.model, module_name)
    #             set_module(self.model, module_name, packed_module.to("cpu"))
    #     else:
    #         if to_cpu:
    #             module = module.to("cpu")
    #         set_module(self.model, module_name, module)
    #     if self.compress_context.is_immediate_saving:
    #         module = get_module(self.model, module_name)
    #         module.to("cpu")
    #         shard_writer.write(module, module_name, False)
    #         module.to("meta")

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
