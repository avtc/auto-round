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
from auto_round.utils.peak_watch import PeakWatcher
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
            if self._should_offload_after_pack(self.compress_context) or envs.AR_DISK_STREAM_MODEL:
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

            if logger.isEnabledFor(logging.DEBUG):
                self._log_device_inventory(
                    {"input_others": input_others},
                    f"block {i}",
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
            if logger.isEnabledFor(logging.DEBUG):
                # post-tune sample: the VmHWM delta vs the pre-tune line
                # measures this block's during-tuning transient peak
                self._log_device_inventory(
                    None,
                    f"block {i} post",
                    extra_buckets={"cache-remaining": input_others_extra_blocks or {}},
                )
            if self._peak_watch is not None:
                self._peak_watch.set_phase("write")
                self._peak_watch.log(f"block {i}")
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
        # after the context finished loading)
        streamer = getattr(self.model_context, "checkpoint_streamer", None)
        if streamer is not None and getattr(self, "amp", False):
            streamer.load_dtype = self.amp_dtype

        if not self.need_calib:
            return self._quantize_zero_shot()

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
        jumped = False
        for rs in resume_states:
            if rs is None or rs.resume_index <= 0:
                continue
            entry = rs.load_input_ids()
            if entry is None:
                continue
            calib_state["fp_inputs"] = entry
            q_input = rs.load_q_input()
            if q_input is not None and self.alg_composer.need_quanted_input():
                calib_state["q_inputs"] = q_input
            jumped = True
        if jumped:
            logger.info("[stream] resume: calibration chain jumped to the saved frontier entry")

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
                    holder["write"] = _time.perf_counter() - _t0
                    if rs is not None:
                        # crash-durability contract, same as the serial path:
                        # the manifest may claim this block done only after
                        # its tensors are durably flushed to a shard file
                        self.shard_writer._flush_shard()
                        _t0 = _wtime.perf_counter()
                        rs.mark_block_done(block_name, q_snap, None if is_model_last else fp_snap)
                        holder["snap"] = _wtime.perf_counter() - _t0
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
        """Release reserved-but-unallocated CUDA segments (resume-rebuild debris).

        The resume rebuild (chain jump, shard adoption, hydration) frees
        transient CUDA allocations whose segments stay cached by the
        allocator; the first post-resume block then OOMs on fragmentation
        even though its live set fits comfortably (measured: ~6.8G reserved
        but unallocated right before an OOM on a block the fresh run had
        quantized in the same co-located placement).
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if reason:
                    parts = []
                    for idx in range(torch.cuda.device_count()):
                        alloc = torch.cuda.memory_allocated(idx) / 2**30
                        reserved = torch.cuda.memory_reserved(idx) / 2**30
                        parts.append(f"cuda:{idx} alloc {alloc:.2f}G reserved {reserved:.2f}G")
                    logger.info("[stream] cuda state after %s: %s", reason, "; ".join(parts))
        except Exception:  # noqa: BLE001  diagnostics only; never break the run
            pass

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
            if not torch.cuda.is_available():
                return False
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type != "cuda":
                return False
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            gap = torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
            if gap < min_gap_bytes:
                return False
            torch.cuda.empty_cache()
            if not getattr(self, "_frag_release_logged", False):
                self._frag_release_logged = True
                logger.info(
                    "[stream] released %.2fG of cached free segments on %s before tuning "
                    "(allocator fragmentation hygiene; one-time log)",
                    gap / 2**30,
                    dev,
                )
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
        per_dev: dict = collections.defaultdict(lambda: collections.defaultdict(int))
        top_thr = 0.1 * 2**30  # list tensors >= 0.1G alongside the buckets
        big: list = []  # (nbytes, "dev:name") tuples when the inventory threshold is set

        def _add(dev: str, bucket: str, t, name: str = None) -> None:
            nbytes = t.numel() * t.element_size()
            per_dev[dev][bucket] += nbytes
            if top_thr and nbytes >= top_thr:
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
            for _t in pending.values():
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
            logger.debug(
                "[stream-mem] %s host: rss %.2fG (peak %.2fG) | %s | residual(rss-tracked) %.2fG",
                tag,
                rss_gb,
                peak_gb,
                host_parts or "no tracked cpu tensors",
                max(0.0, rss_gb - tracked_gb),
            )
            if top_thr:
                top = sorted((b for b in big if b[1].startswith("cpu:")), reverse=True)[:12]
                if top:
                    logger.debug(
                        "[stream-mem] %s host top tensors: %s",
                        tag,
                        ", ".join(f"{n / 2**30:.2f}G {name}" for n, name in top),
                    )
                # regions classify the RESIDUAL (dead memory holds no tensor
                # objects): [heap] => allocator fragmentation; anonymous
                # mappings => CUDA pinned pools / torch host caches;
                # file-backed => checkpoint mmaps
                regions = sorted(psutil.Process().memory_maps(grouped=False), key=lambda m: -m.rss)[:8]
                region_parts = "; ".join(
                    f"rss {m.rss / 2**30:.2f}G size {m.size / 2**30:.2f}G {m.path or '[anon]'}"
                    for m in regions
                    if m.rss > 0
                )
                if region_parts:
                    logger.debug("[stream-mem] %s host regions: %s", tag, region_parts)
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
            if top_thr:
                top = sorted((b for b in big if b[1].startswith(f"{dev}:")), reverse=True)[:12]
                if top:
                    logger.debug(
                        "[stream-mem] %s %s top tensors: %s",
                        tag,
                        dev,
                        ", ".join(f"{n / 2**30:.2f}G {name.split(':', 1)[1]}" for n, name in top),
                    )
            other = max(0.0, alloc - tracked / 2**30)
            logger.debug(
                "[stream-mem] %s %s: alloc %.2fG / reserved %.2fG | %s | other(alloc-tracked) %.2fG",
                tag,
                dev,
                alloc,
                reserved,
                parts or "no tracked tensors",
                other,
            )

    def _resolve_stream_stage_devices(self):
        """Resolve ``stream_prefetch_devices`` into a list of staging devices.

        Returns None for host-RAM staging (the default). "auto" uses every CUDA
        device except the quant device (all of them when only one exists);
        explicit lists accept ints (``cuda:k``), device strings, or torch.device.
        GPU staging with a CPU quant device falls back to host RAM: blocks
        would quantize on CPU, so VRAM staging buys nothing there.
        """
        value = getattr(self, "stream_prefetch_devices", None)
        if not value:
            return None
        quant_dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        forced = isinstance(value, str) and value.strip().lower() == "on"
        if isinstance(value, str) and value.strip().lower() in ("auto", "on"):
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
            pool = allowed if allowed else [torch.device("cuda", i) for i in range(n_gpu)]
            devices = [d for d in pool if d != quant_dev]
            # The primary joins the rotation when its free VRAM comfortably
            # holds the LARGEST block (block sizes vary by layer type - linear-attention vs
            # full-attention vs first/last): parent working set + block + tuning
            # transients must coexist there for one block at a time.
            primary_fit = self._primary_fits_largest_block(quant_dev)
            if primary_fit is not None:
                largest_gb, free_gb = primary_fit
                devices = list(pool)
                logger.info(
                    "[stream] auto staging includes the primary %s: largest block %.2fG fits free %.2fG "
                    "(headroom 3.0G for tuning transients)",
                    quant_dev,
                    largest_gb,
                    free_gb,
                )
            elif len(pool) == 1 and not devices:
                # sole GPU and the largest block does not fit with headroom:
                # do not stage on it, host RAM takes the lookahead
                logger.info("%s: largest block does not fit free VRAM on the sole GPU", ram_last_resort)
                return None
            if forced and not devices:
                logger.info("%s: no usable staging GPU", ram_last_resort)
                return None
        else:
            devices = [torch.device(f"cuda:{d}") if isinstance(d, int) else torch.device(d) for d in value]
            if any(d.type == "cuda" for d in devices) and any(d.type == "cpu" for d in devices):
                raise ValueError(
                    "mixed GPU/CPU staging devices are not supported: staged blocks are "
                    "quantized in place (round-robin homes), so staging must be all-GPU "
                    "or CPU-only (host RAM); pass e.g. --stream_prefetch auto or --stream_prefetch cpu"
                )
            if quant_dev.type != "cuda" and any(d.type == "cuda" for d in devices):
                logger.warning(
                    "[stream] GPU staging devices ignored with CPU quant device %s; using host RAM", quant_dev
                )
                return None
        for d in devices:
            if d.type == "meta":
                raise ValueError(f"invalid staging device {d}: meta tensors hold no data")
        depth = int(getattr(self, "stream_prefetch", 0) or 0)
        shown_depth = depth if depth else 1
        logger.info(
            "[stream] staging %d block(s) deep on %s; blocks quantize in place (round-robin homes)",
            shown_depth,
            [str(d) for d in devices],
        )
        return devices or None

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

        all_blocks = self.quant_block_list or get_block_names(self.model)
        flat_block_names = [name for group in all_blocks for name in group]

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
            logger.info("[stream_calibration] chain initialized with %d row(s)", summary["rows"])
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
                if self._trim_host_heap():
                    # the replay leaves granular-read debris that would raise
                    # every later block's peak (VmHWM keeps the high-water mark)
                    logger.info("[stream] host heap trimmed after chain rebuild")
                self._release_cuda_cache("resume rebuild")

            if streamer is not None:
                # startup reads (embeddings / chain init) touch shards the
                # block loop never revisits - on fresh runs too, not only
                # resume; close them before the prefetch pipeline starts so
                # their mappings stop counting in RSS
                streamer.release_startup_handles_()

        # Prefetch pipeline: a background reader stages upcoming blocks ahead
        # of the quantize loop. With staging devices the blocks land directly
        # on those devices (round-robin by block index) and are quantized in
        # place there; otherwise they wait in host RAM. Lookahead is ONE block
        # regardless of staging-device count: quantize time per block dwarfs
        # its load time, so deeper staging only holds extra block-sized VRAM.
        raw_depth = getattr(self, "stream_prefetch", 0)
        stage_devices = self._resolve_stream_stage_devices() if (streamer is not None and raw_depth != 0) else None
        if raw_depth == 0 or raw_depth is None:
            prefetch_depth = 0 if raw_depth == 0 else 1
        else:
            prefetch_depth = int(raw_depth)
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
            streamer.start_prefetch(prefetch_names, depth=prefetch_depth, stage_devices=stage_devices)

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
        _bg_pack = None

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
        _peak_watch = PeakWatcher() if logger.isEnabledFor(logging.DEBUG) else None
        if _peak_watch is not None:
            _peak_watch.start()
        flat_block_names = [name for group in all_blocks for name in group]
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

                # ── Infrastructure: materialize ───────────────────────────
                _t_load = _time.perf_counter()
                _load_sub = {} if envs.AR_PERF_COUNTERS else None
                _t_seg = _time.perf_counter()
                if _peak_watch is not None:
                    _peak_watch.set_phase("load")
                if streamer is not None:
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
                    if logger.isEnabledFor(logging.DEBUG):
                        # diagnostics: accounted separately so the io figure
                        # stays honest about the actual load cost
                        self._log_device_inventory(calib_state, f"block {stream_block_idx}")
                        _t_seg = _mark_load_seg(_load_sub, "inv", _t_seg)
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
                        calib_state["fp_inputs"][0],
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
                if _bg_pack_eligible:
                    # pack + write of the FINISHED block move to a background
                    # pipeline thread: they run on this block's (now idle)
                    # ping-pong home while the loop advances to the next
                    # block's tune on the other group. Snapshots of the next
                    # block's inputs are captured NOW -- the main loop mutates
                    # calib_state as soon as it advances.
                    _q_snap = calib_state.get("q_inputs") if calib_state is not None else None
                    _fp_snap = calib_state["fp_inputs"] if calib_state is not None else None
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
                if not _bg_pack_eligible and self.compress_context.is_immediate_saving:
                    _t_write = _time.perf_counter()
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
                    _t_write = _time.perf_counter() - _t_write
                    _t_snap = 0.0
                    if rs is not None:
                        # crash-durability contract, same as the serial path:
                        # the manifest may claim this block done only after its
                        # tensors are durably flushed to a shard file
                        self.shard_writer._flush_shard()
                        _t0 = _time.perf_counter()
                        is_model_last = g_idx == len(all_blocks) - 1 and k_idx == len(block_names) - 1
                        rs.mark_block_done(
                            block_name,
                            calib_state.get("q_inputs") if calib_state is not None else None,
                            (None if is_model_last else calib_state["fp_inputs"]) if calib_state is not None else None,
                        )
                        _t_snap = _time.perf_counter() - _t0
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

                if logger.isEnabledFor(logging.DEBUG):
                    # post-tune sample: catches the during-tuning transient
                    # peak via VmHWM that the pre-tune snapshot misses
                    self._log_device_inventory(calib_state, f"block {stream_block_idx} post")
                if _peak_watch is not None:
                    _peak_watch.set_phase("write")
                    _peak_watch.log(f"block {stream_block_idx}")
                    _peak_watch.reset_run_max()
                clear_memory()
                self._trim_host_heap()
                memory_monitor.log_summary()
                stream_block_idx += 1  # consumed a staging slot: rotate the round-robin home
                pbar.update(1)
            if _peak_watch is not None:
                _peak_watch.stop()
            blocks_before += len(block_names)

        # Pipeline lifecycle: model-level teardown (also finalizes rotation)
        if _bg_pack is not None:
            # final save/index write below reads the ShardWriter state; the
            # last block's pack pipeline must be complete first
            self._join_bg_pack(_bg_pack)
            _bg_pack = None
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
            remain_layer_names.append(n)
        for name in remain_layer_names:
            logger.info(f"Quantizing remaining layer {name} on CPU.")
            if streamer is not None:
                # load the layer itself; streaming its parent prefix would
                # materialize the parent's whole subtree (every block weight)
                streamer.load_module_(get_module(self.model, name), name, device="cpu")
            self.alg_composer.compress_layer_outside_block(get_module(self.model, name))
            # Outside-block layers (embed_tokens/lm_head/etc.) are typically few so just
            # log a summary after each one.
            clear_memory()
            memory_monitor.log_summary()

        # Convert remaining fp8
        convert_module_to_hp_if_necessary(self.model, self.amp_dtype, self.device)
        if self.compress_context.low_cpu_mem_usage and streamer is None:
            self._offloader.reload(self.model)
        if streamer is not None and self.compress_context.is_immediate_saving:
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
                if pname not in streamer.weight_map:
                    logger.warning(f"[stream] root tensor {pname} missing from checkpoint; skipped")
                    continue
                streamer.fetch_into_(self.model, pname)
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
            logger.info(
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
                if tensor.device.type != "meta" or pname not in streamer.weight_map:
                    continue
                if any(pname == q or pname.startswith(q + ".") for q in quantized_prefixes):
                    continue
                streamer.fetch_into_(self.model, pname)

            # computed buffers (rotary inv_freq & friends) never appear in a
            # checkpoint; rebuild them so the model is not mixed meta/real at
            # export time (mixed meta makes packing silently skip everything)
            from auto_round.utils.streaming_calibration import materialize_residual_meta

            materialize_residual_meta(self.model, self.model_context.model.config, torch.device("cpu"))

            # Block groups that exist only in the checkpoint (e.g. an MTP
            # layer, which the modeling code does not instantiate) have no module
            # to quantize or write; pass their tensors through verbatim so the
            # export stays complete. The scan is anchored on the model's own
            # block list (any block-path spelling), not on a hard-coded prefix.
            claimed_blocks = {b for block in all_blocks for b in block}
            parent_prefixes = {b.rsplit(".", 1)[0] for b in claimed_blocks if "." in b}
            passed = set()
            for parent in sorted(parent_prefixes):
                for n in streamer.tensor_names:
                    if not n.startswith(parent + "."):
                        continue
                    layer_id = n[len(parent) + 1 :].split(".")[0]
                    if not layer_id.isdigit():
                        continue
                    blk = f"{parent}.{layer_id}"
                    if blk in claimed_blocks or blk in passed:
                        continue
                    passed.add(blk)
                    names = streamer.names_under(blk)
                    if not names:
                        continue
                    logger.info(
                        f"[stream] {blk} has no module counterpart; "
                        f"writing {len(names)} checkpoint tensors verbatim"
                    )
                    for n2 in names:
                        self.shard_writer.save_tensor(n2, streamer.fetch(n2, raw=True))
        if self.compress_context.is_immediate_saving:
            self.shard_writer.write(is_finalize=True)

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
            if any(rs is not None and rs.resume_index > 0 for rs in resume_states) and self.shard_writer is not None:
                self.shard_writer.adopt_existing_shards()

        _mem_inv = logger.isEnabledFor(logging.DEBUG)
        _peak_watch = PeakWatcher() if logger.isEnabledFor(logging.DEBUG) else None
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
