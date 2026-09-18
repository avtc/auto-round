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
import re
import time
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
from auto_round.compressors.predictor_tree import (
    analyze_predictor_group,
    bind_predictor_forward,
    build_predictor_tree,
    checkpoint_only_roots,
    list_checkpoint_tensors,
    pick_sibling_layer,
    synthesize_predictor_e,
)
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
    to_standard_regex,
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
            # `_finalize_compress_context`, which disables `low_cpu_mem_usage` for
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
            if self.compress_context.low_cpu_mem_usage or envs.AR_DISK_STREAM_MODEL or disk_streaming:
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

            # keep the chain tail alive for tail-fed external layers (lm_head)
            # and predictor trees (MTP): the last block's fp reference and
            # quantized-chain outputs are their inputs (lm_head applies the
            # final norm; the tree consumes the raw rows)
            if getattr(self, "_tail_fed_layers_", None) or getattr(self, "_predictor_plan_", None):
                self._lm_head_chain_tail_ = (new_q_input, reference_output)

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
            from auto_round.utils.device import log_cuda_memory_census

            # phase boundary: the block loop just freed its large tuning
            # buffers; returning them to the driver keeps the allocator pool
            # compact before the (potentially huge) outside-block wrappers
            # are built, instead of reserving fragmented segments nobody can use
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_cuda_memory_census(f"outside-block loop entry {name}")
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
        # lm_head-class external layers are tail-fed: the block loop's chain
        # output (fp reference + quantized rows through the final norm) is
        # their input, so they neither join the upfront capture call nor the
        # outside-block q-capture pass - no extra whole-model forwards
        self._tail_fed_layers_ = []
        self._lm_head_chain_tail_ = None
        self._lm_head_norm_name_ = None
        lm_head_name = self._resolve_lm_head_name_(layer_names)
        if lm_head_name is not None:
            self._tail_fed_layers_ = [lm_head_name]
            self._lm_head_norm_name_ = self._discover_final_norm_(all_blocks)
        if not self.has_variable_block_shape:
            to_cache_block_names = [block[0] for block in all_blocks]
        else:
            to_cache_block_names = flatten_list(all_blocks)
        _last_cache_name = to_cache_block_names[-1] if len(to_cache_block_names) > 1 else None
        to_cache_layer_names = [n for n in layer_names if n not in self._tail_fed_layers_]
        if self.super_group_size is not None:
            to_cache_layer_names = []
        if len(layer_names) > 0:
            logger.info(
                "Starting to cache block inputs. This may be slow due to external block layers: %s", layer_names
            )
        else:
            logger.info("start to cache block inputs")
        self._prepare_predictor_tuning_(all_blocks)
        all_inputs = self.cache_data(
            to_cache_block_names,
            self.calibration_context.nsamples,
            to_cache_layer_names,
            last_cache_name=_last_cache_name,
        )
        # Raw token IDs from the tokenizer, cached during calibration for use in quantize_block.
        input_ids_cache = all_inputs.pop("input_ids", None)
        self.inputs = all_inputs
        self._snapshot_predictor_aux_(all_inputs, to_cache_block_names)

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
            )
            resume_states = []
            for group_idx, block_names in enumerate(all_blocks):
                sig = compute_run_signature(
                    model_dir,
                    scheme_desc,
                    dataset_desc,
                    self.calibration_context.nsamples,
                    self.calibration_context.seqlen,
                    block_names,
                )
                resume_states.append(
                    ResumeState(os.path.join(envs.AR_RESUME_DIR, f"group_{group_idx}"), sig, block_names)
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

        # tail-fed layers (lm_head) receive their inputs from the block loop's
        # chain output through the final norm - no input-capture entries
        #
        # the predictor trees also consume the raw chain tail: park it on the
        # host FIRST so the lm_head tune below and the tree tune afterwards
        # both read host-resident rows (the raw tail's ~2x rows on the tune
        # device would collide with the wrapper's value/grad buffers)
        if getattr(self, "_predictor_plan_", None) is not None:
            tail = getattr(self, "_lm_head_chain_tail_", None)
            if tail is not None:
                q_rows, fp_rows = tail
                q_rows = self._chain_hidden_rows(q_rows) if q_rows is not None else None
                fp_rows = self._chain_hidden_rows(fp_rows)
                if fp_rows is not None:
                    self._lm_head_chain_tail_ = (
                        [r.detach().to("cpu", copy=True) for r in q_rows] if q_rows is not None else None,
                        [r.detach().to("cpu", copy=True) for r in fp_rows],
                    )
                # drop every alias to the pre-park rows: the locals kept the
                # raw device-side storages alive through the whole lane otherwise
                del tail, q_rows, fp_rows
                clear_memory()
        tail_inputs = {}
        for tail_name in list(getattr(self, "_tail_fed_layers_", []) or []):
            if tail_name not in layer_names:
                continue
            derived = self._lm_head_tail_inputs_(tail_name)
            if derived is not None:
                tail_inputs[tail_name] = derived
                self._attach_tail_imatrix_(tail_name, derived[0])
        if tail_inputs:
            layer_inputs = dict(layer_inputs)
            for tail_name, (fp_rows, _q_rows) in tail_inputs.items():
                layer_inputs[tail_name] = fp_rows
            # when no predictor tree needs the raw tail, release it now so its
            # residency does not collide with the wrapper's value/grad buffers;
            # with a plan, the tree stage consumes it and releases afterwards
            if getattr(self, "_predictor_plan_", None) is None:
                self._lm_head_chain_tail_ = None

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
            self._tune_predictor_trees_(token_ids)
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
            capture_names = [n for n in layer_names if n not in tail_inputs]
            # the predictor q-tail rides this pass too - keep it when a tree is active
            if capture_names:
                logger.info("starting to cache layer inputs for %s, this may be quite slow ", capture_names)
                q_layer_inputs = self.cache_data([], self.calibration_context.nsamples, layer_names=capture_names)
            else:
                logger.info("outside-block layer inputs come from the calibration chain tail; no extra pass")
            if hasattr(self.model, "hf_device_map") and len(self.model.hf_device_map) > 1:
                accelerate.hooks.remove_hook_from_submodules(
                    self.model
                )  # self.model.hf_device_map has not been changed
        if not self.compress_context.is_immediate_saving:
            self.model = mv_module_from_gpu(self.model)
        clear_memory()
        for layer_name in layer_names:
            if layer_name in tail_inputs:
                # tail rows are parked on host by design and the tune loop
                # streams them per micro-batch; pulling the whole set onto
                # cache_device is the capture-path contract, not ours
                layer_input = layer_inputs[layer_name]
            else:
                layer_input = to_device(layer_inputs[layer_name], self.compress_context.cache_device)
            if layer_name in tail_inputs:
                q_layer_input = tail_inputs[layer_name][1]
            else:
                q_layer_input = q_layer_inputs.get(layer_name, None) if q_layer_inputs is not None else None
                q_layer_input = to_device(q_layer_input, self.compress_context.cache_device)
            try:
                self.alg_composer.compress_layer_outside_block(
                    get_module(self.model, layer_name),
                    fp_inputs=layer_input,
                    q_inputs=q_layer_input,
                    input_ids=token_ids,
                )
            except Exception as e:  # pylint: disable=broad-except
                # containment: a failed tune leaves the layer unquantized and
                # the export pass (missing-tensors) completes the artifact,
                # instead of losing a run whose blocks are already tuned and
                # streamed
                logger.warning(
                    "outside-block layer %s tuning failed (%s: %s); leaving it to the export path",
                    layer_name,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                del layer_input
                clear_memory(q_layer_input)
                clear_memory()
                continue
            if self.compress_context.is_immediate_packing:
                immediate_pack(layer_name, self.layer_config)

            if self.compress_context.is_immediate_saving:
                m = get_module(self.model, layer_name)
                self.shard_writer.write(m, name=layer_name, is_finalize=False)
            del layer_input
            clear_memory(q_layer_input)
            memory_monitor.log_summary()

        self._tune_predictor_trees_(token_ids)

    # ── Predictor-tree (MTP) tuning ─────────────────────────────────────

    def _prepare_predictor_tuning_(self, all_blocks) -> None:
        """Discover pinned checkpoint-only predictor trees before calibration.

        Reads checkpoint metadata once (tensor names/shapes only); the fp/q
        chain tails come from the block loop's stored last-block output (the
        same rows a final-norm pre-hook would see - see
        ``_tune_predictor_trees_``). No-op - and zero cost - when the model
        has no checkpoint-only groups or no local checkpoint directory.
        """
        self._predictor_plan_ = None
        self._predictor_fp_tail_ = None
        self._predictor_q_tail_ = None
        self._predictor_tree_aux_ = None
        cfg = getattr(self.model_context.model, "config", None)
        source_dir = getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None)
        if not source_dir or not os.path.isdir(source_dir):
            return
        ckpt = list_checkpoint_tensors(source_dir)
        roots = checkpoint_only_roots(ckpt, self.model_context.model)
        if not roots:
            return
        root_tensor_names = [n for r in roots for n in ckpt if n.startswith(r + ".")]
        norm_name = self._discover_final_norm_(all_blocks)
        if norm_name is None:
            logger.info(
                "checkpoint-only groups %s found but no final norm discovered; trees stay on the export path",
                ", ".join(roots),
            )
            return
        self._predictor_plan_ = {"ckpt": ckpt, "roots": roots, "norm": norm_name}
        if not self._pins_for_tree_(root_tensor_names):
            logger.info("predictor trees %s follow the run's schema (no layer-config pins)", ", ".join(roots))

    def _discover_final_norm_(self, all_blocks) -> Optional[str]:
        """Name-agnostic final-norm discovery: the last block-external norm-like leaf.

        Accepts any leaf whose parameters are ALL 1-D (RMSNorm has one,
        LayerNorm-with-bias has two - GPT-J/OPT-class models) and whose name
        says norm/ln_f; lm_head-class projections carry a 2-D weight and never
        match."""
        block_prefixes = [name for block in all_blocks for name in block]
        best = None
        for name, m in self.model_context.model.named_modules():
            if not name:
                continue
            if any(name == b or name.startswith(b + ".") for b in block_prefixes):
                continue
            if list(m.children()):
                continue
            params = list(m.parameters())
            if params and all(p.dim() == 1 for p in params):
                low = name.lower()
                if "norm" in low or "ln_f" in low:
                    best = name
        return best

    def _pins_for_tree_(self, tree_tensor_names) -> dict:
        """Layer-config entries (exact or regex) matching tree tensor paths.

        Pins for checkpoint-only tensors are popped from ``layer_config`` into
        ``regex_config`` by the resolver (the modules did not exist at resolve
        time), so BOTH sources must be consulted or the predictor stage would
        silently never activate. Exact in-model entries take precedence.
        """
        sources = {}
        sources.update(getattr(self, "regex_config", None) or {})
        sources.update(dict(self.layer_config))
        pins = {}
        for key, cfg in sources.items():
            if not isinstance(cfg, dict) or not check_to_quantized(cfg):
                continue
            regex = re.compile(to_standard_regex(key))
            if key in tree_tensor_names or any(regex.search(n) for n in tree_tensor_names):
                pins[key] = cfg
        return pins

    def _stamp_pin_(self, module, name: str, pin: dict) -> None:
        # same field set as the canonical plan boundary (apply_plan_to_model)
        # so future scheme fields reach predictor-tree modules too
        from dataclasses import fields as dc_fields

        from auto_round.schemes import QuantizationScheme

        scheme_keys = tuple(f.name for f in dc_fields(QuantizationScheme)) + ("scale_dtype",)
        for attr in scheme_keys:
            # ALWAYS set the attribute: readers (the row-blocked wrapper reads
            # orig_layer.scale_dtype) require it to exist; None is a valid
            # value they resolve to defaults, mirroring apply_plan_to_model
            setattr(module, attr, pin.get(attr))
        # explicit None must not defeat the defaults (AutoScheme emits None);
        # same idiom as the base plan application
        module.act_bits = pin.get("act_bits") or 16
        module.act_sym = pin.get("act_sym") if pin.get("act_sym") is not None else True
        module.act_data_type = pin.get("act_data_type") or None
        module.global_name = name

    def _detach_tree_(self, group: str) -> None:
        """Remove a (possibly partial) predictor tree from the model.

        Attached tree tensors would satisfy the export missing-tensors check,
        silently disabling the verbatim copy the export path would otherwise
        perform for the group.
        """
        segs = group.split(".")
        try:
            parent = (
                self.model_context.model.get_submodule(".".join(segs[:-1]))
                if len(segs) > 1
                else self.model_context.model
            )
        except AttributeError:
            return
        parent._modules.pop(segs[-1], None)

    def _attach_pinned_tree_(self, group, ckpt, info, pins, all_blocks, source_dir):
        """Materialize one predictor tree and stamp pins on its Linears.

        Returns ``(shell, pinned_module_names, claimed_tensor_names)`` or None
        (with a warning logged) when no sibling covers the tree, the build
        fails, or the pins match no Linear.
        """
        picked = pick_sibling_layer(self.model_context.model, ckpt, info, all_blocks)
        if picked is None:
            logger.warning("predictor tree %s skipped: no sibling layer covers its tensors", group)
            return None
        sibling_name, sibling = picked
        try:
            claimed = build_predictor_tree(self.model_context.model, source_dir, ckpt, info, sibling)
        except Exception as e:  # degrade, never die: the export path still handles the group
            logger.warning("predictor tree %s skipped: %s: %s", group, type(e).__name__, e)
            self._detach_tree_(group)
            logger.info("detached partially materialized predictor tree %s", group)
            return None
        logger.info("tuning predictor tree %s (layer from sibling %s)", group, sibling_name)
        shell = self.model_context.model.get_submodule(group)
        # canonical layer-config resolution on the attached tree: user pins
        # resolve with the repo's own precedence and EVERY entry carries the
        # full scheme field set - the same dict the export writes into
        # quantization_config; unpinned tree modules resolve to the run's
        # scheme
        resolved = self._resolve_tree_pins_(group, pins)
        pinned = []
        for n, m in shell.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            module_name = f"{group}.{n}" if n else group
            entry = resolved.get(module_name)
            if entry is None:
                continue
            self._stamp_pin_(m, module_name, entry)
            self.layer_config[module_name] = entry
            pinned.append(module_name)
        if not pinned:
            logger.warning(
                "predictor tree %s: pins %s matched no Linear; leaving to the export pass", group, list(pins)
            )
            self._detach_tree_(group)
            return None
        return shell, pinned, claimed, resolved

    def _resolve_tree_pins_(self, group, user_pins) -> dict:
        """Canonical layer-config resolution over the attached tree modules.

        Reuses the compressor's resolver (``resolve_layer_config``): user
        pins expand and resolve with the repo's own precedence, and every
        entry carries the full scheme field set - the same dict registered in
        ``layer_config`` and exported into ``quantization_config``. Unpinned
        tree modules resolve to the run's scheme defaults.
        """
        from auto_round.compressors.config_resolution.contracts import ResolvedScheme
        from auto_round.compressors.layer_config_resolver import _resolve_layer_config_presets, resolve_layer_config

        # outside-block modules are opt-in in the resolver (entries appear only
        # when the user pins them); the tree stage's policy is schema-follow,
        # so seed every tree Linear with the canonical scheme default and let
        # user pins win through the resolver's own expansion/precedence
        _, default_dict, _, _ = _resolve_layer_config_presets(
            {}, self.model_context.model, self.ignore_layers, self.scheme_context, self.scale_dtype, True
        )
        tree_defaults = {}
        for n, mod in self.model_context.model.named_modules():
            if isinstance(mod, torch.nn.Linear) and (n == group or n.startswith(group + ".")):
                tree_defaults[n] = copy.deepcopy(default_dict)
        merged = {**tree_defaults, **{k: dict(v) for k, v in (user_pins or {}).items()}}
        resolved = resolve_layer_config(
            model=self.model_context.model,
            scheme=ResolvedScheme.from_scheme(self.scheme_context),
            layer_config=merged,
            scale_dtype=self.scale_dtype,
            supported_types=self.supported_types,
            inner_supported_types=self.inner_supported_types,
            quant_block_list=None,
            ignore_layers=self.ignore_layers,
            quant_lm_head=False,
            enable_gguf_official_mixed=False,
            is_mllm=self.model_context.is_mllm,
            format=self._formats_policy_string(),
        )
        return dict(resolved)

    def _tune_tree_heads_(
        self,
        group,
        ckpt,
        info,
        user_pins,
        resolved_pins,
        claimed,
        source_dir,
        new_q_output,
        reference_output,
        token_ids,
    ):
        """Tune remaining 2-D tensors (e.g. the vocab head) on the tree's outputs.

        Head modules are created first, then the canonical resolver runs once
        over the now-complete module set (user pins with the repo precedence;
        schema defaults for the rest), and each created head tunes on the
        tree's fp/q outputs with its resolved entry registered for export.
        """
        role_names = {info[r] for r in ("fc", "norm_e", "norm_h", "final_norm") if info.get(r)}
        heads = []
        for n in (n for n in ckpt if n.startswith(group + ".")):
            if n in claimed or n in role_names or not n.endswith(".weight"):
                continue
            shape = ckpt[n][0]
            if len(shape) != 2:
                continue
            path = n[: -len(".weight")]
            try:
                self.model_context.model.get_submodule(path)
            except AttributeError:
                head = torch.nn.Linear(int(shape[1]), int(shape[0]), bias=False)
                from auto_round.compressors.predictor_tree import ensure_module_path, load_checkpoint_tensor

                parent = ensure_module_path(self.model_context.model, path)
                parent.add_module(path.rsplit(".", 1)[-1], head)
                with torch.no_grad():
                    head.weight.data.copy_(load_checkpoint_tensor(source_dir, ckpt, n))
            heads.append(path)
        if not heads:
            return
        resolved = self._resolve_tree_pins_(group, user_pins)
        for path in heads:
            entry = resolved.get(path)
            if entry is None:
                continue
            if entry.get("data_type") == "float" or int(entry.get("bits") or 0) >= 16:
                continue  # full-precision pin: leave the tensor verbatim for the export
            mod = self.model_context.model.get_submodule(path)
            self._stamp_pin_(mod, path, entry)
            self.layer_config[path] = entry
            logger.info("tuning predictor tensor %s on the tree outputs", path)
            self.alg_composer.compress_layer_outside_block(
                mod,
                fp_inputs=reference_output,
                q_inputs=new_q_output,
                input_ids=token_ids,
            )
            if self.compress_context.is_immediate_packing:
                immediate_pack(path, self.layer_config)

    @staticmethod
    def _chain_hidden_rows(chain_state):
        """A chain input/output as a plain list of per-sample row tensors.

        The chain keeps rows as a list, or a dict of per-key row lists for
        block classes with structured outputs (e.g. gated-delta-net): take its
        ``hidden_states`` rows."""
        rows = chain_state.get("hidden_states") if isinstance(chain_state, dict) else chain_state
        if isinstance(rows, dict):
            rows = next(iter(rows.values()))
        return rows

    def _resolve_lm_head_name_(self, layer_names) -> Optional[str]:
        """lm_head's module name from the outside-block plan, or ``None``."""
        if not layer_names:
            return None
        candidates = [n for n in layer_names if n == "lm_head" or n.rsplit(".", 1)[-1] == "lm_head"]
        if not candidates:
            candidates = [n for n in layer_names if "lm_head" in n]
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                "multiple lm_head candidates in the outside-block plan %s; tuning %s", candidates, candidates[0]
            )
        return candidates[0]

    def _quantizer_requests_q_inputs_(self) -> bool:
        """Whether the active quantizer(s) maintain the quantized-input chain.

        SignRound defaults ``enable_quanted_input`` to True, RTN (iters=0)
        defaults to False; drives the log level of the FP-only tail fallback.
        """
        quantizers = getattr(self.alg_composer, "block_quantizer", None)
        if quantizers is None:
            return False
        if not isinstance(quantizers, (list, tuple)):
            quantizers = [quantizers]
        return any(bool(getattr(q, "enable_quanted_input", False)) for q in quantizers)

    def _attach_tail_imatrix_(self, lm_head_name, fp_rows) -> None:
        """Attach the fp-input imatrix for lm_head from the chain tail rows.

        In the capture path this statistic was accumulated by the quantizer's
        fp-input forward hook while the collection walk executed lm_head; with
        the single-block-target early-stop the walk never reaches lm_head, so
        the same math (fp32 column sums of squares over all token rows, plus
        the row count for the RTN normalization) runs directly over the tail
        rows - the identical inputs the hook would have seen. Never overwrites
        an existing statistic.
        """
        module = get_module(self.model_context.model, lm_head_name)
        if module is None or hasattr(module, "imatrix"):
            return
        if not fp_rows:
            return
        total = None
        count = 0
        for row in fp_rows:
            flattened = row.reshape(-1, row.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)
            total = squared if total is None else total + squared.to(total.device)
            count += flattened.shape[0]
        module.imatrix = total
        module.imatrix_cnt = count
        logger.info("[lm_head] attached the fp-input imatrix for %s from %d chain-tail rows", lm_head_name, count)

    def _lm_head_tail_inputs_(self, lm_head_name):
        """``(fp_rows, q_rows)`` for lm_head from the calibration chain tail.

        The block loop's chain output is the RAW last-block output; lm_head
        consumes POST-final-norm states, so the final norm is applied to the
        rows (its weights stream in when still meta). Returns ``None`` on any
        mismatch - the caller then keeps the closed-form path for the layer.
        """
        tail = getattr(self, "_lm_head_chain_tail_", None)
        if tail is None:
            logger.warning("[lm_head] %s keeps the input-capture path: the block loop kept no chain tail", lm_head_name)
            return None
        new_q_output, reference_output = tail
        fp_rows = self._chain_hidden_rows(reference_output)
        if (
            not isinstance(fp_rows, (list, tuple))
            or len(fp_rows) == 0
            or not all(isinstance(r, torch.Tensor) for r in fp_rows)
        ):
            logger.warning(
                "[lm_head] %s keeps the input-capture path: unexpected chain-tail row format (%s)",
                lm_head_name,
                type(fp_rows).__name__,
            )
            return None
        q_rows = self._chain_hidden_rows(new_q_output) if new_q_output is not None else None
        q_requested = self._quantizer_requests_q_inputs_()
        if not isinstance(q_rows, (list, tuple)) or len(q_rows) != len(fp_rows):
            if q_requested:
                logger.warning(
                    "[lm_head] %s tunes on FP chain inputs (enable_quanted_input cannot be honored): "
                    "quantized chain rows are missing or malformed",
                    lm_head_name,
                )
            else:
                # RTN (iters=0) defaults enable_quanted_input to False: the q chain
                # was never maintained by configuration, so this is the expected
                # path, not a degradation - and the search ignores q rows anyway.
                logger.info(
                    "[lm_head] %s tunes on FP chain inputs (quantized-input chain disabled by config)",
                    lm_head_name,
                )
            q_rows = None
        norm_name = getattr(self, "_lm_head_norm_name_", None)
        norm_mod = get_module(self.model_context.model, norm_name) if norm_name else None
        if norm_mod is None:
            logger.warning(
                "[lm_head] %s keeps the input-capture path: cannot locate the final norm that feeds it",
                lm_head_name,
            )
            return None
        # a wrongly picked leaf is detectable: the real final norm scales the
        # hidden dim lm_head consumes
        in_features = getattr(get_module(self.model_context.model, lm_head_name), "in_features", None)
        if in_features is not None and norm_mod.weight.numel() != in_features:
            logger.warning(
                "[lm_head] %s keeps the input-capture path: candidate final norm %s does not match lm_head's "
                "input width (%d vs %d)",
                lm_head_name,
                norm_name,
                norm_mod.weight.numel(),
                in_features,
            )
            return None
        if any(p.is_meta for p in norm_mod.parameters()):
            offloader = getattr(self, "_offloader", None)
            if offloader is None:
                logger.warning(
                    "[lm_head] %s keeps the input-capture path: the final norm is still meta and no offloader "
                    "is available to load it",
                    lm_head_name,
                )
                return None
            offloader.reload(self.model_context.model, norm_name)
            materialize_model_(norm_mod)
        with torch.no_grad():
            # Compute on the norm's device (its params stay put); results park
            # on the HOST: the tune loop streams rows per micro-batch
            # (torch.cat(...).to(device)), so keeping the whole set resident on
            # cache_device only collides with the wrapper's value/grad buffers
            # on tight GPUs. Only one per-sample transient is away from host
            # at a time.
            ndev, dt = norm_mod.weight.device, norm_mod.weight.dtype
            fp_rows = [norm_mod(r.to(ndev).to(dt)).to("cpu") for r in fp_rows]
            if q_rows is not None:
                q_rows = [norm_mod(r.to(ndev).to(dt)).to("cpu") for r in q_rows]
        return fp_rows, q_rows

    def _snapshot_predictor_aux_(self, all_inputs, to_cache_block_names) -> None:
        """Keep the last block group's auxiliary inputs (attention mask,
        position ids) for the predictor-tree stage.

        Excludes the block's primary row key under BOTH spellings: entries are
        cached as "hidden_states" and renamed to "input_ids" only later -
        leaking either into aux would override the tree's true hidden input.
        Values are shared list references that survive the block loop's pops.
        """
        self._predictor_tree_aux_ = None
        last_first = to_cache_block_names[-1] if to_cache_block_names else None
        entry = all_inputs.get(last_first) if last_first else None
        if isinstance(entry, dict):
            self._predictor_tree_aux_ = {k: v for k, v in entry.items() if k not in ("input_ids", "hidden_states")}

    def _tune_predictor_trees_(self, token_ids) -> None:
        """Materialize and tune pinned checkpoint-only predictor trees (MTP).

        Runs after the block loop and the outside-block layer loop: fp/q tails
        were captured with the final-norm hook; the tree joins as one extra
        block (fp reference vs quantized chain), then any remaining pinned 2-D
        tensors (e.g. the vocabulary head) tune layer-wise on the tree's
        outputs. Trees are attached under checkpoint-spelled paths so pins,
        packing, saving and the missing-tensors pass see checkpoint names.
        """
        plan = getattr(self, "_predictor_plan_", None)
        if plan is None:
            return
        try:
            self._tune_predictor_trees_impl_(token_ids, plan)
        finally:
            # the raw chain tail fed both this stage and any tail-fed external
            # layers; nothing needs it after the trees
            self._lm_head_chain_tail_ = None

    def _tune_predictor_trees_impl_(self, token_ids, plan) -> None:
        """Body of :meth:`_tune_predictor_trees_` (plan already resolved)."""
        ckpt, roots, norm_name = plan["ckpt"], plan["roots"], plan["norm"]
        # fp/q tails come from the block loop's stored chain tail: the raw
        # last-block outputs are exactly the rows a final-norm pre-hook would
        # have captured (the norm's input), so no extra calibration pass and
        # no early-stop override are needed anywhere
        tail = getattr(self, "_lm_head_chain_tail_", None)
        fp_tail = q_tail = None
        if tail is not None:
            q_out, ref_out = tail
            fp_tail = self._chain_hidden_rows(ref_out)
            q_tail = self._chain_hidden_rows(q_out) if q_out is not None else None
            self._predictor_fp_tail_ = fp_tail
            self._predictor_q_tail_ = q_tail
        if not fp_tail:
            logger.warning("predictor trees %s stay on the export path: no chain tail kept by the block loop", roots)
            return
        cfg = getattr(self.model_context.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        hidden = getattr(text_cfg, "hidden_size", None)
        embed_getter = getattr(self.model_context.model, "get_input_embeddings", None)
        embed = embed_getter() if callable(embed_getter) else None
        if token_ids and embed is not None:
            e_rows = [synthesize_predictor_e(ids, embed=embed) for ids in token_ids]
        else:
            e_rows = None
        if e_rows is None:
            logger.warning("predictor trees %s skipped: no cached token ids for the e-side input", roots)
            return
        all_blocks = get_block_names(self.model_context.model)
        source_dir = getattr(cfg, "name_or_path", None) or getattr(cfg, "_name_or_path", None)
        # the tree stage follows the lm_head tune on the same device; reclaim
        # its allocator cache first so the tree's tuning state and activations
        # start from a defragmented pool. The streaming lane gets this for
        # free (every prior block is parked to meta and cleared per group) -
        # the data-driven lane needs the explicit clear.
        clear_memory()
        for group in roots:
            info = analyze_predictor_group(ckpt, group, hidden)
            if info is None:
                logger.info("checkpoint-only group %s is not a predictor tree; leaving it to the export pass", group)
                continue
            names_under = [n for n in ckpt if n.startswith(group + ".")]
            pins = self._pins_for_tree_(names_under)
            if not pins:
                logger.info("predictor tree %s follows the run's schema (no layer-config pins)", group)
            # containment: any failure from here through the tune leaves the
            # tree to the export pass (missing-tensors covers whatever this
            # run leaves unquantized) instead of killing the run after every
            # block has already been tuned and streamed
            try:
                self._tune_one_predictor_tree_(
                    group, ckpt, info, pins, all_blocks, source_dir, fp_tail, token_ids, e_rows
                )
            except Exception as e:  # pylint: disable=broad-except
                if "OutOfMemory" in type(e).__name__:
                    # the only vantage that sees the working set that failed
                    from auto_round.utils.device import log_cuda_memory_census

                    log_cuda_memory_census(f"predictor tree {group} OOM (at failure)", device_manager.device)
                logger.warning(
                    "predictor tree %s tuning failed (%s: %s); leaving it to the export path",
                    group,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                try:
                    self._detach_tree_(group)
                except Exception:  # pylint: disable=broad-except
                    pass
                clear_memory()
                continue

    def _tune_one_predictor_tree_(
        self, group, ckpt, info, pins, all_blocks, source_dir, fp_tail, token_ids, e_rows
    ) -> None:
        """Attach, tune, and pack one predictor tree (failures handled by the caller)."""
        attached = self._attach_pinned_tree_(group, ckpt, info, pins, all_blocks, source_dir)
        if attached is None:
            return
        shell, pinned, claimed, resolved_pins = attached
        bind_predictor_forward(
            shell,
            {
                "norm_e": info["norm_e"][: -len(".weight")],
                "norm_h": info["norm_h"][: -len(".weight")],
                "fc": info["fc"][: -len(".weight")],
                "layer": info["layer_root"],
                "final_norm": info["final_norm"][: -len(".weight")] if info.get("final_norm") else None,
            },
            e=e_rows[0],
            model=self.model_context.model,
        )
        aux = dict(self._predictor_tree_aux_ or {})
        aux["_predictor_e"] = e_rows
        _, input_others = self._preprocess_block_inputs({"input_ids": fp_tail, **aux})
        from auto_round.algorithms.composer import BlockContext

        # same infrastructure the block loop applies: materialize meta
        # tensors, honor the amp dtype policy, and place the tree on the
        # tuning device (compress_block treats placement as caller's job)
        materialize_model_(shell)
        convert_module_to_hp_if_necessary(shell, self.model_context.amp_dtype, device_manager.device)
        shell = self.alg_composer.dispatch_block(shell, fp_tail, input_others)
        # mirrors the lm_head lane's wrapper-ready census (quantizer.py,
        # quantize_layer_outside_block): the tree block tunes via compress_block,
        # which carries no census of its own - this is the last clean vantage
        # before the tune's value/grad/activation allocations begin. Header
        # only: the tensor-list walk runs in the failure handler
        from auto_round.utils.device import log_cuda_memory_census

        log_cuda_memory_census(f"predictor tree ready {group}", device_manager.device, walk=False)
        ctx = BlockContext(
            model=self.model_context.model,
            block_names=[group],
            block_name=group,
            block_index=len(all_blocks),
            bs=self.calibration_context.batch_size,
            block_cnt=len(all_blocks) + 1,
        )
        new_q_output, reference_output = self.alg_composer.compress_block(
            shell,
            fp_tail,
            input_others,
            block_ctx=ctx,
            q_inputs=self._predictor_q_tail_,
            input_ids=token_ids,
        )
        self._tune_tree_heads_(
            group, ckpt, info, pins, resolved_pins, claimed, source_dir, new_q_output, reference_output, token_ids
        )
        if self.compress_context.is_immediate_packing:
            for module_name in pinned:
                immediate_pack(module_name, self.layer_config)
        clear_memory()

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
            from auto_round.utils.model import move_to_device_preserving_cpu_pinned, pin_ngram_embeddings_on_cpu_

            pin_ngram_embeddings_on_cpu_(block)
            block = move_to_device_preserving_cpu_pinned(block, device)

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
                from auto_round.utils.model import (
                    move_to_device_preserving_cpu_pinned,
                    place_ngram_embeddings_for_tuning_,
                )

                place_ngram_embeddings_for_tuning_(block)
                block = move_to_device_preserving_cpu_pinned(block, device)
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
