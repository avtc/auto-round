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

import re

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
from auto_round.utils.device_manager import device_manager

_EXPERT_RE = re.compile(r"^(?P<parent>.*)\.experts\.\d+\.(?P<proj>[^.]+)$")

from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
)


def register_imatrix_hooks(model, *, with_count: bool = False):
    """Attach forward hooks collecting per-module imatrix statistics.

    Each hooked module accumulates ``module.imatrix`` (fp32 sum of squared
    input activations per input channel) and ``module.imatrix_cnt`` (rows
    seen). Normalization (division by the count) is deferred to the consumer.
    Returns the hook handles.
    """

    def collect_imatrix(module, input, output):
        input = input[0] if isinstance(input, (tuple, list)) else input
        flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
        squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

        if not hasattr(module, "imatrix"):
            module.imatrix = squared
            if with_count:
                module.imatrix_cnt = input.shape[0]
            return
        module.imatrix += squared.to(module.imatrix.device)
        if with_count:
            module.imatrix_cnt += input.shape[0]

    handles = []
    for _, module in model.named_modules():
        if check_to_quantized(module):
            handles.append(module.register_forward_hook(collect_imatrix))
    return handles


@register_pipeline_member(RTNConfig)
class RTNQuantizer(BaseQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)

    @torch.no_grad()
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
        """Apply zero-shot RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (unused in RTN).
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Empty dict — zero-shot RTN has no tunable parameters to track.
        """

        for _name, m in block.named_modules():
            if check_to_quantized(m):
                self._quantize_layer_via_rtn(m, disable_opt_rtn=True)
        return {}


@register_pipeline_member(OptimizedRTNConfig)
class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        super().__init__(config)

    def can_compile_block_forward(self):
        return False

    def register_fp_input_forward_hooks(self, block):
        """Register FP-input hooks: imatrix."""
        handles = super().register_fp_input_forward_hooks(block)
        handles.extend(self._register_imatrix_hooks(block, with_count=True))
        return handles

    def _register_imatrix_hooks(self, model, *, with_count: bool = False):
        return register_imatrix_hooks(model, with_count=with_count)

    @torch.no_grad()
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
    ):
        """Apply imatrix-informed RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (unused in RTN).
            **kwargs: Reserved for forward-compatibility with future parameters.
        """
        # Normalize imatrix (cheap elementwise divides), then quantize the
        # block's target modules - fanned out over GPUs when several exist
        targets = []
        for name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if hasattr(m, "global_name") and check_to_quantized(m):
                targets.append(m)
        self._quantize_targets(targets)

    def _split_expert_batches(self, targets: list, n_jobs_hint: int = 0):
        """Partition same-shape expert projections into batchable groups.

        Experts of one layer share weight shapes (e.g. 192 x ``[1536, 4096]`
        gate/up/down projections). The per-group search is row-independent, so
        a whole group can be quantized in one call by stacking weights along
        the output dim - same results as per-module calls, one search's worth
        of per-module overhead instead of 192.

        With ``n_jobs_hint`` > number of groups, each group is split into
        roughly equal chunks so the multi-GPU fan-out fills every device
        (three monolithic groups would otherwise pin the search to three
        GPUs no matter how many are visible). Chunking keeps results
        bit-identical: rows are independent.
        """
        grouped = {}
        singles = []
        for m in targets:
            match = _EXPERT_RE.match(getattr(m, "global_name", "") or "")
            if (
                match is None
                or type(m) is not nn.Linear
                or bool(getattr(m, "sym", False))
                or not isinstance(getattr(m, "group_size", None), int)
                or getattr(m, "group_size", None) <= 0
                or getattr(m, "super_bits", None) is not None
                or getattr(m, "data_type", "int") != "int"
            ):
                singles.append(m)
                continue
            key = (match["parent"], match["proj"], tuple(m.weight.shape), m.bits, m.group_size)
            grouped.setdefault(key, []).append(m)
        batches = [g for g in grouped.values() if len(g) >= 2]
        if n_jobs_hint > len(batches) > 0:
            import math

            chunks_per_group = max(1, round(n_jobs_hint / len(batches)))
            split = []
            for g in batches:
                size = math.ceil(len(g) / chunks_per_group)
                split.extend(g[i : i + size] for i in range(0, len(g), size))
            batches = split
        return batches, singles

    def _expert_search_active(self) -> bool:
        """Whether the optimized-RTN asym search runs for expert modules.

        Mirrors the MoE heuristic in ``_quantize_layer_via_rtn``: experts skip
        the search unless explicitly forced on (``enable_opt_rtn``), so batching
        must not change that decision - only batch when the search would run.
        """
        cfg = self.config
        if bool(getattr(cfg, "disable_opt_rtn", False)):
            return False
        if getattr(cfg, "orig_disable_opt_rtn", None) is None and getattr(self.model_context, "is_moe_model", False):
            return False
        return getattr(cfg, "asym_search", "auto") != "minmax"

    def _quantize_expert_batch(self, mods: list, device) -> bool:
        """Quantize same-shape expert projections in one stacked search call.

        Returns ``False`` when the batch cannot be handled here (caller falls
        back to per-module tuning). Reproduces the per-module wrapper outputs:
        quantize-dequantized weights written in place, plus ``scale``/``zp``/
        ``q_scale_thresh`` attributes matching the unwrapper's conventions.
        """
        m0 = mods[0]
        bits, g = m0.bits, m0.group_size
        try:
            from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym
        except ImportError:
            return False

        max_elems = 2**28  # ~1 GiB fp32 stacked weights per call
        per_call = max(1, max_elems // m0.weight.numel())
        import time as _time

        phase_log = getattr(self, "_expert_phase_log", None)
        for s in range(0, len(mods), per_call):
            chunk = mods[s : s + per_call]
            dev = torch.device(device)
            t0 = _time.perf_counter()
            weights = torch.cat([m.weight.data.reshape(m.weight.shape[0], -1) for m in chunk], dim=0).to(dev)
            imat_rows = []
            for m in chunk:
                if hasattr(m, "imatrix"):
                    imat_rows.append(m.imatrix.to(dev).unsqueeze(0).expand(m.weight.shape[0], -1))
            if imat_rows and len(imat_rows) < len(chunk):  # mixed: fill gaps uniformly
                it = iter(imat_rows)
                imat_rows = [
                    next(it) if hasattr(m, "imatrix") else torch.ones(m.weight.shape, device=dev) for m in chunk
                ]
            # uniform weighting needs no materialized tensor: qw=None is
            # bit-identical to a full ones imatrix and skips ~module-size
            # allocations per expert
            imat = torch.cat(imat_rows, dim=0) if imat_rows else None
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t1 = _time.perf_counter()

            qdq, scale, zp = quant_tensor_opt_rtn_asym(
                weights,
                bits=bits,
                group_size=g,
                v=0.0,
                q_scale_thresh=1e-5,
                imatrix=imat,
                scale_dtype=getattr(m0, "scale_dtype", torch.float16),
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t2 = _time.perf_counter()
            row_w = row_g = 0
            with torch.no_grad():
                for m in chunk:
                    out = m.weight.shape[0]
                    if m.weight.shape[1] % g != 0:  # padded input dim: not row-divisible here
                        return False
                    n_groups_m = out * (m.weight.shape[1] // g)
                    m.weight.data.copy_(qdq[row_w : row_w + out].reshape(m.weight.shape).to(m.weight.device))
                    m.scale = scale[row_g : row_g + n_groups_m].reshape(out, -1).to("cpu")
                    m.zp = zp[row_g : row_g + n_groups_m].reshape(out, -1).to("cpu")
                    m.q_scale_thresh = 1e-5
                    m.weight.grad = None
                    row_w += out
                    row_g += n_groups_m
            if torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            t3 = _time.perf_counter()
            if phase_log is not None:
                # (stack+imatrix hop, search, writeback) per chunk-batch
                phase_log.append((t1 - t0, t2 - t1, t3 - t2))
        return True

    def _quantize_targets(self, targets: list) -> None:
        """Quantize a block's target modules, fanned out over available GPUs.

        Per-module tuning (optimized-RTN scale/zp search, NeUQI joint search)
        is independent across modules: each module's weight is hopped to its
        worker's device, tuned, and written back under a lock. Round-robin
        assignment; per-module results are identical to serial tuning.
        """
        if not targets:
            return
        batches = []
        n_gpu = torch.cuda.device_count()
        parallel_tuning = getattr(self.config, "parallel_tuning", None)
        if parallel_tuning is None:
            parallel_tuning = n_gpu > 1
        if getattr(self.config, "batch_expert_tuning", True) and self._expert_search_active():
            hint = n_gpu if parallel_tuning and n_gpu > 1 else 0
            batches, targets = self._split_expert_batches(targets, n_jobs_hint=hint)
            if batches:
                n_batched = sum(len(b) for b in batches)
                logger.info(
                    "[OptRTN] expert batching: %d expert modules in %d batched groups.",
                    n_batched,
                    len(batches),
                )

        n_workers = min(len(batches) + len(targets), n_gpu) if parallel_tuning and n_gpu > 1 else 1

        def _run_batch(mods, device):
            if not self._quantize_expert_batch(mods, device):
                for m in mods:  # unexpected shape case: per-module fallback
                    self.quantize_layer_outside_block(m, device_override=device)

        if n_workers == 1:
            device = device_manager.device
            for b in batches:
                _run_batch(b, device)
            for m in targets:
                self.quantize_layer_outside_block(m)
            return

        logger.info(
            "[OptRTN] tuning fan-out: %d jobs (%d batched expert groups, %d modules) " "across %d GPUs (round-robin).",
            len(batches) + len(targets),
            len(batches),
            len(targets),
            n_workers,
        )
        from concurrent.futures import ThreadPoolExecutor

        import time as _time

        self._expert_phase_log = []
        job_durations = []  # (seconds, kind, name) appended by worker threads

        def _timed(kind, name, fn, *args, **kwargs):
            t0 = _time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                job_durations.append((_time.perf_counter() - t0, kind, name))

        t_wall = _time.perf_counter()
        jobs = [("batch", b) for b in batches] + [("single", m) for m in targets]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for i, (kind, payload) in enumerate(jobs):
                dev = f"cuda:{i % n_workers}"
                name = getattr(payload[0] if kind == "batch" else payload, "global_name", "?")
                if kind == "batch":
                    futures.append(pool.submit(_timed, kind, f"{name} x{len(payload)}", _run_batch, payload, dev))
                else:
                    futures.append(
                        pool.submit(_timed, kind, name, self.quantize_layer_outside_block, payload, device_override=dev)
                    )
            for f in futures:
                f.result()  # propagate the first worker exception
        elapsed = _time.perf_counter() - t_wall
        durations = sorted(job_durations, reverse=True)
        slowest = ", ".join(f"{n} {d:.1f}s" for d, k, n in durations[:3])
        phases = getattr(self, "_expert_phase_log", None) or []
        if phases:
            stack_t = sum(p[0] for p in phases)
            search_t = sum(p[1] for p in phases)
            write_t = sum(p[2] for p in phases)
            logger.info(
                "[OptRTN] fan-out done in %.1fs | slowest jobs: %s | expert-batch (summed over jobs): "
                "stack+hop %.1fs search %.1fs writeback %.1fs",
                elapsed,
                slowest,
                stack_t,
                search_t,
                write_t,
            )
        else:
            logger.info("[OptRTN] fan-out done in %.1fs | slowest jobs: %s", elapsed, slowest)
        self._expert_phase_log = None
