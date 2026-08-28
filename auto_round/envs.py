# Copyright (c) 2025 Intel Corporation
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
# Note: the design of this module is inspired by vLLM's envs.py
# For detailed usage and configuration guide, see: docs/environments.md

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    AR_LOG_LEVEL: str = "INFO"
    AR_USE_MODELSCOPE: bool = "False"
    AR_MODEL_FREE_SHARD_PARALLELISM: Optional[int] = None
    AUTO_ROUND_CACHE: Optional[str] = None
    AUTO_ROUND_GGUF_AUTO_UPDATE: bool = False
    LLAMA_CPP_ROOT: Optional[str] = None
    AR_AUTO_SCHEME_NSAMPLES: Optional[int] = None
    AR_AUTO_SCHEME_BATCH_SIZE: Optional[int] = None
    AR_AUTO_SCHEME_CACHE: Optional[str] = None
    AR_AUTO_SCHEME_NO_SERIAL_FALLBACK: bool = False
    AR_ENABLE_AUTO_SCHEME_PARALLEL: bool = True
    AR_NVFP4_E5M3_CACHE_HP_WEIGHT: bool = False
    AR_DISK_STREAM_MODEL: bool = False
    AR_RESUME_DIR: Optional[str] = None
    AR_FORCE_MOE_ROUTING_ALL_EXPERTS: bool = False
    AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE: bool = True
    AR_BLOCK_PARALLEL_WORKER: bool = False
    AR_BLOCK_PARALLEL_RANK: int = -1
    AR_BLOCK_PARALLEL_RESULTS: Optional[str] = None
    AR_BLOCK_PARALLEL_SHARED_NONBLOCKS: Optional[str] = None
    AR_BLOCK_PARALLEL_SHARED_INPUTS: Optional[str] = None
    AR_SCHEME_MEM_INVENTORY: bool = False


def _get_optional_positive_int_env(name: str) -> Optional[int]:
    """Read an optional env var that must be a positive integer when set."""
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


environment_variables: dict[str, Callable[[], Any]] = {
    # this is used for configuring the default logging level
    "AR_LOG_LEVEL": lambda: os.getenv("AR_LOG_LEVEL", "INFO").upper(),
    "AR_ENABLE_COMPILE_PACKING": lambda: os.getenv("AR_ENABLE_COMPILE_PACKING", "0").lower() in ("1", "true", "yes"),
    "AR_USE_MODELSCOPE": lambda: os.getenv("AR_USE_MODELSCOPE", "False").lower() in ["1", "true"],
    "AR_WORK_SPACE": lambda: os.getenv("AR_WORK_SPACE", "ar_work_space").lower(),
    "AR_ENABLE_UNIFY_MOE_INPUT_SCALE": lambda: os.getenv("AR_ENABLE_UNIFY_MOE_INPUT_SCALE", "False").lower()
    in ["1", "true"],
    "AR_OMP_NUM_THREADS": lambda: os.getenv("AR_OMP_NUM_THREADS", None),
    "AR_DISABLE_OFFLOAD": lambda: os.getenv("AR_DISABLE_OFFLOAD", "0").lower() in ("1", "true", "yes"),
    "AR_DISABLE_DATASET_SUBPROCESS": lambda: os.getenv("AR_DISABLE_DATASET_SUBPROCESS", "0").lower() in ("1", "true"),
    "AR_DISABLE_COPY_MTP_WEIGHTS": lambda: os.getenv("AR_DISABLE_COPY_MTP_WEIGHTS", "0").lower()
    in ("1", "true", "yes"),
    # Device for the disk-streamed calibration forward pass in
    # LLMCalibrator.collect()'s calibrate_on_cpu branch (targeted block
    # re-quantization). Unset = upstream behavior (cpu). Set to e.g. "cuda:0"
    # to run the whole pass on GPU -- see el_requantize_blocks.py.
    "AR_CALIB_STREAM_DEVICE": lambda: os.getenv("AR_CALIB_STREAM_DEVICE", None),
    "AR_ACT_SCALE": lambda: float(os.getenv("AR_ACT_SCALE", "1.0")),
    "AR_ENABLE_ACT_MINMAX_TUNING": lambda: os.getenv("AR_ENABLE_ACT_MINMAX_TUNING", "0").lower()
    in ("1", "true", "yes"),
    "AR_FUSE_ONLINE_ROTATION": lambda: os.getenv("AR_FUSE_ONLINE_ROTATION", "0").lower() in ("1", "true", "yes"),
    # Controls the search range ratio for symmetric int scale search in
    # `auto_round.data_type.int.search_scales`. The search bound is
    # `nmax * AR_SEARCH_SCALE_RATIO` (default None).
    "AR_SEARCH_SCALE_RATIO": lambda: (
        float(os.getenv("AR_SEARCH_SCALE_RATIO")) if os.getenv("AR_SEARCH_SCALE_RATIO") is not None else None
    ),
    # Number of coarse (log-spaced) and fine (additive) scale candidates for the
    # NeUQI search in ``auto_round.data_type.neuqi``.
    "AR_NEUQI_COARSE": lambda: int(os.getenv("AR_NEUQI_COARSE", "64")),
    "AR_NEUQI_FINE": lambda: int(os.getenv("AR_NEUQI_FINE", "32")),
    # Zero-point sweep backend for the NeUQI search: "auto" serves the batched
    # sweep with the extension Triton kernel on CUDA and falls back to the
    # torch.compile-fused sweep (reference eager sweep elsewhere); "triton"
    # forces the Triton kernel, "compile" forces the torch.compile sweeps on any
    # device, "eager" always uses the reference chunked sweep.
    "AR_NEUQI_BACKEND": lambda: os.getenv("AR_NEUQI_BACKEND", "auto").lower(),
    # Experimental init-search recipes for the SignRound tuning path (iters>0):
    # "" (default) = status-quo min/max init; see docs/environments.md for the
    # recipe table, composability rules, and the BPT arms.
    "AR_TUNE_RECIPE": lambda: os.getenv("AR_TUNE_RECIPE", "").strip().lower(),
    # Post-tuning per-group least-squares scale refit with the integer grid
    # and zero points frozen (composable with any AR_TUNE_RECIPE; also
    # applies to plain minmax-init runs).
    "AR_POST_SCALE_REFIT": lambda: os.getenv("AR_POST_SCALE_REFIT", "0").lower() in ("1", "true", "yes"),
    # Post-quantization per-Linear bias correction: one extra no-grad pass of
    # the calibration rows per block, b = mean(y_fp - y_q) absorbed into bias.
    "AR_BIAS_CORRECT": lambda: os.getenv("AR_BIAS_CORRECT", "0").lower() in ("1", "true", "yes"),
    # When set, the block tuning loop logs a per-block per-stage timing
    # table (staging / forward / loss / sync / backward / snapshot / step,
    # plus CUDA-event GPU times and the host-sync bubble). Diagnostic only;
    # adds no per-iteration sync points.
    "AR_TUNE_PROFILE": lambda: os.getenv("AR_TUNE_PROFILE", "0").lower() in ("1", "true", "yes", "on"),
    # Single-process data parallelism for the block tuning loop: the global
    # calibration batch is sharded across AR_TUNE_DDP_WORLD GPUs (the block's
    # home + mirrors), gradients exchanged with a halving-doubling all-reduce
    # so every replica applies the identical SignSGD step. 1 (default) keeps
    # the exact serial path; must divide the calibration batch size and be a
    # power of two (2/4/8 at batch 8).
    "AR_TUNE_DDP_WORLD": lambda: int(os.getenv("AR_TUNE_DDP_WORLD", "1") or 1),
    # Optional explicit mirror devices for AR_TUNE_DDP_WORLD, comma-separated
    # ("cuda:1,cuda:2"; the block home is always rank 0). Default: home + next
    # GPUs in ascending order, per-mirror VRAM-guarded.
    "AR_TUNE_DDP_DEVICES": lambda: os.getenv("AR_TUNE_DDP_DEVICES", ""),
    # Ping-pong device groups for AR_TUNE_DDP_WORLD, syntax "0,1,2,3;4,5,6,7":
    # streaming homes alternate between group leaders, each block's mirrors
    # are its own group, and the idle group prefetches the next block.
    "AR_TUNE_DDP_GROUPS": lambda: os.getenv("AR_TUNE_DDP_GROUPS", ""),
    # DDP gradient TRANSPORT dtype: fp32 (exact) | bf16 (half wire bytes) |
    # int8 (quarter wire bytes; symmetric per-segment amax scaling -- sign-SGD
    # consumes sign(mean-grad), and the quantization step stays far below any
    # sign-relevant magnitude for realistic gradient spreads). The legacy
    # AR_TUNE_DDP_BF16_GRAD=1 alias maps to bf16; the explicit transport
    # value wins.
    "AR_TUNE_DDP_GRAD_TRANSPORT": lambda: (
        lambda v: (
            v
            if v in ("fp32", "bf16", "int8")
            else (_ for _ in ()).throw(ValueError(f"AR_TUNE_DDP_GRAD_TRANSPORT must be fp32|bf16|int8, got {v!r}"))
        )
    )(
        os.getenv("AR_TUNE_DDP_GRAD_TRANSPORT", "").strip().lower()
        or ("bf16" if os.getenv("AR_TUNE_DDP_BF16_GRAD", "0").lower() in ("1", "true", "yes") else "fp32")
    ),
    "AR_TUNE_DDP_MERGE_STATS": lambda: os.getenv("AR_TUNE_DDP_MERGE_STATS", "0").lower() in ("1", "true", "yes"),
    # Sign-cast gradient exchange (default ON): the SignRound update consumes
    # only sign(mean-grad), so the tuner exchanges int8 SIGNS after the
    # reduce-scatter instead of all-gathering the averaged values -- bitwise
    # identical across ranks, strictly more faithful than a lossy transport
    # all-gather (which can round tiny averaged gradients to zero and lose
    # their sign), and the all-gather wire shrinks 4x (world=8: the exchange
    # loses the payload-heavy all-gather steps that dominated its cost).
    # Only taken when the optimizer is pure sign-SGD (no momentum); set to 0
    # to force the full averaged-value halving-doubling allreduce.
    "AR_TUNE_DDP_SIGN_EXCHANGE": lambda: os.getenv("AR_TUNE_DDP_SIGN_EXCHANGE", "1").lower() in ("1", "true", "yes"),
    # Shard-constrained DDP sampling (default ON): with the distributed
    # calibration pool (device r owns samples [r*shard, (r+1)*shard)), each
    # replica draws its per-iteration sub-batch from its OWN shard instead
    # of a globally shuffled batch -- every pool read stays device-local
    # (a global batch scatters pieces across devices, costing cross-device
    # copies in every replica's reference/input cat, ~10-40 ms/iter at
    # world=8). Same per-epoch coverage; set to 0 for the global sampler.
    "AR_TUNE_DDP_SHARD_SAMPLER": lambda: os.getenv("AR_TUNE_DDP_SHARD_SAMPLER", "1").lower() in ("1", "true", "yes"),
    # Concurrent-shard cap for hook-carrying collection passes (default 4,
    # 0 = uncapped): forward hooks force dynamo graph breaks, leaving the
    # compiled block runner as python-bound eager sections that GIL-convoy
    # beyond a handful of threads -- measured at world=8, the hook pass runs
    # at 3-5x its world=4 per-sample cost (uniformly across shards), while
    # hookless passes scale cleanly and are never capped. Only applies when
    # AR_TUNE_DDP_MERGE_STATS=1 lets hook passes shard at all. Values above
    # the engaged world are no-ops; the knee is CPU-dependent (weaker CPUs
    # convoy earlier).
    # Persistent replica worker pool for the DDP tune loop (default ON):
    # run_threaded is called twice per iteration (forward shards + mirror
    # optimizer steps) and spawning/joining `world` fresh threads each call
    # costs ~1-2 ms per thread of setup plus GIL churn -- a measurable slice
    # of the per-iteration host gap. The pool keeps one daemon worker per
    # replica for the block's lifetime (same first-by-index exception
    # semantics as the spawn path). Set to 0 to restore spawn-per-call
    # (collection passes always use the spawn path -- no lifecycle there).
    "AR_TUNE_DDP_THREAD_POOL": lambda: os.getenv("AR_TUNE_DDP_THREAD_POOL", "1").lower() in ("1", "true", "yes"),
    # Delayed loss read + snapshot ring for the DDP tune loop (default ON):
    # the per-iteration host wait on loss.item() drains the whole GPU chain
    # of the finished iteration; deferring the read to the next iteration
    # overlaps that drain with the newly enqueued forward. Best-params
    # selection semantics are unchanged (same pairs, same strict-less rule,
    # resolved one iteration later; unresolved iterations are dropped on a
    # grid re-swap). Falls back to the immediate read when dynamic_max_gap
    # is set (its early-stop decision needs the current loss in-loop). Set
    # to 0 to restore the immediate read.
    "AR_TUNE_DDP_DELAYED_LOSS": lambda: os.getenv("AR_TUNE_DDP_DELAYED_LOSS", "1").lower() in ("1", "true", "yes"),
    # Flat v-param storage for the DDP tune loop (default ON): rebuild every
    # wrapper tuning parameter as a view of per-(kind, lr) group parameters
    # spanning one fp32 storage, so autograd accumulates into contiguous
    # group grads and the per-iteration gradient gather/scatter collapses to
    # a zero-copy strided view (numerics unchanged: same values, same order,
    # same optimizer math). Built only when DDP engages; any layout mismatch
    # (non-fp32 / mixed-device params) falls back to the legacy per-param
    # storage. Set to 0 to restore per-parameter leaf storage.
    "AR_TUNE_DDP_FLAT_VPARAMS": lambda: os.getenv("AR_TUNE_DDP_FLAT_VPARAMS", "1").lower() in ("1", "true", "yes"),
    # Diagnostic (default OFF): after each block, scan gc for flat-scale cuda:0
    # fp32 tensors that survived the block teardown and log their referrers --
    # attributes the per-block VRAM growth without a debugger.
    "AR_DEBUG_FLAT_LEAK": lambda: os.getenv("AR_DEBUG_FLAT_LEAK", "0").lower() in ("1", "true", "yes"),
    "AR_TUNE_COLL_HOOK_SHARDS": lambda: (
        lambda v: (
            int(v)
            if v.isdigit()
            else (_ for _ in ()).throw(
                ValueError(f"AR_TUNE_COLL_HOOK_SHARDS must be a non-negative integer, got {v!r}")
            )
        )
    )(os.getenv("AR_TUNE_COLL_HOOK_SHARDS", "4").strip()),
    # Allreduce algorithm for the DDP gradient exchange: auto | oneshot |
    # halving. halving-doubling moves the bandwidth-optimal 2*(W-1)/W of the
    # payload per rank but pays 2*log2(W) DEPENDENT exchange steps; one-shot
    # broadcasts each rank's transport-reduced buffer to every peer in a
    # single concurrent wave (2*(W-1) of payload per rank, one latency hop)
    # and reduces locally in canonical order. When per-exchange latency
    # dominates wire bytes (reduced transports on small payloads) one-shot
    # wins decisively. auto picks one-shot for bf16/int8 at world<=8 and
    # fp32 at world<=4 (fp32 one-shot at world=8 would park ~2.85 GB of
    # staging per device); the explicit values force the algorithm.
    "AR_TUNE_DDP_ALLREDUCE": lambda: (
        lambda v: (
            v
            if v in ("auto", "oneshot", "halving")
            else (_ for _ in ()).throw(ValueError(f"AR_TUNE_DDP_ALLREDUCE must be auto|oneshot|halving, got {v!r}"))
        )
    )(os.getenv("AR_TUNE_DDP_ALLREDUCE", "auto").strip().lower()),
    # Background ready-transforms (default ON when eligible): while block N
    # tunes on its ping-pong group, early-load block N+1 on the idle group's
    # home device and apply the weight-only layer-wise transforms (e.g. PreSINQ
    # folds) there, taking the transform off the critical path. Eligibility
    # still requires >=2 ping-pong staging groups + an active layer-wise
    # transform + streaming prefetch; activation-dependent preprocessors (AWQ
    # scale/clip) stay in-loop either way. Set to 1 to opt out.
    "AR_DISABLE_BG_READY_TRANSFORMS": lambda: os.getenv("AR_DISABLE_BG_READY_TRANSFORMS", "0").lower()
    in ("1", "true", "yes"),
    # Background pack pipeline (default ON when eligible): the finished
    # block's immediate-pack + shard-write tail runs in a background thread
    # on its (now idle) ping-pong home while the loop advances to the next
    # block's tune on the other group. Eligibility still requires >=2 staging
    # groups + streaming + immediate packing; exactly one pipeline thread
    # runs at a time. Set to 1 to opt out.
    "AR_DISABLE_BG_PACK": lambda: os.getenv("AR_DISABLE_BG_PACK", "0").lower() in ("1", "true", "yes"),
    # Overlapped gradient exchange (default ON when eligible): per-parameter
    # post-accumulate-grad hooks fire DURING backward, encode each bucket on a
    # per-device side stream and P2P-copy it into preallocated staging on all
    # peers; sync_grads then only waits the side streams and runs the canonical
    # per-bucket sum. DEFAULT OFF: the per-hook fan-out issues copies from
    # multiple autograd threads with per-copy device/stream context switches,
    # which drops the P2P fast path (~2 GB/s vs ~13 GB/s single-threaded),
    # making the overlapped exchange SLOWER than the classic sequential one
    # (world=4 bf16: 825 vs 472 ms/iter; int8: 580 vs 458). Set to "0" to
    # re-enable; the structure is kept for a single-threaded fan-out rework.
    "AR_DISABLE_OVERLAP_EXCHANGE": lambda: os.getenv("AR_DISABLE_OVERLAP_EXCHANGE", "1").lower()
    in ("1", "true", "yes"),
    # alt2 (alternating re-grid): iterations of the SECOND tuning round after
    # the mid-tune re-grid; 0 = half of --iters.
    "AR_ALT2_ITERS2": lambda: int(os.getenv("AR_ALT2_ITERS2", "0") or 0),
    # qoff (FP-reference-chain) tuning unblocker: inject per-channel
    # quantization noise into the FP block inputs during SignRound tuning.
    "AR_QOFF_NOISE": lambda: os.getenv("AR_QOFF_NOISE", "0").lower() in ("1", "true", "yes"),
    # Directory for those noise stats: when set, every quantized block also
    # writes block_<idx>.pt (per-channel mean/var of y_fp - y_q); the tuning
    # run reads them back for injection.
    "AR_QOFF_NOISE_STATS": lambda: os.getenv("AR_QOFF_NOISE_STATS", "").strip(),
    # Post-BPT serial qon touch-up: re-tune this many iterations per block on
    # the real quantized chain, starting each wrapper from the tuned grid.
    "AR_TOUCHUP_ITERS": lambda: int(os.getenv("AR_TOUCHUP_ITERS", "0") or 0),
    # Memory layout of the fused zero-point sweep expression: "auto" picks by
    # device (group-axis-last reduction on CUDA, zero-point-axis-last elsewhere),
    # "last"/"mid" force one layout for A/B measurements.
    "AR_NEUQI_LAYOUT": lambda: os.getenv("AR_NEUQI_LAYOUT", "auto").lower(),
    # All-candidates batched zero-point sweep for the NeUQI search: "auto" batches
    # every coarse/fine scale candidate into one fused kernel per group chunk on
    # CUDA, "on" forces it on any fused-capable device, "off" keeps the
    # per-candidate loop.
    "AR_NEUQI_BATCH": lambda: os.getenv("AR_NEUQI_BATCH", "auto").lower(),
    "AR_STREAM_MEM_INVENTORY": lambda: os.getenv("AR_STREAM_MEM_INVENTORY", "0").lower() in ("1", "true", "yes"),
    "AR_DISABLE_TUNING_FANOUT": lambda: os.getenv("AR_DISABLE_TUNING_FANOUT", "0").lower() in ("1", "true", "yes"),
    # Sinkhorn loop backend for the Pre-SINQ transform: "auto" serves the loop
    # with the extension Triton kernels on CUDA (2.1-2.8x over the eager loop,
    # fp64-ulp parity) and the eager reference loop elsewhere; "triton" forces
    # the kernels, "compile" the torch.compile fused graph (opt-in), "eager"
    # the reference loop. Any failure permanently reverts to the torch loops.
    "AR_PRESINQ_BACKEND": lambda: os.getenv("AR_PRESINQ_BACKEND", "auto").lower(),
    # Minimum value to which torch._dynamo cache_size_limit /
    # accumulated_cache_size_limit / recompile_limit are bumped when
    # ``enable_torch_compile`` is used. The default of 16 is enough to cover
    # all distinct linear-weight shapes inside one transformer block (q/k/v/
    # o_proj, gate/up/down_proj, ...) so that per-layer static recompiles do
    # not exceed dynamo's default limit (8) and fall back to eager.
    "AR_DYNAMO_CACHE_SIZE_LIMIT": lambda: int(os.getenv("AR_DYNAMO_CACHE_SIZE_LIMIT", "16")),
    "AR_MODEL_FREE_SHARD_PARALLELISM": lambda: _get_optional_positive_int_env("AR_MODEL_FREE_SHARD_PARALLELISM"),
    "AUTO_ROUND_CACHE": lambda: os.getenv("AUTO_ROUND_CACHE", None),
    "AUTO_ROUND_GGUF_AUTO_UPDATE": lambda: os.getenv("AUTO_ROUND_GGUF_AUTO_UPDATE", "0").lower()
    in ("1", "true", "yes", "on"),
    "LLAMA_CPP_ROOT": lambda: os.getenv("LLAMA_CPP_ROOT", None),
    # Controls the default number of calibration samples used by AutoScheme scoring
    # when ``AutoScheme.nsamples`` is not explicitly set.
    # When unset, AutoScheme uses 16.
    "AR_AUTO_SCHEME_NSAMPLES": lambda: _get_optional_positive_int_env("AR_AUTO_SCHEME_NSAMPLES"),
    # Controls the default batch size used by AutoScheme scoring
    # when ``AutoScheme.batch_size`` is not explicitly set.
    # When unset, AutoScheme uses its built-in heuristic (8 for low GPU memory mode, 1 for normal mode).
    "AR_AUTO_SCHEME_BATCH_SIZE": lambda: _get_optional_positive_int_env("AR_AUTO_SCHEME_BATCH_SIZE"),
    # Controls the default calibration sequence length used by AutoScheme scoring
    # when ``AutoScheme.seqlen`` is not explicitly set.
    # When unset, AutoScheme uses its built-in heuristic (128 for MoE models, 256 otherwise).
    "AR_AUTO_SCHEME_SEQLEN": lambda: _get_optional_positive_int_env("AR_AUTO_SCHEME_SEQLEN"),
    # Stores persistent AutoScheme scoring results independently from AR_WORK_SPACE,
    # whose contents are temporary working data and may be cleaned after a run.
    "AR_AUTO_SCHEME_CACHE": lambda: os.getenv("AR_AUTO_SCHEME_CACHE", None),
    "AR_AUTO_SCHEME_NO_SERIAL_FALLBACK": lambda: os.getenv("AR_AUTO_SCHEME_NO_SERIAL_FALLBACK", "0").lower()
    in ("1", "true", "yes"),
    # Enables AutoScheme to score schemes in parallel. Enabled by default;
    # set it to 0 when workers could exhaust host RAM or device memory.
    "AR_ENABLE_AUTO_SCHEME_PARALLEL": lambda: os.getenv("AR_ENABLE_AUTO_SCHEME_PARALLEL", "1").lower()
    in ("1", "true", "yes"),
    # Controls whether NVFP4 E5M3 quant linear caches a dequantized high-
    # precision weight after the first forward instead of dequantizing on
    # every call. When enabled, the packed weight buffers are released after
    # the cache is materialized, trading lower runtime overhead for higher
    # steady-state memory usage.
    "AR_NVFP4_E5M3_CACHE_HP_WEIGHT": lambda: os.getenv("AR_NVFP4_E5M3_CACHE_HP_WEIGHT", "0").lower()
    in ("1", "true", "yes", "on"),
    # When set, the model is built as a meta-device skeleton and streamed
    # block-by-block from disk during quantization instead of being fully
    # materialized on CPU RAM up front.
    "AR_DISK_STREAM_MODEL": lambda: os.getenv("AR_DISK_STREAM_MODEL", "0").lower() in ("1", "true", "yes"),
    # When set to a directory path, the per-block tuning loop checkpoints its
    # progress there after each completed block, and resumes from the first
    # not-yet-completed block on a fresh run against the same directory --
    # instead of restarting the whole tuning pass from block 0 after a
    # crash/kill. See auto_round/utils/resume.py.
    "AR_RESUME_DIR": lambda: os.getenv("AR_RESUME_DIR", None),
    # When enabled, MoE routing can be overridden in selected model wrappers
    # to rotate token assignments across all experts for calibration coverage.
    "AR_FORCE_MOE_ROUTING_ALL_EXPERTS": lambda: os.getenv("AR_FORCE_MOE_ROUTING_ALL_EXPERTS", "0").lower()
    in ("1", "true", "yes"),
    # vLLM fused kernels require q/k/v and gate/up projections to use one
    # weight global scale. Disable only for runtimes without that requirement.
    "AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE": lambda: os.getenv("AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE", "1").lower()
    not in ("0", "false", "no", "off"),
    # Internal sentinel set in worker processes spawned for block-parallel
    # tuning; prevents workers from spawning further parallel runs.
    "AR_BLOCK_PARALLEL_WORKER": lambda: os.getenv("AR_BLOCK_PARALLEL_WORKER", "0").lower() in ("1", "true", "yes"),
    # Internal: advisory worker rank for live assignment targeting; injected
    # by the parent into worker envs, recorded in block result payloads.
    "AR_BLOCK_PARALLEL_RANK": lambda: int(os.getenv("AR_BLOCK_PARALLEL_RANK", "-1")),
    # Internal: directory where parallel tuning workers dump tuned scale/zp.
    "AR_BLOCK_PARALLEL_RESULTS": lambda: os.getenv("AR_BLOCK_PARALLEL_RESULTS", None),
    # Internal: mmap-shared non-block tensors + calibration inputs written by
    # the parent and consumed by workers (and by the parent on resume).
    "AR_BLOCK_PARALLEL_SHARED_NONBLOCKS": lambda: os.getenv("AR_BLOCK_PARALLEL_SHARED_NONBLOCKS", None),
    "AR_BLOCK_PARALLEL_SHARED_INPUTS": lambda: os.getenv("AR_BLOCK_PARALLEL_SHARED_INPUTS", None),
    # Logs a per-device live-tensor census during AutoScheme scoring (debug
    # aid for memory-driven worker placement).
    "AR_SCHEME_MEM_INVENTORY": lambda: os.getenv("AR_SCHEME_MEM_INVENTORY", "0").lower() in ("1", "true", "yes"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_config(**kwargs):
    """
    Set configuration values for environment variables.

    Args:
        **kwargs: Keyword arguments where keys are environment variable names
                 and values are the desired values to set.

    Example:
        set_config(AR_LOG_LEVEL="DEBUG", AR_USE_MODELSCOPE=True)
    """
    for key, value in kwargs.items():
        if key in environment_variables:
            # Convert value to appropriate string format
            if key == "AR_USE_MODELSCOPE":
                # Handle boolean values for boolean env flags
                str_value = "true" if value in [True, "True", "true", "1", 1] else "false"
            else:
                # For other variables, convert to string
                str_value = str(value)

            # Set the environment variable
            os.environ[key] = str_value
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {key!r}")
