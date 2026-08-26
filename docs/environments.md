# AutoRound Environment Variables Configuration

English | [简体中文](./environments_CN.md)

This document describes the environment variables used by AutoRound for configuration and their usage.

## Overview

AutoRound uses a centralized environment variable management system through the `envs.py` module. This system provides lazy evaluation of environment variables and programmatic configuration capabilities.

## Available Environment Variables

### AR_LOG_LEVEL
- **Description**: Controls the default logging level for AutoRound
- **Default**: `"INFO"`
- **Valid Values**: `"TRACE"`,  `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Usage**: Set this to control the verbosity of AutoRound logs

```bash
export AR_LOG_LEVEL=DEBUG
```

### AR_ENABLE_COMPILE_PACKING
- **Description**: Enables compile packing optimization
- **Default**: `False` (equivalent to `"0"`)
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) for enabling; any other value for disabling
- **Usage**: Enable this for performance optimizations during packing FP4 tensors into `uint8`.

```bash
export AR_ENABLE_COMPILE_PACKING=1
```

### AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE
- **Description**: Makes fused NVFP4 weight projections use one shared weight global scale. This applies to `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`, as required by vLLM fused kernels.
- **Default**: `True` (equivalent to `"1"`)
- **Valid Values**: `"0"`, `"false"`, `"no"`, or `"off"` (case-insensitive) disable sharing; any other value enables it.
- **Usage**: Disable only when exporting for a runtime that does not require fused projections to share a global scale.

```bash
export AR_NVFP4_FUSED_LAYER_GLOBAL_SCALE=0
```

### AR_USE_MODELSCOPE
- **Description**: Controls whether to use ModelScope for model downloads
- **Default**: `False`
- **Valid Values**: `"1"`, `"true"` (case-insensitive) for enabling; any other value for disabling
- **Usage**: Enable this to use ModelScope instead of Hugging Face Hub for model downloads

```bash
export AR_USE_MODELSCOPE=true
```

### AR_WORK_SPACE
- **Description**: Sets the workspace directory for AutoRound operations
- **Default**: `"ar_work_space"`
- **Usage**: Specify a custom directory for AutoRound to store temporary files and outputs

```bash
export AR_WORK_SPACE=/path/to/custom/workspace
```

### AR_ROTATION_FIX_LINEAR_ATTN
- **Description**: Gate for the linear-attention offline-R1 fix used by rotation transforms (SpinQuant) on hybrid-attention models. When enabled, `layer.linear_attn` projections absorb R1 and `input_layernorm` gamma is folded into them, keeping offline R1 mathematically equivalent. Default off preserves the legacy behavior. Only read by rotation-transform code paths.
- **Default**: `"0"`
- **Valid Values**: `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive) enable; any other value disables
- **Usage**: Enable for offline-R1 rotation runs on hybrid linear-attention architectures

```bash
export AR_ROTATION_FIX_LINEAR_ATTN=1
```

### AR_DISABLE_OFFLOAD
- **Description**: Forcibly disables the weight offloading feature in `OffloadManager`. Useful during development and debugging to skip all offload/reload overhead.
- **Default**: `False` (equivalent to `"0"`)
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) for disabling offload; any other value keeps the default behavior
- **Usage**: Set this to bypass offloading entirely

```bash
export AR_DISABLE_OFFLOAD=1
```

### AR_DISABLE_DATASET_SUBPROCESS
- **Description**: Only for research. Disables the use of a subprocess for dataset preprocessing. By default, AutoRound uses a subprocess to ensure all temporary memory is reclaimed by the OS.
- **Default**: `False`
- **Valid Values**: `"1"`, `"true"` (case-insensitive) for disabling; any other value for enabling
- **Usage**: Set this to run dataset preprocessing in the main process

```bash
export AR_DISABLE_DATASET_SUBPROCESS=true
```

### AR_ACT_SCALE
- **Description**: Only for research. Controls the scaling factor applied to activation min/max values during activation quantization. A value less than 1.0 shrinks the clipping range, which can reduce outlier impact.
- **Default**: `1.0`
- **Valid Values**: float>=0.0, e.g. `0.8`, `0.9`, `1.0`
- **Usage**: Set this to adjust the activation clipping range

```bash
export AR_ACT_SCALE=0.9
```

### AR_ENABLE_ACT_MINMAX_TUNING
- **Description**: Enables tuning of activation min/max scale parameters (`act_min_scale`, `act_max_scale`) during quantization optimization. When enabled, these scales become tunable instead of remaining fixed at `1.0`.
- **Default**: `False` (equivalent to `"0"`)
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) for enabling tuning; any other value keeps tuning disabled
- **Usage**: Set this to enable activation min-max scale tuning

```bash
export AR_ENABLE_ACT_MINMAX_TUNING=1
```

### AR_SEARCH_SCALE_RATIO
- **Description**: Controls the search range ratio used by the symmetric INT scale search in `auto_round.data_type.int.search_scales`. The search bound is `nmax * AR_SEARCH_SCALE_RATIO`, where `nmax = 2^(bits-1)`. Smaller values restrict the search to a tighter neighborhood around the initial scale (faster, less thorough); larger values broaden the search (slower, may improve accuracy on outlier-heavy weights).
- **Default**: unset → falls back to the built-in default (`0.5`, i.e. `nmax/2`).
- **Valid Values**: positive float, e.g. `0.25`, `0.5`, `0.75`, `1.0`
- **Usage**: Set this to override the default scale-search range

```bash
export AR_SEARCH_SCALE_RATIO=0.75
```

### AR_DYNAMO_CACHE_SIZE_LIMIT
- **Description**: Minimum value to which `torch._dynamo`'s `cache_size_limit`, `accumulated_cache_size_limit`, and `recompile_limit` are bumped when `torch.compile` is enabled (the default except on Windows). The same compiled quant function is reused across every linear layer in a transformer block (q/k/v/o_proj, gate/up/down_proj, ...) but each layer has a different weight shape, so per-layer static recompiles quickly exceed dynamo's default limit (8) and trigger a noisy fallback to eager. Raising the limit keeps static-shape compilation (best perf) and just allows more cache entries.
- **Default**: `16`
- **Valid Values**: positive integer
- **Usage**: Increase if your model has more than 16 distinct linear-weight shapes per block (rare).

```bash
export AR_DYNAMO_CACHE_SIZE_LIMIT=32
```

### AR_MODEL_FREE_SHARD_PARALLELISM
- **Description**: Controls how many weight shards are processed concurrently during model-free quantization. Increasing this value improves resource utilization but consumes more RAM.
  - Auto policy (when the variable is **not** set): `shard_count // 4`, capped at **10**, minimum **1**. For example, 8 shards → 2 workers; 40 shards → 10 workers.
  - The effective parallelism is always capped to the actual number of shards.
- **Default**: unset → auto policy applies (`shard_count // 4`, max 10, min 1)
- **Valid Values**: any positive integer; these are not restricted to specific values — e.g. `2`, `4`, `6`, `8`
- **Usage**: Set this to override the automatic parallelism selection

```bash
export AR_MODEL_FREE_SHARD_PARALLELISM=4
```

### AR_AUTO_SCHEME_NSAMPLES
- **Description**: Controls the default number of calibration samples used by AutoScheme scoring when `AutoScheme.nsamples` is not explicitly set.
- **Default**: unset → 16
- **Valid Values**: any positive integer, e.g. `8`, `16`, `32`
- **Usage**: Set this to override the automatic sample-count selection for AutoScheme

```bash
export AR_AUTO_SCHEME_NSAMPLES=1  # set 1 for quick execution
```

### AR_AUTO_SCHEME_BATCH_SIZE
- **Description**: Controls the default batch size used by AutoScheme scoring when `AutoScheme.batch_size` is not explicitly set.
- **Default**: unset → built-in heuristic applies (8 for low GPU memory mode, 1 for normal mode)
- **Valid Values**: any positive integer, e.g. `1`, `2`, `4`
- **Usage**: Set this to override the default batch size for AutoScheme

```bash
export AR_AUTO_SCHEME_BATCH_SIZE=1
```

### AR_AUTO_SCHEME_SEQLEN
- **Description**: Controls the default calibration sequence length used by AutoScheme scoring when `AutoScheme.seqlen` is not explicitly set.
- **Default**: unset → built-in heuristic applies (128 for MoE models, 256 otherwise)
- **Valid Values**: any positive integer, e.g. `256`, `512`, `1024`
- **Usage**: Set this to override the default sequence length for AutoScheme (2-bit schemes usually benefit from `1024`)

```bash
export AR_AUTO_SCHEME_SEQLEN=1024
```

### AR_AUTO_SCHEME_NO_SERIAL_FALLBACK
- **Description**: Turn a parallel-scoring failure into a hard error instead of falling back to serial scoring. Useful when the serial pass is known to be unable to run (or would take workers-count times longer): completed schemes and batches are persisted in the per-scheme cache, so a rerun scores only the failed parts.
- **Default**: unset -> parallel scoring failure falls back to serial
- **Valid Values**: `1`, `true`, `yes`
- **Usage**: Set this to fail fast on parallel scoring errors

```bash
export AR_AUTO_SCHEME_NO_SERIAL_FALLBACK=1
```

### AR_AUTO_SCHEME_CACHE
- **Description**: Stores persistent per-scheme AutoScheme scoring JSON files. This directory is independent of `AR_WORK_SPACE`, which is reserved for temporary working data.
- **Default**: `~/.cache/auto_round`
- **Valid Values**: any writable directory path
- **Usage**: Set this to place reusable AutoScheme scores in a different cache directory

```bash
export AR_AUTO_SCHEME_CACHE=/path/to/auto_scheme_cache
```

### AR_ENABLE_AUTO_SCHEME_PARALLEL
- **Description**: Enables multiprocessing across AutoScheme candidates. It can be combined with `AR_DISK_STREAM_MODEL=1`; each worker then builds its own meta-model skeleton and streams blocks independently. Disable it when concurrent workers could exhaust host RAM or device memory.
- **Default**: `"1"` (schemes are scored in parallel when multiprocessing requirements are met)
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) enable parallel scoring; any other value disables parallel scoring
- **Usage**: Set this to `0` before running AutoScheme to force serial candidate scoring

```bash
export AR_ENABLE_AUTO_SCHEME_PARALLEL=0
```

### AR_NEUQI_COARSE
- **Description**: Number of coarse (log-spaced) scale candidates explored by the NeUQI search before the fine additive refinement.
- **Default**: `"64"`
- **Valid Values**: any positive integer
- **Usage**: Lower for faster searches, raise for exhaustive scale sweeps

```bash
export AR_NEUQI_COARSE=64
```

### AR_NEUQI_FINE
- **Description**: Number of fine (additive) scale refinement candidates per coarse candidate in the NeUQI search.
- **Default**: `"32"`
- **Valid Values**: any positive integer
- **Usage**: Lower for faster searches, raise for finer scale resolution

```bash
export AR_NEUQI_FINE=32
```

### AR_NEUQI_BACKEND
- **Description**: Backend chain for the NeUQI zero-point sweep. `"auto"` (default) keeps the reference chunked eager sweep on CPU and, on CUDA, serves the batched sweep with the hand-written Triton kernel from `auto_round_extension.triton.neuqi_sweep` (registers-resident: each weight element is loaded once and every (candidate, zero-point) loss is computed from registers), falling back to the `torch.compile`-fused sweep when Triton is unavailable. `"triton"` forces the Triton kernel, `"compile"` forces the `torch.compile` sweeps on any device, and `"eager"` always uses the reference sweep. All backends evaluate the exact same losses over the exact same candidate grid (selections identical up to near-ties: <~0.1% of groups flip, worst observed relative loss difference ~5e-5 on an RTX 3090, with the Triton kernel matching that tie profile exactly) and remove the HBM-traffic bottleneck of the brute-force grid on large batched expert searches (measured on a 2M-group RTX 3090 sweep: 12.3 s eager -> 0.59 s fused per-candidate -> 0.49 s compiled-batched -> 0.22 s Triton, ~57x). Every stage latches down permanently on failure: Triton -> compiled batched -> compiled per-candidate -> eager. Requires `AR_NEUQI_BATCH` to be enabled for the Triton/compiled-batched stages (it is by default).
- **Default**: `"auto"`
- **Valid Values**: `"auto"`, `"triton"`, `"compile"`, `"eager"` (unrecognized values are treated as `"eager"`)
- **Usage**: Pin a specific stage for A/B comparisons, or disable compilation entirely

```bash
export AR_NEUQI_BACKEND=eager
```

### AR_NEUQI_LAYOUT
- **Description**: Memory layout of the fused zero-point sweep expression. The sweep reduces squared errors over the group axis for every integer zero point, and the two axes can be arranged either way: `"last"` lays the tensor out as `[groups, zero_points, group_size]` with the reduction over the contiguous last dimension (the canonical Triton/CUDA fusion shape; a middle-dimension reduction measured no faster than eager on an RTX 3090), while `"mid"` lays it out as `[groups, group_size, zero_points]` (faster under the eager TensorIterator path and the compiled CPU backend). `"auto"` (default) selects by device: `"last"` on CUDA, `"mid"` elsewhere. Both layouts compute identical losses up to the fp32 summation order over the group (last-ulp ties).
- **Default**: `"auto"`
- **Valid Values**: `"auto"`, `"last"`, `"mid"` (unrecognized values follow the device-based `"auto"` rule)
- **Usage**: A/B the two layouts on a given device

```bash
export AR_NEUQI_LAYOUT=mid
```

### AR_PRESINQ_BACKEND
- **Description**: Sinkhorn loop backend for the Pre-SINQ transform. `"auto"` (default) serves the loop with the hand-written Triton kernels from `auto_round_extension.triton.presinq_sinkhorn` on CUDA (three fp64 kernels per iteration — one read of the weight tile with both stds computed from registers, a deterministic two-phase column reduction, and the slack-hardened imbalance tracker) and uses the eager reference loop elsewhere. Measured on an RTX 3090: 2.1–2.8x faster than the eager std-once loop across dense/concat/expert/pooled shapes (~4.3x vs the original port), fp64-ulp parity (~1e-15), and the pooled MoE norm fold (huge concatenated consumer matrices, e.g. Hunyuan-A13B-class layers) runs within 24 GiB where the eager loop OOMs. `"triton"` forces the Triton kernels (falls back permanently to the torch loops on any failure — including a forced attempt without CUDA); `"eager"` always uses the eager loop; `"compile"` forces the `torch.compile` fused graph (opt-in: measured at best on par and up to 20% slower than eager on an RTX 3090). The eager loop itself is bit-exact with the previous release and ~1.8x faster (per-iteration stds computed once and reused, unused scaled-matrix materialization skipped).
- **Default**: `"auto"`
- **Valid Values**: `"auto"`, `"triton"`, `"eager"`, `"compile"` (unrecognized values are treated as `"eager"`)
- **Usage**: Pin a backend for A/B comparisons or on architectures where the measurements differ

```bash
export AR_PRESINQ_BACKEND=eager
```

### AR_DISABLE_TUNING_FANOUT

- **Type**: bool (`1`/`true`/`yes` to enable; default off)
- **Description**: Disables the multi-GPU per-module tuning fan-out in the RTN/NeUQI zero-shot path (the `[OptRTN] tuning fan-out: ... across N GPUs` round-robin that hops module weights to worker devices). With the env set, all scale/zp searches run serially on the primary device. Per-module results are identical either way; this is an isolation/forensics knob (single-stream Triton launches, no fan-out threads), not a speed knob. An explicit `parallel_tuning=True` config kwarg still wins over the env. Does not affect block-parallel tuning (BPT) or data-driven SignRound tuning, which have their own switches.

### AR_NEUQI_BATCH
- **Description**: All-candidates batched zero-point sweep. `"auto"` (default) folds every coarse and fine scale candidate of a pass into a single fused kernel per group chunk on CUDA (the per-candidate fused sweep spends most of its remaining wall time in per-candidate dispatch and bookkeeping launches); `"on"` forces the batched sweep on any fused-capable device (e.g. for A/B or tests); `"off"` keeps the per-candidate loop. The batched kernel computes the min over zero points in-kernel and emits only `[groups, candidates]` best losses and winning zero points per launch. Selections follow the same first-minimum tie rule as the sequential sweep; after the single-candidate fusion this is the second speedup stage on top of the same candidate grid (21x measured for the first stage on an RTX 3090). Any failure of a batched call (e.g. out of memory from the larger symbolic intermediate) permanently latches the process back to the per-candidate fused sweep for the remaining candidates.
- **Default**: `"auto"`
- **Valid Values**: `"auto"`, `"on"`, `"off"` (unrecognized values follow the `"auto"` rule)
- **Usage**: Pin the per-candidate loop for A/B comparisons of the batching stage

```bash
export AR_NEUQI_BATCH=off
```

### AR_NVFP4_E5M3_CACHE_HP_WEIGHT
- **Description**: Controls whether `NVFP4E5M3QuantLinear` caches a dequantized high-precision weight after the first forward pass, instead of dequantizing the packed FP4 weight on every call.
- **Default**: `False` (equivalent to `"0"`)
- **Valid Values**: `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive) enable caching; any other value disables caching
- **Usage**: Enable this when repeated inference throughput matters more than memory footprint. The current implementation releases `weight_packed` and `weight_scale` after materializing the cached high-precision weight, so steady-state memory usage increases and the cache cannot be cleared back to packed storage.

```bash
export AR_NVFP4_E5M3_CACHE_HP_WEIGHT=1
```

### AR_DISK_STREAM_MODEL
- **Description**: When enabled, `AutoRound(model=<path>, ...)` builds the model as a meta-device skeleton instead of fully materializing the checkpoint on CPU RAM up front, and streams each decoder block's real weights from the checkpoint's safetensors shards on demand -- materializing right before a block is used (calibration, tuning, or `AutoScheme` sensitivity scoring) and freeing it back to meta right after. This keeps peak CPU RAM roughly flat regardless of checkpoint size, instead of proportional to it. Non-block parameters (embeddings, `lm_head`, final norm) are still loaded up front, since they are typically small. Text-model AutoScheme scoring also supports combining this with parallel scoring, which is enabled by default; each worker streams its own block copy.
- **Default**: `False`
- **Valid Values**: `"1"`, `"true"`, `"yes"` (case-insensitive) for enabling; any other value for disabling
- **Usage**: Enable this to quantize checkpoints larger than available CPU RAM + GPU VRAM combined. Only applies when `model` is a string (local directory) path; has no effect on already-loaded model objects.

```bash
export AR_DISK_STREAM_MODEL=1
```

### AR_POST_SCALE_REFIT
- **Description**: Post-tuning per-group least-squares scale refit with the integer grid and zero points frozen. One closed-form step (the exact minimizer for frozen integers), imatrix-weighted when an imatrix is present; monotone non-increasing weighted MSE by construction. Composable with any `AR_TUNE_RECIPE` (and with plain minmax-init runs); applies to asymmetric int layers with standard group sizes (symmetric layers keep their grid, logged once).
- **Default**: `0` (off)
- **Valid Values**: `0`, `1` (also `true`/`yes`)

```bash
export AR_POST_SCALE_REFIT=1
```

### AR_BIAS_CORRECT
- **Description**: Post-quantization bias correction per transformer block: `b = mean over calibration tokens(y_fp - y_q)` at the block's residual-stream boundary, absorbed into the bias of the block's residual-feeding projection (the last Linear/Conv1D with out_features == hidden; routed-expert modules are deprioritized since they execute on token subsets only). Bias is created when absent — native in all export formats, so vLLM sees it without changes. Reuses the chain forward on the qon path (zero extra forwards); adds one no-grad pass on the qoff path. Serial/streaming only — a hard error under block-parallel workers (biases are not part of worker result files). Composable with any `AR_TUNE_RECIPE` and `AR_POST_SCALE_REFIT`.
- **Default**: `0` (off)
- **Valid Values**: `0`, `1` (also `true`/`yes`)

```bash
export AR_BIAS_CORRECT=1
```

### AR_ALT2_ITERS2
- **Description**: Iterations of the SECOND tuning round when `AR_TUNE_RECIPE=alt2` (alternating re-grid). Round 1 tunes the rounding on the init-search grid for `iters - iters2` iterations; then every alt2 layer re-runs the NeUQI search on the perturbed effective weights, re-anchors the grid, resets the rounding params to zero, and round 2 tunes `iters2` more iterations on the fresh grid. Best-params cache from round 1 is discarded at the switch (the grid changed).
- **Default**: `0` (half of `--iters`)
- **Valid Values**: `0` (auto: half), or `1` … `iters-1`

```bash
export AR_TUNE_RECIPE=alt2 AR_ALT2_ITERS2=10   # + --iters 20 --asym
```

### AR_QOFF_NOISE
- **Description**: qoff (FP-reference-chain) tuning unblocker. BPT/serial qoff tuning optimizes each block against FP-reference inputs that the deployed quantized chain never produces — the measured deployment-mismatch regression. When enabled, the tuning forwards inject per-channel quantization noise `mean + std*eps` (stats of the previous quantized block's output drift, `eps` deterministic per block seed) into the FP inputs; the cached FP inputs are never modified in place. Block 0 skips injection (its inputs are embeddings). Guards: hard error under qon (quantized-input chaining already sees real inputs), without `AR_QOFF_NOISE_STATS`, or on missing/width-mismatched stats, on the zero-shot path (no tuning), and when nblocks > 1 (one stats file per block contract).
- **Default**: `0` (off)
- **Valid Values**: `0`, `1` (also `true`/`yes`)

```bash
# 1) collection pass (cheap, e.g. --iters 0): writes block_<idx>.pt stats
export AR_QOFF_NOISE_STATS=/mnt/bigdisk/qoff_stats
# 2) tuning pass (qoff/BPT): injects them
export AR_QOFF_NOISE=1 AR_QOFF_NOISE_STATS=/mnt/bigdisk/qoff_stats
```

### AR_TOUCHUP_ITERS
- **Description**: Post-BPT serial qon touch-up: after a block-parallel (qoff) run has tuned every block, a serial rerun with this env re-tunes N iterations per block on the REAL quantized chain (qon). Each SignRound wrapper starts anchored to the BPT-tuned (scale, zp) pair — exactly where the parallel run left off — with rounding params reset to zero and margins still tunable; the improved result overwrites the block's result file, so a later apply/export uses the touched grid. The run signature includes the touch-up count: changing N invalidates stale resume artifacts. Guards: serial only (unset in worker environments), requires quantized-input chaining (qon), an existing results dir, and a complete result file for every block.
- **Default**: `0` (off)
- **Valid Values**: `0` (off), or a small positive iteration count (2–5 typical)

```bash
# 1) BPT pass (results in AR_RESUME_DIR)
# 2) serial touch-up on the quantized chain:
export AR_TOUCHUP_ITERS=5   # same AR_RESUME_DIR; no --enable_block_parallel_tuning; qon enabled
```

### AR_TUNE_RECIPE
- **Description**: Experimental init-search recipe for the SignRound tuning path (`--iters > 0`). The recipe replaces the per-group min/max tuning grid with a searched grid: `neuqi_*` anchors the joint (scale, integer zero-point) search (`neuqi_search_scale_zero`, imatrix-weighted); `opt_rtn_qon` anchors the symmetric scale-clip search. `neuqi_frozen_qon` additionally pins the `min_scale`/`max_scale` tuning margins at 1.0 (grid fully fixed, only rounding tunes). `neuqi_fp` vs `neuqi_qon` differ only in which chain shaped the init imatrix (FP-reference vs quantized) — under streaming qon the live imatrix is already chain-faithful. Recipes apply to int data types with standard (non-tuple) group sizes; unsupported layouts keep the min/max grid.
- **Default**: unset (status-quo min/max init)
- **Valid Values**: `minmax_qon` (explicit control arm), `neuqi_qon`, `neuqi_frozen_qon`, `neuqi_fp` (BPT/qoff-compatible), `opt_rtn_qon` (symmetric only), `alt2`, `neuqi_it0` (zero-shot reference marker; requires `--iters 0`)
- **Usage**: Race quantization recipes by KLD on a small model. `neuqi_*` requires `--asym`; `opt_rtn_qon` requires symmetric. Composable with `AR_POST_SCALE_REFIT` and `AR_BIAS_CORRECT`.

```bash
export AR_TUNE_RECIPE=neuqi_qon   # + --iters 20 --asym --imatrix_enabled true
```

### AR_RESUME_DIR
- **Description**: When set to a directory path, the per-block tuning loop checkpoints its progress there after each completed block, and resumes from the first not-yet-completed block on a fresh run against the same directory -- instead of restarting the whole tuning pass from block 0 after a crash or kill.
- **Default**: unset (no resumability)
- **Valid Values**: any writable directory path
- **Usage**: Set this for long-running quantization jobs on large checkpoints where a mid-run crash would otherwise be expensive to restart from scratch.

```bash
export AR_RESUME_DIR=/path/to/resume/state
```

Cross-mode resume: the serial (`AR_DISK_STREAM_MODEL`), streaming (`--stream_quantization`), and block-parallel (BPT) execution paths share the same per-block manifests. A run interrupted in any mode can be continued in the same or any other mode: shards already written for completed blocks are adopted as-is (never overwritten or re-quantized), and blocks a crashed BPT run tuned but never packed get their worker results (scale/zp) applied and packed instead of re-searched.

## Usage Examples

### Setting Environment Variables

#### Using Shell Commands
```bash
# Set logging level to DEBUG
export AR_LOG_LEVEL=DEBUG

# Enable compile packing
export AR_ENABLE_COMPILE_PACKING=1

# Use ModelScope for downloads
export AR_USE_MODELSCOPE=true

# Set custom workspace
export AR_WORK_SPACE=/tmp/autoround_workspace
```

#### Using Python Code
```python
from auto_round.envs import set_config

# Configure multiple environment variables at once
set_config(
    AR_LOG_LEVEL="DEBUG",
    AR_USE_MODELSCOPE=True,
    AR_ENABLE_COMPILE_PACKING=True,
    AR_WORK_SPACE="/tmp/autoround_workspace",
)
```

### Checking Environment Variables

#### Using Python Code
```python
from auto_round import envs

# Access environment variables (lazy evaluation)
log_level = envs.AR_LOG_LEVEL
use_modelscope = envs.AR_USE_MODELSCOPE
enable_packing = envs.AR_ENABLE_COMPILE_PACKING
workspace = envs.AR_WORK_SPACE

print(f"Log Level: {log_level}")
print(f"Use ModelScope: {use_modelscope}")
print(f"Enable Compile Packing: {enable_packing}")
print(f"Workspace: {workspace}")
```

#### Checking if Variables are Explicitly Set
```python
from auto_round.envs import is_set

# Check if environment variables are explicitly set
if is_set("AR_LOG_LEVEL"):
    print("AR_LOG_LEVEL is explicitly set")
else:
    print("AR_LOG_LEVEL is using default value")
```

## Configuration Best Practices

1. **Development Environment**: Set `AR_LOG_LEVEL=TRACE` or `AR_LOG_LEVEL=DEBUG` for detailed logging during development
2. **Production Environment**: Use `AR_LOG_LEVEL=WARNING` or `AR_LOG_LEVEL=ERROR` to reduce log noise
3. **Chinese Users**: Consider setting `AR_USE_MODELSCOPE=true` for better model download performance
4. **Performance Optimization**: Enable `AR_ENABLE_COMPILE_PACKING=1` if you have sufficient computational resources
5. **Custom Workspace**: Set `AR_WORK_SPACE` to a directory with sufficient disk space for model processing

## Notes

- Environment variables are evaluated lazily, meaning they are only read when first accessed
- The `set_config()` function provides a convenient way to configure multiple variables programmatically
- Boolean values for `AR_USE_MODELSCOPE` are automatically converted to appropriate string representations
- All environment variable names are case-sensitive
- Changes made through `set_config()` will affect the current process and any child processes
