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
- **Description**: Sinkhorn loop backend for the Pre-SINQ transform. `"auto"` (default) runs the sinkhorn iteration loop as a fused `torch.compile` graph on non-CPU devices — one fused kernel instead of ~25 launches per iteration, most impactful for the small per-expert folds of MoE models — and uses the eager reference loop on CPU. `"compile"` forces the compiled loop on any device; `"eager"` always uses the reference loop. Both sides evaluate identical fp64 math: the eager loop is bit-exact with the previous release (per-iteration stds are computed once and reused, and the unused scaled-matrix materialization is skipped — ~1.8x faster by itself), while the compiled loop agrees with it to ~1e-15 (its tracker uses a 1e-12 comparison slack so that exact-equality decisions — which eager makes exactly — cannot flip under compiled reduction-order noise). Any compile or runtime failure permanently reverts to the eager loop for the rest of the run.
- **Default**: `"auto"`
- **Valid Values**: `"auto"`, `"compile"`, `"eager"` (unrecognized values are treated as `"eager"`)
- **Usage**: Force the reference loop for A/B comparisons, or fuse on CPU for experimentation

```bash
export AR_PRESINQ_BACKEND=eager
```

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

### AR_RESUME_DIR
- **Description**: When set to a directory path, the per-block tuning loop checkpoints its progress there after each completed block, and resumes from the first not-yet-completed block on a fresh run against the same directory -- instead of restarting the whole tuning pass from block 0 after a crash or kill.
- **Default**: unset (no resumability)
- **Valid Values**: any writable directory path
- **Usage**: Set this for long-running quantization jobs on large checkpoints where a mid-run crash would otherwise be expensive to restart from scratch.

```bash
export AR_RESUME_DIR=/path/to/resume/state
```

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
