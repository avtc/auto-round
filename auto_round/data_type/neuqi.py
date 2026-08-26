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
"""Near-optimal uniform (scale, zero-point) search for asymmetric integer quantization.

This is a clean-room reimplementation of the integer-zero-point variant of the NeUQI
objective (arXiv 2505.17595, "NeUQI: Near-Optimal Uniform Quantization Parameter
Initialization for Low-Bit LLMs"). The reference implementation carries no license,
so no code is copied from it; this module implements the published optimization from
scratch. Only the integer-zero-point variant is implemented because the paper's
floating-point zero point is not consumable by stock int4 kernels (GPTQ/AWQ/Marlin
assume an integral zero point).

For each quantization group the following problem is solved:

    min_{s > 0, z in {0, .., 2**bits - 1}}
        sum_i qw_i * (s * (clamp(round(w_i / s) + z, 0, 2**bits - 1) - z) - w_i)**2

where ``qw`` is an optional per-element weighting (e.g. the activation ``imatrix``).
For a fixed scale the loss decomposes over the integer zero points, so the optimal
``z`` is found exactly by enumerating all ``2**bits`` candidates (16 for 4-bit).
The scale is searched on a coarse log grid bounded by the min-max scale (scales
beyond the min-max range are never optimal) followed by a fine additive refinement
around the best coarse entry.

The search replaces the plain min/max initialization of the asymmetric optimized-RTN
path and is a strict generalization of the symmetric ``search_scales`` grid (which
fixes ``z = 0``).

Performance: the per-candidate zero-point sweep is a pure elementwise + reduction
chain, so on CUDA it is fused into a single generated kernel via ``torch.compile``
(``AR_NEUQI_BACKEND=auto``, the default), and all coarse/fine candidates of a pass
are additionally folded into one such kernel per group chunk (``AR_NEUQI_BATCH``)
so the search costs a handful of launches instead of candidates x chunks. On CUDA
the batched chunk is preferentially served by a hand-written Triton kernel
(``auto_round_extension.triton.neuqi_sweep``) that loads each weight element into
registers once and computes every (candidate, zero-point) loss from registers,
removing the L2 re-read amplification of the compiled expression. Measured on an
RTX 3090 (2M groups, g=64, bits=4, 64+32 grid): 12.3 s eager -> 0.59 s fused
per-candidate -> 0.49 s compiled-batched -> 0.22 s Triton (~57x). The fused sweeps
evaluate exactly the same losses on exactly the same candidate grid; only the fp32
summation order inside the reduction may differ, so selections can flip solely on
near-ties: on random, heavy-tailed and imatrix-weighted inputs the flips stay below
~0.1% of groups with worst-case relative loss differences of ~5e-5 (measured on an
RTX 3090; ~1e-6 on CPU), with the Triton kernel matching that tie profile exactly.
The eager chunked sweep remains the reference path (CPU default) and the ultimate
fallback; every stage latches down permanently on failure (Triton -> compiled
batched -> compiled per-candidate -> eager).
"""

import math
import threading
from functools import lru_cache

import torch

from auto_round import envs
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger

# Upper bound on the number of elements of the temporary [chunk, group, zero_point]
# loss tensor, used to cap peak memory regardless of the input size. The bound is
# honored by both the eager and the fused sweep, so a dynamo recompile-limit fallback
# to eager execution degrades speed only, never correctness or memory.
_MAX_TMP_ELEMS = 2**25


@lru_cache(maxsize=1)
def _log_search_engaged(coarse_n: int, fine_n: int) -> None:
    logger.info("[NeUQI] joint (scale, zero-point) search active (coarse=%d, fine=%d)", coarse_n, fine_n)


@torch.compiler.disable
def _coarse_grid(lo: float, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Log-spaced coarse scale fractions, always built eagerly.

    torch.logspace lowers to an fp64 ``libdevice.pow`` inside compiled
    regions, and inductor has been observed to fuse that into the batched
    sweep's index-select math, where Triton rejects fp64 in the integer
    index guards ("unexpected type fp64" -- BPT worker crash, first MoE
    block under ``--enable_torch_compile``). Building the tiny [n] grid
    eagerly keeps it a plain tensor input to any traced region.

    Deliberately NOT cached: the expert search fans out from a thread pool
    (one thread per device), and a cached tensor would be created on the
    first caller's stream and read from the others' streams with no event
    ordering -- a cross-stream read-before-write window that surfaced as a
    device-side assert on the streaming parent (hidden under
    CUDA_LAUNCH_BLOCKING=1). Recreating 64 floats per call is free and
    keeps creation on the calling thread's own stream.
    """
    return torch.logspace(math.log10(lo), 0.0, n, device=device, dtype=dtype)


def _zp_expr_last(data, qw, scale, zp_grid, maxq):
    """Zero-point sweep with the group dimension last: ``[C, Z, g]``, sum over ``dim=-1``.

    Reduction over the contiguous last dimension is the canonical fused shape for
    the Triton/CUDA codegen: the scheduler can keep the whole pointwise chain
    (round -> clamp -> dequant -> squared error -> group sum -> zero-point min) in
    registers and stream the group axis with coalesced loads. The equivalent
    middle-dim reduction (``[C, g, Z]``) measured no faster than eager on CUDA
    (RTX 3090) because the expanded buffer must be materialized or transposed to
    reduce a non-contiguous axis; it is kept as ``_zp_expr_mid`` where it is the
    faster layout (CPU, including eager execution).

    Same operands and operand order per logical (group, zero-point) element as the
    reference sweep, so losses are identical up to the fp32 summation order over
    the group (last-ulp ties only).

    Args:
        data: [chunk, g] float32 group data.
        qw: [chunk, g] float32 per-element weights or ``None``.
        scale: [chunk, 1] float32 scale candidate.
        zp_grid: [n_zp] float32 integral zero-point candidates ``0 .. maxq``.
        maxq: maximum integer value (``2**bits - 1``).

    Returns:
        (min_loss [chunk], argmin_zp [chunk]) with the first-minimum tie rule of
        ``Tensor.min`` (same winner as the eager sweep on equal losses).
    """
    # Stabilize rounding: values far outside the grid saturate identically after
    # clamping, so clamping r first never changes clamp(r + z, 0, maxq) for z >= 0.
    # Broadcasts give [C, Z, g]: data/qw over the middle (stride-0) zero-point
    # axis, scale shared across the group, reduction over the last (contiguous) dim.
    d = data.unsqueeze(-2)  # [C, 1, g]
    sc = scale.unsqueeze(-1)  # [C, 1, 1]
    zp = zp_grid.view(1, -1, 1)  # [1, Z, 1]
    r = torch.round(d / sc).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r + zp).clamp_(0, maxq)
    deq = sc * (q - zp)
    diff = deq - d
    loss = diff * diff
    if qw is not None:
        loss = loss * qw.unsqueeze(-2)
    loss = loss.sum(dim=-1)  # [C, Z]
    return loss.min(dim=-1)


def _zp_expr_mid(data, qw, scale, zp_grid, maxq):
    """Zero-point sweep with the zero-point dimension last: ``[C, g, Z]``, sum over ``dim=1``.

    Byte-identical operand order to the historical chunked sweep. This is the
    faster layout for eager execution and for the compiled CPU backend (the
    zero-point broadcast rides the contiguous last axis, which TensorIterator
    vectorizes well); it loses to ``_zp_expr_last`` under the Triton/CUDA
    scheduler for the reasons given above.

    Same args and returns as ``_zp_expr_last``.
    """
    # Stabilize rounding: values far outside the grid saturate identically after
    # clamping, so clamping r first never changes clamp(r + z, 0, maxq) for z >= 0.
    r = torch.round(data / scale).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r.unsqueeze(-1) + zp_grid).clamp_(0, maxq)
    deq = scale.unsqueeze(-1) * (q - zp_grid)
    diff = deq - data.unsqueeze(-1)
    loss = diff * diff
    if qw is not None:
        loss = loss * qw.unsqueeze(-1)
    loss = loss.sum(dim=1)  # [chunk, n_zp]
    return loss.min(dim=-1)


def _zp_expr_batched(data, qw, scales, zp_grid, maxq):
    """All-candidates zero-point sweep in one call: ``[C, K, Z, g]``, group last.

    Evaluates every per-group scale candidate (``scales``, [C, K]) against every
    integer zero point in a single fused kernel: the group-axis reduction and the
    min over zero points are fused in-kernel, so the kernel outputs only the
    [C, K] best losses and [C, K] winning zero points -- one launch covers what
    the per-candidate sweep needs K launches and K bookkeeping rounds for. This
    removes the Python-dispatch bound of the per-candidate loop (the residual
    cost after single-candidate fusion) at the price of a larger symbolic
    intermediate, which is why eager execution of this expression is never used
    (see ``_eval_batched_chunk``).

    Same per-(group, candidate, zero-point) operand order as ``_zp_expr_last``;
    the caller applies the first-minimum-over-candidates rule with ``Tensor.min``,
    matching the sequential sweep's strict-improvement tie semantics.

    Args:
        data: [chunk, g] float32 group data.
        qw: [chunk, g] float32 per-element weights or ``None``.
        scales: [chunk, K] float32 per-group scale candidates.
        zp_grid: [n_zp] float32 integral zero-point candidates ``0 .. maxq``.
        maxq: maximum integer value (``2**bits - 1``).

    Returns:
        (best_loss [chunk, K], best_zp [chunk, K]) -- per-candidate minimum over
        zero points, first-minimum tie rule.
    """
    d = data.unsqueeze(-2).unsqueeze(-2)  # [C, 1, 1, g]
    sc = scales.unsqueeze(-1).unsqueeze(-1)  # [C, K, 1, 1]
    zp = zp_grid.view(1, 1, -1, 1)  # [1, 1, Z, 1]
    r = torch.round(d / sc).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r + zp).clamp_(0, maxq)
    deq = sc * (q - zp)
    diff = deq - d
    loss = diff * diff
    if qw is not None:
        loss = loss * qw.unsqueeze(-2).unsqueeze(-2)
    loss = loss.sum(dim=-1)  # [C, K, Z]
    return loss.min(dim=-1)  # [C, K], [C, K]


def _zp_expr_for(device_type: str):
    """Pick the sweep layout for a device, per ``AR_NEUQI_LAYOUT``.

    ``auto`` (default) selects by device: the last-dim-group reduction on CUDA
    (canonical Triton fusion shape), the zero-point-last layout elsewhere (faster
    under TensorIterator and the compiled CPU backend). "last"/"mid" force a
    layout for A/B measurements.
    """
    layout = getattr(envs, "AR_NEUQI_LAYOUT", "auto")
    if layout == "last":
        return _zp_expr_last
    if layout == "mid":
        return _zp_expr_mid
    return _zp_expr_last if device_type == "cuda" else _zp_expr_mid


# torch.compile'd zero-point sweeps, keyed by the expression (layout) -- lazily
# created on first CUDA use / forced backend.
_fused_zp_fns = {}
# Set when a fused invocation raised (e.g. no host C++ compiler for Inductor);
# the process then permanently uses the eager sweep.
_fused_zp_broken = False
# Set when the all-candidates batched sweep raised; the process then permanently
# uses the per-candidate fused sweep.
_batched_broken = False


def _zp_wants_compile(device_type: str) -> bool:
    """Whether a fused torch.compile sweep should be used, per ``AR_NEUQI_BACKEND``."""
    if _BACKEND_OVERRIDE == "eager":
        return False
    backend = _BACKEND_OVERRIDE or envs.AR_NEUQI_BACKEND
    if backend in ("compile", "triton"):
        return True
    if backend == "auto":
        return device_type == "cuda"
    return False  # "eager" (and any unrecognized value): reference sweep


# Extension Triton sweep (auto_round_extension.triton.neuqi_sweep), lazily
# resolved once; ``None`` when the extension/triton is unavailable on the host.
_triton_sweep = None
_triton_checked = False
_triton_broken = False


def _triton_sweep_fn():
    """Resolve the extension Triton sweep once; ``None`` when unavailable."""
    global _triton_sweep, _triton_checked
    if not _triton_checked:
        _triton_checked = True
        try:
            from auto_round_extension.triton.neuqi_sweep import neuqi_sweep_triton

            _triton_sweep = neuqi_sweep_triton
            logger.info("[NeUQI] Triton zero-point sweep engaged (auto_round_extension.triton.neuqi_sweep)")
        except Exception as e:
            logger.info("[NeUQI] Triton sweep unavailable (%s); using the torch.compile sweeps", e)
            _triton_sweep = None
    return _triton_sweep


_BACKEND_OVERRIDE = None  # set by backend_override(); None = env-driven

_sweep_warm_lock = threading.Lock()
_sweep_warmed_devices: set = set()  # device indices whose Triton sweep was warmed once


def ensure_sweep_warmup(device) -> None:
    """Serialized, synchronized first launch of the extension Triton sweep.

    The first Triton launch JIT-compiles the kernel, initializes the launcher
    and writes the on-disk cache; doing that while unrelated GPU work is in
    flight (e.g. inductor compile workers active during SignRound wrapper
    init, or several fan-out threads first-hitting the sweep at once) has been
    observed to leave the sweep producing corrupt selections that surface as
    out-of-bounds gather indices downstream (device-side asserts, vanishing
    under CUDA_LAUNCH_BLOCKING=1). Warming once per device -- under a lock,
    followed by a full device synchronize -- makes every subsequent launch a
    cached, race-free launch. Failures latch the existing Triton-broken
    fallback instead of raising.
    """
    if device.type != "cuda":
        return
    with _sweep_warm_lock:
        if device.index in _sweep_warmed_devices:
            return
        _sweep_warmed_devices.add(device.index)
        if not _zp_wants_triton(device.type):
            return
        fn = _triton_sweep_fn()
        if fn is None:
            return
        global _triton_broken
        try:
            data = torch.randn(4, 128, device=device, dtype=torch.float32)
            scales = torch.rand(4, 2, device=device, dtype=torch.float32) * 0.01 + 0.001
            loss, _zp = fn(data, torch.ones_like(data), scales, 15)
            torch.cuda.synchronize(device)
            if not torch.isfinite(loss).all():
                raise RuntimeError("warmup produced non-finite losses")
            logger.info("[NeUQI] Triton sweep warmed on %s", device)
        except Exception as e:  # noqa: BLE001  warmup failure must never kill a run
            _triton_broken = True
            logger.warning("[NeUQI] Triton sweep warmup failed on %s (%s); using torch.compile sweeps", device, e)


class backend_override:
    """Context manager forcing the sweep backend (bypasses AR_NEUQI_BACKEND).

    The extension Triton sweep intermittently device-asserts when invoked
    from SignRound wrapper init (recipe init-searches) on multi-GPU
    data-driven runs -- passes under CUDA_LAUNCH_BLOCKING=1, never faults in
    the zero-shot quantizer that also uses it. Until that is root-caused,
    wrapper-init searches pin the torch.compile backend: one-shot searches
    cost ~0.2s/layer there vs Triton's ~0.07s -- noise over a whole model.
    """

    def __init__(self, backend: str):
        self._backend = backend

    def __enter__(self):
        global _BACKEND_OVERRIDE
        self._saved = _BACKEND_OVERRIDE
        _BACKEND_OVERRIDE = self._backend
        return self

    def __exit__(self, *exc):
        global _BACKEND_OVERRIDE
        _BACKEND_OVERRIDE = self._saved
        return False


def _zp_wants_triton(device_type: str) -> bool:
    """Whether the extension Triton sweep should be tried, per ``AR_NEUQI_BACKEND``."""
    if envs.AR_NEUQI_BACKEND == "triton":
        return True  # explicit user intent outranks the wrapper-init pin
    if _BACKEND_OVERRIDE == "compile" or _BACKEND_OVERRIDE == "eager":
        return False
    backend = _BACKEND_OVERRIDE or envs.AR_NEUQI_BACKEND
    if backend == "triton":
        return True
    return backend == "auto" and device_type == "cuda"


def _get_fused_zp_fn(expr):
    """Lazily compile one sweep layout; callers fall back to eager on any failure."""
    fn = _fused_zp_fns.get(expr)
    if fn is None:
        from auto_round.utils.device import _bump_dynamo_cache_limit

        # Shape variants (group size x bits x qw-present x chunk tail) must never
        # trip dynamo's recompile limit: exceeding it silently runs the frame in
        # eager, which is memory-safe here (chunking bounds the expansion) but slow.
        _bump_dynamo_cache_limit(64)
        fn = torch.compile(expr, dynamic=False)
        _fused_zp_fns[expr] = fn
        logger.info("[NeUQI] zero-point sweep fused via torch.compile (layout=%s)", expr.__name__)
    return fn


def _eval_zp_chunk(data, qw, scale, zp_grid, maxq):
    """Dispatch one chunk of the zero-point sweep to the fused or eager path."""
    global _fused_zp_broken
    expr = _zp_expr_for(data.device.type)
    if not _fused_zp_broken and _zp_wants_compile(data.device.type):
        try:
            return _get_fused_zp_fn(expr)(data, qw, scale, zp_grid, maxq)
        except Exception as e:  # pragma: no cover - depends on host toolchain
            _fused_zp_broken = True
            logger.warning("[NeUQI] fused zero-point sweep failed (%s); using the eager sweep", e)
    return expr(data, qw, scale, zp_grid, maxq)


def _zp_batch_wanted(device_type: str) -> bool:
    """Whether the all-candidates batched sweep should run, per ``AR_NEUQI_BATCH``.

    "auto" (default) batches on CUDA when the fused backend engages there;
    "on" forces it on any fused-capable device (tests, A/B); "off" keeps the
    per-candidate loop. The batched expression is only ever executed compiled:
    eager execution of [C, K, Z, g] would materialize the full expansion.
    """
    mode = getattr(envs, "AR_NEUQI_BATCH", "auto")
    if mode == "off":
        return False
    if mode == "on":
        return True
    return device_type == "cuda" and _zp_wants_compile(device_type)


def _eval_batched_chunk(data, qw, scales, zp_grid, maxq):
    """One chunk of the all-candidates batched sweep, or ``None`` when unavailable.

    Preference order: the extension Triton sweep (registers-resident, one data
    read per element -- the fastest evaluator on CUDA), then the compiled
    batched expression, then ``None`` so the caller falls back to the
    per-candidate sweep. Strict-improvement bookkeeping makes a partial pass
    followed by a fallback converge to the same selections (ties within the
    usual fp32 summation tolerance).
    """
    global _triton_broken, _batched_broken
    if not _triton_broken and _zp_wants_triton(data.device.type) and _zp_batch_wanted(data.device.type):
        fn = _triton_sweep_fn()
        if fn is not None:
            try:
                return fn(data, qw, scales, maxq)
            except Exception as e:
                _triton_broken = True
                if data.device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.warning("[NeUQI] Triton sweep failed (%s); using the torch.compile sweeps", e)
    if _batched_broken or not _zp_wants_compile(data.device.type):
        return None
    try:
        return _get_fused_zp_fn(_zp_expr_batched)(data, qw, scales, zp_grid, maxq)
    except Exception as e:
        _batched_broken = True
        if data.device.type == "cuda":
            torch.cuda.empty_cache()  # a failed launch may hold pooled memory
        logger.warning("[NeUQI] batched zero-point sweep failed (%s); using the per-candidate sweep", e)
        return None


def _best_zp_for_scale(data, qw, scale, zp_grid, maxq, chunk):
    """Evaluate all integer zero points for one per-group scale candidate.

    Args:
        data: [N, g] float32 group data.
        qw: [N, g] float32 per-element weights or ``None``.
        scale: [N, 1] float32 scale candidates.
        zp_grid: [n_zp] float32 integral zero-point candidates.
        maxq: maximum integer value (``2**bits - 1``).
        chunk: number of groups per evaluation slice (bounds the temporary).

    Returns:
        best_loss: [N] float32, loss of the best zero point per group.
        best_zp: [N] float32, integral-valued optimal zero point per group.
    """
    n_groups = data.shape[0]
    best_loss = torch.full((n_groups,), float("inf"), device=data.device, dtype=data.dtype)
    best_zp = torch.zeros(n_groups, device=data.device, dtype=data.dtype)

    for start in range(0, n_groups, chunk):
        stop = min(start + chunk, n_groups)
        min_loss, argmin_zp = _eval_zp_chunk(
            data[start:stop],
            qw[start:stop] if qw is not None else None,
            scale[start:stop],
            zp_grid,
            maxq,
        )
        best_loss[start:stop] = min_loss
        best_zp[start:stop] = argmin_zp.to(data.dtype)

    return best_loss, best_zp


def _lower_scale_ratio(bits: int) -> float:
    """Fraction of the min-max scale below which clipping is never beneficial."""
    if bits >= 4:
        return 0.25
    if bits == 3:
        return 0.15
    return 0.05


def neuqi_search_scale_zero(data, bits, qw=None, q_scale_thresh=1e-5, coarse_n=None, fine_n=None):
    """Joint near-optimal (scale, integer zero-point) search per quantization group.

    Args:
        data: [N, g] float32 tensor of already-grouped weights.
        bits: quantization bit width.
        qw: optional [N, g] float32 per-element loss weights (e.g. imatrix).
        q_scale_thresh: minimum scale magnitude.
        coarse_n: number of coarse log-spaced scale candidates (env ``AR_NEUQI_COARSE``).
        fine_n: number of fine additive candidates around the best coarse entry
            (env ``AR_NEUQI_FINE``).

    Returns:
        scale: [N, 1] float32 per-group scales.
        zp: [N, 1] float32 per-group integral-valued zero points.
    """
    if coarse_n is None:
        coarse_n = envs.AR_NEUQI_COARSE if envs.AR_NEUQI_COARSE else 64
    if fine_n is None:
        fine_n = envs.AR_NEUQI_FINE if envs.AR_NEUQI_FINE else 32
    _log_search_engaged(coarse_n, fine_n)
    if torch.is_tensor(data):
        ensure_sweep_warmup(data.device)

    maxq = int(2**bits) - 1
    data = data.to(torch.float32)
    if qw is not None:
        qw = qw.to(torch.float32)

    zp_grid = torch.arange(0, maxq + 1, device=data.device, dtype=data.dtype)
    chunk = max(1, _MAX_TMP_ELEMS // (data.shape[1] * (maxq + 1)))

    wmin = torch.clamp(data.min(dim=-1).values, max=0).unsqueeze(-1)
    wmax = torch.clamp(data.max(dim=-1).values, min=0).unsqueeze(-1)
    s0 = ((wmax - wmin) / maxq).clamp_(min=q_scale_thresh)  # [N, 1] min-max scale

    lo = _lower_scale_ratio(bits)
    coarse = _coarse_grid(lo, coarse_n, data.device, torch.float32)

    best_loss = torch.full((data.shape[0],), float("inf"), device=data.device, dtype=torch.float32)
    best_frac = torch.ones(data.shape[0], device=data.device, dtype=torch.float32)
    best_zp = torch.zeros(data.shape[0], device=data.device, dtype=torch.float32)
    n_groups = data.shape[0]
    batch_wanted = _zp_batch_wanted(data.device.type)

    # Coarse pass: shared log-spaced fractions of the per-group min-max scale.
    # Batched fast path first: every candidate in one fused call per group chunk
    # (AR_NEUQI_BATCH); the per-candidate loop is the reference and the fallback.
    coarse_done = False
    if batch_wanted:
        chunk_b = max(1, _MAX_TMP_ELEMS // (coarse_n * (maxq + 1)))
        for start in range(0, n_groups, chunk_b):
            stop = min(start + chunk_b, n_groups)
            scales = s0[start:stop] * coarse.view(1, -1)  # [chunk, K] shared fracs
            out = _eval_batched_chunk(
                data[start:stop], qw[start:stop] if qw is not None else None, scales, zp_grid, maxq
            )
            if out is None:
                break  # latched failure / fused backend off: redo per candidate
            loss_k, zp_k = out  # [chunk, K] each
            loss, k_idx = loss_k.min(dim=1)  # first-min over candidates
            zp = zp_k.gather(1, k_idx.unsqueeze(1)).squeeze(1)
            frac = coarse.index_select(0, k_idx)
            improved = loss < best_loss[start:stop]
            best_loss[start:stop] = torch.where(improved, loss, best_loss[start:stop])
            best_zp[start:stop] = torch.where(improved, zp, best_zp[start:stop])
            best_frac[start:stop] = torch.where(improved, frac, best_frac[start:stop])
        else:
            coarse_done = True
    if not coarse_done:
        for idx in range(coarse_n):
            scale = s0 * coarse[idx]
            loss, zp = _best_zp_for_scale(data, qw, scale, zp_grid, maxq, chunk)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_zp = torch.where(improved, zp, best_zp)
            best_frac = torch.where(improved, torch.full_like(best_frac, coarse[idx]), best_frac)

    # Fine pass: additive grid between the coarse neighbors bracketing each winner.
    best_idx = torch.argmin(torch.abs(coarse.unsqueeze(0) - best_frac.unsqueeze(1)), dim=1)
    frac_lo = coarse[(best_idx - 1).clamp_(min=0)]
    frac_hi = coarse[(best_idx + 1).clamp_(max=coarse_n - 1)]

    steps = torch.arange(1, fine_n + 1, device=data.device, dtype=torch.float32) / (fine_n + 1)
    fine_done = fine_n == 0
    if batch_wanted and not fine_done:
        # per-group fracs: frac_lo/hi differ per group, steps shared
        one_m_steps = 1.0 - steps
        chunk_b = max(1, _MAX_TMP_ELEMS // (fine_n * (maxq + 1)))
        for start in range(0, n_groups, chunk_b):
            stop = min(start + chunk_b, n_groups)
            fracs = frac_lo[start:stop].unsqueeze(1) * one_m_steps + frac_hi[start:stop].unsqueeze(1) * steps
            scales = s0[start:stop] * fracs  # [chunk, K]
            out = _eval_batched_chunk(
                data[start:stop], qw[start:stop] if qw is not None else None, scales, zp_grid, maxq
            )
            if out is None:
                break
            loss_k, zp_k = out
            loss, k_idx = loss_k.min(dim=1)
            zp = zp_k.gather(1, k_idx.unsqueeze(1)).squeeze(1)
            frac = fracs.gather(1, k_idx.unsqueeze(1)).squeeze(1)
            improved = loss < best_loss[start:stop]
            best_loss[start:stop] = torch.where(improved, loss, best_loss[start:stop])
            best_zp[start:stop] = torch.where(improved, zp, best_zp[start:stop])
            best_frac[start:stop] = torch.where(improved, frac, best_frac[start:stop])
        else:
            fine_done = True
    if not fine_done:
        for step in steps:
            frac = frac_lo * (1.0 - step) + frac_hi * step
            scale = s0 * frac.unsqueeze(-1)
            loss, zp = _best_zp_for_scale(data, qw, scale, zp_grid, maxq, chunk)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_zp = torch.where(improved, zp, best_zp)
            best_frac = torch.where(improved, frac, best_frac)

    scale = (best_frac.unsqueeze(-1) * s0).clamp_(min=q_scale_thresh)
    return scale, best_zp.unsqueeze(-1)


@register_dtype("opt_rtn_int_asym")
def quant_tensor_opt_rtn_asym(
    tensor, bits=4, group_size=-1, v=0, q_scale_thresh=1e-5, imatrix=None, scale_dtype=torch.float16, **kwargs
):
    """Quantize/dequantize with a joint near-optimal (scale, integer zero-point) search.

    Asymmetric counterpart of ``opt_rtn_int_sym``: fills the previously missing
    optimized path for asymmetric int quantization (plain min/max before). Set the

    Args:
        tensor: Tensor to quantize.
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8).
        group_size: Number of elements sharing a scale.
        v: Rounding value perturbation.
        q_scale_thresh: Minimum scale magnitude for numerical stability.
        imatrix: Optional per-column importance weights (activation imatrix).
        scale_dtype: dtype of the returned scale (kernels support fp16/fp32).

    Returns:
        (qdq_result, scale, zero_point) matching ``quant_tensor_asym`` conventions.
    """
    from auto_round.data_type.gguf import _imatrix_handle_zero

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2**bits) - 1

    qw = None
    if imatrix is not None:
        if imatrix.dim() == 1:
            # per-column importance shared by every row of this tensor
            qw = imatrix.reshape(1, -1)
            qw = reshape_pad_tensor_by_group_size(qw, group_size, val=1e-5)[0].view(1, -1)
            qw = qw.expand(tensor.numel() // qw.numel(), -1)
            qw = qw.reshape(tensor.shape)
        else:
            # per-row importance (stacked same-shape modules: each module keeps
            # its own column weights); must already match the tensor's shape
            qw = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0]
            if qw.shape != tensor.shape:
                raise ValueError(
                    f"per-row imatrix shape {tuple(imatrix.shape)} incompatible with tensor {tuple(tensor.shape)}"
                )
        qw = _imatrix_handle_zero(qw, tensor, bits, group_size)

    scale, zp = neuqi_search_scale_zero(tensor.to(torch.float32), bits, qw=qw, q_scale_thresh=q_scale_thresh)
    scale = torch.clamp(scale.to(scale_dtype), min=q_scale_thresh)
    zp = zp.to(scale_dtype)

    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


logger.info("[NeUQI] opt_rtn_int_asym registered (clean-room, arXiv 2505.17595)")
