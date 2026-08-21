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
"""

import math
from functools import lru_cache

import torch

from auto_round import envs
from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger

# Upper bound on the number of elements of the temporary [chunk, group, zero_point]
# loss tensor, used to cap peak memory regardless of the input size.
_MAX_TMP_ELEMS = 2**25


@lru_cache(maxsize=1)
def _log_search_engaged(coarse_n: int, fine_n: int) -> None:
    logger.info("[NeUQI] joint (scale, zero-point) search active (coarse=%d, fine=%d)", coarse_n, fine_n)


@lru_cache(maxsize=1)
def _log_disabled() -> None:
    logger.info("[NeUQI] disabled via AR_DISABLE_NEUQI; using plain min/max asym quantization")


def _hist_chunk_eval(data_c, qwv, Aw, Bw, scale_c, maxq):
    """Zero-point sweep for one chunk and one scale candidate (see
    ``_loss_all_zp_hist``). Per-element products are passed in so callers can
    reuse them across all candidates of a search. Scatters accumulate in fp32;
    only the small [chunk, bins] aggregates are composed in fp64 (the expanded
    loss is a difference of large terms, and float32 cancellation flips
    near-optimal choices on heavy-tailed groups)."""
    n_zp = maxq + 1
    rmin, rmax = -(maxq + 1), 2 * maxq + 1
    nbins = rmax - rmin + 1

    r = torch.round(data_c / scale_c).clamp_(min=rmin, max=rmax)  # integer-valued
    bins = (r - rmin).to(torch.int64)  # [c, g] bin index per element
    zeros32 = torch.zeros(data_c.shape[0], nbins, device=data_c.device, dtype=torch.float32)
    n_b = zeros32.scatter_add_(1, bins, qwv)  # per-bin sum of weights
    A_b = torch.zeros_like(zeros32).scatter_add_(1, bins, Aw)  # sum qw*w
    B_b = torch.zeros_like(zeros32).scatter_add_(1, bins, Bw)  # sum qw*w^2
    n_b, A_b, B_b = n_b.double(), A_b.double(), B_b.double()
    b = torch.arange(rmin, rmax + 1, device=data_c.device, dtype=torch.float64)
    bn_b = b.unsqueeze(0) * A_b  # sum qw*r*w (r constant per bin)
    b2n_b = (b * b).unsqueeze(0) * n_b  # sum qw*r^2

    # exclusive prefix sums: P[:, i] = aggregates of bins < i
    P_n = torch.cat([torch.zeros_like(n_b[:, :1]), n_b.cumsum(dim=1)], dim=1)
    P_A = torch.cat([torch.zeros_like(A_b[:, :1]), A_b.cumsum(dim=1)], dim=1)
    P_B = torch.cat([torch.zeros_like(B_b[:, :1]), B_b.cumsum(dim=1)], dim=1)
    P_bn = torch.cat([torch.zeros_like(bn_b[:, :1]), bn_b.cumsum(dim=1)], dim=1)
    P_b2n = torch.cat([torch.zeros_like(b2n_b[:, :1]), b2n_b.cumsum(dim=1)], dim=1)

    z = torch.arange(0, n_zp, device=data_c.device, dtype=torch.float64)
    idx_low = (maxq + 1 - z).long()  # bin index of threshold -z
    idx_high = (2 * maxq + 1 - z).long()  # bin index of threshold maxq-z

    # low: bins <= -z saturate at deq = -s*z
    N_lo = P_n[:, idx_low + 1]
    A_lo = P_A[:, idx_low + 1]
    B_lo = P_B[:, idx_low + 1]
    # high: bins >= maxq-z saturate at deq = s*(maxq-z)
    tot_n, tot_A, tot_B = P_n[:, -1:], P_A[:, -1:], P_B[:, -1:]
    N_hi = tot_n - P_n[:, idx_high]
    A_hi = tot_A - P_A[:, idx_high]
    B_hi = tot_B - P_B[:, idx_high]
    # mid: -z < r < maxq-z keep deq = s*r
    B_md = tot_B - B_lo - B_hi
    M1_md = P_bn[:, -1:] - P_bn[:, idx_low + 1] - (P_bn[:, -1:] - P_bn[:, idx_high])
    M2_md = P_b2n[:, -1:] - P_b2n[:, idx_low + 1] - (P_b2n[:, -1:] - P_b2n[:, idx_high])

    s = scale_c.double()  # [c, 1]
    sz = s * z.unsqueeze(0)  # [c, n_zp]
    m = (maxq - z).unsqueeze(0)  # [c, n_zp]
    loss = (
        (sz * sz) * N_lo
        + 2.0 * sz * A_lo
        + B_lo
        + (s * s) * M2_md
        - 2.0 * s * M1_md
        + B_md
        + (s * s) * (m * m) * N_hi
        - 2.0 * s * m * A_hi
        + B_hi
    )
    min_loss, argmin_zp = loss.min(dim=-1)
    return min_loss.to(torch.float32), argmin_zp.to(torch.float32)


def _loss_all_zp_hist(data, qw, scale, maxq):
    """Evaluate every integer zero point for one scale candidate via binning.

    Mathematically identical to ``_best_zp_for_scale`` (same stabilization
    clamp, same exhaustive z sweep, same first-min tie rule) but reformulated
    for memory bandwidth: r = round(w/s) is integer-valued, so grouping
    elements into r-bins and taking prefix sums over ~3*2^bits bins yields all
    zero-point losses without materializing the [groups, g, 2^bits] grid
    (a 16x data expansion that dominates the brute-force runtime).

    Floating-point summation order differs from the brute force, so losses
    can differ in the last ulp and exact ties may resolve differently.
    """
    n_groups = data.shape[0]
    group_size = data.shape[1]
    chunk = max(1, _MAX_TMP_ELEMS // group_size)

    best_loss = torch.full((n_groups,), float("inf"), device=data.device, dtype=torch.float32)
    best_zp = torch.zeros(n_groups, device=data.device, dtype=torch.float32)

    for start in range(0, n_groups, chunk):
        stop = min(start + chunk, n_groups)
        data_c = data[start:stop]
        qwv = qw[start:stop] if qw is not None else torch.ones_like(data_c)
        Aw = qwv * data_c
        Bw = qwv * data_c * data_c
        loss, zp = _hist_chunk_eval(data_c, qwv, Aw, Bw, scale[start:stop], maxq)
        best_loss[start:stop] = loss
        best_zp[start:stop] = zp

    return best_loss, best_zp

def _best_zp_for_scale(data, qw, scale, maxq):
    """Evaluate all integer zero points for one per-group scale candidate.

    Args:
        data: [N, g] float32 group data.
        qw: [N, g] float32 per-element weights or ``None``.
        scale: [N, 1] float32 scale candidates.
        maxq: maximum integer value (``2**bits - 1``).

    Returns:
        best_loss: [N] float32, loss of the best zero point per group.
        best_zp: [N] float32, integral-valued optimal zero point per group.
    """
    n_groups = data.shape[0]
    group_size = data.shape[1]
    n_zp = maxq + 1
    chunk = max(1, _MAX_TMP_ELEMS // (group_size * n_zp))

    zp_grid = torch.arange(0, n_zp, device=data.device, dtype=data.dtype)

    best_loss = torch.full((n_groups,), float("inf"), device=data.device, dtype=data.dtype)
    best_zp = torch.zeros(n_groups, device=data.device, dtype=data.dtype)

    for start in range(0, n_groups, chunk):
        stop = min(start + chunk, n_groups)
        data_c = data[start:stop]
        qw_c = qw[start:stop] if qw is not None else None
        scale_c = scale[start:stop]

        # Stabilize rounding: values far outside the grid saturate identically after
        # clamping, so clamping r first never changes clamp(r + z, 0, maxq) for z >= 0.
        r = torch.round(data_c / scale_c).clamp_(min=-maxq - 1, max=2 * maxq + 1)
        q = (r.unsqueeze(-1) + zp_grid).clamp_(0, maxq)
        deq = scale_c.unsqueeze(-1) * (q - zp_grid)
        diff = deq - data_c.unsqueeze(-1)
        loss = diff * diff
        if qw_c is not None:
            loss = loss * qw_c.unsqueeze(-1)
        loss = loss.sum(dim=1)  # [chunk, n_zp]

        min_loss, argmin_zp = loss.min(dim=-1)
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

    maxq = int(2**bits) - 1
    data = data.to(torch.float32)
    if qw is not None:
        qw = qw.to(torch.float32)

    wmin = torch.clamp(data.min(dim=-1).values, max=0).unsqueeze(-1)
    wmax = torch.clamp(data.max(dim=-1).values, min=0).unsqueeze(-1)
    s0 = ((wmax - wmin) / maxq).clamp_(min=q_scale_thresh)  # [N, 1] min-max scale

    lo = _lower_scale_ratio(bits)
    coarse = torch.logspace(math.log10(lo), 0.0, coarse_n, device=data.device, dtype=torch.float32)

    best_loss = torch.full((data.shape[0],), float("inf"), device=data.device, dtype=torch.float32)
    best_frac = torch.ones(data.shape[0], device=data.device, dtype=torch.float32)
    best_zp = torch.zeros(data.shape[0], device=data.device, dtype=torch.float32)

    use_hist = envs.AR_NEUQI_SWEEP != "brute"

    best_loss = torch.full((data.shape[0],), float("inf"), device=data.device, dtype=torch.float32)
    best_frac = torch.ones(data.shape[0], device=data.device, dtype=torch.float32)
    best_zp = torch.zeros(data.shape[0], device=data.device, dtype=torch.float32)

    if not use_hist:
        # Coarse pass: shared log-spaced fractions of the per-group min-max scale.
        for idx in range(coarse_n):
            scale = s0 * coarse[idx]
            loss, zp = _best_zp_for_scale(data, qw, scale, maxq)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_zp = torch.where(improved, zp, best_zp)
            best_frac = torch.where(improved, torch.full_like(best_frac, coarse[idx]), best_frac)

        # Fine pass: additive grid between the coarse neighbors bracketing each winner.
        best_idx = torch.argmin(torch.abs(coarse.unsqueeze(0) - best_frac.unsqueeze(1)), dim=1)
        frac_lo = coarse[(best_idx - 1).clamp_(min=0)]
        frac_hi = coarse[(best_idx + 1).clamp_(max=coarse_n - 1)]

        steps = torch.arange(1, fine_n + 1, device=data.device, dtype=torch.float32) / (fine_n + 1)
        for step in steps:
            frac = frac_lo * (1.0 - step) + frac_hi * step
            scale = s0 * frac.unsqueeze(-1)
            loss, zp = _best_zp_for_scale(data, qw, scale, maxq)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_zp = torch.where(improved, zp, best_zp)
            best_frac = torch.where(improved, frac, best_frac)
    else:
        # Histogram sweep: chunk-outer so the per-element products qw*w and
        # qw*w^2 are built once per chunk and reused across all candidates.
        group_size = data.shape[1]
        chunk = max(1, _MAX_TMP_ELEMS // group_size)
        n_groups = data.shape[0]
        for cstart in range(0, n_groups, chunk):
            cstop = min(cstart + chunk, n_groups)
            data_c = data[cstart:cstop]
            s0_c = s0[cstart:cstop]
            qwv = qw[cstart:cstop] if qw is not None else torch.ones_like(data_c)
            Aw = qwv * data_c
            Bw = qwv * data_c * data_c

            c_loss = torch.full((cstop - cstart,), float("inf"), device=data.device, dtype=torch.float32)
            c_frac = torch.ones(cstop - cstart, device=data.device, dtype=torch.float32)
            c_zp = torch.zeros(cstop - cstart, device=data.device, dtype=torch.float32)

            for idx in range(coarse_n):
                loss, zp = _hist_chunk_eval(data_c, qwv, Aw, Bw, s0_c * coarse[idx], maxq)
                improved = loss < c_loss
                c_loss = torch.where(improved, loss, c_loss)
                c_zp = torch.where(improved, zp, c_zp)
                c_frac = torch.where(improved, torch.full_like(c_frac, coarse[idx]), c_frac)

            best_idx = torch.argmin(torch.abs(coarse.unsqueeze(0) - c_frac.unsqueeze(1)), dim=1)
            frac_lo = coarse[(best_idx - 1).clamp_(min=0)]
            frac_hi = coarse[(best_idx + 1).clamp_(max=coarse_n - 1)]
            steps = torch.arange(1, fine_n + 1, device=data.device, dtype=torch.float32) / (fine_n + 1)
            for step in steps:
                frac = frac_lo * (1.0 - step) + frac_hi * step
                loss, zp = _hist_chunk_eval(data_c, qwv, Aw, Bw, s0_c * frac.unsqueeze(-1), maxq)
                improved = loss < c_loss
                c_loss = torch.where(improved, loss, c_loss)
                c_zp = torch.where(improved, zp, c_zp)
                c_frac = torch.where(improved, frac, c_frac)

            best_loss[cstart:cstop] = c_loss
            best_frac[cstart:cstop] = c_frac
            best_zp[cstart:cstop] = c_zp

    scale = (best_frac.unsqueeze(-1) * s0).clamp_(min=q_scale_thresh)
    return scale, best_zp.unsqueeze(-1)


@register_dtype("opt_rtn_int_asym")
def quant_tensor_opt_rtn_asym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    q_scale_thresh=1e-5,
    imatrix=None,
    scale_dtype=torch.float16,
    **kwargs
):
    """Quantize/dequantize with a joint near-optimal (scale, integer zero-point) search.

    Asymmetric counterpart of ``opt_rtn_int_sym``: fills the previously missing
    optimized path for asymmetric int quantization (plain min/max before). Set the
    ``AR_DISABLE_NEUQI`` env var to revert to the plain min/max behavior.

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
    from auto_round.data_type.int import quant_tensor_asym

    if envs.AR_DISABLE_NEUQI:
        _log_disabled()
        return quant_tensor_asym(
            tensor, bits=bits, group_size=group_size, v=v, q_scale_thresh=q_scale_thresh, **kwargs
        )

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


logger.info("[NeUQI] opt_rtn_int_asym registered (clean-room, arXiv 2505.17595; AR_DISABLE_NEUQI=%s)", bool(envs.AR_DISABLE_NEUQI))
