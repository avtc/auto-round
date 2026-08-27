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
"""Triton kernels for the Pre-SINQ Sinkhorn loop (fp64, CUDA).

One iteration of the loop is served by three kernels:

* ``_k1_pass`` (grid: H-chunks x B) reads each ``m`` tile [BH, LP] exactly
  once, folds the mu divisions in registers, and emits the per-row stds
  ``s1[B, H]`` plus per-chunk column partials (sum, sumsq) — no ``cur``
  materialization, no atomics;
* ``_k2_update`` (grid: B) tree-reduces the column partials into column
  stds, reduces the s1 extremes, and runs the imbalance tracker with the
  same hardenings as the compiled arm (tracker seeded from +inf so
  iteration 0 always captures the identity candidate; comparisons with a
  1e-12 relative slack so exact-equality decisions cannot flip under
  reduction-order noise) plus the lm1 / mu1_star updates and, on the first
  iteration, tgt;
* ``_k3_rows`` (grid: H-chunks x B) applies the sal_row / lm2 update and
  captures mu2_star.

Measured on an RTX 3090 (fp64, n_iter=4, tile batches as produced by
``column_scales``): 2.1-2.8x faster than the std-once eager loop across
dense/concat/expert/pooled shapes and ~4.3x vs the original port, with
agreement at fp64-ulp level (~1e-15) — tighter than the compiled arm. On
the pooled MoE norm fold (a [592896, 4096] concat of 386 consumer
matrices, e.g. Hunyuan-A13B-class layers) the eager loop exceeds 24 GiB
and OOMs where these kernels still run.

Two Triton pitfalls are load-bearing here (both found by the staged GPU
bench, both covered by regression tests):

* python float literals are fp32 — a binding clamp would return the fp32
  bound (fp32(0.7) differs from 0.7 by 1.7e-8), so every non-exact
  constant (0.7, -0.3, 1e-3, 1e-6, 1e-12) is a ``tl.full(..., tl.float64)``;
* a load following a store to the same pointer within one program is not
  guaranteed to observe it — tgt is kept in a register on the first
  iteration instead of being loaded back.
"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on host toolchain
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e


if triton is not None:

    @triton.jit
    def _k1_pass(m_ptr, lm1_ptr, lm2_ptr, s1_ptr, csum_ptr, csq_ptr, H, L, NB, BH: tl.constexpr, LP: tl.constexpr):
        pc = tl.program_id(0)
        b = tl.program_id(1)
        rows = pc * BH + tl.arange(0, BH)
        cols = tl.arange(0, LP)
        rmask = rows < H
        cmask = cols < L
        mmask = rmask[:, None] & cmask[None, :]
        tile = tl.load(m_ptr + (b * H + rows[:, None]) * L + cols[None, :], mask=mmask, other=0.0)
        mu1 = tl.exp(tl.load(lm1_ptr + b * L + cols, mask=cmask, other=0.0))
        mu2 = tl.exp(tl.load(lm2_ptr + b * H + rows, mask=rmask, other=0.0))
        cur = tl.where(cmask[None, :], (tile / mu1[None, :]) / mu2[:, None], 0.0)
        Lf = L.to(tl.float64)
        rsum = tl.sum(cur, 1)
        rsq = tl.sum(cur * cur, 1)
        rvar = tl.maximum(rsq / Lf - (rsum / Lf) * (rsum / Lf), 0.0) * (Lf / (Lf - 1.0))
        tl.store(s1_ptr + b * H + rows, tl.sqrt(rvar), mask=rmask)
        cw = tl.where(rmask[:, None], cur, 0.0)  # masked rows contribute zero
        part = (pc * NB + b) * LP + cols
        tl.store(csum_ptr + part, tl.sum(cw, 0))
        tl.store(csq_ptr + part, tl.sum(cw * cw, 0))

    @triton.jit
    def _k2_update(
        csum_ptr,
        csq_ptr,
        s1_ptr,
        lm1_ptr,
        m1s_ptr,
        tgt_ptr,
        imbmin_ptr,
        gate_ptr,
        better_ptr,
        H,
        L,
        NB,
        P,
        FIRST: tl.constexpr,
        LP: tl.constexpr,
        CP: tl.constexpr,
        SH: tl.constexpr,
    ):
        b = tl.program_id(0)
        cols = tl.arange(0, LP)
        cmask = cols < L
        # clamp/add constants as fp64 tensors: python float literals are fp32 in
        # triton, and a binding clamp would return the fp32 bound (e.g. fp32(0.7)
        # differs from 0.7 by 1.7e-8 -- exactly the sal/log error this caused)
        c_lo = tl.full((), 0.7, tl.float64)
        c_hi = tl.full((), 2.0, tl.float64)
        w_lo = tl.full((), 1e-3, tl.float64)
        w_hi = tl.full((), 1e3, tl.float64)
        lm_lo = tl.full((), -0.3, tl.float64)
        lm_hi = tl.full((), 10.0, tl.float64)
        tiny = tl.full((), 1e-12, tl.float64)
        # tree-reduce column partials over H-chunks (deterministic order)
        cs = tl.zeros((LP,), dtype=tl.float64)
        cq = tl.zeros((LP,), dtype=tl.float64)
        for p0 in range(0, P, CP):
            pidx = p0 + tl.arange(0, CP)
            pmask = (pidx < P)[:, None] & cmask[None, :]
            off = (pidx * NB + b)[:, None] * LP + cols[None, :]
            cs += tl.sum(tl.load(csum_ptr + off, mask=pmask, other=0.0), 0)
            cq += tl.sum(tl.load(csq_ptr + off, mask=pmask, other=0.0), 0)
        Hf = H.to(tl.float64)
        cvar = tl.maximum(cq / Hf - (cs / Hf) * (cs / Hf), 0.0) * (Hf / (Hf - 1.0))
        s2 = tl.sqrt(cvar)
        # s1 extremes over H (fp64 accumulators: python float inits are fp32 and
        # would change type inside the loop)
        s1_min_raw = tl.full((SH,), float("inf"), dtype=tl.float64)
        s1_max_raw = tl.full((SH,), float("-inf"), dtype=tl.float64)
        s1_min_c = tl.full((SH,), float("inf"), dtype=tl.float64)
        for h0 in range(0, H, SH):
            hidx = h0 + tl.arange(0, SH)
            vals = tl.load(s1_ptr + b * H + hidx, mask=hidx < H, other=float("inf"))
            s1_min_raw = tl.minimum(s1_min_raw, tl.min(vals, 0))
            s1_max_raw = tl.maximum(s1_max_raw, tl.max(tl.where(hidx < H, vals, float("-inf")), 0))
            s1_min_c = tl.minimum(s1_min_c, tl.min(tl.clamp(vals, w_lo, w_hi), 0))
        s1_min_raw = tl.min(s1_min_raw, 0)
        s1_max_raw = tl.max(s1_max_raw, 0)
        s1_min_c = tl.min(s1_min_c, 0)
        s2_min_raw = tl.min(tl.where(cmask, s2, float("inf")), 0)
        s2_max_raw = tl.max(tl.where(cmask, s2, float("-inf")), 0)
        # imbalance + tracker (inf seed, 1e-12 slack; slack written as x + x*1e-12
        # because a python (1.0 + 1e-12) constant is fp32 in triton and rounds to 1.0)
        ib = tl.maximum(s1_max_raw, s2_max_raw) / tl.maximum(tl.minimum(s1_min_raw, s2_min_raw), tiny)
        imb_min = tl.load(imbmin_ptr + b)
        better = ib <= imb_min + imb_min * 1e-12
        imb_min = tl.minimum(imb_min, ib)
        gate = tl.load(gate_ptr + b)
        rising = ib > imb_min + imb_min * 1e-12
        gate = tl.minimum(gate + rising.to(tl.float64), 1.0)
        g = 1.0 - gate
        if FIRST:
            # tgt from the identity candidate (lm == 0 on iteration 0). Kept in a
            # register: a load following the store to tgt_ptr in the same program
            # is not guaranteed to observe it.
            s2_min_c = tl.min(tl.where(cmask, tl.clamp(s2, w_lo, w_hi), float("inf")), 0)
            tgt = tl.minimum(s1_min_c, s2_min_c) + tl.full((), 1e-6, tl.float64)
            tl.store(tgt_ptr + b, tgt)
        else:
            tgt = tl.load(tgt_ptr + b)
        lm1 = tl.load(lm1_ptr + b * L + cols, mask=cmask, other=0.0)
        m1s = tl.load(m1s_ptr + b * L + cols, mask=cmask, other=0.0)
        m1s = tl.where(better, tl.exp(lm1), m1s)
        sal = tl.log(tl.clamp(tl.clamp(s2, w_lo, w_hi) / tgt, c_lo, c_hi))
        lm1 = tl.clamp(lm1 + sal * g, lm_lo, lm_hi)
        tl.store(m1s_ptr + b * L + cols, m1s, mask=cmask)
        tl.store(lm1_ptr + b * L + cols, lm1, mask=cmask)
        tl.store(imbmin_ptr + b, imb_min)
        tl.store(gate_ptr + b, gate)
        tl.store(better_ptr + b, better.to(tl.int32))

    @triton.jit
    def _k3_rows(s1_ptr, lm2_ptr, m2s_ptr, tgt_ptr, gate_ptr, better_ptr, H, BH: tl.constexpr):
        pc = tl.program_id(0)
        b = tl.program_id(1)
        rows = pc * BH + tl.arange(0, BH)
        rmask = rows < H
        # fp64 clamp constants (python float literals are fp32 in triton)
        c_lo = tl.full((), 0.7, tl.float64)
        c_hi = tl.full((), 2.0, tl.float64)
        w_lo = tl.full((), 1e-3, tl.float64)
        w_hi = tl.full((), 1e3, tl.float64)
        lm_lo = tl.full((), -0.3, tl.float64)
        lm_hi = tl.full((), 10.0, tl.float64)
        s1 = tl.load(s1_ptr + b * H + rows, mask=rmask, other=1.0)
        lm2 = tl.load(lm2_ptr + b * H + rows, mask=rmask, other=0.0)
        m2s = tl.load(m2s_ptr + b * H + rows, mask=rmask, other=1.0)
        tgt = tl.load(tgt_ptr + b)
        g = 1.0 - tl.load(gate_ptr + b)
        better = tl.load(better_ptr + b) > 0
        m2s = tl.where(better, tl.exp(lm2), m2s)
        sal = tl.log(tl.clamp(tl.clamp(s1, w_lo, w_hi) / tgt, c_lo, c_hi))
        lm2 = tl.clamp(lm2 + sal * g, lm_lo, lm_hi)
        tl.store(m2s_ptr + b * H + rows, m2s, mask=rmask)
        tl.store(lm2_ptr + b * H + rows, lm2, mask=rmask)


_BH = 32  # rows per K1/K3 program


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def sinkhorn_log_triton(matrix, order=8):
    """Run the Sinkhorn loop on CUDA; returns ``(mu1 [.., L], mu2 [.., H, 1])``.

    Same candidates and updates as the eager loop (see the module docstring
    for the tracker hardenings); fp64 in/out. ``matrix`` is ``[B, H, L]`` (or
    ``[H, L]``) contiguous on CUDA; ``L >= 2`` is required (std of a single
    element is undefined, matching eager's NaN).
    """
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    m = matrix.to(torch.float64)
    assert m.is_cuda and m.is_contiguous()
    squeeze = m.dim() == 2
    if squeeze:
        m = m.unsqueeze(0)
    B, H, L = m.shape
    assert L >= 2, "std undefined for L==1; use the eager path"
    LP = _next_pow2(L)
    P = triton.cdiv(H, _BH)
    dev = m.device
    lm1 = torch.zeros(B, L, dtype=torch.float64, device=dev)
    lm2 = torch.zeros(B, H, dtype=torch.float64, device=dev)
    m1s = torch.ones(B, L, dtype=torch.float64, device=dev)
    m2s = torch.ones(B, H, dtype=torch.float64, device=dev)
    s1 = torch.empty(B, H, dtype=torch.float64, device=dev)
    csum = torch.empty(P, B, LP, dtype=torch.float64, device=dev)
    csq = torch.empty(P, B, LP, dtype=torch.float64, device=dev)
    tgt = torch.zeros(B, dtype=torch.float64, device=dev)
    imb_min = torch.full((B,), float("inf"), dtype=torch.float64, device=dev)
    gate = torch.zeros(B, dtype=torch.float64, device=dev)
    better = torch.zeros(B, dtype=torch.int32, device=dev)
    cp = max(8, 2048 // LP)
    # launch under the matrix's device context: Triton kernels launch on the
    # CURRENT cuda device and reject/misread pointers from other GPUs (streaming
    # block homes live on non-primary devices)
    with torch.cuda.device(dev):
        for k in range(order):
            _k1_pass[(P, B)](m, lm1, lm2, s1, csum, csq, H, L, B, BH=_BH, LP=LP, num_warps=4)
            _k2_update[(B,)](
                csum,
                csq,
                s1,
                lm1,
                m1s,
                tgt,
                imb_min,
                gate,
                better,
                H,
                L,
                B,
                P,
                FIRST=(k == 0),
                LP=LP,
                CP=cp,
                SH=4096,
                num_warps=4,
            )
            _k3_rows[(P, B)](s1, lm2, m2s, tgt, gate, better, H, BH=_BH, num_warps=4)
    if squeeze:
        return m1s[0], m2s[0].unsqueeze(-1)
    return m1s, m2s.unsqueeze(-1)
