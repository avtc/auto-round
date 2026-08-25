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
"""Triton kernel for the NeUQI all-candidates zero-point sweep.

One launch evaluates every per-group scale candidate (``scales``, [C, K])
against every integer zero point for a chunk of groups. Unlike the
``torch.compile``-fused expression -- whose kernel parallelizes over the
(candidate, zero-point) lanes and therefore re-reads each weight element from
L2 once per lane -- this kernel loads each element of a [BC, g] data tile into
registers exactly once and computes all K x Z losses from registers
(amplification 1x). Measured on an RTX 3090 (2M groups, g=64, bits=4, 64+32
candidate grid, fp32): 0.216 s vs 0.488 s (compiled batched expression) and
12.3 s (eager chunked sweep). Selections match the torch paths' tie profile
(same ~0.07% flipped groups, worst fp64 relative loss difference ~5e-5): the
kernel rounds with round-half-to-even (matching ``torch.round``; libdevice
``round`` is half-away-from-zero and would change selections) and keeps the
first-minimum zero-point rule via a strict ``<`` with ascending zero points.

Contract (``neuqi_sweep_triton``):
    data:   [C, g] float32, contiguous, CUDA
    qw:     [C, g] float32, contiguous, CUDA, or ``None``
    scales: [C, K] float32, contiguous, CUDA
    maxq:   maximum integer value (``2**bits - 1``)
    -> (loss [C, K] float32, argmin zero point [C, K] int32)
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
    def _neuqi_sweep_kernel(
        data_ptr,
        qw_ptr,
        scales_ptr,
        loss_ptr,
        zp_ptr,
        C,
        G,
        K,
        NZ,
        MAXQ: tl.constexpr,
        HAS_QW: tl.constexpr,
        BC: tl.constexpr,
        GP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c_off = pid * BC + tl.arange(0, BC)
        g_off = tl.arange(0, GP)
        cm = c_off < C
        gm = g_off < G
        d = tl.load(data_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        if HAS_QW:
            qwt = tl.load(qw_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        else:
            qwt = tl.zeros((BC, GP), dtype=tl.float32) + 1.0
        for k in range(0, K):
            sc = tl.load(scales_ptr + c_off * K + k, mask=cm, other=1.0)  # [BC]
            x = d / sc[:, None]
            fl = tl.floor(x)
            fr = x - fl
            # round-half-to-even (matches torch.round)
            fl_odd = fl - 2.0 * tl.floor(fl * 0.5) == 1.0
            take_up = (fr > 0.5) | ((fr == 0.5) & fl_odd)
            r = fl + take_up.to(tl.float32)
            r = tl.minimum(tl.maximum(r, -MAXQ - 1.0), 2.0 * MAXQ + 1.0)
            best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
            bz = tl.zeros((BC,), dtype=tl.int32)
            for z in range(0, NZ):
                q = tl.minimum(tl.maximum(r + z, 0.0), MAXQ)
                err = sc[:, None] * (q - z) - d
                loss2 = err * err * qwt
                acc = tl.sum(tl.where(gm[None, :], loss2, 0.0), axis=1)
                better = acc < best  # strict: ascending z keeps the first minimum
                best = tl.where(better, acc, best)
                bz = tl.where(better, z, bz)
            tl.store(loss_ptr + c_off * K + k, best, mask=cm)
            tl.store(zp_ptr + c_off * K + k, bz, mask=cm)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def neuqi_sweep_triton(data, qw, scales, maxq):
    """Run the all-candidates zero-point sweep on CUDA (see module docstring)."""
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    assert data.is_cuda and data.dtype == torch.float32 and data.is_contiguous()
    assert scales.is_contiguous() and scales.dtype == torch.float32
    if qw is not None:
        assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.shape == data.shape

    c, g = data.shape
    k = scales.shape[1]
    nz = maxq + 1
    gp = _next_pow2(g)
    bc = max(1, min(64, 4096 // gp))
    loss = torch.empty((c, k), device=data.device, dtype=torch.float32)
    zp = torch.empty((c, k), device=data.device, dtype=torch.int32)
    _neuqi_sweep_kernel[(triton.cdiv(c, bc),)](
        data,
        qw if qw is not None else data,  # dummy pointer when unweighted (dead lanes)
        scales,
        loss,
        zp,
        c,
        g,
        k,
        nz,
        MAXQ=maxq,
        HAS_QW=qw is not None,
        BC=bc,
        GP=gp,
        num_warps=4,
    )
    return loss, zp
