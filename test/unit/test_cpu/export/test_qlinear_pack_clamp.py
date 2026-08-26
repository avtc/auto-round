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
"""Packers must clamp quantized levels into [0, maxq] before bit-packing.

round(W/s + zp) can exceed the level range whenever the searched (scale, zp)
grid does not fully cover the weight range (the search itself evaluates the
clamped loss). Packing the unclamped value shifts extra bits into the
NEIGHBORING packed nibble -- silently corrupting it -- so the packed artifact
no longer reproduces the clamped fake-quantization the search optimized.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from auto_round.export.export_to_awq.utils import WQLinear_GEMM
from auto_round_extension.torch.qlinear_torch import QuantLinear as QuantLinearDirect
from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as QuantLinearMinusOne

OUT_F, IN_F, GROUP = 64, 64, 32


def _make_inputs(bits, seed=0, with_overshoot=True):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT_F, IN_F, generator=g) * 0.05
    scale = torch.rand(OUT_F, IN_F // GROUP, generator=g) * 0.005 + 0.02
    zp = torch.randint(0, 2**bits, (OUT_F, IN_F // GROUP), generator=g).float()
    if with_overshoot:
        # force q = round(w/s + zp) far outside [0, maxq] on dedicated rows:
        # scales ~0.02..0.025, so +/-3.0 / 0.02 ~ +/-150 levels of overshoot
        w[0, :] = 3.0
        w[1, :] = -3.0
        zp[0, :] = 0  # negative side undershoots as well
    return w, scale, zp


def _unpack_qweight(qlayer):
    bits = qlayer.bits
    npk = 32 // bits
    maxq = 2**bits - 1
    wf = torch.arange(0, 32, bits, dtype=torch.int32)
    w_int = qlayer.qweight.unsqueeze(1).expand(-1, npk, -1) >> wf.unsqueeze(-1)
    w_int = (w_int & maxq).reshape(-1, qlayer.outfeatures)  # [in, out]
    return w_int.t().to(torch.int64)  # [out, in]


def _unpack_qzeros(qlayer):
    bits = qlayer.bits
    npk = 32 // bits
    maxq = 2**bits - 1
    wf = torch.arange(0, 32, bits, dtype=torch.int32)
    z = qlayer.qzeros.unsqueeze(2).expand(-1, -1, npk) >> wf.unsqueeze(0)
    return (z & maxq).to(torch.int64)  # [in/g, out/npk-1 packed] -> per nibble [in/g, out]


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_clamps_levels_direct_zp(bits):
    torch.manual_seed(0)
    w, scale, zp = _make_inputs(bits)
    q = QuantLinearDirect(bits, GROUP, IN_F, OUT_F, bias=False)
    q.device = torch.device("cpu")
    q.pack(SimpleNamespace(weight=w, bias=None), scale, zp, None, device="cpu")

    expected = torch.clamp(
        torch.round(w / scale.repeat_interleave(GROUP, 1) + zp.repeat_interleave(GROUP, 1)), 0, 2**bits - 1
    ).to(torch.int64)
    got = _unpack_qweight(q)
    assert got.min() >= 0 and got.max() <= 2**bits - 1
    assert torch.equal(got, expected), "packed levels must equal the clamped fake-quantization"


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_clamps_levels_minus_one_zp(bits):
    torch.manual_seed(0)
    w, scale, zp = _make_inputs(bits)
    q = QuantLinearMinusOne(bits, GROUP, IN_F, OUT_F, bias=False)
    q.device = torch.device("cpu")
    q.pack(SimpleNamespace(weight=w, bias=None), scale, zp, None, device="cpu")

    expected = torch.clamp(
        torch.round(w / scale.repeat_interleave(GROUP, 1) + zp.repeat_interleave(GROUP, 1)), 0, 2**bits - 1
    ).to(torch.int64)
    assert torch.equal(_unpack_qweight(q), expected)


def test_minus_one_zp_packing_survives_zero_zero_point():
    """zp=0 wraps to -1 under the classic (zp-1) packing; the raw -1 word
    corrupts every nibble sharing its packed word. The stored value must be
    clamped so only that one zero-point degrades (to a decodable +1)."""
    torch.manual_seed(0)
    bits = 4
    w, scale, zp = _make_inputs(bits, with_overshoot=False)
    zp[:, :] = 0  # every group sits at the unrepresentable zero-point
    q = QuantLinearMinusOne(bits, GROUP, IN_F, OUT_F, bias=False)
    q.device = torch.device("cpu")
    q.pack(SimpleNamespace(weight=w, bias=None), scale, zp, None, device="cpu")

    z_nibbles = _unpack_qzeros(q)  # [in/g, out] per-nibble values
    # pre-fix: (0-1) << shift makes whole words -1 -> every nibble reads maxq
    assert z_nibbles.max() <= 1, "zp=0 must not smear garbage across the packed word"
    # neighbors of a clamped cell keep their own packed value: all nibbles equal
    # either 0 (the clamped zp-1) -- uniform here by construction
    assert (z_nibbles == 0).all()
    # and the module still dequantizes (forward runs, no all-ones corruption)
    out = q(torch.eye(IN_F))
    assert torch.isfinite(out).all()


def test_awq_pack_clamps_levels():
    """The AWQ packer shares the unclamped-round spill bug: an out-of-range
    level shifts bits into the neighboring packed nibble (AWQ interleave
    order), corrupting it. Packed levels must equal the clamped grid."""
    torch.manual_seed(0)
    bits, maxq = 4, 15
    w, scale, zp = _make_inputs(bits)  # [out, in], [out, in/g], [out, in/g]
    lin = nn.Linear(IN_F, OUT_F, bias=False)
    with torch.no_grad():
        lin.weight.copy_(w)
    q = WQLinear_GEMM.from_linear(lin, bits, GROUP, scales=scale.t(), zeros=zp.t(), device="cpu")

    expected_q = torch.clamp(
        torch.round(w / scale.repeat_interleave(GROUP, 1) + zp.repeat_interleave(GROUP, 1)), 0, maxq
    ).to(torch.int32)

    def _pack_rows(t):  # mirrors the packer: nibbles along dim 1, AWQ interleave
        pack_num = 32 // bits
        order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7]) * bits
        packed = t.reshape(t.shape[0], t.shape[1] // pack_num, pack_num) << order
        return torch.sum(packed, dim=-1).to(torch.int32)

    # qweight packs q transposed ([in, out/8], nibbles along the output dim);
    # qzeros pack zp along the output dim ([in/g, out/8])
    assert torch.equal(q.qweight, _pack_rows(expected_q.t().contiguous()))
    assert torch.equal(q.qzeros, _pack_rows(zp.to(torch.int32).t().contiguous()))


def test_direct_zp_roundtrip_in_range():
    """Sanity: for in-range data the packed module reproduces clamped
    fake-quantization bit-exactly (clamp must be a no-op there)."""
    torch.manual_seed(0)
    bits, maxq = 4, 15
    w, scale, zp = _make_inputs(bits, with_overshoot=False)
    # keep q in-range: shrink weights relative to scale
    w = w * 0.05
    q = QuantLinearDirect(bits, GROUP, IN_F, OUT_F, bias=False)
    q.device = torch.device("cpu")
    q.pack(SimpleNamespace(weight=w, bias=None), scale, zp, None, device="cpu")
    s16 = scale.half().float().repeat_interleave(GROUP, 1)
    zrep = zp.repeat_interleave(GROUP, 1)
    expected = torch.clamp(torch.round(w / s16 + zrep), 0, maxq).to(torch.int64)
    assert torch.equal(_unpack_qweight(q), expected)
    # dequantized weights match the reference reconstruction
    ref_w = (expected.to(torch.float32) - zrep) * s16
    got_w = q(torch.eye(IN_F)).t()
    torch.testing.assert_close(got_w, ref_w, rtol=0, atol=1e-3)
