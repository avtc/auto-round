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
"""CUDA parity tests for the Pre-SINQ Triton sinkhorn against the eager loop.

The shapes include the two regression shapes found by the staged GPU bench:
``[B=8, H=256, L=64]`` (tgt load-after-store) and non-power-of-two ``L`` (lane
padding). Parity is asserted at fp64-ulp level (~1e-15), tighter than the
compiled arm. Skipped on hosts without CUDA+triton.
"""

import pytest
import torch

from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales, sinkhorn_log

sinkhorn_triton = pytest.importorskip(
    "auto_round_extension.triton.presinq_sinkhorn", reason="extension/triton unavailable"
).sinkhorn_log_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize(
    "B, H, L, order",
    [
        (4, 64, 16, 4),  # sanity shape
        (8, 256, 64, 4),  # regression: tgt load-after-store at L=64
        (16, 4096, 64, 4),  # expert-batch shape (multi-chunk column reduce)
        (100, 13, 5, 3),  # non-power-of-two L (lane padding)
        (3, 64, 16, 0),  # order=0 -> identity
    ],
)
def test_triton_matches_eager_shapes(B, H, L, order):
    torch.manual_seed(B * 131 + H * 7 + L)
    m = (torch.randn(B, H, L, device="cuda") * torch.logspace(-1, 1, L, device="cuda")).to(torch.float64)
    _, mu1_e, mu2_e = sinkhorn_log(m, order=order)
    mu1_t, mu2_t = sinkhorn_triton(m, order=order)
    if order == 0:
        assert torch.equal(mu1_t, mu1_e) and torch.equal(mu2_t, mu2_e)
    else:
        torch.testing.assert_close(mu1_t, mu1_e, rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(mu2_t, mu2_e, rtol=1e-12, atol=1e-15)


def test_triton_adversarial_tie_trajectory():
    """The seed-0 tile batch whose exact-equality ties cascade under naive
    comparisons: mu agreement AND equal achieved imbalance are required."""
    torch.manual_seed(0)
    m = torch.randn(2048, 768, device="cuda") * torch.logspace(-1, 1, 768, device="cuda")
    m = m.view(2048, 12, 64).permute(1, 0, 2).contiguous().to(torch.float64)
    _, mu1_e, mu2_e = sinkhorn_log(m, order=4)
    mu1_t, mu2_t = sinkhorn_triton(m, order=4)
    torch.testing.assert_close(mu1_t, mu1_e, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(mu2_t, mu2_e, rtol=1e-9, atol=1e-12)

    def imb_of(mat):
        s1, s2 = mat.std(-1), mat.std(-2)
        return torch.maximum(s1.amax(-1), s2.amax(-1)) / torch.minimum(s1.amin(-1), s2.amin(-1)).clamp_min(1e-12)

    torch.testing.assert_close(
        imb_of(m / mu1_t.unsqueeze(-2) / mu2_t), imb_of(m / mu1_e.unsqueeze(-2) / mu2_e), rtol=1e-9, atol=1e-12
    )


def test_triton_end_to_end_column_scales():
    torch.manual_seed(3)
    mats = [torch.randn(128, 256, device="cuda") * torch.logspace(-1, 1, 256, device="cuda")]
    from auto_round import envs

    saved = envs.AR_PRESINQ_BACKEND
    envs.AR_PRESINQ_BACKEND = "eager"
    try:
        ref = column_scales(mats, group_size=64, n_iter=4)
    finally:
        envs.AR_PRESINQ_BACKEND = saved
    # direct: emulate the triton mu1 provider through the production driver
    import auto_round.algorithms.transforms.presinq.sinkhorn as S

    saved_state = (S._triton_checked, S._triton_sweep, S._triton_broken)
    S._triton_checked, S._triton_sweep, S._triton_broken = True, sinkhorn_triton, False
    try:
        envs.AR_PRESINQ_BACKEND = "triton"
        got = column_scales(mats, group_size=64, n_iter=4)
    finally:
        envs.AR_PRESINQ_BACKEND, (S._triton_checked, S._triton_sweep, S._triton_broken) = (saved, saved_state)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-15)
