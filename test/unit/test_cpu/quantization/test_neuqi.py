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
import logging

import pytest
import torch

from auto_round.data_type.int import quant_tensor_asym
from auto_round.data_type.neuqi import _log_search_engaged, neuqi_search_scale_zero, quant_tensor_opt_rtn_asym
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import get_quant_func


def _heavy_tailed_data(n_groups=64, group_size=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = torch.randn(n_groups, group_size, generator=gen)
    # inject a few large outliers per group to make min/max suboptimal
    data[:, :5] *= 120.0
    return data


def _mse(x, y):
    return ((x.float() - y.float()) ** 2).mean().item()


class TestNeuqiSearch:
    def test_hist_sweep_matches_brute_force(self):
        """The histogram zero-point sweep selects (scale, zp) with losses equal
        to the brute-force grid within float noise; near-ties may flip but both
        choices are optimal to ~1e-5 relative loss."""
        from auto_round.data_type.neuqi import _best_zp_for_scale, _loss_all_zp_hist

        torch.manual_seed(5)
        maxq = 15
        for qw_on in (True, False):
            data = torch.randn(4096, 64) * 0.05
            qw = torch.rand(4096, 64) + 0.1 if qw_on else None
            scale = data.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) * (torch.rand(4096, 1) * 1.4 + 0.05)
            bl, bz = _best_zp_for_scale(data, qw, scale, maxq)
            hl, hz = _loss_all_zp_hist(data, qw, scale, maxq)
            assert torch.allclose(bl, hl, rtol=1e-4, atol=1e-8)
            # wherever they disagree the histogram choice must be no worse
            assert (hl <= bl + 1e-4 * bl.abs() + 1e-8).all().item()

    def test_brute_force_available_via_env(self, monkeypatch):
        """AR_NEUQI_SWEEP=brute restores the reference grid sweep."""
        import auto_round.data_type.neuqi as N

        monkeypatch.setattr(N.envs, "AR_NEUQI_SWEEP", "brute")
        torch.manual_seed(5)
        data = torch.randn(512, 64) * 0.05
        qw = torch.rand(512, 64) + 0.1
        scale, zp = N.neuqi_search_scale_zero(data, bits=4, qw=qw, coarse_n=4, fine_n=3)
        assert scale.shape == (512, 1) and zp.shape == (512, 1)
        assert (zp >= 0).all() and (zp <= 15).all()

    def test_beats_minmax_on_heavy_tailed_data(self):
        data = _heavy_tailed_data()
        bits, group_size = 4, 128

        minmax_qdq, _, _ = quant_tensor_asym(data.clone(), bits=bits, group_size=group_size)
        neuqi_qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=bits, group_size=group_size)

        assert _mse(neuqi_qdq, data) < _mse(minmax_qdq, data)

    def test_zero_point_integral_and_in_range(self):
        data = _heavy_tailed_data(n_groups=32)
        scale, zp = neuqi_search_scale_zero(data, bits=4)
        assert torch.all(zp == zp.round())
        assert torch.all(zp >= 0) and torch.all(zp <= 15)
        assert torch.all(scale > 0)

    def test_close_to_dense_exhaustive_reference(self):
        """On a small case the search must approach a dense exhaustive grid."""
        gen = torch.Generator().manual_seed(3)
        n, g, bits = 8, 16, 2
        maxq = 2**bits - 1
        data = torch.randn(n, g, generator=gen)
        data[:, 0] *= 50.0

        scale, zp = neuqi_search_scale_zero(data, bits=bits)
        r = torch.round(data / scale)
        q = (r + zp).clamp(0, maxq)
        our_loss = (((scale * (q - zp)) - data) ** 2).sum().item()

        # brute force: dense additive scale grid per group x all integer zero points
        best = torch.full((n,), float("inf"))
        wmin = torch.clamp(data.min(-1).values, max=0)
        wmax = torch.clamp(data.max(-1).values, min=0)
        s0 = ((wmax - wmin) / maxq).clamp(min=1e-5)
        for i in range(401):
            frac = 0.05 + (1.0 - 0.05) * i / 400
            s = (s0 * frac).unsqueeze(-1)
            r = torch.round(data / s)
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                loss = ((s * (q - z) - data) ** 2).sum(-1)
                best = torch.minimum(best, loss)
        exhaustive_loss = best.sum().item()
        # within 1% of the (much denser) exhaustive grid
        assert our_loss <= exhaustive_loss * 1.01 + 1e-9

    def test_weighting_changes_solution(self):
        """A per-element weight that de-emphasizes outliers must pull the scale down."""
        gen = torch.Generator().manual_seed(1)
        data = torch.randn(16, 64, generator=gen)
        data[:, :4] *= 100.0

        qw = torch.ones_like(data)
        qw[:, :4] = 1e-4  # ignore outliers

        scale_plain, _ = neuqi_search_scale_zero(data, bits=4)
        scale_weighted, _ = neuqi_search_scale_zero(data, bits=4, qw=qw)
        # ignoring outliers must never increase the scale and must shrink it somewhere
        assert torch.all(scale_weighted <= scale_plain + 1e-6)
        assert torch.any(scale_weighted < scale_plain)

        def weighted_loss(scale):
            maxq = 15
            r = torch.round(data / scale).clamp(-maxq - 1, 2 * maxq + 1)
            losses = []
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                losses.append((((scale * (q - z) - data) ** 2) * qw).sum(-1))
            return torch.stack(losses, -1).min(-1).values.sum()

        assert weighted_loss(scale_weighted) < weighted_loss(scale_plain)

    def test_constant_group_is_exact(self):
        data = torch.full((4, 128), 0.25)
        qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        # the fp16 scale cast limits exactness to one quantization step (0.25 / 15)
        assert (qdq - data).abs().max() <= 0.25 / 15
        assert torch.all(scale > 0)

    def test_shapes_preserved(self):
        gen = torch.Generator().manual_seed(2)
        for shape in [(4, 33, 128), (17, 65), (3, 5, 7, 11)]:
            tensor = torch.randn(*shape, generator=gen)
            qdq, scale, zp = quant_tensor_opt_rtn_asym(tensor.clone(), bits=4, group_size=32)
            assert qdq.shape == tensor.shape


class _LogCapture(logging.Handler):
    records: list

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class TestNeuqiLogging:
    def test_engaged_log_line(self):
        _log_search_engaged.cache_clear()
        cap = _LogCapture()
        ar_logger = logging.getLogger("autoround")
        ar_logger.addHandler(cap)
        try:
            quant_tensor_opt_rtn_asym(_heavy_tailed_data(n_groups=4).clone(), bits=4, group_size=128)
        finally:
            ar_logger.removeHandler(cap)
        assert any("[NeUQI]" in m and "search active" in m for m in cap.records)

    def test_disabled_log_line(self, monkeypatch):
        from auto_round.data_type.neuqi import _log_disabled

        monkeypatch.setenv("AR_DISABLE_NEUQI", "1")
        _log_disabled.cache_clear()
        cap = _LogCapture()
        ar_logger = logging.getLogger("autoround")
        ar_logger.addHandler(cap)
        try:
            quant_tensor_opt_rtn_asym(_heavy_tailed_data(n_groups=4).clone(), bits=4, group_size=128)
        finally:
            ar_logger.removeHandler(cap)
        assert any("[NeUQI]" in m and "disabled" in m for m in cap.records)
        monkeypatch.delenv("AR_DISABLE_NEUQI")


class TestNeuqiIntegration:
    def test_registration_and_dispatch(self):
        assert "opt_rtn_int_asym" in QUANT_FUNC_WITH_DTYPE
        func, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0)
        assert name == "opt_rtn_int_asym"
        assert func is QUANT_FUNC_WITH_DTYPE["opt_rtn_int_asym"]
        # symmetric path must remain untouched
        _, sym_name = get_quant_func("int", 4, True, disable_opt_rtn=False, group_size=None, iters=0)
        assert sym_name == "opt_rtn_int_sym"

    def test_disable_env_reverts_to_plain_asym(self, monkeypatch):
        monkeypatch.setenv("AR_DISABLE_NEUQI", "1")
        data = _heavy_tailed_data(n_groups=16)
        qdq_disabled, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        qdq_plain, _, _ = quant_tensor_asym(data.clone(), bits=4, group_size=128)
        assert torch.equal(qdq_disabled, qdq_plain)
        monkeypatch.delenv("AR_DISABLE_NEUQI")
        qdq_enabled, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        assert _mse(qdq_enabled, data) < _mse(qdq_plain, data)

    def test_imatrix_weighting_improves_weighted_mse(self):
        data = _heavy_tailed_data(n_groups=32)
        imatrix = torch.rand(128) + 0.1

        qdq_plain, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        qdq_im, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128, imatrix=imatrix)

        qw = imatrix.expand(32, 128)
        weighted = lambda q: (((q - data) ** 2) * qw).sum().item()  # noqa: E731
        assert weighted(qdq_im) < weighted(qdq_plain)

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_bits_smoke(self, bits):
        data = _heavy_tailed_data(n_groups=8, group_size=64, seed=bits)
        qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=bits, group_size=64)
        maxq = 2**bits - 1
        assert torch.all(zp >= 0) and torch.all(zp <= maxq)
        assert torch.all(scale > 0)
        assert qdq.shape == data.shape


class TestAsymSearchAPI:
    """RTNConfig(asym_search) enum: auto | neuqi | minmax (asym path only)."""

    def test_default_auto_resolves_neuqi(self):
        func, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0)
        assert name == "opt_rtn_int_asym"

    def test_minmax_resolves_plain_asym(self):
        from auto_round.data_type.int import quant_tensor_asym as plain_asym

        func, name = get_quant_func(
            "int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax"
        )
        assert "opt_rtn" not in name
        assert name.endswith("_asym")
        assert func is plain_asym

    def test_minmax_does_not_touch_sym_path(self):
        _, sym_name = get_quant_func(
            "int", 4, True, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax"
        )
        assert sym_name == "opt_rtn_int_sym"

    def test_minmax_output_matches_plain_quant(self):
        torch.manual_seed(7)
        data = _heavy_tailed_data(n_groups=8, group_size=64, seed=7)
        func, _ = get_quant_func(
            "int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax"
        )
        from auto_round.data_type.int import quant_tensor_asym

        expected = quant_tensor_asym(data.clone(), bits=4, group_size=64)
        actual = func(data.clone(), bits=4, group_size=64)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_force_neuqi_accepted_when_engageable(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=False, asym_search="neuqi")
        assert cfg.asym_search == "neuqi"

    def test_non_auto_with_sym_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match="asymmetric path only"):
            RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, asym_search="neuqi")
        with pytest.raises(ValueError, match="asymmetric path only"):
            RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, asym_search="minmax")

    def test_neuqi_with_opt_rtn_disabled_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match='asym_search="neuqi" requires the optimized-RTN path'):
            RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=True, asym_search="neuqi")

    def test_unknown_value_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match="not one of"):
            RTNConfig(bits=4, group_size=32, sym=False, asym_search="grid")

    def test_default_is_auto(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig(bits=4, group_size=32, sym=False)
        assert cfg.asym_search == "auto"
