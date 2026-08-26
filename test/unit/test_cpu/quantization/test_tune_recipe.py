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
"""AR_TUNE_RECIPE: searched (scale, zp) grids as SignRound tuning init.

Design: .featyard/designs/2026-08-26-tuning-recipes-design.md §3 — the STE
quant functions re-derive (scale, zp) from per-group (tensor_min, tensor_max);
anchoring min=-zp*s, max=(maxq-zp)*s reproduces a searched grid exactly.
"""

import pytest
import torch

from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym
from auto_round.data_type.utils import get_quant_func


class TestAnchorMath:
    """The anchor formula must reproduce any searched grid through the STE fn."""

    def test_asym_anchor_reproduces_scale_zp(self):
        torch.manual_seed(0)
        bits, gs = 4, 32
        maxq = 2**bits - 1
        s = (torch.rand(6, 1) * 0.02 + 0.001).float()  # searched scale [N,1]
        zp = torch.randint(0, maxq + 1, (6, 1)).float()  # integral zp
        w = (torch.randn(6, gs) * 0.05).float()

        wmin = -(zp * s).squeeze(-1)
        wmax = ((maxq - zp) * s).squeeze(-1)
        _, scale_out, zp_out = quant_tensor_asym(
            w, bits=bits, group_size=gs, tensor_min=wmin, tensor_max=wmax, scale_dtype=torch.float32
        )
        assert torch.equal(zp_out.squeeze(-1), zp.squeeze(-1)), "zp must re-derive exactly"
        torch.testing.assert_close(scale_out.squeeze(-1), s.squeeze(-1), rtol=0, atol=1e-7)

    def test_sym_anchor_reproduces_scale(self):
        torch.manual_seed(1)
        bits, gs = 4, 32
        maxq = 2 ** (bits - 1)
        s = (torch.rand(6, 1) * 0.02 + 0.001).float()
        w = (torch.randn(6, gs) * 0.05).float()
        wmin = -(s * maxq).squeeze(-1)
        wmax = (s * maxq).squeeze(-1)
        qdq, scale_out, _ = quant_tensor_sym(
            w, bits=bits, group_size=gs, tensor_min=wmin, tensor_max=wmax, scale_dtype=torch.float32
        )
        # balanced magnitudes derive a NEGATIVE scale in quant_tensor_sym
        # (2*(wmax<wmin)-1 == -1 on ties); qdq is identical either way
        torch.testing.assert_close(scale_out.squeeze(-1).abs(), s.squeeze(-1), rtol=0, atol=1e-7)
        # the negative-scale tie convention orients the clamp window as
        # [-(maxq-1), +maxq]*s (same orientation search_scales itself uses)
        q_ref = s * torch.clamp(torch.round(w / s), -(maxq - 1), maxq)
        torch.testing.assert_close(qdq, q_ref, rtol=0, atol=1e-6)


class TestRecipeAnchors:
    def test_neuqi_anchor_matches_manual_search(self, monkeypatch):
        from auto_round.data_type.neuqi import neuqi_search_scale_zero
        from auto_round.wrapper import _compute_recipe_anchors

        torch.manual_seed(2)
        bits, gs = 4, 16
        w = torch.randn(8, gs).float()
        qw = torch.rand(8, gs).float() + 0.1

        s_ref, zp_ref = neuqi_search_scale_zero(w.clone(), bits, qw=qw.clone())
        wmin, wmax, frozen = _compute_recipe_anchors(w, bits, gs, qw, "neuqi_qon", torch.device("cpu"))
        assert frozen is False
        maxq = 2**bits - 1
        # anchors re-derive the searched grid through the STE contract
        _, scale_out, zp_out = quant_tensor_asym(
            w, bits=bits, group_size=gs, tensor_min=wmin, tensor_max=wmax, scale_dtype=torch.float32
        )
        torch.testing.assert_close(scale_out.squeeze(-1), s_ref.squeeze(-1), rtol=0, atol=1e-7)
        assert torch.equal(zp_out.squeeze(-1), zp_ref.squeeze(-1))

    def test_neuqi_frozen_flags_margins(self):
        from auto_round.wrapper import _compute_recipe_anchors

        torch.manual_seed(3)
        w = torch.randn(4, 16).float()
        _, _, frozen = _compute_recipe_anchors(w, 4, 16, None, "neuqi_frozen_qon", torch.device("cpu"))
        assert frozen is True

    def test_opt_rtn_sym_anchor(self):
        from auto_round.data_type.int import search_scales
        from auto_round.wrapper import _compute_recipe_anchors

        torch.manual_seed(4)
        bits, gs = 4, 16
        w = torch.randn(8, gs).float()
        s_ref = search_scales(w.clone(), bits).abs()
        wmin, wmax, frozen = _compute_recipe_anchors(w, bits, gs, None, "opt_rtn_qon", torch.device("cpu"))
        assert frozen is False
        qdq, scale_out, _ = quant_tensor_sym(
            w, bits=bits, group_size=gs, tensor_min=wmin, tensor_max=wmax, scale_dtype=torch.float32
        )
        # magnitude matches; sign is the tie-convention artifact (see TestAnchorMath)
        torch.testing.assert_close(scale_out.squeeze(-1).abs(), s_ref.squeeze(-1), rtol=0, atol=1e-7)
        # same qdq grid as quantizing with the searched scale directly, with
        # the search's own window orientation [-(maxq-1), +maxq]*s
        _nmax = 2 ** (bits - 1)
        q_ref = s_ref * torch.clamp(torch.round(w / s_ref), -(_nmax - 1), _nmax)
        torch.testing.assert_close(qdq, q_ref, rtol=0, atol=1e-6)

    def test_minmax_and_unknown_recipes_return_none(self):
        from auto_round.wrapper import _compute_recipe_anchors

        w = torch.randn(4, 16).float()
        assert _compute_recipe_anchors(w, 4, 16, None, "minmax_qon", torch.device("cpu")) is None
        assert _compute_recipe_anchors(w, 4, 16, None, "", torch.device("cpu")) is None


class TestGuardMatrix:
    """get_quant_func validates AR_TUNE_RECIPE combos with actionable errors."""

    def _g(self, **kw):
        base = dict(dtype="int", bits=4, sym=False, disable_opt_rtn=False, group_size=128, iters=20)
        base.update(kw)
        return get_quant_func(**base)

    def test_invalid_recipe_name(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "bogus")
        with pytest.raises(ValueError, match="not one of"):
            self._g()

    def test_tuning_recipe_with_iters0(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        with pytest.raises(ValueError, match="iters>0"):
            self._g(iters=0)

    def test_neuqi_it0_with_iters(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_it0")
        with pytest.raises(ValueError, match="zero-shot reference"):
            self._g()

    def test_alt2_not_implemented(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "alt2")
        with pytest.raises(NotImplementedError, match="alt2"):
            self._g()

    def test_neuqi_requires_asym(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        with pytest.raises(ValueError, match="asymmetric"):
            self._g(sym=True)

    def test_opt_rtn_requires_sym(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "opt_rtn_qon")
        with pytest.raises(ValueError, match="symmetric"):
            self._g()

    def test_recipe_allows_neuqi_with_iters(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_fp")
        fn, name = self._g(asym_search="neuqi")
        assert name.startswith("int_asym")  # SignRound STE path engaged

    def test_unset_neuqi_iters_still_raises(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_RECIPE", raising=False)
        with pytest.raises(ValueError, match="zero-shot path"):
            self._g(asym_search="neuqi")

    def test_unset_is_status_quo(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_RECIPE", raising=False)
        fn, name = self._g()
        assert name.startswith("int_asym")


class TestWrapperIntegration:
    """The recipe anchors land in WrapperLinear init (weight_min/max + margins)."""

    @staticmethod
    def _armed_linear(out=8, inp=32, gs=16):
        import torch.nn as nn

        layer = nn.Linear(inp, out, bias=True)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
        layer.bits, layer.group_size, layer.sym = 4, gs, False
        layer.data_type, layer.act_bits, layer.act_sym = "int", 16, True
        layer.scale_dtype = torch.float16
        layer.iters = 20
        return layer

    def _wrapper(self, recipe, monkeypatch):
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", recipe)
        layer = self._armed_linear()
        w = WrapperLinear(
            layer,
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=20,
        )
        return w

    def test_neuqi_frozen_anchors_and_pins_margins(self, monkeypatch):
        w = self._wrapper("neuqi_frozen_qon", monkeypatch)
        assert w.weight_min is not None and w.weight_max is not None
        assert "value" in w.params, "rounding params must stay tunable"
        assert "min_scale" not in w.params and "max_scale" not in w.params, "frozen margins"
        assert float(w.min_scale) == 1.0 and float(w.max_scale) == 1.0

    def test_neuqi_margins_stay_tunable(self, monkeypatch):
        w = self._wrapper("neuqi_qon", monkeypatch)
        assert "min_scale" in w.params and "max_scale" in w.params
        assert "value" in w.params

    def test_unset_recipe_is_status_quo(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_RECIPE", raising=False)
        w = self._wrapper("", monkeypatch)
        assert "min_scale" in w.params and "value" in w.params
        # min/max grid is the raw per-group range (un-anchored)
        import torch as _t

        gmin = _t.clamp(w.orig_layer.weight.reshape(8, 2, 16).amin(dim=-1).flatten(), max=0)
        _t.testing.assert_close(w.weight_min, gmin, rtol=0, atol=1e-7)
