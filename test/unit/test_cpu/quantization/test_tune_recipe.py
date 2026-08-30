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

    @pytest.fixture(autouse=True)
    def _pin_mixed_latch(self, monkeypatch):
        """Guards default to strict; a leaked mixed-pool latch from an earlier
        test must not silence them."""
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", False, raising=False)

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

    def test_mixed_pool_neuqi_recipe_skips_sym_layers(self, monkeypatch):
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_frozen_qon")
        fn, name = self._g(sym=True)
        assert name.startswith("int_sym")

    def test_mixed_pool_opt_rtn_skips_asym_layers(self, monkeypatch):
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        monkeypatch.setenv("AR_TUNE_RECIPE", "opt_rtn_qon")
        fn, name = self._g(sym=False)
        assert name.startswith("int_asym")

    def test_mixed_latch_does_not_mask_other_errors(self, monkeypatch):
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        monkeypatch.setenv("AR_TUNE_RECIPE", "bogus")
        with pytest.raises(ValueError, match="not one of"):
            self._g()


class TestRecipeAppliesToLayer:
    """recipe_applies_to_layer: mixed pools anchor only matching layers."""

    def test_matrix(self):
        from auto_round.data_type.utils import recipe_applies_to_layer

        # neuqi_* anchors the asymmetric path only
        assert recipe_applies_to_layer("neuqi_frozen_qon", sym=False) is True
        assert recipe_applies_to_layer("neuqi_qon", sym=True) is False
        assert recipe_applies_to_layer("neuqi_fp", sym=True) is False
        # opt_rtn_qon anchors the symmetric path only
        assert recipe_applies_to_layer("opt_rtn_qon", sym=True) is True
        assert recipe_applies_to_layer("opt_rtn_qon", sym=False) is False
        # no-anchor recipes
        assert recipe_applies_to_layer("", True) is False
        assert recipe_applies_to_layer("minmax_qon", False) is False
        assert recipe_applies_to_layer("neuqi_it0", False) is False
        assert recipe_applies_to_layer("touchup", False) is False


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


class TestWrapperMixedSymPool:
    """Mixed sym/asym pools: recipes anchor only layers of their sym class."""

    @staticmethod
    def _linear(sym):
        import torch.nn as nn

        layer = nn.Linear(32, 8, bias=True)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
        layer.bits, layer.group_size, layer.sym = 4, 16, sym
        layer.data_type, layer.act_bits, layer.act_sym = "int", 16, True
        layer.scale_dtype = torch.float16
        layer.iters = 20
        return layer

    def _wrapper(self, recipe, sym, monkeypatch):
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", recipe)
        return WrapperLinear(
            self._linear(sym),
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=20,
        )

    def test_neuqi_recipe_skips_sym_layer(self, monkeypatch):
        from auto_round.data_type import utils as dt_utils

        # latch mirrors a real mixed-pool run (parse_scheme sets it on the scheme)
        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        w = self._wrapper("neuqi_frozen_qon", sym=True, monkeypatch=monkeypatch)
        assert w._tune_recipe == "", "sym layer must not take a neuqi anchor"
        assert "min_scale" in w.params, "margins stay tunable (no frozen pin)"
        import torch as _t

        gmin = _t.clamp(w.orig_layer.weight.reshape(8, 2, 16).amin(dim=-1).flatten(), max=0)
        _t.testing.assert_close(w.weight_min, gmin, rtol=0, atol=1e-7)

    def test_opt_rtn_recipe_skips_asym_layer(self, monkeypatch):
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        w = self._wrapper("opt_rtn_qon", sym=False, monkeypatch=monkeypatch)
        assert w._tune_recipe == ""
        assert "min_scale" in w.params

    def test_matching_layers_still_anchor(self, monkeypatch):
        w = self._wrapper("neuqi_frozen_qon", sym=False, monkeypatch=monkeypatch)
        assert w._tune_recipe == "neuqi_frozen_qon"
        assert float(w.min_scale) == 1.0 and float(w.max_scale) == 1.0


class TestPostScaleRefit:
    """AR_POST_SCALE_REFIT: exact LS scale refit, integers+zp frozen."""

    @staticmethod
    def _quantized(bits=4, gs=16, n=8, seed=5):
        torch.manual_seed(seed)
        w = torch.randn(n, gs).float() * 0.05
        s = (torch.rand(n, 1) * 0.004 + 0.002).float()
        maxq = 2**bits - 1
        zp = torch.randint(0, maxq + 1, (n, 1)).float()
        q = torch.clamp(torch.round(w / s + zp), 0, maxq)
        qdq = (s * (q - zp)).to(w.dtype)
        return w, s, zp, q, qdq

    def test_refit_never_increases_mse(self):
        from auto_round.wrapper import _refit_scale_grid

        w, s, zp, q, qdq = self._quantized()
        qdq_new, s_new = _refit_scale_grid(w, qdq.clone(), s.clone(), zp, 16)
        mse0 = ((qdq - w) ** 2).sum()
        mse1 = ((qdq_new - w) ** 2).sum()
        assert mse1 <= mse0 + 1e-12, f"refit increased MSE: {mse0} -> {mse1}"

    def test_integers_and_zp_frozen(self):
        from auto_round.wrapper import _refit_scale_grid

        w, s, zp, q, qdq = self._quantized(seed=6)
        qdq_new, s_new = _refit_scale_grid(w, qdq.clone(), s.clone(), zp, 16)
        q_rec = torch.round(qdq_new / s_new + zp)
        q_ref = torch.round(qdq / s + zp)
        assert torch.equal(q_rec, q_ref), "integer grid must not move"

    def test_imatrix_weighted_objective_improves(self):
        from auto_round.wrapper import _refit_scale_grid

        w, s, zp, q, qdq = self._quantized(seed=7)
        qw = (torch.rand_like(w) * 10) ** 2  # strong per-element weighting
        qdq_new, _ = _refit_scale_grid(w, qdq.clone(), s.clone(), zp, 16, qw=qw)
        wmse0 = (qw * (qdq - w) ** 2).sum()
        wmse1 = (qw * (qdq_new - w) ** 2).sum()
        assert wmse1 <= wmse0 + 1e-9

    def test_degenerate_group_keeps_scale(self):
        from auto_round.wrapper import _refit_scale_grid

        w = torch.zeros(2, 16)
        s = torch.full((2, 1), 0.01)
        zp = torch.zeros(2, 1)
        qdq = torch.zeros_like(w)  # d == 0 everywhere
        qdq_new, s_new = _refit_scale_grid(w, qdq, s, zp, 16)
        torch.testing.assert_close(s_new, s)


class TestBiasCorrect:
    """AR_BIAS_CORRECT: block-output b = mean(y_fp - y_q) absorbed into the residual sink."""

    @staticmethod
    def _toy_block(d=8, seed=3):
        import torch.nn as nn

        torch.manual_seed(seed)
        block = nn.Sequential(nn.Linear(d, 16, bias=True), nn.Linear(16, d, bias=False))
        return block

    def test_block_output_mean_restored(self):
        from auto_round.algorithms.composer import _apply_block_bias_correction

        block = self._toy_block()
        x = torch.randn(64, 8)
        y_fp = block(x)
        with torch.no_grad():  # simulate quantization error in BOTH linears
            block[0].weight.add_(torch.randn_like(block[0].weight) * 0.02)
            block[1].weight.add_(torch.randn_like(block[1].weight) * 0.02)
        y_q = block(x)
        assert not torch.allclose(y_fp.mean(0), y_q.mean(0), atol=1e-4)

        assert _apply_block_bias_correction(block, y_fp, y_q) is True
        assert block[1].bias is not None, "correction lands on the residual sink (last proj)"
        y_corr = block(x)
        torch.testing.assert_close(y_corr.mean(0), y_fp.mean(0), rtol=0, atol=1e-4)

    def test_absent_bias_created(self):
        from auto_round.algorithms.composer import _apply_block_bias_correction

        block = self._toy_block(seed=4)
        x = torch.randn(32, 8)
        y_fp = block(x)
        with torch.no_grad():
            block[1].weight.mul_(1.01)
        y_q = block(x)
        _apply_block_bias_correction(block, y_fp, y_q)
        assert block[1].bias is not None and block[1].bias.shape == (8,)

    def test_expert_sink_deprioritized(self):
        """Routed experts (token-subset execution) never win over the shared sink."""
        import torch.nn as nn

        from auto_round.algorithms.composer import _apply_block_bias_correction

        class _MoEish(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = nn.Linear(8, 8)
                self.shared = nn.Linear(8, 8)
                # experts LAST in named_modules order: without deprioritization
                # pool[-1] would pick mlp_experts.2 as the sink
                self.mlp_experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

            def forward(self, x):
                return self.shared(self.hidden(x)) + sum(e(x) for e in self.mlp_experts) / 3.0

        block = _MoEish()
        expert_bias_before = [e.bias.detach().clone() for e in block.mlp_experts]
        shared_bias_before = block.shared.bias.detach().clone()
        x = torch.randn(16, 8)
        y_fp = block(x)
        with torch.no_grad():
            block.shared.weight.mul_(0.98)
        y_q = block(x)
        _apply_block_bias_correction(block, y_fp, y_q)
        assert not torch.equal(block.shared.bias.detach(), shared_bias_before), "sink must be shared"
        for e, before in zip(block.mlp_experts, expert_bias_before):
            assert torch.equal(e.bias.detach(), before), "expert biases must stay untouched"

    def test_as_hidden_tensor_dict(self):
        from auto_round.algorithms.composer import _as_hidden_tensor

        t = torch.zeros(2, 3)
        assert _as_hidden_tensor({"hidden_states": t, "foo": 1}) is t
        assert _as_hidden_tensor({"last_hidden_state": t}) is t
        assert _as_hidden_tensor((t,)) is t
        assert _as_hidden_tensor(t) is t

    def test_off_by_default(self):
        import auto_round.envs as envs

        assert not envs.AR_BIAS_CORRECT


class TestQoffNoise:
    """AR_QOFF_NOISE: deterministic per-channel noise injection + guards."""

    @staticmethod
    def _stats_file(tmp_path, hidden=8, mean_val=0.05, var_val=0.04):
        import torch as _t

        d = tmp_path / "stats"
        d.mkdir(exist_ok=True)
        _t.save(
            {"mean": _t.full((hidden,), mean_val), "var": _t.full((hidden,), var_val)},
            d / "block_0000.pt",
        )
        return str(d)

    def test_injection_deterministic_and_math(self, tmp_path, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        x = [torch.zeros(2, 5, 8) for _ in range(2)]

        a = _maybe_qoff_noise(list(x), 1, enable_quanted_input=False)
        b = _maybe_qoff_noise(list(x), 1, enable_quanted_input=False)
        for ta, tb in zip(a, b):
            torch.testing.assert_close(ta, tb, rtol=0, atol=0)  # same seed -> identical
        for tx in x:
            assert torch.equal(tx, torch.zeros_like(tx)), "cached inputs never modified in place"

        # var=0 => x' == x + mean exactly
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path, var_val=0.0))
        c = _maybe_qoff_noise(list(x), 1, enable_quanted_input=False)
        for tc, tx in zip(c, x):
            torch.testing.assert_close(tc - tx, torch.full_like(tx, 0.05), rtol=0, atol=1e-6)

    def test_env_off_is_identity(self, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.delenv("AR_QOFF_NOISE", raising=False)
        x = [torch.zeros(2, 4)]
        out = _maybe_qoff_noise(x, 3, enable_quanted_input=False)
        assert out is x

    def test_qon_guard(self, tmp_path, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        with pytest.raises(ValueError, match="qoff"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 1, enable_quanted_input=True)

    def test_missing_stats_dir_guard(self, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.delenv("AR_QOFF_NOISE_STATS", raising=False)
        with pytest.raises(ValueError, match="AR_QOFF_NOISE_STATS"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 1, enable_quanted_input=False)

    def test_missing_file_guard(self, tmp_path, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        with pytest.raises(ValueError, match="block_0001.pt"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 2, enable_quanted_input=False)  # needs block 1 stats

    def test_block_zero_skips(self, tmp_path, monkeypatch):
        import auto_round.envs as envs
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        x = [torch.zeros(1, 4, 8)]
        assert _maybe_qoff_noise(x, 0, enable_quanted_input=False) is x

    def test_collection_stats_finite(self, tmp_path):
        from auto_round.algorithms.composer import _collect_qoff_noise_stats_from_outputs

        y_fp = torch.zeros(4, 6, 8)
        y_q = torch.randn(4, 6, 8) * 0.01
        path = str(tmp_path / "s" / "block_0000.pt")
        mean, var = _collect_qoff_noise_stats_from_outputs(y_fp, y_q, path)
        assert mean.shape == (8,) and var.shape == (8,)
        assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var >= 0).all()
        import os

        assert os.path.exists(path)


class TestTouchup:
    """AR_TOUCHUP_ITERS: post-BPT serial qon touch-up init."""

    def test_wrapper_anchors_from_touchup_pair(self):
        """The tuned (scale, zp) pair re-derives bit-exactly through the STE grid."""
        from auto_round.data_type.int import quant_tensor_asym
        from auto_round.wrapper import WrapperLinear

        layer = TestWrapperIntegration._armed_linear()
        bits, gs = 4, 16
        maxq = float(2**bits - 1)
        s = torch.full((16, 1), 0.0031)  # per-group scale [out*groups, 1]
        zp = torch.randint(0, int(maxq) + 1, (16, 1)).float()
        layer._touchup_scale, layer._touchup_zp = s, zp
        w = WrapperLinear(
            layer,
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=5,
        )
        assert w._tune_recipe == "touchup"
        assert "value" in w.params and "min_scale" in w.params, "rounding + margins still tunable"
        _, scale_out, zp_out = quant_tensor_asym(
            layer.weight,
            bits=bits,
            group_size=gs,
            tensor_min=w.weight_min,
            tensor_max=w.weight_max,
            scale_dtype=torch.float32,
        )
        torch.testing.assert_close(scale_out.squeeze(-1), s.squeeze(-1), rtol=0, atol=1e-7)
        torch.testing.assert_close(zp_out.squeeze(-1), zp.squeeze(-1), rtol=0, atol=1e-5)

    def test_env_off_is_status_quo(self):
        import auto_round.envs as envs

        assert envs.AR_TOUCHUP_ITERS == 0
        from auto_round.wrapper import WrapperLinear

        layer = TestWrapperIntegration._armed_linear()
        w = WrapperLinear(
            layer,
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=5,
        )
        assert w._tune_recipe == "" and not hasattr(layer, "_touchup_scale")
        assert "value" in w.params and "min_scale" in w.params

    def test_apply_touchup_init_sets_attrs(self, tmp_path):
        import torch.nn as nn

        from auto_round.compressors import block_parallel as bp
        from auto_round.compressors.orchestrator import CompressionOrchestrator, _tuned_layer_key

        block = nn.Sequential(TestWrapperIntegration._armed_linear())
        block[0].global_name = "model.layers.0.mlp.down_proj"

        class _Ctx:
            pass

        class _Model(nn.Module):
            def __init__(self, blk):
                super().__init__()
                self.layers = nn.ModuleList([blk])

        model = _Model(block)
        results = str(tmp_path)
        key = _tuned_layer_key("layers.0", "0", block[0])
        bp.save_block_results(results, "layers.0", {key: {"scale": torch.full((16, 1), 0.002), "zp": 3}})

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = _Ctx()
        orch.model_context.model = model
        orch._apply_touchup_init(block, "layers.0", results)
        assert hasattr(block[0], "_touchup_scale") and abs(float(block[0]._touchup_scale[0, 0]) - 0.002) < 1e-7
        assert block[0]._touchup_zp == 3

    def test_signature_changes_with_touchup_iters(self, monkeypatch):
        import auto_round.envs as envs
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)

        class _NS:
            pass

        orch.model_context = _NS()
        orch.model_context.disk_stream_model_dir = None
        m = _NS()
        m.config = _NS()
        m.config._name_or_path = "test-model"
        orch.model_context.model = m
        orch.scheme = "int4"
        orch.quantizer = _NS()
        orch.quantizer.layer_config = None
        composer = _NS()
        composer.need_quanted_input = lambda: True
        orch._alg_composer = composer
        orch.dataset = None
        orch.calibration_context = _NS()
        orch.calibration_context.nsamples = 8
        orch.calibration_context.seqlen = 512

        monkeypatch.setenv("AR_TOUCHUP_ITERS", "0")
        sig0 = orch._parallel_run_signature([["model.layers.0"]])
        monkeypatch.setenv("AR_TOUCHUP_ITERS", "5")
        sig5 = orch._parallel_run_signature([["model.layers.0"]])
        monkeypatch.setenv("AR_TOUCHUP_ITERS", "7")
        sig7 = orch._parallel_run_signature([["model.layers.0"]])
        assert sig0 != sig5 != sig7, "touch-up count must invalidate stale resume artifacts"


class TestReviewFixes:
    """Regression tests for the code-review findings (imatrix, sym, act-path, Conv1D)."""

    def test_cross_env_guards(self, monkeypatch):
        import auto_round.envs as envs
        from auto_round.data_type import get_quant_func

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        with pytest.raises(ValueError, match="AR_QOFF_NOISE=1 injects"):
            get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=0, asym_search="auto")

        monkeypatch.delenv("AR_QOFF_NOISE", raising=False)
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        monkeypatch.setenv("AR_TOUCHUP_ITERS", "5")
        with pytest.raises(ValueError, match="one or the other"):
            get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=20, asym_search="auto")

    def test_recipe_anchor_with_1d_imatrix(self, monkeypatch):
        """The standard 1D [in_features] imatrix must not crash the anchor search."""
        from auto_round.wrapper import _compute_recipe_anchors

        torch.manual_seed(11)
        w = torch.randn(6, 32) * 0.05
        im1d = torch.rand(32) * 4 + 0.5  # what register_imatrix_hooks attaches
        for recipe in ("neuqi_qon", "opt_rtn_qon"):
            out = _compute_recipe_anchors(w, 4, 32, im1d, recipe, torch.device("cpu"))
            assert out is not None and out[0].shape == (6,)

    def test_refit_with_1d_imatrix(self):
        from auto_round.wrapper import _refit_scale_grid

        torch.manual_seed(12)
        w, s, zp, q, qdq = TestPostScaleRefit._quantized(seed=12)
        im1d = torch.rand(32) * 4 + 0.5
        qdq_new, _ = _refit_scale_grid(w, qdq.clone(), s.clone(), zp, 16, bits=4, qw=im1d)
        assert torch.isfinite(qdq_new).all()

    def test_refit_skips_sym_layers(self, monkeypatch):
        import logging

        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_POST_SCALE_REFIT", "1")
        layer = TestWrapperIntegration._armed_linear()
        layer.sym = True  # symmetric layer
        w = WrapperLinear(
            layer,
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=5,
        )
        w.unwrapper({"value": torch.tensor(0.0), "min_scale": torch.tensor(1.0), "max_scale": torch.tensor(1.0)})
        # sym layer keeps its quant_tensor_sym grid (scale magnitude = s from min/max)

    def test_touchup_sym_fails_fast(self):
        from auto_round.wrapper import WrapperLinear

        layer = TestWrapperIntegration._armed_linear()
        layer.sym = True
        layer._touchup_scale = torch.full((16, 1), 0.003)
        layer._touchup_zp = 3.0
        with pytest.raises(ValueError, match="asym layers only"):
            WrapperLinear(
                layer,
                enable_minmax_tuning=True,
                enable_round_tuning=True,
                enable_norm_bias_tuning=False,
                device="cpu",
                enable_torch_compile=False,
                disable_opt_rtn=True,
                asym_search="auto",
                iters=5,
            )

    def test_act_quant_call_skips_recipe_guards(self, monkeypatch):
        """neuqi_qon + W4A8 (act_sym=True) must not crash wrapper init (R1-10)."""
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        layer = TestWrapperIntegration._armed_linear()
        layer.act_bits = 8  # activation quant engaged, sym default True
        layer.act_data_type = "int"
        layer.act_dynamic = True
        from auto_round.wrapper import WrapperLinear

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
            enable_act_quant=True,
        )
        assert w.act_quant_func is not None

    def test_override_disables_triton_enables_compile(self):
        from auto_round.data_type.neuqi import _zp_wants_compile, _zp_wants_triton, backend_override

        assert _zp_wants_triton("cuda"), "default auto should want triton on cuda"
        with backend_override("compile"):
            assert not _zp_wants_triton("cuda")
            assert _zp_wants_compile("cuda")
        assert _zp_wants_triton("cuda"), "override must restore env-driven behavior"

    def test_override_eager_kills_both(self):
        from auto_round.data_type.neuqi import _zp_wants_compile, _zp_wants_triton, backend_override

        with backend_override("eager"):
            assert not _zp_wants_triton("cuda")
            assert not _zp_wants_compile("cuda")

    def test_override_nests_and_restores(self):
        from auto_round.data_type.neuqi import _zp_wants_triton, backend_override

        with backend_override("compile"):
            with backend_override("eager"):
                assert not _zp_wants_triton("cuda")
            assert not _zp_wants_triton("cuda"), "inner exit restores outer override"
        assert _zp_wants_triton("cuda")


class TestTritonEscapeHatch:
    """AR_NEUQI_BACKEND=triton outranks the wrapper-init compile pin (forensics path)."""

    def test_explicit_triton_wins_over_override(self, monkeypatch):
        import auto_round.envs as envs
        from auto_round.data_type.neuqi import _zp_wants_triton, backend_override

        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        with backend_override("compile"):
            assert _zp_wants_triton("cuda"), "explicit triton must stay available for forensics"


class TestSweepWarmup:
    """Serialized per-device Triton sweep warmup (replaces the compile pin)."""

    def test_cpu_is_noop(self):
        from auto_round.data_type.neuqi import _sweep_warmed_devices, ensure_sweep_warmup

        before = set(_sweep_warmed_devices)
        ensure_sweep_warmup(torch.device("cpu"))
        assert _sweep_warmed_devices == before, "cpu devices are never warmed"

    def test_wrapper_init_search_no_longer_pins_compile(self):
        """The pin was replaced by the warmup; source must not reference backend_override."""
        import inspect

        import auto_round.wrapper as wrapper_mod

        src = inspect.getsource(wrapper_mod._compute_recipe_anchors)
        assert "backend_override" not in src, "wrapper-init searches must keep the Triton default"


class TestTuningFanoutEnv:
    """AR_DISABLE_TUNING_FANOUT flips the auto default to serial; explicit kwarg wins."""

    def test_env_disables_auto_fanout(self, monkeypatch):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        monkeypatch.setenv("AR_DISABLE_TUNING_FANOUT", "1")
        cfg = RTNConfig(bits=4)
        assert cfg.parallel_tuning is False

    def test_env_ignores_off(self, monkeypatch):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        monkeypatch.delenv("AR_DISABLE_TUNING_FANOUT", raising=False)
        cfg = RTNConfig(bits=4)
        assert cfg.parallel_tuning is False  # serial default; opt in with True

    def test_explicit_kwarg_wins_over_env(self, monkeypatch):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        monkeypatch.setenv("AR_DISABLE_TUNING_FANOUT", "1")
        cfg = RTNConfig(bits=4, parallel_tuning=True)
        assert cfg.parallel_tuning is True


class TestBiasCorrectAllRows:
    """The design: b = mean over ALL calibration rows (not the first sample)."""

    def test_drift_in_later_samples_is_corrected(self):
        import torch.nn as nn

        from auto_round.algorithms.composer import _apply_block_bias_correction

        torch.manual_seed(0)
        d = 8
        block = nn.Sequential(nn.Linear(d, 16, bias=True), nn.Linear(16, d, bias=False))
        # sample 0: zero drift; samples 1-3: +1 per-channel drift on ch 0..3
        y_fp = [torch.zeros(1, 5, d) for _ in range(4)]
        y_q = [torch.zeros(1, 5, d) for _ in range(4)]
        for i in range(1, 4):
            y_q[i][..., 0] = 1.0
            y_q[i][..., 1] = -1.0
        assert _apply_block_bias_correction(block, y_fp, y_q) is True
        # old (out[0]) behavior gave b == 0; b = mean(y_fp - y_q) over all rows
        assert abs(block[1].bias[0].item() + 0.75) < 1e-6
        assert abs(block[1].bias[1].item() - 0.75) < 1e-6
        assert torch.all(block[1].bias[2:] == 0)

    def test_noise_stats_use_all_rows(self, tmp_path):
        import os

        from auto_round.algorithms.composer import _collect_qoff_noise_stats_from_outputs

        torch.manual_seed(1)
        y_fp = [torch.randn(1, 16, 4) for _ in range(3)]
        y_q = [t + 0.5 for t in y_fp]
        path = os.path.join(tmp_path, "n", "block_0000.pt")
        mean, var = _collect_qoff_noise_stats_from_outputs(y_fp, y_q, path)
        assert mean.shape == (4,) and var.shape == (4,)
        torch.testing.assert_close(mean, torch.full((4,), -0.5), rtol=0, atol=1e-5)
        torch.testing.assert_close(var, torch.zeros(4), rtol=0, atol=1e-4)  # constant noise
        assert os.path.exists(path)


class TestDeferredRecipeAnchor:
    """Mirrors-first flow: deferred anchor == immediate anchor, bit-exact."""

    @staticmethod
    def _armed(out=8, inp=32, gs=16):
        import torch.nn as nn

        layer = nn.Linear(inp, out, bias=True)
        with torch.no_grad():
            layer.weight.normal_(0, 0.02)
        layer.bits, layer.group_size, layer.sym = 4, gs, False
        layer.data_type, layer.act_bits, layer.act_sym = "int", 16, True
        layer.scale_dtype = torch.float16
        layer.iters = 20
        return layer

    @classmethod
    def _wrap(cls, recipe, monkeypatch, defer=False, weight=None):
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", recipe)
        layer = cls._armed()
        if weight is not None:
            with torch.no_grad():
                layer.weight.copy_(weight)
        return WrapperLinear(
            layer,
            enable_minmax_tuning=True,
            enable_round_tuning=True,
            enable_norm_bias_tuning=False,
            device="cpu",
            enable_torch_compile=False,
            disable_opt_rtn=True,
            asym_search="auto",
            iters=20,
            defer_recipe_anchor=defer,
        )

    def test_deferred_anchor_matches_immediate_frozen(self, monkeypatch):
        w_imm = self._wrap("neuqi_frozen_qon", monkeypatch)
        w_def = self._wrap("neuqi_frozen_qon", monkeypatch, defer=True, weight=w_imm.orig_layer.weight.data)
        assert w_def._recipe_anchor_deferred is True
        assert w_def.anchor_recipe_grid() is True
        torch.testing.assert_close(w_def.weight_min, w_imm.weight_min, rtol=0, atol=0)
        torch.testing.assert_close(w_def.weight_max, w_imm.weight_max, rtol=0, atol=0)
        for w in (w_imm, w_def):
            assert "min_scale" not in w.params and "max_scale" not in w.params
            assert float(w.min_scale) == 1.0 and float(w.max_scale) == 1.0
        assert w_def._tune_recipe_frozen_margins is True

    def test_reanchor_preserves_pinned_margin_storage(self, monkeypatch):
        """Template-cache hit: a re-anchor must not orphan the pinned margin
        storage. Captured graphs bake the storage pointer of ``min_scale``/
        ``max_scale``; the frozen-recipe pin must FILL IN PLACE (same storage)
        instead of allocating a fresh 1.0 tensor, or the cached graph reads
        recycled foreign memory as the scale margins (deterministic huge loss
        from iter 0 while eager stays correct)."""
        w = self._wrap("neuqi_frozen_qon", monkeypatch, defer=True)
        assert w.anchor_recipe_grid() is True
        ptr_min, ptr_max = w.min_scale.data_ptr(), w.max_scale.data_ptr()
        # hit flow: sync re-arms the deferred flag, the anchor runs AGAIN on
        # the adopted mirror (same wrapper object the graph was captured on)
        w._recipe_anchor_deferred = True
        assert w.anchor_recipe_grid() is True
        assert w.min_scale.data_ptr() == ptr_min, "pin orphaned min_scale storage"
        assert w.max_scale.data_ptr() == ptr_max, "pin orphaned max_scale storage"
        assert float(w.min_scale) == 1.0 and float(w.max_scale) == 1.0
        # registration semantics unchanged: constants, out of the tuned pool
        assert "min_scale" not in w.params and "max_scale" not in w.params
        assert "min_scale" not in dict(w.named_parameters())
        assert "max_scale" not in dict(w.named_parameters())
        assert not isinstance(w.min_scale, torch.nn.Parameter)

    def test_shard_helper_broadcasts_grids(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _shard_recipe_anchor

        class _RG:
            def __init__(self, reps):
                self.replicas = reps
                self.home = reps[0]

            def run_threaded(self, fns):
                for fn in fns:
                    fn()

        import torch.nn as nn

        base = self._armed()
        blocks = []
        for i in range(2):
            monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_frozen_qon")
            layer = self._armed()
            with torch.no_grad():
                layer.weight.copy_(base.weight.data)  # identical across "replicas"
            blk = nn.Sequential()
            blk.add_module("l", layer)
            blk.l = self._wrap.__func__.__self__._wrap_layer(layer) if False else _wrap_layer(layer)
            blocks.append(blk)
        _shard_recipe_anchor(_RG(blocks))
        for blk in blocks:
            assert getattr(blk.l, "_recipe_anchor_deferred", False) is False
            assert "min_scale" not in blk.l.params
        torch.testing.assert_close(blocks[0].l.weight_min, blocks[1].l.weight_min, rtol=0, atol=0)
        torch.testing.assert_close(blocks[0].l.weight_max, blocks[1].l.weight_max, rtol=0, atol=0)


def _wrap_layer(layer):
    from auto_round.wrapper import WrapperLinear

    return WrapperLinear(
        layer,
        enable_minmax_tuning=True,
        enable_round_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_torch_compile=False,
        disable_opt_rtn=True,
        asym_search="auto",
        iters=20,
        defer_recipe_anchor=True,
    )


class TestFanOutDefault:
    def test_parallel_tuning_defaults_serial(self, monkeypatch):
        monkeypatch.delenv("AR_DISABLE_TUNING_FANOUT", raising=False)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig()
        assert cfg.parallel_tuning is False


class TestEnableNeuqiDefaultRecipe:
    """--enable_neuqi + iters>0: AR_TUNE_RECIPE defaults to neuqi_frozen_qon
    (the NeUQI init reaches SignRound through the recipe anchor)."""

    def test_helper_defaults_frozen_recipe(self, monkeypatch):
        import os

        from auto_round.data_type.utils import maybe_default_tune_recipe

        monkeypatch.delenv("AR_TUNE_RECIPE", raising=False)
        try:
            assert maybe_default_tune_recipe("neuqi", 200) == "neuqi_frozen_qon"
            assert os.environ["AR_TUNE_RECIPE"] == "neuqi_frozen_qon"
        finally:
            os.environ.pop("AR_TUNE_RECIPE", None)

    def test_helper_respects_explicit_recipe(self, monkeypatch):
        import os

        from auto_round.data_type.utils import maybe_default_tune_recipe

        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        assert maybe_default_tune_recipe("neuqi", 200) is None
        assert os.environ["AR_TUNE_RECIPE"] == "neuqi_qon"

    def test_helper_no_default_at_iters0_or_without_neuqi(self, monkeypatch):
        import os

        from auto_round.data_type.utils import maybe_default_tune_recipe

        monkeypatch.delenv("AR_TUNE_RECIPE", raising=False)
        try:
            assert maybe_default_tune_recipe("neuqi", 0) is None
            assert maybe_default_tune_recipe("auto", 200) is None
            assert maybe_default_tune_recipe("minmax", 200) is None
            assert "AR_TUNE_RECIPE" not in os.environ
        finally:
            os.environ.pop("AR_TUNE_RECIPE", None)

    def test_wrapper_anchors_from_defaulted_recipe(self, monkeypatch):
        # the compressor materializes the default in the env; the wrapper then
        # anchors asym layers and skips sym layers in a mixed pool
        from auto_round.data_type import utils as dt_utils

        monkeypatch.setattr(dt_utils, "_MIXED_SYM_POOL", True, raising=False)
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_frozen_qon")
        w = TestWrapperMixedSymPool()._wrapper("neuqi_frozen_qon", sym=False, monkeypatch=monkeypatch)
        assert w._tune_recipe == "neuqi_frozen_qon"
        assert float(w.min_scale) == 1.0 and float(w.max_scale) == 1.0
        w_sym = TestWrapperMixedSymPool()._wrapper("neuqi_frozen_qon", sym=True, monkeypatch=monkeypatch)
        assert w_sym._tune_recipe == ""
        assert "min_scale" in w_sym.params

    def test_get_quant_func_accepts_neuqi_iters_with_defaulted_recipe(self, monkeypatch):
        # compressor-driven runs always see a materialized recipe; the direct
        # call path with the defaulted recipe resolves the asym STE function
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_frozen_qon")
        fn, name = TestGuardMatrix()._g(asym_search="neuqi")
        assert name.startswith("int_asym")
