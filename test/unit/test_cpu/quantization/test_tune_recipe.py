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

    def test_alt2_guards(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_RECIPE", "alt2")
        self._g()  # implemented now: resolves cleanly
        with pytest.raises(ValueError, match="alt2"):
            self._g(sym=True)
        with pytest.raises(ValueError, match="2 tuning iterations"):
            self._g(iters=1)

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


class TestAlt2:
    """alt2: alternating re-grid — switch math, dispatch, wrapper re-grid."""

    def test_switch_iter_math(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import alt2_switch_iter

        assert alt2_switch_iter(20, "alt2", 0) == 10  # default half
        assert alt2_switch_iter(20, "alt2", 5) == 15
        assert alt2_switch_iter(20, "neuqi_qon", 5) is None  # other recipes
        assert alt2_switch_iter(1, "alt2", 0) is None  # needs >= 2 iters
        assert alt2_switch_iter(0, "alt2", 0) is None

    def test_switch_iter_bounds_guard(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import alt2_switch_iter

        for bad in (0, -1, 20, 21):
            if bad == 0:  # 0 = half, valid
                continue
            try:
                alt2_switch_iter(20, "alt2", bad)
                raised = False
            except ValueError as e:
                raised = "AR_ALT2_ITERS2" in str(e)
            if bad in (-1, 20, 21):
                assert raised, f"bounds guard failed for {bad}"

    def test_dispatch_alt2_no_longer_raises(self, monkeypatch):
        from auto_round.data_type import get_quant_func

        monkeypatch.setenv("AR_TUNE_RECIPE", "alt2")
        fn, dt = get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=20, asym_search="auto")
        assert callable(fn)
        # sym alt2 and iters<2 must fail fast
        for sym, iters in ((True, 20), (False, 1)):
            try:
                get_quant_func("int", 4, sym, disable_opt_rtn=True, group_size=128, iters=iters, asym_search="auto")
                raised = False
            except ValueError:
                raised = True
            assert raised, f"alt2 guard missing for sym={sym} iters={iters}"

    def test_wrapper_regrid_anchors_and_resets_v(self, monkeypatch):
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", "alt2")
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
            iters=20,
        )
        assert w._tune_recipe == "alt2"
        assert "value" in w.params
        with torch.no_grad():  # perturb v as round 1 would
            w.params["value"].uniform_(-0.4, 0.4)
        min_before, max_before = w.weight_min.clone(), w.weight_max.clone()

        ds = w.alt2_regrid()
        assert ds is not None and ds >= 0
        assert torch.equal(w.params["value"].data, torch.zeros_like(w.params["value"])), "v reset to 0"
        # anchors are a fresh search on the perturbed weights -> may move
        assert w.weight_min.shape == min_before.shape

    def test_regrid_skips_non_alt2_layers(self, monkeypatch):
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
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
            iters=20,
        )
        assert w.alt2_regrid() is None  # not an alt2 layer


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
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

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
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

        monkeypatch.delenv("AR_QOFF_NOISE", raising=False)
        x = [torch.zeros(2, 4)]
        out = _maybe_qoff_noise(x, 3, enable_quanted_input=False)
        assert out is x

    def test_qon_guard(self, tmp_path, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        with pytest.raises(ValueError, match="qoff"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 1, enable_quanted_input=True)

    def test_missing_stats_dir_guard(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.delenv("AR_QOFF_NOISE_STATS", raising=False)
        with pytest.raises(ValueError, match="AR_QOFF_NOISE_STATS"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 1, enable_quanted_input=False)

    def test_missing_file_guard(self, tmp_path, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        monkeypatch.setenv("AR_QOFF_NOISE_STATS", self._stats_file(tmp_path))
        with pytest.raises(ValueError, match="block_0001.pt"):
            _maybe_qoff_noise([torch.zeros(1, 4, 8)], 2, enable_quanted_input=False)  # needs block 1 stats

    def test_block_zero_skips(self, tmp_path, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _maybe_qoff_noise

        import auto_round.envs as envs

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
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        import auto_round.envs as envs

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

    def test_cross_env_guards(self, monkeypatch):
        from auto_round.data_type import get_quant_func

        import auto_round.envs as envs

        monkeypatch.setenv("AR_ALT2_ITERS2", "10")
        monkeypatch.setenv("AR_TUNE_RECIPE", "neuqi_qon")
        with pytest.raises(ValueError, match="AR_ALT2_ITERS2 only applies"):
            get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=20, asym_search="auto")

        monkeypatch.setenv("AR_ALT2_ITERS2", "0")
        monkeypatch.setenv("AR_QOFF_NOISE", "1")
        with pytest.raises(ValueError, match="AR_QOFF_NOISE=1 injects"):
            get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=0, asym_search="auto")

        monkeypatch.delenv("AR_QOFF_NOISE", raising=False)
        monkeypatch.setenv("AR_TOUCHUP_ITERS", "5")
        with pytest.raises(ValueError, match="one or the other"):
            get_quant_func("int", 4, False, disable_opt_rtn=True, group_size=128, iters=20, asym_search="auto")

    def test_regrid_rederives_searched_grid(self, monkeypatch):
        """After alt2_regrid the STE grid equals a manual search on w + v*s (R1-17)."""
        from auto_round.data_type.int import quant_tensor_asym
        from auto_round.data_type.neuqi import neuqi_search_scale_zero
        from auto_round.wrapper import WrapperLinear

        monkeypatch.setenv("AR_TUNE_RECIPE", "alt2")
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
            iters=20,
        )
        with torch.no_grad():
            w.params["value"].uniform_(-0.4, 0.4)
            # simulate tuned margins != 1.0
            w.params["min_scale"].fill_(0.95)
            w.params["max_scale"].fill_(1.05)
            # snapshot the pre-switch state the manual reference must replay
            v_snap = w.params["value"].detach().clone()
            maxq = float(2**4 - 1)
            s_eff = (w.weight_max.float().reshape(-1, 1) * 1.05 - w.weight_min.float().reshape(-1, 1) * 0.95) / maxq
            w_eff_snap = layer.weight.float().reshape(16, 16) + v_snap.reshape(16, 16) * s_eff
            w.alt2_regrid()
        # margins must be back at 1.0 against the new anchors
        torch.testing.assert_close(w.min_scale.detach(), torch.full_like(w.min_scale.detach(), 1.0), rtol=0, atol=1e-6)
        torch.testing.assert_close(w.max_scale.detach(), torch.full_like(w.max_scale.detach(), 1.0), rtol=0, atol=1e-6)
        # STE re-derivation matches a manual search on the pre-switch effective weights
        with torch.no_grad():
            s_man, zp_man = neuqi_search_scale_zero(w_eff_snap, 4)
            _, scale_out, zp_out = quant_tensor_asym(
                layer.weight,
                bits=4,
                group_size=16,
                tensor_min=w.weight_min,
                tensor_max=w.weight_max,
                scale_dtype=torch.float32,
            )
        torch.testing.assert_close(scale_out.squeeze(-1), s_man.squeeze(-1), rtol=0, atol=1e-7)
        torch.testing.assert_close(zp_out.squeeze(-1), zp_man.squeeze(-1), rtol=0, atol=1e-5)


class TestBackendOverride:
    """Recipe wrapper-init searches pin the compile backend (Triton race workaround)."""

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
        from auto_round.data_type.neuqi import _zp_wants_triton, backend_override

        import auto_round.envs as envs

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
