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
"""Tests for the Pre-SINQ transform (Sinkhorn-normalised weight folding).

Covers: sinkhorn scale properties, function-preserving exactness of the folds
(dense attention, hybrid linear-attention, MoE, optional v<->o), the (1+weight)
RMSNorm convention, config routing and the SpinQuant mutual-exclusion guard.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.transforms.presinq import PreSINQConfig, PreSINQRotation
from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales, sinkhorn_log

D = 64  # hidden
I = 96  # intermediate


# ---------------------------------------------------------------------------
# Tiny model fixtures
# ---------------------------------------------------------------------------
class StdRMSNorm(nn.Module):
    """Standard RMSNorm: out = normed(x) * weight (gamma IS the weight)."""

    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return normed * self.weight


class OnePlusRMSNorm(StdRMSNorm):
    """qwen3_next/qwen3_5 convention: out = normed(x) * (1 + weight)."""

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return normed * (1 + self.weight)


class TinyAttention(nn.Module):
    """MHA (heads == kv heads) so the optional v<->o fold is exact."""

    def __init__(self, d: int, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.hd = d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.hd).transpose(1, 2)

    def forward(self, x):
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(x)), self._split(self.v_proj(x))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1) @ v
        return self.o_proj(att.transpose(1, 2).reshape(x.shape))


class TinyLinearAttn(nn.Module):
    """qwen3_5 GatedDeltaNet-style surface: in_proj_{qkv,z,a,b} + out_proj."""

    def __init__(self, d: int):
        super().__init__()
        self.in_proj_qkv = nn.Linear(d, 3 * d, bias=False)
        self.in_proj_z = nn.Linear(d, d, bias=False)
        self.in_proj_a = nn.Linear(d, d, bias=False)
        self.in_proj_b = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv[..., :D], qkv[..., D : 2 * D], qkv[..., 2 * D :]
        h = F.silu(q) * torch.sigmoid(self.in_proj_z(x)) + k * torch.tanh(self.in_proj_a(x)) + v * self.in_proj_b(x)
        return self.out_proj(h)


class TinyMLP(nn.Module):
    def __init__(self, d: int, i: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyMoEMLP(nn.Module):
    def __init__(self, d: int, i: int, n_experts: int = 2):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([TinyMLP(d, i) for _ in range(n_experts)])

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)
        return sum(w[..., e : e + 1] * self.experts[e](x) for e in range(len(self.experts)))


class TinyLayer(nn.Module):
    def __init__(self, d: int, i: int, linear_attn: bool = False, moe: bool = False):
        super().__init__()
        self.input_layernorm = StdRMSNorm(d)
        self.post_attention_layernorm = StdRMSNorm(d)
        self.self_attn = None
        self.linear_attn = None
        if linear_attn:
            self.linear_attn = TinyLinearAttn(d)
        else:
            self.self_attn = TinyAttention(d)
        self.mlp = TinyMoEMLP(d, i) if moe else TinyMLP(d, i)

    def forward(self, x):
        attn = self.self_attn if self.self_attn is not None else self.linear_attn
        return x + attn(self.input_layernorm(x)) + self.mlp(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, d: int = D, i: int = I, layers: int = 2, linear_attn: bool = False, moe: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, d)
        self.layers = nn.ModuleList([TinyLayer(d, i, linear_attn, moe) for _ in range(layers)])
        self.norm = StdRMSNorm(d)
        self.lm_head = nn.Linear(d, 32, bias=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def _rel_err(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-12))


def _forward_logits(model):
    torch.manual_seed(0)
    ids = torch.randint(0, 32, (2, 16))
    with torch.no_grad():
        return model(ids)


def _assert_exact(model, atol_rel=1e-5):
    before = _forward_logits(model).clone()
    PreSINQRotation(PreSINQConfig()).apply_to_model(model, data_type="int", use_tqdm=False)
    after = _forward_logits(model)
    assert _rel_err(after, before) < atol_rel, f"fold changed the function: rel_err={_rel_err(after, before):.3e}"


# ---------------------------------------------------------------------------
# sinkhorn port
# ---------------------------------------------------------------------------
class TestSinkhorn:
    def test_shapes_and_positivity(self):
        torch.manual_seed(0)
        W = torch.randn(128, 64, dtype=torch.float64) * torch.logspace(-2, 2, 64)
        scaled, mu1, mu2 = sinkhorn_log(W, order=4)
        assert scaled.shape == W.shape and mu1.shape == (64,) and mu2.shape == (128, 1)
        assert bool((mu1 > 0).all()) and bool((mu2 > 0).all())
        assert torch.allclose(scaled, (W / mu1) / mu2)

    def test_column_scales_median_normalised(self):
        torch.manual_seed(0)
        W = [torch.randn(50, 64) * torch.logspace(-1, 1, 64)]
        t = column_scales(W, group_size=32, n_iter=4)
        assert t.shape == (64,) and torch.isfinite(t).all() and bool((t > 0).all())
        assert abs(float(t.median()) - 1.0) < 1e-9

    def test_block_auto_adjust(self):
        t = column_scales([torch.randn(8, 60)], group_size=64, n_iter=1)
        assert t.shape == (60,)

    @staticmethod
    def _reference_impl(matrix, order, clip_min, clip_max, eps, stop):
        """The original per-iteration loop (std recomputed inside imbalance and
        again for the update; scaled always computed) — the bit-exactness oracle."""
        m = matrix.to(torch.float64)

        def imbalance(mat):
            s1 = mat.std(dim=-1)
            s2 = mat.std(dim=-2)
            s_min = torch.minimum(s1.amin(dim=-1), s2.amin(dim=-1)).clamp_min(1e-12)
            s_max = torch.maximum(s1.amax(dim=-1), s2.amax(dim=-1))
            return s_max / s_min

        imb_min = torch.full(m.shape[:-2], float("inf"), dtype=torch.float64, device=m.device)
        tgt_small = (
            torch.minimum(m.std(-1).clamp(clip_min, clip_max).amin(-1), m.std(-2).clamp(clip_min, clip_max).amin(-1))
            + eps
        )
        log_mu1 = torch.zeros(*m.shape[:-2], m.shape[-1], dtype=torch.float64, device=m.device)
        log_mu2 = torch.zeros(*m.shape[:-2], m.shape[-2], 1, dtype=torch.float64, device=m.device)
        mu1_star = log_mu1.exp().clone()
        mu2_star = log_mu2.exp().clone()
        imb_min = torch.minimum(imb_min, imbalance(m))
        gate = torch.zeros_like(imb_min)
        for _ in range(order):
            cur = (m / log_mu1.exp().unsqueeze(-2)) / log_mu2.exp()
            ib = imbalance(cur)
            better = ib <= imb_min
            imb_min = torch.minimum(imb_min, ib)
            mu1_star = torch.where(better.unsqueeze(-1), log_mu1.exp(), mu1_star)
            mu2_star = torch.where(better.unsqueeze(-1).unsqueeze(-1), log_mu2.exp(), mu2_star)
            if stop:
                rising = (ib > imb_min).to(torch.float64)
                gate = torch.clip(gate + rising, max=1.0)
            g = 1.0 - gate
            std_r = cur.std(dim=-1).clamp(clip_min, clip_max)
            std_c = cur.std(dim=-2).clamp(clip_min, clip_max)
            sal_col = (std_c / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()
            sal_row = (std_r / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()
            log_mu1 = (log_mu1 + sal_col * g.unsqueeze(-1)).clip(-0.3, 10.0)
            log_mu2 = (log_mu2 + sal_row.unsqueeze(-1) * g.unsqueeze(-1).unsqueeze(-1)).clip(-0.3, 10.0)
        scaled = m / mu1_star.unsqueeze(-2) / mu2_star
        return scaled, mu1_star, mu2_star

    def test_optimized_loop_is_bit_exact(self):
        """The std-once / exp-hoisted refactor must reproduce the original loop
        bit-for-bit on every return value."""
        torch.manual_seed(1)
        cases = [
            (torch.randn(128, 64) * torch.logspace(-2, 2, 64), 4),
            (torch.randn(3, 96, 48) * torch.logspace(-2, 2, 48), 8),
            (torch.randn(32, 100), 0),
        ]
        for W, order in cases:
            for stop in (True, False):
                ref = self._reference_impl(W, order, 1e-3, 1e3, 1e-6, stop)
                got = sinkhorn_log(W, order=order, stop_on_increasing_imbalance=stop)
                for r, gv in zip(ref, got):
                    assert torch.equal(r, gv), f"order={order} stop={stop} {tuple(W.shape)}"

    def test_want_scaled_false(self):
        torch.manual_seed(2)
        W = torch.randn(64, 48) * torch.logspace(-2, 2, 48)
        ref_scaled, ref_mu1, ref_mu2 = sinkhorn_log(W, order=4)
        scaled, mu1, mu2 = sinkhorn_log(W, order=4, want_scaled=False)
        assert scaled is None
        assert torch.equal(mu1, ref_mu1) and torch.equal(mu2, ref_mu2)


class TestSinkhornBackend:
    """AR_PRESINQ_BACKEND compile arm: same math fused via torch.compile."""

    def test_policy(self, monkeypatch):
        from auto_round import envs
        from auto_round.algorithms.transforms.presinq.sinkhorn import _wants_compile

        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "auto")
        assert _wants_compile("cuda") is True
        assert _wants_compile("cpu") is False
        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "compile")
        assert _wants_compile("cpu") is True
        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "eager")
        assert _wants_compile("cuda") is False

    @pytest.mark.enable_torch_compile
    def test_compiled_close_to_eager(self, monkeypatch):
        pytest.importorskip("torch._inductor")
        from auto_round import envs
        from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales as cs

        torch.manual_seed(3)
        W = [torch.randn(96, 128) * torch.logspace(-1, 1, 128)]
        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "eager")
        t_ref = cs(W, group_size=32, n_iter=4)
        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "compile")
        import auto_round.algorithms.transforms.presinq.sinkhorn as S

        S._compiled_broken = False
        S._compiled_body = None
        t_c = cs(W, group_size=32, n_iter=4)
        if S._compiled_broken:
            pytest.skip("torch.compile/inductor unavailable in this environment")
        torch.testing.assert_close(t_c, t_ref, rtol=1e-10, atol=1e-12)

    def test_compile_failure_latches(self, monkeypatch):
        from auto_round import envs
        import auto_round.algorithms.transforms.presinq.sinkhorn as S
        from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales as cs

        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "compile")

        def boom(*a, **k):
            raise RuntimeError("simulated compile failure")

        monkeypatch.setattr(S, "_compiled_body", boom)
        monkeypatch.setattr(S, "_compiled_broken", False)
        torch.manual_seed(4)
        W = [torch.randn(64, 64)]
        t_fb = cs(W, group_size=32, n_iter=2)
        assert S._compiled_broken is True
        monkeypatch.setattr(envs, "AR_PRESINQ_BACKEND", "eager")
        torch.testing.assert_close(t_fb, cs(W, group_size=32, n_iter=2))


# ---------------------------------------------------------------------------
# fold exactness
# ---------------------------------------------------------------------------
class TestFoldExactness:
    def test_dense_attention(self):
        torch.manual_seed(7)
        _assert_exact(TinyModel())

    def test_hybrid_linear_attn(self):
        torch.manual_seed(7)
        _assert_exact(TinyModel(linear_attn=True))

    def test_moe(self):
        torch.manual_seed(7)
        _assert_exact(TinyModel(moe=True))

    def test_multiple_repeats(self):
        torch.manual_seed(7)
        model = TinyModel()
        before = _forward_logits(model).clone()
        PreSINQRotation(PreSINQConfig(n_repeat=3)).apply_to_model(model, data_type="int", use_tqdm=False)
        assert _rel_err(_forward_logits(model), before) < 1e-5

    def test_normalize_outproj_mha(self):
        torch.manual_seed(7)
        model = TinyModel()
        before = _forward_logits(model).clone()
        PreSINQRotation(PreSINQConfig(normalize_outproj=True)).apply_to_model(model, data_type="int", use_tqdm=False)
        assert _rel_err(_forward_logits(model), before) < 1e-5

    def test_one_plus_weight_norm(self):
        """(1+weight) norms must fold t into the effective gamma (1+w), not w."""
        torch.manual_seed(7)
        model = TinyModel()
        for layer in model.layers:
            layer.input_layernorm.__class__ = OnePlusRMSNorm
            layer.post_attention_layernorm.__class__ = OnePlusRMSNorm
        _assert_exact(model)


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------
class TestPlumbing:
    def test_registry_lookup(self):
        from auto_round.algorithms.transforms.base import BaseRotation

        rotation = BaseRotation.from_config(PreSINQConfig())
        assert isinstance(rotation, PreSINQRotation)

    def test_normalize_str_and_dict(self):
        from auto_round.algorithms.transforms import normalize_rotation_config

        assert isinstance(normalize_rotation_config("presinq"), PreSINQConfig)
        cfg = normalize_rotation_config({"algorithm": "presinq", "n_repeat": 2, "bogus": 1})
        assert isinstance(cfg, PreSINQConfig) and cfg.n_repeat == 2

    def test_routing_keeps_optimized_rtn(self, monkeypatch):
        from auto_round.autoround import AutoRound as NewAutoRound
        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor

        captured = {}

        def _fake_init(self, config, **kwargs):
            captured["config"] = config

        monkeypatch.setattr(Compressor, "__init__", _fake_init)
        monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *a, **k: False)
        monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *a, **k: False)
        monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *a, **k: "llm")

        NewAutoRound(
            "dummy-model",
            scheme="W4A16",
            alg_configs=[PreSINQConfig(), RTNConfig(group_size=32, disable_opt_rtn=False)],
            seqlen=8,
            nsamples=1,
        )
        configs = captured["config"] if isinstance(captured["config"], list) else [captured["config"]]
        rtn = next(c for c in configs if isinstance(c, RTNConfig))
        assert isinstance(rtn, OptimizedRTNConfig), "PreSINQ must not silently drop the optimized-RTN search"
        assert any(isinstance(c, PreSINQConfig) for c in configs)

    def test_forced_imatrix_asym_without_neuqi_raises(self, monkeypatch):
        """asym + forced imatrix without the NeUQI joint search: get_quant_func
        resolves the plain min/max rtn_*_asym function which never consumes the
        imatrix -- must stop at validation with the --enable_neuqi fix."""
        from auto_round.autoround import AutoRound as NewAutoRound
        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor

        monkeypatch.setattr(Compressor, "__init__", lambda self, config, **kw: None)
        monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *a, **k: False)
        monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *a, **k: False)
        monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *a, **k: "llm")
        cfg = RTNConfig(group_size=32, sym=False, disable_opt_rtn=False)  # asym_search=auto
        cfg.forced_imatrix = True
        with pytest.raises(ValueError, match="enable_neuqi"):
            NewAutoRound("dummy-model", scheme="W4A16", alg_configs=[cfg], seqlen=8, nsamples=1)

    def test_forced_imatrix_with_opt_rtn_disabled_raises(self, monkeypatch):
        """--imatrix_enabled true x --disable_opt_rtn is inconsistent: the

        imatrix only weights the optimized-RTN search, so forcing it on while

        disabling the search must stop at validation, not collect statistics

        nothing consumes."""

        from auto_round.autoround import AutoRound as NewAutoRound

        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor

        monkeypatch.setattr(Compressor, "__init__", lambda self, config, **kw: None)

        monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *a, **k: False)

        monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *a, **k: False)

        monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *a, **k: "llm")

        cfg = RTNConfig(group_size=32, sym=False, disable_opt_rtn=True)

        cfg.forced_imatrix = True

        with pytest.raises(ValueError, match="imatrix_enabled true requires the optimized-RTN"):

            NewAutoRound("dummy-model", scheme="W4A16", alg_configs=[cfg], seqlen=8, nsamples=1)

    def test_enable_opt_rtn_beats_w8_auto_rule(self):
        """--enable_opt_rtn records a non-None sentinel, so the W8A16/W8A8

        efficiency heuristic must NOT flip the search off when the user asked

        for it explicitly."""

        forced_on = RTNConfig(bits=8, act_bits=16, data_type="int", enable_opt_rtn=True)

        assert forced_on.orig_disable_opt_rtn is False, "explicit enable must pin the sentinel"

        assert forced_on.disable_opt_rtn is False

        unset = RTNConfig(bits=8, act_bits=16, data_type="int")

        assert unset.orig_disable_opt_rtn is None, "unset leaves room for the auto rule"

    def test_forced_imatrix_marker_honored_in_selection(self, monkeypatch):
        """--imatrix_enabled true must survive the scheme rules: asym resolves

        the imatrix off, but the forced marker keeps the OptimizedRTNConfig

        class with the imatrix enabled (autoround selection honors it)."""

        from auto_round.autoround import AutoRound as NewAutoRound

        from auto_round.compressors.orchestrator import CompressionOrchestrator as Compressor

        captured = {}

        def _fake_init(self, config, **kwargs):

            captured["config"] = config

        monkeypatch.setattr(Compressor, "__init__", _fake_init)

        monkeypatch.setattr("auto_round.utils.is_mllm_model", lambda *a, **k: False)

        monkeypatch.setattr("auto_round.utils.is_diffusion_model", lambda *a, **k: False)

        monkeypatch.setattr("auto_round.utils.model.detect_model_type", lambda *a, **k: "llm")

        cfg = RTNConfig(group_size=32, sym=False, asym_search="neuqi", disable_opt_rtn=False)  # asym + NeUQI

        cfg.forced_imatrix = True

        NewAutoRound(
            "dummy-model",
            scheme="W4A16",
            alg_configs=[cfg],
            seqlen=8,
            nsamples=1,
        )

        selected = captured["config"] if not isinstance(captured["config"], list) else captured["config"][0]

        assert isinstance(selected, OptimizedRTNConfig), "forced imatrix must keep the optimized-RTN class"

        assert selected.enable_imatrix is True

    def test_spinquant_mutual_exclusion(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

        with pytest.raises(ValueError, match="[Pp]re-?SINQ.*SpinQuant"):
            AlgorithmComposer([PreSINQConfig(), SpinQuantConfig(r1=True), RTNConfig()])

    def test_streamed_without_layerwise_raises(self):
        from types import SimpleNamespace

        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer([PreSINQConfig(group_size=16, n_iter=1, n_repeat=1), RTNConfig(group_size=16)])
        composer._layerwise_rotation = False
        composer._orchestrator_ref = SimpleNamespace(stream_quantization=True)
        meta_model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32)).to("meta")
        with pytest.raises(ValueError, match=r"layer-wise.*--layerwise_rotation"):
            composer.apply_model_transforms(meta_model)

    def test_meta_params_without_stream_attr_raise(self):
        # meta skeleton detection must not depend on the stream_quantization attr
        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer([PreSINQConfig(group_size=16, n_iter=1, n_repeat=1), RTNConfig(group_size=16)])
        composer._layerwise_rotation = False
        composer._orchestrator_ref = None
        meta_model = nn.Sequential(nn.Linear(32, 32)).to("meta")
        with pytest.raises(ValueError, match="layer-wise"):
            composer.apply_model_transforms(meta_model)

    def test_streamed_with_layerwise_defers_without_materializing(self):
        from types import SimpleNamespace

        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer([PreSINQConfig(group_size=16, n_iter=1, n_repeat=1), RTNConfig(group_size=16)])
        composer._layerwise_rotation = True
        composer._orchestrator_ref = SimpleNamespace(stream_quantization=True)
        meta_model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32)).to("meta")
        model = composer.apply_model_transforms(meta_model)
        assert model is meta_model
        assert any(p.device.type == "meta" for p in model.parameters()), "layerwise mode must not materialize"

    def test_layerwise_fallback_to_full_rotation_guarded(self, monkeypatch):
        """A rotation without layerwise support falls back to whole-model

        rotation even in layerwise mode; on a streamed skeleton that fallback

        must hit the same actionable guard instead of folding meta weights."""

        from types import SimpleNamespace

        from auto_round.algorithms.composer import AlgorithmComposer

        from auto_round.algorithms.transforms.presinq.apply import PreSINQRotation

        monkeypatch.setattr(PreSINQRotation, "supports_layerwise", False)

        composer = AlgorithmComposer([PreSINQConfig(group_size=16, n_iter=1, n_repeat=1), RTNConfig(group_size=16)])

        composer._layerwise_rotation = True

        composer._orchestrator_ref = SimpleNamespace(stream_quantization=True)

        meta_model = nn.Sequential(nn.Linear(32, 32)).to("meta")

        with pytest.raises(ValueError, match="layer-wise"):

            composer.apply_model_transforms(meta_model)

    def test_unstreamed_full_rotation_still_runs(self):
        from types import SimpleNamespace

        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer([PreSINQConfig(group_size=16, n_iter=1, n_repeat=1), RTNConfig(group_size=16)])
        composer._layerwise_rotation = False
        composer._orchestrator_ref = SimpleNamespace(stream_quantization=False)
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))
        out = composer.apply_model_transforms(model)
        assert all(p.device.type != "meta" for p in out.parameters())


# ---------------------------------------------------------------------------
# Review-driven edge cases
# ---------------------------------------------------------------------------
class TinyGQAAttention(nn.Module):
    """GQA (2 q heads, 1 kv head) with a biased v_proj; carries head_dim."""

    def __init__(self, d: int, heads: int = 2, kv_heads: int = 1):
        super().__init__()
        assert d % heads == 0 and d % kv_heads == 0
        self.head_dim = d // heads
        self.num_key_value_heads = kv_heads
        self.heads, self.kv_heads = heads, kv_heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        hd, h, kv = self.head_dim, self.heads, self.kv_heads
        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
        v = self.v_proj(x).view(b, t, kv, hd).transpose(1, 2).repeat_interleave(h // kv, dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1) @ v
        return self.o_proj(att.transpose(1, 2).reshape(x.shape))


class TinyMLAAttention(nn.Module):
    """MLA-style: q_proj + kv_a_proj_with_mqa both consume the norm output."""

    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, 2 * d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        kv = self.kv_a_proj_with_mqa(x)
        h = F.silu(self.q_proj(x)) * torch.sigmoid(kv[..., :D]) + kv[..., D:]
        return self.o_proj(h)


class TinyWeirdAttention(nn.Module):
    """Attention with an unknown norm consumer (should abort the fold, not break)."""

    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.weird_proj = nn.Linear(d, d, bias=False)  # unknown consumer

    def forward(self, x):
        base = F.silu(self.q_proj(x)) * torch.sigmoid(self.k_proj(x)) + self.v_proj(x)
        return self.o_proj(base) + self.weird_proj(x)


class TinySharedExpertMoE(nn.Module):
    def __init__(self, d: int, i: int, n_experts: int = 2):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([TinyMLP(d, i) for _ in range(n_experts)])
        self.shared_expert = TinyMLP(d, i)
        self.shared_expert_gate = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)
        out = sum(w[..., e : e + 1] * self.experts[e](x) for e in range(len(self.experts)))
        return out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class TinyFusedMLP(nn.Module):
    """Fused gate_up projection (get_proj returns the same module for both)."""

    def __init__(self, d: int, i: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(d, 2 * i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def forward(self, x):
        gu = self.gate_up_proj(x)
        g, u = gu[..., : gu.shape[-1] // 2], gu[..., gu.shape[-1] // 2 :]
        return self.down_proj(F.silu(g) * u)


class TinyBiasedMLP(TinyMLP):
    def __init__(self, d: int, i: int):
        super().__init__(d, i)
        self.up_proj = nn.Linear(d, i, bias=True)  # bias must scale with rows
        with torch.no_grad():
            self.up_proj.bias.normal_()


class BiasedNormLayerMixin:
    """Patch a layer's norms to carry a bias (fold must skip, not break)."""


class TestReviewEdgeCases:
    def _model_with(self, attn_cls=None, mlp_cls=None, **kwargs):
        torch.manual_seed(11)
        model = TinyModel(**kwargs)
        if attn_cls is not None:
            for layer in model.layers:
                layer.self_attn = attn_cls(D)
        if mlp_cls is not None:
            for layer in model.layers:
                layer.mlp = mlp_cls(D, I)
        return model

    def test_gqa_v_o_fold_exact(self):
        model = self._model_with(attn_cls=lambda d: TinyGQAAttention(d, heads=2, kv_heads=1))
        before = _forward_logits(model).clone()
        PreSINQRotation(PreSINQConfig(normalize_outproj=True)).apply_to_model(model, data_type="int", use_tqdm=False)
        assert _rel_err(_forward_logits(model), before) < 1e-5

    def test_mla_extra_consumer_exact(self):
        _assert_exact(self._model_with(attn_cls=TinyMLAAttention))

    def test_unknown_consumer_skips_safely(self):
        model = self._model_with(attn_cls=TinyWeirdAttention)
        norm_w = model.layers[0].input_layernorm.weight.detach().clone()
        _assert_exact(model)  # function preserved (fold skipped)
        assert torch.equal(model.layers[0].input_layernorm.weight.detach(), norm_w)

    def test_shared_expert_moe_exact(self):
        torch.manual_seed(11)
        model = TinyModel()
        for layer in model.layers:
            layer.mlp = TinySharedExpertMoE(D, I)
        _assert_exact(model)

    def test_fused_gate_up_skips_safely(self):
        model = self._model_with(mlp_cls=TinyFusedMLP)
        _assert_exact(model)

    def test_up_proj_bias_scaled(self):
        model = self._model_with(mlp_cls=TinyBiasedMLP)
        _assert_exact(model)

    def test_norm_with_bias_skips(self):
        model = self._model_with()
        for layer in model.layers:
            layer.input_layernorm.bias = nn.Parameter(torch.zeros(D))
            layer.post_attention_layernorm.bias = nn.Parameter(torch.zeros(D))
        _assert_exact(model)  # untouched -> trivially exact


class TestShadowArithmetic:
    def test_cumulative_scales_single_writeback(self):
        """Multi-pass folds == one fp64 product applied once (single dtype rounding)."""
        import auto_round.algorithms.transforms.presinq.apply as apply_mod

        torch.manual_seed(5)
        w = nn.Parameter(torch.randn(50, 40, dtype=torch.float32))
        rot = apply_mod.PreSINQRotation(PreSINQConfig())
        t1 = torch.rand(40).double() + 0.5
        t2 = torch.rand(50).double() + 0.5  # row scale
        t3 = torch.rand(40).double() + 0.5
        rot._apply_col(w, t1)
        rot._apply_row(w, t2)
        rot._apply_col(w, t3)
        expected = (w.data.double() * (t1 * t3).view(1, -1) * t2.view(-1, 1)).to(w.dtype)
        rot._finalize()
        assert torch.equal(w.data, expected)
        rot._states.clear()

    def test_effective_reflects_cumulative_scales(self):
        import auto_round.algorithms.transforms.presinq.apply as apply_mod

        torch.manual_seed(6)
        w = nn.Parameter(torch.randn(12, 10))
        rot = apply_mod.PreSINQRotation(PreSINQConfig())
        c = torch.rand(10).double() + 0.5
        r = torch.rand(12).double() + 0.5
        rot._apply_col(w, c)
        rot._apply_row(w, r)
        expected = (w.data.double() * c.view(1, -1) * r.view(-1, 1)).to(torch.float32)
        eff = rot._effective(w)
        assert torch.allclose(eff, expected, atol=1e-6)
        rot._states.clear()


class TestLayerwise:
    def test_layerwise_protocol_matches_full_model(self):
        """Per-layer rotate_layer(n_repeat local passes) == full-model apply_to_model.

        Valid because every fold is layer-local; also proves exactness of the
        simulated composer protocol (prepare -> rotate per layer -> finalize).
        """
        torch.manual_seed(21)
        full = TinyModel(linear_attn=True, moe=True)
        torch.manual_seed(21)
        lw = TinyModel(linear_attn=True, moe=True)
        before = _forward_logits(lw).clone()

        rot = PreSINQRotation(PreSINQConfig())
        rot.prepare_layerwise(lw, data_type="int")
        for i, layer in enumerate(lw.layers):
            rot.rotate_layer(layer, i)
        rot.finalize_layerwise(lw)

        PreSINQRotation(PreSINQConfig()).apply_to_model(full, data_type="int", use_tqdm=False)

        for (n1, p1), (n2, p2) in zip(full.named_parameters(), lw.named_parameters()):
            assert n1 == n2 and torch.allclose(p1.detach(), p2.detach(), atol=1e-6), f"mismatch at {n1}"
        assert _rel_err(_forward_logits(lw), before) < 1e-5

    def test_rotate_layer_idempotent_safe(self):
        torch.manual_seed(22)
        m = TinyModel()
        before = _forward_logits(m).clone()
        rot = PreSINQRotation(PreSINQConfig())
        rot.prepare_layerwise(m, data_type="int")
        for i, layer in enumerate(m.layers):
            rot.rotate_layer(layer, i)
            rot.rotate_layer(layer, i)  # second call must stay function-preserving
        assert _rel_err(_forward_logits(m), before) < 1e-5


class TestDevicePolicy:
    def test_cpu_inputs_prefer_cuda_when_available(self, monkeypatch):
        from auto_round.algorithms.transforms.presinq import sinkhorn as sk

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        # single-CPU-device inputs must NOT short-circuit to CPU (regression:
        # the first policy version fast-pathed cpu-resident weights to cpu)
        w = [torch.randn(4, 8)]
        assert sk._select_device(w).type == "cuda"
        # mixed devices -> cuda too
        assert sk._select_device([torch.randn(4, 8), torch.randn(4, 8).to("meta")]).type == "cuda"

    def test_cpu_used_only_without_cuda(self, monkeypatch):
        from auto_round.algorithms.transforms.presinq import sinkhorn as sk

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert sk._select_device([torch.randn(4, 8)]).type == "cpu"


class TestHy3Naming:
    """hy_v3 (HYV3) module naming: mlp.shared_mlp, mlp.router.gate, mlp.experts."""

    def _hy3_moe_mlp(self, hidden=16, inter=8, n_experts=3):
        import torch.nn as nn

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden, inter, bias=False)
                self.up_proj = nn.Linear(hidden, inter, bias=False)
                self.down_proj = nn.Linear(inter, hidden, bias=False)

        class Router(nn.Module):  # hy_v3: mlp.router.gate (nested Linear)
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(hidden, n_experts, bias=False)

        class MoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([Blk() for _ in range(n_experts)])
                self.shared_mlp = Blk()
                self.router = Router()

        return MoE()

    def test_helpers_cover_hy3_names(self):
        from auto_round.algorithms.transforms.presinq.apply import _moe_blocks
        from auto_round.algorithms.transforms.spinquant.rotation_utils import get_router_linears

        mlp = self._hy3_moe_mlp()
        blocks = _moe_blocks(mlp, mlp.experts)
        assert len(blocks) == 4  # 3 experts + shared_mlp
        routers = get_router_linears(mlp)
        assert routers == [mlp.router.gate]

    def test_hy3_moe_fold_is_exact(self):
        """Fold on an hy_v3-shaped MoE layer preserves its function exactly."""
        import torch.nn as nn

        torch.manual_seed(0)

        class Layer(nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.post_attention_layernorm = StdRMSNorm(16)
                self.mlp = mlp

        layer = Layer(self._hy3_moe_mlp())
        rot = PreSINQRotation(PreSINQConfig(n_repeat=1))
        h = torch.randn(5, 16)  # one fixed input for both passes
        before = self._forward(layer, h)
        rot.rotate_layer(layer, 0)
        after = self._forward(layer, h)
        assert torch.allclose(before, after, rtol=1e-3, atol=1e-4)

    def _forward(self, layer, h=None):
        """MoE forward: sigmoid-router top-1 over experts + shared, biases on logits."""
        import torch.nn.functional as F

        if h is None:
            h = torch.randn(5, 16)
        x = layer.post_attention_layernorm(h)
        mlp = layer.mlp
        logits = F.linear(x, mlp.router.gate.weight) + torch.arange(3.0)  # expert_bias analogue
        w = torch.sigmoid(logits) * (logits > 0)
        w = w / w.sum(-1, keepdim=True)
        exp_out = torch.zeros(5, 16)
        for i, blk in enumerate(mlp.experts):
            exp_out += w[:, i : i + 1] * blk.down_proj(F.silu(blk.gate_proj(x)) * blk.up_proj(x))
        sh = mlp.shared_mlp
        return exp_out + sh.down_proj(F.silu(sh.gate_proj(x)) * sh.up_proj(x))

    def test_expert_fold_fanout_matches_serial(self):
        """Parallel expert folds must reproduce the serial loop exactly.

        Exercises the ThreadPoolExecutor path by forcing CPU worker devices
        (no CUDA needed locally); the block folds are device-independent.
        """
        import copy

        torch.manual_seed(1)
        layer_a = self._hy3_moe_mlp(n_experts=5)
        layer_b = copy.deepcopy(layer_a)

        rot_serial = PreSINQRotation(PreSINQConfig(n_repeat=1))
        rot_serial._fold_moe(layer_a, layer_a.experts, None)

        rot_par = PreSINQRotation(PreSINQConfig(n_repeat=1))
        rot_par._fold_devices = lambda n: ["cpu", "cpu"]  # force the fan-out path
        rot_par._fold_moe(layer_b, layer_b.experts, None)

        for (na, wa), (nb, wb) in zip(layer_a.named_parameters(), layer_b.named_parameters()):
            assert na == nb
            assert torch.allclose(wa, wb, atol=1e-12), na
        assert abs(rot_serial._stats_log_t - rot_par._stats_log_t) < 1e-6
        assert rot_serial._stats_n == rot_par._stats_n


class TestLazyMoEMaterialization:
    """Weight transforms must run after lazy-MoE materialization."""

    def test_apply_model_transforms_materializes_lazy_moe(self):
        """Fused-MoE replacement modules hold expert weights on the meta device
        until materialize_weights() runs. apply_model_transforms runs earlier and
        reads/writes every weight, so it must materialize the model first; a meta
        expert weight otherwise crashes the transform pass."""
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase

        class LazyExpert(ReplacementModuleBase):
            def __init__(self, original):
                super().__init__(original)
                dim = original.gate_proj.weight.shape[0]
                self.gate_proj = nn.Linear(dim, dim, bias=False, device="meta")
                self.up_proj = nn.Linear(dim, dim, bias=False, device="meta")
                self.down_proj = nn.Linear(dim, dim, bias=False, device="meta")

            @classmethod
            def original_module_class(cls):
                return "TinyMLP"

            @classmethod
            def from_original(cls, original, config):
                return cls(original)

            def _materialize_weights(self):
                src = self._get_original_module()
                self.gate_proj.weight = nn.Parameter(src.gate_proj.weight.detach().clone())
                self.up_proj.weight = nn.Parameter(src.up_proj.weight.detach().clone())
                self.down_proj.weight = nn.Parameter(src.down_proj.weight.detach().clone())

        model = TinyModel(moe=True)
        moe_layer = model.layers[0]
        assert hasattr(moe_layer.mlp, "experts"), "fixture must place a MoE MLP in layer 0"
        originals = list(moe_layer.mlp.experts)
        moe_layer.mlp.experts = nn.ModuleList([LazyExpert(e) for e in originals])

        composer = AlgorithmComposer([PreSINQConfig(n_iter=1, n_repeat=1), RTNConfig()])
        out = composer.apply_model_transforms(model)

        for expert in out.layers[0].mlp.experts:
            for param in expert.parameters():
                assert param.device.type != "meta"

    def test_layerwise_mode_does_not_materialize_lazy_moe(self):
        """Layer-wise rotation must not materialize the whole model: streamed
        models may be intentionally larger than memory, and transforms apply
        per block after the block loop materializes it."""
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase

        class LazyExpert(ReplacementModuleBase):
            def __init__(self, original):
                super().__init__(original)
                dim = original.gate_proj.weight.shape[0]
                self.gate_proj = nn.Linear(dim, dim, bias=False, device="meta")

            @classmethod
            def original_module_class(cls):
                return "TinyMLPLazy"

            @classmethod
            def from_original(cls, original, config):
                return cls(original)

            def _materialize_weights(self):
                raise AssertionError("layer-wise mode must not materialize expert weights")

        model = TinyModel(moe=True)
        moe_layer = model.layers[0]
        originals = list(moe_layer.mlp.experts)
        moe_layer.mlp.experts = nn.ModuleList([LazyExpert(e) for e in originals])

        composer = AlgorithmComposer([PreSINQConfig(n_iter=1, n_repeat=1), RTNConfig()])
        composer._layerwise_rotation = True
        composer.apply_model_transforms(model)  # must not raise / not materialize
