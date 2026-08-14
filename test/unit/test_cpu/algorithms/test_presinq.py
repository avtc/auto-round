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
    PreSINQRotation(PreSINQConfig()).apply_to_model(model, data_type="int")
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
        PreSINQRotation(PreSINQConfig(n_repeat=3)).apply_to_model(model, data_type="int")
        assert _rel_err(_forward_logits(model), before) < 1e-5

    def test_normalize_outproj_mha(self):
        torch.manual_seed(7)
        model = TinyModel()
        before = _forward_logits(model).clone()
        PreSINQRotation(PreSINQConfig(normalize_outproj=True)).apply_to_model(model, data_type="int")
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

    def test_spinquant_mutual_exclusion(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

        with pytest.raises(ValueError, match="[Pp]re-?SINQ.*SpinQuant"):
            AlgorithmComposer([PreSINQConfig(), SpinQuantConfig(r1=True), RTNConfig()])


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
        PreSINQRotation(PreSINQConfig(normalize_outproj=True)).apply_to_model(model, data_type="int")
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
