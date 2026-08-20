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
"""Tests for the offline-fused block-Hadamard rotation.

Covers: rotation-matrix properties, function-preserving exactness on a mini
hybrid model (dense attention + GatedDeltaNet-style linear attention + MoE,
mixing both RMSNorm weight conventions), opt-in config plumbing, and the
rotation-family mutual exclusion in the composer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.block_hadamard import (
    BlockHadamardConfig,
    BlockHadamardRotation,
    build_block_rotation,
)
from auto_round.algorithms.transforms.block_hadamard.apply import BlockHadamardRotation as _BHR

HIDDEN = 128
BLOCK = 64
VOCAB = 64
N_EXPERTS = 2


class MiniRMSNorm(nn.Module):
    """Standard convention: output = pure_norm(x) * weight (weight init 1)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        v = x.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return v.to(x.dtype) * self.weight


class MiniRMSNormOnePlus(nn.Module):
    """Qwen3.5-style convention: output = pure_norm(x) * (1 + weight)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        v = x.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return v.to(x.dtype) * (1.0 + self.weight)


class MiniAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        a = torch.softmax((q * k).mean(-1, keepdim=True) / (q.shape[-1] ** 0.5), dim=1)
        return self.o_proj(a * v)


class MockLinearAttn(nn.Module):
    """GatedDeltaNet-shaped module: same projection names, arbitrary nonlinear mix.

    Exactness only requires every projection to be functionally equivalent -
    whatever nonlinear mixing happens in the projected space commutes with the
    weight-side fusion.
    """

    def __init__(self, dim):
        super().__init__()
        self.in_proj_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.in_proj_z = nn.Linear(dim, dim, bias=False)
        self.in_proj_a = nn.Linear(dim, dim, bias=False)
        self.in_proj_b = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.in_proj_qkv(x).chunk(3, dim=-1)
        z, a, b = self.in_proj_z(x), self.in_proj_a(x), self.in_proj_b(x)
        h = torch.sigmoid(q) * torch.tanh(k) * v * torch.sigmoid(z) + a * b
        return self.out_proj(h)


class MiniMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniMoE(nn.Module):
    def __init__(self, dim, n_experts):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)  # router
        self.experts = nn.ModuleList([MiniMLP(dim) for _ in range(n_experts)])

    def forward(self, x):
        probs = torch.softmax(self.gate(x), dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out = out + probs[..., i:i + 1] * expert(x)
        return out


class MiniLayer(nn.Module):
    def __init__(self, dim, attn_kind, norm_cls):
        super().__init__()
        self.input_layernorm = norm_cls(dim)
        if attn_kind == "dense":
            self.self_attn = MiniAttn(dim)
        else:
            self.linear_attn = MockLinearAttn(dim)
        self.post_attention_layernorm = norm_cls(dim)
        self.mlp = MiniMLP(dim)

    def forward(self, x):
        attn = self.self_attn if hasattr(self, "self_attn") else self.linear_attn
        x = x + attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMoELayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_layernorm = MiniRMSNorm(dim)
        self.self_attn = MiniAttn(dim)
        self.post_attention_layernorm = MiniRMSNorm(dim)
        self.mlp = MiniMoE(dim, N_EXPERTS)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.model.layers = nn.ModuleList(
            [
                MiniLayer(HIDDEN, "dense", MiniRMSNorm),
                MiniLayer(HIDDEN, "linear", MiniRMSNormOnePlus),
                MiniMoELayer(HIDDEN),
            ]
        )
        self.model.norm = MiniRMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(self.model.norm(h))


def _rand_init(model, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)


class TestBlockRotationMatrix:
    def test_auto_resolution(self):
        from auto_round.algorithms.transforms.block_hadamard import resolve_auto_block_size

        # largest pow2 divisor STRICTLY below full width (Full is degenerate:
        # no blocks left for MassDiff to balance, quality collapses)
        assert resolve_auto_block_size(5120) == 1024
        assert resolve_auto_block_size(4096) == 2048  # pow2 hidden -> step down once
        assert resolve_auto_block_size(6144) == 2048
        assert resolve_auto_block_size(128) == 64
        # odd x 2 hidden with no valid sub-full pow2 divisor
        with pytest.raises(ValueError, match="auto block_size"):
            resolve_auto_block_size(2)

    def test_default_is_auto(self):
        assert BlockHadamardConfig().block_size == 0

    def test_orthogonal_symmetric_deterministic(self):
        r1 = build_block_rotation(BLOCK, seed=7, randomized=True)
        r2 = build_block_rotation(BLOCK, seed=7, randomized=True)
        assert torch.allclose(r1, r2)
        assert torch.allclose(r1 @ r1.T, torch.eye(BLOCK, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(r1, r1.T, atol=1e-10)

    def test_plain_hadamard(self):
        r = build_block_rotation(BLOCK, randomized=False)
        assert torch.allclose(r @ r.T, torch.eye(BLOCK, dtype=torch.float64), atol=1e-10)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            build_block_rotation(3)


class TestFusionExactness:
    def _run(self, **cfg_overrides):
        torch.manual_seed(0)
        model = MiniModel()
        _rand_init(model)
        model.eval()
        ids = torch.randint(0, VOCAB, (2, 16))
        with torch.no_grad():
            before = model(ids)
        cfg = BlockHadamardConfig(**{"block_size": BLOCK, **cfg_overrides})
        rot = _BHR(cfg)
        rot.apply_to_model(model)
        with torch.no_grad():
            after = model(ids)
        torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)

    def test_default_randomized(self):
        self._run()

    def test_plain(self):
        self._run(randomized=False)

    def test_smaller_block(self):
        self._run(block_size=32)

    def test_gamma_folded_to_neutral(self):
        torch.manual_seed(0)
        model = MiniModel()
        _rand_init(model)
        rot = _BHR(BlockHadamardConfig(block_size=BLOCK))
        rot.apply_to_model(model)
        lin_layer = model.model.layers[1]
        assert torch.allclose(lin_layer.input_layernorm.weight.data, torch.zeros(HIDDEN), atol=1e-7)
        dense_layer = model.model.layers[0]
        assert torch.allclose(dense_layer.input_layernorm.weight.data, torch.ones(HIDDEN), atol=1e-7)


class TiedModel(MiniModel):
    def __init__(self):
        super().__init__()
        self.lm_head.weight = self.model.embed_tokens.weight  # [VOCAB, HIDDEN] tie


class TestValidation:
    def test_tied_head_refused(self):
        model = TiedModel()
        with pytest.raises(ValueError, match="tied"):
            _BHR(BlockHadamardConfig(block_size=BLOCK)).apply_to_model(model)

    def test_indivisible_hidden_refused(self):
        model = MiniModel()
        with pytest.raises(ValueError, match="divisible"):
            _BHR(BlockHadamardConfig(block_size=96)).apply_to_model(model)

    def test_no_layers_refused(self):
        with pytest.raises(ValueError, match="decoder layers"):
            _BHR(BlockHadamardConfig(block_size=BLOCK)).apply_to_model(nn.Linear(4, 4))


class TestPlumbing:
    def test_registry_key(self):
        from auto_round.algorithms.transforms import normalize_rotation_config

        assert "block_hadamard" in BaseRotation._REGISTRY
        cfg = normalize_rotation_config("block_hadamard")
        assert isinstance(cfg, BlockHadamardConfig)
        assert cfg.block_size == 0
        cfg2 = normalize_rotation_config({"algorithm": "block_hadamard", "block_size": 32})
        assert isinstance(cfg2, BlockHadamardConfig) and cfg2.block_size == 32
        inst = BaseRotation.from_config(cfg)
        assert isinstance(inst, BlockHadamardRotation)

    def test_apply_model_transforms_materializes_lazy_moe(self):
        """Weight transforms must run after lazy-MoE materialization.

        Fused-MoE replacement modules hold expert weights on the meta device
        until materialize_weights() runs (first block touch in the quantize
        loop). apply_model_transforms runs earlier and reads/writes every
        weight, so it must materialize the model first; a meta expert weight
        otherwise crashes the transform pass.
        """
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
                return "MiniMLP"

            @classmethod
            def from_original(cls, original, config):
                return cls(original)

            def _materialize_weights(self):
                src = self._get_original_module()
                self.gate_proj.weight = nn.Parameter(src.gate_proj.weight.detach().clone())
                self.up_proj.weight = nn.Parameter(src.up_proj.weight.detach().clone())
                self.down_proj.weight = nn.Parameter(src.down_proj.weight.detach().clone())

        model = MiniModel()
        moe_layer = model.model.layers[2]
        originals = list(moe_layer.mlp.experts)
        moe_layer.mlp.experts = nn.ModuleList([LazyExpert(e) for e in originals])

        composer = AlgorithmComposer([BlockHadamardConfig(), RTNConfig()])
        out = composer.apply_model_transforms(model)

        for expert in out.model.layers[2].mlp.experts:
            for param in expert.parameters():
                assert param.device.type != "meta"

    def test_layerwise_mode_skips_full_materialization(self):
        """Layer-wise rotation must not force full-model materialization.

        Layer-wise mode exists for models intentionally kept beyond memory
        residency; transforms run per block after the block loop materializes
        it. Materializing the whole model here would defeat that mode.
        """
        from unittest import mock as _mock

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.modeling import fused_moe

        model = MiniModel()
        composer = AlgorithmComposer([BlockHadamardConfig(), RTNConfig()])
        composer._layerwise_rotation = True
        with _mock.patch.object(fused_moe, "materialize_model_") as mat:
            composer.apply_model_transforms(model)
        mat.assert_not_called()

    def test_composer_accepts_and_excludes(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.presinq import PreSINQConfig
        from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantConfig

        composer = AlgorithmComposer([BlockHadamardConfig(), RTNConfig()])
        assert [type(c).__name__ for c in composer._rotation_configs] == ["BlockHadamardConfig"]

        composer = AlgorithmComposer([PreSINQConfig(), BlockHadamardConfig(), RTNConfig()])
        assert len(composer._rotation_configs) == 2

        with pytest.raises(ValueError, match="mutually exclusive"):
            AlgorithmComposer([BlockHadamardConfig(), SpinQuantConfig(), RTNConfig()])
