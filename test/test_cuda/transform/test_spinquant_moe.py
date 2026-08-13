"""MoE tests for the SpinQuant/QuaRot rotation implementation.

Covers the MoE gaps fixed in the SpinQuant pipeline:
  - R1 (online & offline) now rotates expert gate/up/down projections and
    (offline only) router weights, keeping the model mathematically equivalent
  - R4 hooks are registered exactly on the down_proj modules whose weights
    were fused (hook set == fused set), including MoE experts
  - RMSNorm gamma fusion covers experts and router linears

Uses a tiny randomly-initialised Qwen3Moe model (no downloads needed).

Note: ``test/conftest.py`` contains a workaround that stubs torchvision
when the installed torchvision is incompatible with torch (it otherwise
breaks every transformers model import in this environment).
"""

import sys

import pytest
import torch
import torch.nn as nn

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    get_proj,
    get_router_linears,
    iter_mlp_blocks,
)
from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

NUM_EXPERTS = 4


def _tiny_qwen3_moe(seed: int = 0) -> Qwen3MoeForCausalLM:
    """Build a tiny random-init Qwen3Moe model (fp32, CPU-friendly)."""
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return Qwen3MoeForCausalLM(cfg).eval()


def _spinquant_hooks(module: nn.Module) -> list:
    return [h for h in module._forward_pre_hooks.values() if getattr(h, "_spinquant_hook", False)]


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


# ═══════════════════════════════════════════════════════════════════════════════
# Traversal helper tests (unfused expert layout)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoeTraversal:
    def test_iter_mlp_blocks_finds_experts(self):
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)
        mlp = model.model.layers[0].mlp

        blocks = list(iter_mlp_blocks(mlp))
        kinds = [k for _, k in blocks]
        assert kinds.count("expert") == NUM_EXPERTS
        assert "dense" not in kinds  # Qwen3Moe has no dense MLP in sparse layers

        for block, kind in blocks:
            assert get_proj(block, "gate") is not None
            assert get_proj(block, "up") is not None
            assert get_proj(block, "down") is not None

    def test_router_detected(self):
        model = _tiny_qwen3_moe()
        mlp = model.model.layers[0].mlp
        routers = get_router_linears(mlp)
        assert len(routers) == 1
        assert routers[0] is mlp.gate
        assert routers[0].weight.shape == (NUM_EXPERTS, 128)


# ═══════════════════════════════════════════════════════════════════════════════
# Equivalence tests: rotated model must match the original model's logits
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoeRotationEquivalence:
    def test_online_r1_r2_r4_equivalence(self):
        """Online R1 + offline R2 + R4 (hook+fuse) on unfused MoE experts."""
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)

        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)

        # rotation_size=128 exercises block rotation for R4 (expert in=256)
        cfg = SpinQuantConfig(
            r1=True, r2=True, r3=False, r4=True,
            rotation_size=128, online_r1_rotation=True,
        )
        SpinQuantPreprocessor(model, cfg).preprocess()

        out = _logits(model, input_ids)
        assert torch.isfinite(out).all()
        rel = (out - ref).norm() / ref.norm().clamp_min(1e-12)
        assert rel < 1e-3, f"relative logits error too large: {rel:.2e}"

    def test_offline_r1_r2_r4_equivalence(self):
        """Offline R1 fuses RMSNorm gamma + rotates router & experts losslessly."""
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)

        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)

        cfg = SpinQuantConfig(
            r1=True, r2=True, r3=False, r4=True,
            rotation_size=128, online_r1_rotation=False,
        )
        SpinQuantPreprocessor(model, cfg).preprocess()

        out = _logits(model, input_ids)
        assert torch.isfinite(out).all()
        rel = (out - ref).norm() / ref.norm().clamp_min(1e-12)
        assert rel < 1e-3, f"relative logits error too large: {rel:.2e}"


# ═══════════════════════════════════════════════════════════════════════════════
# Coverage & hook/fuse consistency
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoeCoverage:
    def test_online_r1_rotates_and_hooks_experts(self):
        """Every expert gate/up gets rotated weights + a matching online hook;
        the router is left completely untouched (no hook, no rotation)."""
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)
        mlp = model.model.layers[0].mlp

        expert0 = getattr(mlp.experts, "0")
        w_gate_before = expert0.gate_proj.weight.data.clone()
        w_router_before = mlp.gate.weight.data.clone()

        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=True)
        SpinQuantPreprocessor(model, cfg).preprocess()

        # Expert gate/up: weights rotated AND hook registered
        assert not torch.equal(expert0.gate_proj.weight.data, w_gate_before)
        assert len(_spinquant_hooks(expert0.gate_proj)) == 1
        assert len(_spinquant_hooks(expert0.up_proj)) == 1

        # Router: untouched
        assert torch.equal(mlp.gate.weight.data, w_router_before)
        assert len(mlp.gate._forward_pre_hooks) == 0

    def test_offline_r1_rotates_router(self):
        """Offline R1 must rotate router weights (residual stream is rotated)."""
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)
        mlp = model.model.layers[0].mlp
        w_router_before = mlp.gate.weight.data.clone()

        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        SpinQuantPreprocessor(model, cfg).preprocess()

        assert not torch.equal(mlp.gate.weight.data, w_router_before)
        assert len(mlp.gate._forward_pre_hooks) == 0  # routers never get hooks

    def test_r4_hook_fuse_consistency(self):
        """Hook set == fused set: with rotation_size=128, expert down_proj
        (in=256, divisible) gets hook+fuse; a hypothetical incompatible
        module would get neither."""
        model = _tiny_qwen3_moe()
        prepare_model_for_moe_quantization(model)

        downs_before = {}
        for name, mod in model.named_modules():
            if "down_proj" in name and isinstance(mod, nn.Linear):
                downs_before[name] = mod.weight.data.clone()
        assert len(downs_before) == 2 * NUM_EXPERTS  # 2 layers × 4 experts

        cfg = SpinQuantConfig(r1=False, r2=False, r3=False, r4=True, rotation_size=128)
        SpinQuantPreprocessor(model, cfg).preprocess()

        for name, mod in model.named_modules():
            if "down_proj" in name and isinstance(mod, nn.Linear):
                compatible = mod.in_features % 128 == 0
                hooked = len(_spinquant_hooks(mod)) == 1
                fused = not torch.equal(mod.weight.data, downs_before[name])
                assert hooked == compatible, f"{name}: hook presence mismatch"
                assert fused == compatible, f"{name}: weight fusion mismatch"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
