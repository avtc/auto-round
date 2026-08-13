"""Architecture-generality tests for the SpinQuant/QuaRot rotation pipeline.

Covers the fixes made after the MoE architecture evaluation:
  - R2: v_proj bias rotation, paired v/o check, output-gated attention skip
  - Offline R1: explicit refusal on unsupported architectures (hybrid
    linear-attention, MLA, fused experts) instead of silently breaking
    equivalence
  - Architecture-generic MLP attribute aliases (feed_forward /
    block_sparse_moe), fused gate+up projections (input_linear /
    gate_up_proj), layer-level shared_mlp
  - VL-safe text-backbone location (language_model nesting)

All models are tiny and randomly initialised — no downloads needed.
"""

import sys

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    get_mlp_module,
    get_proj,
    iter_layer_mlp_blocks,
)
from auto_round.modeling.fused_moe.replace_modules import apply_replacements, materialize_model_


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def _rel_err(out, ref):
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _assert_equivalent(model, input_ids, ref, preprocess_cfg):
    SpinQuantPreprocessor(model, preprocess_cfg).preprocess()
    out = _logits(model, input_ids)
    assert torch.isfinite(out).all()
    rel = _rel_err(out, ref)
    assert rel < 1e-3, f"relative logits error too large: {rel:.2e}"


def _tiny_qwen3_dense(seed=0):
    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval()


def _tiny_granitemoe(seed=0):
    from transformers.models.granitemoe import GraniteMoeConfig, GraniteMoeForCausalLM

    torch.manual_seed(seed)
    cfg = GraniteMoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return GraniteMoeForCausalLM(cfg).eval()


def _tiny_granitemoeshared(seed=0):
    from transformers.models.granitemoeshared import GraniteMoeSharedConfig, GraniteMoeSharedForCausalLM

    torch.manual_seed(seed)
    cfg = GraniteMoeSharedConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        shared_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return GraniteMoeSharedForCausalLM(cfg).eval()


def _tiny_llama4_text(seed=0):
    from transformers.models.llama4 import Llama4ForCausalLM, Llama4TextConfig

    torch.manual_seed(seed)
    cfg = Llama4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        intermediate_size_mlp=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=1,
        moe_layers=[0, 1],
    )
    return Llama4ForCausalLM(cfg).eval()


def _tiny_qwen3_next(seed=0):
    from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"],
    )
    return Qwen3NextForCausalLM(cfg).eval()


def _tiny_deepseek_v3(seed=0):
    from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

    torch.manual_seed(seed)
    cfg = DeepseekV3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        kv_lora_rank=32,
        q_lora_rank=None,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        max_position_embeddings=128,
    )
    return DeepseekV3ForCausalLM(cfg).eval()


# ═══════════════════════════════════════════════════════════════════════════════
# R2 fixes
# ═══════════════════════════════════════════════════════════════════════════════


class TestR2Fixes:
    def test_r2_rotates_v_proj_bias(self):
        """v_proj bias must be rotated per head together with the weight."""
        model = _tiny_qwen3_dense()
        # Simulate a trained checkpoint: non-zero v_proj bias.
        with torch.no_grad():
            for layer in model.model.layers:
                v = layer.self_attn.v_proj
                if v.bias is None:
                    v.bias = nn.Parameter(torch.zeros(v.weight.shape[0], dtype=v.weight.dtype))
                v.bias.normal_(0, 0.02)

        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=False, r2=True, r3=False, r4=False)
        _assert_equivalent(model, input_ids, ref, cfg)

    def test_r2_skipped_for_mla(self):
        """MLA attention (deepseek_v3) has o_proj but no v_proj: R2 must skip
        the whole pair, not rotate o_proj one-sidedly."""
        model = _tiny_deepseek_v3()
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=False, r2=True, r3=False, r4=False)
        _assert_equivalent(model, input_ids, ref, cfg)

    def test_r2_skipped_for_output_gated_attention(self):
        """qwen3_next full attention is output-gated; R2 must be skipped."""
        model = _tiny_qwen3_next()
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=False, r2=True, r3=False, r4=False)
        _assert_equivalent(model, input_ids, ref, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Offline R1 explicit refusal
# ═══════════════════════════════════════════════════════════════════════════════


class TestOfflineR1Refusal:
    def test_refusal_on_hybrid_linear_attention(self):
        """qwen3_next has GatedDeltaNet layers without self_attn → refuse."""
        model = _tiny_qwen3_next()
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        with pytest.raises(ValueError, match="[Oo]ffline R1"):
            SpinQuantPreprocessor(model, cfg).preprocess()

    def test_refusal_on_fused_experts(self):
        """llama4 experts stay fused (no linear_loop support) → refuse offline."""
        model = _tiny_llama4_text()
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        with pytest.raises(ValueError, match="[Oo]ffline R1"):
            SpinQuantPreprocessor(model, cfg).preprocess()

    def test_refusal_on_mla(self):
        """deepseek_v3 MLA attention lacks q/k/v_proj → refuse offline R1."""
        model = _tiny_deepseek_v3()
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        with pytest.raises(ValueError, match="[Oo]ffline R1"):
            SpinQuantPreprocessor(model, cfg).preprocess()

    def test_online_r1_r4_work_on_mla(self):
        """Online R1 + R4 remain valid for deepseek_v3 (R2 auto-skipped)."""
        model = _tiny_deepseek_v3()
        from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

        prepare_model_for_moe_quantization(model)
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=True)
        _assert_equivalent(model, input_ids, ref, cfg)

    def test_online_r1_still_works_on_hybrid(self):
        """Online R1 + R4 remain valid for qwen3_next (residual stream untouched)."""
        model = _tiny_qwen3_next()
        from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

        prepare_model_for_moe_quantization(model)
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        # R2 auto-skipped via output-gating detection.
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=True)
        _assert_equivalent(model, input_ids, ref, cfg)

    def test_online_r1_covers_linear_attn_layer_experts(self):
        """Hybrid models: experts of linear-attention layers (no self_attn)
        must also be rotated and hooked by online R1 — attention and MLP
        rotation are independent."""
        model = _tiny_qwen3_next()
        from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

        prepare_model_for_moe_quantization(model)

        # layer 0 is a linear_attention layer (no self_attn)
        linear_layer = model.model.layers[0]
        assert not hasattr(linear_layer, "self_attn")
        expert0 = getattr(linear_layer.mlp.experts, "0")
        w_before = expert0.gate_proj.weight.data.clone()

        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=True)
        SpinQuantPreprocessor(model, cfg).preprocess()

        assert not torch.equal(expert0.gate_proj.weight.data, w_before), "expert gate_proj was not rotated"

        def n_hooks(mod):
            return sum(1 for h in mod._forward_pre_hooks.values() if getattr(h, "_spinquant_hook", False))

        assert n_hooks(expert0.gate_proj) == 1
        assert n_hooks(expert0.up_proj) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture-generic MLP aliases
# ═══════════════════════════════════════════════════════════════════════════════


class TestArchitectureAliases:
    def test_granitemoe_block_sparse_moe(self):
        """granitemoe: block_sparse_moe naming + input_linear/output_linear experts."""
        model = _tiny_granitemoe()
        model = apply_replacements(model)
        materialize_model_(model)

        layer = model.model.layers[0]
        assert get_mlp_module(layer) is layer.block_sparse_moe
        kinds = [k for _, k in iter_layer_mlp_blocks(layer)]
        assert kinds.count("expert") == 4

        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)

        for online in (True, False):
            model_i = _tiny_granitemoe()
            model_i = apply_replacements(model_i)
            materialize_model_(model_i)
            ref_i = _logits(model_i, input_ids)
            cfg = SpinQuantConfig(
                r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=online
            )
            _assert_equivalent(model_i, input_ids, ref_i, cfg)

    def test_granitemoeshared_layer_level_shared_mlp(self):
        """granitemoeshared: shared_mlp hangs on the layer; must be covered too."""
        model = _tiny_granitemoeshared()
        model = apply_replacements(model)
        materialize_model_(model)

        layer = model.model.layers[0]
        kinds = [k for _, k in iter_layer_mlp_blocks(layer)]
        assert kinds.count("expert") == 4
        assert "shared" in kinds  # layer.shared_mlp found

        shared = layer.shared_mlp
        assert get_proj(shared, "gate") is shared.input_linear
        assert get_proj(shared, "up") is shared.input_linear  # fused gate+up
        assert get_proj(shared, "down") is shared.output_linear

        input_ids = torch.randint(0, 256, (2, 16))
        for online in (True, False):
            model_i = _tiny_granitemoeshared()
            model_i = apply_replacements(model_i)
            materialize_model_(model_i)
            ref_i = _logits(model_i, input_ids)
            cfg = SpinQuantConfig(
                r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=online
            )
            _assert_equivalent(model_i, input_ids, ref_i, cfg)

    def test_llama4_feed_forward_online(self):
        """llama4: feed_forward naming; shared expert + router covered online;
        fused experts skipped with a warning (partial but equivalent)."""
        model = _tiny_llama4_text()
        layer = model.model.layers[0]
        assert get_mlp_module(layer) is layer.feed_forward

        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=True)
        _assert_equivalent(model, input_ids, ref, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# VL-safe text-backbone location
# ═══════════════════════════════════════════════════════════════════════════════


class TestVLTextBackbone:
    def _tiny_qwen3_vl_moe(self, seed=0):
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

        torch.manual_seed(seed)
        cfg = Qwen3VLMoeConfig(
            text_config=dict(
                vocab_size=256,
                hidden_size=128,
                intermediate_size=256,
                moe_intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=64,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=128,
            ),
            vision_config=dict(
                depth=2,
                hidden_size=64,
                intermediate_size=128,
                num_heads=2,
                out_hidden_size=128,
                patch_size=16,
                spatial_merge_size=1,
            ),
        )
        return Qwen3VLMoeForConditionalGeneration(cfg).eval()

    def test_vl_offline_and_online_equivalence(self):
        """Rotation must target the TEXT backbone of a VL model: layers,
        embeddings and arch info all live under model.language_model /
        config.text_config.  Both offline and online R1 must stay equivalent.
        """
        from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

        input_ids = torch.randint(0, 256, (2, 16))
        for online in (True, False):
            model = self._tiny_qwen3_vl_moe()
            prepare_model_for_moe_quantization(model)
            ref = _logits(model, input_ids)
            cfg = SpinQuantConfig(
                r1=True, r2=True, r3=False, r4=True, rotation_size=128, online_r1_rotation=online
            )
            _assert_equivalent(model, input_ids, ref, cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
