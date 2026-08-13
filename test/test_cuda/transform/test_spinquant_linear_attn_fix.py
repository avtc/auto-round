# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for the gated linear-attention offline-R1 fix
(``AR_ROTATION_FIX_LINEAR_ATTN``) on hybrid-attention models (qwen3_next /
qwen3_5 GatedDeltaNet layers).

The default offline-R1 path refuses hybrid linear-attention models because the
``linear_attn`` layers (no ``self_attn``) would be skipped, leaving their MLP
and linear_attn projections uncompensated while the residual stream is rotated
— breaking equivalence.  When ``AR_ROTATION_FIX_LINEAR_ATTN=1`` three things
happen together, keeping offline R1 mathematically exact:

  - the ``input_layernorm`` (1 + weight)-convention gamma is folded into the
    following projections (Qwen3Next/Qwen3_5 RMSNorms use ``output*(1+weight)``
    with ``weight`` zero-initialised; folding the delta would silently zero the
    projection);
  - ``linear_attn.in_proj_*`` absorb R1^-1 on input channels and
    ``out_proj`` absorbs R1 on output channels;
  - the layer's MLP is rotated too.

Uses a tiny DENSE hybrid model (num_experts=0) to isolate the linear-attention
handling from MoE concerns and to run on transformers >= 4.57 without the MoE
linearisation step.
"""

import sys

import pytest
import torch

from auto_round.algorithms.transforms.spinquant import SpinQuantConfig
from auto_round.algorithms.transforms.spinquant.preprocessor import SpinQuantPreprocessor

_ENV = "AR_ROTATION_FIX_LINEAR_ATTN"


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids).logits


def _rel_err(out, ref):
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _tiny_qwen3_next_dense(seed=0):
    """Tiny DENSE hybrid model: layer 0 = GatedDeltaNet, layer 1 = full attn."""
    from transformers.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=0,  # dense: isolates the linear-attention fix from MoE
        full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"],
    )
    return Qwen3NextForCausalLM(cfg).eval()


class TestLinearAttnFix:
    def test_offline_r1_refuses_when_fix_disabled(self, monkeypatch):
        """Default behaviour: offline R1 refuses hybrid linear-attention."""
        monkeypatch.delenv(_ENV, raising=False)
        model = _tiny_qwen3_next_dense()
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        with pytest.raises(ValueError, match="[Oo]ffline R1"):
            SpinQuantPreprocessor(model, cfg).preprocess()

    def test_offline_r1_equivalent_when_fix_enabled(self, monkeypatch):
        """AR_ROTATION_FIX_LINEAR_ATTN=1: offline R1 is exact on the hybrid model."""
        monkeypatch.setenv(_ENV, "1")
        model = _tiny_qwen3_next_dense()
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        SpinQuantPreprocessor(model, cfg).preprocess()
        out = _logits(model, input_ids)
        assert torch.isfinite(out).all()
        assert _rel_err(out, ref) < 1e-3

    def test_offline_r1_r2_equivalent_when_fix_enabled(self, monkeypatch):
        """R2 auto-skips on output-gated qwen3_next full attention; with the
        linear-attn fix, offline R1+R2 stays equivalent."""
        monkeypatch.setenv(_ENV, "1")
        model = _tiny_qwen3_next_dense()
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        cfg = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False, online_r1_rotation=False)
        SpinQuantPreprocessor(model, cfg).preprocess()
        out = _logits(model, input_ids)
        assert torch.isfinite(out).all()
        assert _rel_err(out, ref) < 1e-3

    def test_offline_r1_rotates_linear_attn_projections(self, monkeypatch):
        """Sanity: with the fix on, the linear_attn projections are actually
        modified (not silently skipped)."""
        monkeypatch.setenv(_ENV, "1")
        model = _tiny_qwen3_next_dense()
        linear_attn = model.model.layers[0].linear_attn
        assert not hasattr(model.model.layers[0], "self_attn")
        before = {p: getattr(linear_attn, p).weight.data.clone() for p in ("in_proj_qkvz", "in_proj_ba", "out_proj")}
        cfg = SpinQuantConfig(r1=True, r2=False, r3=False, r4=False, online_r1_rotation=False)
        SpinQuantPreprocessor(model, cfg).preprocess()
        for p, w0 in before.items():
            assert not torch.equal(getattr(linear_attn, p).weight.data, w0), f"{p} was not rotated"

    def test_no_regression_on_standard_rmsnorm_model(self, monkeypatch):
        """The (1 + weight) probe must report False for a standard RMSNorm
        (qwen3 dense), so env ON leaves standard models byte-identical."""
        from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

        from auto_round.algorithms.transforms.spinquant.rotation_utils import _is_one_plus_weight_norm

        monkeypatch.setenv(_ENV, "1")
        cfg = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            tie_word_embeddings=False,
        )
        model = Qwen3ForCausalLM(cfg).eval()
        assert _is_one_plus_weight_norm(model.model.layers[0].input_layernorm) is False
        input_ids = torch.randint(0, 256, (2, 16))
        ref = _logits(model, input_ids)
        SpinQuantPreprocessor(
            model, SpinQuantConfig(r1=True, r2=True, r3=False, r4=False, online_r1_rotation=False)
        ).preprocess()
        out = _logits(model, input_ids)
        assert torch.isfinite(out).all()
        assert _rel_err(out, ref) < 1e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
