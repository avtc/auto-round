# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unpinned MTP/nextn tensors pass through at the source-exact float type in
GGUF exports (parity with the non-GGUF paths); pins still quantize as pinned."""

import torch
from gguf import GGMLQuantizationType

from auto_round.export.export_to_gguf.convert import get_qtype_by_layer_config

Q4 = GGMLQuantizationType.Q4_0
BF16 = GGMLQuantizationType.BF16
F32 = GGMLQuantizationType.F32
F16 = GGMLQuantizationType.F16


class TestNextnPassthrough:
    def test_remapped_nextn_layer_passes_through(self):
        # predictor layers sit at index >= the base block count after remap
        got = get_qtype_by_layer_config(
            {},
            "model.layers.80.self_attn.q_proj.weight",
            Q4,
            explicit_only=True,
            source_dtype=torch.bfloat16,
            base_block_count=80,
        )
        assert got == BF16

    def test_native_mtp_prefix_passes_through_without_config(self):
        got = get_qtype_by_layer_config({}, "mtp.fc.weight", Q4, explicit_only=True, source_dtype=torch.float16)
        assert got == F16

    def test_source_fp32_maps_to_f32(self):
        got = get_qtype_by_layer_config({}, "nextn.proj.weight", Q4, explicit_only=True, source_dtype=torch.float32)
        assert got == F32

    def test_body_layer_keeps_default(self):
        got = get_qtype_by_layer_config(
            {},
            "model.layers.3.mlp.gate_proj.weight",
            Q4,
            explicit_only=True,
            source_dtype=torch.bfloat16,
            base_block_count=80,
        )
        assert got is None  # explicit_only: no pin, not a predictor tensor

    def test_pin_still_quantizes(self):
        cfg = {"model.layers.80.mlp.experts.0.gate_proj": {"bits": 8, "sym": True}}
        got = get_qtype_by_layer_config(
            cfg,
            "model.layers.80.mlp.experts.0.gate_proj.weight",
            Q4,
            explicit_only=True,
            source_dtype=torch.bfloat16,
            base_block_count=80,
        )
        assert got == GGMLQuantizationType.Q8_0

    def test_float_pin_unchanged(self):
        cfg = {"model.layers.80.fc": {"bits": 16}}
        got = get_qtype_by_layer_config(
            cfg,
            "model.layers.80.fc.weight",
            Q4,
            explicit_only=True,
            source_dtype=torch.bfloat16,
            base_block_count=80,
        )
        assert got == BF16
