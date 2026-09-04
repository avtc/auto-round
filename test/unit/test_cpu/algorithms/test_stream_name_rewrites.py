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
"""Checkpoint-name alias resolution must be registry-driven, never hard-coded.

Two layers, consulted in order:
  * transformers' per-family conversion registry (families transformers
    knows; e.g. checkpoints spelling ``mlp.shared_mlp`` where the modeling
    code exposes ``mlp.shared_experts``),
  * auto-round's ``register_checkpoint_name_rewrites`` fallback for model
    families absent from transformers.

Both the streaming materializer (checkpoint -> module direction) and the
export revert (module -> checkpoint direction) read the same layers, so load
and export spellings can never diverge.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from auto_round.utils.checkpoint_streamer import CheckpointStreamer

_HY_V3 = "hy_v3"  # model_type whose registry entries exercise all three renames


def _make_checkpoint(tmp_path, tensors, model_type):
    save_file(tensors, os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump({"model_type": model_type}, f)
    return str(tmp_path)


def _make_moe_block():
    import torch.nn as nn

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4, bias=False)
            self.shared_experts = nn.ModuleDict(
                {
                    "gate_proj": nn.Linear(8, 8, bias=False),
                    "up_proj": nn.Linear(8, 8, bias=False),
                    "down_proj": nn.Linear(8, 8, bias=False),
                }
            )
            self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(4))

        def forward(self, x):
            return x

    with torch.device("meta"):
        return MoE()


class TestTransformersRegistryRewrites:
    def test_registry_renames_materialize_on_miss(self, tmp_path):
        ckpt = {
            "model.layers.1.mlp.router.gate.weight": torch.randn(4, 8),
            "model.layers.1.mlp.shared_mlp.gate_proj.weight": torch.randn(8, 8),
            "model.layers.1.mlp.shared_mlp.up_proj.weight": torch.randn(8, 8),
            "model.layers.1.mlp.shared_mlp.down_proj.weight": torch.randn(8, 8),
            "model.layers.1.mlp.expert_bias": torch.randn(4),
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.randn(8, 8),
        }
        path = _make_checkpoint(tmp_path, ckpt, _HY_V3)
        streamer = CheckpointStreamer(path)
        assert streamer._model_type == _HY_V3  # noqa: SLF001 - direct check of detection
        block = _make_moe_block()

        loaded = streamer.load_module_(block, "model.layers.1.mlp")

        assert "model.layers.1.mlp.router.gate.weight" in loaded
        assert block.gate.weight.device.type == "cpu"
        assert torch.equal(block.gate.weight.data, ckpt["model.layers.1.mlp.router.gate.weight"])
        assert torch.equal(
            block.shared_experts["gate_proj"].weight.data, ckpt["model.layers.1.mlp.shared_mlp.gate_proj.weight"]
        )
        assert torch.equal(
            block.shared_experts["down_proj"].weight.data, ckpt["model.layers.1.mlp.shared_mlp.down_proj.weight"]
        )
        assert torch.equal(block.e_score_correction_bias.data, ckpt["model.layers.1.mlp.expert_bias"])
        # every parameter left the meta device
        for name, p in block.named_parameters():
            assert p.device.type != "meta", name

    def test_export_side_revert_covers_renamed_families(self):
        # The writer's generic revert must recover checkpoint spellings from
        # the transformers registry even when the model instance never ran
        # from_pretrained (meta/config-built: instance mapping empty/absent).
        from auto_round.utils.common import (
            get_reverse_checkpoint_conversion_mapping,
            revert_checkpoint_conversion_mapping,
        )

        class StubModel:
            config = type("C", (), {"model_type": _HY_V3})()
            _checkpoint_conversion_mapping = ""  # empty str on config-built instances

        reverse = get_reverse_checkpoint_conversion_mapping(StubModel())
        assert reverse, "expected a non-empty reverse mapping for the renamed family"
        assert (
            revert_checkpoint_conversion_mapping("model.layers.1.mlp.shared_experts.gate_proj.weight", reverse)
            == "model.layers.1.mlp.shared_mlp.gate_proj.weight"
        )
        assert (
            revert_checkpoint_conversion_mapping("model.layers.1.mlp.gate.weight", reverse)
            == "model.layers.1.mlp.router.gate.weight"
        )
        # names outside the renamed families pass through untouched
        assert (
            revert_checkpoint_conversion_mapping("model.layers.1.mlp.experts.3.down_proj.weight", reverse)
            == "model.layers.1.mlp.experts.3.down_proj.weight"
        )


class TestDetectionEdgeCases:
    def test_missing_config_json_falls_back_to_strict(self, tmp_path):
        ckpt = {"blk.a.weight": torch.ones(2, 2)}
        save_file(ckpt, os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})
        streamer = CheckpointStreamer(str(tmp_path))
        assert streamer._model_type is None  # noqa: SLF001 - direct check, no config.json present

    def test_prefetch_path_also_applies_rewrites(self, tmp_path):
        # prefetch staging fetches by checkpoint name; the rewrite happens at
        # module assignment, so prefetch and direct fetch must agree
        ckpt = {"model.layers.1.mlp.router.gate.weight": torch.randn(4, 8)}
        path = _make_checkpoint(tmp_path, ckpt, _HY_V3)
        streamer = CheckpointStreamer(path)
        t = streamer.fetch("model.layers.1.mlp.router.gate.weight")
        assert torch.equal(t, ckpt["model.layers.1.mlp.router.gate.weight"])
