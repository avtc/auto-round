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
"""Restoring packed-module state from output shards at export time."""

import json
import os

import torch
from compressed_tensors.quantization import QuantizationStatus

from auto_round.export.export_to_llmcompressor.export import _restore_packed_modules_from_shards


def _fake_layer():
    import torch.nn as nn

    lin = nn.Linear(8, 4, bias=False)
    lin.bits, lin.group_size, lin.sym, lin.data_type = 4, 128, True, "int"
    lin.act_bits, lin.act_sym, lin.act_data_type = 16, True, None
    return lin


def _write_shards(tmp_path, tensors):
    from safetensors.torch import save_file

    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))
    index = {"metadata": {"total_size": 1}, "weight_map": {k: shard.name for k in tensors}}
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    return shard


class TestRestorePackedFromShards:
    def test_metadata_only_marks_compressed_without_tensors(self, tmp_path):
        lin = _fake_layer()
        model = torch.nn.Module()
        model.add_module("blk", lin)
        packed = torch.randint(-127, 128, (4, 4), dtype=torch.int8)
        _write_shards(
            tmp_path,
            {
                "blk.weight_packed": packed,
                "blk.weight_scale": torch.ones(4, 1),
                "blk.weight_shape": torch.tensor([4, 8]),
            },
        )
        n = _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=True)
        assert n == 1
        assert lin.quantization_status is QuantizationStatus.COMPRESSED
        assert lin.quantization_scheme is not None
        # the packed weights stay in the shards; nothing materialized in RAM
        assert not hasattr(lin, "weight_packed") and not hasattr(lin, "weight_scale")

    def test_full_restore_reads_tensors(self, tmp_path):
        lin = _fake_layer()
        model = torch.nn.Module()
        model.add_module("blk", lin)
        packed = torch.randint(-127, 128, (4, 4), dtype=torch.int8)
        scale = torch.ones(4, 1)
        _write_shards(tmp_path, {"blk.weight_packed": packed, "blk.weight_scale": scale})
        n = _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=False)
        assert n == 1
        assert torch.equal(lin.weight_scale.data, scale)
        assert getattr(lin, "quantization_status", None) is QuantizationStatus.COMPRESSED

    def test_index_json_drives_lookup(self, tmp_path):
        # no .safetensors scan needed: the index alone maps names to shards
        lin = _fake_layer()
        model = torch.nn.Module()
        model.add_module("blk", lin)
        shard = tmp_path / "model-00001-of-00001.safetensors"
        from safetensors.torch import save_file

        save_file({"blk.weight_packed": torch.zeros(4, 4, dtype=torch.int8)}, str(shard))
        index = {"weight_map": {"blk.weight_packed": shard.name}}
        with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index, f)
        n = _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=True)
        assert n == 1

    def test_absent_packed_state_is_noop(self, tmp_path):
        lin = _fake_layer()
        model = torch.nn.Module()
        model.add_module("blk", lin)
        _write_shards(tmp_path, {"other.weight_packed": torch.zeros(1)})
        assert _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=True) == 0
        assert getattr(lin, "quantization_status", None) is None


class TestRestoreRenamedShardKeys:
    """Shards may carry checkpoint-side names (conversion aliases): the
    restore must resolve module names before the packed-key lookup."""

    def test_resolver_marks_module_packed_under_checkpoint_name(self, tmp_path):
        lin = _fake_layer()
        model = torch.nn.Module()
        container = torch.nn.Module()
        se = torch.nn.Module()
        se.add_module("gate_proj", lin)
        container.add_module("shared_experts", se)
        model.add_module("mlp", container)
        packed = torch.randint(-127, 128, (4, 4), dtype=torch.int8)
        # shard key uses the CHECKPOINT name (shared_mlp), module uses the
        # transformers name (shared_experts)
        _write_shards(
            tmp_path,
            {
                "mlp.shared_mlp.gate_proj.weight_packed": packed,
                "mlp.shared_mlp.gate_proj.weight_scale": torch.ones(4, 1),
            },
        )
        resolver = lambda n: n.replace("shared_experts", "shared_mlp")  # noqa: E731
        n = _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=True, name_resolver=resolver)
        assert n == 1
        assert lin.quantization_status is QuantizationStatus.COMPRESSED

    def test_resolver_miss_leaves_module_unmarked(self, tmp_path):
        lin = _fake_layer()
        model = torch.nn.Module()
        model.add_module("blk", lin)
        _write_shards(tmp_path, {"other.weight_packed": torch.zeros(1)})
        resolver = lambda n: None  # noqa: E731
        assert (
            _restore_packed_modules_from_shards(model, str(tmp_path), metadata_only=True, name_resolver=resolver) == 0
        )
        assert getattr(lin, "quantization_status", None) is None
