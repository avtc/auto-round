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

import os
from types import SimpleNamespace

import pytest
import torch

from auto_round.compressors.shard_writer import ShardWriter
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext


class _ToyBlock(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class _DiffusionStyleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_ToyBlock()])
        self.proj_out = torch.nn.Linear(4, 2)
        self.config = SimpleNamespace(model_type="toy-diffusion")


class _FormatStub:

    def get_backend_name(self):
        return "auto_round"


def _make_writer(model, output_dir, monkeypatch):
    ShardWriter.reset()
    compress_context = SimpleNamespace(formats=[_FormatStub()], output_dir=output_dir)
    model_context = SimpleNamespace(is_diffusion=False)
    monkeypatch.setattr(CompressContext, "get_context", classmethod(lambda cls: compress_context))
    monkeypatch.setattr(ModelContext, "get_context", classmethod(lambda cls: model_context))
    return ShardWriter(model, bits=4, max_shard_size="1MB", safe_serialization=False)


def test_finalize_saves_tail_layer_when_tie_word_embeddings_missing(tmp_path, monkeypatch):
    model = _DiffusionStyleModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    assert writer.lm_head_name == "proj_out"
    assert not hasattr(model.config, "tie_word_embeddings")

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "proj_out.weight" in saved_tensors, "proj_out must be saved when tie_word_embeddings is absent"
    assert "proj_out.bias" in saved_tensors


class _LMStyleModel(torch.nn.Module):
    """Model whose config explicitly sets tie_word_embeddings=True."""

    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_ToyBlock()])
        self.lm_head = torch.nn.Linear(4, 2, bias=False)
        self.config = SimpleNamespace(model_type="toy-lm", tie_word_embeddings=True)


class _ToyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_transposed = False


class _FusedExpertsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.experts = _ToyExperts()
        self.talker = torch.nn.Module()
        self.talker.experts = _ToyExperts()
        self.config = SimpleNamespace(model_type="qwen3_omni_moe")


def test_finalize_skips_lm_head_when_tie_word_embeddings_true(tmp_path, monkeypatch):
    """Complementary test: when tie_word_embeddings=True the lm_head should be
    skipped (not written to disk) and offloaded to meta."""
    model = _LMStyleModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    assert writer.lm_head_name == "lm_head"

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "lm_head.weight" not in saved_tensors, "lm_head must be skipped when tied"
    assert model.lm_head.weight.device.type == "meta"


def test_expand_fused_experts_for_skipped_talker_prefix(tmp_path, monkeypatch):
    """Talker fused 3D weights must be expanded to exact per-expert 2D keys.

    Real Qwen3-Omni-MoE exports attach reverse checkpoint conversion mappings for
    MoE projections. If we apply that mapping to the fused talker tensor before
    expanding it, the fused tensor is saved under a wildcard key such as
    ``talker.experts.*.gate_proj.weight``. That breaks reload because
    transformers expects concrete per-expert 2D keys after save_pretrained.
    """
    model = _FusedExpertsModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)
    writer.reverse_checkpoint_conversion_mapping = {
        r"experts\.gate_up_proj$": ["experts.*.gate_proj.weight", "experts.*.up_proj.weight"]
    }

    fused_gate_up = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    writer._add_tensor("talker.experts.gate_up_proj", fused_gate_up)
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "talker.experts.gate_up_proj" not in saved_tensors
    assert "talker.experts.*.gate_proj.weight" not in saved_tensors
    assert "talker.experts.0.gate_proj.weight" in saved_tensors
    assert "talker.experts.0.up_proj.weight" in saved_tensors
    assert torch.equal(saved_tensors["talker.experts.0.gate_proj.weight"], fused_gate_up[0, :3, :])
    assert torch.equal(saved_tensors["talker.experts.0.up_proj.weight"], fused_gate_up[0, 3:, :])


def test_do_not_expand_fused_experts_outside_skipped_prefixes(tmp_path, monkeypatch):
    model = _FusedExpertsModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    fused_gate_up = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    writer._add_tensor("block.experts.gate_up_proj", fused_gate_up)
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "block.experts.gate_up_proj" in saved_tensors
    assert "block.experts.0.gate_proj.weight" not in saved_tensors


def test_finalize_offloads_module_with_tensor_in_parameters(tmp_path, monkeypatch):
    model = _DiffusionStyleModel()
    model.transformer_blocks[0].linear._parameters["weight"] = model.transformer_blocks[0].linear.weight.to("cpu")
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    offloaded_weight = model.transformer_blocks[0].linear._parameters["weight"]
    assert isinstance(offloaded_weight, torch.nn.Parameter)
    assert offloaded_weight.device.type == "meta"


def test_default_max_shard_size_is_fixed(tmp_path, monkeypatch):
    writer = _make_writer(_DiffusionStyleModel(), str(tmp_path), monkeypatch)
    assert writer.max_shard_size == 1 * 1024**2

    ShardWriter.reset()
    compress_context = SimpleNamespace(formats=[_FormatStub()], output_dir=str(tmp_path))
    model_context = SimpleNamespace(is_diffusion=False)
    monkeypatch.setattr(CompressContext, "get_context", classmethod(lambda cls: compress_context))
    monkeypatch.setattr(ModelContext, "get_context", classmethod(lambda cls: model_context))
    default_writer = ShardWriter(_DiffusionStyleModel(), bits=4, safe_serialization=False)
    assert default_writer.max_shard_size == 5 * 1024**3


def test_oversized_tensor_does_not_leave_tiny_preceding_shard(tmp_path, monkeypatch):
    writer = _make_writer(_DiffusionStyleModel(), str(tmp_path), monkeypatch)
    writer.max_shard_size = 1024

    writer._add_tensor("small", torch.zeros(1, dtype=torch.uint8))
    writer._add_tensor("large", torch.zeros(2048, dtype=torch.uint8))
    writer._flush_shard()

    assert writer.shard_counter == 1
    assert set(writer.current_shard_tensors) == set()


class TestAdoptExistingShards:
    """Resume support: a fresh writer must adopt shards a crashed run already wrote.

    Without adoption the resumed writer starts at shard_counter=0 and its first
    flush silently overwrites model-shard-00001.safetensors, destroying the
    crashed run's tensors and dropping them from the final index.
    """

    def _writer(self, output_dir, monkeypatch):
        ShardWriter.reset()
        compress_context = SimpleNamespace(formats=[_FormatStub()], output_dir=output_dir)
        model_context = SimpleNamespace(is_diffusion=False)
        monkeypatch.setattr(CompressContext, "get_context", classmethod(lambda cls: compress_context))
        monkeypatch.setattr(ModelContext, "get_context", classmethod(lambda cls: model_context))
        return ShardWriter(torch.nn.Module(), bits=4, max_shard_size="1MB", safe_serialization=True)

    def test_adopt_continues_numbering_and_index_covers_all(self, tmp_path, monkeypatch):
        from safetensors.torch import load_file

        out = str(tmp_path)
        t0 = torch.randn(4, 4)
        w1 = self._writer(out, monkeypatch)
        w1.save_tensor("blk.0.w", t0)
        w1._flush_shard()
        # crash: no finalize, singleton reset
        w2 = self._writer(out, monkeypatch)
        adopted = w2.adopt_existing_shards()
        assert adopted == 1
        assert "blk.0.w" in w2._all_saved
        assert w2.shard_counter == 1
        t1 = torch.randn(4, 4)
        w2.save_tensor("blk.1.w", t1)
        w2.finalize()

        import json

        index = json.loads(open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8").read())
        assert index["metadata"]["total_shards"] == 2
        assert "blk.0.w" in index["weight_map"]
        assert "blk.1.w" in index["weight_map"]
        # the crashed run's shard must survive byte-identical (no overwrite)
        shard0 = os.path.join(out, "model-00001-of-00002.safetensors")
        assert torch.equal(load_file(shard0)["blk.0.w"], t0)
        shard1 = os.path.join(out, "model-00002-of-00002.safetensors")
        assert torch.equal(load_file(shard1)["blk.1.w"], t1)

    def test_adopt_deletes_incomplete_tail_shard(self, tmp_path, monkeypatch):
        out = str(tmp_path)
        w1 = self._writer(out, monkeypatch)
        w1.save_tensor("blk.0.w", torch.randn(4, 4))
        w1._flush_shard()
        w1.save_tensor("blk.1.w", torch.randn(4, 4))
        w1._flush_shard()
        # crash mid-flush of a third shard: truncated file on disk
        tail = os.path.join(out, "model-shard-00003.safetensors")
        with open(tail, "wb") as f:
            f.write(b"\x00" * 17)  # header length + partial json

        w2 = self._writer(out, monkeypatch)
        adopted = w2.adopt_existing_shards()
        assert adopted == 2
        assert not os.path.exists(tail), "incomplete tail shard must be deleted"
        assert w2.shard_counter == 2

    def test_adopt_deletes_truncated_data_tail_shard(self, tmp_path, monkeypatch):
        out = str(tmp_path)
        w1 = self._writer(out, monkeypatch)
        w1.save_tensor("blk.0.w", torch.randn(4, 4))
        w1._flush_shard()
        # crash AFTER the header flushed but DURING the data write: the tail
        # shard parses as valid safetensors but its declared data is truncated
        import struct as _struct

        from safetensors.torch import save_file

        full = os.path.join(out, "model-shard-00002.safetensors")
        save_file({"blk.1.w": torch.randn(8, 8)}, full)
        with open(full, "r+b") as f:
            header_len = _struct.unpack("<Q", f.read(8))[0]
            f.truncate(8 + header_len + 16)  # cut inside the tensor data

        w2 = self._writer(out, monkeypatch)
        adopted = w2.adopt_existing_shards()
        assert adopted == 1
        assert not os.path.exists(full), "header-valid but truncated tail shard must be deleted"
        # counter = max adopted index; the next flush continues after it
        assert w2.shard_counter == 1

    def test_adopt_raises_on_corrupt_non_tail_shard(self, tmp_path, monkeypatch):
        out = str(tmp_path)
        w1 = self._writer(out, monkeypatch)
        w1.save_tensor("blk.0.w", torch.randn(4, 4))
        w1._flush_shard()
        w1.save_tensor("blk.1.w", torch.randn(4, 4))
        w1._flush_shard()
        # corrupt the FIRST shard (not the tail): real corruption
        with open(os.path.join(out, "model-shard-00001.safetensors"), "r+b") as f:
            f.write(b"\xff\xff\xff\xff\xff\xff\xff\xff")

        w2 = self._writer(out, monkeypatch)
        with pytest.raises(RuntimeError, match="corrupt"):
            w2.adopt_existing_shards()

    def test_adopt_refuses_bin_shards(self, tmp_path, monkeypatch):
        out = str(tmp_path)
        with open(os.path.join(out, "model-shard-00001.bin"), "wb") as f:
            f.write(b"junk")
        w = self._writer(out, monkeypatch)
        with pytest.raises(RuntimeError, match="safetensors"):
            w.adopt_existing_shards()

    def test_adopt_temp_and_final_same_ordinal_keeps_temp(self, tmp_path, monkeypatch):
        """A reused output dir can hold a stale finalized shard from a previous
        lineage next to the current lineage's temp shard at the same ordinal:
        the temp (resumable artifact) must win regardless of listdir order and
        the stale final must be named, not silently dropped."""
        from safetensors.torch import load_file, save_file

        out = str(tmp_path)
        stale = torch.randn(4, 4) + 100.0
        save_file({"stale.w": stale}, os.path.join(out, "model-00001-of-00009.safetensors"))
        w1 = self._writer(out, monkeypatch)
        t0 = torch.randn(4, 4)
        w1.save_tensor("blk.0.w", t0)
        w1._flush_shard()  # leaves model-shard-00001.safetensors
        w2 = self._writer(out, monkeypatch)
        adopted = w2.adopt_existing_shards()
        assert adopted == 1
        assert "blk.0.w" in w2._all_saved
        assert "stale.w" not in w2._all_saved
        t1 = torch.randn(4, 4)
        w2.save_tensor("blk.1.w", t1)
        w2.finalize()
        import json

        index = json.loads(open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8").read())
        assert "blk.0.w" in index["weight_map"] and "blk.1.w" in index["weight_map"]
        assert "stale.w" not in index["weight_map"]
        assert torch.equal(load_file(os.path.join(out, "model-00001-of-00002.safetensors"))["blk.0.w"], t0)

    def test_add_tensor_dedups_across_conversion_spellings(self, tmp_path, monkeypatch):
        """The duplicate guard normalizes to the checkpoint-side spelling first:
        a renamed-family tensor offered under both spellings is written once."""
        w = self._writer(str(tmp_path), monkeypatch)
        w.reverse_checkpoint_conversion_mapping = {"mlp.gate": "mlp.router.gate"}
        t = torch.randn(4, 4)
        w.save_tensor("model.layers.0.mlp.router.gate.weight", t)  # checkpoint-side
        w.save_tensor("model.layers.0.mlp.gate.weight", t.clone())  # module-side alias
        assert w.total_param_elems == t.numel(), "the aliased spelling was written a second time"
