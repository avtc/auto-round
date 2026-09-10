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


class _PackedLinear(torch.nn.Module):
    """The packed form a previous process flushed for a quantized module."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.zeros(8, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(2))


class _StaleWrapperLinear(torch.nn.Module):
    """Mimics a quant wrapper left on a resume-skipped module: carries the
    `orig_layer` marker (kept out of state_dict) and exposes the fp weight
    under the plain ``.weight`` name, like the real DataWrapper does."""

    def __init__(self):
        super().__init__()
        inner = torch.nn.Linear(4, 4, bias=False)
        self.__dict__["orig_layer"] = inner
        self.weight = inner.weight


class _ResumedModel(torch.nn.Module):

    def __init__(self, linear):
        super().__init__()
        block = torch.nn.Module()
        block.linear = linear
        self.blocks = torch.nn.ModuleList([block])
        self.embed = torch.nn.Linear(4, 4, bias=False)


def test_finalize_drops_stale_wrapper_weights_of_resume_skipped_modules(tmp_path, monkeypatch):
    """A resumed process never re-packs blocks its resume manifest marks done.
    Their packed tensors already live in shards flushed by the interrupted
    process, while the live model still holds the quant wrappers with their fp
    ``.weight``. finalize() must NOT write those stale fp tensors -- otherwise
    the artifact contains BOTH forms for the same module and a loader may
    silently serve the unquantized fp layer."""
    import glob

    monkeypatch.delenv("AR_RESUME_DIR", raising=False)
    # Process 1 (interrupted): flushed the PACKED form of blocks.0.linear.
    prev = _ResumedModel(_PackedLinear())
    writer1 = _make_writer(prev, str(tmp_path), monkeypatch)
    writer1.save_module(prev.blocks[0].linear, name="blocks.0.linear")
    writer1._flush_shard()
    ShardWriter.reset()

    # Process 2 (resumed): same output_dir; skipped module is still a wrapper
    # with its real fp weight; a later block's write triggers shard discovery
    # exactly like the real per-block loop does.
    monkeypatch.setenv("AR_RESUME_DIR", str(tmp_path / "resume"))
    model = _ResumedModel(_StaleWrapperLinear())
    writer2 = _make_writer(model, str(tmp_path), monkeypatch)
    writer2.save_module(model.embed, name="embed")
    writer2._flush_shard()
    writer2.write(None, is_finalize=True)

    tensors = {}
    for f in sorted(glob.glob(str(tmp_path / "*.bin"))):
        tensors.update(torch.load(f, map_location="cpu"))
    assert "blocks.0.linear.qweight" in tensors, "packed form from the old shard must be kept"
    assert "blocks.0.linear.weight" not in tensors, "stale fp wrapper weight must be dropped"
    assert "embed.weight" in tensors, "untouched modules must still be saved"
