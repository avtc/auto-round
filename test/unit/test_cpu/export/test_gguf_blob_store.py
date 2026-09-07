# Copyright (c) 2025 Intel Corporation
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

import gc
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from auto_round.export.export_to_gguf.blob_store import (  # noqa: E402
    GgufBlobStore,
    RecordingGgufWriter,
    assemble_gguf_from_blobs,
)


class _FakeConversion:
    """Minimal conversion-instance stand-in for recorder attach/finalize."""

    def __init__(self, arch="llama", endianess="little", fname_out="out.gguf"):
        from gguf import GGUFWriter

        self.model_arch = gguf.MODEL_ARCH[arch.upper()]
        self.endianess = endianess
        self.fname_out = fname_out
        self.gguf_writer = GGUFWriter(path=None, arch=arch)


@pytest.fixture()
def store(tmp_path):
    return GgufBlobStore(tmp_path / "export")


from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES  # noqa: E402


def _quant_bytes(cols, qtype):
    block, tsize = GGML_QUANT_SIZES[qtype.name.lower()]
    assert cols % block == 0
    return (cols // block) * tsize


_RNG = np.random.default_rng(1234)


def _add_q4_tensor(recorder, name, rows, cols=256, qtype=gguf.GGMLQuantizationType.Q4_K):
    data = _RNG.integers(0, 255, size=(rows, _quant_bytes(cols, qtype)), dtype=np.uint8)
    recorder.add_tensor(name, data, raw_dtype=qtype)
    return data


def test_recorder_records_kv_ops_and_tensors(store):
    rec = store.attach_recorder(_FakeConversion(), "text")
    rec.add_uint32("general.block_count", 2)
    rec.add_string("general.architecture", "llama")
    rec.add_array("tokenizer.ggml.tokens", np.asarray(["a", "b", "c"], dtype=np.str_))
    _add_q4_tensor(rec, "blk.0.attn_q.weight", 8)

    assert rec.kv_ops[0]["method"] == "add_uint32"
    assert rec.kv_ops[0]["args"] == ["general.block_count", 2]
    assert rec.tensors[0]["blk.0.attn_q.weight"].shape == (8, 256)
    assert rec.tensors[0]["blk.0.attn_q.weight"].dtype == "Q4_K"
    # membership check used by the conversion loop keeps working
    from auto_round.export.export_to_gguf.convert import _gguf_writer_has_tensor

    assert _gguf_writer_has_tensor(rec, "blk.0.attn_q.weight")
    assert not _gguf_writer_has_tensor(rec, "blk.0.attn_k.weight")


def test_flush_writes_shards_and_manifest(store, tmp_path):
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    data = _add_q4_tensor(rec, "blk.0.attn_q.weight", 8)
    rec.add_string("general.architecture", "llama")
    assert store.flush("text") == 1

    manifest = json.loads((store.blob_dir / "manifest.json").read_text())
    entry = manifest["roles"]["text"]["tensors"][0]
    assert entry["name"] == "blk.0.attn_q.weight"
    assert entry["dtype"] == "Q4_K"
    assert entry["shard"] == "blob-text-00001.safetensors"

    from safetensors.numpy import load_file

    saved = load_file(str(store.blob_dir / entry["shard"]))
    assert np.array_equal(saved["blk.0.attn_q.weight"], data)
    # second flush with nothing pending is a no-op that keeps the manifest intact
    assert store.flush("text") == 0
    assert len(json.loads((store.blob_dir / "manifest.json").read_text())["roles"]["text"]["tensors"]) == 1


def test_finalize_and_assemble_roundtrip(store, tmp_path):
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    q4 = _add_q4_tensor(rec, "blk.0.attn_q.weight", 8)
    f32 = np.ones((4,), dtype=np.float32)
    rec.add_tensor("blk.0.attn_norm.weight", f32, raw_dtype=gguf.GGMLQuantizationType.F32)
    rec.add_uint32("general.block_count", 1)
    rec.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_K_M)
    rec.add_array("some.int.array", [3, 1, 4])
    out = store.finalize_role(conv, "text")
    assert out == tmp_path / "m.gguf"
    assert not out.exists()  # finalize only persists blobs; assembly writes the file

    [produced] = assemble_gguf_from_blobs(store.root)
    assert produced == out
    reader = gguf.GGUFReader(str(produced))
    fields = {f.name: f for f in reader.fields.values()}
    assert fields["general.block_count"].parts[-1][0] == 1
    assert int(fields["general.file_type"].parts[-1][0]) == int(gguf.LlamaFileType.MOSTLY_Q4_K_M)
    tensors = {t.name: t for t in reader.tensors}
    assert set(tensors) == {"blk.0.attn_q.weight", "blk.0.attn_norm.weight"}
    assert tensors["blk.0.attn_q.weight"].tensor_type == gguf.GGMLQuantizationType.Q4_K
    assert tensors["blk.0.attn_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert np.array_equal(tensors["blk.0.attn_norm.weight"].data, f32)

    # byte-identical re-assembly (determinism); the reader mem-maps the file,
    # which blocks re-opening it for writing on Windows until it is released
    first = produced.read_bytes()
    del reader, tensors, fields
    gc.collect()
    assemble_gguf_from_blobs(store.root)
    assert produced.read_bytes() == first


def test_assembly_matches_interactive_writer_bytes(store, tmp_path):
    """Blob path must equal what a real GGUFWriter fed the same calls produces."""
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    tensors = {}
    rng = np.random.default_rng(0)
    for i in range(3):
        data = rng.integers(0, 255, size=(16, _quant_bytes(256, gguf.GGMLQuantizationType.Q4_0)), dtype=np.uint8)
        name = f"blk.{i}.attn_q.weight"
        rec.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        tensors[name] = data
    norm = rng.standard_normal(64).astype(np.float32)
    rec.add_tensor("blk.0.attn_norm.weight", norm, raw_dtype=gguf.GGMLQuantizationType.F32)
    tensors["blk.0.attn_norm.weight"] = norm
    rec.add_uint32("general.block_count", 3)
    store.finalize_role(conv, "text")
    [blob_out] = assemble_gguf_from_blobs(store.root)

    direct = tmp_path / "direct.gguf"
    w = gguf.GGUFWriter(path=None, arch="llama")
    w.add_uint32("general.block_count", 3)
    for name, data in tensors.items():
        raw = gguf.GGMLQuantizationType.Q4_0 if name.endswith("attn_q.weight") else gguf.GGMLQuantizationType.F32
        w.add_tensor(name, data, raw_dtype=raw)
    w.write_header_to_file(path=str(direct))
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert blob_out.read_bytes() == direct.read_bytes()


def test_total_parameter_count_matches_real_writer(store):
    conv = _FakeConversion()
    rec = store.attach_recorder(conv, "text")
    _add_q4_tensor(rec, "blk.0.attn_q.weight", 8, cols=256)
    gate = _add_q4_tensor(rec, "blk.0.ffn_gate_exps.weight", 8, cols=256)
    # regroup the same bytes as 2 experts x 4 rows so expert counting sees a
    # 3-D fused tensor on BOTH writers
    gate = gate.reshape(4, 2, -1)
    rec.tensors[0]["blk.0.ffn_gate_exps.weight"]  # noqa: B018 - keep name in manifest
    rec2_gate = gate.copy()
    rec.tensors[0]["blk.0.ffn_gate_exps.weight"].shape = (4, 2, 256)
    real = gguf.GGUFWriter(path=None, arch="llama")
    real.add_tensor(
        "blk.0.attn_q.weight",
        np.zeros((8, _quant_bytes(256, gguf.GGMLQuantizationType.Q4_K)), np.uint8),
        raw_dtype=gguf.GGMLQuantizationType.Q4_K,
    )
    real.add_tensor("blk.0.ffn_gate_exps.weight", rec2_gate, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    assert rec.get_total_parameter_count() == real.get_total_parameter_count()


def test_adopt_existing_restores_entries(store, tmp_path):
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
    store.flush("text")

    fresh = GgufBlobStore(store.root)
    adopted = fresh.adopt_existing()
    assert adopted == 1
    state = fresh._role_state("text")
    assert state["shard_counter"] == 1
    # the next flush lands in a NEW shard, keeping the old bytes intact
    rec2 = fresh.attach_recorder(_FakeConversion(fname_out=str(tmp_path / "m2.gguf")), "text")
    _add_q4_tensor(rec2, "blk.1.attn_q.weight", 4)
    fresh.flush("text")
    manifest = json.loads((fresh.blob_dir / "manifest.json").read_text())
    shards = [e["shard"] for e in manifest["roles"]["text"]["tensors"]]
    assert shards == ["blob-text-00001.safetensors", "blob-text-00002.safetensors"]


def test_assembly_is_little_endian_by_default(store, tmp_path):
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
    store.finalize_role(conv, "text")
    [out] = assemble_gguf_from_blobs(store.root)
    header = out.read_bytes()[:12]
    assert header[:4] == b"GGUF"
    assert int.from_bytes(header[4:8], "little") == 3  # GGUF v3, little-endian


def test_float16_passthrough_and_bytes_kv(store, tmp_path):
    conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
    rec = store.attach_recorder(conv, "text")
    f16 = np.asarray([[1.5, -2.5, 3.5, 4.5]], dtype=np.float16)
    rec.add_tensor("token_embd.weight", f16, raw_dtype=gguf.GGMLQuantizationType.F16)
    rec.add_string("tokenizer.ggml.pre", bytes([0x1, 0x2, 0x3]))
    store.finalize_role(conv, "text")
    [out] = assemble_gguf_from_blobs(store.root)
    reader = gguf.GGUFReader(str(out))
    tensors = {t.name: t for t in reader.tensors}
    assert tensors["token_embd.weight"].tensor_type == gguf.GGMLQuantizationType.F16
    assert np.array_equal(tensors["token_embd.weight"].data, f16)
    pre = reader.fields["tokenizer.ggml.pre"].parts[-1]
    assert bytes(bytearray(np.asarray(pre).view(np.uint8).ravel())) == bytes([0x1, 0x2, 0x3])


class TestBlobModeBranches:
    def test_gguf_blob_mode_detection(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        gguf = SimpleNamespace(is_gguf=lambda: True)
        ct = SimpleNamespace(is_gguf=lambda: False)

        def fake(formats, stream):
            return SimpleNamespace(formats=formats, model_context=SimpleNamespace(stream_quantization=stream))

        assert CompressionOrchestrator._gguf_blob_mode(fake([gguf], True)) is True
        assert CompressionOrchestrator._gguf_blob_mode(fake([gguf], False)) is False
        assert CompressionOrchestrator._gguf_blob_mode(fake([ct], True)) is False
        assert CompressionOrchestrator._gguf_blob_mode(fake([gguf, gguf], True)) is False
        assert CompressionOrchestrator._gguf_blob_mode(fake(None, True)) is False

    def test_write_finished_block_skips_ct_writes_in_blob_mode(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        calls = []
        leaf = SimpleNamespace(to=lambda device: calls.append(("leaf.to", device)))

        class _Block:
            def named_modules(self):
                mod = SimpleNamespace(
                    children=lambda: iter([]),
                    state_dict=lambda: {"weight": 1},
                    global_name="model.layers.0.norm",
                    to=lambda device: calls.append(("module.to", device)),
                )
                return iter([("norm", mod)])

            def to(self, device):
                calls.append(("block.to", device))

        monkeypatch.setattr(orch, "get_module", lambda model, name: leaf)
        monkeypatch.setattr(orch, "set_module", lambda model, name, mod: calls.append(("set_module", name)))
        written = []
        fake = SimpleNamespace(
            _gguf_blob_mode=lambda: True,
            model=SimpleNamespace(),
            shard_writer=SimpleNamespace(
                write=lambda **kw: written.append(kw), _flush_shard=lambda: written.append("flush")
            ),
        )
        rs = SimpleNamespace(mark_block_done=lambda *a, **kw: calls.append(("mark_done", a[0])))
        CompressionOrchestrator._write_finished_block_(fake, _Block(), "model.layers.0", set(), rs, None, None, True)
        assert written == [], "no compressed-tensors writes in blob mode"
        assert ("block.to", "meta") in calls
        assert ("mark_done", "model.layers.0") in calls

        fake._gguf_blob_mode = lambda: False
        CompressionOrchestrator._write_finished_block_(fake, _Block(), "model.layers.0", set(), rs, None, None, True)
        assert written, "CT path keeps writing"


@pytest.mark.slow
class TestStreamGgufBlobE2E:
    """End-to-end: stream-quantize a tiny model straight to blob shards + assembled GGUF.

    Uses the synthetic tiny Llama checkpoint (same one as the streaming
    equivalence suite) plus a gpt2-style fast tokenizer so the GGUF vocab
    embedding path is exercised without a network-fetched fixture.
    """

    @pytest.fixture(scope="class")
    def tiny_checkpoint(self, tmp_path_factory):
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        torch.manual_seed(11)
        model = LlamaForCausalLM(cfg)
        d = tmp_path_factory.mktemp("tiny_gguf_ckpt")
        from tokenizers import Tokenizer
        from tokenizers import decoders, models as tk_models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        # BPE + ByteLevel keeps the GGUF vocab embedding on the recognized
        # gpt2 path (a plain WordLevel/Whitespace tokenizer is rejected by
        # get_vocab_base_pre)
        tk = Tokenizer(tk_models.BPE(vocab={"[UNK]": 0, "a": 1, "b": 2}, merges=[]))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(str(d))
        model.save_pretrained(str(d), max_shard_size="40KB")
        return str(d)

    def test_stream_quantize_assembles_gguf(self, tiny_checkpoint, tmp_path, monkeypatch):
        gguf = pytest.importorskip("gguf")
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.autoround import AutoRound
        from auto_round.export.export_to_gguf.blob_store import GgufBlobStore
        from auto_round.export.export_to_gguf import llama_cpp_conversion as lcc

        # the synthetic tokenizer cannot reproduce any real model's
        # pre-tokenizer checksum; the blob machinery under test is agnostic
        # to which pre-tokenizer the vocab claims. The conversion classes load
        # under the top-level "conversion" package (cached module), so patch
        # them there - the auto_round copy of the same files is not the one
        # the live export instantiates.
        lcc.get_conversion(tiny_checkpoint)
        conversion_llama = importlib.import_module("conversion.llama")
        monkeypatch.setattr(conversion_llama.LlamaModel, "get_vocab_base_pre", lambda self, tokenizer: "gpt2")
        GgufBlobStore.reset_singletons()
        out_dir = str(tmp_path / "out")
        ar = AutoRound(
            tiny_checkpoint,
            scheme="gguf:q4_0",
            alg_configs=[RTNConfig(group_size=32, disable_opt_rtn=False)],
            stream_quantization=True,
            stream_prefetch="off",
            format="gguf:q4_0",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(out_dir, format="gguf:q4_0")

        blobs = Path(ar.output_dir) / "gguf-blobs"
        assert (blobs / "manifest.json").is_file(), "blob manifest missing"
        manifest = json.loads((blobs / "manifest.json").read_text())
        entries = manifest["roles"]["text"]["tensors"]
        assert entries, "no blob tensors recorded"

        # NOTE: on Windows, absolute model paths defeat the "/"-split leaf
        # derivation in _get_export_dir (a pre-existing quirk; relative paths
        # and POSIX paths derive correctly), so the assembled file's exact
        # folder can differ - the manifest's recorded out_path is the source
        # of truth for where assembly put it.
        out_path = Path(manifest["roles"]["text"]["out_path"])
        assert out_path.is_file(), f"assembled .gguf missing at {out_path}"
        reader = gguf.GGUFReader(str(out_path))
        names = {t.name for t in reader.tensors}
        manifest_names = {e["name"] for e in entries}
        assert names == manifest_names, "assembled tensors must match the manifest exactly"
        dtypes = {t.name: t.tensor_type for t in reader.tensors}
        assert dtypes["blk.0.attn_q.weight"] == gguf.GGMLQuantizationType.Q4_0
        assert dtypes["blk.0.ffn_norm.weight"] == gguf.GGMLQuantizationType.F32
        assert manifest["roles"]["text"]["kv_ops"], "metadata must be captured for replay"


class TestMtpNextnRemap:
    """Embedded nextn enablement: name remap + tree counting (streaming blob path)."""

    @pytest.fixture()
    def qwen_mtp_cls(self):
        from auto_round.export.export_to_gguf.conversion.qwen import Qwen3NextModel

        return Qwen3NextModel

    def test_remap_table(self, qwen_mtp_cls):
        remap = qwen_mtp_cls._remap_mtp_name_
        assert remap("mtp.fc.weight", 64) == "model.layers.64.eh_proj.weight"
        assert remap("mtp.pre_fc_norm_embedding.weight", 64) == "model.layers.64.enorm.weight"
        assert remap("mtp.pre_fc_norm_hidden.weight", 64) == "model.layers.64.hnorm.weight"
        assert remap("mtp.norm.weight", 64) == "model.layers.64.shared_head.norm.weight"
        assert remap("mtp.layers.0.self_attn.q_proj.weight", 64) == "model.layers.64.self_attn.q_proj.weight"
        assert remap("model.mtp.fc.weight", 64) == "model.layers.64.eh_proj.weight"
        # non-MTP names pass through untouched
        assert remap("model.layers.3.fc1.weight", 64) == "model.layers.3.fc1.weight"

    def test_count_mtp_layers(self):
        from auto_round.export.export_to_gguf.export import _count_mtp_layers

        def _model(names):
            return SimpleNamespace(named_modules=lambda: [(n, None) for n in names])

        assert _count_mtp_layers(_model(["model.layers.0.a", "mtp.fc", "mtp.layers.0.x", "mtp.layers.0.y"])) == 1
        assert _count_mtp_layers(_model(["model.mtp.layers.0.a", "model.mtp.layers.1.b"])) == 2
        assert _count_mtp_layers(_model(["mtp.fc", "mtp.norm"])) == 0
        assert _count_mtp_layers(_model(["model.layers.0.a"])) == 0

    def test_create_conversion_model_include_mtp(self, monkeypatch):
        from auto_round.export.export_to_gguf.export import _create_conversion_model

        constructed = {}

        class _FakeMtp:
            supports_mtp_export = True
            no_mtp = True

            def __init__(self, hparams, **kwargs):
                constructed["hparams"] = hparams
                constructed["kwargs"] = kwargs
                self.no_mtp = _FakeMtp.no_mtp

        class _FakePlain:
            supports_mtp_export = False

            def __init__(self, hparams, **kwargs):
                constructed["plain"] = True

        inst = _create_conversion_model(_FakeMtp, {"a": 1}, include_mtp=True, extra=2)
        assert inst.no_mtp is False and _FakeMtp.no_mtp is True, "class flag must be restored"
        inst = _create_conversion_model(_FakeMtp, {"a": 1})
        assert inst.no_mtp is True
        constructed.pop("plain", None)
        _create_conversion_model(_FakePlain, {"a": 1}, include_mtp=True)
        assert constructed["plain"], "non-MTP classes construct directly regardless of the flag"


class TestBlobResumeAndRerun:
    """R1 hardening: crash-resume adoption, in-process reruns, recorder parity."""

    def test_resume_adopts_prior_blobs_and_rebuilds_recorder(self, store, tmp_path):
        import json as _json

        conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
        rec = store.attach_recorder(conv, "text")
        _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
        store.flush("text")

        # simulated crash + new process: fresh store adopts the manifest
        fresh = GgufBlobStore(store.root)
        assert fresh.adopt_existing() == 1
        conv2 = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
        rec2 = fresh.attach_recorder(conv2, "text")
        assert "blk.0.attn_q.weight" in rec2.tensors[0], "adopted tensors must replay into the recorder"
        # the next flush continues at the next shard index - the crashed run's
        # shard bytes are never overwritten
        _add_q4_tensor(rec2, "blk.1.attn_q.weight", 4)
        fresh.flush("text")
        shards = sorted(p.name for p in fresh.blob_dir.glob("blob-text-*.safetensors"))
        assert shards == ["blob-text-00001.safetensors", "blob-text-00002.safetensors"]
        manifest = _json.loads((fresh.blob_dir / "manifest.json").read_text())
        names = [e["name"] for e in manifest["roles"]["text"]["tensors"]]
        assert names == ["blk.0.attn_q.weight", "blk.1.attn_q.weight"]

    def test_rerun_resets_via_clear_instances(self, store, tmp_path):
        from auto_round.export.export_to_gguf.export import _clear_gguf_model_instances

        conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
        rec = store.attach_recorder(conv, "text")
        _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
        store.flush("text")

        _clear_gguf_model_instances()
        fresh = GgufBlobStore.get_or_create(store.root)
        assert fresh._role_state("text")["recorder"] is None
        assert fresh._role_state("text")["entries"] == []

    def test_recorder_rejects_duplicate_tensor_names(self, store):
        conv = _FakeConversion()
        rec = store.attach_recorder(conv, "text")
        _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
        with pytest.raises(ValueError, match="Duplicated tensor name"):
            _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)

    def test_decode_refuses_foreign_enum_modules(self, store):
        from auto_round.export.export_to_gguf.blob_store import _decode_value

        with pytest.raises(ValueError, match="refusing to decode"):
            _decode_value({"__enum__": ["os.path", "SomeEnum", "MEMBER"]})

    def test_mtp_packing_block_name_normalization(self):
        from auto_round.export.export_to_gguf.convert import _remap_mtp_checkpoint_name_

        class _Inst:
            no_mtp = False
            hparams = {"num_hidden_layers": 64}
            _original_block_count = None

            @classmethod
            def _remap_mtp_name_(cls, name, base):
                from auto_round.export.export_to_gguf.conversion.qwen import Qwen3NextModel

                return Qwen3NextModel._remap_mtp_name_(name, base)

        inst = _Inst()
        # the packing-block filter feeds tensor-style names through the remap
        assert _remap_mtp_checkpoint_name_(inst, "model.mtp.layers.0.self_attn") == "model.layers.64.self_attn"
        assert _remap_mtp_checkpoint_name_(inst, "model.mtp.layers.0.weight")[: -len(".weight")] == "model.layers.64"
        assert _remap_mtp_checkpoint_name_(inst, "model.layers.3") == "model.layers.3"
        inst.no_mtp = True
        assert _remap_mtp_checkpoint_name_(inst, "model.mtp.layers.0") == "model.mtp.layers.0"


class TestR2Hardening:
    """R2 review contracts: early-instance registration, fail-fast, parity."""

    def test_ensure_blob_conversion_registers_global(self, monkeypatch, tmp_path):
        """The MTP-aware early instance must become the pack-time global."""
        import auto_round.export.export_to_gguf.export as gguf_export
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        created = {}

        def _fake_create(output_dir, model, layer_config, backend, **kwargs):
            created["kwargs"] = kwargs
            inst = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
            created["instance"] = inst
            return inst

        import auto_round.compressors.utils as orch_utils

        monkeypatch.setattr(gguf_export, "create_model_class", _fake_create)
        monkeypatch.setattr(orch_utils, "_get_save_folder_name", lambda fmt, *a, **kw: str(tmp_path / "out"))
        monkeypatch.setattr(gguf_export, "_clear_gguf_model_instances", lambda: None, raising=False)
        gguf_export.gguf_model_instance_global = None

        class _Streamer:
            weight_map = {"mtp.fc.weight": "a.safetensors", "mtp.layers.0.a.weight": "a.safetensors"}

        orch = SimpleNamespace(
            _gguf_blob_mode=lambda: True,
            model=SimpleNamespace(),
            layer_config={},
            formats=[SimpleNamespace(get_backend_name=lambda: "gguf:q4_0")],
            model_context=SimpleNamespace(),
            device="cpu",
        )
        try:
            CompressionOrchestrator._ensure_gguf_blob_conversion_(orch, _Streamer())
            assert created["kwargs"]["mtp_checkpoint_names"] == list(_Streamer.weight_map)
            assert gguf_export.gguf_model_instance_global == [
                created["instance"]
            ], "the early instance must be registered as the pack-time global"
            # a second call must not clobber an existing registration
            other = object()
            gguf_export.gguf_model_instance_global = [other]
            CompressionOrchestrator._ensure_gguf_blob_conversion_(orch, None)
            assert gguf_export.gguf_model_instance_global == [other]
        finally:
            gguf_export.gguf_model_instance_global = None

    def test_recorder_raw_shape_matches_real_writer(self, store):
        rec = store.attach_recorder(_FakeConversion(), "text")
        data = np.zeros((4, _quant_bytes(256, gguf.GGMLQuantizationType.Q4_K)), np.uint8)
        rec.add_tensor("blk.0.attn_q.weight", data, raw_shape=(4, 144), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        info = rec.tensors[0]["blk.0.attn_q.weight"]
        real = gguf.GGUFWriter(path=None, arch="llama")
        real.add_tensor("blk.0.attn_q.weight", data, raw_shape=(4, 144), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
        real_info = real.tensors[-1]["blk.0.attn_q.weight"] if isinstance(real.tensors[-1], dict) else None
        if real_info is not None:
            assert tuple(info.shape) == tuple(real_info.shape)
