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
            regex_config={},
            _layer_config_with_regex_pins_=lambda: {},
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


class TestMmprojInstanceUnderEarlyRegistration:
    """The streaming orchestrator registers the text instance early; the
    mmproj instance must still be created for MLLM runs at save time."""

    @pytest.fixture(autouse=True)
    def _reset_global(self):
        import auto_round.export.export_to_gguf.export as gguf_export

        self.had = "gguf_model_instance_global" in getattr(gguf_export, "__dict__", {})
        gguf_export.gguf_model_instance_global = None
        yield
        if not self.had:
            gguf_export.__dict__.pop("gguf_model_instance_global", None)

    def _run(self, monkeypatch, tmp_path, pre_registered, mllm):
        import gguf as gguf_pkg

        import auto_round.export.export_to_gguf.export as gguf_export
        from auto_round.export.export_to_gguf.config import ModelType

        created = []

        class _FakeInst:
            def __init__(self, model_type):
                self.model_arch = gguf_pkg.MODEL_ARCH.MMPROJ if model_type == ModelType.MMPROJ else object()
                self.fname_out = str(tmp_path / (f"{model_type.name}-out"))
                self.wrote = False

            def write(self):
                self.wrote = True

        def _fake_create(output_dir, model, layer_config, backend, **kwargs):
            inst = _FakeInst(kwargs.get("model_type", ModelType.TEXT))
            created.append(inst)
            return inst

        monkeypatch.setattr(gguf_export, "create_model_class", _fake_create)
        pre = None
        if pre_registered:
            pre = _FakeInst(ModelType.TEXT)
            gguf_export.gguf_model_instance_global = [pre]
        else:
            # the real latch is attribute *presence*; delete it for the lazy path
            gguf_export.__dict__.pop("gguf_model_instance_global", None)
        # save_quantized_as_gguf clears the global in its finally block
        gguf_export.save_quantized_as_gguf(
            str(tmp_path / "out"),
            model=object(),
            layer_config={},
            mllm=mllm,
            blob_store=None,
        )
        return created, pre

    def test_mmproj_created_when_text_pre_registered(self, monkeypatch, tmp_path):
        """Regression: the mllm append used to live inside the
        `global not in globals()` guard, so the early-registered streaming
        instance suppressed mmproj creation entirely."""
        from auto_round.export.export_to_gguf.config import ModelType

        created, pre = self._run(monkeypatch, tmp_path, pre_registered=True, mllm=True)
        assert pre.wrote, "the early-registered text instance must still be written"
        assert any(
            i.model_arch == gguf.MODEL_ARCH.MMPROJ for i in created
        ), "mmproj instance must be created even when the text instance was registered early"

    def test_mmproj_not_duplicated_without_mllm(self, monkeypatch, tmp_path):
        created, _ = self._run(monkeypatch, tmp_path, pre_registered=True, mllm=False)
        assert not any(i.model_arch == gguf.MODEL_ARCH.MMPROJ for i in created)

    def test_lazy_path_still_creates_both(self, monkeypatch, tmp_path):
        created, _ = self._run(monkeypatch, tmp_path, pre_registered=False, mllm=True)
        kinds = [i.model_arch == gguf.MODEL_ARCH.MMPROJ for i in created]
        assert kinds == [False, True], "lazy path must create text first, then mmproj"


class TestStreamGgufBlobMtpE2E:
    """End-to-end: checkpoint-only MTP tensors survive a streaming blob export
    as embedded nextn blocks (Qwen3-Next topology)."""

    @pytest.fixture(scope="class")
    def tiny_qwen3next_ckpt(self, tmp_path_factory):
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

        cfg = Qwen3NextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=8,
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
        )
        torch.manual_seed(12)
        model = Qwen3NextForCausalLM(cfg)
        d = tmp_path_factory.mktemp("tiny_q3n_ckpt")
        from tokenizers import Tokenizer
        from tokenizers import decoders, models as tk_models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(tk_models.BPE(vocab={"[UNK]": 0, "a": 1, "b": 2}, merges=[]))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(str(d))
        model.save_pretrained(str(d), max_shard_size="40KB")

        # append a checkpoint-only mtp.* group: a full clone of decoder layer 0
        # plus the fc mixer and the three norms
        import json
        import os

        from safetensors import safe_open
        from safetensors.torch import save_file

        idx_path = os.path.join(str(d), "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        last = sorted(set(idx["weight_map"].values()))[-1]
        sd = {}
        for shard in sorted(set(idx["weight_map"].values())):
            with safe_open(os.path.join(str(d), shard), framework="pt") as fh:
                sd.update({k: fh.get_tensor(k) for k in fh.keys()})
        h = cfg.hidden_size
        mtp = {}
        for k, v in sd.items():
            if k.startswith("model.layers.0."):
                mtp["mtp." + k[len("model.") :]] = v.clone()
        mtp["mtp.fc.weight"] = torch.randn(h, 2 * h)
        mtp["mtp.pre_fc_norm_embedding.weight"] = torch.randn(h)
        mtp["mtp.pre_fc_norm_hidden.weight"] = torch.randn(h)
        mtp["mtp.norm.weight"] = torch.randn(h)
        with safe_open(os.path.join(str(d), last), framework="pt") as fh:
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        tensors.update(mtp)
        save_file(tensors, os.path.join(str(d), last), metadata={"format": "pt"})
        for k in mtp:
            idx["weight_map"][k] = last
        with open(idx_path, "w") as f:
            json.dump(idx, f)
        return str(d)

    @pytest.mark.timeout(300)
    @pytest.mark.timeout(300)
    def test_nextn_pins_and_kv_in_assembled_gguf(self, tiny_qwen3next_ckpt, tmp_path, monkeypatch):
        """One pinned MTP run covers: nextn presence + kv, and that user pins
        reach the assembled dtypes (merged from the unpinned + pins e2es; the
        unpinned scheme-default dtype is covered by the mmproj e2e body).
        Also verifies the late-materializing regex pin ('.*mtp.*' resolves
        before the predictor tree exists - 27B regression)."""
        gguf = pytest.importorskip("gguf")
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.autoround import AutoRound
        from auto_round.export.export_to_gguf import llama_cpp_conversion as lcc
        from auto_round.export.export_to_gguf.blob_store import GgufBlobStore

        lcc.get_conversion(tiny_qwen3next_ckpt)
        conversion_qwen = importlib.import_module("conversion.qwen")
        monkeypatch.setattr(conversion_qwen.Qwen3NextModel, "get_vocab_base_pre", lambda self, tokenizer: "gpt2")
        GgufBlobStore.reset_singletons()
        out_dir = str(tmp_path / "out")
        ar = AutoRound(
            tiny_qwen3next_ckpt,
            scheme="gguf:q4_0",
            alg_configs=[RTNConfig(group_size=32, disable_opt_rtn=True)],
            layer_config={
                ".*mtp.*": {"bits": 8},
                "lm_head": {"bits": 8},
                "embed_tokens": {"bits": 16, "data_type": "float"},
            },
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
        manifest = json.loads((blobs / "manifest.json").read_text())
        entries = manifest["roles"]["text"]["tensors"]
        names = {e["name"] for e in entries}
        for want in ("blk.2.nextn.eh_proj.weight", "blk.2.nextn.enorm.weight", "blk.2.nextn.hnorm.weight"):
            assert want in names, f"{want} missing from blob tensors; got {sorted(n for n in names if 'nextn' in n)}"
        cloned = {n for n in names if n.startswith("blk.2.") and ".nextn." not in n}
        assert len(cloned) >= 5, f"cloned MTP decoder-layer tensors missing: {sorted(cloned)[:5]}"

        out_path = Path(manifest["roles"]["text"]["out_path"])
        assert out_path.is_file(), f"assembled .gguf missing at {out_path}"
        reader = gguf.GGUFReader(str(out_path))
        kv_names = {str(k) for k in reader.fields}
        nextn_kv = [k for k in kv_names if "nextn_predict_layers" in k]
        assert nextn_kv, f"nextn_predict_layers kv missing; fields: {sorted(kv_names)[:8]}"
        field = reader.fields[nextn_kv[0]]
        val = field.parts[-1].view(np.int32)[0]
        assert int(val) == 1
        dtypes = {t.name: t.tensor_type for t in reader.tensors}
        assert dtypes["output.weight"] == gguf.GGMLQuantizationType.Q8_0, "lm_head bits-8 pin did not reach the export"
        assert (
            dtypes["blk.2.nextn.eh_proj.weight"] == gguf.GGMLQuantizationType.Q8_0
        ), ".*mtp.* bits-8 pin did not reach the nextn tensors"
        cloned_q = [
            n
            for n in dtypes
            if n.startswith("blk.2.")
            and ".nextn." not in n
            and "norm" not in n
            and "ssm_" not in n
            and "ffn_gate_inp" not in n
            and not n.endswith(".bias")
        ]
        assert cloned_q, "no cloned MTP layer tensors in the artifact"
        assert all(
            dtypes[n] == gguf.GGMLQuantizationType.Q8_0 for n in cloned_q
        ), f"cloned MTP layer tensors not Q8_0: {[(n, dtypes[n].name) for n in cloned_q[:4]]}"
        assert dtypes["token_embd.weight"] == gguf.GGMLQuantizationType.F32, "embed float pin did not keep F32"
        assert manifest["roles"]["text"]["kv_ops"], "metadata must be captured for replay"


@pytest.mark.slow
class TestStreamGgufBlobMmprojE2E:
    """End-to-end MLLM blob run: crash mid-quantize, resume, and assert BOTH
    roles assemble with their roots and pins intact (merged: two-role mmproj
    e2e + the former llama crash-resume e2e)."""

    @pytest.fixture(scope="class")
    def vl_ckpt(self, tmp_path_factory):
        # depth 3: the patch-merger Sequential ends at mlp.2, which needs a
        # vision block count >= 3 to resolve in the gguf tensor map (real
        # models have depth >= 27; the shared 2-layer fixture cannot map it)
        import shutil
        import sys

        sys.path.insert(0, "test")
        from helpers import qwen_2_5_vl_name_or_path, save_tiny_model

        d = tmp_path_factory.mktemp("tiny_q25vl_blob")
        p = save_tiny_model(qwen_2_5_vl_name_or_path, str(d / "model"), num_layers=3, is_mllm=True)
        yield p
        shutil.rmtree(p, ignore_errors=True)

    @pytest.mark.timeout(900)
    def test_stream_quantize_assembles_mmproj(self, vl_ckpt, tmp_path, monkeypatch):
        # qwen2_vl checkpoints ship flat pre-nesting keys; qwen2_5_vl ships
        # the same flat layout, so both need the streamer's family rewrite
        gguf = pytest.importorskip("gguf")
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.autoround import AutoRound
        from auto_round.export.export_to_gguf import llama_cpp_conversion as lcc
        from auto_round.export.export_to_gguf.blob_store import GgufBlobStore

        lcc.get_conversion(vl_ckpt)
        conversion_qwenvl = importlib.import_module("conversion.qwenvl")
        monkeypatch.setattr(conversion_qwenvl.Qwen2VLModel, "get_vocab_base_pre", lambda self, tokenizer: "gpt2")
        from unittest import mock

        from auto_round.utils.resume import ResumeState

        monkeypatch.setenv("AR_RESUME_DIR", str(tmp_path / "resume"))
        # the VL fixture ties its embeddings (no lm_head tensor in the
        # checkpoint): pin only the embed float here; the lm_head Q8 pin is
        # covered by the nextn e2e on the untied fixture
        pins = {
            "embed_tokens": {"bits": 16, "data_type": "float"},
        }

        def _run(out_dir):
            # a resumed process starts with fresh conversion state; the first
            # run dies mid-quantize, so the instance global must be cleared
            # between in-process runs
            import auto_round.export.export_to_gguf.export as gguf_export

            gguf_export._clear_gguf_model_instances()
            GgufBlobStore.reset_singletons()
            ar = AutoRound(
                vl_ckpt,
                scheme="gguf:q4_0",
                alg_configs=[RTNConfig(group_size=32, disable_opt_rtn=True)],
                layer_config=pins,
                quant_nontext_module=False,
                stream_quantization=True,
                stream_prefetch="off",
                format="gguf:q4_0",
                disable_model_free=True,
                device_map="cpu",
                low_gpu_mem_usage=True,
                low_cpu_mem_usage=True,
            )
            ar.quantize_and_save(out_dir, format="gguf:q4_0")
            return ar

        # phase 1: crash after the first durable block mark - the resumed run
        # must not drop outside-block tensors (27B regression) nor either role
        original_mark = ResumeState.mark_block_done
        crashed = []

        def crash_after_first(self, block_name, q_input, input_ids):
            original_mark(self, block_name, q_input, input_ids)
            crashed.append(block_name)
            if len(crashed) == 1:
                raise RuntimeError("simulated crash")

        with mock.patch.object(ResumeState, "mark_block_done", crash_after_first):
            with pytest.raises(RuntimeError, match="simulated crash"):
                _run(str(tmp_path / "out1"))
        assert crashed, "crash injection never fired"

        ar = _run(str(tmp_path / "out2"))

        blobs = Path(ar.output_dir) / "gguf-blobs"
        manifest = json.loads((blobs / "manifest.json").read_text())
        roles = manifest["roles"]
        assert "mmproj" in roles, f"mmproj role missing from manifest: {sorted(roles)}"
        assert roles["mmproj"]["tensors"], "mmproj role recorded no tensors"
        for role in ("text", "mmproj"):
            out_path = Path(roles[role]["out_path"])
            assert out_path.is_file(), f"assembled {role} gguf missing at {out_path}"
            reader = gguf.GGUFReader(str(out_path))
            manifest_names = {e["name"] for e in roles[role]["tensors"]}
            reader_names = {t.name for t in reader.tensors}
            assert reader_names == manifest_names, f"{role}: assembled tensors must match the manifest"
        mmproj_reader = gguf.GGUFReader(str(Path(roles["mmproj"]["out_path"])))
        # gguf names vision-tower tensors v.* and merger tensors mm.*
        assert any(
            t.name.startswith("v.") for t in mmproj_reader.tensors
        ), "mmproj gguf carries no vision-tower tensors"
        assert any(t.name.startswith("mm.") for t in mmproj_reader.tensors), "no merger tensors"
        # resumed-run contract (folded from the llama crash-resume e2e):
        # outside-block roots survive the resume and the pins hold
        text_reader = gguf.GGUFReader(str(Path(roles["text"]["out_path"])))
        text_dtypes = {t.name: t.tensor_type for t in text_reader.tensors}
        assert "token_embd.weight" in text_dtypes, "resumed run dropped the embedding"
        # the VL fixture ties its embeddings: output.weight is intentionally
        # absent from the container (tied models read token_embd); the untied
        # lm_head root is covered by the nextn e2e
        assert "output_norm.weight" in text_dtypes, "resumed run dropped the final norm"
        assert text_dtypes["token_embd.weight"] == gguf.GGMLQuantizationType.F32, "embed float pin"
        body_q = [
            n for n in text_dtypes if n.startswith("blk.") and "norm" not in n and not n.endswith(".bias")
        ]
        assert body_q and all(
            text_dtypes[n] == gguf.GGMLQuantizationType.Q4_0 for n in body_q
        ), "unpinned body lost the scheme default"


class TestMoeImatrixContract:
    """The per-source imatrix contract only matters for imatrix-aware quants."""

    def test_moe_imatrix_required_by_qtype(self):
        import gguf as gguf_pkg

        from auto_round.export.export_to_gguf.moe_adapter import moe_imatrix_required

        assert moe_imatrix_required(gguf_pkg.GGMLQuantizationType.IQ4_XS)
        assert moe_imatrix_required(gguf_pkg.GGMLQuantizationType.IQ2_XXS)
        # K-quants feed the imatrix into their double-quant search
        assert moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q2_K)
        assert moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q4_K)
        assert moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q6_K)
        assert not moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q4_0)
        assert not moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q8_0)
        assert not moe_imatrix_required(gguf_pkg.GGMLQuantizationType.Q8_K)


class TestFloatPinKeepsTensorUnquantized:
    """A 16-bit (float) layer_config pin must keep a GGUF tensor unquantized
    instead of applying the file-type default, in the float type that is exact
    for the source dtype (bf16 -> BF16, fp16 -> F16, fp32 -> F32)."""

    def test_float_pin_maps_by_source_dtype(self):
        import gguf as gguf_pkg

        from auto_round.export.export_to_gguf.convert import get_qtype_by_layer_config

        cfg = {"embed_tokens": {"bits": 16, "data_type": "float"}, "lm_head": {"bits": 8}}
        for fallback in (gguf_pkg.GGMLQuantizationType.Q4_0, gguf_pkg.GGMLQuantizationType.Q6_K):
            got = get_qtype_by_layer_config(
                cfg, "embed_tokens.weight", fallback, explicit_only=True, source_dtype=torch.bfloat16
            )
            assert got == gguf_pkg.GGMLQuantizationType.BF16, (fallback, got)
        assert (
            get_qtype_by_layer_config(
                cfg,
                "embed_tokens.weight",
                gguf_pkg.GGMLQuantizationType.Q6_K,
                explicit_only=True,
                source_dtype=torch.float16,
            )
            == gguf_pkg.GGMLQuantizationType.F16
        )
        assert (
            get_qtype_by_layer_config(
                cfg,
                "embed_tokens.weight",
                gguf_pkg.GGMLQuantizationType.Q6_K,
                explicit_only=True,
                source_dtype=torch.float32,
            )
            == gguf_pkg.GGMLQuantizationType.F32
        )
        # unknown dtype stays on llama.cpp's canonical F16
        assert (
            get_qtype_by_layer_config(
                cfg, "embed_tokens.weight", gguf_pkg.GGMLQuantizationType.Q6_K, explicit_only=True
            )
            == gguf_pkg.GGMLQuantizationType.F16
        )
        assert get_qtype_by_layer_config(cfg, "lm_head.weight", gguf_pkg.GGMLQuantizationType.Q4_0) == (
            gguf_pkg.GGMLQuantizationType.Q8_0
        )

    def test_float_pin_wins_through_resolve_restored_qtype(self):
        import gguf as gguf_pkg

        from auto_round.export.export_to_gguf.convert import resolve_restored_qtype

        cfg = {"model.embed_tokens": {"bits": 16, "data_type": "float"}}
        for dtype, want in (
            (torch.bfloat16, gguf_pkg.GGMLQuantizationType.BF16),
            (None, gguf_pkg.GGMLQuantizationType.F16),
        ):
            got = resolve_restored_qtype(
                cfg,
                ("model.embed_tokens.weight",),
                "model.embed_tokens.weight",
                "token_embd.weight",
                gguf_pkg.GGMLQuantizationType.Q6_K,  # the large-tensor mix default
                [],
                allow_recipe_fallback=True,
                source_dtype=dtype,
            )
            assert got == want, (dtype, got)


class TestDiscardShardsAfterAssembly:
    """Blob shards are the resumable intermediate: once every role is
    assembled they are removed automatically (manifest kept as the record)."""

    def test_discard_shards_removes_shards_keeps_manifest(self, store, tmp_path):
        import json as _json

        conv = _FakeConversion(fname_out=str(tmp_path / "m.gguf"))
        rec = store.attach_recorder(conv, "text")
        _add_q4_tensor(rec, "blk.0.attn_q.weight", 4)
        store.flush("text")

        assert store.discard_shards() == 1
        assert not list(store.blob_dir.glob("blob-*.safetensors")), "shards must be gone"
        assert (store.blob_dir / "manifest.json").is_file(), "manifest must stay"
        manifest = _json.loads((store.blob_dir / "manifest.json").read_text())
        assert manifest["roles"]["text"]["tensors"], "manifest still records what was assembled"

    def test_save_calls_discard_after_all_roles(self, monkeypatch, tmp_path):
        """The save path discards only AFTER every instance was written."""
        import auto_round.export.export_to_gguf.export as gguf_export
        from auto_round.export.export_to_gguf.config import ModelType

        calls = []

        class _FakeInst:
            model_arch = object()
            fname_out = str(tmp_path / "out")

            def write(self):
                calls.append("write")

        class _FakeStore:
            def __init__(self):
                self.discarded = False

            def finalize_role(self, inst, role):
                calls.append(f"finalize:{role}")

            def assemble(self, roles, progress=False):
                calls.append(f"assemble:{roles[0]}")
                return [tmp_path / f"{roles[0]}.gguf"]

            def discard_shards(self):
                assert calls[-1].startswith("assemble"), calls
                self.discarded = True
                calls.append("discard")

        fake_store = _FakeStore()
        gguf_export.__dict__.pop("gguf_model_instance_global", None)
        monkeypatch.setattr(
            gguf_export,
            "create_model_class",
            lambda *a, **kw: _FakeInst(),
        )
        gguf_export.save_quantized_as_gguf(
            str(tmp_path / "out"), model=object(), layer_config={}, mllm=False, blob_store=fake_store
        )
        assert calls == ["write", "finalize:text", "assemble:text", "discard"], calls
        assert fake_store.discarded
