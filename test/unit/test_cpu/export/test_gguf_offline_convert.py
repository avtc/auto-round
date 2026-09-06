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

"""Offline GGUF conversion from an exported checkpoint directory (Route B).

stream_quantization exports compressed-tensors packed shards; the bundled
llama.cpp conversion reads such a directory lazily and unpacks the packed
weights itself, so the offline pass is a thin, streaming-free bridge.
"""

import json
import os

import pytest

gguf = pytest.importorskip("gguf", reason="gguf package required for offline conversion tests")


def _convert(ckpt, out, fmt):
    from auto_round.export.export_to_gguf.offline_convert import convert_checkpoint_to_gguf

    return convert_checkpoint_to_gguf(ckpt, output_dir=out, gguf_format=fmt)


def _quantize_to_ct(tiny_qwen_model_path, export_dir):
    from auto_round import AutoRound

    AutoRound(
        tiny_qwen_model_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=0,
        data_type="int",
        nsamples=1,
        seqlen=8,
    ).quantize_and_save(output_dir=export_dir, inplace=False, format="auto_round:llm_compressor")
    # quantize_and_save nests per-format output in a subdirectory
    return os.path.join(export_dir, os.listdir(export_dir)[0])


class TestConvertArgumentValidation:
    def test_missing_config_json_raises(self, tmp_path):
        from auto_round.export.export_to_gguf.offline_convert import convert_checkpoint_to_gguf

        with pytest.raises(ValueError, match="no config.json"):
            convert_checkpoint_to_gguf(str(tmp_path), output_dir=str(tmp_path / "out"))

    def test_unsupported_format_raises(self, tmp_path, tiny_qwen_model_path):
        from auto_round.export.export_to_gguf.offline_convert import convert_checkpoint_to_gguf

        with pytest.raises(ValueError, match="unsupported gguf format"):
            convert_checkpoint_to_gguf(tiny_qwen_model_path, output_dir=str(tmp_path / "out"), gguf_format="gguf:q9_9")

    def test_auto_format_raises(self, tmp_path, tiny_qwen_model_path):
        """gguf:auto passes the ftype table but only a real format is convertible."""
        from auto_round.export.export_to_gguf.offline_convert import convert_checkpoint_to_gguf

        with pytest.raises(ValueError, match="unsupported gguf format"):
            convert_checkpoint_to_gguf(tiny_qwen_model_path, output_dir=str(tmp_path / "out"), gguf_format="gguf:auto")


class TestConvertCli:
    def test_dispatch_and_parser(self, tmp_path, monkeypatch):
        from auto_round.cli import main as cli_main

        parser = cli_main.build_convert_parser(prog="auto_round convert")
        args = parser.parse_args(["--model", "some_dir", "--format", "gguf:q4_0", "--output_dir", "out"])
        assert args.model == "some_dir" and args.format == "gguf:q4_0" and args.output_dir == "out"

        captured = {}

        def fake_convert(ckpt, output_dir, gguf_format):
            captured.update(ckpt=ckpt, output_dir=output_dir, gguf_format=gguf_format)
            return "done.gguf"

        import auto_round.export.export_to_gguf.offline_convert as oc

        monkeypatch.setattr(oc, "convert_checkpoint_to_gguf", fake_convert)
        cli_main.run_convert(["--model", "some_dir", "--format", "gguf:q8_0", "--output_dir", "out"])
        assert captured == {"ckpt": "some_dir", "output_dir": "out", "gguf_format": "gguf:q8_0"}

    def test_normalize_routes_convert(self):
        from auto_round.cli.main import _normalize_cli_invocation

        assert _normalize_cli_invocation(["convert", "--model", "x"])[0] == "convert"


def _ct_required():
    """compressed-tensors exports are the primary input; skip e2e without it."""
    pytest.importorskip("compressed_tensors", reason="compressed-tensors required for CT export e2e")


@pytest.mark.slow
class TestOfflineConvertEndToEnd:
    def test_convert_llm_compressor_export(self, tmp_path, tiny_qwen_model_path):
        _ct_required()
        export_dir = _quantize_to_ct(tiny_qwen_model_path, str(tmp_path / "ct_export"))
        assert any(f.endswith(".safetensors") for f in os.listdir(export_dir))

        path = _convert(export_dir, str(tmp_path / "gguf"), "gguf:q4_0")
        assert path.endswith(".gguf") and os.path.isfile(path)

        reader = gguf.GGUFReader(path)
        names = [tensor.name for tensor in reader.tensors]
        assert any(n.startswith("blk.") for n in names)
        dtypes = {t.tensor_type for t in reader.tensors}
        assert gguf.GGMLQuantizationType.Q4_0 in dtypes

    def test_convert_official_mixed_format(self, tmp_path, tiny_qwen_model_path):
        _ct_required()
        export_dir = _quantize_to_ct(tiny_qwen_model_path, str(tmp_path / "ct_export"))
        path = _convert(export_dir, str(tmp_path / "gguf"), "gguf:q4_k_m")
        reader = gguf.GGUFReader(path)
        # mixed formats keep per-tensor types: float norms plus at least two
        # distinct quantized types (the official mix boosts attention-v and
        # ffn-down tensors)
        float_types = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}
        quantized = {t.tensor_type for t in reader.tensors} - float_types
        assert gguf.GGMLQuantizationType.F32 in {t.tensor_type for t in reader.tensors}
        assert len(quantized) >= 2, f"official mixed should select several quantized types, got {quantized}"

    def test_convert_f16_stores_genuine_float16(self, tmp_path, tiny_qwen_model_path):
        """Float passthrough must CAST: f32 bytes tagged F16 would corrupt the file."""
        _ct_required()
        path = _convert(tiny_qwen_model_path, str(tmp_path / "gguf"), "gguf:f16")
        reader = gguf.GGUFReader(path)
        two_d = [t for t in reader.tensors if len(t.shape) == 2]
        assert two_d, "expected 2D tensors in the file"
        assert all(t.tensor_type == gguf.GGMLQuantizationType.F16 for t in two_d)

    def test_convert_bf16_checkpoint_roundtrips(self, tmp_path, tiny_qwen_model_path):
        """bf16 delegates to the reference float-storage implementation (the
        torch bf16 packer does not run on CPU)."""
        path = _convert(tiny_qwen_model_path, str(tmp_path / "gguf"), "gguf:bf16")
        reader = gguf.GGUFReader(path)
        two_d = [t for t in reader.tensors if len(t.shape) == 2]
        assert two_d
        assert all(t.tensor_type == gguf.GGMLQuantizationType.BF16 for t in two_d)
        # spot-check one tensor decodes to a sane value
        import numpy as np

        t = two_d[0]
        raw = np.frombuffer(t.data, dtype=np.uint8).reshape(-1, 2)
        bits = (raw[:, 0].astype(np.uint16) | (raw[:, 1].astype(np.uint16) << 8)).astype(np.uint16)
        vals = (bits.astype(np.uint32) << 16).view(np.float32)
        assert np.isfinite(vals).all()

    def test_convert_is_deterministic(self, tmp_path, tiny_qwen_model_path):
        _ct_required()
        export_dir = _quantize_to_ct(tiny_qwen_model_path, str(tmp_path / "ct_export"))

        a = _convert(export_dir, str(tmp_path / "g1"), "gguf:q4_0")
        b = _convert(export_dir, str(tmp_path / "g2"), "gguf:q4_0")
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()

    def test_convert_plain_bf16_checkpoint(self, tmp_path, tiny_qwen_model_path):
        path = _convert(tiny_qwen_model_path, str(tmp_path / "gguf"), "gguf:q4_0")
        reader = gguf.GGUFReader(path)
        assert len(reader.tensors) > 0
