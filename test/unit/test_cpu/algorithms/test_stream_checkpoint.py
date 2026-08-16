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
"""Tests for stream_checkpoint: meta-device model + per-block tensor streaming."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from auto_round.utils.checkpoint_streamer import CheckpointStreamer


def _make_sharded_checkpoint(tmpdir, tensors_by_shard):
    """Write safetensors shards + index; returns the directory path."""
    weight_map = {}
    for i, (shard_name, tensors) in enumerate(tensors_by_shard.items()):
        save_file(tensors, os.path.join(tmpdir, shard_name), metadata={"format": "pt"})
        for k in tensors:
            weight_map[k] = shard_name
    with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return str(tmpdir)


class TestCheckpointStreamer:
    def test_fetch_and_load_module(self, tmp_path):
        torch.manual_seed(0)
        a = torch.randn(4, 8)
        b = torch.randn(4, 4)
        c = torch.randn(8, 2)
        path = _make_sharded_checkpoint(
            tmp_path, {"model-00001-of-00002.safetensors": {"blk.a.weight": a, "blk.b.weight": b},
                       "model-00002-of-00002.safetensors": {"blk.c.weight": c}}
        )
        streamer = CheckpointStreamer(path)
        assert set(streamer.tensor_names) == {"blk.a.weight", "blk.b.weight", "blk.c.weight"}
        assert streamer.names_under("blk") == ["blk.a.weight", "blk.b.weight", "blk.c.weight"]
        assert streamer.names_under("blk.a") == ["blk.a.weight"]
        assert torch.equal(streamer.fetch("blk.c.weight"), c)

        import torch.nn as nn

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4, bias=False)
                self.b = nn.Linear(4, 4, bias=False)
                self.c = nn.Linear(2, 8, bias=False)

        with torch.device("meta"):
            blk = Blk()
        loaded = streamer.load_module_(blk, "blk")
        assert set(loaded) == {"blk.a.weight", "blk.b.weight", "blk.c.weight"}
        assert torch.equal(blk.a.weight.data, a) and blk.a.weight.device.type == "cpu"

    def test_load_module_device_and_errors(self, tmp_path):
        torch.manual_seed(1)
        path = _make_sharded_checkpoint(tmp_path, {"model.safetensors": {"m.w.weight": torch.randn(3, 3)}})
        streamer = CheckpointStreamer(path)
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(3, 3, bias=False)

        with torch.device("meta"):
            m = M()
        streamer.load_module_(m, "m")
        with pytest.raises(ValueError):  # nothing matches
            streamer.load_module_(m, "nonexistent.prefix")
        with pytest.raises(KeyError):
            streamer.fetch("not.a.tensor")
        class Wrong(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(4, 4, bias=False)

        with torch.device("meta"):
            wrong = Wrong()
        streamer2 = CheckpointStreamer(path)
        with pytest.raises(ValueError):  # shape mismatch
            streamer2.load_module_(wrong, "m")

    def test_single_file_checkpoint(self, tmp_path):
        t = {"x.weight": torch.ones(2, 2)}
        save_file(t, os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})
        streamer = CheckpointStreamer(str(tmp_path))
        assert torch.equal(streamer.fetch("x.weight"), t["x.weight"])


@pytest.mark.slow
class TestStreamQuantizeEquivalence:
    """stream_checkpoint=True must produce the same export as the normal flow."""

    @pytest.fixture(scope="class")
    def tiny_checkpoint(self, tmp_path_factory):
        """Tiny local causal LM checkpoint, sharded (no network access)."""
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        )
        torch.manual_seed(7)
        model = LlamaForCausalLM(cfg)
        # fat-tailed weights: guarantees the opt-RTN clip search moves scales
        # away from the min/max default (near-uniform weights would not)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.dim() >= 2:
                    outlier_mask = torch.rand_like(p) < 0.02
                    p.mul_(0.1).add_(outlier_mask.float() * torch.randn_like(p))
        d = tmp_path_factory.mktemp("tiny_ckpt")
        # minimal fast tokenizer (no sentencepiece dependency)
        from tokenizers import Tokenizer, models as tk_models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(tk_models.WordLevel(vocab={"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tk)
        tok.save_pretrained(str(d))
        # max_shard_size forces multiple shards + an index file
        model.save_pretrained(str(d), max_shard_size="40KB")
        return str(d)

    @staticmethod
    def _quantize(model_path, out_dir, stream):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.presinq import PreSINQConfig
        from auto_round.autoround import AutoRound

        ar = AutoRound(
            model_path,
            scheme="W4A16",
            alg_configs=[PreSINQConfig(group_size=16, n_iter=2, n_repeat=1), RTNConfig(group_size=16, disable_opt_rtn=False)],
            layerwise_rotation=True,
            stream_checkpoint=stream,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(out_dir, format="auto_round")
        return ar.output_dir  # resolved export dir (may add a name/scheme suffix)

    @pytest.mark.xfail(
        strict=True,
        reason="requires the layer-wise rotation protocol wiring (per-block transform lifecycle), "
        "which lands in an immediate follow-up commit",
    )

    def test_streamed_export_matches_normal(self, tiny_checkpoint, tmp_path):
        """Streamed zero-shot export must match the normal (data-driven) run on
        everything the streaming mode controls: tensor inventory, all
        non-quantized tensors (incl. Pre-SINQ folds - bit-exact), and configs.
        Quantized layers may differ in packing: the data-driven path passes a
        pile-10k imatrix into the clip search while streamed zero-shot uses a
        pure weight-MSE search (by design - calibration-free)."""
        import shutil

        ck_normal = str(tmp_path / "ck_normal")
        ck_stream = str(tmp_path / "ck_stream")
        shutil.copytree(tiny_checkpoint, ck_normal)
        shutil.copytree(tiny_checkpoint, ck_stream)
        normal = self._quantize(ck_normal, str(tmp_path / "normal"), stream=False)
        streamed = self._quantize(ck_stream, str(tmp_path / "streamed"), stream=True)

        def load_tensors(d):
            from safetensors import safe_open

            out = {}
            idx = os.path.join(d, "model.safetensors.index.json")
            files = []
            if os.path.exists(idx):
                with open(idx) as f:
                    files = sorted(set(json.load(f)["weight_map"].values()))
            else:
                files = ["model.safetensors"]
            for fn in files:
                with safe_open(os.path.join(d, fn), framework="pt") as f:
                    for k in f.keys():
                        out[k] = f.get_tensor(k)
            return out

        t_normal, t_streamed = load_tensors(normal), load_tensors(streamed)
        assert set(t_normal) == set(t_streamed), (
            f"tensor name mismatch: only-normal={set(t_normal) - set(t_streamed)} "
            f"only-streamed={set(t_streamed) - set(t_normal)}"
        )
        quant_suffixes = ("qweight", "qzeros", "scales", "g_idx")
        n_bitexact, n_quant = 0, 0
        for k in t_normal:
            if k.endswith(quant_suffixes):
                n_quant += 1
                continue
            assert torch.equal(t_normal[k], t_streamed[k]), f"non-quantized tensor {k} differs"
            n_bitexact += 1
        assert n_bitexact > 0 and n_quant > 0  # both classes present
        with open(os.path.join(normal, "quantization_config.json")) as f:
            cn = json.load(f)
        with open(os.path.join(streamed, "quantization_config.json")) as f:
            cs = json.load(f)
        assert cn == cs

    @pytest.mark.xfail(
        strict=True,
        reason="requires the layer-wise rotation protocol wiring (per-block transform lifecycle), "
        "which lands in an immediate follow-up commit",
    )

    def test_search_runs_under_streaming(self, tiny_checkpoint, tmp_path):
        """Zero-shot streaming must still run the optimized clip search:
        with fat-tailed weights the searched scales must differ from plain RTN."""
        import shutil

        ck_a = str(tmp_path / "ck_a")
        ck_b = str(tmp_path / "ck_b")
        shutil.copytree(tiny_checkpoint, ck_a)
        shutil.copytree(tiny_checkpoint, ck_b)
        d_opt = self._quantize(ck_a, str(tmp_path / "a"), stream=True)  # disable_opt_rtn=False
        # plain: same call but disable the search
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.presinq import PreSINQConfig
        from auto_round.autoround import AutoRound

        ar = AutoRound(
            ck_b, scheme="W4A16",
            alg_configs=[PreSINQConfig(group_size=16, n_iter=2, n_repeat=1), RTNConfig(group_size=16, disable_opt_rtn=True)],
            layerwise_rotation=True, stream_checkpoint=True, format="auto_round",
            disable_model_free=True, device_map="cpu", low_gpu_mem_usage=True, low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(str(tmp_path / "b"), format="auto_round")
        d_plain = ar.output_dir

        from safetensors import safe_open

        def get_scales(d):
            out = {}
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".safetensors"):
                    with safe_open(os.path.join(d, fn), framework="pt") as f:
                        for k in f.keys():
                            if k.endswith("scales") and "down_proj" in k:
                                out[k] = f.get_tensor(k)
            return out

        so, sp = get_scales(d_opt), get_scales(d_plain)
        assert so and set(so) == set(sp)
        differing = [k for k in so if not torch.equal(so[k], sp[k])]
        assert differing, "opt-RTN clip search did not change any scales under stream_checkpoint"
