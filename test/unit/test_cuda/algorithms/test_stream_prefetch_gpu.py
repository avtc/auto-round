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
"""GPU-staging parity for stream_quantization: prefetch/auto must not change a
single exported bit versus the plain streamed run (staging only decides WHERE
the next block waits - a GPU home or the rotation - never values).

On CPU-only boxes the CPU-tier family cannot exercise this: prefetch-auto with
a CPU quant device degrades to host-RAM staging, the same destination as the
explicit ``staged`` arm there. Only a CUDA quant device walks the GPU staging
chain. Single-GPU agents still cover real GPU staging: the sole-GPU branch
stages on the quant device itself when the largest block fits (host RAM is the
last resort, not the single-GPU default); the cross-device rotation
([quant_dev, cuda:1]) additionally engages on multi-GPU agents.
"""

import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


def _make_tiny_checkpoint(tmpdir):
    """Tiny sharded causal LM checkpoint, fully local (no network access)."""
    from tokenizers import Tokenizer
    from tokenizers import models as tk_models
    from tokenizers import pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    torch.manual_seed(7)
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                outlier_mask = torch.rand_like(p) < 0.02
                p.mul_(0.1).add_(outlier_mask.float() * torch.randn_like(p))
    d = str(tmpdir)
    tk = Tokenizer(tk_models.WordLevel(vocab={"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(d)
    model.save_pretrained(d, max_shard_size="40KB")
    return d


def _quantize(model_path, out_dir, stream_prefetch):
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    ar = AutoRound(
        model_path,
        scheme="W4A16",
        alg_configs=[RTNConfig(group_size=16)],
        stream_quantization=True,
        stream_prefetch=stream_prefetch,
        format="auto_round",
        disable_model_free=True,
        device_map="cuda:0",
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
    )
    ar.quantize_and_save(out_dir, format="auto_round")
    return ar.output_dir


def _load_all(d):
    from safetensors import safe_open

    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(d, fn), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


@pytest.mark.slow
class TestStreamPrefetchGpuEquivalence:
    @pytest.fixture(scope="class")
    def tiny_checkpoint(self, tmp_path_factory):
        return _make_tiny_checkpoint(tmp_path_factory.mktemp("tiny_gpu_ckpt"))

    def test_prefetch_auto_bit_identical_to_plain(self, tiny_checkpoint, tmp_path):
        import shutil

        arms = {}
        for name, sp in (("plain", "off"), ("auto", "auto")):
            ck = str(tmp_path / f"ck_{name}")
            shutil.copytree(tiny_checkpoint, ck)
            arms[name] = _quantize(ck, str(tmp_path / name), sp)

        t = {name: _load_all(d) for name, d in arms.items()}
        assert set(t["plain"]) == set(t["auto"]), "prefetch changed the tensor inventory"
        for k in t["plain"]:
            assert torch.equal(t["plain"][k], t["auto"][k]), f"tensor {k} differs under prefetch/auto"
        with open(os.path.join(arms["plain"], "quantization_config.json")) as f:
            cp = json.load(f)
        with open(os.path.join(arms["auto"], "quantization_config.json")) as f:
            ca = json.load(f)
        assert cp == ca, "prefetch changed the quantization config"
