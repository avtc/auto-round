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
import time

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

    def test_prefetch_serves_cached_tensors(self, tmp_path):
        """Prefetched tensors are served from the host-RAM cache and match disk."""
        torch.manual_seed(2)
        tensors = {
            "blk.a.weight": torch.randn(4, 8),
            "blk.b.weight": torch.randn(4, 4),
            "blk.c.weight": torch.randn(8, 2),
        }
        path = _make_sharded_checkpoint(
            tmp_path, {"model-00001-of-00002.safetensors": {"blk.a.weight": tensors["blk.a.weight"],
                                                       "blk.b.weight": tensors["blk.b.weight"]},
                       "model-00002-of-00002.safetensors": {"blk.c.weight": tensors["blk.c.weight"]}}
        )
        streamer = CheckpointStreamer(path)
        streamer.start_prefetch(["blk"], depth=1)
        try:
            deadline = time.monotonic() + 10.0
            while streamer.prefetch_pending() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not streamer.prefetch_pending(), "prefetch did not finish"
            for name, ref in tensors.items():
                assert torch.equal(streamer.fetch(name), ref)
            assert not streamer._prefetch_cache  # fully consumed
        finally:
            streamer.stop_prefetch()
        # after stop, plain disk reads still work
        assert torch.equal(streamer.fetch("blk.a.weight"), tensors["blk.a.weight"])

    def test_prefetch_depth_and_consumption(self, tmp_path):
        """The reader keeps at most ``depth`` prefixes staged ahead and releases
        a slot only after the consumer reports the prefix consumed."""
        torch.manual_seed(3)
        shards = {}
        for i in range(4):
            shards[f"model-{i:05d}.safetensors"] = {f"b{i}.w.weight": torch.randn(2, 2)}
        path = _make_sharded_checkpoint(tmp_path, shards)
        streamer = CheckpointStreamer(path)
        streamer.start_prefetch(["b0", "b1", "b2", "b3"], depth=1)

        def wait_staged(prefix):
            deadline = time.monotonic() + 10.0
            while prefix not in streamer._prefetch_staged and time.monotonic() < deadline:
                time.sleep(0.01)
            assert prefix in streamer._prefetch_staged, f"{prefix} never staged"

        try:
            wait_staged("b0")
            time.sleep(0.2)  # a wrongly-unbounded reader would stage b1..b3 now
            assert streamer._prefetch_staged == ["b0"], streamer._prefetch_staged
            assert torch.equal(streamer.fetch("b0.w.weight"), shards["model-00000.safetensors"]["b0.w.weight"])
            streamer.prefetch_consumed("b0")
            for p in ("b1", "b2", "b3"):
                wait_staged(p)
                streamer.prefetch_consumed(p)
            assert not streamer._prefetch_remaining and not streamer._prefetch_staged
            assert not streamer._prefetch_cache
        finally:
            streamer.stop_prefetch()

    def test_prefetch_error_surfaces(self, tmp_path):
        """A reader failure is re-raised at the next fetch."""
        path = _make_sharded_checkpoint(tmp_path, {"model.safetensors": {"m.w.weight": torch.randn(2, 2)}})
        streamer = CheckpointStreamer(path)
        streamer.start_prefetch(["missing"], depth=1)
        try:
            deadline = time.monotonic() + 10.0
            while streamer.prefetch_error() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert streamer.prefetch_error() is not None, "reader failure not recorded"
            with pytest.raises(RuntimeError, match="prefetch"):
                streamer.fetch("m.w.weight")
        finally:
            streamer.stop_prefetch()

    def test_prefetch_stages_to_devices(self, tmp_path):
        """With stage_devices the reader lands each prefix on its assigned
        device (round-robin by prefix index) and fetch() moves tensors back to
        whatever device the consumer asks for."""
        torch.manual_seed(4)
        tensors = {f"b{i}.w.weight": torch.randn(3, 3) for i in range(4)}
        shards = {f"model-{i:05d}.safetensors": {f"b{i}.w.weight": tensors[f"b{i}.w.weight"]} for i in range(4)}
        path = _make_sharded_checkpoint(tmp_path, shards)
        stage = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        streamer = CheckpointStreamer(path)
        streamer.start_prefetch(["b0", "b1", "b2", "b3"], depth=4, stage_devices=[stage])
        try:
            deadline = time.monotonic() + 10.0
            while len(streamer._prefetch_staged) < 4 and streamer.prefetch_error() is None \
                    and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(streamer._prefetch_staged) == 4, streamer._prefetch_staged
            for name in list(streamer._prefetch_cache):
                assert streamer._prefetch_cache[name].device == stage
            # consumer on a different device gets a move, not an error
            target = torch.device("cpu") if stage.type == "cuda" else None
            for name, ref in tensors.items():
                got = streamer.fetch(name, device=target)
                assert got.device == (target or stage)
                assert torch.equal(got.to("cpu"), ref)
        finally:
            streamer.stop_prefetch()

    def test_prefetch_invalid_stage_device_raises(self, tmp_path):
        """Meta staging devices are rejected eagerly: staging to meta would
        silently produce empty tensors instead of an error."""
        path = _make_sharded_checkpoint(tmp_path, {"model.safetensors": {"m.w.weight": torch.randn(2, 2)}})
        streamer = CheckpointStreamer(path)
        with pytest.raises(ValueError, match="meta"):
            streamer.start_prefetch(["m"], depth=1, stage_devices=[torch.device("meta")])
        # no reader thread left behind
        assert streamer._prefetch_thread is None

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
    def _quantize(
        model_path,
        out_dir,
        stream,
        stream_prefetch=0,
        stream_prefetch_gpus=None,
        stream_calibration=False,
        dataset=None,
    ):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.presinq import PreSINQConfig
        from auto_round.compressors.entry import AutoRound

        kwargs = {}
        if dataset is not None:
            kwargs["dataset"] = dataset
            kwargs["seqlen"] = 32
            kwargs["nsamples"] = 8
        ar = AutoRound(
            model_path,
            scheme="W4A16",
            alg_configs=[PreSINQConfig(group_size=16, n_iter=2, n_repeat=1), RTNConfig(group_size=16, disable_opt_rtn=False)],
            layerwise_rotation=True,
            stream_checkpoint=stream,
            stream_prefetch=stream_prefetch,
            stream_prefetch_gpus=stream_prefetch_gpus,
            stream_calibration=stream_calibration,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        ar.quantize_and_save(out_dir, format="auto_round")
        return ar.output_dir  # resolved export dir (may add a name/scheme suffix)

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

    def test_stream_calibration_matches_data_driven(self, tiny_checkpoint, tmp_path):
        """stream_calibration=True must reproduce the data-driven run exactly:
        the streaming pass forwards the same rows through the same block chain
        (same attention-mask rule, row-by-row) so the imatrix statistics — and
        therefore the searched quantized weights — are identical."""
        import shutil

        torch.manual_seed(7)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(8)]  # vocab_size=64
        ck_a = str(tmp_path / "ck_a")
        ck_b = str(tmp_path / "ck_b")
        shutil.copytree(tiny_checkpoint, ck_a)
        shutil.copytree(tiny_checkpoint, ck_b)
        data_driven = self._quantize(ck_a, str(tmp_path / "a"), stream=False, dataset=rows)
        streamed_calib = self._quantize(
            ck_b, str(tmp_path / "b"), stream=True, stream_calibration=True, dataset=rows
        )

        def load_all(d):
            from safetensors import safe_open

            out = {}
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(d, fn), framework="pt") as f:
                    for k in f.keys():
                        out[k] = f.get_tensor(k)
            return out

        t_a, t_b = load_all(data_driven), load_all(streamed_calib)
        assert set(t_a) == set(t_b), (
            f"tensor name mismatch: only-a={set(t_a) - set(t_b)} only-b={set(t_b) - set(t_a)}"
        )
        n_exact, n_close, n_diff = 0, 0, 0
        for k in t_a:
            if torch.equal(t_a[k], t_b[k]):
                n_exact += 1
            elif torch.allclose(t_a[k].float(), t_b[k].float(), atol=1e-6):
                n_close += 1
            else:
                n_diff += 1
        assert n_diff == 0, f"{n_diff} tensors differ beyond tolerance"
        assert n_exact + n_close == len(t_a)

    def test_partial_layer_config_resolves_format(self, tiny_checkpoint, tmp_path):
        """layer_config entries are partial overrides: unset keys fall back to
        the global scheme. Format resolution must not assume every entry
        carries the full key set — asym scheme (RTNConfig sym=False, the NeUQI
        path) hits the AutoAWQ enablement check in formats.py, which raised
        KeyError('bits') on partial entries (hy3 NeUQI-asym server run)."""
        import shutil

        ck = str(tmp_path / "ck_partial")
        shutil.copytree(tiny_checkpoint, ck)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.compressors.entry import AutoRound

        ar = AutoRound(
            ck,
            scheme="W4A16",
            alg_configs=[RTNConfig(group_size=8, sym=False)],  # asym global scheme (NeUQI path)
            layer_config={
                ".*model.layers.0.": {"bits": 16, "data_type": "float"},
                ".*shared_mlp": {"group_size": 8},  # partial: no bits key
                ".*experts": {"group_size": 8},
            },
            layerwise_rotation=True,
            stream_checkpoint=True,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        ar.post_init()  # resolves formats; raised KeyError('bits') before the fix
        assert ar.formats is not None

    def test_prefetched_export_matches_streamed(self, tiny_checkpoint, tmp_path):
        """stream_prefetch only changes WHERE the tensors come from (host-RAM
        cache instead of a synchronous disk read); the exported checkpoint must
        stay bit-identical to the un-prefetched streaming run."""
        import shutil

        ck_a = str(tmp_path / "ck_a")
        ck_b = str(tmp_path / "ck_b")
        shutil.copytree(tiny_checkpoint, ck_a)
        shutil.copytree(tiny_checkpoint, ck_b)
        plain = self._quantize(ck_a, str(tmp_path / "plain"), stream=True)
        prefetched = self._quantize(ck_b, str(tmp_path / "prefetched"), stream=True, stream_prefetch=1)

        def load_all(d):
            from safetensors import safe_open

            out = {}
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(d, fn), framework="pt") as f:
                    for k in f.keys():
                        out[k] = f.get_tensor(k)
            return out

        t_plain, t_prefetched = load_all(plain), load_all(prefetched)
        assert set(t_plain) == set(t_prefetched)
        for k in t_plain:
            assert torch.equal(t_plain[k], t_prefetched[k]), f"tensor {k} differs under prefetch"

    def test_staged_export_matches_streamed(self, tiny_checkpoint, tmp_path):
        """Device staging (round-robin block homes + tensors preloaded onto the
        staging devices) must not change any exported bit: the quantization
        math is device-independent, staging only changes where tensors wait.
        Exercises the full home-rotation plumbing via CPU staging devices; the
        multi-GPU variant runs on CUDA hosts."""
        import shutil

        ck_a = str(tmp_path / "ck_a")
        ck_b = str(tmp_path / "ck_b")
        shutil.copytree(tiny_checkpoint, ck_a)
        shutil.copytree(tiny_checkpoint, ck_b)
        stage_devs = ["cpu"] if torch.cuda.device_count() < 2 else ["cuda:1"]
        plain = self._quantize(ck_a, str(tmp_path / "plain"), stream=True)
        staged = self._quantize(
            ck_b, str(tmp_path / "staged"), stream=True, stream_prefetch=2, stream_prefetch_gpus=stage_devs
        )

        def load_all(d):
            from safetensors import safe_open

            out = {}
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".safetensors"):
                    continue
                with safe_open(os.path.join(d, fn), framework="pt") as f:
                    for k in f.keys():
                        out[k] = f.get_tensor(k)
            return out

        t_plain, t_staged = load_all(plain), load_all(staged)
        assert set(t_plain) == set(t_staged)
        for k in t_plain:
            assert torch.equal(t_plain[k], t_staged[k]), f"tensor {k} differs under device staging"

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
        from auto_round.compressors.entry import AutoRound

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
