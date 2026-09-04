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
"""Tests for stream_quantization: meta-device model + per-block tensor streaming."""

import json
import os
import time
from pathlib import Path

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
            tmp_path,
            {
                "model-00001-of-00002.safetensors": {"blk.a.weight": a, "blk.b.weight": b},
                "model-00002-of-00002.safetensors": {"blk.c.weight": c},
            },
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

    def test_single_file_checkpoint(self, tmp_path):
        t = {"x.weight": torch.ones(2, 2)}
        save_file(t, os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})
        streamer = CheckpointStreamer(str(tmp_path))
        assert torch.equal(streamer.fetch("x.weight"), t["x.weight"])

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
            while (
                len(streamer._prefetch_staged) < 4 and streamer.prefetch_error() is None and time.monotonic() < deadline
            ):
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
            tmp_path,
            {
                "model-00001-of-00002.safetensors": {
                    "blk.a.weight": tensors["blk.a.weight"],
                    "blk.b.weight": tensors["blk.b.weight"],
                },
                "model-00002-of-00002.safetensors": {"blk.c.weight": tensors["blk.c.weight"]},
            },
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

    def test_pick_stage_device_headroom(self, tmp_path, monkeypatch):
        """Staging needs block bytes + search headroom free; picks the first
        round-robin GPU that qualifies, None when none does (caller waits)."""
        import auto_round.utils.checkpoint_streamer as cs

        gpu0, gpu1 = torch.device("cuda:0"), torch.device("cuda:1")
        fake_free = {gpu0: 4 << 30, gpu1: 14 << 30}
        monkeypatch.setattr(cs, "device_free_bytes", lambda d: fake_free.get(torch.device(d)))

        headroom = 4 << 30
        # an 8.5 GiB block needs 12.5 GiB: gpu0 (4) skipped, gpu1 (14) taken
        assert cs.pick_stage_device([gpu0, gpu1], 0, 8 * 1024**3 + (512 << 20), headroom) == gpu1
        # a small block fits gpu0 directly (4 GiB free >= 0 + 4 GiB headroom)
        assert cs.pick_stage_device([gpu0, gpu1], 0, 0, headroom) == gpu0
        # non-CUDA devices report unknown free and are always eligible
        cpu = torch.device("cpu")
        assert cs.pick_stage_device([cpu], 0, 8 << 30, headroom) == cpu
        # all GPUs below block+headroom -> None (no CPU fallback here)
        fake_free[gpu1] = 1 << 30
        assert cs.pick_stage_device([gpu0, gpu1], 0, 8 << 30, headroom) is None

    def test_prefetch_rescue_buffer_capped_at_one(self, tmp_path, monkeypatch):
        """With every GPU below the headroom the reader stages exactly ONE
        block into host RAM (rescue buffer) and waits for VRAM for the rest;
        no OOM, no unbounded RAM staging."""
        import time

        import auto_round.utils.checkpoint_streamer as cs

        torch.manual_seed(3)
        tensors = {"blk.a.weight": torch.randn(4, 4), "blk.b.weight": torch.randn(4, 4)}
        path = _make_sharded_checkpoint(tmp_path, {"model.safetensors": tensors})
        streamer = CheckpointStreamer(path)

        gpu = torch.device("cuda:0")
        state = {"free": 1 << 30}
        monkeypatch.setattr(cs, "device_free_bytes", lambda d: state["free"])
        monkeypatch.setattr(cs.CheckpointStreamer, "_staging_search_headroom", 4 << 30)
        # no CUDA locally: the device CHOICE is what matters, not the transfer
        monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

        streamer.start_prefetch(["blk.a", "blk.b"], depth=2, stage_devices=[gpu])
        try:
            deadline = time.monotonic() + 5
            while streamer._prefetch_staged != ["blk.a"] and time.monotonic() < deadline:
                time.sleep(0.05)
            # blk.a took the rescue slot (host RAM); blk.b must wait for VRAM
            time.sleep(1.0)
            assert streamer._prefetch_staged == ["blk.a"]
            assert streamer.prefetch_error() is None

            state["free"] = 20 << 30  # VRAM frees up
            deadline = time.monotonic() + 5
            while streamer._prefetch_staged != ["blk.a", "blk.b"] and time.monotonic() < deadline:
                time.sleep(0.05)
            assert streamer.prefetch_error() is None
        finally:
            streamer.stop_prefetch()

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

    def test_check_ids_in_vocab(self):
        from auto_round.utils.streaming_calibration import _check_ids_in_vocab

        rows = [torch.tensor([[1, 2, 3]]), torch.tensor([[0, 5, 63]])]
        _check_ids_in_vocab(rows, 64)  # in range: no raise
        bad = [torch.tensor([[1, 2, 150000]])]  # ids from a different model's tokenizer
        with pytest.raises(ValueError, match="different model's tokenizer"):
            _check_ids_in_vocab(bad, 120832)


class TestStageDeviceResolution:
    """stream_prefetch_devices kwarg + back-compat for the old _gpus spelling."""

    def _resolve(self, **attrs):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        stub = SimpleNamespace(stream_prefetch=2, device="cpu", **attrs)
        return CompressionOrchestrator._resolve_stream_stage_devices(stub)

    def test_none_means_host_ram(self):
        assert self._resolve(stream_prefetch_devices=None) is None

    def test_cuda_list_with_cpu_quant_falls_back_to_ram(self):
        # all-GPU staging list with a CPU quant device -> host RAM by design
        devices = self._resolve(stream_prefetch_devices=["cuda:0", "cuda:1"])
        assert devices is None

    def test_mixed_list_raises_even_with_cpu_quant(self):
        with pytest.raises(ValueError, match="mixed GPU/CPU staging"):
            self._resolve(stream_prefetch_devices=["cuda:0", "cpu"])

    def test_cpu_only_list_resolves_to_cpu(self):
        devices = self._resolve(stream_prefetch_devices=["cpu"])
        assert [str(d) for d in devices] == ["cpu"]

    def test_int_list_with_cpu_quant_falls_back_to_ram(self):
        # ints mean cuda:k; GPU staging with a CPU quant device buys nothing
        assert self._resolve(stream_prefetch_devices=[0, 1]) is None

    def test_meta_device_rejected(self):
        with pytest.raises(ValueError, match="meta tensors hold no data"):
            self._resolve(stream_prefetch_devices=["cpu", "meta"])

    def test_on_value_parses(self):
        from auto_round.cli.main import _parse_stream_prefetch

        assert _parse_stream_prefetch("on") == (None, "on")
        assert _parse_stream_prefetch("off") == (0, None)
        assert _parse_stream_prefetch("auto") == (None, "auto")

    def test_on_with_cpu_quant_stays_enabled_as_ram(self):
        # 'on' never silently disables: with a CPU quant device the chain
        # lands on host-RAM staging (devices None) without a warning-only bail
        assert self._resolve(stream_prefetch_devices="on") is None

    def test_auto_with_cpu_quant_is_none(self):
        assert self._resolve(stream_prefetch_devices="auto") is None

    def test_mixed_gpu_cpu_list_rejected_with_cuda_quant(self):
        # in-place round-robin quantization: all-GPU or CPU-only, never mixed
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        stub = SimpleNamespace(stream_prefetch=None, device="cuda:0", stream_prefetch_devices=["cuda:1", "cpu"])
        with pytest.raises(ValueError, match="mixed GPU/CPU staging"):
            CompressionOrchestrator._resolve_stream_stage_devices(stub)


class TestStreamingRoutingWaiver:
    """Calibration-demand resolution under stream_quantization: RTN stays waived
    (weight-only); SignRound (iters>0) is admitted when the chained calibration
    is active (auto-engaged) and falls back to the data-driven path without it."""

    def _needs(self, configs, **attrs):
        from types import SimpleNamespace

        from auto_round.compressors.base import BaseCompressor

        fields = dict(
            quantize_config=object(),
            _alg_configs=configs,
            stream_quantization=False,
            stream_calibration=False,
            scheme="W4A16",
            static_kv_dtype=None,
            static_attention_dtype=None,
            layer_config=None,
        )
        fields["_layer_config_needs_calibration"] = lambda check: False
        fields.update(attrs)
        return BaseCompressor._needs_calibration_data(SimpleNamespace(**fields))

    def test_auto_scheme_streamed_does_not_force_calib(self):
        """AutoScheme scoring runs in workers / from cache -- the streaming
        parent never forwards, so under stream_quantization the scheme itself
        must not force the data-driven path (streaming governs quantization)."""
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        scheme = AutoScheme(avg_bits=3.5, options="W3A16,W4A16")
        assert self._needs([], scheme=scheme, stream_quantization=True) is False

    def test_auto_scheme_non_streamed_still_forces_calib(self):
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        scheme = AutoScheme(avg_bits=3.5, options="W3A16,W4A16")
        assert self._needs([], scheme=scheme, stream_quantization=False) is True

    def test_optimized_rtn_streamed_waived_without_calibration(self):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

        assert self._needs([OptimizedRTNConfig(group_size=16)], stream_quantization=True) is False

    def test_signround_streamed_with_calibration_waived(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        stub = SignRoundConfig(group_size=16, iters=1)
        assert self._needs([stub], stream_quantization=True, stream_calibration=True) is False

    def test_mixed_rtn_signround_streamed_with_calibration_waived(self):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        configs = [OptimizedRTNConfig(group_size=16), SignRoundConfig(group_size=16, iters=1)]
        assert self._needs(configs, stream_quantization=True, stream_calibration=True) is False

    def test_signround_streamed_without_chain_falls_back_to_data_driven(self):
        """Auto-engage normally handles this in __init__; if a SignRound config
        slips in later without the chain, the safe route is the data-driven
        path -- never silent weight-only tuning under streaming."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        assert self._needs([SignRoundConfig(group_size=16, iters=1)], stream_quantization=True) is True

    def test_signround_unstreamed_stays_data_driven(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        assert self._needs([SignRoundConfig(group_size=16, iters=1)]) is True

    def test_unsupported_quantizer_streamed_stays_data_driven(self):
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        assert self._needs([AWQConfig(group_size=16)], stream_quantization=True, stream_calibration=True) is True

    def test_rule_enabled_imatrix_streamed_stays_waived_silently(self):
        """imatrix on by scheme rules (no flag): the pre-existing weight-only
        waiver applies -- no error, imatrix simply not collected."""
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

        assert self._needs([OptimizedRTNConfig(group_size=16)], stream_quantization=True) is False


class TestStreamModeExclusivity:
    """stream_quantization is mutually exclusive with AR_DISK_STREAM_MODEL
    (hard error), and quant_nontext_module is rejected under streaming."""

    def _make_context(self, monkeypatch, *, env_disk_stream=False, mllm=False):
        from auto_round import envs
        from auto_round.context.model import ModelContext

        if env_disk_stream:
            monkeypatch.setenv("AR_DISK_STREAM_MODEL", "1")
        else:
            monkeypatch.delenv("AR_DISK_STREAM_MODEL", raising=False)
        monkeypatch.setattr("auto_round.context.model.is_mllm_model", lambda *a, **k: mllm)
        monkeypatch.setattr("auto_round.context.model.is_diffusion_model", lambda *a, **k: False)
        # BaseContext memoizes instances and skips __init__ on re-construction:
        # reset around the construction so this test builds a fresh
        # ModelContext and never leaks the instance to later tests.
        ModelContext.reset_context()
        try:
            return ModelContext("dummy-model", stream_quantization=True)
        finally:
            ModelContext.reset_context()

    def test_disk_stream_env_plus_stream_quantization_raises(self, monkeypatch):
        import pytest

        with pytest.raises(ValueError, match="mutually exclusive"):
            self._make_context(monkeypatch, env_disk_stream=True)

    def test_stream_quantization_with_mllm_routes_to_meta_loader(self, monkeypatch):
        """Multimodal + streaming no longer errors: the mllm branch routes into
        _load_model_on_meta (arch-resolved skeleton + processor stack; vision
        tower stays meta for the export pass-through)."""
        import torch.nn as nn

        from auto_round.context.model import ModelContext

        called = {"meta": 0}

        class _TinyModule(nn.Module):
            dtype = torch.float32  # _set_amp_dtype reads it
            config = None

            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 4)

        def _fake_meta(self):
            called["meta"] += 1
            self.model = _TinyModule()  # minimal module so the __init__ tail survives
            self.tokenizer = None

        monkeypatch.setattr(ModelContext, "_load_model_on_meta", _fake_meta)
        ctx = self._make_context(monkeypatch, mllm=True)
        assert called["meta"] == 1, "mllm + stream_quantization must use the streaming loader"
        assert ctx.is_mllm is True

    def test_quant_nontext_module_under_streaming_raises(self, monkeypatch):
        """quant_nontext_module + streaming must fail fast: the streaming chain
        feeds text hidden states block-to-block and cannot drive vision blocks
        (silently including them would quantize on garbage statistics)."""
        from types import SimpleNamespace

        import pytest

        from auto_round.compressors.base import BaseCompressor

        stub = SimpleNamespace(stream_quantization=True)
        with pytest.raises(ValueError, match="quant_nontext_module=True is not supported under stream_quantization"):
            BaseCompressor._validate_stream_options(stub, quant_nontext_module=True)
        # sanity: allowed combination passes
        BaseCompressor._validate_stream_options(stub, quant_nontext_module=False)

    def test_stream_quantization_with_mllm_and_env_var_still_raises(self, monkeypatch):
        import pytest

        with pytest.raises(ValueError, match="mutually exclusive"):
            self._make_context(monkeypatch, mllm=True, env_disk_stream=True)

    def test_text_path_without_env_proceeds_to_meta_load(self, monkeypatch):
        """Sanity: the guards must not fire on the normal text streaming path
        (the meta loader then fails on the dummy checkpoint -- any other
        error than the two guards is acceptable progress)."""
        import pytest

        with pytest.raises(Exception) as excinfo:
            self._make_context(monkeypatch)
        assert "mutually exclusive" not in str(excinfo.value)
        assert "multimodal" not in str(excinfo.value)


class TestStreamRowsCap:
    """nsamples caps chained rows for every dataset type (list datasets were
    previously forwarded in full -- inconsistent with the data-driven
    calibrator, which stops at nsamples for any dataset)."""

    def test_list_dataset_capped_at_nsamples(self):
        from auto_round.utils.streaming_calibration import _normalize_rows

        rows = [torch.randint(0, 64, (1, 32)) for _ in range(8)]
        out = _normalize_rows(rows, tokenizer=None, seqlen=32, nsamples=3)
        assert len(out) == 3

    def test_list_dataset_shorter_than_nsamples_untouched(self):
        from auto_round.utils.streaming_calibration import _normalize_rows

        rows = [torch.randint(0, 64, (1, 32)) for _ in range(2)]
        out = _normalize_rows(rows, tokenizer=None, seqlen=32, nsamples=8)
        assert len(out) == 2

    def test_short_rows_dropped_by_skip_rule(self):
        from auto_round.utils.streaming_calibration import _normalize_rows

        rows = [torch.randint(0, 64, (1, 8)), torch.randint(0, 64, (1, 32))]  # first < seqlen
        out = _normalize_rows(rows, tokenizer=None, seqlen=32, nsamples=8)
        assert len(out) == 1


class TestOffloadAfterPackDecision:
    """Per-block offload writes must be skipped once the block is already
    flushed to output shards (is_immediate_saving): the file would be dead
    weight until process exit (a large MoE easily writes hundreds of GB)."""

    def _should_offload(self, low_cpu_mem_usage, is_immediate_saving):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        ctx = SimpleNamespace(low_cpu_mem_usage=low_cpu_mem_usage, is_immediate_saving=is_immediate_saving)
        return CompressionOrchestrator._should_offload_after_pack(ctx)

    def test_offloads_when_not_immediate_saving(self):
        assert self._should_offload(low_cpu_mem_usage=True, is_immediate_saving=False) is True

    def test_skips_when_immediate_saving_flushed_the_block(self):
        assert self._should_offload(low_cpu_mem_usage=True, is_immediate_saving=True) is False

    def test_skips_when_low_cpu_mem_usage_off(self):
        assert self._should_offload(low_cpu_mem_usage=False, is_immediate_saving=False) is False


class TestDiskStreamEnvRotationGuard:
    """_assert_model_foldable_in_place must also refuse whole-model rotation
    under AR_DISK_STREAM_MODEL (env-var offloader path builds the same meta
    skeleton; the guard used to key on the --stream_quantization flag only)."""

    def _composer(self, monkeypatch, env_value):
        from types import SimpleNamespace

        from auto_round import envs
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        if env_value:
            monkeypatch.setenv("AR_DISK_STREAM_MODEL", "1")
        else:
            monkeypatch.delenv("AR_DISK_STREAM_MODEL", raising=False)
        orch = SimpleNamespace(
            layerwise_rotation=False, stream_quantization=False, model_context=None, compress_context=None
        )
        return AlgorithmComposer([RTNConfig(group_size=16)], orchestrator=orch)

    def test_env_var_eager_rotation_raises(self, monkeypatch):
        import pytest
        import torch.nn as nn

        composer = self._composer(monkeypatch, True)
        with pytest.raises(ValueError, match="can only run layer-wise"):
            composer._assert_model_foldable_in_place(nn.Linear(4, 4))

    def test_flag_path_still_raises(self, monkeypatch):
        from types import SimpleNamespace

        import pytest
        import torch.nn as nn

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        composer = AlgorithmComposer(
            [RTNConfig(group_size=16)],
            orchestrator=SimpleNamespace(
                layerwise_rotation=False, stream_quantization=True, model_context=None, compress_context=None
            ),
        )
        with pytest.raises(ValueError, match="can only run layer-wise"):
            composer._assert_model_foldable_in_place(nn.Linear(4, 4))

    def test_no_streaming_no_meta_raises_nothing(self, monkeypatch):
        import torch.nn as nn

        composer = self._composer(monkeypatch, False)
        composer._assert_model_foldable_in_place(nn.Linear(4, 4))  # must not raise


class TestStreamFeatureAutoEngage:
    """_auto_engage_stream_features: layerwise rotation + calibration chain
    engage by themselves under stream_quantization when the run needs them."""

    @staticmethod
    def _engage(configs, **attrs):
        from types import SimpleNamespace

        from auto_round.compressors.base import BaseCompressor

        fields = dict(
            layerwise_rotation=None,
            stream_quantization=False,
            stream_calibration=False,
            _alg_configs=configs,
        )
        fields.update(attrs)
        stub = SimpleNamespace(**fields)
        BaseCompressor._auto_engage_stream_features(stub)
        return stub

    def test_layerwise_stays_off_without_rotations(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        stub = self._engage([RTNConfig(group_size=16)], stream_quantization=True)
        assert stub.layerwise_rotation is False

    def test_calibration_auto_engages_for_signround(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        stub = self._engage([SignRoundConfig(group_size=16, iters=1)], stream_quantization=True)
        assert stub.stream_calibration is True

    def test_calibration_not_engaged_for_weight_only(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        stub = self._engage([RTNConfig(group_size=16)], stream_quantization=True)
        assert stub.stream_calibration is False

    def test_calibration_auto_engages_for_rule_enabled_imatrix(self):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig

        cfg = OptimizedRTNConfig(group_size=16)  # sym int4: imatrix on by scheme rules
        cfg.enable_imatrix = True
        stub = self._engage([cfg], stream_quantization=True)
        assert stub.stream_calibration is True

    def test_calibration_not_engaged_for_imatrix_off(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        stub = self._engage([RTNConfig(group_size=16)], stream_quantization=True)
        assert stub.stream_calibration is False

    def test_calibration_not_engaged_without_streaming(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        stub = self._engage([SignRoundConfig(group_size=16, iters=1)])
        assert stub.stream_calibration is False


@pytest.mark.slow
class TestStreamQuantizeEquivalence:
    """stream_quantization=True must produce the same export as the normal flow."""

    @pytest.fixture(scope="class")
    def tiny_checkpoint(self, tmp_path_factory):
        """Tiny local causal LM checkpoint, sharded (no network access)."""
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
        from tokenizers import Tokenizer
        from tokenizers import models as tk_models
        from tokenizers import pre_tokenizers
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
        stream_prefetch_devices=None,
        dataset=None,
    ):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.autoround import AutoRound

        kwargs = {}
        if dataset is not None:
            kwargs["dataset"] = dataset
            kwargs["seqlen"] = 32
            kwargs["nsamples"] = 8
        ar = AutoRound(
            model_path,
            scheme="W4A16",
            alg_configs=[RTNConfig(group_size=16, disable_opt_rtn=False)],
            stream_quantization=stream,
            stream_prefetch=stream_prefetch,
            stream_prefetch_devices=stream_prefetch_devices,
            **kwargs,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(out_dir, format="auto_round")
        return ar.output_dir  # resolved export dir (may add a name/scheme suffix)

    def test_auto_scheme_with_streaming(self, tiny_checkpoint, tmp_path, monkeypatch):
        """AutoScheme + stream_quantization: scoring never needs the streaming
        parent (pass 1 fills the AR_AUTO_SCHEME_CACHE without streaming; pass 2
        resolves from cache) and the quantization itself runs in the zero-shot
        streaming loop -- streaming governs quantization, not the scheme."""
        import shutil

        from auto_round import envs
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
        from auto_round.autoround import AutoRound

        cache_dir = str(tmp_path / "as_cache")
        monkeypatch.setenv("AR_AUTO_SCHEME_CACHE", str(cache_dir))
        # fp16 embed/lm_head inflate the achievable floor; W2+W4 span it
        scheme = AutoScheme(avg_bits=5.0, options=("W2A16", "W4A16"), nsamples=1, ignore_scale_zp_bits=True)

        def _run(ckpt, out_dir, stream):
            return AutoRound(
                ckpt,
                scheme=scheme,
                iters=0,
                nsamples=1,
                seqlen=32,
                stream_quantization=stream,
                format="auto_round",
                disable_model_free=True,
                device_map="cpu",
                low_gpu_mem_usage=True,
                low_cpu_mem_usage=True,
            ).quantize_and_save(out_dir, format="auto_round")

        # same model path both passes: the scheme cache is keyed on it (quantization
        # happens in memory; the on-disk checkpoint stays pristine)
        _run(tiny_checkpoint, str(tmp_path / "pass1"), stream=False)
        assert any(p.name.endswith(".json") for p in Path(cache_dir).glob("*")), "pass 1 must fill the AutoScheme cache"
        _, out_dir = _run(tiny_checkpoint, str(tmp_path / "pass2"), stream=True)
        with open(os.path.join(out_dir, "config.json")) as f:
            qconf = json.load(f)["quantization_config"]
        assert qconf["quant_method"] == "auto-round"

    def test_auto_scheme_streaming_uncached_no_workers_raises(self, tiny_checkpoint, tmp_path, monkeypatch):
        """Single-run streaming + uncached schemes on a box with no CUDA scoring
        workers: must fail with an actionable error, never attempt in-process
        scoring on the meta skeleton (meta tensor copy)."""
        from auto_round import envs
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
        from auto_round.autoround import AutoRound

        monkeypatch.setenv("AR_AUTO_SCHEME_CACHE", str(tmp_path / "empty_cache"))
        scheme = AutoScheme(avg_bits=5.0, options=("W2A16", "W4A16"), nsamples=1, ignore_scale_zp_bits=True)

        with pytest.raises(RuntimeError, match="AutoScheme scoring cannot run in-process"):
            AutoRound(
                tiny_checkpoint,
                scheme=scheme,
                iters=0,
                nsamples=1,
                seqlen=32,
                stream_quantization=True,
                format="auto_round",
                disable_model_free=True,
                device_map="cpu",
                low_gpu_mem_usage=True,
                low_cpu_mem_usage=True,
            ).quantize_and_save(str(tmp_path / "out"), format="auto_round")

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
        (same attention-mask rule, row-by-row) so the imatrix statistics - and
        therefore the searched quantized weights - are identical."""
        import shutil

        torch.manual_seed(7)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(8)]  # vocab_size=64
        ck_a = str(tmp_path / "ck_a")
        ck_b = str(tmp_path / "ck_b")
        shutil.copytree(tiny_checkpoint, ck_a)
        shutil.copytree(tiny_checkpoint, ck_b)
        data_driven = self._quantize(ck_a, str(tmp_path / "a"), stream=False, dataset=rows)
        streamed_calib = self._quantize(ck_b, str(tmp_path / "b"), stream=True, dataset=rows)

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
        assert set(t_a) == set(t_b), f"tensor name mismatch: only-a={set(t_a) - set(t_b)} only-b={set(t_b) - set(t_a)}"
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

    def test_streamed_signround_qon_chains_quantized_inputs(self, tiny_checkpoint, tmp_path, monkeypatch):
        """iters>0 under stream_quantization routes into the streaming loop
        (routing waiver) and chains each block's quantized outputs as the next
        block's q_inputs (qon), mirroring the data-driven loop."""
        import shutil

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.autoround import AutoRound

        torch.manual_seed(11)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(4)]
        ck = str(tmp_path / "ck_qon")
        shutil.copytree(tiny_checkpoint, ck)

        captured = []
        returned = []
        orig = AlgorithmComposer.compress_block

        def spy(self, block, fp_inputs, input_others, *args, **kwargs):
            captured.append(kwargs.get("q_inputs", "absent"))
            result = orig(self, block, fp_inputs, input_others, *args, **kwargs)
            returned.append(result)
            return result

        monkeypatch.setattr(AlgorithmComposer, "compress_block", spy)
        ar = AutoRound(
            ck,
            scheme="W4A16",
            alg_configs=[SignRoundConfig(group_size=16, iters=1, lr=1e-4)],
            layerwise_rotation=False,
            stream_quantization=True,
            dataset=rows,
            seqlen=32,
            nsamples=4,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        out_dir = ar.quantize_and_save(str(tmp_path / "qon"), format="auto_round")
        assert out_dir, "streamed SignRound qon run must produce an export"
        block_calls = [q for q in captured if q != "absent"]
        assert len(block_calls) >= 2, f"expected >=2 chained block calls, got {len(block_calls)}"
        assert block_calls[0] is None, "first block has no upstream quantized input"
        assert any(q is not None for q in block_calls[1:]), "qon chain must feed quantized outputs downstream"
        # identity: block k+1 receives exactly block k's returned quantized output
        for k in range(min(len(block_calls), len(returned)) - 1):
            if returned[k][0] is not None:
                assert (
                    block_calls[k + 1] is returned[k][0]
                ), f"qon chain leak: block {k + 2} did not receive block {k + 1}'s quantized output"

    def test_streamed_signround_qoff_keeps_q_inputs_none(self, tiny_checkpoint, tmp_path, monkeypatch):
        """enable_quanted_input=False (qoff): every block tunes against the
        fp reference chain only; q_inputs must stay None block-to-block."""
        import shutil

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.autoround import AutoRound

        torch.manual_seed(11)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(4)]
        ck = str(tmp_path / "ck_qoff")
        shutil.copytree(tiny_checkpoint, ck)

        captured = []
        orig = AlgorithmComposer.compress_block

        def spy(self, block, fp_inputs, input_others, *args, **kwargs):
            captured.append(kwargs.get("q_inputs", "absent"))
            return orig(self, block, fp_inputs, input_others, *args, **kwargs)

        monkeypatch.setattr(AlgorithmComposer, "compress_block", spy)
        ar = AutoRound(
            ck,
            scheme="W4A16",
            alg_configs=[SignRoundConfig(group_size=16, iters=1, lr=1e-4)],
            layerwise_rotation=False,
            stream_quantization=True,
            enable_quanted_input=False,
            dataset=rows,
            seqlen=32,
            nsamples=4,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(str(tmp_path / "qoff"), format="auto_round")
        block_calls = [q for q in captured if q != "absent"]
        assert len(block_calls) >= 2
        assert all(q is None for q in block_calls), "qoff must not chain quantized inputs"

    def test_streamed_signround_auto_engages_calibration(self, tiny_checkpoint, tmp_path, monkeypatch):
        """stream_quantization + iters>0 with stream_calibration unset: the
        activation chain auto-engages -- the run completes with real chained
        tuning inputs instead of silently skipping the tuning data."""
        import shutil

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.autoround import AutoRound

        torch.manual_seed(11)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(4)]
        ck = str(tmp_path / "ck_nocal")
        shutil.copytree(tiny_checkpoint, ck)

        chained = {"calls": 0}
        orig = AlgorithmComposer.compress_block

        def spy(self, block, fp_inputs, input_others, *args, **kwargs):
            if fp_inputs is not None:
                chained["calls"] += 1
            return orig(self, block, fp_inputs, input_others, *args, **kwargs)

        monkeypatch.setattr(AlgorithmComposer, "compress_block", spy)
        ar = AutoRound(
            ck,
            scheme="W4A16",
            alg_configs=[SignRoundConfig(group_size=16, iters=1, lr=1e-4)],
            stream_quantization=True,
            dataset=rows,
            seqlen=32,
            nsamples=4,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        assert ar.stream_calibration is True, "chain must auto-engage for iters>0 under streaming"
        out_dir = ar.quantize_and_save(str(tmp_path / "nocal"), format="auto_round")
        assert out_dir
        assert chained["calls"] >= 2, "auto-engaged chain must feed every block"

    def test_streamed_signround_runs_prepare_run_and_alg_ext_wrapper(self, tiny_checkpoint, tmp_path, monkeypatch):
        """The streaming zero-shot loop must run the model-level lifecycle
        before its block loop: SignRoundV2Quantizer.prepare_run binds the
        optimized wrapper (SignRoundOptimizedWrapperLinear). Skipping it left
        V2 tuning silently on the plain min/max wrapper (worse init, worse KL).
        """
        import shutil
        from functools import partial

        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear
        from auto_round.autoround import AutoRound
        from auto_round.wrapper import wrapper_block

        torch.manual_seed(11)
        rows = [torch.randint(0, 64, (1, 32)) for _ in range(4)]
        ck = str(tmp_path / "ck_ext")
        shutil.copytree(tiny_checkpoint, ck)

        prepared = {"calls": 0}
        orig_prepare = AlgorithmComposer.prepare_run

        def prepare_spy(self, composer=None):
            prepared["calls"] += 1
            return orig_prepare(self, composer=composer)

        wrapper_at_compress = []
        orig = AlgorithmComposer.compress_block

        def spy(self, block, fp_inputs, input_others, *args, **kwargs):
            wrapper_at_compress.append(self.block_quantizer.wrapper_block)
            return orig(self, block, fp_inputs, input_others, *args, **kwargs)

        monkeypatch.setattr(AlgorithmComposer, "prepare_run", prepare_spy)
        monkeypatch.setattr(AlgorithmComposer, "compress_block", spy)
        ar = AutoRound(
            ck,
            scheme="W4A16",
            alg_configs=[SignRoundConfig(group_size=16, iters=1, lr=1e-4, enable_alg_ext=True)],
            layerwise_rotation=False,
            stream_quantization=True,
            dataset=rows,
            seqlen=32,
            nsamples=4,
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            low_gpu_mem_usage=True,
            low_cpu_mem_usage=True,
        )
        out_dir = ar.quantize_and_save(str(tmp_path / "ext"), format="auto_round")
        assert out_dir, "streamed SignRound alg-ext run must produce an export"
        assert prepared["calls"] >= 1, "streaming zero-shot loop never called alg_composer.prepare_run"
        assert wrapper_at_compress, "no compress_block calls observed"
        expected = partial(wrapper_block, wrapper_cls=SignRoundOptimizedWrapperLinear)
        for w in wrapper_at_compress:
            assert w.func is wrapper_block, "wrapper_block must stay the shared wrapper function"
            assert w.keywords.get("wrapper_cls") is SignRoundOptimizedWrapperLinear, (
                "alg-ext tuning under streaming must use the optimized wrapper; "
                "the plain min/max WrapperLinear means prepare_run was skipped"
            )

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
            ck_b, str(tmp_path / "staged"), stream=True, stream_prefetch=2, stream_prefetch_devices=stage_devs
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

    def test_unclaimed_block_passthrough(self, tiny_checkpoint, tmp_path):
        """Checkpoint block groups with no module counterpart (e.g. an MTP layer
        transformers does not model) must be written verbatim."""
        import json
        import os
        import shutil

        from safetensors import safe_open
        from safetensors.torch import save_file

        src = shutil.copytree(tiny_checkpoint, str(tmp_path / "ck"))
        extra = {
            "model.layers.3.eh_proj.weight": torch.randn(16, 32),
            "model.layers.3.enorm.weight": torch.randn(32),
        }
        idx_path = os.path.join(src, "model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        last = sorted(set(idx["weight_map"].values()))[-1]
        with safe_open(os.path.join(src, last), framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        tensors.update(extra)
        save_file(tensors, os.path.join(src, last), metadata={"format": "pt"})
        for k in extra:
            idx["weight_map"][k] = last
        with open(idx_path, "w") as f:
            json.dump(idx, f)

        out = self._quantize(src, str(tmp_path / "out"), stream=True)

        idx_file = os.path.join(out, "model.safetensors.index.json")
        single_file = os.path.join(out, "model.safetensors")
        if os.path.exists(idx_file):
            with open(idx_file) as f:
                wm = json.load(f)["weight_map"]
            shard = os.path.join(out, wm["model.layers.3.eh_proj.weight"])
        else:
            assert os.path.exists(single_file), f"no export tensors in {out}"
            shard = single_file
        from safetensors import safe_open

        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            assert "model.layers.3.eh_proj.weight" in keys, "unclaimed block tensor dropped from export"
            assert "model.layers.3.enorm.weight" in keys
            assert torch.equal(f.get_tensor("model.layers.3.eh_proj.weight"), extra["model.layers.3.eh_proj.weight"])
            assert torch.equal(f.get_tensor("model.layers.3.enorm.weight"), extra["model.layers.3.enorm.weight"])


class TestAutoStagingScopesToDeviceMap:
    """Auto staging never reaches outside the user's --device_map.

    An explicit device map is a sandbox declaration (other GPUs may belong to
    other jobs); the no-flag default resolves to every visible GPU, so
    scoping changes nothing there.
    """

    def _resolve(self, monkeypatch, device_list, quant="cuda:1", primary_fit=None):
        from types import SimpleNamespace

        import torch as _torch

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        monkeypatch.setattr(_torch.cuda, "device_count", lambda: 4)
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        stub = SimpleNamespace(
            stream_prefetch=None,
            device=quant,
            stream_prefetch_devices="auto",
            _primary_fits_largest_block=lambda dev: primary_fit,
        )
        monkeypatch.setattr(
            "auto_round.compressors.orchestrator.device_manager",
            SimpleNamespace(device_list=device_list),
        )
        return CompressionOrchestrator._resolve_stream_stage_devices(stub)

    def test_auto_excludes_gpus_outside_device_map(self, monkeypatch):
        devices = self._resolve(monkeypatch, device_list=["cuda:1", "cuda:2"])
        # cuda:0 and cuda:3 are visible but not in the map: never staged
        assert [str(d) for d in devices] == ["cuda:2"]

    def test_auto_primary_joins_only_within_device_map(self, monkeypatch):
        devices = self._resolve(monkeypatch, device_list=["cuda:1", "cuda:2"], primary_fit=(6.9, 22.3))
        assert sorted(str(d) for d in devices) == ["cuda:1", "cuda:2"]

    def test_auto_default_map_uses_all_visible_gpus(self, monkeypatch):
        # no --device_map: device_list still resolves to every visible GPU
        devices = self._resolve(monkeypatch, device_list=[f"cuda:{i}" for i in range(4)], primary_fit=(6.9, 22.3))
        assert sorted(str(d) for d in devices) == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    def test_auto_sole_gpu_in_map_without_fit_falls_back_to_ram(self, monkeypatch):
        assert self._resolve(monkeypatch, device_list=["cuda:1"], primary_fit=None) is None


class TestResumeCudaCacheRelease:
    def test_releases_reserved_segments_when_cuda_available(self, monkeypatch):
        from types import SimpleNamespace

        import torch as _torch

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        calls = []
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(_torch.cuda, "empty_cache", lambda: calls.append(1))
        CompressionOrchestrator._release_cuda_cache("resume rebuild")
        assert calls == [1]

    def test_never_raises_without_cuda(self, monkeypatch):
        import torch as _torch

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        monkeypatch.setattr(_torch.cuda, "is_available", lambda: False)
        CompressionOrchestrator._release_cuda_cache("resume rebuild")


class TestLoadBreakdown:
    """AR_PERF_COUNTERS load sub-phase breakdown for the [perf] line."""

    def test_empty_when_no_parts(self):
        from auto_round.compressors.orchestrator import _format_load_breakdown

        assert _format_load_breakdown({}) == ""
        assert _format_load_breakdown(None) == ""

    def test_subresolution_segments_fold_away(self):
        from auto_round.compressors.orchestrator import _format_load_breakdown

        assert _format_load_breakdown({"io": 0.01, "close": 0.02}) == ""

    def test_breakdown_lists_significant_segments(self):
        from auto_round.compressors.orchestrator import _format_load_breakdown

        out = _format_load_breakdown({"io": 6.9, "close": 0.06, "inv": 2.0})
        assert out == " (io 6.9s, close 0.1s, inv 2.0s)"

    def test_streamer_reports_fetch_segments(self, tmp_path, monkeypatch):
        import logging

        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        from auto_round.utils import checkpoint_streamer as cs_mod

        handler = _Cap()
        cs_mod.logger.addHandler(handler)
        try:
            streamer, block = self._tiny_fixture(tmp_path)
            streamer.load_module_(block, "blk", device=None)
        finally:
            cs_mod.logger.removeHandler(handler)
        perf_lines = [r for r in records if r.startswith("[perf] streamer")]
        assert len(perf_lines) == 1
        assert "read" in perf_lines[0] and "tensors" in perf_lines[0]

    @staticmethod
    def _tiny_fixture(tmp_path):
        import torch.nn as nn

        streamer = CheckpointStreamer(
            _make_sharded_checkpoint(tmp_path, {"single.safetensors": {"blk.a.weight": torch.randn(4, 8)}})
        )

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4, bias=False)

        with torch.device("meta"):
            blk = Blk()
        return streamer, blk


class TestFragmentationRelease:
    """Phase-boundary release of provably-free cached segments before tuning."""

    def _mk(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        return CompressionOrchestrator, SimpleNamespace()

    def test_releases_when_gap_exceeds_threshold(self, monkeypatch):
        import torch as _torch

        cls, stub = self._mk()
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(_torch.cuda, "memory_reserved", lambda idx: 13 * 2**30)
        monkeypatch.setattr(_torch.cuda, "memory_allocated", lambda idx: 6 * 2**30)
        monkeypatch.setattr(_torch.cuda, "current_device", lambda: 0)
        calls = []
        monkeypatch.setattr(_torch.cuda, "empty_cache", lambda: calls.append(1))
        assert cls._release_cached_segments_if_fragmented(stub, "cuda:0") is True
        assert calls == [1]

    def test_noop_when_gap_small(self, monkeypatch):
        import torch as _torch

        cls, stub = self._mk()
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(_torch.cuda, "memory_reserved", lambda idx: 7 * 2**30)
        monkeypatch.setattr(_torch.cuda, "memory_allocated", lambda idx: 6 * 2**30)
        monkeypatch.setattr(_torch.cuda, "current_device", lambda: 0)
        calls = []
        monkeypatch.setattr(_torch.cuda, "empty_cache", lambda: calls.append(1))
        assert cls._release_cached_segments_if_fragmented(stub, "cuda:0") is False
        assert calls == []

    def test_cpu_device_never_releases(self, monkeypatch):
        import torch as _torch

        cls, stub = self._mk()
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        calls = []
        monkeypatch.setattr(_torch.cuda, "empty_cache", lambda: calls.append(1))
        assert cls._release_cached_segments_if_fragmented(stub, "cpu") is False
        assert calls == []


class TestPrefetchConsumerWaits:
    """load_module_ must wait out an in-flight prefetch of the same prefix.

    Without the wait the consumer falls back to its own disk read while the
    reader keeps staging the same block - two full copies on the staging
    device, one freed only afterwards (block-sized reserved-but-unallocated
    gap, first-forward OOMs on tightly-fitting GPUs).
    """

    def test_consumer_waits_for_inflight_staging(self, tmp_path, monkeypatch):
        import threading
        import time as _time

        torch.manual_seed(0)
        w = torch.randn(4, 8)
        path = _make_sharded_checkpoint(tmp_path, {"single.safetensors": {"blk.a.weight": w}})
        streamer = CheckpointStreamer(path)

        import torch.nn as nn

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4, bias=False)

        with torch.device("meta"):
            blk = Blk()

        read_threads = []
        real_read = streamer._read_tensor

        def _slow_read(name, handles, order):
            read_threads.append(threading.get_ident())
            _time.sleep(0.05)
            return real_read(name, handles, order)

        monkeypatch.setattr(streamer, "_read_tensor", _slow_read)
        streamer.start_prefetch(["blk"], depth=1, stage_devices=None)
        try:
            streamer.load_module_(blk, "blk", device=None)
        finally:
            streamer.stop_prefetch()
        # every tensor came from the reader's staging, never the main thread
        assert read_threads and set(read_threads) != {threading.get_ident()}
        assert torch.equal(blk.a.weight.data, w)

    def test_non_enqueued_prefix_does_not_wait(self, tmp_path, monkeypatch):
        import threading
        import time as _time

        w = torch.randn(4, 8)
        e = torch.randn(8, 4)
        path = _make_sharded_checkpoint(tmp_path, {"single.safetensors": {"blk.a.weight": w, "embed.weight": e}})
        streamer = CheckpointStreamer(path)

        import torch.nn as nn

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 4, bias=False)

        class Emb(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(8, 4, dtype=torch.float32).to("meta"))

        started = threading.Event()
        release = threading.Event()

        real_read = streamer._read_tensor

        def _gated_read(name, handles, order):
            if name.startswith("blk."):
                started.set()
                release.wait(timeout=10)
            return real_read(name, handles, order)

        monkeypatch.setattr(streamer, "_read_tensor", _gated_read)
        streamer.start_prefetch(["blk"], depth=1, stage_devices=None)
        try:
            assert started.wait(timeout=10)  # reader is mid-staging blk
            with torch.device("meta"):
                emb = Emb()
            streamer.load_module_(emb, "embed", device=None)  # not enqueued: must not block
            assert torch.equal(emb.weight.data, e)
        finally:
            release.set()
            streamer.stop_prefetch()
