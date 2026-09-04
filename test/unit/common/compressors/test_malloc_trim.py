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
import torch

import auto_round.envs as envs
from auto_round.compressors.orchestrator import CompressionOrchestrator


class TestMallocTrim:
    def test_env_default_off_and_lazy_read(self, monkeypatch):
        assert envs.AR_STREAM_MALLOC_TRIM is False
        monkeypatch.setenv("AR_STREAM_MALLOC_TRIM", "1")
        assert envs.AR_STREAM_MALLOC_TRIM is True
        monkeypatch.setenv("AR_STREAM_MALLOC_TRIM", "0")
        assert envs.AR_STREAM_MALLOC_TRIM is False

    def test_trim_host_heap_never_raises_and_returns_bool(self):
        # On non-glibc hosts (Windows/macOS) the ctypes load fails and the
        # helper must degrade to a quiet False, never raise.
        result = CompressionOrchestrator._trim_host_heap()
        assert isinstance(result, bool)

    def test_trim_sites_gated_by_env_in_source(self):
        # The two streaming-loop call sites must sit behind the env flag so
        # the default behavior is unchanged.
        import inspect

        src = inspect.getsource(CompressionOrchestrator)
        gated = [ln for ln in src.splitlines() if "_trim_host_heap()" in ln and "def " not in ln]
        assert len(gated) == 2, "expected exactly 2 call sites"
        for ln in gated:
            idx = src.splitlines().index(ln)
            window = "\n".join(src.splitlines()[max(0, idx - 3) : idx + 1])
            assert "AR_STREAM_MALLOC_TRIM" in window, f"call site not env-gated:\n{window}"

    def test_trim_helper_isolated_from_model_state(self):
        # must not require any instance state (called as static from the loop)
        with torch.no_grad():
            t = torch.zeros(1024, 1024)
            CompressionOrchestrator._trim_host_heap()
            t += 1
        assert bool(t.sum() > 0)


class TestStreamMemTop:
    def test_env_parse(self, monkeypatch):
        import auto_round.envs as envs

        assert envs.AR_STREAM_MEM_TOP == 0.0
        monkeypatch.setenv("AR_STREAM_MEM_TOP", "0.25")
        assert abs(envs.AR_STREAM_MEM_TOP - 0.25) < 1e-9
        monkeypatch.setenv("AR_STREAM_MEM_TOP", "not-a-float")
        assert envs.AR_STREAM_MEM_TOP == 0.0


class TestDropFileCache:
    def test_env_parse(self, monkeypatch):
        import auto_round.envs as envs

        assert envs.AR_STREAM_DROP_FILE_CACHE is False
        monkeypatch.setenv("AR_STREAM_DROP_FILE_CACHE", "1")
        assert envs.AR_STREAM_DROP_FILE_CACHE is True

    def test_drop_file_cache_never_raises(self, tmp_path):
        # non-POSIX hosts must silently no-op; a bogus path must not raise
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        CheckpointStreamer._drop_file_cache(str(tmp_path / "missing.safetensors"))


class TestShardPool:
    def test_env_default_and_bounds(self, monkeypatch):
        import auto_round.envs as envs

        assert envs.AR_STREAM_SHARD_POOL == 2
        monkeypatch.setenv("AR_STREAM_SHARD_POOL", "4")
        assert envs.AR_STREAM_SHARD_POOL == 4
        monkeypatch.setenv("AR_STREAM_SHARD_POOL", "0")
        assert envs.AR_STREAM_SHARD_POOL == 1  # clamped
        monkeypatch.setenv("AR_STREAM_SHARD_POOL", "bogus")
        import pytest

        with pytest.raises(ValueError):
            _ = envs.AR_STREAM_SHARD_POOL  # noqa: B018


class TestConsumedShardClose:
    def _streamer(self, monkeypatch, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        s = object.__new__(CheckpointStreamer)
        s.weight_map = {"a.w": "s0", "b.w": "s0", "c.w": "s1"}
        s._format = "safetensors"
        s._open_handles = {}
        s._open_order = []
        s._shard_unread = {}
        s._shard_path = lambda shard: str(tmp_path / shard)
        return s

    def test_closes_only_when_last_tensor_read(self, monkeypatch, tmp_path):
        s = self._streamer(monkeypatch, tmp_path)
        closed = []
        fake = SimpleNamespace(__exit__=lambda self_, *a: closed.append("s0"))
        s._open_handles["s0"] = fake
        s._open_order.append("s0")
        s._close_if_consumed_("s0", s._open_handles, s._open_order)
        assert closed == []  # two tensors still unread
        s._shard_unread["s0"].discard("a.w")
        s._close_if_consumed_("s0", s._open_handles, s._open_order)
        assert closed == []  # one still unread
        s._shard_unread["s0"].discard("b.w")
        s._close_if_consumed_("s0", s._open_handles, s._open_order)
        assert closed == ["s0"]  # last read -> closed + evicted
        assert "s0" not in s._open_handles and "s0" not in s._open_order

    def test_closes_in_both_pools(self, monkeypatch, tmp_path):
        s = self._streamer(monkeypatch, tmp_path)
        s._prefetch_handles = {}
        s._prefetch_handle_order = []
        exit_calls = []
        for pool in (s._open_handles, s._prefetch_handles):
            fake = SimpleNamespace(__exit__=lambda self_, *a: exit_calls.append(id(pool)))
            pool["s1"] = fake
        s._open_order.append("s1")
        s._prefetch_handle_order.append("s1")
        s._shard_unread["s1"] = set()  # all tensors of s1 already read
        s._close_if_consumed_("s1", s._open_handles, s._open_order)
        assert len(exit_calls) == 2
        assert not s._open_handles and not s._prefetch_handles


from types import SimpleNamespace  # noqa: E402


class TestStartupRelease:
    def _streamer(self, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        s = object.__new__(CheckpointStreamer)
        s._open_handles = {}
        s._open_order = []
        s._shard_path = lambda shard: str(tmp_path / shard)
        return s

    def test_release_closes_and_clears_all_pools(self, tmp_path):
        s = self._streamer(tmp_path)
        exited = []
        s._open_handles["s0"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s0"))
        s._open_handles["s1"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s1"))
        s._open_order.extend(["s0", "s1"])
        s.release_startup_handles_()
        assert exited == ["s0", "s1"]
        assert not s._open_handles and not s._open_order


class TestPlannedSetClose:
    def _streamer(self, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        s = object.__new__(CheckpointStreamer)
        s.weight_map = {"blk0.a": "s0", "vis.b": "s0", "blk1.a": "s1"}
        s._format = "safetensors"
        s._open_handles = {}
        s._open_order = []
        s._shard_unread = {}
        s._shard_path = lambda shard: str(tmp_path / shard)
        return s

    def test_closes_shard_with_no_planned_reads(self, tmp_path):
        s = self._streamer(tmp_path)
        # s0 fully read except the never-planned vision tensor
        s._shard_unread["s0"] = {"vis.b"}
        s._shard_unread["s1"] = {"blk1.a"}  # blk1 not yet read -> still serving
        exited = []
        s._open_handles["s0"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s0"))
        s._open_handles["s1"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s1"))
        s._open_order.extend(["s0", "s1"])
        s.close_shards_not_serving_(["blk1"])  # only blk1 remains planned
        assert exited == ["s0"]  # s0 serves nothing planned (vis.b never read)
        assert "s0" not in s._open_handles
        assert "s1" in s._open_handles  # still serving blk1

    def test_keeps_shard_serving_future_block(self, tmp_path):
        s = self._streamer(tmp_path)
        s._shard_unread["s0"] = {"blk0.a", "vis.b"}
        exited = []
        s._open_handles["s0"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s0"))
        s._open_order.append("s0")
        s.close_shards_not_serving_(["blk0"])  # blk0 still planned
        assert exited == []


class TestPerBlockClose:
    def test_env_and_close_main_pool(self, monkeypatch, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        import auto_round.envs as envs

        assert envs.AR_STREAM_CLOSE_PER_BLOCK is False
        monkeypatch.setenv("AR_STREAM_CLOSE_PER_BLOCK", "1")
        assert envs.AR_STREAM_CLOSE_PER_BLOCK is True

        s = object.__new__(CheckpointStreamer)
        s._open_handles = {}
        s._open_order = []
        s._shard_path = lambda shard: str(tmp_path / shard)
        s._prefetch_handles = {"s9": SimpleNamespace(__exit__=lambda self_, *a: None)}
        s._prefetch_handle_order = ["s9"]
        exited = []
        s._open_handles["s0"] = SimpleNamespace(__exit__=lambda self_, *a: exited.append("s0"))
        s._open_order.append("s0")
        s.close_main_pool_()
        assert exited == ["s0"]
        assert not s._open_handles and not s._open_order
        assert "s9" in s._prefetch_handles  # prefetch pool untouched
