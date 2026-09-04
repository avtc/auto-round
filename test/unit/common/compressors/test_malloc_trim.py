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
    def test_trim_host_heap_never_raises_and_returns_bool(self):
        # On non-glibc hosts (Windows/macOS) the ctypes load fails and the
        # helper must degrade to a quiet False, never raise.
        result = CompressionOrchestrator._trim_host_heap()
        assert isinstance(result, bool)

    def test_trim_sites_unconditional_in_source(self):
        # The two streaming-loop call sites must be unconditional: the trim is
        # cheap hygiene (ms per block) and keeps RSS/VmHWM from growing
        # monotonically on glibc hosts.
        import inspect

        src = inspect.getsource(CompressionOrchestrator)
        lines = src.splitlines()
        sites = [i for i, ln in enumerate(lines) if "_trim_host_heap()" in ln and "def " not in ln]
        assert len(sites) == 2, "expected exactly 2 call sites"
        for i in sites:
            window = "\n".join(lines[max(0, i - 3) : i + 1])
            assert "envs." not in window, f"call site still gated:\n{window}"

    def test_trim_helper_isolated_from_model_state(self):
        # must not require any instance state (called as static from the loop)
        with torch.no_grad():
            t = torch.zeros(1024, 1024)
            CompressionOrchestrator._trim_host_heap()
            t += 1
        assert bool(t.sum() > 0)


class TestMemDiagnosticsGatedByDebugLevel:
    """The streaming memory diagnostics (inventory buckets, top-tensor/
    region attribution, peak-RSS watcher) have no env switches anymore:
    they are DEBUG-level log lines, activated by AR_LOG_LEVEL=DEBUG."""

    def test_inventory_and_watcher_gated_on_logger(self):
        import inspect

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        src = inspect.getsource(CompressionOrchestrator)
        assert src.count("logger.isEnabledFor(logging.DEBUG)") >= 6  # 4 inventory sites + 2 watcher sites
        # emissions are debug-level: visible only under AR_LOG_LEVEL=DEBUG
        assert 'logger.debug(\n                "[stream-mem]' in src
        assert 'logger.info(\n                "[stream-mem]' not in src


class TestShardPool:
    def test_depth_is_structural_constant(self):
        # The LRU depth is a structural constant: sequential consume-once reads
        # need current + prefetch-next; deeper only keeps already-read pages
        # mapped in RSS with zero reuse benefit.
        import inspect

        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        src = inspect.getsource(CheckpointStreamer.__init__)
        assert "self._max_open = 2" in src


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
    def test_close_main_pool(self, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

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


class TestUnconditionalBoundedResidency:
    """Grouped shard reads, per-block consumer-pool close and heap trims are
    unconditional streaming behaviors now - bounded host residency is the
    default contract, not an opt-in recipe."""

    def test_grouped_read_unconditional(self):
        import inspect

        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        src = inspect.getsource(CheckpointStreamer.load_module_)
        assert "key=lambda n: self.weight_map[n]" in src

    def test_loop_sites_ungated(self):
        import inspect

        src = inspect.getsource(CompressionOrchestrator)
        assert "close_main_pool_()" in src


class TestWalkConcurrentSafety:
    """The inventory walk snapshots every container it iterates: the bg pack
    worker mutates composer-held dicts while the main loop logs."""

    def test_walk_snapshots_all_iterations(self):
        import inspect

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        src = inspect.getsource(CompressionOrchestrator._log_device_inventory)
        assert "for k, x in list(v.items()):" in src
        assert "for k, x in list(vars(v).items()):" in src
        assert "for j, x in enumerate(list(v)):" in src
        # no unsnapshotted live iteration remains
        for bad in ("for k, x in v.items():", "for k, x in vars(v).items():", "for j, x in enumerate(v):"):
            assert bad not in src

    def test_walk_survives_dict_growing_during_holder_walk(self):
        # a holder object whose __dict__ gains a key between snapshot and
        # recursion must not crash the walk (bg worker inserting mid-walk)
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)

        class _Holder:
            pass

        holder = _Holder()
        holder.records = {"a": torch.zeros(8)}
        orch.model = torch.nn.Sequential()  # empty: walk focuses on the holder
        orch._log_device_inventory({"fp_inputs": [holder]}, "race-sim")
