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
"""Background pack pipeline (AR_STREAM_BG_PACK=auto|1|0).

The finished block's immediate-pack + shard-write tail runs in a background
thread on its (now idle) ping-pong home while the loop advances to the next
block's tune on the other group. Tests cover the worker contract (pack ->
leaf saves -> write -> flush -> mark_block_done ordering, snapshot args,
meta release) and failure surfacing at join time.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from auto_round.compressors.orchestrator import CompressionOrchestrator


class _Ctx:
    is_immediate_packing = True
    is_immediate_saving = True


class _NoSaveCtx(_Ctx):
    is_immediate_saving = False


class _Writer:
    def __init__(self):
        self.calls = []
        self.flushed = 0

    def write(self, name=None, **kw):
        self.calls.append(("write", name))

    def _flush_shard(self):
        self.calls.append(("flush", None))
        self.flushed += 1


class _RS:
    def __init__(self):
        self.calls = []

    def mark_block_done(self, name, q, fp):
        self.calls.append(("mark", name, q, fp))


class TestSnapshotChainRows:
    """_snapshot_chain_rows_: resume snapshots must be immune to chain mutation.

    The bg-finish worker serializes the snapshot seconds after capture; the
    main loop resets the shared chain containers meanwhile (27B evidence:
    saved entry read back as rows[128] with a None first row)."""

    def test_deep_copies_tensors_and_containers(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        rows = [torch.ones(2, 3), {"hidden_states": [torch.zeros(1, 1)]}]
        snap = CompressionOrchestrator._snapshot_chain_rows_(rows)
        rows[0][0, 0] = 99
        rows[1]["hidden_states"][0][0, 0] = 99
        assert snap[0][0, 0] == 1
        assert snap[1]["hidden_states"][0][0, 0] == 0
        # copies are detached and on the host
        assert not snap[0].requires_grad and snap[0].device.type == "cpu"

    def test_none_slots_and_scalars_pass_through(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        snap = CompressionOrchestrator._snapshot_chain_rows_([None, 3, torch.tensor([1.0])])
        assert snap[0] is None and snap[1] == 3 and torch.equal(snap[2], torch.tensor([1.0]))

    def test_none_state_and_dict_roundtrip(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        assert CompressionOrchestrator._snapshot_chain_rows_(None) is None
        d = {"a": [torch.tensor(2.0)], "b": (torch.tensor(3.0),)}
        snap = CompressionOrchestrator._snapshot_chain_rows_(d)
        assert isinstance(snap, dict) and isinstance(snap["b"], tuple)
        assert snap["a"][0].item() == 2.0 and snap["b"][0].item() == 3.0


class TestBgPackEnv:
    def test_tri_state_parse(self, monkeypatch):
        import pytest

        from auto_round import envs

        monkeypatch.delenv("AR_STREAM_BG_PACK", raising=False)
        assert envs.AR_STREAM_BG_PACK == "auto"
        monkeypatch.setenv("AR_STREAM_BG_PACK", "1")
        assert envs.AR_STREAM_BG_PACK == "on"
        monkeypatch.setenv("AR_STREAM_BG_PACK", "0")
        assert envs.AR_STREAM_BG_PACK == "off"
        monkeypatch.setenv("AR_STREAM_BG_PACK", "bogus")
        with pytest.raises(ValueError):
            _ = envs.AR_STREAM_BG_PACK

    def test_resolve_mode_matrix(self):
        import pytest

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        r = CompressionOrchestrator._resolve_bg_pack_mode
        # auto: on exactly when supported
        assert r("auto", 2, True) is True
        assert r("auto", 1, True) is False
        assert r("auto", 2, False) is False
        # 0: always serialized
        assert r("off", 2, True) is False
        # 1: required - fails loudly when unsupported, never silent fallback
        assert r("on", 2, True) is True
        with pytest.raises(ValueError, match="AR_STREAM_BG_PACK=1 requires"):
            r("on", 1, True)
        with pytest.raises(ValueError, match="AR_STREAM_BG_PACK=1 requires"):
            r("on", 2, False)


class TestBgPackWorker:
    def _orchestrator(self, ctx):
        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        model = nn.Sequential()
        blk = nn.Sequential()
        norm = nn.LayerNorm(4)
        norm.global_name = "blk.norm"
        blk.add_module("norm", norm)
        model.add_module("blk", blk)
        orch.model = model
        orch.compress_context = ctx
        orch.shard_writer = _Writer()
        return orch, model, blk

    def _run(self, ctx, is_last=False, pack_impl=None):
        from auto_round.compressors import utils as cutils

        orch, model, blk = self._orchestrator(ctx)
        writer, rs = orch.shard_writer, _RS()
        q_snap, fp_snap = object(), object()
        orig = cutils.immediate_pack_block
        if pack_impl is not None:
            cutils.immediate_pack_block = pack_impl
        try:
            t = orch._start_bg_pack_block(blk, "blk", "cpu", {"bits": 4}, 1, set(), rs, q_snap, fp_snap, is_last)
            orch._join_bg_pack(t)
        finally:
            if pack_impl is not None:
                cutils.immediate_pack_block = orig
        return orch, writer, rs, model, blk, (q_snap, fp_snap)

    def test_worker_packs_writes_marks_in_order(self):
        pack_seen = []

        def _pack(block, name, layer_config, nblocks=1, device=None):
            pack_seen.append((name, layer_config, nblocks, device))

        orch, writer, rs, model, blk, snaps = self._run(_Ctx(), pack_impl=_pack)
        assert pack_seen == [("blk", {"bits": 4}, 1, "cpu")]
        kinds = [c[0] for c in writer.calls]
        # leaf save then block-scope write, then flush BEFORE mark_block_done
        assert kinds == ["write", "write", "flush"]
        assert writer.calls[0] == ("write", "blk.norm") and writer.calls[1] == ("write", "blk")
        assert rs.calls == [("mark", "blk", snaps[0], snaps[1])]
        # snapshots passed through, not live dict reads
        assert rs.calls[0][2] is snaps[0] and rs.calls[0][3] is snaps[1]
        # block released to meta
        assert all(p.device.type == "meta" for p in blk.parameters())

    def test_model_last_drops_fp_snapshot(self):
        _, _, rs, *_ = self._run(_Ctx(), is_last=True)
        assert len(rs.calls) == 1
        assert rs.calls[0][3] is None  # fp snapshot dropped on the model-last block

    def test_worker_without_saving_moves_to_cpu(self):
        orch, writer, rs, model, blk, _ = self._run(_NoSaveCtx())
        assert writer.calls == [] and rs.calls == []
        assert all(p.device.type == "cpu" for p in blk.parameters())

    def test_worker_failure_surfaces_at_join(self):
        def _boom(*a, **kw):
            raise ValueError("pack exploded")

        with pytest.raises(RuntimeError, match="pack exploded"):
            self._run(_Ctx(), pack_impl=_boom)


class TestFormatHostBuckets:
    """[stream-mem] host lines must not drown real residents in zero-bucket noise."""

    def test_zero_and_subresolution_buckets_collapse(self):
        from auto_round.compressors.orchestrator import _format_host_buckets

        buckets = {f"block:{i}": 0 for i in range(80)}
        buckets["block:1"] = int(0.4 * 2**30)  # real resident
        buckets["block:2"] = 2 * 2**20  # 2 MiB renders as 0.00G -> negligible
        out = _format_host_buckets(buckets)
        assert out.startswith("block:1=0.40G")
        assert "[79 negligible buckets]" in out
        assert "block:2=" not in out
        assert "block:0=" not in out

    def test_all_real_buckets_listed_sorted(self):
        from auto_round.compressors.orchestrator import _format_host_buckets

        out = _format_host_buckets({"chain": 2 * 2**30, "quantizer": int(0.45 * 2**30)})
        assert out == "chain=2.00G, quantizer=0.45G"

    def test_empty_buckets_render_empty_string(self):
        from auto_round.compressors.orchestrator import _format_host_buckets

        assert _format_host_buckets({}) == ""


class TestMainLoopBlockOwnership:
    """With immediate saving the finished block belongs to the pack pipeline,
    never to the main loop: moving it from the loop races the worker's
    compress (weights toward cpu, search scales still on the home device)."""

    def test_immediate_saving_blocks_main_loop_move(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        assert CompressionOrchestrator._main_loop_may_move_block_off_gpu(True) is False

    def test_non_saving_path_still_moves(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        assert CompressionOrchestrator._main_loop_may_move_block_off_gpu(False) is True


class TestBgFinishForBlobMode:
    """Blob-mode finish worker: packing stays serial, the block finish
    (meta-park + resume snapshot) overlaps on a background thread."""

    def test_resolve_bg_finish_mode(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator as CO

        # blob mode: the finish worker does no GPU math, so no device-count
        # requirement; only an explicit "off" disables it
        assert CO._resolve_bg_finish_mode(True, "auto") is True
        assert CO._resolve_bg_finish_mode(True, "1") is True
        assert CO._resolve_bg_finish_mode(True, "off") is False
        # non-blob formats use the full bg-pack pipeline instead
        assert CO._resolve_bg_finish_mode(False, "auto") is False

    def test_finish_worker_runs_off_main_and_receives_captured_refs(self):
        import threading

        from auto_round.compressors.orchestrator import CompressionOrchestrator as CO

        calls = []

        def _fake_write(block, block_name, tied, rs, q_snap, fp_snap, is_model_last):
            calls.append(
                {
                    "block_name": block_name,
                    "thread": threading.current_thread().name,
                    "is_main": threading.current_thread() is threading.main_thread(),
                    "q": q_snap,
                    "fp": fp_snap,
                    "last": is_model_last,
                }
            )

        orch = SimpleNamespace(_write_finished_block_=_fake_write)
        import auto_round.compressors.orchestrator as orch_mod

        q, fp = object(), object()
        t = orch_mod.CompressionOrchestrator._start_bg_finish_block(
            orch, "blk", "model.layers.0", set(), None, q, fp, False
        )
        t.join(timeout=10)
        assert not t.is_alive()
        assert getattr(t, "autoround_state", {}).get("exc") is None, "worker must not fail"
        assert len(calls) == 1
        assert calls[0]["block_name"] == "model.layers.0"
        assert not calls[0]["is_main"], "finish must run off the main thread"
        assert calls[0]["q"] is q and calls[0]["fp"] is fp and calls[0]["last"] is False

    def test_finish_worker_failure_surfaces_at_join(self):
        import threading

        import auto_round.compressors.orchestrator as orch_mod

        def _boom(*a, **kw):
            raise RuntimeError("snapshot exploded")

        orch = SimpleNamespace(_write_finished_block_=_boom)
        t = orch_mod.CompressionOrchestrator._start_bg_finish_block(
            orch, "blk", "model.layers.0", set(), None, None, None, False
        )
        t.join(timeout=10)
        with pytest.raises(RuntimeError, match="refusing to continue"):
            orch_mod.CompressionOrchestrator._join_bg_pack(t)
        assert threading.main_thread()  # sanity: back on main
