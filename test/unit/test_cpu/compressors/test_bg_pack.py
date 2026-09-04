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

    def test_old_env_name_gone(self):
        from auto_round import envs

        assert not hasattr(envs, "AR_DISABLE_BG_PACK")


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
