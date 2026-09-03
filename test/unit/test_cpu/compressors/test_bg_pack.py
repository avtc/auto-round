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
"""Background pack pipeline (default ON; AR_DISABLE_BG_PACK opts out).

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
    def test_default_on_and_opt_out(self, monkeypatch):
        from auto_round import envs

        monkeypatch.delenv("AR_DISABLE_BG_PACK", raising=False)
        assert envs.AR_DISABLE_BG_PACK is False
        monkeypatch.setenv("AR_DISABLE_BG_PACK", "1")
        assert envs.AR_DISABLE_BG_PACK is True


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
