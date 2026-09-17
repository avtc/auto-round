# coding=utf-8 -*-
# Copyright (c) 2026, AMD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF CONDITIONS OF ANY KIND, either express
# implied. See the License for the specific language governing permissions and
# limitations under the License.

"""AWQ sharded grid search: the parallel seam evaluates the smoothing grid on
per-shard deepcopies of the block and merges per-point loss SUMS; the merged
losses reproduce the serial loop's choice (same best ratio, same scales)."""

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.sign_round.tune_parallel import TuneParallelContext
from auto_round.algorithms.transforms.awq.base import AWQTransform
from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping
from auto_round.algorithms.transforms.awq.qdq import QDQTool


class _FakeBlock(nn.Module):
    """Parent IS the block; the balance layer is the inner linear."""

    def __init__(self, din=8, doubt=4):
        super().__init__()
        self.lin = nn.Linear(din, doubt, bias=False)
        self.register_buffer("dummy", torch.zeros(1))

    def forward(self, x):
        return self.lin(x)


def _make_model(block):
    """Wrapper model exposing the block at ``model.layers.0``."""

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.layers = nn.ModuleList([block])
            self.model = inner

    return _Root()


def _make_transform(n_grid=4, model=None):
    from types import SimpleNamespace

    tr = AWQTransform.__new__(AWQTransform)
    # model_context is a read-only property backed by the run context
    tr._BaseAlgorithm__run_ctx = SimpleNamespace(model_context=SimpleNamespace(model=model))
    tr.n_grid = n_grid
    tr.duo_scaling = False
    tr._awq_seqlen = None
    tr._smooth_batch_size = None
    tr._parent_args_cache = {}
    tr._parallel_reduce = None
    tr._qdq_tool = QDQTool(bits=4, group_size=-1, sym=True, data_type="int")
    return tr


def _make_mapping(block):
    block.global_name = "model.layers.0"
    block.lin.global_name = "model.layers.0.lin"
    return ResolvedMapping(
        smooth_name="model.layers.0.smooth",
        smooth_layer=block.lin,
        balance_names=["model.layers.0.lin"],
        balance_layers=[block.lin],
        parent_name="model.layers.0",
        parent=block,
    )


def _run_search(tr, block, mapping, x_mean, calls):
    tr._parent_args_cache[mapping.parent] = [(args, kwargs) for args, kwargs in calls]
    return tr._grid_search_scales(mapping, x_mean, block_prefix=block.global_name)


def _make_calls(block, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [((torch.randn(2, block.lin.in_features, generator=g),), {}) for _ in range(n)]


class TestShardedGridParity:
    def test_sharded_choice_matches_serial(self):
        torch.manual_seed(7)
        block = _FakeBlock()
        serial_tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        calls = _make_calls(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5

        serial = _run_search(serial_tr, block, mapping, x_mean, calls)

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr = _make_transform(model=_make_model(block))
        engaged = {"merged": None}
        orig_sharded = AWQTransform._sharded_grid_losses

        def spy(self, *a, **kw):
            engaged["merged"] = orig_sharded(self, *a, **kw)
            return engaged["merged"]

        AWQTransform._sharded_grid_losses = spy
        try:
            tr.set_parallel_reduce(ctx.reduce)
            sharded = _run_search(tr, block, mapping, x_mean, calls)
        finally:
            AWQTransform._sharded_grid_losses = orig_sharded

        assert engaged["merged"] is not None, "sharded path silently declined"
        assert serial is not None and sharded is not None
        assert torch.allclose(serial, sharded, atol=1e-6)
        # home weights untouched by the sharded walk
        for args, kwargs in calls:
            ref = block(*args, **kwargs)
        assert torch.isfinite(ref).all()

    def test_uneven_calls_shard_and_match_serial(self):
        """3 calls over 2 replicas: ceil/floor split, every call kept, the
        merged losses reproduce the serial choice (sum-merge is
        partition-invariant)."""
        torch.manual_seed(11)
        block = _FakeBlock()
        serial_tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        calls = _make_calls(block, n=3)
        x_mean = torch.rand(block.lin.in_features) + 0.5
        serial = _run_search(serial_tr, block, mapping, x_mean, calls)

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr = _make_transform(model=_make_model(block))
        engaged = {"merged": None}
        orig_sharded = AWQTransform._sharded_grid_losses

        def spy(self, *a, **kw):
            engaged["merged"] = orig_sharded(self, *a, **kw)
            return engaged["merged"]

        AWQTransform._sharded_grid_losses = spy
        try:
            tr.set_parallel_reduce(ctx.reduce)
            sharded = _run_search(tr, block, mapping, x_mean, calls)
        finally:
            AWQTransform._sharded_grid_losses = orig_sharded

        assert engaged["merged"] is not None, "uneven split must shard, not decline"
        assert serial is not None and sharded is not None
        assert torch.allclose(serial, sharded, atol=1e-6)

    def test_single_call_declines_to_serial(self):
        torch.manual_seed(13)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        calls = _make_calls(block, n=1)
        x_mean = torch.rand(block.lin.in_features) + 0.5
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr.set_parallel_reduce(ctx.reduce)
        out = _run_search(tr, block, mapping, x_mean, calls)
        assert out is not None  # serial answered

    def test_out_of_block_parent_warns_and_runs_serial(self):
        torch.manual_seed(5)
        block = _FakeBlock()
        model = _make_model(block)
        outer = nn.Module()
        outer.model = model  # block still reachable, but the parent sits at root level
        # parent = a module whose global_name lies outside the block prefix
        block.global_name = "model.layers.0"
        block.lin.global_name = "model.layers.0.lin"

        tr = _make_transform(model=_make_model(block))
        from auto_round.algorithms.transforms.awq.mappings import ResolvedMapping

        class _Outside(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 4, bias=False)

            def forward(self, x):
                return self.lin(x)

        outside = _Outside()
        outside.global_name = "outside.holder"
        tr._BaseAlgorithm__run_ctx = None  # rebuilt below with the outside module in the model
        from types import SimpleNamespace

        root = _make_model(block)
        root.outside = outside
        tr._BaseAlgorithm__run_ctx = SimpleNamespace(model_context=SimpleNamespace(model=root))
        mapping = ResolvedMapping("s", block.lin, ["n"], [block.lin], "p", outside)
        calls = [((torch.randn(2, 8),), {}) for _ in range(4)]
        tr._parent_args_cache[outside] = list(calls)
        x_mean = torch.rand(block.lin.in_features) + 0.5

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr.set_parallel_reduce(ctx.reduce)
        warned = []
        import auto_round.algorithms.transforms.awq.base as awq_base

        orig_warn = awq_base.logger.warning
        awq_base.logger.warning = lambda *a, **kw: warned.append(a[0] % a[1:] if len(a) > 1 else a[0])
        try:
            out = tr._grid_search_scales(mapping, x_mean, block_prefix="model.layers.0")
        finally:
            awq_base.logger.warning = orig_warn
        assert out is not None  # serial loop answered
        assert any("outside block" in w for w in warned)

    def test_clip_search_sharded_matches_serial(self):
        """The clip search rides the map seam: per-layer results, concatenated
        in shard order, identical to the serial per-layer search."""
        torch.manual_seed(17)
        block = _FakeBlock()
        model = _make_model(block)
        block.global_name = "model.layers.0"
        block.lin.global_name = "model.layers.0.lin"
        tr = _make_transform(model=model)
        tr._clip_input_feat = {}
        tr.clip_n_sample_token = 512
        tr.clip_n_grid = 20
        tr.clip_max_shrink = 0.5
        from auto_round.algorithms.transforms.awq.qdq import QDQTool

        tr._qdq_tool = QDQTool(bits=4, group_size=-1, sym=True, data_type="int")
        g = torch.Generator().manual_seed(2)
        feats = [torch.randn(64, block.lin.in_features, generator=g) for _ in range(4)]
        lins = [block.lin] + [torch.nn.Linear(block.lin.in_features, 4, bias=False) for _ in range(3)]
        jobs = [(lin, feat, f"name{i}") for i, (lin, feat) in enumerate(zip(lins, feats))]

        serial = [tr._compute_best_clip(lin, feat) for lin, feat, _ in jobs]

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr.set_parallel_reduce(ctx.reduce, map_fn=ctx.map)
        sharded = tr._sharded_clip_results("model.layers.0", jobs)
        assert sharded is not None
        for set, par in zip(serial, sharded):
            assert set is not None and par is not None
            assert torch.allclose(set[0], par[0], atol=1e-6)
            assert torch.allclose(set[1], par[1], atol=1e-6)

    def test_parallel_reduce_none_by_default(self):
        tr = _make_transform()
        assert tr._parallel_reduce is None


class TestGridSplitPerfLine:
    """AR_PERF_COUNTERS-gated qdq/replay split for both grid-search modes."""

    def _capture(self):
        import logging as _logging

        from auto_round.logger import logger as ar_logger

        records = []

        class _Handler(_logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Handler(level=_logging.INFO)
        ar_logger.addHandler(handler)
        try:
            yield records
        finally:
            ar_logger.removeHandler(handler)

    def test_serial_split_line_gated(self):
        import auto_round.envs as envs

        torch.manual_seed(11)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5

        gen = self._capture()
        records = next(gen)
        try:
            prev = getattr(envs, "AR_PERF_COUNTERS", False)
            had_attr = "AR_PERF_COUNTERS" in vars(envs)
            envs.AR_PERF_COUNTERS = False
            _run_search(tr, block, mapping, x_mean, _make_calls(block))
            assert not any("awq grid split" in r for r in records)

            envs.AR_PERF_COUNTERS = True
            _run_search(tr, block, mapping, x_mean, _make_calls(block))
            lines = [r for r in records if "awq grid split" in r]
            assert len(lines) == 1, lines
            assert "mode=serial" in lines[0]
            for key in ("refs=", "qdq=", "replay="):
                assert key in lines[0]
        finally:
            # a real module attr would shadow the dynamic env-var lookup
            if had_attr:
                envs.AR_PERF_COUNTERS = prev
            else:
                delattr(envs, "AR_PERF_COUNTERS")

    def test_sharded_split_line_gated(self):
        import auto_round.envs as envs

        torch.manual_seed(13)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr.set_parallel_reduce(ctx.reduce)

        gen = self._capture()
        records = next(gen)
        try:
            prev = getattr(envs, "AR_PERF_COUNTERS", False)
            had_attr = "AR_PERF_COUNTERS" in vars(envs)
            envs.AR_PERF_COUNTERS = True
            _run_search(tr, block, mapping, x_mean, _make_calls(block))
            lines = [r for r in records if "awq grid split" in r]
            assert len(lines) == 2, lines  # sharded buckets + coordinator final
            assert "mode=sharded" in lines[0] and "prep=" in lines[0]
            assert "name=" in lines[0]
            assert "mode=final" in lines[1] and "name=" in lines[1]
        finally:
            # a real module attr would shadow the dynamic env-var lookup
            if had_attr:
                envs.AR_PERF_COUNTERS = prev
            else:
                delattr(envs, "AR_PERF_COUNTERS")


class TestShardedReplayStaging:
    """Per-replica args staging: one move per item, reused across all grid points."""

    def test_moves_args_once_per_item(self, monkeypatch):
        import auto_round.algorithms.transforms.awq.base as awq_base

        torch.manual_seed(17)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5
        calls = _make_calls(block, n=4)

        moved = {"n": 0}
        orig_move = awq_base.move_to_device

        def counting_move(v, dev):
            if isinstance(v, torch.Tensor):
                moved["n"] += 1
            return orig_move(v, dev)

        monkeypatch.setattr(awq_base, "move_to_device", counting_move)

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr.set_parallel_reduce(ctx.reduce)
        tr.n_grid = 4  # 4 points -> without the hoist: 4 points x n calls + refs

        _run_search(tr, block, mapping, x_mean, calls)

        # one call arg per item, moved exactly once (refs + 4 points reuse it)
        assert moved["n"] == len(calls), moved["n"]

    def test_parity_holds_after_staging(self):
        torch.manual_seed(19)
        block = _FakeBlock()
        serial_tr = _make_transform(model=_make_model(block))
        mapping = _make_mapping(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5
        calls = _make_calls(block)

        serial = _run_search(serial_tr, block, mapping, x_mean, calls)

        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr = _make_transform(model=_make_model(block))
        tr.set_parallel_reduce(ctx.reduce)
        sharded = _run_search(tr, block, mapping, x_mean, calls)

        assert serial is not None and sharded is not None
        assert torch.allclose(serial, sharded, atol=1e-6)


class TestCaptureParkingGate:
    """Parent-args capture parks to CPU only under low_gpu_mem."""

    def _capture_with(self, low_gpu_mem):
        from types import SimpleNamespace

        torch.manual_seed(23)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        tr._block_mappings = {"model.layers.0": [_make_mapping(block)]}
        tr._activation_stats = {}
        tr._clip_input_feat = {}
        tr.apply_clip = False
        tr._BaseAlgorithm__run_ctx = SimpleNamespace(
            model_context=SimpleNamespace(model=_make_model(block)),
            compress_context=SimpleNamespace(low_gpu_mem_usage=low_gpu_mem),
        )
        mapping = _make_mapping(block)
        calls = _make_calls(block, n=2)

        handles = tr.register_fp_input_forward_hooks(block)
        try:
            for args, kwargs in calls:
                block(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
        captured = tr._parent_args_cache[mapping.parent]
        return captured, calls

    def test_parks_when_low_gpu_mem(self):
        captured, calls = self._capture_with(low_gpu_mem=True)
        assert len(captured) == len(calls)
        for (cargs, _), (oargs, _) in zip(captured, calls):
            assert cargs[0].device.type == "cpu"

    def test_keeps_on_device_when_vram_allowed(self):
        captured, calls = self._capture_with(low_gpu_mem=False)
        assert len(captured) == len(calls)
        # same values, decision attribute reflects the gate
        for (cargs, _), (oargs, _) in zip(captured, calls):
            assert torch.equal(cargs[0], oargs[0].detach())

    def test_explicit_park_overrides_vram_allowance(self):
        """The composer's serial-lane park_capture=True wins over low_gpu_mem=False."""
        from types import SimpleNamespace

        torch.manual_seed(29)
        block = _FakeBlock()
        tr = _make_transform(model=_make_model(block))
        tr._block_mappings = {"model.layers.0": [_make_mapping(block)]}
        tr._activation_stats = {}
        tr._clip_input_feat = {}
        tr.apply_clip = False
        tr._BaseAlgorithm__run_ctx = SimpleNamespace(
            model_context=SimpleNamespace(model=_make_model(block)),
            compress_context=SimpleNamespace(low_gpu_mem_usage=False),
        )
        calls = _make_calls(block, n=1)
        handles = tr.register_fp_input_forward_hooks(block, park_capture=True)
        try:
            for args, kwargs in calls:
                block(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
        cargs, _ = tr._parent_args_cache[_make_mapping(block).parent][0]
        assert cargs[0].device.type == "cpu"


class TestGridLossParityControlled:
    """Per-point grid losses must match serial vs sharded on identical inputs.

    AWQ's analog of the tune lane's controlled-draws parity test: with the data
    pinned (same block, same cached calls), the only difference between the two
    lanes is the fp32 summation split across shard partials.
    """

    def test_per_point_losses_match_serial(self):
        torch.manual_seed(37)
        block = _FakeBlock()
        mapping = _make_mapping(block)
        x_mean = torch.rand(block.lin.in_features) + 0.5
        calls = _make_calls(block, n=4)

        # serial: record each grid point's loss via _compute_parent_loss spy
        serial_losses = []
        orig_cpl = AWQTransform._compute_parent_loss

        def cpl_spy(self, parent, kwargs_list, fp16_outputs):
            val = orig_cpl(self, parent, kwargs_list, fp16_outputs)
            serial_losses.append(float(val))
            return val

        serial_tr = _make_transform(model=_make_model(block))
        AWQTransform._compute_parent_loss = cpl_spy
        try:
            serial_scales = _run_search(serial_tr, block, mapping, x_mean, calls)
        finally:
            AWQTransform._compute_parent_loss = orig_cpl
        assert serial_losses and all(v != float("inf") for v in serial_losses)

        # sharded: capture the merged per-point sums + count
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        tr = _make_transform(model=_make_model(block))
        engaged = {"merged": None}
        orig_sharded = AWQTransform._sharded_grid_losses

        def spy(self, *a, **kw):
            engaged["merged"] = orig_sharded(self, *a, **kw)
            return engaged["merged"]

        AWQTransform._sharded_grid_losses = spy
        try:
            tr.set_parallel_reduce(ctx.reduce)
            sharded_scales = _run_search(tr, block, mapping, x_mean, calls)
        finally:
            AWQTransform._sharded_grid_losses = orig_sharded

        merged = engaged["merged"]
        assert merged is not None
        n_grid = len(serial_losses)
        per_point = (merged[:n_grid] / merged[-1].clamp(min=1)).tolist()
        for s_val, p_val in zip(serial_losses, per_point):
            assert abs(s_val - p_val) <= 1e-6 + 1e-5 * abs(s_val), (s_val, p_val)

        # the chosen candidate is the same point in both lanes
        assert torch.allclose(serial_scales, sharded_scales, atol=1e-6)
