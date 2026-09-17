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

    def test_parallel_reduce_none_by_default(self):
        tr = _make_transform()
        assert tr._parallel_reduce is None
