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

"""CPU-tier tests for the ZeRO-2-lite tune-state sharding engine.

All tests are hermetic: a tiny toy "wrapper" module mimics the real
wrapper contract (``params`` dict with a round key ``v`` and min/max keys,
an ``orig_layer`` marker, forward reading ``params['v']``), so no model
download or GPU is needed. The parity anchor compares one ZeRO iteration
(fwd/bwd on disjoint shards + deposit/reduce + shard step) against the
mathematically equivalent full-batch reference.
"""

import types

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.quantization.sign_round.zero2 import (
    ZeroReplicaGroup,
    _is_round_key,
    is_zero_candidate,
    split_bounds,
)


class _ToyLinear(nn.Module):
    def __init__(self, fan_in=6, fan_out=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(fan_out, fan_in, generator=g))
        self.v = nn.Parameter(torch.randn(fan_out, fan_in, generator=g) * 0.05)
        self.params = {"v": self.v, "scale_max": nn.Parameter(torch.ones(1))}
        self.orig_layer = types.SimpleNamespace(bits=4)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight + self.v * 0.1)


class _ToyBlock(nn.Module):
    def __init__(self, n_modules=2):
        super().__init__()
        first = _ToyLinear(fan_in=6, fan_out=4, seed=0)
        rest = [_ToyLinear(fan_in=4, fan_out=4, seed=i + 1) for i in range(n_modules - 1)]
        self.layers = nn.ModuleList([first] + rest)
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        for lin in self.layers:
            x = torch.tanh(lin(x))
        return self.norm(x)


class _CpuPlan:
    def __init__(self, world):
        self.world = world
        self.devices = [torch.device("cpu")] * world


def _make_group(world=2, n_modules=2):
    torch.manual_seed(7)
    block = _ToyBlock(n_modules)
    lrs = {}
    for _, m in block.named_modules():
        if is_zero_candidate(m):
            lrs[id(m.params["v"])] = 0.5
    plan = _CpuPlan(world)
    group = ZeroReplicaGroup(block, plan, lr_by_param=lrs)
    return block, group


class TestSplitBounds:
    def test_exact_and_uneven(self):
        assert split_bounds(8, 2) == [(0, 4), (4, 8)]
        assert split_bounds(7, 3) == [(0, 3), (3, 5), (5, 7)]

    def test_covers_all(self):
        for numel, world in [(13, 4), (1, 2), (64, 8)]:
            bounds = split_bounds(numel, world)
            assert bounds[0][0] == 0 and bounds[-1][1] == numel
            for (_, hi), (lo2, _) in zip(bounds, bounds[1:]):
                assert hi == lo2


class TestKeys:
    def test_round_vs_minmax(self):
        assert _is_round_key("v") and _is_round_key("bias_v")
        assert not _is_round_key("scale_max") and not _is_round_key("scale_min")

    def test_candidate(self):
        assert is_zero_candidate(_ToyLinear())
        assert not is_zero_candidate(nn.Linear(2, 2))


class TestBuildAndTeardown:
    def test_shards_are_exact_and_teardown_roundtrips(self):
        torch.manual_seed(7)
        block = _ToyBlock(2)
        v0 = {n: m.params["v"].detach().clone() for n, m in block.named_modules() if is_zero_candidate(m)}
        lrs = {}
        for _, m in block.named_modules():
            if is_zero_candidate(m):
                lrs[id(m.params["v"])] = 0.5
        group = ZeroReplicaGroup(block, _CpuPlan(2), lr_by_param=lrs)
        # the shards -- not the (shape-shared, transient) stages -- are the
        # source of truth and hold the bitwise-exact original values
        for e in group.entries:
            vals = torch.cat([group._v_shard[o][e.uid] for o in range(group.world)])
            assert torch.equal(vals, v0[e.module_name].reshape(-1))
        group.capture_best()
        with torch.no_grad():
            # post-capture shard pollution must NOT leak into teardown output
            group._v_shard[0][group.entries[0].uid].add_(123.0)
        group.step()  # g_shards are zero -> sign(0)=0, no-op
        group.teardown()
        for n, m in block.named_modules():
            if is_zero_candidate(m):
                assert torch.equal(m.params["v"].data, v0[n])
        # home tree has no shells left
        for _, m in block.named_modules():
            assert type(m).__name__ != "_ZeROShell"


class TestParityAnchor:
    def test_one_iteration_matches_full_batch_reference(self):
        """ZeRO world=2 over disjoint shards == full-batch sign step.

        The toy loss is mean((f(x) - y)^2) with no masking, so the global
        grad = mean of the two shard grads; the reference applies the same
        SignRound update to the full v directly. bf16 inbox transport makes
        the exchange approximate; tolerance covers the bf16 rounding of the
        mean (sign flips only when a grad element is ~0).
        """

        def run(world):
            torch.manual_seed(11)
            block = _ToyBlock(2)
            xs = [torch.randn(3, 6) for _ in range(4)]
            ws = [torch.randn(3, 4) for _ in range(4)]
            lrs = {}
            for _, m in block.named_modules():
                if is_zero_candidate(m):
                    lrs[id(m.params["v"])] = 0.5
            group = ZeroReplicaGroup(block, _CpuPlan(world), lr_by_param=lrs)
            shards = [[0, 2], [1, 3]] if world == 2 else [list(range(4))]
            losses = [None] * world

            def rep_step(r):
                rep = group.replicas[r]
                x = torch.cat([xs[j] for j in shards[r]])
                y = torch.cat([ws[j] for j in shards[r]])
                out = rep(x)
                loss = torch.mean((out - y) ** 2)
                losses[r] = loss.item()
                loss.backward()

            group.run_threaded([lambda r=r: rep_step(r) for r in range(world)])
            group.reduce_inboxes()
            mean_loss = sum(losses) / world
            group.step()
            group.teardown()
            vs = {n: m.params["v"].detach().clone() for n, m in block.named_modules() if is_zero_candidate(m)}
            return vs, mean_loss

        zero_vs, zero_loss = run(2)
        # build the full-batch reference without the ZeRO engine
        torch.manual_seed(11)
        block = _ToyBlock(2)
        xs = [torch.randn(3, 6) for _ in range(4)]
        ws = [torch.randn(3, 4) for _ in range(4)]
        x = torch.cat(xs)
        y = torch.cat(ws)
        out = block(x)
        loss = torch.mean((out - y) ** 2)
        loss.backward()
        with torch.no_grad():
            for _, m in block.named_modules():
                if is_zero_candidate(m):
                    m.params["v"].sub_(torch.sign(m.params["v"].grad), alpha=0.5)
        assert zero_loss == pytest.approx(loss.item(), rel=1e-6)
        for n, m in block.named_modules():
            if is_zero_candidate(m):
                a, b = zero_vs[n], m.params["v"].detach()
                assert torch.allclose(a, b, atol=2e-2), (n, (a - b).abs().max())

    def test_grads_freed_during_backward(self):
        block, group = _make_group()
        seen_full = []

        def rep_step(r):
            rep = group.replicas[r]
            loss = torch.mean(rep(torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        # after backward, no round leaf carries a full-size grad
        for rep in group.replicas:
            for _, m in rep.named_modules():
                if is_zero_candidate(m):
                    assert m.params["v"].grad is None
        group.reduce_inboxes()
        group.teardown()


class TestShardMath:
    def test_stage_gather_picks_up_shard_updates(self):
        block, group = _make_group()
        # mutate shards directly, gather, verify stages reflect it
        with torch.no_grad():
            e = group.entries[0]
            group._v_shard[0][e.uid].add_(1.0)
        group._gather_replica(1)
        stage = group._stages[1][tuple(e.param_by_replica[1].shape)].reshape(-1)
        lo, hi = e.bounds[0]
        assert torch.equal(stage[lo:hi], group._v_shard[0][e.uid])
