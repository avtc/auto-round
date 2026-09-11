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

import copy
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
    plan = _CpuPlan(world)
    group = ZeroReplicaGroup(block, plan)
    return block, group


def _lr_map(block, lr=0.5):
    out = {}
    for _, m in block.named_modules():
        if is_zero_candidate(m):
            out[id(m.params["v"])] = lr
    return out


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
        group = ZeroReplicaGroup(block, _CpuPlan(2))
        # the shards -- not the (shape-shared, transient) stages -- are the
        # source of truth and hold the bitwise-exact original values
        for e in group.entries:
            vals = torch.cat([group._v_shard[o][e.uid] for o in range(group.world)])
            assert torch.equal(vals, v0[e.module_name].reshape(-1))
        group.capture_best()
        with torch.no_grad():
            # post-capture shard pollution must NOT leak into teardown output
            group._v_shard[0][group.entries[0].uid].add_(123.0)
        group.step(_lr_map(block))  # g_shards are zero -> sign(0)=0, no-op
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
            group = ZeroReplicaGroup(block, _CpuPlan(world))
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
            group.step(lrs)
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


class TestExchangeReset:
    def test_reset_exchange_clears_inboxes_and_grad_shards(self):
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        assert any(
            any(s is not None for s in group._inbox[o][e.uid]) for o in range(group.world) for e in group.entries
        )
        group.reduce_inboxes()
        # folding clears the slots (R1-3): a grad-less later iteration folds zero
        assert any(group._g_shard[o][e.uid].abs().sum() > 0 for o in range(group.world) for e in group.entries)
        for o in range(group.world):
            for e in group.entries:
                assert all(s is None for s in group._inbox[o][e.uid])
        group.reset_exchange()
        for o in range(group.world):
            for e in group.entries:
                assert all(s is None for s in group._inbox[o][e.uid])
                assert float(group._g_shard[o][e.uid].abs().sum()) == 0.0
        group.teardown()

    def test_sync_grads_allreduces_minmax_only(self):
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        # minmax params keep classic per-replica grads (no hooks on them)
        minmax_per_replica = []
        for rep in group.replicas:
            ps = []
            for _, m in rep.named_modules():
                if is_zero_candidate(m):
                    ps.append(m.params["scale_max"])
            minmax_per_replica.append(ps)
        # craft distinguishable grads: replica r gets value r+1 on every element
        for r, ps in enumerate(minmax_per_replica):
            for p in ps:
                p.grad = torch.full_like(p, float(r + 1))
        group.sync_grads(minmax_per_replica)
        for r, ps in enumerate(minmax_per_replica):
            for p in ps:
                # mean over replicas of (1, 2) == 1.5
                assert torch.allclose(p.grad, torch.full_like(p, 1.5), atol=1e-2)
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


class TestScheduler:
    def test_stage_buffers_reused_across_iterations(self):
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        stage_ids_before = {k: id(t) for r in range(group.world) for k, t in group._stages[r].items()}
        for _ in range(3):
            group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
            group.sync_grads(_minmax_per_replica(group))
            group.step(_lr_map(block))
        stage_ids_after = {k: id(t) for r in range(group.world) for k, t in group._stages[r].items()}
        assert stage_ids_before == stage_ids_after
        group.teardown()

    def test_backward_recompute_regathers_per_module(self):
        """Same-shape modules share one stage; only the backward recompute's
        re-gather can give module 1 its OWN gradient after module 2's forward
        overwrote the shared stage."""
        torch.manual_seed(3)
        block = _ToyBlock(2)
        # make the two modules' values distinguishable
        with torch.no_grad():
            block.layers[0].v.fill_(0.10)
            block.layers[1].v.fill_(-0.10)
        group = ZeroReplicaGroup(block, _CpuPlan(2))
        x = torch.randn(5, 6)

        # engine grad for module 0 (world=2 halves the batch, but grads on
        # round leaves deposit the SUM of shard contributions pre-mean --
        # compare against a reference scaled accordingly)
        ref_block = copy.deepcopy(block)
        x_all = torch.cat([x, x])
        out = ref_block(x_all)
        loss = torch.mean(out**2)
        g_ref = torch.autograd.grad(loss, [ref_block.layers[0].params["v"]])[0]

        def rep_step(r):
            loss_r = torch.mean(group.replicas[r](x) ** 2)
            loss_r.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        group.reduce_inboxes()
        # engine grad shard for module 0: each replica contributed
        # d(mean over its shard) -- two equal shards sum to the full-batch
        # mean gradient (no 1/world scaling in the deposit, matching the
        # full-mirror lane where sync divides by world at exchange; here the
        # owner folds raw deposits, so compare after multiplying by world)
        e0 = next(e for e in group.entries if e.module_name == "layers.0")
        g_engine = torch.cat([group._g_shard[o][e0.uid] for o in range(group.world)])
        # deposits hold the SUM of the two equal-size shard-mean grads, i.e.
        # world x the full-batch mean gradient (sign-invariant at the step)
        assert torch.allclose(g_engine, group.world * g_ref.reshape(-1), atol=5e-2)
        group.teardown()

    def test_multi_iteration_progression_matches_reference(self):
        """Three iterations of the full ZeRO cycle == three manual
        full-batch sign steps (bf16 transport tolerance)."""
        torch.manual_seed(23)
        block = _ToyBlock(2)
        xs = [torch.randn(3, 6) for _ in range(4)]
        ws = [torch.randn(3, 4) for _ in range(4)]
        lr = 0.1
        group = ZeroReplicaGroup(block, _CpuPlan(2))
        shards = [[0, 2], [1, 3]]
        for _it in range(3):
            losses = [None, None]

            def rep_step(r):
                rep = group.replicas[r]
                x = torch.cat([xs[j] for j in shards[r]])
                y = torch.cat([ws[j] for j in shards[r]])
                loss = torch.mean((rep(x) - y) ** 2)
                losses[r] = loss.item()
                loss.backward()

            group.run_threaded([lambda r=r: rep_step(r) for r in range(2)])
            group.reduce_inboxes()
            group.step(_lr_map(block, lr))
        group.capture_best()
        group.teardown()
        zero_vs = {n: m.params["v"].detach().clone() for n, m in block.named_modules() if is_zero_candidate(m)}

        torch.manual_seed(23)
        ref = _ToyBlock(2)
        for _it in range(3):
            out = ref(torch.cat(xs))
            loss = torch.mean((out - torch.cat(ws)) ** 2)
            loss.backward()
            with torch.no_grad():
                for _, m in ref.named_modules():
                    if is_zero_candidate(m):
                        m.params["v"].sub_(torch.sign(m.params["v"].grad), alpha=lr)
                        m.params["v"].grad = None
        for n, m in ref.named_modules():
            if is_zero_candidate(m):
                assert torch.allclose(zero_vs[n], m.params["v"].detach(), atol=5e-2), n


class TestEngagement:
    def test_world_one_rejected(self):
        block = _ToyBlock(1)
        with pytest.raises(ValueError):
            ZeroReplicaGroup(block, _CpuPlan(1))

    def test_no_round_params_rejected(self):
        class _Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        with pytest.raises(ValueError):
            ZeroReplicaGroup(_Plain(), _CpuPlan(2))


def _minmax_per_replica(group):
    out = []
    for rep in group.replicas:
        ps = []
        for _, m in rep.named_modules():
            if is_zero_candidate(m):
                ps.append(m.params["scale_max"])
        out.append(ps)
    return out


class TestR1Regressions:
    def test_mirror_stages_track_shard_updates(self):
        """R1-1 regression: every replica's shell must gather ITS OWN stage
        from the current shards, independent of which worker thread runs it."""
        block, group = _make_group()
        with torch.no_grad():
            group._v_shard[1][group.entries[0].uid].fill_(7.0)
        # forward replica 1 (runs in this thread -- shells are thread-agnostic)
        group.replicas[1](torch.randn(2, 6))
        e = group.entries[0]
        stage1 = group._stages[1][tuple(e.param_by_replica[1].shape)].reshape(-1)
        lo, hi = e.bounds[1]
        assert torch.equal(stage1[lo:hi], group._v_shard[1][e.uid])
        group.teardown()

    def test_wrapper_seen_once_in_module_walk(self):
        """R1-2 regression: the shell must not double-register the wrapper,
        which duplicated every tuning param in optimizer collections."""
        block, group = _make_group(n_modules=2)
        for rep in group.replicas:
            with_orig = [n for n, m in rep.named_modules() if hasattr(m, "orig_layer")]
            assert len(with_orig) == 2, with_orig  # exactly one per toy module
        group.teardown()

    def test_stale_inbox_cleared_after_fold(self):
        """R1-3 regression: a later iteration with no deposit must fold ZERO,
        not the previous iteration's slots."""
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        group.reduce_inboxes()
        e = group.entries[0]
        assert float(torch.cat([group._g_shard[o][e.uid] for o in range(group.world)]).abs().sum()) > 0
        # iteration 2: no forward/backward at all
        group.reduce_inboxes()
        for o in range(group.world):
            assert float(group._g_shard[o][e.uid].abs().sum()) == 0.0
            assert all(s is None for s in group._inbox[o][e.uid])
        group.teardown()

    def test_shell_delegates_wrapper_attributes(self):
        block, group = _make_group()
        shell = next(m for m in block.modules() if type(m).__name__ == "_ZeROShell")
        assert shell.orig_layer is shell.wrapped.orig_layer
        # plain attribute of the wrapped linear (e.g. its weight)
        assert shell.weight is shell.wrapped.weight
        group.teardown()

    def test_hooks_removed_and_pool_shutdown_at_teardown(self):
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        pool = group._pool
        group.teardown()
        assert group._hooks == []
        assert group._pool is None
        assert group.replicas == [block]


class TestR2Regressions:
    def test_world3_distinct_params_per_replica(self):
        """R2-1 regression: each replica must bind its OWN param copy -- the
        flat mirror map kept only the last mirror, so at world>=3 one mirror
        tuned stale and another got duplicate deposit hooks."""
        torch.manual_seed(5)
        block = _ToyBlock(2)
        group = ZeroReplicaGroup(block, _CpuPlan(3))
        for e in group.entries:
            ids = [id(p) for p in e.param_by_replica]
            assert len(set(ids)) == group.world, f"{e.uid}: {ids}"
            # exactly one deposit hook per replica param
            for r in range(group.world):
                p = e.param_by_replica[r]
                hooks = getattr(p, "_post_accumulate_grad_hooks", None)
                assert hooks is not None and len(hooks) == 1, (e.uid, r)
        # every replica's stage tracks its own shards
        with torch.no_grad():
            group._v_shard[1][group.entries[0].uid].fill_(3.0)
        group.replicas[1](torch.randn(2, 6))
        e = group.entries[0]
        stage = group._stages[1][tuple(e.param_by_replica[1].shape)].reshape(-1)
        lo, hi = e.bounds[1]
        assert torch.equal(stage[lo:hi], group._v_shard[1][e.uid])
        group.teardown()

    def test_engine_rejects_momentum(self):
        block = _ToyBlock(1)
        with pytest.raises(ValueError, match="momentum"):
            ZeroReplicaGroup(block, _CpuPlan(2), momentum=0.9)

    def test_collect_best_params_filters(self):
        from auto_round.compressors.utils import collect_best_params

        block = _ToyBlock(1)
        out_all = collect_best_params(block)
        out_round = collect_best_params(block, exclude_round=True)
        out_mm = collect_best_params(block, exclude_minmax=True)
        n = "layers.0"
        assert set(out_all[n]) == {"v", "scale_max"}
        assert set(out_round[n]) == {"scale_max"}
        assert set(out_mm[n]) == {"v"}
        assert collect_best_params(block, exclude_round=True, exclude_minmax=True) == {}

    def test_shell_missing_attribute_raises(self):
        block, group = _make_group()
        shell = next(m for m in block.modules() if type(m).__name__ == "_ZeROShell")
        with pytest.raises(AttributeError):
            shell.definitely_not_a_real_attribute
        group.teardown()

    def test_sync_grads_geometry_mismatch_warns_and_skips(self):
        block, group = _make_group()

        def rep_step(r):
            loss = torch.mean(group.replicas[r](torch.randn(3, 6)) ** 2)
            loss.backward()

        group.run_threaded([lambda r=r: rep_step(r) for r in range(group.world)])
        minmax = _minmax_per_replica(group)
        for ps in minmax:
            for p in ps:
                p.grad = torch.full_like(p, 1.0)
        # hand in only ONE replica's list: geometry mismatch -> skip, grads untouched
        group.sync_grads(minmax[:1])
        for r, ps in enumerate(minmax):
            for p in ps:
                assert torch.allclose(p.grad, torch.full_like(p, 1.0))
        group.teardown()


class TestR3:
    def test_flat_single_wrapper_block_rejected(self):
        with pytest.raises(ValueError, match="container block"):
            ZeroReplicaGroup(_ToyLinear(), _CpuPlan(2))


class TestBlockHasTuningEntries:
    """Shared DDP decline check: all-float blocks, minmax-only blocks."""

    def test_plain_block_has_no_entries(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        assert block_has_tuning_entries(torch.nn.Sequential(torch.nn.Linear(8, 8))) is False

    def test_minmax_only_block_counts_as_tunable(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class MinmaxOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"wmax": nn.Parameter(torch.ones(1))}

        # full-mirror can still tune minmax params; only the zero lane declines
        assert block_has_tuning_entries(torch.nn.Sequential(MinmaxOnly())) is True

    def test_round_block_counts(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class RoundHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"v": nn.Parameter(torch.ones(4))}

        assert block_has_tuning_entries(torch.nn.Sequential(RoundHolder())) is True


class TestBlockHasRoundEntries:
    """All-float pinned / minmax-only blocks must be detectable before engagement."""

    def test_plain_block_has_no_entries(self):
        from auto_round.algorithms.quantization.sign_round.zero2 import block_has_round_entries

        block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        assert block_has_round_entries(block) is False

    def test_minmax_only_block_has_no_entries(self):
        from auto_round.algorithms.quantization.sign_round.zero2 import block_has_round_entries

        class MinmaxOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"wmax": nn.Parameter(torch.ones(1)), "wmin": nn.Parameter(torch.zeros(1))}

        block = torch.nn.Sequential(MinmaxOnly())
        assert block_has_round_entries(block) is False

    def test_block_with_round_entry_is_detected(self):
        from auto_round.algorithms.quantization.sign_round.zero2 import block_has_round_entries

        class RoundHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"v": nn.Parameter(torch.ones(4)), "wmax": nn.Parameter(torch.ones(1))}

        block = torch.nn.Sequential(torch.nn.Linear(8, 8), RoundHolder())
        assert block_has_round_entries(block) is True
