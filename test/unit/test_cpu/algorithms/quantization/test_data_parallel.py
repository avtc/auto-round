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
"""Tests for single-process data parallelism of the SignRound tuning loop."""

import copy

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.data_parallel import (
    DDPPlan,
    ReplicaGroup,
    ReplicaThreadPool,
    _DelayedBestTracker,
    _flat_grad_buffer,
    _param_grad_buffers,
    _rebuild_flat_views,
    _write_back_grads,
    build_flat_tuning_params,
    halving_doubling_allreduce,
    resolve_ddp_plan,
    run_threaded_spawn,
    sign_exchange_allreduce,
)


class TestHalvingDoublingAllreduce:
    @pytest.mark.parametrize("world", [2, 4, 8])
    @pytest.mark.parametrize("n", [96, 100])  # divisible and non-divisible by world
    def test_matches_reference_mean(self, world, n):
        torch.manual_seed(0)
        bufs = [torch.randn(n) for _ in range(world)]
        expected = torch.stack(bufs).sum(0) / world
        halving_doubling_allreduce(bufs, scale=1.0 / world)
        for b in bufs:
            assert torch.allclose(b, expected, rtol=1e-5, atol=1e-6)

    def test_all_ranks_identical_after_exchange(self):
        torch.manual_seed(1)
        bufs = [torch.randn(64) for _ in range(4)]
        halving_doubling_allreduce(bufs, scale=0.25)
        for b in bufs[1:]:
            assert torch.equal(b, bufs[0])

    def test_world_one_scales_only(self):
        b = torch.randn(8)
        halving_doubling_allreduce([b], scale=0.5)
        assert torch.allclose(b, b)  # no crash; scaled in place

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            halving_doubling_allreduce([torch.randn(4) for _ in range(3)])

    def test_bf16_exchange_close(self):
        torch.manual_seed(2)
        bufs = [torch.randn(128) for _ in range(4)]
        expected = torch.stack(bufs).sum(0) / 4
        halving_doubling_allreduce(bufs, scale=0.25, transport="bf16")
        for b in bufs:
            assert torch.allclose(b, expected, rtol=1e-2, atol=1e-2)


class TestSignExchangeAllreduce:
    @pytest.mark.parametrize("world", [2, 4, 8])
    @pytest.mark.parametrize("n", [96, 100, 5])  # divisible, non-divisible, smaller than world
    def test_matches_fp32_sign_of_mean(self, world, n):
        torch.manual_seed(0)
        bufs = [torch.randn(n) for _ in range(world)]
        bufs[0][0] = 0.0  # an exact zero in the sum must survive as sign 0
        avg = torch.stack(bufs).sum(0) / world
        expected = torch.sign(avg)
        sign_exchange_allreduce(bufs, transport="fp32")
        for b in bufs:
            assert torch.equal(b, expected)  # fp32 transport: exact signs of the mean
            assert torch.equal(b, bufs[0])  # ranks bitwise identical

    def test_bf16_transport_signs_match_outside_tiny_band(self):
        torch.manual_seed(1)
        bufs = [torch.randn(512) for _ in range(4)]
        avg = torch.stack(bufs).sum(0) / 4
        expected = torch.sign(avg)
        sign_exchange_allreduce(bufs, transport="bf16")
        mism = bufs[0] != expected
        # the reduce-scatter partials round through bf16, so a sign may flip
        # only where the averaged gradient is essentially zero
        assert (avg[mism].abs() < 5e-2).all()
        for b in bufs:
            assert torch.equal(b, bufs[0])

    def test_int8_transport_values_and_rank_identity(self):
        torch.manual_seed(2)
        bufs = [torch.randn(256) for _ in range(8)]
        sign_exchange_allreduce(bufs, transport="int8")
        for b in bufs:
            assert set(b.unique().tolist()) <= {-1.0, 0.0, 1.0}
            assert torch.equal(b, bufs[0])

    def test_world_one_signs_in_place(self):
        torch.manual_seed(3)
        vals = torch.randn(16)
        vals[4] = 0.0
        expected = torch.sign(vals)
        sign_exchange_allreduce([vals])
        assert torch.equal(vals, expected)

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            sign_exchange_allreduce([torch.randn(4) for _ in range(3)])

    def test_optim_sign_update_is_unchanged(self):
        # the optimizer step is param.add_(sign(d_p), alpha=-lr): feeding it
        # pre-signed gradients (-1/0/1) must produce the identical update as
        # feeding it the full averaged gradients
        g_avg = torch.randn(32)
        p_ref = torch.randn(32)
        p_signed = p_ref.clone()
        lr = 1e-2
        p_ref.add_(torch.sign(g_avg), alpha=-lr)
        p_signed.add_(torch.sign(torch.sign(g_avg)), alpha=-lr)
        assert torch.equal(p_ref, p_signed)


class TestGradBuffers:
    def test_flatten_writeback_roundtrip(self):
        torch.manual_seed(3)
        params = [torch.nn.Parameter(torch.zeros(10, 6)), torch.nn.Parameter(torch.zeros(7))]
        for p in params:
            p.grad = torch.randn_like(p)
        originals = [p.grad.clone() for p in params]
        (buf,) = _param_grad_buffers([params])
        assert buf.numel() == 67
        # simulate averaged buffer
        buf.mul_(2.0)
        _write_back_grads(buf, params)
        for p, o in zip(params, originals):
            assert torch.allclose(p.grad, o * 2.0)

    def test_none_grads_skipped(self):
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = None
        assert _param_grad_buffers([[p]]) == [None]


class TestResolveDDPPlan:
    def test_disabled_by_world_one(self):
        plan = resolve_ddp_plan(1, torch.device("cuda:0"), batch_size=8)
        assert plan.world == 1 and not plan.enabled

    def test_cpu_home_disabled(self):
        plan = resolve_ddp_plan(4, torch.device("cpu"), batch_size=8)
        assert plan.world == 1 and any("not CUDA" in n for n in plan.notes)

    def test_batch_not_divisible(self):
        plan = resolve_ddp_plan(3, torch.device("cuda:0"), batch_size=8)
        assert plan.world == 1 and any("divisible" in n for n in plan.notes)

    def test_rotation_from_home(self):
        plan = resolve_ddp_plan(4, torch.device("cuda:1"), batch_size=8, visible_cuda_devices=[0, 1, 2, 3])
        assert plan.devices[0] == torch.device("cuda", 1)
        assert len(plan.devices) == 4
        assert plan.shard_size == 2

    def test_vram_guard_skips_mirror(self):
        free = {torch.device("cuda", 1): 1 << 30}  # 1GiB free on mirror candidate
        plan = resolve_ddp_plan(
            2,
            torch.device("cuda:0"),
            batch_size=8,
            visible_cuda_devices=[0, 1],
            vram_free_bytes=free,
            mirror_footprint_bytes=2 << 30,
        )
        assert plan.world == 1 and any("VRAM guard" in n or "skip" in n for n in plan.notes)

    def test_explicit_devices(self):
        plan = resolve_ddp_plan(
            3, torch.device("cuda:0"), batch_size=12, explicit_devices=["cuda:0", "cuda:5", "cuda:2"]
        )
        assert plan.devices == [torch.device("cuda:0"), torch.device("cuda:5"), torch.device("cuda:2")]


class TestReplicaGroup:
    def _block(self):
        torch.manual_seed(0)
        block = torch.nn.Sequential()
        wrapper = torch.nn.Module()
        wrapper.orig_layer = torch.nn.Linear(8, 8)  # separate original, like WrapperLinear
        wrapper.params = {"v": torch.nn.Parameter(torch.zeros(8, 8))}
        block.add_module("m", wrapper)
        return block

    def test_mirrors_created_and_torn_down(self):
        block = self._block()
        plan = DDPPlan(world=3, devices=[torch.device("cpu")] * 3, shard_size=8)
        group = ReplicaGroup(block, plan)
        assert group.world == 3  # cpu home + 2 mirrors (same device, still separate copies)
        assert len(group.mirrors) == 2
        # mirrors are independent copies
        group.mirrors[0].m.params["v"].data.add_(1.0)
        assert group.home.m.params["v"].abs().sum() == 0
        group.teardown()
        assert group.mirrors == []

    def test_sync_grads_averages(self):
        block = self._block()
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=8)
        group = ReplicaGroup(block, plan)
        per_replica = group.round_params()
        for rep_ps, val in zip(per_replica, [1.0, 3.0]):
            for p in rep_ps:
                p.grad = torch.full_like(p, val)
        group.sync_grads(per_replica)
        for rep_ps in per_replica:
            for p in rep_ps:
                assert torch.allclose(p.grad, torch.full_like(p, 2.0))
        group.teardown()

    def test_sync_grads_sign_exchange(self):
        block = self._block()
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=8)
        group = ReplicaGroup(block, plan)
        per_replica = group.round_params()
        for rep_ps, val in zip(per_replica, [1.0, 3.0]):
            for p in rep_ps:
                p.grad = torch.full_like(p, val)
        group.sync_grads(per_replica, sign_exchange=True)
        for rep_ps in per_replica:
            for p in rep_ps:
                # signs of the averaged gradient (2.0), not the average
                assert torch.equal(p.grad, torch.ones_like(p.grad))
        group.teardown()

    def test_sync_grads_sign_exchange_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_DDP_SIGN_EXCHANGE", "0")
        block = self._block()
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=8)
        group = ReplicaGroup(block, plan)
        per_replica = group.round_params()
        for rep_ps, val in zip(per_replica, [1.0, 3.0]):
            for p in rep_ps:
                p.grad = torch.full_like(p, val)
        group.sync_grads(per_replica, sign_exchange=True)
        for rep_ps in per_replica:
            for p in rep_ps:
                assert torch.allclose(p.grad, torch.full_like(p, 2.0))  # averaged, not signed
        group.teardown()

    def test_broadcast_module_attrs(self):
        block = self._block()
        block.m.weight_min = torch.full((8,), 2.0)
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=8)
        group = ReplicaGroup(block, plan)
        block.m.weight_min.mul_(5.0)
        group.broadcast_module_attrs(("weight_min",))
        for m in group.mirrors:
            assert torch.allclose(m.m.weight_min, torch.full((8,), 10.0))
        group.teardown()


class TestDDPEnvGate:
    def test_envs_default_world_one(self, monkeypatch):
        from auto_round import envs

        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        assert envs.AR_TUNE_DDP_WORLD == 1
        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "4")
        assert envs.AR_TUNE_DDP_WORLD == 4
        monkeypatch.setenv("AR_TUNE_DDP_DEVICES", "cuda:1,cuda:2")
        assert envs.AR_TUNE_DDP_DEVICES == "cuda:1,cuda:2"
        monkeypatch.setenv("AR_TUNE_DDP_BF16_GRAD", "1")
        assert envs.AR_TUNE_DDP_GRAD_TRANSPORT == "bf16"  # legacy alias resolves


class TestRelocateParams:
    def test_params_dict_entries_move_in_place(self):
        # the dict entry must stay the SAME object: wrapper _init_params
        # aliases params[name] with the registered _parameters entry, and the
        # optimizer collects from the dict while the forward reads the
        # registered Parameter -- replacing the object here silently broke
        # mirror tuning (grads landed on an object nobody stepped)
        from auto_round.algorithms.quantization.sign_round.data_parallel import _relocate_params

        wrapper = torch.nn.Module()
        wrapper.params = {"v": torch.nn.Parameter(torch.randn(4, 4))}
        old = wrapper.params["v"]
        old.grad = torch.ones(4, 4)
        _relocate_params(wrapper, torch.device("cpu"))
        new = wrapper.params["v"]
        assert isinstance(new, torch.nn.Parameter)
        assert new is old  # object identity preserved; data moved in place
        assert torch.equal(new.detach(), old.detach())

    def test_non_dict_and_non_params_untouched(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _relocate_params

        wrapper = torch.nn.Module()
        wrapper.params = {"note": "text", "buf": torch.zeros(3)}  # plain tensor stays as-is type-wise
        _relocate_params(wrapper, torch.device("cpu"))
        assert wrapper.params["note"] == "text"
        assert isinstance(wrapper.params["buf"], torch.Tensor) and not isinstance(
            wrapper.params["buf"], torch.nn.Parameter
        )


class TestReplicaThreadPool:
    def _block_group(self, world=4):
        block = torch.nn.Linear(2, 2)
        plan = DDPPlan(world=world, devices=[torch.device("cpu")] * world, shard_size=4)
        return ReplicaGroup(block, plan)

    def test_pool_executes_all_callables(self):
        group = self._block_group(4)
        ran = []
        group.run_threaded([lambda i=r: ran.append(i) for r in range(4)])
        assert sorted(ran) == [0, 1, 2, 3]
        # second round reuses the same workers
        group.run_threaded([lambda i=r: ran.append(i + 10) for r in range(4)])
        assert sorted(ran) == [0, 1, 2, 3, 10, 11, 12, 13]
        group.teardown()

    def test_pool_exception_first_by_index_and_survives(self):
        group = self._block_group(4)

        def boom(tag):
            def _f():
                raise ValueError(tag)

            return _f

        with pytest.raises(ValueError, match="second"):
            group.run_threaded([lambda: None, boom("second"), boom("third"), lambda: None])
        # the pool stays usable after a failing round
        ran = []
        group.run_threaded([lambda i=r: ran.append(i) for r in range(4)])
        assert sorted(ran) == [0, 1, 2, 3]
        group.teardown()

    def test_pool_env_opt_out_uses_spawn(self, monkeypatch):
        from auto_round import envs

        monkeypatch.setattr(envs, "AR_TUNE_DDP_THREAD_POOL", False)
        group = self._block_group(2)
        ran = []
        group.run_threaded([lambda: ran.append(0), lambda: ran.append(1)])
        assert sorted(ran) == [0, 1]
        assert getattr(group, "_pool", None) is None  # no pool created under opt-out
        group.teardown()

    def test_teardown_joins_workers(self):
        import threading as _threading

        before = _threading.active_count()
        group = self._block_group(3)
        group.run_threaded([lambda: None for _ in range(3)])
        assert _threading.active_count() >= before + 3  # workers alive mid-life
        group.teardown()
        assert _threading.active_count() == before  # all joined

    def test_spawn_function_semantics(self):
        ran = []

        def boom():
            raise RuntimeError("spawn boom")

        with pytest.raises(RuntimeError, match="spawn boom"):
            run_threaded_spawn([lambda: ran.append(0), boom, lambda: ran.append(2)])
        assert sorted(ran) == [0, 2]

    def test_pool_width_matches_world_not_first_call(self):
        group = self._block_group(4)
        # a narrower first call must not pin the pool to the wrong size:
        # it falls back to spawn, and the pool (when created) is world-wide
        ran = []
        group.run_threaded([lambda: ran.append(0)])  # width 1 < world 4
        assert ran == [0]
        group.run_threaded([lambda i=r: ran.append(i) for r in range(4)])
        assert sorted(ran) == [0, 0, 1, 2, 3]
        assert group._pool is not None and group._pool.n == 4
        group.teardown()

    def test_pool_class_run_and_shutdown(self):
        pool = ReplicaThreadPool(2)
        order = []
        pool.run([lambda: order.append("a"), lambda: order.append("b")])
        assert sorted(order) == ["a", "b"]
        with pytest.raises(ValueError, match="wrong count"):
            pool.run([lambda: None])  # mismatched length is refused
        pool.shutdown()


class TestDelayedBestTracker:
    """Ring + delayed-read selection must match immediate selection exactly.

    The tracker defers loss(i).item() to iteration i+1 (host pipelining);
    these tests pin the SELECTION contract: same argmin, same strict-less
    tie rule, same promoted snapshot values as an immediate reference.
    """

    @staticmethod
    def _snap(tag):
        # nested dict mirroring collect_best_params shape, value-tagged
        return {"layers.0.mlp": {"v": torch.full((4,), float(tag))}}

    def _run(self, tracker, losses, track=True):
        for i, v in enumerate(losses):
            tracker.stage(self._snap(i) if track else None, [torch.tensor(float(v), dtype=torch.float64)], i)
            tracker.resolve()

    def test_selection_matches_immediate_reference(self):
        sequences = [
            [3.0, 2.0, 1.0],  # decreasing
            [1.0, 2.0, 3.0],  # min at iter 0
            [2.0, 2.0, 2.0],  # all ties -> first wins (strict less)
            [5.0, 4.0, 5.0, 1.0, 2.0],  # late min
            [3.0, 2.0, 1.0, 0.5],  # min at last iter
            [7.0, 3.0, 3.0, 5.0, 9.0, 2.9],  # tie then near-tie
        ]
        for losses in sequences:
            ref_loss, ref_iter = min((v, i) for i, v in enumerate(losses))
            tr = _DelayedBestTracker()
            self._run(tr, losses)
            assert tr.best_loss == ref_loss, losses
            assert tr.best_iter == ref_iter, losses
            won = tr.best_params["layers.0.mlp"]["v"]
            assert torch.equal(won, torch.full((4,), float(ref_iter))), losses

    def test_pending_required_before_resolve(self):
        tr = _DelayedBestTracker()
        assert tr.resolve() is None  # nothing staged yet
        tr.stage(self._snap(0), [torch.tensor(1.0)], 0)
        assert tr.resolve() == 1.0
        assert tr.resolve() is None  # pending consumed

    def test_unresolved_loss_value_is_exact(self):
        # resolve returns the raw sum; caller applies its own scaling
        tr = _DelayedBestTracker()
        tr.stage(None, [torch.tensor(1.5), torch.tensor(2.5)], 3)
        assert tr.resolve() == 4.0
        assert tr.best_loss is None  # None snapshot never promotes
        assert tr.best_params is None
        assert tr.last_promoted is False

    def test_last_promoted_flag_tracks_promotions(self):
        tr = _DelayedBestTracker()
        tr.stage(self._snap(0), [torch.tensor(5.0, dtype=torch.float64)], 0)
        assert tr.resolve() == 5.0 and tr.last_promoted is True  # first stage promotes
        tr.stage(self._snap(1), [torch.tensor(4.0, dtype=torch.float64)], 1)
        assert tr.resolve() == 4.0 and tr.last_promoted is True
        tr.stage(self._snap(2), [torch.tensor(6.0, dtype=torch.float64)], 2)
        assert tr.resolve() == 6.0 and tr.last_promoted is False  # no improvement

    def test_best_survives_staging_without_reuse_copies(self):
        tr = _DelayedBestTracker()
        losses = [5.0, 1.0, 4.0, 3.0, 2.0]  # best at iter 1, never improved
        self._run(tr, losses)
        assert tr.best_iter == 1
        # snapshots are fresh allocations: the promoted best keeps the iter-1
        # values unchanged while later iterations stage their own copies
        assert torch.equal(tr.best_params["layers.0.mlp"]["v"], torch.full((4,), 1.0))
        # at most two snapshot dicts alive at any time (pending + best)
        tr.stage(self._snap(9), [torch.tensor(9.0, dtype=torch.float64)], 9)
        assert tr.best_iter == 1  # pending staged, not yet resolved

    def test_reset_clears_everything(self):
        tr = _DelayedBestTracker()
        self._run(tr, [3.0, 2.0])
        tr.reset()
        assert tr.best_loss is None and tr.best_params is None and tr.best_iter is None
        assert tr.resolve() is None  # pending dropped too
        # fresh selection after reset (alt2 re-grid semantics)
        self._run(tr, [9.0, 8.0])
        assert tr.best_loss == 8.0 and tr.best_iter == 1

    def test_delayed_env_gate_default_and_opt_out(self, monkeypatch):
        from auto_round import envs

        assert envs.AR_TUNE_DDP_DELAYED_LOSS is True
        monkeypatch.setattr(envs, "AR_TUNE_DDP_DELAYED_LOSS", False)
        assert envs.AR_TUNE_DDP_DELAYED_LOSS is False


class _FakeWrap(torch.nn.Module):
    """Minimal wrapper stand-in: params dict + module-attr dual aliasing."""

    def __init__(self, shape=(4,), keys=("value", "min_scale", "max_scale")):
        super().__init__()
        self.orig_layer = torch.nn.Linear(3, 3, bias=False)
        self.params = {}
        for k in keys:
            p = torch.nn.Parameter(torch.randn(shape))
            self.params[k] = p
            setattr(self, k, p)


class TestFlatVParams:
    """Flat v-param storage: views must alias the group parameters exactly."""

    @staticmethod
    def _block():
        block = torch.nn.Module()
        block.m1 = _FakeWrap((4,))
        block.m2 = _FakeWrap((6,))
        return block

    @staticmethod
    def _lrs(m):
        return 0.01

    @staticmethod
    def _mlrs(m):
        return 0.02

    def test_views_alias_flat_storage(self):
        block = self._block()
        assert build_flat_tuning_params(block, self._lrs, self._mlrs) is True
        for m in (block.m1, block.m2):
            for key in m.params:
                v = m.params[key]
                assert not isinstance(v, torch.nn.Parameter)  # a view, not a leaf
                assert key not in m._parameters  # stale leaf registration removed
                assert getattr(m, key) is v  # attr aliasing preserved
        # write through the view -> visible in the flat group parameter
        man = block._tune_flat_manifest
        e = [x for x in man if x[0] == "m1" and x[1] == "value"][0]
        block.m1.params["value"].data.zero_()
        gp = block._tune_flat_groups[e[2]]
        assert gp.data[e[3] : e[3] + e[4]].abs().sum().item() == 0.0

    def test_manifest_order_and_groups(self):
        block = self._block()
        assert build_flat_tuning_params(block, self._lrs, self._mlrs) is True
        man = block._tune_flat_manifest

        # flat order == legacy exchange order: round keys (named_modules x key
        # order) first, then minmax keys -- the r_ps + m_ps concatenation
        def _kind(k):
            return 1 if ("min" in k or "max" in k) else 0

        legacy = [(n, k) for n, m in block.named_modules() if hasattr(m, "orig_layer") for k in m.params]
        legacy = sorted(legacy, key=lambda nk: _kind(nk[1]))
        # buffer order == manifest sorted by (group, offset) == legacy r_ps + m_ps
        in_buffer_order = sorted(man, key=lambda x: (x[2], x[3]))
        assert [(e[0], e[1]) for e in in_buffer_order] == legacy
        # one group per (kind, lr); round groups indexed before minmax groups
        gk = block._tune_flat_group_lrs
        kinds = [k for k, _lr in gk]
        assert kinds == sorted(kinds, key=lambda k: 0 if k == "round" else 1)
        assert len(gk) == len(set(gk))  # one group per (kind, lr) pair
        # every group param covers its manifest slice exactly
        for gidx, gp in enumerate(block._tune_flat_groups):
            es = [e for e in man if e[2] == gidx]
            assert sum(e[4] for e in es) == gp.numel()

    def test_sign_sgd_steps_through_views(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        block = self._block()
        build_flat_tuning_params(block, self._lrs, self._mlrs)
        groups = list(block._tune_flat_groups)
        opt = SignSGD([{"params": [gp], "lr": 0.1} for gp in groups], lr=0.1)
        before = [gp.detach().clone() for gp in groups]
        loss = sum(
            (m.params[k] * 2.0).sum() for _n, m in block.named_modules() if hasattr(m, "params") for k in m.params
        )
        loss.backward()
        for gp in groups:
            assert gp.grad is not None  # autograd reached the group leaf via views
        opt.step()
        for gp, b in zip(groups, before):
            assert not torch.equal(gp.detach(), b)  # stepped in place

    def test_rebuild_views_on_mirror(self):
        block = self._block()
        build_flat_tuning_params(block, self._lrs, self._mlrs)
        from auto_round.algorithms.quantization.sign_round.data_parallel import _deepcopy_flat_safe

        with pytest.raises(RuntimeError):
            copy.deepcopy(block)  # plain deepcopy refuses non-leaf views
        mirror = _deepcopy_flat_safe(block)
        # simulate repair: fresh group parameter objects on the mirror
        fresh = torch.nn.ParameterList([torch.nn.Parameter(gp.detach().clone()) for gp in mirror._tune_flat_groups])
        fresh[0].data.zero_()  # distinguishable values
        mirror._tune_flat_groups = fresh
        assert _rebuild_flat_views(mirror) is True
        mods = dict(mirror.named_modules())
        e = [x for x in mirror._tune_flat_manifest if x[1] == "value"][0]
        mv = mods[e[0]].params[e[1]]
        assert mv.numel() == e[4]
        # view reads the MIRROR's fresh group, not home's storage
        assert mv.abs().sum().item() == 0.0

    def test_flat_grad_buffer_zero_copy_adjacent(self):
        base = torch.zeros(10)
        p1 = torch.nn.Parameter(base[0:6])
        p2 = torch.nn.Parameter(base[6:10])
        p1.grad = torch.ones(6)
        p2.grad = torch.ones(4)
        # grads must share one contiguous storage for the zero-copy path
        p1.grad = base[0:6].clone().requires_grad_(False)  # separate: returns None
        assert _flat_grad_buffer([p1, p2]) is None
        # now build genuinely adjacent grads
        g = torch.ones(10)
        p1.grad = g[0:6]
        p2.grad = g[6:10]
        buf = _flat_grad_buffer([p1, p2])
        assert buf is not None and buf.numel() == 10
        assert buf.data_ptr() == p1.grad.data_ptr()  # zero-copy over the chain
        buf[3] = 99.0
        assert p1.grad[3].item() == 99.0 and p2.grad[0].item() == 1.0  # aliases

    def test_flat_build_skips_non_fp32(self):
        block = self._block()
        orig = block.m1.params["value"]
        with torch.no_grad():
            block.m1.params["value"] = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
        setattr(block.m1, "value", block.m1.params["value"])
        assert build_flat_tuning_params(block, self._lrs, self._mlrs) is False
        assert isinstance(block.m1.params["value"], torch.nn.Parameter)  # untouched
        assert block.m1.params["value"] is not orig
        assert getattr(block, "_tune_flat_groups", None) is None

    def test_parked_grads_share_one_buffer(self):
        block = self._block()
        build_flat_tuning_params(block, self._lrs, self._mlrs)
        loss = sum(
            (m.params[k] * 2.0).sum() for _n, m in block.named_modules() if hasattr(m, "params") for k in m.params
        )
        loss.backward()
        for gp in block._tune_flat_groups:
            assert gp.grad is not None
        buf = _flat_grad_buffer(list(block._tune_flat_groups))
        assert buf is not None, "parked grads must be storage-adjacent after backward"
        assert buf.numel() == block._tune_flat_base.numel()
        # zero in place (set_to_none=False semantics) keeps the parking
        for gp in block._tune_flat_groups:
            gp.grad.zero_()
        assert _flat_grad_buffer(list(block._tune_flat_groups)) is not None

    def test_flat_vs_legacy_equivalence(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        def vals(block):
            return {
                f"{n}.{k}": m.params[k].detach().clone()
                for n, m in block.named_modules()
                if hasattr(m, "params")
                for k in m.params
            }

        def legacy_block(seed):
            torch.manual_seed(seed)
            b = torch.nn.Module()
            b.m1 = _FakeWrap((6,))
            b.m2 = _FakeWrap((8,))
            return b

        weights = [1.7, -0.9]
        legacy = [legacy_block(11 + r) for r in range(2)]
        flats = [legacy_block(11 + r) for r in range(2)]
        for b in flats:
            assert build_flat_tuning_params(b, self._lrs, self._mlrs)
        for b in legacy + flats:
            assert set(vals(b)) == set(vals(legacy[0]))
        lopts = [
            SignSGD(
                [
                    {
                        "params": [
                            m.params[k] for _n, m in b.named_modules() if hasattr(m, "params") for k in m.params
                        ],
                        "lr": 0.05,
                    }
                ],
                lr=0.05,
            )
            for b in legacy
        ]
        fopts = [SignSGD([{"params": list(b._tune_flat_groups), "lr": 0.05}], lr=0.05) for b in flats]
        for _it in range(10):
            for blocks, opts in ((legacy, lopts), (flats, fopts)):
                for b, w in zip(blocks, weights):
                    sum(
                        (m.params[k] * w).sum() for _n, m in b.named_modules() if hasattr(m, "params") for k in m.params
                    ).backward()
            # legacy: gather + exchange + scatter
            per_replica = [
                [m.params[k] for _n, m in b.named_modules() if hasattr(m, "params") for k in m.params] for b in legacy
            ]
            bufs = _param_grad_buffers(per_replica)
            halving_doubling_allreduce(bufs, scale=0.5)
            for buf, ps in zip(bufs, per_replica):
                _write_back_grads(buf, ps)
            # flat: zero-copy exchange over parked grads
            fbufs = []
            for b in flats:
                fb = _flat_grad_buffer(list(b._tune_flat_groups))
                assert fb is not None
                fbufs.append(fb)
            halving_doubling_allreduce(fbufs, scale=0.5)
            for o in lopts + fopts:
                o.step()
                o.zero_grad(set_to_none=False)
        for r in range(2):
            va, vb = vals(legacy[r]), vals(flats[r])
            for k in va:
                assert torch.equal(va[k], vb[k]), f"replica {r} key {k} diverged"

    def test_flat_env_gate_default_and_opt_out(self, monkeypatch):
        from auto_round import envs

        assert envs.AR_TUNE_DDP_FLAT_VPARAMS is True
        monkeypatch.setattr(envs, "AR_TUNE_DDP_FLAT_VPARAMS", False)
        assert envs.AR_TUNE_DDP_FLAT_VPARAMS is False


class TestRunThreaded:
    def test_exception_propagates_to_caller(self):
        block = torch.nn.Linear(2, 2)
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=4)
        group = ReplicaGroup(block, plan)

        def boom():
            raise RuntimeError("mirror failed")

        with pytest.raises(RuntimeError, match="mirror failed"):
            group.run_threaded([lambda: None, boom])
        group.teardown()

    def test_all_run_even_when_one_fails(self):
        ran = []
        block = torch.nn.Linear(2, 2)
        plan = DDPPlan(world=3, devices=[torch.device("cpu")] * 3, shard_size=4)
        group = ReplicaGroup(block, plan)

        def ok(i):
            ran.append(i)

        def boom():
            raise ValueError("x")

        with pytest.raises(ValueError):
            group.run_threaded([lambda i=0: ok(i), boom, lambda i=2: ok(i)])
        assert sorted(ran) == [0, 2]
        group.teardown()


class TestRelocatePlainState:
    def _wrapper(self):
        w = torch.nn.Module()
        w.orig_layer = torch.nn.Linear(8, 8)
        w.params = {"v": torch.nn.Parameter(torch.zeros(8, 8))}
        w.weight_min = torch.zeros(8)
        w.weight_max = torch.ones(8)
        w.device = torch.device("cuda:0")
        w.output_device = torch.device("cuda:0")
        return w

    def test_plain_tensors_and_device_attrs_relocated(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        target = torch.device("cuda:1")
        wrapper = self._wrapper()
        moved = []

        def fake_move(t, dev):
            moved.append((tuple(t.shape), dev))
            return t

        monkeypatch.setattr(dp, "_move_tensor", fake_move)
        dp._relocate_params(wrapper, target)
        assert wrapper.weight_min is wrapper.weight_min  # replaced via the seam
        assert (tuple(wrapper.weight_min.shape), target) in moved
        assert (tuple(wrapper.weight_max.shape), target) in moved
        assert wrapper.device == target and wrapper.output_device == target
        # params dict entries remain fresh Parameters (v moved via the seam too)
        assert isinstance(wrapper.params["v"], torch.nn.Parameter)
        assert any(shape == (8, 8) for shape, _dev in moved)

    def test_meta_tensors_left_alone(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        moved = []
        monkeypatch.setattr(dp, "_move_tensor", lambda t, dev: moved.append(1) or t)
        w = self._wrapper()
        w.skeleton = torch.zeros(4, device="meta")
        dp._relocate_params(w, torch.device("cuda:1"))
        assert w.skeleton.device.type == "meta"


class TestDDPGroups:
    def test_parse_group_syntax(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import parse_ddp_groups

        groups = parse_ddp_groups("0,1,2,3;4,5,6,7")
        assert groups == [
            [torch.device("cuda", i) for i in (0, 1, 2, 3)],
            [torch.device("cuda", i) for i in (4, 5, 6, 7)],
        ]
        assert parse_ddp_groups("") is None
        assert parse_ddp_groups(None) is None

    def test_plan_uses_group_containing_home(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import parse_ddp_groups, resolve_ddp_plan

        groups = parse_ddp_groups("0,1,2,3;4,5,6,7")
        plan = resolve_ddp_plan(4, torch.device("cuda:5"), batch_size=8, groups=groups)
        assert plan.devices[0] == torch.device("cuda", 5)
        assert plan.world == 4
        assert set(plan.devices) == {torch.device("cuda", i) for i in (4, 5, 6, 7)}

    def test_plan_home_outside_groups_disables(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import parse_ddp_groups, resolve_ddp_plan

        groups = parse_ddp_groups("0,1,2,3;4,5,6,7")
        plan = resolve_ddp_plan(4, torch.device("cuda:0"), batch_size=8, groups=parse_ddp_groups("4,5,6,7"))
        assert plan.world == 1 and any("group" in n for n in plan.notes)


class TestShardedNoGradForward:
    def _runner(self):
        class _R:
            def __init__(self):
                self.calls = []

            def __call__(self, block, inputs, others, indices=None, cache_device=None):
                return self.forward(block, inputs, others, indices, cache_device)

            def forward(self, block, inputs, others, indices, cache_device=None):
                import torch

                idx = list(indices) if indices is not None else list(range(len(inputs)))
                self.calls.append((str(block), idx, str(cache_device)))
                rows = [block(torch.tensor([float(i), float(i)])) for i in idx]
                return torch.stack(rows).reshape(len(idx), 1)

        return _R()

    def test_shards_split_outputs_concat_in_order(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import sharded_nograd_forward

        block = nn.Linear(2, 1)
        inputs = list(range(8))  # opaque to the helper; indices matter
        runner = self._runner()
        out = sharded_nograd_forward(
            runner,
            block,
            inputs,
            {},
            out_device=torch.device("cpu"),
            devices=[torch.device("cpu")] * 4,
            sample_count=8,
        )
        # serial contract: per-sample list, in order -- NOT one cat'd tensor
        assert isinstance(out, list) and len(out) == 8
        assert all(t.shape == (1, 1) for t in out)
        got = [i for _b, idx, _c in runner.calls for i in idx]
        assert got == list(range(8))
        assert len(runner.calls) == 4

    def test_hook_pass_shard_cap(self):
        from unittest.mock import patch

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        block = nn.Linear(2, 1)
        runner = self._runner()
        devices = [torch.device("cpu")] * 8
        with patch.object(dp.logger, "info"):
            out = dp.sharded_nograd_forward(
                runner,
                block,
                [torch.randn(1, 2, 2) for _ in range(8)],
                {},
                out_device=torch.device("cpu"),
                devices=devices,
                sample_count=8,
                max_devices=4,
            )
        # 8 samples over the first 4 devices -> 2-sample shards, in order
        assert len(runner.calls) == 4
        got = [i for _b, idx, _c in runner.calls for i in idx]
        assert got == list(range(8))
        assert [len(idx) for _b, idx, _c in runner.calls] == [2, 2, 2, 2]
        assert isinstance(out, list) and len(out) == 8
        # cap of 0 leaves all devices sharded
        runner2 = self._runner()
        dp.sharded_nograd_forward(
            runner2,
            block,
            [torch.randn(1, 2, 2) for _ in range(8)],
            {},
            out_device=torch.device("cpu"),
            devices=devices,
            sample_count=8,
            max_devices=0,
        )
        assert len(runner2.calls) == 8

    def test_hook_pass_shard_cap_logging(self):
        from unittest.mock import patch

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        block = nn.Linear(2, 1)
        runner = self._runner()
        with patch.object(dp.logger, "info") as info:
            dp.sharded_nograd_forward(
                runner,
                block,
                [torch.randn(1, 2, 2) for _ in range(8)],
                {},
                out_device=torch.device("cpu"),
                devices=[torch.device("cpu")] * 8,
                sample_count=8,
                max_devices=4,
            )
        msgs = [c.args[0] % c.args[1:] if len(c.args) > 1 else c.args[0] for c in info.call_args_list]
        assert any("capping concurrent shards at 4 of 8" in m for m in msgs)

    def test_sharded_collect_breakdown_logged(self):
        from unittest.mock import patch

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        block = nn.Linear(2, 1)
        runner = self._runner()
        with patch.object(dp.logger, "info") as info:
            dp.sharded_nograd_forward(
                runner,
                block,
                list(range(8)),
                {},
                out_device=torch.device("cpu"),
                devices=[torch.device("cpu")] * 4,
                sample_count=8,
            )
        lines = [c.args[0] for c in info.call_args_list if "sharded collect breakdown" in c.args[0]]
        assert lines, "expected the per-pass collection breakdown log"
        line = lines[0]
        # setup / fwd_wall / fwd_gpu / split / merge all present, and the
        # %-format args interpolate cleanly (calls' args carry the numbers)
        for key in ("setup=", "fwd_wall", "fwd_gpu", "split=", "merge="):
            assert key in line
        call = info.call_args_list[[c.args[0] for c in info.call_args_list].index(line)]
        msg = line % call.args[1:]
        assert "world=4" in msg and "nan" not in msg

    def test_input_pool_pre_distributed(self):
        from unittest.mock import patch

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        block = nn.Linear(2, 1)
        pool = [torch.randn(1, 2, 2) for _ in range(8)]  # tensor pool: placed
        runner = self._runner()
        with patch.object(dp, "distribute_pool", wraps=dp.distribute_pool) as dist:
            dp.sharded_nograd_forward(
                runner,
                block,
                pool,
                {},
                out_device=torch.device("cpu"),
                devices=[torch.device("cpu")] * 4,
                sample_count=8,
            )
            # shard-local reads: the input pool is placed on the collect
            # devices BEFORE the threaded forwards
            dist.assert_called_once_with(pool, [torch.device("cpu")] * 4)
        # explicit sample_count != len(inputs): skip pre-distribution
        with patch.object(dp, "distribute_pool", wraps=dp.distribute_pool) as dist2:
            dp.sharded_nograd_forward(
                runner,
                block,
                pool,
                {},
                out_device=torch.device("cpu"),
                devices=[torch.device("cpu")] * 4,
                sample_count=4,
            )
            dist2.assert_not_called()
        # opaque (non-tensor) pools keep the historic passthrough contract
        with patch.object(dp, "distribute_pool", wraps=dp.distribute_pool) as dist3:
            dp.sharded_nograd_forward(
                runner,
                block,
                list(range(8)),
                {},
                out_device=torch.device("cpu"),
                devices=[torch.device("cpu")] * 4,
                sample_count=8,
            )
            dist3.assert_not_called()

    def test_indivisible_falls_back_serial(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import sharded_nograd_forward

        block = nn.Linear(2, 1)
        runner = self._runner()
        out = sharded_nograd_forward(
            runner,
            block,
            list(range(7)),
            {},
            out_device=torch.device("cpu"),
            devices=[torch.device("cpu")] * 4,
            sample_count=7,
        )
        assert len(runner.calls) == 1  # serial fallback


class TestShardedForwardStructure:
    def test_sharded_returns_serial_per_sample_list(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import sharded_nograd_forward

        class _Runner:
            def __init__(self):
                self.last_output_dict = None
                self.batch_dim = 0

            def split_outputs(self, output):
                return list(torch.split(output, 1, dim=self.batch_dim))

            def __call__(self, block, inputs, input_others, indices=None, cache_device=None):
                idx = list(indices) if indices is not None else list(range(len(inputs)))
                # mimic BlockForwardRunner: indices -> batched tensor [n, S, H]
                outs = [block(torch.ones(1, 3, 2)) for _i in idx]
                return torch.cat(outs, dim=0)

        block = torch.nn.Linear(2, 2)
        runner = _Runner()
        inputs = [torch.zeros(1, 3, 2) for _ in range(4)]
        out = sharded_nograd_forward(runner, block, inputs, {}, torch.device("cpu"), [torch.device("cpu")] * 2)
        # serial contract: list of per-sample [1, S, H] tensors in order
        assert isinstance(out, list) and len(out) == 4
        assert all(t.shape == (1, 3, 2) for t in out)
        # the partial last-shard dict residue must be cleared
        assert runner.last_output_dict is None


class TestHookGate:
    def test_collect_forward_with_hooks_stays_serial(self):
        """Hook-carrying collection passes must not shard (mirror stats are lost)."""
        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer.__new__(AlgorithmComposer)

        class _BF:
            def __call__(self, block, inputs, input_others, cache_device=None):
                return ["serial"]

        composer.block_forward = _BF()
        composer._coll_devs = [torch.device("cpu"), torch.device("cpu")]

        out = composer._collect_forward(object(), [1, 2], {}, torch.device("cpu"), allow_shard=False)
        assert out == ["serial"]


class TestMirrorStatMerge:
    def _hooked_block(self):
        import torch.nn as nn

        block = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))

        def collect_imatrix(module, inputs, output):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            sq = torch.sum(x.pow(2), dim=0)
            if hasattr(module, "imatrix"):
                module.imatrix += sq
            else:
                module.imatrix = sq

        for m in block:
            m.register_forward_hook(collect_imatrix)
        return block

    def test_merged_imatrix_equals_serial(self):
        """home-shard + mirror-shard + merge == serial all-samples total.

        Exercises the merge directly (on CPU all devices collapse onto the
        home block, whose threaded hook accumulation would race -- production
        always assigns distinct CUDA devices per replica).
        """
        import copy as _copy

        from auto_round.algorithms.quantization.sign_round.data_parallel import _merge_mirror_stats

        torch.manual_seed(0)
        inputs = [torch.randn(1, 3, 4) for _ in range(4)]

        block0 = self._hooked_block()
        serial_block = _copy.deepcopy(block0)
        with torch.no_grad():
            for x in inputs:
                serial_block(x)
        serial = {n: getattr(m, "imatrix").clone() for n, m in serial_block.named_modules() if hasattr(m, "imatrix")}

        home = block0  # identical weights: module-1 inputs depend on module-0 weights
        mirror = _copy.deepcopy(home)  # hooks copied; mirror accumulates its own shard
        with torch.no_grad():
            for x in inputs[:2]:
                home(x)
            for x in inputs[2:]:
                mirror(x)
        _merge_mirror_stats(home, [mirror])
        for n, m in home.named_modules():
            if hasattr(m, "imatrix"):
                torch.testing.assert_close(m.imatrix, serial[n], rtol=1e-5, atol=1e-5)

    def test_act_max_merged_as_max(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import _merge_mirror_stats

        home = nn.Linear(2, 2)
        home.act_max = torch.tensor([1.0, 5.0])
        mirror = nn.Linear(2, 2)
        mirror.act_max = torch.tensor([3.0, 2.0])
        _merge_mirror_stats(home, [mirror])
        assert home.act_max.tolist() == [3.0, 5.0]


class TestDistributedPool:
    def test_shard_samplers_align_with_pool_layout(self):
        from auto_round.compressors.utils import shard_samplers

        # nsamples=128, world=8, 1 sample per replica per iteration
        sams = shard_samplers(128, 8, 1)
        assert sams is not None and len(sams) == 8
        shard = 128 // 8
        seen = set()
        for r, s in enumerate(sams):
            for _ in range(shard):  # one full epoch per replica
                batch = s.next_batch()
                assert len(batch) == 1
                j = batch[0]
                assert r * shard <= j < (r + 1) * shard  # shard-local draws only
                seen.add(j)
        assert seen == set(range(128))  # complete, disjoint coverage per epoch

    def test_shard_samplers_multi_draw_and_fallbacks(self):
        from auto_round.compressors.utils import IndexSampler, shard_samplers

        # batch 8 / world 4 -> 2 samples per replica from a 32-piece shard
        sams = shard_samplers(128, 4, 2)
        assert sams is not None
        for r, s in enumerate(sams):
            b = s.next_batch()
            assert len(b) == 2
            assert all(32 * r <= j < 32 * (r + 1) for j in b)
        # not applicable: world 1, indivisible pool, oversized draw
        assert shard_samplers(128, 1, 8) is None
        assert shard_samplers(130, 8, 1) is None  # 130 % 8 != 0
        assert shard_samplers(8, 4, 3) is None  # draw 3 > shard 2

    def test_index_sampler_explicit_indices(self):
        import random

        from auto_round.compressors.utils import IndexSampler

        random.seed(0)
        s = IndexSampler(4, 2, indices=[10, 11, 12, 13])
        drawn = [j for _ in range(2) for j in s.next_batch()]
        assert set(drawn) <= {10, 11, 12, 13}
        assert s.nsamples == 4

    def test_distribute_pool_noop_on_single_device(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import distribute_pool

        pool = [torch.randn(1, 2, 2) for _ in range(4)]
        before = [t.data_ptr() for t in pool]
        distribute_pool(pool, [torch.device("cpu")])  # world < 2 -> untouched
        assert [t.data_ptr() for t in pool] == before
        distribute_pool(pool, [torch.device("cpu")] * 4)  # already local
        assert [t.data_ptr() for t in pool] == before

    def test_distribute_pool_indivisible_untouched(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import distribute_pool

        pool = [torch.randn(1, 2, 2) for _ in range(5)]
        before = [t.data_ptr() for t in pool]
        distribute_pool(pool, [torch.device("cpu")] * 2)  # 5 % 2 != 0
        assert [t.data_ptr() for t in pool] == before

    def test_sharded_outputs_stay_on_replica_device(self):
        """The per-shard runner call must receive cache_device == replica device."""
        from auto_round.algorithms.quantization.sign_round.data_parallel import sharded_nograd_forward

        seen = []

        class _Runner:
            last_output_dict = None

            def split_outputs(self, output):
                return list(torch.split(output, 1, dim=0))

            def __call__(self, block, inputs, input_others, indices=None, cache_device=None):
                seen.append(cache_device)
                outs = [block(torch.ones(1, 2, 2)) for _i in indices]
                return torch.cat(outs, dim=0).to(cache_device)

        block = torch.nn.Linear(2, 2)
        sharded_nograd_forward(_Runner(), block, list(range(4)), {}, torch.device("cpu"), [torch.device("cpu")] * 2)
        # every shard was told to park its outputs on ITS OWN device (here cpu)
        assert all(d == torch.device("cpu") for d in seen) and len(seen) == 2


class TestCatDeviceSafe:
    def test_same_device_selection_is_copy_free(self):
        from auto_round.algorithms.block_runner import _cat_device_safe

        tensors = [torch.randn(1, 3, 2) for _ in range(4)]
        ptrs = [t.data_ptr() for t in tensors]
        out = _cat_device_safe(tensors, dim=0)
        assert out.shape == (4, 3, 2)
        assert [t.data_ptr() for t in tensors] == ptrs  # no silent copies

    def test_cat_matches_torch_cat(self):
        from auto_round.algorithms.block_runner import _cat_device_safe

        tensors = [torch.randn(1, 3, 2) for _ in range(3)]
        torch.testing.assert_close(_cat_device_safe(tensors, 0), torch.cat(tensors, dim=0))


class TestExpectPoolLocal:
    def test_no_warning_when_local(self, caplog):
        import logging

        from auto_round.algorithms.quantization.sign_round.data_parallel import expect_pool_local

        pieces = [torch.randn(2, 2) for _ in range(3)]
        with caplog.at_level(logging.WARNING, logger="auto_round"):
            expect_pool_local(pieces, torch.device("cpu"), "test-site-local")
        assert not [r for r in caplog.records if "test-site-local" in r.message]

    def test_warns_once_per_site_on_misplacement(self):
        from unittest.mock import patch

        from auto_round.algorithms.quantization.sign_round import data_parallel as dp
        from auto_round.algorithms.quantization.sign_round.data_parallel import expect_pool_local

        dp._pool_move_warned.discard("test-site-stray")
        pieces = [torch.randn(2, 2) for _ in range(3)]
        stray = torch.randn(2, 2, device="meta")  # CPU box: simulate a stray via meta
        with patch.object(dp.logger, "warning") as warn:
            expect_pool_local(pieces + [stray], torch.device("cpu"), "test-site-stray")
            expect_pool_local(pieces + [stray], torch.device("cpu"), "test-site-stray")  # latched
        assert warn.call_count == 1
        args = warn.call_args[0]
        assert args[1] == "test-site-stray" and args[2] == 1 and args[3] == 4  # 1 stray of 4
        assert "meta" in str(args[5])


class TestEffectiveGroups:
    def test_env_groups_win_over_registry(self):
        import auto_round.envs as envs_mod
        from auto_round.algorithms.quantization.sign_round import data_parallel as dp

        dp.set_effective_ddp_groups([[torch.device("cuda", 4), torch.device("cuda", 5)]])
        try:
            envs_mod.AR_TUNE_DDP_GROUPS = "0,1,2,3;4,5,6,7"
            got = dp.effective_ddp_groups()
            assert [str(d) for g in got for d in g] == [
                "cuda:0",
                "cuda:1",
                "cuda:2",
                "cuda:3",
                "cuda:4",
                "cuda:5",
                "cuda:6",
                "cuda:7",
            ]
        finally:
            envs_mod.AR_TUNE_DDP_GROUPS = ""
            dp.set_effective_ddp_groups(None)

    def test_registry_used_when_env_empty(self):
        from auto_round.algorithms.quantization.sign_round import data_parallel as dp

        dp.set_effective_ddp_groups([[torch.device("cuda", 0), torch.device("cuda", 1)]])
        try:
            got = dp.effective_ddp_groups()
            assert got is not None and len(got) == 1 and len(got[0]) == 2
        finally:
            dp.set_effective_ddp_groups(None)

    def test_none_when_unset(self):
        from auto_round.algorithms.quantization.sign_round import data_parallel as dp

        dp.set_effective_ddp_groups(None)
        assert dp.effective_ddp_groups() is None


class TestBf16TransportOrder:
    def test_xchg_moves_bf16_bytes_not_fp32(self):
        # regression: .to(device, dtype) casts on the SOURCE and ships fp32;
        # _xchg must cast-down -> move bf16 -> cast-up
        from auto_round.algorithms.quantization.sign_round.data_parallel import _xchg

        src = torch.randn(1000, dtype=torch.float32)
        got = _xchg(src, torch.device("cpu"), torch.float32, "bf16")
        assert got.dtype == torch.float32
        # transported through bf16: agreement at bf16 resolution, not exact
        torch.testing.assert_close(got, src, rtol=5e-2, atol=5e-2)
        assert not torch.equal(got, src)
        # fp32 path stays exact
        got32 = _xchg(src, torch.device("cpu"), torch.float32, "fp32")
        assert torch.equal(got32, src)


class TestGradTransportSwitch:
    def test_env_default_alias_and_override(self, monkeypatch):
        from auto_round import envs

        monkeypatch.delenv("AR_TUNE_DDP_GRAD_TRANSPORT", raising=False)
        monkeypatch.delenv("AR_TUNE_DDP_BF16_GRAD", raising=False)
        assert envs.AR_TUNE_DDP_GRAD_TRANSPORT == "fp32"
        monkeypatch.setenv("AR_TUNE_DDP_BF16_GRAD", "1")  # legacy alias
        assert envs.AR_TUNE_DDP_GRAD_TRANSPORT == "bf16"
        monkeypatch.setenv("AR_TUNE_DDP_GRAD_TRANSPORT", "int8")  # explicit wins
        assert envs.AR_TUNE_DDP_GRAD_TRANSPORT == "int8"

    def test_env_invalid_raises(self, monkeypatch):
        from auto_round import envs

        monkeypatch.setenv("AR_TUNE_DDP_GRAD_TRANSPORT", "fp16")
        with pytest.raises(ValueError, match="fp32\|bf16\|int8"):
            _ = envs.AR_TUNE_DDP_GRAD_TRANSPORT

    def test_xchg_int8_accuracy_and_signs(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _xchg

        g = torch.randn(10000, dtype=torch.float32)
        got = _xchg(g, torch.device("cpu"), torch.float32, "int8")
        step = g.abs().amax() / 127.0
        torch.testing.assert_close(got, g, rtol=0, atol=float(step))  # within one quantization step
        # signs preserved for anything above one step (SignSGD consumes sign(mean))
        significant = g.abs() > step
        assert (torch.sign(got[significant]) == torch.sign(g[significant])).float().mean() > 0.999
        # zero segment is safe
        z = torch.zeros(100)
        torch.testing.assert_close(_xchg(z, torch.device("cpu"), torch.float32, "int8"), z)

    def test_allreduce_int8_close_to_mean(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import halving_doubling_allreduce

        bufs = [torch.randn(4096) for _ in range(4)]
        ref = torch.stack(bufs).mean(0)
        for transport in ("fp32", "bf16", "int8"):
            work = [b.clone() for b in bufs]
            halving_doubling_allreduce(work, scale=0.25, transport=transport)
            atol = {"fp32": 1e-6, "bf16": 2e-2, "int8": 1e-1}[transport]
            torch.testing.assert_close(work[0], ref, rtol=0, atol=atol)


class TestOneShotAllreduce:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("n", [96, 100])
    def test_matches_reference_mean(self, world, n):
        from auto_round.algorithms.quantization.sign_round.data_parallel import one_shot_allreduce

        torch.manual_seed(3)
        bufs = [torch.randn(n) for _ in range(world)]
        expected = torch.stack(bufs).sum(0) / world
        one_shot_allreduce(bufs, scale=1.0 / world)  # fp32 default: exact
        for b in bufs:
            assert torch.allclose(b, expected, rtol=1e-5, atol=1e-6)

    def test_int8_and_bf16_close_and_identical_across_ranks(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import one_shot_allreduce

        torch.manual_seed(4)
        base = [torch.randn(4096) for _ in range(4)]
        expected = torch.stack(base).mean(0)
        for transport, atol in (("bf16", 2e-2), ("int8", 1e-1)):
            bufs = [b.clone() for b in base]
            one_shot_allreduce(bufs, scale=0.25, transport=transport)
            torch.testing.assert_close(bufs[0], expected, rtol=0, atol=atol)
            for b in bufs[1:]:
                assert torch.equal(b, bufs[0])  # every rank holds the same average

    def test_world_one_scales_only(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import one_shot_allreduce

        b = torch.randn(8)
        one_shot_allreduce([b], scale=0.5)
        assert b.shape == (8,)  # scaled in place, no crash

    def test_int8_amax_captured_before_scale_mutation(self):
        # scale is applied per-destination inside the loop; a source whose
        # amax were recomputed AFTER its own acc.mul_(scale) would dequantize
        # every later peer with a shrunken scale -- this pins the ordering.
        from auto_round.algorithms.quantization.sign_round.data_parallel import one_shot_allreduce

        torch.manual_seed(5)
        base = [torch.randn(2048) for _ in range(4)]
        for scale in (1.0 / 4, 0.37):  # non-uniform scale also exercises it
            bufs = [b.clone() for b in base]
            expected = torch.stack(bufs).sum(0) * scale
            one_shot_allreduce(bufs, scale=scale, transport="int8")
            step = torch.stack(base).abs().amax() / 127.0
            torch.testing.assert_close(bufs[0], expected, rtol=0, atol=float(step))


class TestAllreduceModeSwitch:
    def test_env_default_and_validation(self, monkeypatch):
        from auto_round import envs
        from auto_round.algorithms.quantization.sign_round.data_parallel import use_one_shot

        monkeypatch.delenv("AR_TUNE_DDP_ALLREDUCE", raising=False)
        assert envs.AR_TUNE_DDP_ALLREDUCE == "auto"
        # auto policy: halving-doubling everywhere (one-shot measured slower
        # at world=4 int8: 350 ms vs 195 ms); explicit values force the choice
        for world in (2, 4, 8):
            for transport in ("fp32", "bf16", "int8"):
                assert not use_one_shot(world, transport)
        monkeypatch.setenv("AR_TUNE_DDP_ALLREDUCE", "oneshot")  # forces regardless
        assert use_one_shot(8, "fp32")
        monkeypatch.setenv("AR_TUNE_DDP_ALLREDUCE", "halving")
        assert not use_one_shot(4, "int8")
        monkeypatch.setenv("AR_TUNE_DDP_ALLREDUCE", "ring")
        with pytest.raises(ValueError, match="auto\|oneshot\|halving"):
            _ = envs.AR_TUNE_DDP_ALLREDUCE

    @pytest.mark.parametrize("transport", ["fp32", "bf16", "int8"])
    @pytest.mark.parametrize("world", [2, 4, 8])
    def test_one_shot_parity_all_worlds(self, world, transport):
        from auto_round.algorithms.quantization.sign_round.data_parallel import one_shot_allreduce

        torch.manual_seed(6)
        base = [torch.randn(4096) for _ in range(world)]
        expected = torch.stack(base).mean(0)
        bufs = [b.clone() for b in base]
        one_shot_allreduce(bufs, scale=1.0 / world, transport=transport)
        atol = {"fp32": 1e-6, "bf16": 2e-2, "int8": 1e-1}[transport]
        torch.testing.assert_close(bufs[0], expected, rtol=0, atol=atol)
        for b in bufs[1:]:
            assert torch.equal(b, bufs[0])  # bitwise identical across all ranks


class TestOverlapExchange:
    def test_encode_transport(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _encode_transport

        g = torch.randn(10000)
        q, meta = _encode_transport(g, "int8")
        assert q.dtype == torch.int8 and meta is not None
        step = g.abs().amax() / 127.0
        torch.testing.assert_close(q.to(torch.float32) * (meta / 127.0), g, rtol=0, atol=float(step))
        b, m = _encode_transport(g, "bf16")
        assert b.dtype == torch.bfloat16 and m is None
        f, m2 = _encode_transport(g, "fp32")
        assert f is g and m2 is None

    def test_canonical_bucket_sum_matches_mean_and_is_reusable(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import (
            _canonical_bucket_sum,
            _encode_transport,
        )

        torch.manual_seed(7)
        base = [torch.randn(4096) for _ in range(4)]
        for transport, atol in (("fp32", 1e-6), ("bf16", 2e-2), ("int8", 1e-1)):
            enc = [_encode_transport(b, transport) for b in base]
            payloads = [e[0] for e in enc]
            metas = [e[1] for e in enc]
            # call twice: the second call must see identical inputs (no
            # in-place mutation of the payloads through aliasing)
            first = _canonical_bucket_sum(payloads, metas, transport, torch.float32).clone()
            second = _canonical_bucket_sum(payloads, metas, transport, torch.float32)
            assert torch.equal(first, second)
            expected = torch.stack(base).mean(0)
            torch.testing.assert_close(first / 4.0, expected, rtol=0, atol=atol)

    def test_cpu_group_stays_on_sequential_path(self):
        # the overlap machinery is CUDA-only; a CPU group must keep working
        # through the classic exchange and leave _overlap unset
        from auto_round.algorithms.quantization.sign_round.data_parallel import ReplicaGroup

        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(16))

            def forward(self, x):
                return x * self.w

        blk = _Tiny()
        blk.params = {"v": blk.w}  # type: ignore[attr-defined]
        group = ReplicaGroup(
            blk, DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=1), grad_transport="fp32"
        )
        per_rep = group.round_params()
        for ps in per_rep:
            for p in ps:
                p.grad = p.detach().clone()
        group.sync_grads(per_rep)
        for ps in per_rep:
            for p in ps:
                assert p.grad is not None
        assert group._overlap is None
        group.teardown()

    def test_env_opt_out_blocks_overlap(self, monkeypatch):
        from auto_round import envs

        # default off: the per-hook fan-out regresses the exchange (see the
        # AR_DISABLE_OVERLAP_EXCHANGE comment in envs.py)
        monkeypatch.delenv("AR_DISABLE_OVERLAP_EXCHANGE", raising=False)
        assert envs.AR_DISABLE_OVERLAP_EXCHANGE is True
        monkeypatch.setenv("AR_DISABLE_OVERLAP_EXCHANGE", "0")
        assert envs.AR_DISABLE_OVERLAP_EXCHANGE is False
        monkeypatch.setenv("AR_DISABLE_OVERLAP_EXCHANGE", "1")
        assert envs.AR_DISABLE_OVERLAP_EXCHANGE is True
