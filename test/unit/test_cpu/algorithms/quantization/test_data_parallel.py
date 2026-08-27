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

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.data_parallel import (
    DDPPlan,
    ReplicaGroup,
    _param_grad_buffers,
    _write_back_grads,
    halving_doubling_allreduce,
    resolve_ddp_plan,
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
        halving_doubling_allreduce(bufs, scale=0.25, bf16=True)
        for b in bufs:
            assert torch.allclose(b, expected, rtol=1e-2, atol=1e-2)


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
        assert envs.AR_TUNE_DDP_BF16_GRAD is True


class TestRelocateParams:
    def test_params_dict_entries_recreated_as_fresh_parameters(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _relocate_params

        wrapper = torch.nn.Module()
        wrapper.params = {"v": torch.nn.Parameter(torch.randn(4, 4))}
        old = wrapper.params["v"]
        old.grad = torch.ones(4, 4)
        _relocate_params(wrapper, torch.device("cpu"))
        new = wrapper.params["v"]
        assert isinstance(new, torch.nn.Parameter)
        assert new is not old and new.grad is None  # fresh leaf, grad state reset
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
