#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-tier tests for device-mapped placement and the mapped replica group.

CPU normalizes ``torch.device('cpu', i)`` tensors to plain ``cpu``, so
placement correctness is asserted through the resolver mapping (captured),
storage identity (``.data`` replaced), plan device-set bookkeeping, and
fail-loud checks -- not through tensor ``.device`` indices.
"""

import unittest

import pytest
import torch


class FakeGDNContainer(torch.nn.Module):
    """Gated-delta-net style container: parameters held in a plain list."""

    def __init__(self, dim=4):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)
        # the crash class: .to() never moves these
        self.delta_list = [torch.nn.Parameter(torch.randn(dim, dim)) for _ in range(2)]
        self.bias_tuple = (torch.nn.Parameter(torch.randn(dim)),)

    def forward(self, x):
        acc = self.proj(x)
        for d in self.delta_list:
            acc = acc + x @ d
        acc = acc + self.bias_tuple[0]
        return acc


class FakeMoEBlock(torch.nn.Module):
    """Stage-structured fake block: two stages with an expert loop."""

    def __init__(self, dim=4, n_experts=4):
        super().__init__()
        self.attn = torch.nn.Linear(dim, dim)
        experts = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False) for _ in range(n_experts)])
        self.experts = experts
        self.down = torch.nn.Linear(dim, dim)

    def forward(self, x, topk=None):
        h = self.attn(x)
        if topk is None:
            topk = torch.zeros(x.shape[0], dtype=torch.long)
        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            mask = topk == e
            if mask.any():
                out = out + expert(h) * mask.unsqueeze(-1).to(h.dtype)
        return self.down(out)


@pytest.fixture()
def _autoround_log_propagate():
    """Propagate the ``autoround`` logger to the root so caplog sees warnings.

    Self-contained copy of the test_ddp_core fixture (the production logger
    is configured with propagate=False).
    """
    import logging

    logger = logging.getLogger("autoround")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


class TestTensorLeaves:
    """The leaf walker sees unregistered parameters that .to() misses."""

    def test_finds_list_and_tuple_held_params(self):
        from auto_round.algorithms.quantization.sign_round.placement import tensor_leaves

        m = FakeGDNContainer()
        names = {n for n, _t, _c, _k in tensor_leaves(m)}
        assert "proj.weight" in names
        assert any("delta_list[0]" in n for n in names), names
        assert any("delta_list[1]" in n for n in names), names
        assert any("bias_tuple[0]" in n for n in names), names

    def test_plain_to_leaves_them_behind(self):
        # the crash class in structural form: .to() never registers or
        # reseats list-held parameters -- they stay invisible to the module
        # registry, so any later registry-driven move misses them too
        m = FakeGDNContainer()
        original = m.delta_list[0]
        m.to(torch.device("cpu"))
        assert m.delta_list[0] is original
        registered = {id(p) for _n, p in m.named_parameters()}
        assert id(m.delta_list[0]) not in registered
        assert id(m.bias_tuple[0]) not in registered

    def test_place_module_tree_walks_everything(self):
        from auto_round.algorithms.quantization.sign_round.placement import place_module_tree

        m = FakeGDNContainer()
        calls = []

        def device_of(mod_name):
            calls.append(mod_name)
            return torch.device("cpu")

        moved = place_module_tree(m, device_of)
        assert isinstance(moved, int)
        assert isinstance(m.delta_list[0], torch.nn.Parameter)  # identity kept
        assert "" in calls  # root module asked for its target
        # the block still runs after placement
        out = m(torch.randn(2, 4))
        assert out.shape == (2, 4)

    def test_place_module_tree_maps_stages(self):
        from auto_round.algorithms.quantization.sign_round.placement import place_module_tree

        m = FakeMoEBlock()
        # CUDA targets are unreachable on the CPU tier; the mapping contract
        # is which modules get asked, so tag stages by a sentinel in `seen`
        stage = {"": "s0", "attn": "s0", "experts": "s1", "down": "s1"}
        seen = {}

        def device_of(mod_name):
            seen[mod_name] = stage.get(mod_name, "s1")
            return torch.device("cpu")

        place_module_tree(m, device_of)
        # every module in the tree got a stage target (ModuleList submods too)
        assert seen[""] == "s0" and seen["attn"] == "s0"
        assert seen["down"] == "s1" and seen["experts"] == "s1"
        assert any(n.startswith("experts.") and v == "s1" for n, v in seen.items())


class TestStageDevices:
    def test_accelerator_stage_devices_ordered(self):
        from auto_round.algorithms.quantization.sign_round.placement import accelerator_stage_devices

        m = FakeMoEBlock()
        # CPU-only world: no accelerator leaves -> empty
        assert accelerator_stage_devices(m) == []


class TestMappedBuffers(unittest.TestCase):
    def test_param_grad_buffers_staging(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _param_grad_buffers

        a = torch.nn.Parameter(torch.ones(3))
        b = torch.nn.Parameter(torch.ones(4))
        a.grad = torch.full((3,), 1.0)
        b.grad = torch.full((4,), 2.0)
        buf = _param_grad_buffers([[a, b]], staging_device=None)
        self.assertEqual(buf[0].shape, (7,))

        # staging path: buffer materializes on the staging device (bookkeeping
        # only on CPU -- the call must not raise and must stay flat) with
        # VALUE parity to the unstaged flatten (same parts, same order, fp32)
        buf2 = _param_grad_buffers([[a, b]], staging_device=torch.device("cpu"))
        self.assertEqual(buf2[0].shape, (7,))
        self.assertTrue(torch.equal(buf2[0], torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])))
        self.assertEqual(buf2[0].dtype, torch.float32)
        # a mapped group flattens per replica (each with its own staging
        # device) -- two replicas produce their own buffers, no shared device
        c = torch.nn.Parameter(torch.ones(2))
        c.grad = torch.full((2,), 3.0)
        bufs = [_param_grad_buffers([params], staging_device=torch.device("cpu"))[0] for params in ([a], [c])]
        self.assertTrue(torch.equal(bufs[1], torch.tensor([3.0, 3.0])))


class TestMappedResolver:
    """resolve_mapped_ddp_plan: disjoint sets, pricing notes, loud declines."""

    def _stages(self, n):
        return [torch.device("cuda", i) for i in range(n)]

    def test_insufficient_devices_declines(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_mapped_ddp_plan

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)  # host-independent
        block = FakeMoEBlock()
        plan = resolve_mapped_ddp_plan(2, block, self._stages(4), free=None)
        # no candidates at all -> even a smaller world does not fit
        assert plan.world == 1
        assert any("no smaller world fits" in n for n in plan.notes)

    def test_disjoint_sets_when_devices_exist(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        # simulate 8 visible cuda devices for the candidate enumeration
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        block = FakeMoEBlock()
        plan = dp.resolve_mapped_ddp_plan(2, block, self._stages(4), free=None)
        assert plan.world == 2
        sets = plan.replica_devices
        assert [str(d) for d in sets[0]] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        assert [str(d) for d in sets[1]] == ["cuda:4", "cuda:5", "cuda:6", "cuda:7"]

    def test_vram_decline(self, monkeypatch):
        import torch as _t

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(_t.cuda, "device_count", lambda: 8)
        # CPU-hosted fakes carry no accelerator leaves, so feed the pricing
        # helper directly (1 MiB of weights per stage)
        monkeypatch.setattr(dp, "_block_stage_bytes", lambda b: [2**20] * 4)
        block = FakeMoEBlock()
        stages = self._stages(4)
        tiny = 1024
        free = {torch.device("cuda", i): tiny for i in range(8)}
        plan = dp.resolve_mapped_ddp_plan(2, block, stages, free=free)
        # weights exist -> 7x pricing exceeds the tiny free budget -> decline
        assert plan.world == 1
        assert any("exceeds free" in n for n in plan.notes)


class TestMappedReplicaGroup:
    """Mapped mirror build: stage-count and overlap guards fire loudly."""

    def test_stage_count_mismatch_raises(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.placement as placement

        monkeypatch.setattr(placement, "accelerator_stage_devices", lambda b: [torch.device("cuda", 0)])
        block = FakeMoEBlock()
        plan = dp.DDPPlan(2, [torch.device("cuda", 0)], 0)
        plan.replica_devices = [
            [torch.device("cuda", 0), torch.device("cuda", 1)],
            [torch.device("cuda", 2), torch.device("cuda", 3)],
        ]
        with pytest.raises(RuntimeError, match="stage count"):
            dp.ReplicaGroup(block, plan)

    def test_overlapping_sets_raise(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.placement as placement

        monkeypatch.setattr(placement, "accelerator_stage_devices", lambda b: [torch.device("cuda", 0)])
        block = FakeMoEBlock()
        plan = dp.DDPPlan(2, [torch.device("cuda", 0)], 0)
        plan.replica_devices = [
            [torch.device("cuda", 0)],
            [torch.device("cuda", 0)],
        ]
        with pytest.raises(RuntimeError, match="overlap"):
            dp.ReplicaGroup(block, plan)


class FakeExpertLoop(torch.nn.Module):
    """MoE expert-loop crash class: call-time device captures go stale.

    Real hardware: the routed-expert loop scatters/index_copy_s onto a
    device captured when the module was built; moving the experts without
    rebinding the capture crashes the first cross-device op. CPU tiers
    cannot observe tensor device indices (n22576: cpu normalizes cpu:i),
    so the forward emulates the CUDA device-mismatch RULE on the captured
    device objects themselves.
    """

    def __init__(self, dim=4, n_experts=4):
        super().__init__()
        self.experts = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False) for _ in range(n_experts)])
        # call-time binds the placement must rebind (device capture, string
        # staging target, plain tensor cache)
        self._out_device = torch.device("cpu")
        self.device = "cpu"
        self._cache = torch.zeros(dim)

    def forward(self, x, topk=None):
        weight_device = self.experts[0].weight.device
        if self._out_device != weight_device or torch.device(self.device) != weight_device:
            raise RuntimeError(
                f"expert scatter device {self._out_device}/{self.device} != expert weights {weight_device}"
            )
        if topk is None:
            topk = torch.zeros(x.shape[0], dtype=torch.long)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = topk == e
            if mask.any():
                out = out + expert(x) * mask.unsqueeze(-1).to(x.dtype)
        return out + self._cache


class TestCalltimeDeviceBind:
    """The expert-loop crash class: .to() cannot rebind device captures."""

    def test_to_leaves_captures_stale(self):
        m = FakeExpertLoop()
        m._out_device = torch.device("cuda", 0)  # captured at build time
        m.device = "cuda:0"
        m.to(torch.device("cpu"))  # moves the experts, not the captures
        assert m._out_device == torch.device("cuda", 0)
        assert m.device == "cuda:0"
        # the emulated CUDA mismatch rule fires at call time
        with pytest.raises(RuntimeError, match="expert scatter device"):
            m(torch.randn(2, 4))

    def test_relocate_module_state_rebinds(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _relocate_module_state

        m = FakeExpertLoop()
        m._out_device = torch.device("cuda", 0)  # device objects repoint cross-family
        m.device = "cpu:1"  # string tokens repoint same-family only
        m._cache = torch.zeros(4, device=None)
        _relocate_module_state(m, torch.device("cpu"))
        assert m._out_device == torch.device("cpu")
        assert m.device == "cpu"  # same-family string staging target repointed
        assert m._cache.device.type == "cpu"
        out = m(torch.randn(2, 4))  # strict checker passes after the rebind
        assert out.shape == (2, 4)

    def test_other_family_strings_untouched(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import _relocate_module_state

        m = FakeExpertLoop()
        m.device = "hpu:0"  # different accelerator family: NOT repointed
        _relocate_module_state(m, torch.device("cpu"))
        assert m.device == "hpu:0"


class TestSnapshotOnSpanningBlock:
    """collect_best_params over a multi-wrapper block: per-wrapper copies.

    The mapped lane snapshots the HOME block only (mirrors follow the home
    values each iteration), so the snapshot contract is per-wrapper
    detached copies to the cache device -- the CPU tier cannot observe
    stage devices, but the shape of the snapshot (every wrapper, detached,
    off the live leaves) is what the spanning lane consumes.
    """

    def test_collect_best_params_copies_each_wrapper(self):
        from auto_round.compressors.utils import collect_best_params

        class FakeWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.orig_layer = torch.nn.Linear(4, 4)
                self.params = {"scale": torch.nn.Parameter(torch.randn(4)), "v": torch.nn.Parameter(torch.randn(4))}

            def forward(self, x):
                return self.orig_layer(x) * self.params["scale"]

        block = torch.nn.ModuleList([FakeWrapper(), FakeWrapper()])
        snapshot = collect_best_params(block, cache_device="cpu")
        assert set(snapshot) == {"0", "1"}
        for mod_params in snapshot.values():
            assert set(mod_params) == {"scale", "v"}
            for tensor in mod_params.values():
                assert tensor.device.type == "cpu"
                assert not tensor.requires_grad  # detached copy, not the live leaf


class TestMappedProbeFailVisible:
    """House rule: no silent excepts -- the device probe must warn and decline."""

    def test_probe_failure_warns_and_declines(self, monkeypatch, caplog, _autoround_log_propagate):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        def _boom():
            raise OSError("cuda probe unreachable")

        monkeypatch.setattr(torch.cuda, "device_count", _boom)
        block = FakeMoEBlock()
        with caplog.at_level("WARNING"):
            plan = dp.resolve_mapped_ddp_plan(2, block, [torch.device("cuda", 0)], free=None)
        assert plan.world == 1
        assert any("device probe failed" in r.message for r in caplog.records)


class TestMappedPricing:
    """Per-device fit: stage weights x7 + activation allowance vs free VRAM."""

    def test_k2_world2_fits_with_allowance(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        monkeypatch.setattr(dp, "_block_stage_bytes", lambda b: [2**30, 2**30])  # 1 GiB/stage
        block = FakeMoEBlock()
        stages = [torch.device("cuda", 0), torch.device("cuda", 1)]
        # 1 GiB x7 + 2 GiB allowance = 9 GiB needed per device
        free = {torch.device("cuda", i): 9 * 2**30 for i in range(4)}
        plan = dp.resolve_mapped_ddp_plan(2, block, stages, free=free)
        assert plan.world == 2
        assert plan.replica_devices is not None and len(plan.replica_devices) == 2

    def test_k2_world2_declines_when_allowance_tips_it(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        monkeypatch.setattr(dp, "_block_stage_bytes", lambda b: [2**30, 2**30])
        block = FakeMoEBlock()
        stages = [torch.device("cuda", 0), torch.device("cuda", 1)]
        # weights alone would fit in 8 GiB; the 2 GiB allowance tips it over
        free = {torch.device("cuda", i): 8 * 2**30 for i in range(4)}
        plan = dp.resolve_mapped_ddp_plan(2, block, stages, free=free)
        assert plan.world == 1
        assert any("exceeds free" in n for n in plan.notes)


class TestHy3GeometryPricing:
    """hy3 block geometry (3.95B params, K=4) against the per-device guard.

    Per stage: 3.95e9/4 params x 2 B bf16 = 1.975e9 B (1.84 GiB) of weights;
    x7 (fp32 value + grads + best-MSE snapshot alongside the weights, the
    streaming balancer's tune-state multiplier) + the 2 GiB activation
    allowance = ~14.9 GiB per device at D=2.
    """

    STAGE_BYTES = int(3.95e9 / 4 * 2)  # 1_975_000_000

    def _resolve(self, free_gib, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        monkeypatch.setattr(dp, "_block_stage_bytes", lambda b: [self.STAGE_BYTES] * 4)
        free = {torch.device("cuda", i): int(free_gib * 2**30) for i in range(8)}
        return dp.resolve_mapped_ddp_plan(2, FakeMoEBlock(), [torch.device("cuda", i) for i in range(4)], free=free)

    def test_fits_16gib(self, monkeypatch):
        plan = self._resolve(16.0, monkeypatch)
        assert plan.world == 2
        assert any("replica 0" in n for n in plan.notes)

    def test_declines_12gib_with_reason(self, monkeypatch):
        plan = self._resolve(12.0, monkeypatch)
        assert plan.world == 1
        reason = [n for n in plan.notes if "exceeds free" in n]
        assert reason and "14.9GiB" in reason[0] and "12.0GiB" in reason[0]

    def test_decline_names_the_device(self, monkeypatch):
        plan = self._resolve(12.0, monkeypatch)
        assert any("cuda:0" in n for n in plan.notes if "exceeds free" in n)


class TestMetaPlacement:
    """Container-held plain tensors observably move (meta trick).

    torch.device('meta') makes real moves observable on the CPU tier, but
    its TensorImpl is incompatible with the ``.data`` set_data swap -- so
    the Parameter/buffer/bare-attr kinds (identity-preserving swaps) are
    GPU-handoff validations; what the CPU tier CAN observe end-to-end is
    the container-replacement branch (R1-4's dead code, now live) and the
    walker's coverage of every leaf kind.
    """

    class _CacheModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.plain_attr = torch.zeros(3)  # bare tensor attr
            self.cache_list = [torch.zeros(2), torch.ones(2)]  # list-held tensors
            self.cache_dict = {"a": torch.zeros(2)}  # dict-held tensor
            self.cache_tuple = (torch.zeros(2), torch.ones(2))  # tuple-held tensor

    def test_container_tensors_replaced_onto_meta(self):
        from auto_round.algorithms.quantization.sign_round.placement import (
            leaf_device_map,
            place_module_tree,
        )

        root = torch.nn.Module()
        root.cache = self._CacheModule()
        root.proj = torch.nn.Linear(4, 4)  # registered leaves stay cpu
        t_list = root.cache.cache_list[0]
        t_tuple = root.cache.cache_tuple[0]
        moved = place_module_tree(root, lambda name: torch.device("meta") if name == "cache" else torch.device("cpu"))
        assert moved >= 4
        assert root.cache.cache_list[0] is not t_list
        assert root.cache.cache_list[0].device.type == "meta"
        assert root.cache.cache_tuple[0] is not t_tuple
        assert root.cache.cache_tuple[0].device.type == "meta"
        assert root.cache.cache_dict["a"].device.type == "meta"
        assert root.cache.plain_attr.device.type == "meta"
        assert root.proj.weight.device.type == "cpu"  # untouched module

    def test_walker_names_every_leaf_kind(self):
        from auto_round.algorithms.quantization.sign_round.placement import leaf_device_map

        m = FakeGDNContainer()
        m.plain_attr = torch.zeros(3)
        m.cache_list = [torch.zeros(2)]
        m.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
        names = set(leaf_device_map(m))
        assert "proj.weight" in names  # registered param
        assert "inv_freq" in names  # registered buffer
        assert "delta_list[0]" in names  # container param
        assert "bias_tuple[0]" in names  # tuple-held param
        assert "plain_attr" in names  # bare tensor attr
        assert "cache_list[0]" in names  # container plain tensor
        # no doubled qualnames (R1-8)
        assert not any(n.count("cache_list") > 1 for n in names)


class TestMappedResolverGates:
    """Power-of-two + divisibility gates mirror the full-mirror lane."""

    def _stages(self, n):
        return [torch.device("cuda", i) for i in range(n)]

    def test_non_power_of_two_declines(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        plan = dp.resolve_mapped_ddp_plan(3, FakeMoEBlock(), self._stages(2), free=None)
        assert plan.world == 1
        assert any("not a power of two" in n for n in plan.notes)

    def test_batch_divisibility_declines(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        plan = dp.resolve_mapped_ddp_plan(2, FakeMoEBlock(), self._stages(2), free=None, batch_size=7)
        assert plan.world == 1
        assert any("not divisible" in n for n in plan.notes)

    def test_shard_size_set(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        plan = dp.resolve_mapped_ddp_plan(2, FakeMoEBlock(), self._stages(2), free=None, batch_size=8)
        assert plan.world == 2 and plan.shard_size == 4


class TestResolverIntercept:
    """resolve_tune_ddp_plan_ spanning path: engaged mapped / declined raises."""

    class _FakeQuantizer:
        def __init__(self):
            self._resolved_ddp_plan = None
            self.enable_lfq = False
            self.gradient_accumulate_steps = 1
            self.calibration_context = None

    def _patch_spanning(self, monkeypatch, n_stages):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.placement as placement

        stages = [torch.device("cuda", i) for i in range(n_stages)]
        monkeypatch.setattr(placement, "accelerator_stage_devices", lambda b: stages)
        return dp

    def test_spanning_engages_mapped(self, monkeypatch, _autoround_log_propagate):
        dp = self._patch_spanning(monkeypatch, 2)
        engaged = dp.DDPPlan(2, [torch.device("cuda", 0)], 4)
        engaged.replica_devices = [
            [torch.device("cuda", 0), torch.device("cuda", 1)],
            [torch.device("cuda", 2), torch.device("cuda", 3)],
        ]
        monkeypatch.setattr(dp, "resolve_mapped_ddp_plan", lambda *a, **k: engaged)
        q = self._FakeQuantizer()
        plan = dp.resolve_tune_ddp_plan_(
            q, FakeMoEBlock(), [torch.zeros(1)] * 8, None, torch.device("cuda", 0), world=2, log=True
        )
        assert plan.replica_devices is not None and plan.world == 2
        assert q._resolved_ddp_plan is plan  # cached like the full-mirror path

    def test_spanning_declined_raises_requirement(self, monkeypatch, _autoround_log_propagate):
        dp = self._patch_spanning(monkeypatch, 2)
        declined = dp.DDPPlan(1, [torch.device("cuda", 0)], 8, notes=["mapped replicas need 2 extra device(s)"])
        monkeypatch.setattr(dp, "resolve_mapped_ddp_plan", lambda *a, **k: declined)
        q = self._FakeQuantizer()
        with pytest.raises(RuntimeError, match="world=2 is ineligible.*need 2 extra"):
            dp.resolve_tune_ddp_plan_(
                q, FakeMoEBlock(), [torch.zeros(1)] * 8, None, torch.device("cuda", 0), world=2, log=True
            )


class TestWorldDemotion:
    """Q-2: device shortage demotes the world instead of hard-failing."""

    def test_auto_world_demotes_to_fit(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        # 8 visible, home spans 4 -> auto world 8 needs 28 extra devices;
        # only 4 free -> demote to 2 (needs exactly 4)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        monkeypatch.setattr(dp, "_block_stage_bytes", lambda b: [1024] * 4)
        free = {torch.device("cuda", i): 2**40 for i in range(8)}  # ample VRAM
        plan = dp.resolve_mapped_ddp_plan(
            8, FakeMoEBlock(), [torch.device("cuda", i) for i in range(4)], free=free, batch_size=8
        )
        assert plan.world == 2
        assert any("world reduced 8 -> 2" in n for n in plan.notes)
        assert plan.shard_size == 4

    def test_demoted_world_rechecks_divisibility(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        plan = dp.resolve_mapped_ddp_plan(
            8, FakeMoEBlock(), [torch.device("cuda", i) for i in range(4)], free=None, batch_size=7
        )
        # original world 8 divides nothing here (7 %% 8 != 0 declines first)
        assert plan.world == 1

    def test_mixed_families_decline(self):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        stages = [torch.device("cuda", 0), torch.device("xpu", 0)]
        plan = dp.resolve_mapped_ddp_plan(2, FakeMoEBlock(), stages, free=None)
        assert plan.world == 1
        assert any("mixed accelerator families" in n for n in plan.notes)


class TestComposerMappedDecline:
    """Q-1/Q-3: the composer's collection sharding declines for mapped plans.

    The check must fire on the plan RETURNED by the resolver -- the cache is
    empty on the first spanning block, so a cache-first read would miss
    exactly that engagement and build squashed ephemeral mirrors.
    """

    class _FakeComposer:
        def __init__(self, quantizer):
            self.block_quantizer = quantizer
            runner = type("R", (), {"output_config": {"hidden_states": 1}})
            self.block_forward = runner()

    def test_first_engagement_declines_collection(self, monkeypatch, _autoround_log_propagate):
        import auto_round.algorithms.composer as composer_mod
        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        engaged = resolve_tune_ddp_plan_.__globals__["DDPPlan"](2, [torch.device("cuda", 0)], 4)
        engaged.replica_devices = [
            [torch.device("cuda", 0), torch.device("cuda", 1)],
            [torch.device("cuda", 2), torch.device("cuda", 3)],
        ]

        class _Q:  # fresh quantizer: NO cached plan, like the first block
            _resolved_ddp_plan = None
            enable_lfq = False
            gradient_accumulate_steps = 1
            calibration_context = None

        import auto_round.algorithms.quantization.sign_round.placement as placement

        monkeypatch.setattr(
            placement, "accelerator_stage_devices", lambda b: [torch.device("cuda", 0), torch.device("cuda", 1)]
        )
        monkeypatch.setattr(
            "auto_round.algorithms.quantization.sign_round.data_parallel.resolve_tune_ddp_plan_",
            lambda *a, **k: engaged,
        )
        comp = self._FakeComposer(_Q())
        fp_inputs = [torch.zeros(1)] * 8
        got = composer_mod.AlgorithmComposer._ddp_collection_devices.__get__(comp)(type("B", (), {})(), fp_inputs)
        assert got is None  # declined: mapped plan -> serial collection
        assert _Q._resolved_ddp_plan is None  # the mock left the cache empty (first-block condition)


class TestFullMirrorPreferred:
    """A spanning block that FITS a single device takes the full-mirror lane.

    The gather squashes the block onto the home device and the wrappers
    self-stage (the proven path); mapped replicas are only for blocks that
    do not fit a single device.
    """

    def test_spanning_fits_uses_full_mirror(self, monkeypatch, _autoround_log_propagate):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.placement as placement

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        stages = [torch.device("cuda", 0), torch.device("cuda", 1)]
        full_mirror = dp.DDPPlan(2, [torch.device("cuda", 0), torch.device("cuda", 4)], 4)
        monkeypatch.setattr(dp, "resolve_ddp_plan", lambda *a, **k: full_mirror)
        calls = []
        monkeypatch.setattr(dp, "resolve_mapped_ddp_plan", lambda *a, **k: calls.append(1) or dp.DDPPlan(1, stages, 8))
        monkeypatch.setattr(placement, "accelerator_stage_devices", lambda b: stages)

        class _Q:
            _resolved_ddp_plan = None
            enable_lfq = False
            gradient_accumulate_steps = 1
            calibration_context = None

        # probe failure -> free=None -> full mirror preferred, mapped never consulted
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (_ for _ in ()).throw(OSError("no probe")))
        monkeypatch.setattr(dp, "_ENGAGED_LOGGED_SIG", None)  # engaged-path global
        plan = dp.resolve_tune_ddp_plan_(
            _Q(), FakeMoEBlock(), [torch.zeros(1)] * 8, None, torch.device("cuda", 0), world=2, log=True
        )
        assert plan.replica_devices is None and plan.world == 2
        assert calls == []

    def test_spanning_too_big_goes_mapped(self, monkeypatch, _autoround_log_propagate):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp
        import auto_round.algorithms.quantization.sign_round.placement as placement

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        stages = [torch.device("cuda", 0), torch.device("cuda", 1)]
        monkeypatch.setattr(placement, "accelerator_stage_devices", lambda b: stages)
        engaged = dp.DDPPlan(2, [torch.device("cuda", 0)], 4)
        engaged.replica_devices = [stages, [torch.device("cuda", 2), torch.device("cuda", 3)]]
        monkeypatch.setattr(dp, "resolve_mapped_ddp_plan", lambda *a, **k: engaged)

        class _Q:
            _resolved_ddp_plan = None
            enable_lfq = False
            gradient_accumulate_steps = 1
            calibration_context = None

        # every non-home device is starved: no full mirror fits -> mapped
        monkeypatch.setattr(dp, "_ENGAGED_LOGGED_SIG", None)
        free = {torch.device("cuda", i): 1024 for i in range(8)}
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (free[torch.device("cuda", idx)], 0))
        plan = dp.resolve_tune_ddp_plan_(
            _Q(), FakeMoEBlock(), [torch.zeros(1)] * 8, None, torch.device("cuda", 0), world=2, log=True
        )
        assert plan.replica_devices is not None


class TestStageBoundaryHooks:
    """accelerate AlignDevicesHook staging for spanning subtrees (bookkeeping).

    Real cross-device staging is GPU-handoff; the CPU tier pins the no-op
    path (CPU-only subtrees never hook) and mirror retargeting repoints
    execution devices.
    """

    def test_install_skips_cpu_only_subtrees(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        installed = []

        monkeypatch.setattr(
            "accelerate.hooks.add_hook_to_module",
            lambda mod, hook, append=False: installed.append(mod),
        )
        block = torch.nn.Linear(4, 4)  # all leaves on cpu: nothing hooks
        added = dp.install_stage_boundary_hooks_(block, torch.device("cuda", 0))
        assert added == 0 and installed == []

    def test_retarget_repoints_execution_devices(self):
        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        class _Hook:
            def __init__(self, dev):
                self.execution_device = dev

        m = torch.nn.Linear(4, 4)
        m._hf_hook = _Hook(torch.device("cuda", 0))
        moved = dp.retarget_stage_hooks_(m, lambda name: torch.device("cuda", 5))
        assert moved == 1 and m._hf_hook.execution_device == torch.device("cuda", 5)
        assert dp.retarget_stage_hooks_(m, lambda name: torch.device("cuda", 5)) == 0


class TestInitScaleDevice:
    """init_scale must land on the wrapper's device, not the search input's.

    The deferred V2 init-scale search runs on the ORIG WEIGHT's device
    (which can differ from the wrapper's device on a spanning block); the finalize receives vals from the round-robin search
    on ANOTHER replica's device. Both land on self.device -- the wrapper's
    canonical device where min/max_scale already live. Pinned with meta
    stand-ins for the foreign device.
    """

    def _wrapper_stub(self):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        w = SignRoundOptimizedWrapperLinear.__new__(SignRoundOptimizedWrapperLinear)
        layer = type("L", (), {})()
        layer.data_type = "int"
        layer.bits = 4
        layer.group_size = 128
        w.orig_layer = layer
        # meta as the wrapper's device: plain tensors CAN move onto meta
        # (copying OUT of meta is impossible), so the .to(self.device)
        # co-location is observable in the result's device
        w.device = torch.device("meta")
        w.q_scale_thresh = 0.0
        w._compile_own_quant_func = lambda: None
        return w

    def test_search_lands_on_wrapper_device(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        w = self._wrapper_stub()
        # the search input (grouped weight) and result live on the FOREIGN device
        monkeypatch.setattr(
            v2q.SignRoundOptimizedWrapperLinear, "_prepare_init_scale_weight", lambda self: torch.ones(8, 128)
        )
        # the searched scale arrives on the FOREIGN (cpu) device
        monkeypatch.setattr(v2q, "search_optimized_init_scale", lambda *a, **k: torch.ones(8, 1))
        monkeypatch.setattr(v2q, "reshape_imatrix_for_weight", lambda im, wr, gs: None)
        w._run_init_scale_search()
        assert w.init_scale is not None and w.init_scale.device.type == "meta"

    def test_finalize_cross_replica_val(self, monkeypatch):
        w = self._wrapper_stub()
        w.init_scale = None
        # val searched on another replica's (cpu) device
        w._finalize_deferred_init(val=torch.ones(8, 1))
        assert w.init_scale.device.type == "meta"

    def test_finalize_keeps_own_search_result(self):
        w = self._wrapper_stub()
        w.init_scale = torch.ones(8, 1, device="meta")  # already searched locally
        w._finalize_deferred_init(val=None)
        assert w.init_scale.device.type == "meta"
