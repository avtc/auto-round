# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Batched same-shape wrap-search tests: grouping, bit-parity, deferral protocol."""

import threading
import unittest
from unittest import mock

import torch

import auto_round.algorithms.quantization.search_shard as search_shard
from auto_round.algorithms.quantization.search_shard import run_batched_wrap_search


def _mk_inputs(seed=0, shape=(6, 128), dtype="int", bits=4, thresh=1e-5, device="cpu"):
    from auto_round.data_type.utils import resolve_optimized_init_scale_fn

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g)
    im = torch.rand(*shape, generator=g) + 0.5
    fn = resolve_optimized_init_scale_fn(dtype, thresh)
    return w, im, dtype, bits, thresh, device, fn


class FakeDeferred:
    """Duck-typed deferred wrapper: stages inputs, finalizes with the result."""

    supports_batched_search = True

    def __init__(self, seed=0, shape=(6, 128), dtype="int", bits=4, device="cpu", thresh=1e-5, search_fn=None):
        w, im, dt, b, th, dev, fn = _mk_inputs(seed, shape, dtype, bits, thresh, device)
        if search_fn is not None:
            fn = search_fn
        self._deferred_search_inputs = (w, dt, b, im, th, fn)
        self.init_scale = None
        self.finalized_on_thread = None
        self.name = f"m{seed}"

    def _run_deferred_search_now(self):
        w, _dt, b, im, _th, fn = self._deferred_search_inputs
        self.init_scale = fn(w, b, im)
        self._deferred_search_inputs = None

    def finalize_batched_search(self, init_scale):
        self.init_scale = init_scale
        self._deferred_search_inputs = None
        self.finalized_on_thread = threading.current_thread().name


class TestBatchedBitParity(unittest.TestCase):
    def test_stacked_search_matches_individual_exactly(self):
        fakes = [FakeDeferred(seed=i) for i in range(5)]
        individual = [_individual(f) for f in fakes]
        stacked_w = torch.stack([f._deferred_search_inputs[0] for f in fakes])
        stacked_im = torch.stack([f._deferred_search_inputs[3] for f in fakes])
        batched = fakes[0]._deferred_search_inputs[5](stacked_w, 4, stacked_im)
        for i, ref in enumerate(individual):
            self.assertTrue(torch.equal(batched[i], ref), f"module {i} diverged")

    def test_parity_across_shapes_and_bits(self):
        for shape, bits in [((8, 128), 4), ((13, 128), 3), ((4, 64), 2)]:
            fakes = [FakeDeferred(seed=i + bits * 10, shape=shape, bits=bits) for i in range(4)]
            refs = [
                f._deferred_search_inputs[5](f._deferred_search_inputs[0], bits, f._deferred_search_inputs[3])
                for f in fakes
            ]
            batched = fakes[0]._deferred_search_inputs[5](
                torch.stack([f._deferred_search_inputs[0] for f in fakes]),
                bits,
                torch.stack([f._deferred_search_inputs[3] for f in fakes]),
            )
            for i, ref in enumerate(refs):
                self.assertTrue(torch.equal(batched[i], ref))


class TestRunBatchedWrapSearch(unittest.TestCase):
    def test_batches_same_shape_and_finalizes(self):
        fakes = [FakeDeferred(seed=i) for i in range(4)] + [FakeDeferred(seed=99, shape=(3, 128))]
        expected = [_individual(f) for f in fakes]
        handled = run_batched_wrap_search(fakes)
        self.assertTrue(handled)
        for f, ref in zip(fakes, expected):
            self.assertIsNone(f._deferred_search_inputs)
            self.assertIsNotNone(f.init_scale)
            self.assertTrue(torch.equal(f.init_scale, ref), f.name)

    def test_single_deferred_runs_individually(self):
        fakes = [FakeDeferred(seed=0)]
        handled = run_batched_wrap_search(fakes)
        self.assertTrue(handled)  # still handled (individual fallback), inputs consumed
        self.assertIsNotNone(fakes[0].init_scale)

    def test_shape_mismatch_groups_separately(self):
        a = [FakeDeferred(seed=i, shape=(6, 128)) for i in range(2)]
        b = [FakeDeferred(seed=i, shape=(5, 128)) for i in range(2)]
        run_batched_wrap_search(a + b)
        for f in a + b:
            self.assertIsNotNone(f.init_scale)

    def test_dtype_mismatch_groups_separately(self):
        a = FakeDeferred(seed=0)
        b = FakeDeferred(seed=1, dtype="mx_fp4")
        run_batched_wrap_search([a, b])  # different resolved fn -> separate groups, no raise
        self.assertIsNotNone(a.init_scale)
        self.assertIsNotNone(b.init_scale)

    def test_distinct_search_fns_never_share_a_batch(self):
        def fake_fn_a(w, bits, im):
            return w.abs().amax(dim=-1).squeeze(-1)

        def fake_fn_b(w, bits, im):
            return -w.abs().amax(dim=-1).squeeze(-1)

        a = FakeDeferred(seed=0, search_fn=fake_fn_a)
        b = FakeDeferred(seed=1, search_fn=fake_fn_b)
        exp_a = fake_fn_a(a._deferred_search_inputs[0], 4, a._deferred_search_inputs[3])
        exp_b = fake_fn_b(b._deferred_search_inputs[0], 4, b._deferred_search_inputs[3])
        run_batched_wrap_search([a, b])
        # each kept its own search semantics (no shared stack across fns)
        self.assertTrue(torch.equal(a.init_scale, exp_a))
        self.assertTrue(torch.equal(b.init_scale, exp_b))

    def test_empty_list(self):
        self.assertFalse(run_batched_wrap_search([]))

    def test_batch_chunking_respects_cap(self):
        fakes = [FakeDeferred(seed=i) for i in range(6)]
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(torch, "stack", side_effect=spy_stack):
            run_batched_wrap_search(fakes, max_batch=2)
        # 3 chunks x (weights + imatrices) = 6 stacks, each of size 2
        self.assertEqual(sorted(calls), [2] * 6)
        for f in fakes:
            self.assertIsNotNone(f.init_scale)

    def test_element_budget_caps_batch_size(self):
        # 2**28 elements / (w+im) per module: tiny modules force the element cap below the probe cap
        fakes = [FakeDeferred(seed=i, shape=(1024, 128)) for i in range(6)]  # ~0.26M elems/module
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        import auto_round.algorithms.quantization.search_shard as shard_mod

        with mock.patch.object(torch, "stack", side_effect=spy_stack), mock.patch.object(
            shard_mod, "_probe_usable_bytes", return_value=2**40
        ):
            run_batched_wrap_search(fakes)
        # 2**28 // (2 * 1024 * 128) = 1024 -> all 6 in one batch (probe cap 64 governs); then a huge probe
        # with a small module count still takes the element cap path when modules are many:
        self.assertTrue(all(c == 6 for c in calls))

    def test_env_gb_override_caps_batches(self):
        import auto_round.algorithms.quantization.search_shard as shard_mod

        # tiny modules (0.26M elems each): 1 GiB budget would allow ~1000 -> force 2 per batch via GB override
        fakes = [FakeDeferred(seed=i, shape=(1024, 128)) for i in range(4)]
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(shard_mod.envs, "AR_WRAP_SEARCH_BATCH_GB", 0.001), mock.patch.object(
            torch, "stack", side_effect=spy_stack
        ):
            run_batched_wrap_search(fakes)
        # 0.001 GiB = ~268K elements -> 268214 // (2*1024*128=262144) = 1 module/batch;
        # 4 batches x (weights + imatrices) = 8 stacks of size 1
        self.assertEqual(calls, [1] * 8)
        for f in fakes:
            self.assertIsNotNone(f.init_scale)

    def test_env_gb_override_invalid_falls_back(self):
        import auto_round.algorithms.quantization.search_shard as shard_mod

        fakes = [FakeDeferred(seed=i) for i in range(4)]
        with mock.patch.object(shard_mod.envs, "AR_WRAP_SEARCH_BATCH_GB", "not-a-number"):
            handled = run_batched_wrap_search(fakes)
        self.assertTrue(handled)  # default budget applies, no crash
        for f in fakes:
            self.assertIsNotNone(f.init_scale)

    def test_kill_switch_disables(self):
        fakes = [FakeDeferred(seed=i) for i in range(3)]
        with mock.patch.object(search_shard.envs, "AR_DISABLE_SEARCH_SHARD", True):
            handled = run_batched_wrap_search(fakes)
        self.assertFalse(handled)
        self.assertIsNone(fakes[0].init_scale)  # inputs untouched: caller runs them per module
        for f in fakes:
            f._run_deferred_search_now()  # the caller's documented fallback
        self.assertIsNotNone(fakes[0].init_scale)


def _individual(fake):
    w, _dt, b, im, _th, fn = fake._deferred_search_inputs
    return fn(w, b, im)


class TestV2DeferralProtocol(unittest.TestCase):
    def _bare_v2(self):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        w = object.__new__(SignRoundOptimizedWrapperLinear)
        w.init_scale = None
        w._deferred_search_inputs = None
        return w

    def test_run_now_and_finalize_assign(self):
        from auto_round.data_type.utils import resolve_optimized_init_scale_fn

        w = self._bare_v2()
        weight = torch.randn(6, 128)
        imatrix = torch.rand(6, 128) + 0.5
        fn = resolve_optimized_init_scale_fn("int", 1e-5)
        w._deferred_search_inputs = (weight, "int", 4, imatrix, 1e-5, fn)
        ref = fn(weight, 4, imatrix)
        w._run_deferred_search_now()
        self.assertTrue(torch.equal(w.init_scale, ref))
        self.assertIsNone(w._deferred_search_inputs)

        w2 = self._bare_v2()
        w2._deferred_search_inputs = (weight, "int", 4, imatrix, 1e-5, fn)
        w2.finalize_batched_search(ref)
        self.assertTrue(torch.equal(w2.init_scale, ref))
        self.assertIsNone(w2._deferred_search_inputs)


class TestRealV2Construction(unittest.TestCase):
    """The kwarg must survive the REAL base __init__ chain (regression: kwargs swallowed it)."""

    def _layer(self):
        import torch.nn as nn

        layer = nn.Linear(128, 64, bias=False)
        layer.data_type = "int"
        layer.bits = 4
        layer.sym = True
        layer.group_size = 128
        layer.iters = 10
        layer.act_bits = 16
        return layer

    def _make(self, defer_search):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        layer = self._layer()
        return SignRoundOptimizedWrapperLinear(
            layer,
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=False,
            enable_torch_compile=False,
            device="cpu",
            defer_search=defer_search,
        )

    def test_deferred_stages_through_real_init(self):
        w = self._make(defer_search=True)
        self.assertIsNotNone(w._deferred_search_inputs, "defer_search was swallowed by the base __init__ kwargs")
        self.assertIsNone(w.init_scale)
        w._run_deferred_search_now()
        self.assertIsNotNone(w.init_scale)

    def test_non_deferred_searches_inline_through_real_init(self):
        w = self._make(defer_search=False)
        self.assertIsNone(w._deferred_search_inputs)
        self.assertIsNotNone(w.init_scale)

    def test_kwarg_absent_by_default(self):
        w = self._make(defer_search=False)
        self.assertIsNone(w._deferred_search_inputs)  # old callers unchanged


class TestSearchWorkerPicking(unittest.TestCase):
    def _pick(self, working_set, free_map, home="cuda:0"):
        import auto_round.algorithms.quantization.search_shard as shard_mod

        with mock.patch.object(torch.cuda, "device_count", return_value=len(free_map)), mock.patch.object(
            shard_mod, "_probe_usable_bytes", side_effect=lambda k: free_map.get(k)
        ):
            return shard_mod.pick_search_worker_devices(working_set, home_device=home)

    def test_full_home_goes_to_idle_devices(self):
        free = {"cuda:0": 1 * 2**30, "cuda:1": 12 * 2**30, "cuda:2": 12 * 2**30}
        ws = 4 * 2**30
        self.assertEqual(self._pick(ws, free), ["cuda:1", "cuda:2"])  # home excluded, idles viable

    def test_all_full_falls_back_to_home(self):
        free = {"cuda:0": 0, "cuda:1": 1 * 2**30}
        self.assertEqual(self._pick(4 * 2**30, free), ["cuda:0"])

    def test_home_participates_when_it_fits(self):
        free = {"cuda:0": 10 * 2**30, "cuda:1": 10 * 2**30}
        self.assertEqual(self._pick(2 * 2**30, free), ["cuda:0", "cuda:1"])

    def test_kill_switch_pins_home(self):
        import auto_round.algorithms.quantization.search_shard as shard_mod

        with mock.patch.object(shard_mod.envs, "AR_DISABLE_SEARCH_OFFLOAD", True):
            self.assertEqual(shard_mod.pick_search_worker_devices(4 * 2**30, home_device="cuda:3"), ["cuda:3"])

    def test_no_cuda_returns_home(self):
        import auto_round.algorithms.quantization.search_shard as shard_mod

        with mock.patch.object(torch.cuda, "device_count", return_value=0):
            self.assertEqual(shard_mod.pick_search_worker_devices(4 * 2**30, home_device="cpu"), ["cpu"])

    def test_rtn_driver_offloads_to_viable_worker(self):
        # CPU-only: workers = [home] so behavior is the parity path already covered;
        # this pins that the bucket machinery runs end-to-end with the picker patched.
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = TestBatchedRtnSearchParity()._layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", TestBatchedRtnSearchParity()._make_wrapper(layer)))
        with mock.patch(
            "auto_round.algorithms.quantization.search_shard.pick_search_worker_devices",
            return_value=["cpu"],
        ):
            leftovers = run_batched_rtn_search(model, staged)
        self.assertEqual(leftovers, [])
        for i in range(2):
            self.assertIsNotNone(getattr(model, f"l{i}").scale)


class TestWrapperBlockDrivesBatching(unittest.TestCase):
    def _fake_block(self):
        block = torch.nn.Module()
        for i in range(3):
            m = torch.nn.Linear(8, 8, bias=False)
            m.bits = 4
            setattr(block, f"l{i}", m)
        return block

    def test_protocol_class_deferred_and_batched(self):
        import auto_round.wrapper as wrapper_mod

        calls = {"created": 0, "finalized": 0}

        class FakeBatched(torch.nn.Module):
            supports_batched_search = True

            def __init__(self, layer, defer_search=False, **kwargs):
                super().__init__()
                self.orig_layer = layer
                self.init_scale = None
                self._deferred_search_inputs = None
                calls["created"] += 1
                assert defer_search is True
                w = layer.weight.data.reshape(-1, layer.weight.shape[1])
                from auto_round.data_type.utils import resolve_optimized_init_scale_fn

                self._deferred_search_inputs = (
                    w,
                    "int",
                    4,
                    torch.ones_like(w),
                    1e-5,
                    resolve_optimized_init_scale_fn("int", 1e-5),
                )

            def _run_deferred_search_now(self):
                w, _dt, b, im, _th, fn = self._deferred_search_inputs
                self.init_scale = fn(w, b, im)
                self._deferred_search_inputs = None

            def finalize_batched_search(self, init_scale):
                self.init_scale = init_scale
                self._deferred_search_inputs = None
                calls["finalized"] += 1

        block = self._fake_block()
        q, u = wrapper_mod.wrapper_block(
            block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeBatched
        )
        self.assertEqual(q, ["l0", "l1", "l2"])
        self.assertEqual(calls["created"], 3)
        self.assertEqual(calls["finalized"], 3)  # all three searched via the batch driver
        for m in (block.l0, block.l1, block.l2):
            self.assertIsNotNone(m.init_scale)

    def test_kill_switch_runs_per_module(self):
        import auto_round.algorithms.quantization.search_shard as shard_mod
        import auto_round.wrapper as wrapper_mod

        stats = {"inline": 0, "now": 0}

        class FakeBatched(torch.nn.Module):
            supports_batched_search = True

            def __init__(self, layer, defer_search=False, **kwargs):
                super().__init__()
                self.orig_layer = layer
                self.init_scale = None
                self._deferred_search_inputs = None
                if defer_search:
                    self._deferred_search_inputs = ("staged",)
                else:  # kill switch: the class must search inline, as the real wrapper does
                    stats["inline"] += 1
                    self.init_scale = torch.zeros(1)

            def _run_deferred_search_now(self):
                stats["now"] += 1
                self.init_scale = torch.zeros(1)
                self._deferred_search_inputs = None

            def finalize_batched_search(self, init_scale):
                raise AssertionError("must not batch under the kill switch")

        block = self._fake_block()
        with mock.patch.object(shard_mod.envs, "AR_DISABLE_SEARCH_SHARD", True):
            wrapper_mod.wrapper_block(
                block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeBatched
            )
        self.assertEqual(stats["inline"], 3)  # no deferral at all under the kill switch
        self.assertEqual(stats["now"], 0)
        self.assertIsNotNone(block.l0.init_scale)


class TestBatchedRtnSearchParity(unittest.TestCase):
    """iters=0 batching: stacked vs serial must be bit-identical (incl. the imatrix path)."""

    def _layer(self, seed, sym=True, with_imatrix=True):
        import torch.nn as nn

        g = torch.Generator().manual_seed(seed)
        layer = nn.Linear(128, 64, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(64, 128, generator=g))
        layer.data_type = "int"
        layer.bits = 4
        layer.sym = sym
        layer.group_size = 128
        layer.iters = 0
        layer.act_bits = 16
        layer.scale_dtype = torch.float16
        if with_imatrix:
            layer.imatrix = torch.rand(128, generator=g) + 0.5
        return layer

    def _make_wrapper(self, layer):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(
            layer,
            device="cpu",
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=False,
            enable_round_tuning=False,
            enable_torch_compile=False,
            disable_opt_rtn=False,
            iters=0,
        )

    def _run(self, sym, with_imatrix):
        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        # serial arm (production callers run under no_grad)
        serial = []
        with torch.no_grad():
            for i in range(3):
                layer = self._layer(seed=i, sym=sym, with_imatrix=with_imatrix)
                w = self._make_wrapper(layer)
                out = w.unwrapper({})
                serial.append((out.weight.data.clone(), out.scale, out.zp))
        # batched arm
        import torch.nn as nn

        model = nn.Module()
        staged = []
        for i in range(3):
            layer = self._layer(seed=i, sym=sym, with_imatrix=with_imatrix)
            setattr(model, f"l{i}", layer)
            w = self._make_wrapper(layer)
            staged.append((f"l{i}", w))
        leftovers = run_batched_rtn_search(model, staged)
        self.assertEqual(leftovers, [])
        for i in range(3):
            got = getattr(model, f"l{i}")
            ref_w, ref_scale, ref_zp = serial[i]
            self.assertTrue(torch.equal(got.weight.data, ref_w), f"weight mismatch module {i}")
            if isinstance(ref_scale, torch.Tensor):
                self.assertTrue(torch.equal(got.scale, ref_scale), f"scale mismatch module {i}")
            if ref_zp is not None and isinstance(ref_zp, torch.Tensor):
                self.assertTrue(torch.equal(got.zp, ref_zp), f"zp mismatch module {i}")

    def test_parity_sym_with_imatrix(self):
        self._run(sym=True, with_imatrix=True)

    def test_parity_sym_no_imatrix(self):
        self._run(sym=True, with_imatrix=False)

    def test_parity_asym_with_imatrix(self):
        self._run(sym=False, with_imatrix=True)

    def test_oom_chunk_falls_back_per_module(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn import batched_search

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = self._layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", self._make_wrapper(layer)))
        real_fn = staged[0][1].weight_quant_func
        calls = {"n": 0, "raised": 0}

        def boom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:  # only the stacked call raises; per-module fallback delegates
                calls["raised"] += 1
                raise torch.OutOfMemoryError("simulated")
            return real_fn(*a, **k)

        # patch BOTH wrappers with the SAME callable so the group key still matches
        with mock.patch.object(staged[0][1], "weight_quant_func", boom), mock.patch.object(
            staged[1][1], "weight_quant_func", boom
        ):
            leftovers = batched_search.run_batched_rtn_search(model, staged)
        self.assertEqual(calls["raised"], 1)  # the stacked call raised exactly once
        self.assertEqual(calls["n"], 3)  # then the two per-module fallbacks delegated to the real fn
        for i in range(2):
            got = getattr(model, f"l{i}")
            self.assertIsNotNone(getattr(got, "scale", None))  # per-module fallback quantized it
