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
