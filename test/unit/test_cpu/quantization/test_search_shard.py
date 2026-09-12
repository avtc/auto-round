# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed not permitted...
"""Tests for per-weight-device sharding of weight-local quantization searches."""

import threading
import unittest
from unittest import mock

import torch

import auto_round.algorithms.quantization.search_shard as search_shard
from auto_round.algorithms.quantization.search_shard import group_items_by_device, run_items_by_device


def _linear(out=8, inn=8):
    m = torch.nn.Linear(inn, out, bias=False)
    m.bits = 4  # pass check_to_quantized
    return m


class TestGroupItemsByDevice(unittest.TestCase):
    def test_groups_by_device_string(self):
        items = [("a", "cuda:0"), ("b", "cuda:1"), ("c", "cuda:0")]
        groups = group_items_by_device(items, device_of=lambda item: item[1])
        self.assertEqual([k for k in groups], ["cuda:0", "cuda:1"])
        self.assertEqual([(i, it) for i, it in groups["cuda:0"]], [(0, ("a", "cuda:0")), (2, ("c", "cuda:0"))])
        self.assertEqual([(i, it) for i, it in groups["cuda:1"]], [(1, ("b", "cuda:1"))])

    def test_empty_items(self):
        self.assertEqual(list(group_items_by_device([], device_of=lambda it: "cpu").items()), [])

    def test_missing_device_falls_back_to_none_bucket(self):
        items = [("a", None), ("b", "cuda:0")]
        groups = group_items_by_device(items, device_of=lambda item: item[1], none_key="shared")
        self.assertIn("shared", groups)
        self.assertIn("cuda:0", groups)


class TestRunItemsByDevice(unittest.TestCase):
    def test_runs_all_items_and_records_devices(self):
        items = [f"m{i}" for i in range(6)]
        # 3 fake devices, 2 items each
        device_of = lambda it: f"cuda:{int(it[1:]) % 3}"  # noqa: E731
        groups = group_items_by_device(items, device_of=device_of)
        seen = []
        lock = threading.Lock()

        def fn(_idx, item):
            with lock:
                seen.append((item, threading.current_thread().name))

        run_items_by_device(groups, fn)
        self.assertEqual(sorted(s[0] for s in seen), items)
        # items on different devices ran on different threads
        threads = {s[0]: s[1] for s in seen}
        self.assertNotEqual(threads["m0"], threads["m1"])

    def test_exception_propagates_fail_visible(self):
        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cpu")

        def fn(_idx, item):
            if item == "a":
                raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_items_by_device(groups, fn)

    def test_single_group_runs_inline(self):
        # one device -> no thread spawn; fn runs on the calling thread
        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cpu")
        caller = threading.current_thread().name
        seen = []
        run_items_by_device(groups, lambda _idx, it: seen.append(threading.current_thread().name))
        self.assertEqual(seen, [caller, caller])

    def test_cuda_device_ctx_applied(self):
        recorded = []
        real_ctx = torch.cuda.device

        class FakeCtx:
            def __init__(self, dev):
                recorded.append(str(dev))

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cuda:0" if it == "a" else "cuda:1")
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "device", FakeCtx
        ):
            run_items_by_device(groups, lambda _idx, it: None, use_cuda_ctx=True)
        self.assertEqual(sorted(recorded), ["cuda:0", "cuda:1"])  # one ctx per device group


class TestShardEligible(unittest.TestCase):
    def test_eligible_requires_multiple_cuda_devices(self):
        self.assertFalse(search_shard.shard_eligible(["cpu"]))
        self.assertFalse(search_shard.shard_eligible([]))
        self.assertFalse(search_shard.shard_eligible(["cpu", "cpu"]))
        self.assertTrue(search_shard.shard_eligible(["cpu", "cuda:1"]))
        self.assertTrue(search_shard.shard_eligible(["cuda:0", "cuda:1"]))


class TestWrapperBlockShard(unittest.TestCase):
    """wrapper_block threading via a fake wrapper_cls + patched device attribution."""

    def _fake_block(self):
        block = torch.nn.Module()
        block.l0 = _linear()
        block.l1 = _linear()
        return block

    def test_multi_device_threads_wrap(self):
        import auto_round.wrapper as wrapper_mod

        calls = []
        lock = threading.Lock()

        class FakeWrapper(torch.nn.Module):
            def __init__(self, layer, **kwargs):
                super().__init__()
                self.orig_layer = layer
                with lock:
                    calls.append((id(layer), threading.current_thread().name))

        block = self._fake_block()
        layers = [block.l0, block.l1]
        with mock.patch.object(
            wrapper_mod, "_wrap_item_device", side_effect=lambda m, device: f"cuda:{layers.index(m)}"
        ), mock.patch.object(search_shard, "shard_eligible", return_value=True):
            q, u = wrapper_mod.wrapper_block(
                block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeWrapper
            )
        self.assertEqual(q, ["l0", "l1"])  # original order preserved
        self.assertEqual(u, [])
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0][1], calls[1][1])  # ran on different threads
        self.assertTrue(isinstance(block.l0, FakeWrapper))

    def test_single_device_serial_unchanged(self):
        import auto_round.wrapper as wrapper_mod

        engaged = []

        class FakeWrapper(torch.nn.Module):
            def __init__(self, layer, **kwargs):
                super().__init__()
                self.orig_layer = layer

        import auto_round.algorithms.quantization.search_shard as shard_mod

        block = self._fake_block()
        with mock.patch.object(shard_mod, "run_items_by_device", side_effect=lambda *a, **k: engaged.append(1)):
            wrapper_mod.wrapper_block(
                block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeWrapper
            )
        self.assertEqual(engaged, [])  # never engaged on single-device blocks
        self.assertTrue(isinstance(block.l0, FakeWrapper))

    def test_thread_exception_reraised(self):
        import auto_round.wrapper as wrapper_mod

        class BadWrapper(torch.nn.Module):
            def __init__(self, layer, **kwargs):
                super().__init__()
                raise ValueError("search failed")

        block = self._fake_block()
        layers = [block.l0, block.l1]
        with mock.patch.object(
            wrapper_mod, "_wrap_item_device", side_effect=lambda m, device: f"cuda:{layers.index(m)}"
        ), mock.patch.object(search_shard, "shard_eligible", return_value=True):
            with self.assertRaisesRegex(ValueError, "search failed"):
                wrapper_mod.wrapper_block(
                    block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=BadWrapper
                )


class TestRtnSearchShard(unittest.TestCase):
    def test_optimized_rtn_quantize_block_shards(self):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = object.__new__(OptimizedRTNQuantizer)
        calls = []
        lock = threading.Lock()

        def fake_outside_block(m):
            with lock:
                calls.append((m.name, threading.current_thread().name))

        q.quantize_layer_outside_block = fake_outside_block

        block = torch.nn.Module()
        for i, dev in enumerate(["cuda:0", "cuda:1", "cuda:0"]):
            m = _linear()
            m.name = f"m{i}"
            m.tuning_device = dev
            setattr(block, f"m{i}", m)
            m.global_name = f"m{i}"

        with mock.patch.object(search_shard, "shard_eligible", return_value=True):
            q.quantize_block(block, None, None, None, None, None)
        order = [c[0] for c in calls]
        self.assertEqual(sorted(order), ["m0", "m1", "m2"])
        # same-device items keep their relative order; cross-device interleaving is scheduler-dependent
        self.assertLess(order.index("m0"), order.index("m2"))
        names = {c[0]: c[1] for c in calls}
        self.assertNotEqual(names["m0"], names["m1"])

    def test_single_device_runs_serial(self):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = object.__new__(OptimizedRTNQuantizer)
        caller = threading.current_thread().name
        calls = []
        q.quantize_layer_outside_block = lambda m: calls.append(threading.current_thread().name)

        block = torch.nn.Module()
        m = _linear()
        m.name = "m0"
        m.global_name = "m0"
        m.tuning_device = "cpu"
        block.m0 = m
        q.quantize_block(block, None, None, None, None, None)
        self.assertEqual(calls, [caller])
