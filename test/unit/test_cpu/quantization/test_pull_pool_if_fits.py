# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Unit tests for the iters>0 hot-pool bulk pull gate (``_pull_pool_if_fits``).

The helper decides whether a tune-loop pool moves in bulk onto the device
that reads it every iteration (entry device for inputs, loss device for
references). The gate arithmetic is exercised with tensor/block fakes and a
monkeypatched free-memory probe: no real multi-GPU devices are needed.
"""

import unittest
from unittest import mock

import torch

from auto_round.algorithms.quantization.sign_round.quantizer import _pull_pool_if_fits

_GIB = 2**30


class _FakeTensor:  # pylint: disable=too-few-public-methods
    def __init__(self, device, numel, esize=4):
        self.device = torch.device(device)
        self._numel = numel
        self._esize = esize
        self.moved_to = None

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize

    def to(self, device):
        self.moved_to = torch.device(device)
        self.device = torch.device(device)
        return self


class _FakeParam(_FakeTensor):  # pylint: disable=too-few-public-methods
    pass


class _FakeBlock:  # pylint: disable=too-few-public-methods
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return list(self._params)

    def modules(self):
        return [self]


def _run(pool, block, free, target="cuda:1", iters=20, label="tune-reference"):
    with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=free), mock.patch(
        "auto_round.utils.pool_placement._working_allowance_bytes", return_value=int(1.4 * _GIB)
    ):
        return _pull_pool_if_fits(pool, target, block, 8, iters, label)


class TestPullPoolIfFits(unittest.TestCase):
    def test_moves_when_free_covers_pool_and_state(self):
        # 8-GPU-shaped arithmetic: 4 GiB pool, ~6.4 GiB state on the target.
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4) for _ in range(4)]
        block = _FakeBlock([_FakeParam("cuda:1", numel=int(0.46e9))])
        out = _run(pool, block, free=19.5 * _GIB)
        self.assertEqual(len(out), len(pool))
        self.assertTrue(all(t.moved_to == torch.device("cuda:1") for t in out))

    def test_declines_when_state_density_eats_headroom(self):
        # 4-GPU-shaped arithmetic: same pool, ~12.9 GiB state on the target,
        # ~16 GiB free -> reserve alone exceeds free minus pool.
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4) for _ in range(4)]
        block = _FakeBlock([_FakeParam("cuda:1", numel=int(0.92e9))])
        out = _run(pool, block, free=16.1 * _GIB)
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))

    def test_state_charged_only_for_params_on_target(self):
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4)]
        block = _FakeBlock([_FakeParam("cuda:2", numel=int(4 * _GIB))])
        out = _run(pool, block, free=7 * _GIB)
        # peers' state is not charged: 7 - (1.4 + 0.5) >= 4 GiB pool bytes
        self.assertTrue(all(t.moved_to == torch.device("cuda:1") for t in out))

    def test_already_local_pool_is_noop(self):
        pool = [_FakeTensor("cuda:1", numel=_GIB // 4)]
        block = _FakeBlock([])
        out = _run(pool, block, free=0)  # probe value irrelevant
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))

    def test_non_list_pool_untouched(self):
        block = _FakeBlock([])
        out = _run({"a": 1}, block, free=100 * _GIB)
        self.assertEqual(out, {"a": 1})

    def test_probe_failure_keeps_sharded(self):
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4)]
        block = _FakeBlock([])
        # free=None -> decline path (no move); helper never raises
        with mock.patch("auto_round.utils.device.probe_usable_bytes", side_effect=RuntimeError("no cuda")), mock.patch(
            "auto_round.utils.pool_placement._working_allowance_bytes", return_value=0
        ):
            out = _pull_pool_if_fits(pool, "cuda:1", block, 8, 20, "tune-reference")
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))


class TestTuningStateBytes(unittest.TestCase):
    def test_wrapper_params_excluded_from_state(self):
        import torch.nn as nn

        block = nn.Linear(4, 4)  # 16 + 4 params, any device (cpu here)
        # wrapper-style tuning tensor: same numel as the weight, registered
        # nowhere but present in a .params dict; identity-excluded from the count
        value = torch.zeros_like(block.weight)
        block.params = {"value": value}
        from auto_round.algorithms.quantization.sign_round.quantizer import _tuning_state_bytes

        # cpu target: logical params = 20 (weight 16 + bias 4), value excluded
        self.assertEqual(_tuning_state_bytes(block, "cpu"), 20 * 14)


if __name__ == "__main__":
    unittest.main()
