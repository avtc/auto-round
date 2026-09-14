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


def _run(pool, block, free, target="cuda:1", iters=20, label="tune-reference", **kw):
    with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=free), mock.patch(
        "auto_round.utils.pool_placement._working_allowance_bytes", return_value=int(1.4 * _GIB)
    ):
        return _pull_pool_if_fits(pool, target, block, 8, iters, label, **kw)


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

    def test_input_pull_charges_routed_budget(self):
        # the OOMed 4-GPU lane: entry free ~15.9 GiB, pool 4, routed budget
        # 12 GiB (tokens x top_k x hidden x 6, from the pr/streaming formula)
        # -> declines; with no MoE activation cost the same numbers pull
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.quantizer._block_activation_bytes",
            return_value=int(12 * _GIB),
        ):
            pool = [_FakeTensor("cuda:2", numel=_GIB // 4) for _ in range(4)]
            block = _FakeBlock([])
            out = _run(pool, block, free=15.9 * _GIB, target="cuda:0", label="tune-input", charge_activation=True)
            self.assertTrue(all(t.moved_to is None for t in out))
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.quantizer._block_activation_bytes", return_value=0
        ):
            pool2 = [_FakeTensor("cuda:2", numel=_GIB // 4) for _ in range(4)]
            out2 = _run(pool2, block, free=15.9 * _GIB, target="cuda:0", label="tune-input", charge_activation=True)
            self.assertTrue(all(t.moved_to == torch.device("cuda:0") for t in out2))

    def test_routed_budget_formula_reproduces_measured_lane(self):
        # hy3: batch 8 x seq 2048 x top_k 8 x hidden 4096 x fp32 x 6 ~= 12.0 GiB
        # (measured loop retention: 12.7 GiB of batch cats + routed caches)
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):  # routed container (hy3 mlp.experts)
            num_experts = 192

        ref = torch.zeros(1, 2048, 4096)  # fp32 pool sample
        block = nn.Sequential(_Experts(), nn.Linear(4, 4))
        config = type("C", (), {"num_experts_per_tok": 8})()
        got = q._block_activation_bytes(block, [ref], 8, config)
        self.assertAlmostEqual(got / _GIB, 12.0, delta=0.05)

    def test_recorded_dispatch_shapes_beat_missing_config(self):
        # config absent + module attrs absent -> recorded (top_k, rows) decides
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):
            num_experts = 192

        exp = _Experts()
        exp._routed_shape_rec_ = (8, 8 * 2048 * 8)  # seen at batch 8, seq 2048
        block = nn.Sequential(exp)
        ref = torch.zeros(1, 2048, 4096)
        got = q._block_activation_bytes(block, [ref], 8, config=None)
        self.assertAlmostEqual(got / _GIB, 12.0, delta=0.05)

    def test_estimator_and_routed_composed_by_max(self):
        # a big shared expert (plain module) makes the estimator term win;
        # routed alone would undercharge it -- max covers both
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):
            num_experts = 192

        class _Shared(nn.Module):  # plain shared MLP, full-token width
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4096, 13312)

        exp = _Experts()
        exp.gate_proj = nn.Linear(4096, 1536)
        block = nn.Sequential(exp, _Shared())
        block.config = type("C", (), {"num_experts_per_tok": 8})()
        # mark them quantizable-ish for the estimator's check_to_quantized
        for m in (exp.gate_proj, block[1].gate_proj):
            m.orig_layer = m
            m.bits = 4
            m.act_bits = 16
            m.group_size = 128
        ref = torch.zeros(1, 2048, 4096)
        got = q._block_activation_bytes(block, [ref], 8, block.config)
        routed = 8 * 2048 * 8 * 4096 * 4 * 6  # 12 GiB
        # shared 13312-wide gate at full tokens x2 grads ~ 1.56 GiB + expert
        # module output... estimator path runs the real estimator; assert the
        # composition never returns less than the routed term
        self.assertGreaterEqual(got, routed)

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
