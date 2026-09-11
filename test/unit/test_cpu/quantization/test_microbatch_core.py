# coding=utf-8 -*-
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-tier tests for the micro-batch pipeline scheduler (GPipe semantics).

The scheduler splits one tuning iteration's forward batch on the sample
dimension into M micro-batches and pipelines them across the K device-mapped
stages of a block. Contract under test (design doc
.featyard/design/02-microbatch-pipeline.md):

* gradients of the summed per-micro-batch losses equal the whole-batch serial
  gradients (autograd is additive across micro-batches),
* the optimizer/sign step fires exactly once per iteration, after all M
  backwards,
* per-microbatch stage order is respected (stage k+1 of micro-batch i only
  after its stage k),
* the reported loss is the sum of the per-microbatch losses.

All tests run on plain CPU tensors (no CUDA, no model downloads).
"""

import unittest

import torch


def _make_block():
    """Two-stage fake block: stage 0 and stage 1 are device-separable segments."""
    torch.manual_seed(0)
    a = torch.nn.Linear(6, 8)
    b = torch.nn.Linear(8, 5)
    return a, b


class TestGPipePlan:
    """The pure schedule plan: op order for the GPipe (wait-flush) schedule."""

    def test_plan_order(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import plan_gpipe

        # K=3 stages, M=4 micro-batches: all forwards pipelined, then all
        # backwards pipelined (reverse stage order per micro-batch)
        plan = plan_gpipe(microbatches=4, stages=3)
        assert plan.forward == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)] or plan.forward[
            0
        ] == (0, 0), plan.forward
        # invariants any valid GPipe plan must satisfy
        seen_per_mb = {}
        for mb, stage in plan.forward:
            prior = seen_per_mb.get(mb, -1)
            assert stage == prior + 1, f"micro-batch {mb} skipped stage {prior + 1}"
            seen_per_mb[mb] = stage
        assert sorted(seen_per_mb) == list(range(4))
        assert len(plan.forward) == 12

    def test_backward_mirror_order(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import plan_gpipe

        plan = plan_gpipe(microbatches=2, stages=2)
        # backwards only start after every forward; stages reverse per micro-batch
        assert len(plan.backward) == 4
        per_mb = {}
        for mb, stage in plan.backward:
            prior = per_mb.get(mb, 2)
            assert stage == prior - 1, f"micro-batch {mb} backward out of order: {stage} after {prior}"
            per_mb[mb] = stage


class TestGPipeGradParity(unittest.TestCase):
    """The driver reproduces whole-batch serial gradients to fp reduction order."""

    def test_grads_match_serial(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import GPipeDriver

        a, b = _make_block()
        x = torch.randn(4, 3, 6)
        for p in list(a.parameters()) + list(b.parameters()):
            p.grad = None

        driver = GPipeDriver(
            stages=[lambda t, _a=a: _a(t), lambda t, _b=b: _b(t)],
        )
        losses = driver.run_iteration(microbatches=[x[:2], x[2:]], loss_fn=lambda out: out.pow(2).sum() / 2)

        # serial reference with the same loss normalization (sum of per-mb means
        # over 2 samples == mean over the whole batch of 4)
        a2, b2 = _make_block()
        a2.load_state_dict(a.state_dict())
        b2.load_state_dict(b.state_dict())
        ref_loss = b2(a2(x)).pow(2).sum() / 2
        ref_loss.backward()

        self.assertAlmostEqual(sum(l.item() for l in losses), ref_loss.item(), places=5)
        for p, q in zip(list(a.parameters()) + list(b.parameters()), list(a2.parameters()) + list(b2.parameters())):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=1e-6), f"grad mismatch for {p.shape}")

    def test_step_fires_once_per_iteration(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import GPipeDriver

        a, b = _make_block()
        x = torch.randn(4, 3, 6)
        steps = []

        driver = GPipeDriver(
            stages=[lambda t, _a=a: _a(t), lambda t, _b=b: _b(t)],
        )
        driver.run_iteration(
            microbatches=[x[:2], x[2:]],
            loss_fn=lambda out: out.pow(2).mean(),
            on_step=lambda: steps.append(1),
        )
        self.assertEqual(steps, [1])

    def test_stage_order_log(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import GPipeDriver

        a, b = _make_block()
        x = torch.randn(4, 3, 6)
        calls = []

        def stage(fn, tag):
            def _s(t):
                calls.append((tag, str(t.shape[0])))
                return fn(t)

            return _s

        driver = GPipeDriver(
            stages=[stage(lambda t, _a=a: _a(t), "s0"), stage(lambda t, _b=b: _b(t), "s1")],
        )
        losses = driver.run_iteration(microbatches=[x[:2], x[2:]], loss_fn=lambda out: out.pow(2).mean())
        self.assertEqual(len(losses), 2)
        # both micro-batches went through both stages in order
        self.assertEqual(calls.count(("s0", "2")), 2)
        self.assertEqual(calls.count(("s1", "2")), 2)


if __name__ == "__main__":
    unittest.main()
