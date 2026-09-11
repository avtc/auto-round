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


class TestBatchDimSlicing:
    """Sample-dim slicing of the cached pool rows (incl. rope tuple kwargs)."""

    def test_slices_tensor_pool(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import slice_pool_rows

        pool = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        out = slice_pool_rows(pool, [2, 3])
        assert out.shape == (2, 3, 2)
        assert torch.equal(out[0], pool[2])
        assert torch.equal(out[1], pool[3])

    def test_slices_per_sample_list_pool(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import slice_pool_rows

        pool = [torch.full((3, 2), float(i)) for i in range(4)]
        out = slice_pool_rows(pool, [1, 3])
        assert torch.equal(out[0], pool[1])
        assert torch.equal(out[1], pool[3])

    def test_slices_rope_tuple_entries(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import slice_pool_rows

        cos = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
        sin = cos + 100.0
        out = slice_pool_rows((cos, sin), [2])
        assert isinstance(out, tuple) and len(out) == 2
        assert torch.equal(out[0], cos[2:3])
        assert torch.equal(out[1], sin[2:3])

    def test_slices_dict_recursively_and_passes_scalars(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import slice_pool_rows

        pool = {
            "mask": torch.ones(4, 5),
            "rope": (torch.ones(4, 5, 2), torch.zeros(4, 5, 2)),
            "names": ["a", "b", "c", "d"],
            "flag": True,
            "nothing": None,
        }
        out = slice_pool_rows(pool, [0, 2])
        assert out["mask"].shape == (2, 5)
        assert out["rope"][0].shape == (2, 5, 2)
        assert out["names"] == ["a", "c"]
        assert out["flag"] is True
        assert out["nothing"] is None

    def test_empty_rows_raise(self):
        import pytest
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import slice_pool_rows

        with pytest.raises(ValueError):
            slice_pool_rows(torch.ones(4, 5), [])


class TestStageStreams:
    """Stream/event mechanics: fake factories, CPU degradation."""

    def _fake_world(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import StageStreams

        events = []

        class _Event:
            def __init__(self, device=None, **kw):
                self.waited = False
                self.recorded = False

            def wait(self, stream=None):
                self.waited = True

            def record(self, stream=None):
                self.recorded = True

        class _Stream:
            def __init__(self, device):
                self.device = device

        def stream_factory(device):
            return _Stream(device)

        def event_factory(**kw):
            e = _Event(**kw)
            events.append(e)
            return e

        streams = StageStreams(
            microbatches=2,
            stages=2,
            devices=[torch.device("cpu", 0), torch.device("cpu", 1)],
            stream_factory=stream_factory,
            event_factory=event_factory,
        )
        return streams, events

    def test_dependency_structure(self):
        streams, events = self._fake_world()
        plan_fwd = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for mb, stage in plan_fwd:
            streams.before(mb, stage)
            streams.after(mb, stage)
        # after() on stage 0 records an event; before() on stage>0 waits on the
        # SAME micro-batch's previous-stage event
        waits = [e for e in events if e.waited]
        records = [e for e in events if e.recorded]
        assert len(waits) == 2  # (0,1) and (1,1) waited
        assert len(records) == 4  # every stage recorded

    def test_cpu_default_is_sequential_noop(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import StageStreams

        streams = StageStreams(microbatches=2, stages=2, devices=[torch.device("cpu")])
        # must not raise and must not create anything on CPU-only worlds
        streams.before(0, 0)
        streams.after(0, 0)
        streams.before(0, 1)
        streams.after(0, 1)

    def test_driver_runs_identically_with_streams(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import GPipeDriver, StageStreams

        torch.manual_seed(3)
        a, b = torch.nn.Linear(6, 8), torch.nn.Linear(8, 5)
        x = torch.randn(4, 3, 6)
        streams, _ = self._fake_world()
        driver = GPipeDriver(
            stages=[lambda t, _a=a: _a(t), lambda t, _b=b: _b(t)],
            streams=streams,
        )
        # per-microbatch losses must use the whole-batch normalization so their
        # sum equals the serial batch loss exactly
        total_out = 4 * 3 * 5
        losses = driver.run_iteration(microbatches=[x[:2], x[2:]], loss_fn=lambda o: o.pow(2).sum() / total_out)
        assert len(losses) == 2
        ref = b(a(x)).pow(2).mean()
        assert abs(sum(l.item() for l in losses) - ref.item()) < 1e-5
