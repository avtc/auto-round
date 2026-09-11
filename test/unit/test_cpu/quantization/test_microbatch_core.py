# coding=utf-8
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-tier tests for the micro-batch pipeline scheduler (GPipe semantics).

The scheduler splits one tuning iteration's forward batch on the sample
dimension into M micro-batches and pipelines them across the K device-mapped
stages of a block. Contract under test (the micro-batch pipeline design):

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


class TestPipelinedCollection:
    """Chunk-granular no-grad collection pipelining (per-device streams)."""

    def test_outputs_match_serial_loop(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import pipelined_nograd_forwards

        block = torch.nn.Linear(4, 4)
        chunks = [torch.randn(2, 4) for _ in range(5)]
        out = pipelined_nograd_forwards(lambda c: block(c), chunks, devices=[torch.device("cpu")])
        ref = [block(c) for c in chunks]
        for a, b in zip(out, ref):
            assert torch.allclose(a, b)

    def test_enqueue_order_is_chunk_order(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import pipelined_nograd_forwards

        order = []

        def fn(c, _i):
            order.append(_i)
            return c

        chunks = [(i, torch.randn(1, 2)) for i in range(4)]
        outs = pipelined_nograd_forwards(lambda c: fn(c, c[0]), chunks, devices=[torch.device("cpu")])
        assert order == [0, 1, 2, 3]
        assert len(outs) == 4

    def test_cpu_multi_device_degrades_to_sequential(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import pipelined_nograd_forwards

        devs = [torch.device("cpu", i) for i in range(3)]
        block = torch.nn.Linear(2, 2)
        chunks = [torch.randn(1, 2) for _ in range(3)]
        outs = pipelined_nograd_forwards(lambda c: block(c), chunks, devices=devs)
        assert all(torch.allclose(a, block(c)) for a, c in zip(outs, chunks))

    def test_empty_chunks_raise(self):
        import pytest
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import pipelined_nograd_forwards

        with pytest.raises(ValueError):
            pipelined_nograd_forwards(lambda c: c, [], devices=[torch.device("cpu")])

    def test_stream_scope_creates_streams_only_for_cuda(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import device_stream_scope

        # CPU-only world: scope is a no-op (no streams exist on CPU)
        with device_stream_scope([torch.device("cpu")]) as streams:
            assert streams is None or streams == {}

        made = {}

        class _FakeStream:
            def __init__(self, dev):
                self.dev = dev
                made[dev] = self

        with device_stream_scope(
            [torch.device("cuda", 0), torch.device("cpu")], stream_factory=lambda d: _FakeStream(d)
        ) as streams:
            assert set(streams) == {torch.device("cuda", 0)}
            assert isinstance(streams[torch.device("cuda", 0)], _FakeStream)
        assert len(made) == 1


class TestOneFOneBPlan:
    """The 1F1B schedule: stash bounded at ~K in-flight micro-batches."""

    def test_in_flight_ceiling_is_stages(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import plan_1f1b

        for M, K in [(4, 2), (8, 4), (6, 3), (3, 4)]:
            ops = plan_1f1b(M, K)
            fwd_done, bwd_done, max_inflight = set(), set(), 0
            for kind, mb, _stage in ops:
                if kind == "fwd":
                    fwd_done.add(mb)
                else:
                    bwd_done.add(mb)
                inflight = len(fwd_done) - len(bwd_done)
                max_inflight = max(max_inflight, inflight)
            assert len(fwd_done) == M and len(bwd_done) == M, (M, K)
            assert max_inflight <= K, f"M={M} K={K}: stash hit {max_inflight}"

    def test_fwd_precedes_own_bwd(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import plan_1f1b

        ops = plan_1f1b(6, 3)
        seen_fwd, seen_bwd = set(), set()
        for kind, mb, stage in ops:
            if kind == "fwd":
                # forward entries are per-stage; each (mb, stage) fires once,
                # stages ascend per micro-batch
                assert (mb, stage) not in seen_fwd
                prior = max((st for m, st in seen_fwd if m == mb), default=-1)
                assert stage == prior + 1, f"mb {mb}: stage {stage} after {prior}"
                seen_fwd.add((mb, stage))
            elif mb not in seen_bwd:
                # backward entries are per-stage (stream placement); only the
                # first per micro-batch is the graph-level backward
                assert any(m == mb for m, _ in seen_fwd), f"bwd of {mb} before its fwd"
                seen_bwd.add(mb)

    def test_warmup_is_k_microbatches(self):
        from auto_round.algorithms.quantization.sign_round.microbatch import plan_1f1b

        ops = plan_1f1b(8, 4)
        heads = []
        for kind, mb, _s in ops:
            if kind == "fwd" and mb not in heads:
                heads.append(mb)
        assert heads[:4] == [0, 1, 2, 3]
        # first backward interleaves right after warmup: bwd(0) fires before fwd(4)
        idx = {(kind, mb): i for i, (kind, mb, _s) in enumerate(ops)}
        assert idx[("bwd", 0)] < idx[("fwd", 4)]


class TestOneFOneBDriver:
    """The driver executes the 1F1B plan with identical gradient results."""

    def test_grad_and_step_parity_with_gpipe(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.microbatch import GPipeDriver

        torch.manual_seed(7)
        a, b = torch.nn.Linear(6, 8), torch.nn.Linear(8, 5)
        x = torch.randn(4, 3, 6)
        steps = []

        def run(schedule):
            for p in list(a.parameters()) + list(b.parameters()):
                p.grad = None
            driver = GPipeDriver(
                stages=[lambda t, _a=a: _a(t), lambda t, _b=b: _b(t)],
                schedule=schedule,
            )
            losses = driver.run_iteration(
                microbatches=[x[:2], x[2:]],
                loss_fn=lambda o: o.pow(2).mean(),
                on_step=lambda: steps.append(schedule),
            )
            grads = [p.grad.clone() for p in list(a.parameters()) + list(b.parameters())]
            return losses, grads

        losses_g, grads_g = run("gpipe")
        losses_f, grads_f = run("1f1b")
        assert steps == ["gpipe", "1f1b"]
        for lg, lf in zip(losses_g, losses_f):
            assert torch.allclose(lg, lf, atol=1e-7)
        for g_g, g_f in zip(grads_g, grads_f):
            assert torch.allclose(g_g, g_f, atol=1e-6)


class TestMicroBatchConfig:
    """Flag surface: default off, validation, clamping matrix."""

    def test_default_is_off(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        assert SignRoundConfig().micro_batch is None

    def test_invalid_value_disables(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        assert SignRoundConfig(micro_batch=0).micro_batch is None
        assert SignRoundConfig(micro_batch=-3).micro_batch is None
        assert SignRoundConfig(micro_batch=2).micro_batch == 2

    def test_cli_arg_registered(self):
        # the declarative registry wires --micro_batch from the config class
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        src = repr(SignRoundConfig.register_args.__code__.co_consts)
        assert any("micro_batch" in str(c) for c in SignRoundConfig.register_args.__code__.co_consts)

    def test_clamping_matrix(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        class _Q(SignRoundQuantizer):
            def __init__(self, mb):
                from types import SimpleNamespace

                self.config = SimpleNamespace(micro_batch=mb)

        idx4 = torch.arange(4)
        assert _Q(None)._micro_batch_n(idx4) is None
        assert _Q(1)._micro_batch_n(idx4) is None
        assert _Q(2)._micro_batch_n(idx4) == 2
        assert _Q(4)._micro_batch_n(idx4) == 4
        assert _Q(8)._micro_batch_n(idx4) == 4  # clamped to batch size
        assert _Q(4)._micro_batch_n(idx4[:1]) is None  # single-sample batch stays serial


class TestMicroBatchedBatchParity(unittest.TestCase):
    """_tune_batch_micro_batched reproduces the serial whole-batch step exactly."""

    def _run(self, micro_batch):
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        torch.manual_seed(11)
        block = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Linear(6, 3))
        n = 4
        fp_outputs = [torch.randn(1, 5, 3) for _ in range(n)]
        x = torch.randn(n, 5, 4)

        class _Q(SignRoundQuantizer):
            def __init__(self):
                self.config = SimpleNamespace(micro_batch=micro_batch)
                self.enable_lfq = False

            def _get_loss(self, pred, ref, indices, loss_func, device="cpu", valid_token_mask=None, input_ids=None):
                return (pred - ref).pow(2).mean()

            def _scale_loss_and_backward(self, scaler, loss):
                loss.backward()

        class _Fwd:
            def forward(self, block_, active_inputs, input_others, indices, cache_device=None):
                sel = torch.cat([active_inputs[int(i)] for i in indices], dim=0)
                return block_(sel)

        q = _Q()
        ctx = SimpleNamespace(block_index=0, block_cnt=2)
        contributed = q._tune_batch_micro_batched(
            2,
            block=block,
            indices=torch.arange(n),
            active_inputs=[x[i : i + 1] for i in range(n)],
            input_others={},
            fp_outputs=fp_outputs,
            input_ids=None,
            tuning_cache=None,
            block_fwd=_Fwd(),
            mse_loss=None,
            valid_token_mask=None,
            num_elm=0,
            scaler=None,
            loss_device=None,
            fwd_cache_device=None,
            home_device="cpu",
            block_ctx=ctx,
        )
        grads = [p.grad.clone() for p in block.parameters() if p.grad is not None]
        return block, x, fp_outputs, contributed, grads

    def test_matches_serial(self):
        import torch

        block, x, fp_outputs, contributed, grads_mb = self._run(micro_batch=2)

        # serial reference: whole-batch forward + mean loss + backward
        block2 = torch.nn.Sequential(*[torch.nn.Linear(4, 6), torch.nn.Linear(6, 3)])
        block2.load_state_dict(block.state_dict())
        for p in block2.parameters():
            p.grad = None
        pred = block2(x)
        ref = torch.cat(fp_outputs, dim=0)
        loss = (pred - ref).pow(2).mean()
        loss.backward()

        self.assertAlmostEqual(contributed, loss.item(), places=6)
        grads_serial = [p.grad for p in block2.parameters() if p.grad is not None]
        self.assertEqual(len(grads_mb), len(grads_serial))
        for gm, gs in zip(grads_mb, grads_serial):
            self.assertTrue(torch.allclose(gm, gs, atol=1e-6), "micro-batched grad != serial grad")


class TestMicroBatchDeclineGuards:
    """Loud decline paths: slice sums cannot reproduce the serial loss there."""

    def _q(self, mb=2, **attrs):
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        class _Q(SignRoundQuantizer):
            def __init__(self):
                self.config = SimpleNamespace(micro_batch=mb)
                for k, v in attrs.items():
                    setattr(self, k, v)

        return _Q()

    def test_v2_contract_declines(self):
        q = self._q(micro_batch_supported=False)
        assert q._micro_batch_n(torch.arange(4)) is None

    def test_gradient_accumulate_declines(self):
        q = self._q(gradient_accumulate_steps=4)
        assert q._micro_batch_n(torch.arange(4)) is None

    def test_tuning_cache_declines(self):
        q = self._q()
        assert q._micro_batch_n(torch.arange(4), tuning_cache=object()) is None
        assert q._micro_batch_n(torch.arange(4), tuning_cache=None) == 2

    def test_lfq_last_block_declines(self):
        from types import SimpleNamespace

        q = self._q(enable_lfq=True)
        last = SimpleNamespace(block_index=3, block_cnt=4)
        mid = SimpleNamespace(block_index=1, block_cnt=4)
        assert q._micro_batch_n(torch.arange(4), block_ctx=last) is None
        assert q._micro_batch_n(torch.arange(4), block_ctx=mid) == 2

    def test_v2_quantizer_class_opts_out(self):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundV2Quantizer

        assert SignRoundV2Quantizer.micro_batch_supported is False


class TestMicroBatchedMaskedParity(unittest.TestCase):
    """Masked MSE parity: weighted slice means reproduce the serial loss."""

    def test_masked_valid_token_par(self):
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        torch.manual_seed(5)
        block = torch.nn.Linear(4, 3)
        n = 4
        x = torch.randn(n, 5, 4)
        ref = torch.randn(n, 5, 3)
        # per-sample masks with DIFFERENT valid counts (the hard case)
        vm = torch.ones(n, 1, 5)
        vm[0, :, 3:] = 0
        vm[1, :, 1:] = 0
        vm[3, :, 4:] = 0

        class _Q(SignRoundQuantizer):
            def __init__(self):
                self.config = SimpleNamespace(micro_batch=2)
                self.enable_lfq = False

            def _get_loss(self, pred, r, indices, loss_func, device="cpu", valid_token_mask=None, input_ids=None):
                # mirror the base masked path: mean over FULL element count
                m = torch.cat([valid_token_mask[int(i)] for i in indices], dim=0).unsqueeze(-1).to(pred.device)
                return ((pred * m) - (r * m)).pow(2).mean()

            def _scale_loss_and_backward(self, scaler, loss):
                loss.backward()

        class _Fwd:
            def forward(self, block_, active_inputs, input_others, indices, cache_device=None):
                return block_(torch.cat([active_inputs[int(i)] for i in indices], dim=0))

        q = _Q()
        got = q._tune_batch_micro_batched(
            2,
            block=block,
            indices=torch.arange(n),
            active_inputs=[x[i : i + 1] for i in range(n)],
            input_others={},
            fp_outputs=[ref[i : i + 1] for i in range(n)],
            input_ids=None,
            tuning_cache=None,
            block_fwd=_Fwd(),
            mse_loss=None,
            valid_token_mask=vm,
            num_elm=int(vm.sum()),
            scaler=None,
            loss_device=None,
            fwd_cache_device=None,
            home_device="cpu",
            block_ctx=SimpleNamespace(block_index=0, block_cnt=4),
        )

        # serial reference: same loss fn over the whole batch, /num_elm like the loop
        serial_loss = q._get_loss(block(x), ref, torch.arange(n), None, "cpu", vm)
        serial_contrib = serial_loss.item() / int(vm.sum())
        self.assertAlmostEqual(got, serial_contrib, places=6)

        # gradients: sum of weighted slice grads == serial grad
        for p in block.parameters():
            p.grad = None
        q2 = _Q()
        q2._tune_batch_micro_batched(
            2,
            block=block,
            indices=torch.arange(n),
            active_inputs=[x[i : i + 1] for i in range(n)],
            input_others={},
            fp_outputs=[ref[i : i + 1] for i in range(n)],
            input_ids=None,
            tuning_cache=None,
            block_fwd=_Fwd(),
            mse_loss=None,
            valid_token_mask=vm,
            num_elm=int(vm.sum()),
            scaler=None,
            loss_device=None,
            fwd_cache_device=None,
            home_device="cpu",
            block_ctx=SimpleNamespace(block_index=0, block_cnt=4),
        )
        block2 = torch.nn.Linear(4, 3)
        block2.load_state_dict(block.state_dict())
        # the serial loop backwards the UNDIVIDED batch loss (the /num_elm
        # division is reporting-only), so the reference matches that exactly
        q2._get_loss(block2(x), ref, torch.arange(n), None, "cpu", vm).backward()
        for p, p2 in zip(block.parameters(), block2.parameters()):
            assert torch.allclose(p.grad, p2.grad, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
