# coding=utf-8
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Micro-batch pipeline scheduling for device-mapped block tuning.

One tuning iteration forwards the whole ``batch_size`` batch through the
block; on a block whose modules sit on K devices the stages execute
sequentially (utilization 1/K). Splitting the forward batch on the sample
dimension into M micro-batches and overlapping stages across micro-batches
raises utilization to M/(M+K-1). The sample is the floor -- token-dim
splitting is forbidden because full-attention layers need each sample's
whole causal context.

This module owns the *schedule* only:

* :class:`GPipePlan` / :func:`plan_gpipe` -- the pure wait-flush schedule
  (all M forwards pipelined, then all M backwards pipelined; both phases
  fill the pipeline so the same utilization applies to fwd and bwd),
* :class:`GPipeDriver` -- executes the plan against stage callables and
  enforces the tuning-loop contract: micro-batch gradients are additive
  (their sum equals the whole-batch serial gradient up to fp reduction
  order), and the optimizer/sign step fires exactly once per iteration,
  after all M backwards.

Stage callables are composable segments of ONE autograd graph per
micro-batch; backwards therefore run at graph level, in the micro-batch
order the plan's backward phase visits. The (micro-batch, stage) backward
entries exist for stream/event placement when the CUDA-stream mechanics
land; on CPU devices they carry no synchronization meaning.

Production wiring today: ``DeviceStreamScope`` is the only class the
quantizer and calibration loop consume (the tune loop's phase-split lives
in ``_tune_batch_micro_batched`` because whole-block forwards have no
stage-callable decomposition). The plans, driver, slicer, and collection
runner here are the tested foundation for that integration and for
stage-granular stream placement; they carry the schedule contracts
(grad parity, one-step semantics, stash bounds) as CPU-tier unit tests.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch


@dataclass
class GPipePlan:
    """Execution order of a wait-flush (GPipe) micro-batch iteration.

    ``forward``/``backward`` list ``(micro_batch, stage)`` pairs; stage
    indices ascend in forward and descend in backward for every
    micro-batch.
    """

    forward: List[tuple] = field(default_factory=list)
    backward: List[tuple] = field(default_factory=list)


def plan_gpipe(microbatches: int, stages: int) -> GPipePlan:
    """Diagonal (pipeline-fill) order for M micro-batches over K stages.

    Forward: diagonal ``d = mb + stage`` ascending, so micro-batch ``i+1``
    enters stage 0 while micro-batch ``i`` is still in later stages.
    Backward: the same diagonals visited from the far end, mirroring the
    reverse stage order each micro-batch must obey.
    """
    if microbatches < 1 or stages < 1:
        raise ValueError(f"plan_gpipe needs microbatches>=1 and stages>=1, got {microbatches}/{stages}")
    forward = []
    for d in range(microbatches + stages - 1):
        for mb in range(microbatches):
            stage = d - mb
            if 0 <= stage < stages:
                forward.append((mb, stage))
    backward = []
    for d in reversed(range(microbatches + stages - 1)):
        for mb in range(microbatches):
            stage = d - mb
            if 0 <= stage < stages:
                backward.append((mb, stage))
    return GPipePlan(forward=forward, backward=backward)


class GPipeDriver:
    """Runs one tuning iteration under a :class:`GPipePlan`.

    ``stages`` are callables ``tensor -> tensor`` composing one autograd
    graph per micro-batch (the block's device-resident segments in
    execution order). ``loss_fn`` maps a stage-K output to a scalar loss;
    losses of the M micro-batches are reported as a list whose sum is the
    iteration loss. ``on_step`` (the optimizer/sign step hook) fires
    exactly once per iteration, after every micro-batch backward.
    """

    def __init__(
        self,
        stages: Sequence[Callable],
        streams: Optional["StageStreams"] = None,
        schedule: str = "gpipe",
    ):
        if not stages:
            raise ValueError("GPipeDriver needs at least one stage callable")
        if schedule not in ("gpipe", "1f1b"):
            raise ValueError(f"unknown schedule {schedule!r}; expected 'gpipe' or '1f1b'")
        self.stages = list(stages)
        self.streams = streams
        self.schedule = schedule

    def run_iteration(
        self,
        microbatches: Sequence,
        loss_fn: Callable,
        on_step: Optional[Callable] = None,
    ) -> List[torch.Tensor]:
        current = {mb: None for mb in range(len(microbatches))}
        losses = [None] * len(microbatches)
        done = set()

        def _fwd(mb, stage):
            if self.streams is not None:
                self.streams.before(mb, stage)
            if current[mb] is None:
                current[mb] = microbatches[mb]
            current[mb] = self.stages[stage](current[mb])
            if self.streams is not None:
                self.streams.after(mb, stage)

        def _bwd(mb):
            losses[mb] = loss_fn(current[mb])
            losses[mb].backward()
            current[mb] = None
            done.add(mb)

        if self.schedule == "1f1b":
            ops = plan_1f1b(len(microbatches), len(self.stages))
            for kind, mb, stage in ops:
                if kind == "fwd":
                    _fwd(mb, stage)
                elif mb not in done:
                    _bwd(mb)
        else:
            plan = plan_gpipe(len(microbatches), len(self.stages))
            for mb, stage in plan.forward:
                _fwd(mb, stage)
            for mb in range(len(microbatches)):
                losses[mb] = loss_fn(current[mb])
            bwd_order = []
            for mb, _stage in plan.backward:
                if mb not in bwd_order:
                    bwd_order.append(mb)
            for mb in bwd_order:
                losses[mb].backward()
        if on_step is not None:
            on_step()
        return losses


def slice_pool_rows(entry, rows):
    """Slice cached pool entries on the batch (sample) dimension.

    Handles the layouts the calibration cache produces: whole-batch tensors
    (``index_select`` on dim 0), per-sample lists (entry pick), rope-style
    tuples of tensors (each sliced on dim 0), dicts (recursively), and
    scalars (returned unchanged). ``rows`` is a non-empty list of sample
    indices.
    """
    import torch

    if not rows:
        raise ValueError("slice_pool_rows: empty row selection")
    if torch.is_tensor(entry):
        idx = torch.as_tensor(list(rows), dtype=torch.long, device=entry.device)
        return entry.index_select(0, idx)
    if isinstance(entry, list):
        return [entry[int(i)] for i in rows]
    if isinstance(entry, tuple) and entry and all(torch.is_tensor(t) for t in entry):
        return tuple(slice_pool_rows(t, rows) for t in entry)
    if isinstance(entry, dict):
        return {k: slice_pool_rows(v, rows) for k, v in entry.items()}
    return entry


class StageStreams:
    """Per-(micro-batch, stage) stream/event plumbing for one iteration.

    Real cross-stage overlap comes from CUDA streams: each stage's kernels
    are enqueued (by the single driver thread) on the stream of the device
    hosting that stage, and stage ``k+1`` of micro-batch ``i`` waits on an
    event recorded after stage ``k`` of the SAME micro-batch. On CPU-only
    worlds the object is inert (sequential execution is already correct;
    streams do not exist) -- before/after become no-ops.

    ``stream_factory``/``event_factory`` are injectable for testing; when
    they are omitted, streams/events are only created if every device is
    CUDA (otherwise the object stays inert).
    """

    def __init__(self, microbatches, stages, devices, stream_factory=None, event_factory=None):
        self.microbatches = microbatches
        self.stages = stages
        self.devices = list(devices)
        self._streams = {}
        self._events = {}
        self._active = True

        if stream_factory is None or event_factory is None:
            if not all(getattr(d, "type", "cpu") == "cuda" for d in self.devices):
                self._active = False
                return
            stream_factory = lambda d: torch.cuda.Stream(device=d)  # noqa: E731
            event_factory = lambda **kw: torch.cuda.Event(**kw)
        for mb in range(microbatches):
            for stage in range(stages):
                self._streams[(mb, stage)] = stream_factory(self.devices[stage % len(self.devices)])
                self._events[(mb, stage)] = event_factory()

    def before(self, micro_batch: int, stage: int) -> None:
        """Enter stage ``stage`` of ``micro_batch``: wait on its producer."""
        if not self._active:
            return
        if stage > 0:
            self._events[(micro_batch, stage - 1)].wait(stream=self._streams[(micro_batch, stage)])

    def after(self, micro_batch: int, stage: int) -> None:
        """Leave stage ``stage``: record the event stage+1 will wait on."""
        if not self._active:
            return
        self._events[(micro_batch, stage)].record(stream=self._streams[(micro_batch, stage)])


@contextlib.contextmanager
def device_stream_scope(devices, stream_factory=None):
    """Yield per-device CUDA streams and make them current for the block.

    Independent no-grad forwards enqueued back-to-back under this scope
    pipeline across devices for free: every op lands on the current stream
    of its tensors' device (PyTorch resolves op streams from input-tensor
    devices), per-device streams are FIFO across chunks, and cross-device
    copies synchronize their source/target streams internally (event-based,
    handled by torch itself). One stream per device is shared by all chunks
    -- the pipelining comes from enqueue order, not from per-chunk streams.

    CPU devices have no streams; a CPU-only (or mixed) world only gets
    streams for its CUDA devices, and a CPU-only world yields ``{}`` (the
    caller stays on default streams = sequential execution, which is
    already correct there).

    ``stream_factory`` is injectable for tests.
    """
    cuda_devs = [d for d in devices if getattr(d, "type", "cpu") == "cuda"]
    if not cuda_devs:
        yield {}
        return
    real_streams = stream_factory is None
    if stream_factory is None:
        stream_factory = lambda d: torch.cuda.Stream(device=d)  # noqa: E731
    streams = {d: stream_factory(d) for d in cuda_devs}
    if not real_streams:
        # injected (fake) streams: caller only inspects the mapping; entering
        # real CUDA device/stream contexts would crash CUDA-less test builds
        yield streams
        return
    cm_stack = contextlib.ExitStack()
    try:
        for dev, stream in streams.items():
            cm_stack.enter_context(torch.cuda.device(dev))
            cm_stack.enter_context(torch.cuda.stream(stream))
        yield streams
    finally:
        cm_stack.close()


def pipelined_nograd_forwards(forward_fn, chunks, devices):
    """Run independent no-grad chunk forwards under the device stream scope.

    The pre-tune fp-reference pass and the post-tune quantized cascade pass
    walk the calibration pool in ``batch_size`` chunks (e.g. 16 chunked
    [8, 2048] forwards for the default recipe). Each chunk is an
    independent forward; enqueuing them back-to-back inside
    :func:`device_stream_scope` overlaps device-resident segments across
    chunks (utilization 1/K -> #chunks/(#chunks+K-1) at chunk granularity)
    without splitting the block into stages or touching the align hooks.

    Returns the per-chunk outputs in chunk order. On CPU devices this is a
    plain sequential loop.
    """
    if not chunks:
        raise ValueError("pipelined_nograd_forwards: no chunks to run")
    import torch

    with torch.no_grad(), device_stream_scope(devices) as _streams:
        outputs = [forward_fn(chunk) for chunk in chunks]
    return outputs


def plan_1f1b(microbatches: int, stages: int):
    """One-forward-one-backward schedule with the in-flight stash bounded at K.

    After a warmup of ``stages`` micro-batch forwards, each additional
    forward is immediately followed by the backward of the oldest
    in-flight micro-batch (micro-batch gradients are independent and
    additive, so backward order is free). GPipe stashes all M forward
    graphs before the first backward; this schedule bounds the live
    graphs at ~K, which is what unlocks raising ``batch_size`` under the
    same activation memory.

    Returns ``(kind, micro_batch, stage)`` op tuples; forward stage
    entries keep the per-micro-batch stage order, backward entries carry
    the stage they unpin (informational for stream/event placement).
    """
    if microbatches < 1 or stages < 1:
        raise ValueError(f"plan_1f1b needs microbatches>=1 and stages>=1, got {microbatches}/{stages}")
    plan = plan_gpipe(microbatches, stages)
    fwd_by_mb = {}
    for mb, stage in plan.forward:
        fwd_by_mb.setdefault(mb, []).append((mb, stage))
    ops = []
    in_flight = []
    for mb in range(microbatches):
        # pop-before-forward: the oldest in-flight backward fires BEFORE the
        # next forward once the pipeline is full, keeping in-flight <= K at
        # every prefix (the stash bound this schedule exists for)
        if len(in_flight) == stages:
            done = in_flight.pop(0)
            ops.extend(("bwd", done, stage) for stage in reversed(range(stages)))
        ops.extend(("fwd", *pair) for pair in fwd_by_mb[mb])
        in_flight.append(mb)
    while in_flight:
        done = in_flight.pop(0)
        ops.extend(("bwd", done, stage) for stage in reversed(range(stages)))
    return ops


class DeviceStreamScope:
    """Cached per-device CUDA streams, re-enterable across forward calls.

    One :func:`device_stream_scope` context creates fresh streams each time,
    which would break cross-call FIFO ordering. Calibration loops that want
    pipelined batch forwards keep a single ``DeviceStreamScope`` alive and
    re-enter ``context()`` around each forward: every forward's ops enqueue
    on the same per-device streams, so consecutive batches pipeline while
    per-device order stays FIFO. CPU-only worlds carry no streams and the
    context is a no-op.
    """

    def __init__(self, devices, stream_factory=None):

        self.devices = list(devices)
        cuda_devs = [d for d in self.devices if getattr(d, "type", "cpu") == "cuda"]
        self.streams = {}
        if len(cuda_devs) >= 2:
            # a single CUDA device gains nothing from a side stream (no
            # cross-device overlap to expose) and pays bridging overhead
            if stream_factory is None:
                stream_factory = lambda d: torch.cuda.Stream(device=d)  # noqa: E731
            self.streams = {d: stream_factory(d) for d in cuda_devs}

    @contextlib.contextmanager
    def context(self):

        if not self.streams:
            yield {}
            return
        cm_stack = contextlib.ExitStack()
        try:
            for dev, stream in self.streams.items():
                # bridge IN: work staged on the device's default stream before
                # this call (input H2D copies, optimizer steps) must be visible
                # to the scope stream before its kernels run
                stream.wait_stream(torch.cuda.default_stream(dev))
                cm_stack.enter_context(torch.cuda.device(dev))
                cm_stack.enter_context(torch.cuda.stream(stream))
            yield self.streams
        finally:
            for dev, stream in self.streams.items():
                # bridge OUT: work enqueued on the scope stream must complete
                # before anything the caller runs next on the default stream
                # (parameter reads, .item()s, cache captures) -- otherwise the
                # next consumer races the pipelined kernels
                torch.cuda.default_stream(dev).wait_stream(stream)
            cm_stack.close()
