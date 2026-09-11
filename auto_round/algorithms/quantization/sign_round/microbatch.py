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
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


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

    def __init__(self, stages: Sequence[Callable], streams: Optional["StageStreams"] = None):
        if not stages:
            raise ValueError("GPipeDriver needs at least one stage callable")
        self.stages = list(stages)
        self.streams = streams

    def run_iteration(
        self,
        microbatches: Sequence,
        loss_fn: Callable,
        on_step: Optional[Callable] = None,
    ) -> List["torch.Tensor"]:  # noqa: F821 - torch typed lazily for CPU-only tests
        plan = plan_gpipe(len(microbatches), len(self.stages))
        current = {mb: None for mb in range(len(microbatches))}
        losses = [None] * len(microbatches)
        for mb, stage in plan.forward:
            if self.streams is not None:
                self.streams.before(mb, stage)
            if current[mb] is None:
                current[mb] = microbatches[mb]
            current[mb] = self.stages[stage](current[mb])
            if self.streams is not None:
                self.streams.after(mb, stage)
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
        import torch

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
