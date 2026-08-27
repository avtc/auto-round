# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-process data parallelism for the SignRound tuning loop.

Runs one block's iteration loop on the home GPU plus W-1 mirror replicas
(persistent deep copies of the wrapped block). Per iteration the global
calibration batch is sharded across replicas; each replica computes its
forward/loss/backward on its own device; gradients are exchanged with a
halving-doubling all-reduce over flat fp32 buffers so every replica ends
with the same averaged gradient; the deterministic SignSGD step is then
applied locally on every replica (identical inputs -> identical update),
which keeps mirrors in sync without any parameter broadcast.

World size 1 (default) executes none of this code.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from auto_round.logger import logger


@dataclass
class DDPPlan:
    """Resolved data-parallel plan for one block."""

    world: int
    devices: List[torch.device]
    shard_size: int
    notes: List[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.world > 1


def parse_ddp_groups(spec: Optional[str]) -> Optional[List[List[torch.device]]]:
    """Parse AR_TUNE_DDP_GROUPS syntax ``"0,1,2,3;4,5,6,7"`` into device groups.

    Ping-pong staging: blocks alternate between groups (home = group leader,
    following the streaming round-robin over leaders), each block's mirrors
    are the rest of its group, and the other group prefetches the next block.
    """
    if not spec or not str(spec).strip():
        return None
    groups = []
    for part in str(spec).split(";"):
        items = [t.strip() for t in part.split(",") if t.strip()]
        if not items:
            continue
        groups.append([torch.device(f"cuda:{t}") if t.isdigit() else torch.device(t) for t in items])
    return groups or None


def resolve_ddp_plan(
    world: int,
    home: torch.device,
    batch_size: int,
    visible_cuda_devices: Optional[Sequence[int]] = None,
    explicit_devices: Optional[Sequence] = None,
    vram_free_bytes: Optional[int] = None,
    mirror_footprint_bytes: Optional[int] = None,
    margin_bytes: int = 2 * 1024**3,
    groups: Optional[List[List[torch.device]]] = None,
) -> DDPPlan:
    """Pick mirror devices for ``world``-way data parallelism of one block.

    Rules (each demotion is recorded in ``notes``):
    - world <= 1 or non-CUDA home -> disabled
    - batch_size % world != 0 -> disabled (shard means must reproduce the
      global mean exactly)
    - explicit device list -> use as-is after the home (deduplicated)
    - otherwise home + next devices in ascending visible order
    - per-mirror VRAM guard: skip any device that cannot hold the mirror
      footprint with margin; the world shrinks accordingly
    """
    notes: List[str] = []
    if world is None or world <= 1:
        return DDPPlan(1, [home], batch_size, notes)
    if home.type != "cuda":
        notes.append("home device is not CUDA")
        return DDPPlan(1, [home], batch_size, notes)
    world = int(world)
    if groups:
        # ping-pong mode: mirrors are the block's own group (rotated so the
        # streaming home leads); the requested world must match the group size
        own = [g for g in groups if home in g]
        if not own:
            notes.append(f"home {home} is not part of any AR_TUNE_DDP_GROUPS group")
            return DDPPlan(1, [home], batch_size, notes)
        g = own[0]
        if len(g) != world:
            notes.append(f"group size {len(g)} != requested world {world}; using group size")
            world = len(g)
        if batch_size % world != 0:
            notes.append(f"batch_size {batch_size} not divisible by world {world}")
            return DDPPlan(1, [home], batch_size, notes)
        devices = [home] + [d for d in g if d != home]
        if vram_free_bytes is not None and mirror_footprint_bytes is not None:
            kept = [devices[0]]
            for dev in devices[1:]:
                free = vram_free_bytes.get(dev, 0) if hasattr(vram_free_bytes, "get") else vram_free_bytes
                if free < mirror_footprint_bytes + margin_bytes:
                    notes.append(f"skip {dev}: free {free / 2**30:.1f}GiB < footprint+margin")
                    continue
                kept.append(dev)
            devices = kept
        if len(devices) < 2:
            notes.append("no mirror device passed the VRAM guard")
            return DDPPlan(1, [home], batch_size, notes)
        return DDPPlan(len(devices), devices, batch_size // len(devices), notes)
    if batch_size % world != 0:
        notes.append(f"batch_size {batch_size} not divisible by world {world}")
        return DDPPlan(1, [home], batch_size, notes)

    home_idx = home.index if home.index is not None else 0
    if explicit_devices:
        order = [torch.device(d) for d in explicit_devices]
    elif visible_cuda_devices:
        order = [torch.device("cuda", i) for i in sorted(visible_cuda_devices)]
    else:
        order = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

    rotated = [home] + [d for d in order if d != home]
    devices: List[torch.device] = []
    for dev in rotated[:world]:
        if vram_free_bytes is not None and mirror_footprint_bytes is not None:
            free = vram_free_bytes.get(dev, 0) if hasattr(vram_free_bytes, "get") else vram_free_bytes
            if dev != home and free < mirror_footprint_bytes + margin_bytes:
                notes.append(f"skip {dev}: free {free / 2**30:.1f}GiB < footprint+margin")
                continue
        devices.append(dev)

    if len(devices) < 2:
        notes.append("no mirror device passed the VRAM guard")
        return DDPPlan(1, [home], batch_size, notes)
    if len(devices) < world:
        notes.append(f"world reduced {world} -> {len(devices)} by VRAM guard")
    return DDPPlan(len(devices), devices, batch_size // len(devices), notes)


def _move_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to ``device`` (seam for tests on single-device hosts)."""
    return t.to(device)


def _relocate_params(module: torch.nn.Module, device: torch.device) -> None:
    """Move ALL state that ``nn.Module.to()`` cannot see onto ``device``.

    Wrapper modules keep three kinds of non-registered state that a mirrored
    block would otherwise leave on the home device:
    - tunable tensors in the plain ``params`` dict (``v``/``min_scale``/...)
      -- recreated as fresh leaf Parameters (grad state resets, correct for a
      fresh mirror);
    - plain tensor attributes (``weight_min``/``weight_max`` anchors, cached
      imatrix slices, ...) -- moved in place;
    - device-typed attributes (``device``/``output_device``/``tuning_device``)
      -- repointed so wrapper forward staging targets the mirror, not home.
    """
    for _n, m in module.named_modules():
        params = getattr(m, "params", None)
        if isinstance(params, dict):
            for key, val in params.items():
                if isinstance(val, torch.nn.Parameter):
                    moved = _move_tensor(val.detach(), device)
                    params[key] = torch.nn.Parameter(moved, requires_grad=val.requires_grad)
        for key, val in list(m.__dict__.items()):
            if isinstance(val, torch.Tensor) and val.device != device and val.device.type != "meta":
                m.__dict__[key] = _move_tensor(val, device)
            elif isinstance(val, torch.device):
                m.__dict__[key] = device


def _param_grad_buffers(params_by_device: List[List[torch.nn.Parameter]]) -> List[Optional[torch.Tensor]]:
    """Flatten each replica's gradients into one contiguous fp32 buffer."""
    bufs: List[Optional[torch.Tensor]] = []
    for params in params_by_device:
        parts = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if not parts:
            bufs.append(None)
            continue
        buf = torch.cat(parts) if len(parts) > 1 else parts[0].clone()
        bufs.append(buf.to(torch.float32) if buf.dtype != torch.float32 else buf)
    return bufs


def _write_back_grads(buf: Optional[torch.Tensor], params: List[torch.nn.Parameter]) -> None:
    """Scatter an averaged flat buffer back into ``param.grad`` tensors."""
    if buf is None:
        return
    offset = 0
    for p in params:
        if p.grad is None:
            continue
        n = p.grad.numel()
        p.grad.copy_(buf[offset : offset + n].view_as(p.grad))
        offset += n


def halving_doubling_allreduce(buffers: List[torch.Tensor], scale: float = 1.0, bf16: bool = False) -> None:
    """In-place all-reduce across device-resident buffers (single process).

    Chunked recursive halving-doubling: the flat space is split into W
    chunks; the reduce-scatter phase leaves rank r owning reduced chunk r
    (each step exchanges half of the working block with the partner rank);
    the all-gather phase re-exchanges owned chunks until every rank holds
    the fully reduced buffer. Per-rank traffic is 2*(W-1)/W*bytes, same as a
    ring. Cross-device ``to()`` copies use P2P when available. ``scale`` is
    applied at the end (pass ``1/world`` to average). ``bf16`` casts the
    exchanged chunks to bfloat16 (halves the PCIe payload; accumulation
    stays fp32). Requires a power-of-two world (the resolver guarantees it).
    """
    world = len(buffers)
    if world < 2:
        if buffers:
            buffers[0].mul_(scale)
        return
    if world & (world - 1):
        raise ValueError(f"halving_doubling_allreduce needs a power-of-two world, got {world}")

    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world

    # ── reduce-scatter: working block per rank halves each step; each rank
    # keeps one half (adding the partner's copy of it) and discards the other
    # (whose values are added into the partner's kept half). Rank r ends up
    # owning reduced chunk r.
    length = world  # chunks in each rank's working block
    while length > 1:
        half = length // 2
        for rank in range(world):
            base = rank - (rank % length)  # aligned working-block start
            mid, hi = base + half, base + length
            if rank % length < half:
                partner = rank + half  # keep [base, mid)
                seg = buffers[partner][chunk * base : chunk * mid]
                seg = (
                    seg.to(torch.bfloat16).to(buffers[rank].device, buffers[rank].dtype)
                    if bf16
                    else seg.to(buffers[rank].device)
                )
                buffers[rank][chunk * base : chunk * mid].add_(seg)
            else:
                partner = rank - half  # keep [mid, hi)
                seg = buffers[partner][chunk * mid : chunk * hi]
                seg = (
                    seg.to(torch.bfloat16).to(buffers[rank].device, buffers[rank].dtype)
                    if bf16
                    else seg.to(buffers[rank].device)
                )
                buffers[rank][chunk * mid : chunk * hi].add_(seg)
        length = half

    # ── all-gather: working block per rank doubles each step; ranks exchange
    # the chunks the partner owns (copies, no adds)
    length = 2
    while length <= world:
        half = length // 2
        for rank in range(world):
            base = rank - (rank % length)
            mid, hi = base + half, base + length
            if rank % length < half:
                partner = rank + half
                src = buffers[partner][chunk * mid : chunk * hi]
                dst = buffers[rank][chunk * mid : chunk * hi]
                dst.copy_(src.to(torch.bfloat16).to(dst.device, dst.dtype) if bf16 else src.to(dst.device))
            else:
                partner = rank - half
                src = buffers[partner][chunk * base : chunk * mid]
                dst = buffers[rank][chunk * base : chunk * mid]
                dst.copy_(src.to(torch.bfloat16).to(dst.device, dst.dtype) if bf16 else src.to(dst.device))
        length *= 2

    for buf in buffers:
        buf.mul_(scale)


def sharded_nograd_forward(
    runner,
    block,
    inputs,
    input_others: dict,
    out_device: torch.device,
    devices: List[torch.device],
    sample_count: Optional[int] = None,
):
    """Parallelize a no-grad collection forward across ``devices``.

    The collection passes (reference outputs, quantized-output cascade) are
    plain forwards over the whole sample pool on one GPU while the mirrors
    idle. Here the pool is split into equal disjoint shards; an ephemeral
    copy of the block on each device forwards its shard in a parallel thread;
    per-shard outputs are parked on ``out_device`` and concatenated in order.
    Bit-identical to the serial pass: rows are sample-independent and the
    module copies carry identical weights. Falls back to the serial runner
    call when the pool is not divisible or fewer than two devices are given.
    Mirrors are dropped (freed) afterwards.
    """
    world = len(devices)
    n = sample_count if sample_count is not None else (len(inputs) if isinstance(inputs, list) else 0)
    if world < 2 or n < world or n % world != 0:
        return runner(block, inputs, input_others, cache_device=out_device)
    shard = n // world
    shards = [list(range(r * shard, (r + 1) * shard)) for r in range(world)]
    home_dev = _block_device(block)
    mirrors: List[torch.nn.Module] = []
    reps: List[torch.nn.Module] = []
    for dev in devices:
        if dev == home_dev:
            reps.append(block)
            mirrors.append(None)
        else:
            m = copy.deepcopy(block).to(dev)
            _relocate_params(m, dev)
            reps.append(m)
            mirrors.append(m)
    parts: List = [None] * world

    def _run(r):
        rep = reps[r]
        dev_r = _block_device(rep)
        if dev_r.type == "cuda":
            with torch.cuda.device(dev_r):
                parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=out_device)
        else:
            parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=out_device)

    runner_group = ReplicaGroup.__new__(ReplicaGroup)  # reuse only the thread runner
    runner_group.run_threaded([lambda r=r: _run(r) for r in range(world)])
    out = parts[0]
    for part in parts[1:]:
        out = torch.cat([out, part], dim=0)
    mirrors.clear()  # drop mirror refs; the caching allocator reclaims them
    reps.clear()
    return out


class ReplicaGroup:
    """Persistent mirrors of a wrapped block for the iteration loop."""

    def __init__(self, block, plan: DDPPlan, bf16_grad: bool = False, staged_source=None) -> None:
        self.plan = plan
        self.bf16_grad = bf16_grad
        self.home = block
        self.mirrors: List[torch.nn.Module] = []
        self.adopted = 0
        for dev in plan.devices[1:]:  # plan.devices[0] is the home by construction
            mirror = self._make_mirror(block, dev, staged_source)
            _relocate_params(mirror, dev)
            self.mirrors.append(mirror)
        self.replicas = [block] + self.mirrors
        self.world = len(self.replicas)
        logger.info(
            "[tune-ddp] world=%d devices=%s adopted=%d/%d",
            self.world,
            [str(d) for d in plan.devices],
            self.adopted,
            max(0, self.world - 1),
        )

    def _make_mirror(self, block, dev, staged_source):
        """Mirror the block onto ``dev``, adopting prefetched weights when sane.

        ``staged_source`` = (streamer, prefix): when the prefetch reader
        fanned this block out onto ``dev`` AND the home weights are pristine
        (no pre-quantize transform mutated them), the checkpoint-backed
        orig_layer weights are swapped to meta before the device move and
        re-materialized from the local staged copies -- the cross-GPU copy
        then carries only the small tuning state instead of the block
        weights. Falls back to the plain deepcopy path otherwise.
        """
        staged = None
        if staged_source is not None:
            streamer, prefix = staged_source
            try:
                staged = streamer.staged_replica_tensors(prefix, dev)
            except Exception:
                staged = None
        if not staged or not getattr(block, "_stream_weights_pristine", False):
            return copy.deepcopy(block).to(dev)

        mirror = copy.deepcopy(block)
        swapped = []
        for name, param in mirror.named_parameters():
            if ".orig_layer." in name:
                key = (
                    (prefix + "." + name.replace(".orig_layer.", ".")) if prefix else name.replace(".orig_layer.", ".")
                )
                if key in staged:
                    param.data = torch.empty(0, dtype=param.dtype, device="cpu")
                    swapped.append((name, key))
        mirror = mirror.to(dev)  # cheap: big weights ride as empty meta tensors
        params = dict(mirror.named_parameters())
        for name, key in swapped:
            params[name].data = staged[key].to(params[name].device)
        self.adopted += 1
        return mirror

    def round_params(self) -> List[List[torch.nn.Parameter]]:
        out = []
        for rep in self.replicas:
            ps = []
            for _n, m in rep.named_modules():
                if hasattr(m, "orig_layer") and "v" in getattr(m, "params", {}):
                    ps.append(m.params["v"])
            out.append(ps)
        return out

    def sync_grads(self, params_per_replica: List[List[torch.nn.Parameter]]) -> None:
        """All-reduce v-gradients so every replica holds the identical average."""
        bufs = _param_grad_buffers(params_per_replica)
        if any(b is None for b in bufs):
            return
        halving_doubling_allreduce(bufs, scale=1.0 / self.world, bf16=self.bf16_grad)
        for buf, params in zip(bufs, params_per_replica):
            _write_back_grads(buf, params)

    def broadcast_module_attrs(self, attr_names: Tuple[str, ...]) -> None:
        """Copy small anchor attrs (e.g. weight_min/max after a re-grid) to mirrors."""
        home_mods = [m for _n, m in self.home.named_modules() if hasattr(m, "orig_layer")]
        for mirror in self.mirrors:
            mir_mods = [m for _n, m in mirror.named_modules() if hasattr(m, "orig_layer")]
            for hm, mm in zip(home_mods, mir_mods):
                target_dev = next(mm.parameters()).device
                for attr in attr_names:
                    if hasattr(hm, attr):
                        val = getattr(hm, attr)
                        val = val.to(target_dev) if torch.is_tensor(val) else copy.deepcopy(val)
                        setattr(mm, attr, val)

    def run_threaded(self, fns: Sequence) -> None:
        """Run one callable per replica in parallel threads.

        Exceptions propagate: the first failure (by thread index) is re-raised
        in the joining thread -- a swallowed worker failure would otherwise
        leave missing shard losses/grads and corrupt the step silently.
        """
        errors: dict = {}

        def _guarded(idx, fn):
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[idx] = exc

        threads = [threading.Thread(target=_guarded, args=(i, fn)) for i, fn in enumerate(fns)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[min(errors)]

    def teardown(self) -> None:
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
