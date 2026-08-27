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


def resolve_ddp_plan(
    world: int,
    home: torch.device,
    batch_size: int,
    visible_cuda_devices: Optional[Sequence[int]] = None,
    explicit_devices: Optional[Sequence] = None,
    vram_free_bytes: Optional[int] = None,
    mirror_footprint_bytes: Optional[int] = None,
    margin_bytes: int = 2 * 1024**3,
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


def _relocate_params(module: torch.nn.Module, device: torch.device) -> None:
    """Move dict-registered tuning params (``m.params``) to ``device``.

    Wrapper modules keep their tunable tensors in a plain ``params`` dict --
    ``nn.Module.to()`` does not see them, so a mirrored block would compute
    with home-device ``v``/``min_scale``/``max_scale``. Recreate each entry as
    a fresh leaf Parameter on the target device (grad state resets, which is
    correct for a fresh mirror).
    """
    for _n, m in module.named_modules():
        params = getattr(m, "params", None)
        if not isinstance(params, dict):
            continue
        for key, val in params.items():
            if isinstance(val, torch.nn.Parameter):
                params[key] = torch.nn.Parameter(val.detach().to(device), requires_grad=val.requires_grad)


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


class ReplicaGroup:
    """Persistent mirrors of a wrapped block for the iteration loop."""

    def __init__(self, block, plan: DDPPlan, bf16_grad: bool = False) -> None:
        self.plan = plan
        self.bf16_grad = bf16_grad
        self.home = block
        self.mirrors: List[torch.nn.Module] = []
        for dev in plan.devices[1:]:  # plan.devices[0] is the home by construction
            mirror = copy.deepcopy(block).to(dev)
            _relocate_params(mirror, dev)
            self.mirrors.append(mirror)
        self.replicas = [block] + self.mirrors
        self.world = len(self.replicas)
        logger.info("[tune-ddp] world=%d devices=%s", self.world, [str(d) for d in plan.devices])

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
        """Run one callable per replica in parallel threads."""
        threads = [threading.Thread(target=fn) for fn in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def teardown(self) -> None:
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
