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

import contextlib
import copy
import queue
import threading
import time
from collections import OrderedDict
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


_effective_groups_registry: Optional[List[List[torch.device]]] = None


def set_effective_ddp_groups(groups) -> None:
    """Register DDP device groups derived OUTSIDE the env (auto ping-pong).

    The streaming orchestrator can split the staging pool into consecutive
    world-sized groups when AR_TUNE_DDP_GROUPS is unset; the quantizer, the
    composer and the prefetch fan-out must all see the SAME groups, so they
    resolve through effective_ddp_groups() (explicit env wins).
    """
    global _effective_groups_registry
    _effective_groups_registry = [list(g) for g in groups] if groups else None


def effective_ddp_groups():
    """Explicit env groups, else the registered auto groups, else None."""
    from auto_round import envs as _envs

    env_groups = parse_ddp_groups(getattr(_envs, "AR_TUNE_DDP_GROUPS", None))
    if env_groups:
        return env_groups
    return _effective_groups_registry


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


def enable_peer_access(devices) -> list:
    """Explicitly enable CUDA peer access between every device pair.

    A patched driver can expose full-mesh P2P, but the caching allocator
    does not necessarily call cudaDeviceEnablePeerAccess for each ordered
    pair -- and without it every cross-device tensor copy silently stages
    through host memory (~3-4 GB/s instead of the measured 13-26 GB/s),
    which showed up as a multi-x allreduce slowdown. Returns the list of
    enabled "i<->j" pairs for logging.
    """
    import ctypes

    devs = sorted({int(getattr(d, "index", d) if getattr(d, "index", None) is not None else d) for d in devices})
    enabled = []
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        try:
            cudart = ctypes.CDLL("libcudart.so.2")
        except OSError:
            return []
    # the raw cudart calls below bypass torch's device-guard bookkeeping: a
    # leaked cudaSetDevice would silently move the thread's current device
    # and split later CUDA-event pairs across devices (profiler crashes).
    # Hold one torch device context around the whole loop so it is restored.
    saved = torch.cuda.current_device()
    try:
        for i in devs:
            for j in devs:
                if i == j:
                    continue
                try:
                    can = ctypes.c_int(0)
                    if cudart.cudaDeviceCanAccessPeer(ctypes.byref(can), i, j) != 0 or not can.value:
                        continue
                    cudart.cudaSetDevice(i)
                    # cudaErrorPeerAccessAlreadySet (716) is fine
                    cudart.cudaDeviceEnablePeerAccess(j, 0)
                    enabled.append(f"{i}->{j}")
                except Exception:  # noqa: BLE001 - best effort only
                    continue
    finally:
        torch.cuda.set_device(saved)
    return enabled


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


class StagedSourceRef:
    """Deepcopy-safe (streamer, prefix) holder stamped on streamed blocks.

    ``block._stream_prefetch_source`` must survive ``copy.deepcopy(block)``
    (mirror creation, sharded collection copies) without walking into the
    CheckpointStreamer -- its prefetch reader owns ``safe_open`` handles that
    cannot be pickled. All copies therefore SHARE this reference (read-only).
    """

    def __init__(self, streamer, prefix: str) -> None:
        self.streamer = streamer
        self.prefix = prefix

    def unpack(self):
        return self.streamer, self.prefix

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):  # pragma: no cover - defensive
        return (StagedSourceRef, (None, self.prefix))


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
                    # move IN PLACE: the dict entry and the registered
                    # _parameters entry are the SAME object by construction
                    # (wrapper _init_params). Replacing the object here broke
                    # that aliasing -- optimizers collected the dict's dead
                    # clone while the forward/backward used the registered
                    # Parameter, so mirror grads stayed None, sync_grads
                    # silently skipped the allreduce, and the home stepped on
                    # its own shard's gradient only (quality regressed,
                    # monotonically in world size).
                    val.data = _move_tensor(val.detach(), device)
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


def _xchg(seg: torch.Tensor, dev: torch.device, dtype: torch.dtype, transport: str) -> torch.Tensor:
    """Move a peer's segment onto ``dev`` in the requested transport dtype.

    Transport ORDER matters: a combined ``.to(device, dtype)`` cross-device
    copy casts on the SOURCE first and then memcpys -- with an fp32
    destination the wire carried full fp32 bytes and the bf16 transport was
    a no-op on payload (measured: bf16 allreduce time == fp32 on a
    half-duplex-per-link fabric). Cast down on the source, move the reduced
    bytes, cast back up on the receiver instead.

    int8 uses symmetric per-segment scaling: one amax per exchanged segment
    rides along as an fp32 scalar. The step size stays relative to the
    segment max, and the averaged-gradient signs SignSGD consumes are only
    perturbed inside a band far below typical |grad| magnitudes. All int8
    arithmetic stays fp32: the first implementation routed the quantize /
    dequantize through fp64, whose temporaries carry 2x fp32 traffic and
    made the exchange SLOWER than plain fp32 wire (measured 360-390 ms vs
    ~243 fp32 per tune iteration at world=4), on top of ~2.5 GB of extra
    peak VRAM.

    The wire hop itself uses non_blocking=True: the measured allreduce is
    payload-independent across fp32/bf16 (~243/~231 ms), i.e. dominated by
    the per-exchange HOST blocking of a synchronous copy rather than wire
    bytes. Cross-device copy_ is stream-ordered on both endpoints (it
    records an event on the source stream and makes the destination stream
    wait), and every producer/consumer of a region runs on that device's
    current stream in issue order, so dropping the host block lets the
    exchange chain execute back-to-back on the GPUs without races.
    """
    if transport == "fp32":
        return seg.to(dev, non_blocking=True)
    if transport == "bf16":
        return seg.to(torch.bfloat16).to(dev, non_blocking=True).to(dtype)
    if transport == "int8":
        with torch.no_grad():
            src = seg.detach()
            amax = src.abs().amax()
            inv = 127.0 / amax.clamp_min(torch.finfo(src.dtype).tiny)
            q = torch.round(src * inv).clamp_(-127.0, 127.0).to(torch.int8)
            q = q.to(dev, non_blocking=True)
            scale = amax.to(dev, non_blocking=True) / 127.0
            return q.to(dtype).mul_(scale)
    raise ValueError(f"unknown gradient transport {transport!r} (fp32|bf16|int8)")


_ONESHOT_LOGGED = False


def _encode_transport(t: torch.Tensor, transport: str):
    """Encode a gradient tensor for wire transport.

    Returns ``(payload, meta)``: the wire tensor (int8 / bfloat16 / fp32) and
    the int8 per-bucket amax scalar (``None`` for fp32 / bf16). fp32 returns
    the tensor itself (read-only alias -- callers must not mutate it).
    """
    if transport == "int8":
        amax = t.abs().amax()
        inv = 127.0 / amax.clamp_min(torch.finfo(t.dtype).tiny)
        return torch.round(t * inv).clamp_(-127.0, 127.0).to(torch.int8), amax
    if transport == "bf16":
        return t.to(torch.bfloat16), None
    return t, None


def use_one_shot(world: int, transport: str) -> bool:
    """Pick the allreduce algorithm (AR_TUNE_DDP_ALLREDUCE=auto|oneshot|halving).

    One-shot trades bandwidth for latency: 2*(W-1) payload per rank instead
    of halving-doubling's bandwidth-optimal 2*(W-1)/W, but a single
    concurrency wave instead of 2*log2(W) dependent steps. Reduced transports
    (bf16/int8) shrink the payload enough that latency dominates everywhere
    up to world=8; fp32 one-shot only stays comfortable through world=4 --
    at world=8 its (W-1)=7 fp32 payloads plus a dequant temp park ~2.85 GB
    per device, so auto keeps halving-doubling there. Explicit values force
    the algorithm regardless of world/transport.
    """
    from auto_round import envs

    mode = envs.AR_TUNE_DDP_ALLREDUCE
    if mode == "oneshot":
        return True
    if mode == "halving":
        return False
    # auto keeps halving-doubling for now: one-shot MEASURED slower on the
    # reference fabric (350 ms vs 195 ms per tune iteration at world=4 int8,
    # against ~22 ms of actual wire) -- the per-rank wire argument did not
    # survive contact with the machine. Flip to one-shot explicitly to help
    # profile the wave structure (encode/copy/reduce sub-stages are logged).
    return False


def _canonical_bucket_sum(payloads, metas, transport: str, dtype: torch.dtype) -> torch.Tensor:
    """Sum transport-encoded bucket payloads in canonical (source-index) order.

    Every contribution -- including the local one -- enters as the SAME
    dequantized payload every other rank sums, so all ranks end up bitwise
    identical (the same guarantee halving-doubling's exchange tree gives).
    """
    acc = None
    for payload, meta in zip(payloads, metas):
        t = payload.to(dtype)
        if transport == "int8" and meta is not None:
            t = t.mul_(meta.to(dtype) / 127.0)
        if acc is None:
            # fp32 transport aliases the payload (to() is a no-op) -- clone so
            # the in-place adds never mutate the caller's staging slot
            acc = t.clone() if t is payload else t
        else:
            acc.add_(t)
    return acc


def one_shot_allreduce(buffers: List[torch.Tensor], scale: float = 1.0, transport: str = "fp32", prof=None) -> None:
    """In-place all-reduce via a single full-broadcast wave, then local sums.

    Halving-doubling pays 2*log2(W) DEPENDENT exchange steps; each step's
    copy cannot start before the previous step's result exists, so at small
    payloads the chain latency dominates the wire bytes (measured: int8
    world=4 allreduce ~195 ms vs ~30 ms of wire). One-shot instead ships
    every rank's transport-reduced buffer to every peer in ONE concurrent
    wave (W*(W-1) copies) and dequant-adds locally afterwards.

    Staging is (W-1) transport-dtype payloads plus one fp32 temp per DEVICE
    (int8 world=8: ~0.7 GB; bf16 world=8: ~1.6 GB; fp32 world=4: ~1.4 GB).
    See ``use_one_shot`` for the auto policy; AR_TUNE_DDP_ALLREDUCE forces
    either algorithm.
    """
    world = len(buffers)
    if world < 2:
        if buffers:
            buffers[0].mul_(scale)
        return
    if transport not in ("fp32", "bf16", "int8"):
        raise ValueError(f"unknown gradient transport {transport!r} (fp32|bf16|int8)")
    from auto_round.utils.tune_profile import stage as _stage

    with torch.no_grad():
        dt = buffers[0].dtype
        devs = [b.device for b in buffers]
        # wave 0: transport-encode each source ONCE (its own stream); the
        # int8 amax scalars must be captured HERE -- buffers are mutated by
        # the scale below, so recomputing amax later would be wrong.
        with _stage(prof, "os_encode"):
            payloads, scales = [], []
            for src in buffers:
                if transport == "int8":
                    amax = src.abs().amax()
                    inv = 127.0 / amax.clamp_min(torch.finfo(src.dtype).tiny)
                    payloads.append(torch.round(src * inv).clamp_(-127.0, 127.0).to(torch.int8))
                    scales.append(amax)
                elif transport == "bf16":
                    payloads.append(src.to(torch.bfloat16))
                    scales.append(None)
                else:
                    # clone: buffers are mutated in wave 2, and a same-device
                    # .to() later is a no-op that would keep the alias alive
                    payloads.append(src.clone())
                    scales.append(None)
        # wave 1: move every payload (and its int8 scalar) to every peer;
        # the copies overlap on the fabric -- int8 payloads are 1/4 fp32 so
        # W*(W-1) of them fit easily
        with _stage(prof, "os_copy"):
            recv = [[None] * world for _ in range(world)]
            scale_d = [[None] * world for _ in range(world)]
            for r in range(world):
                for d in range(world):
                    if d != r:
                        recv[d][r] = payloads[r].to(devs[d], non_blocking=True)
                        if scales[r] is not None:
                            scale_d[d][r] = scales[r].to(devs[d], non_blocking=True)
        # (wave boundaries are enqueue boundaries; the GPU executes across
        # them asynchronously -- the sub-stages below bracket where each
        # wave's work was ISSUED, and their stage events drain on the home
        # stream, so the numbers show critical-path contribution)
        # wave 2: canonical reduction. Every rank sums the SAME dequantized
        # payloads in the SAME (source-index) order -- including its own,
        # round-tripped through the transport like everyone else's -- so all
        # ranks end up bitwise identical, and the accumulation tree matches
        # across ranks exactly like halving-doubling's exchange tree does.
        # One reusable fp32 temp per device; acc is rebuilt from zero.
        with _stage(prof, "os_reduce"):
            for d in range(world):
                acc = buffers[d]
                acc.zero_()
                for r in range(world):
                    t = (payloads[r] if r == d else recv[d][r]).to(dt)
                    s_r = scales[r] if r == d else scale_d[d][r]
                    if s_r is not None:
                        t.mul_(s_r / 127.0)
                    acc.add_(t)
                acc.mul_(scale)


def halving_doubling_allreduce(buffers: List[torch.Tensor], scale: float = 1.0, transport: str = "fp32") -> None:
    """In-place all-reduce across device-resident buffers (single process).

    Chunked recursive halving-doubling: the flat space is split into W
    chunks; the reduce-scatter phase leaves rank r owning reduced chunk r
    (each step exchanges half of the working block with the partner rank);
    the all-gather phase re-exchanges owned chunks until every rank holds
    the fully reduced buffer. Per-rank traffic is 2*(W-1)/W*bytes, same as a
    ring. Cross-device ``to()`` copies use P2P when available. ``scale`` is
    applied at the end (pass ``1/world`` to average). ``transport`` selects
    the exchange dtype: fp32 (exact), bf16 (half wire bytes) or int8
    (quarter wire bytes, symmetric per-segment amax scaling); accumulation
    stays fp32. Requires a power-of-two world (the resolver guarantees it).
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

    _reduce_scatter_halving(buffers, transport)
    _allgather_doubling(buffers, transport)

    for buf in buffers:
        buf.mul_(scale)


def _reduce_scatter_halving(buffers: List[torch.Tensor], transport: str) -> None:
    """Recursive-halving reduce-scatter across device-resident buffers.

    The flat space is split into W chunks; the working block per rank halves
    each step (each rank keeps one half, adding the partner's transport-
    rounded copy of it). Rank r ends up owning reduced chunk r; the other
    chunks hold partial garbage. Accumulation stays fp32 on the receiver.
    Requires a power-of-two world.
    """
    world = len(buffers)
    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    length = world  # chunks in each rank's working block
    while length > 1:
        half = length // 2
        for rank in range(world):
            base = rank - (rank % length)  # aligned working-block start
            mid, hi = base + half, base + length
            if rank % length < half:
                partner = rank + half  # keep [base, mid)
                seg = buffers[partner][chunk * base : chunk * mid]
                seg = _xchg(seg, buffers[rank].device, buffers[rank].dtype, transport)
                buffers[rank][chunk * base : chunk * mid].add_(seg)
            else:
                partner = rank - half  # keep [mid, hi)
                seg = buffers[partner][chunk * mid : chunk * hi]
                seg = _xchg(seg, buffers[rank].device, buffers[rank].dtype, transport)
                buffers[rank][chunk * mid : chunk * hi].add_(seg)
        length = half


def _allgather_doubling(buffers: List[torch.Tensor], transport: str) -> None:
    """Recursive-doubling all-gather of per-rank owned chunks.

    Mirrors the reduce-scatter block structure: the working block per rank
    doubles each step; ranks exchange the chunks the partner owns (copies,
    no adds). Every rank ends holding every chunk -- bitwise identical when
    the transport is lossless for the buffer dtype (fp32 buffers with fp32
    transport, int8 buffers with fp32 transport).
    """
    world = len(buffers)
    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
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
                dst.copy_(_xchg(src, dst.device, dst.dtype, transport))
            else:
                partner = rank - half
                src = buffers[partner][chunk * base : chunk * mid]
                dst = buffers[rank][chunk * base : chunk * mid]
                dst.copy_(_xchg(src, dst.device, dst.dtype, transport))
        length *= 2


_SIGN_LOGGED = False


def _debug_flat_leak_probe(tag: str) -> None:
    """AR_DEBUG_FLAT_LEAK=1: attribute per-block cuda:0 accumulation.

    Scans ALL gc-tracked cuda:0 tensors >= 64 MB (any dtype): prints the
    per-(dtype, numel) histogram (untruncated survivor count), the
    allocator's allocated/reserved bytes, and referrers for the largest
    survivors -- so a next-block "-start" probe reveals whether the
    PREVIOUS block's tensors really died.
    """
    import gc
    from collections import Counter

    from auto_round import envs

    if not getattr(envs, "AR_DEBUG_FLAT_LEAK", False):
        return
    gc.collect()
    hits = []
    histogram: Counter = Counter()
    bytes_by_key: Counter = Counter()
    for obj in gc.get_objects():
        try:
            if (
                torch.is_tensor(obj)
                and obj.device.type == "cuda"
                and obj.device.index == 0
                and obj.untyped_storage().nbytes() >= 64 * 2**20
            ):
                hits.append(obj)
                # int() coerces SymInt/SymFloat (meta/fake tensors) to plain
                # numbers; symbolic ones raise and are skipped by the guard
                key = (str(obj.dtype).replace("torch.", ""), int(obj.numel()))
                histogram[key] += 1
                bytes_by_key[key] += int(obj.untyped_storage().nbytes())
        except Exception:  # noqa: BLE001 - probe must never kill a run
            continue
    try:
        alloc = torch.cuda.memory_allocated(0) / 2**30
        reserved = torch.cuda.memory_reserved(0) / 2**30
    except Exception:  # noqa: BLE001
        alloc = reserved = -1.0
    hist = " ".join(f"{k[0]}[{k[1]}]x{v}({bytes_by_key[k] / 2**30:.2f}GB)" for k, v in sorted(histogram.items()))
    logger.info(
        "[leak-probe/%s] cuda:0 tensors>=64MB: %d total, alloc=%.2fGB reserved=%.2fGB | %s",
        tag,
        len(hits),
        alloc,
        reserved,
        hist,
    )
    hits.sort(key=lambda t: -int(t.untyped_storage().nbytes()))
    for t in hits[:6]:
        try:
            ptr = t.data_ptr()
        except Exception:  # noqa: BLE001 - FakeTensor/FunctionalTensor
            continue
        names = []
        for r in gc.get_referrers(t)[:8]:
            kind = type(r).__name__
            if isinstance(r, dict):
                kind += "[" + ",".join(str(k) for k in list(r.keys())[:3]) + "]"
            names.append(kind)
            # a module __dict__ holder: name the module and WHAT HOLDS IT
            if isinstance(r, dict) and {"training", "_parameters", "_buffers"} <= set(r.keys()):
                for mod in gc.get_referrers(r):
                    if isinstance(mod, torch.nn.Module):
                        held_by = []
                        for h in gc.get_referrers(mod)[:6]:
                            hk = type(h).__name__
                            if isinstance(h, (list, tuple)):
                                hk += f"[len{len(h)}]"
                            elif isinstance(h, dict):
                                hk += "[" + ",".join(str(k) for k in list(h.keys())[:3]) + "]"
                            held_by.append(hk)
                        names.append(f"MODULE<{type(mod).__name__}#{id(mod) % 100000}> held-by: {held_by}")
                        if type(mod).__name__ == "ReplicaGroup":
                            gh = []
                            for h in gc.get_referrers(mod)[:6]:
                                hk = type(h).__name__
                                if isinstance(h, (list, tuple)):
                                    hk += f"[len{len(h)}]"
                                elif isinstance(h, dict):
                                    hk += "[" + ",".join(str(k) for k in list(h.keys())[:3]) + "]"
                                gh.append(hk)
                            names.append(f"GROUP-HELD-BY: {gh}")
        logger.info(
            "[leak-probe/%s]   ptr=%d %s numel=%d (%.2f GB) referrers: %s",
            tag,
            ptr,
            t.dtype,
            int(t.numel()),
            int(t.untyped_storage().nbytes()) / 2**30,
            names,
        )


def sign_exchange_allreduce(buffers: List[torch.Tensor], transport: str = "fp32") -> None:
    """In-place sign all-reduce for SignSGD gradients (single process).

    The SignRound optimizer consumes ONLY torch.sign(grad) (weight_decay is
    always 0), so the averaged gradient's magnitude never reaches the
    update. This exchanges exactly what the step needs: a recursive-halving
    reduce-scatter (identical transport rounding of the partials as
    halving-doubling) leaves rank r owning the reduced chunk r; each rank
    then computes torch.sign() ONCE on its exact fp32 chunk and the signs
    are all-gathered as int8 -- a lossless wire format 4x smaller than
    fp32. Compared to a full halving-doubling allreduce this REMOVES the
    all-gather's transport rounding (bf16 rounding can zero out tiny
    averaged gradients, losing their sign), so the signs every rank applies
    are bitwise identical and at least as faithful to the true fp32 mean.

    Valid only when the optimizer is pure sign-SGD: a momentum buffer or
    weight decay would mix magnitudes back into the update, so callers must
    gate on momentum == 0 (weight_decay is hard-wired to 0 in the tuner).
    """
    world = len(buffers)
    if world < 2:
        for buf in buffers:
            buf.sign_()
        return
    if world & (world - 1):
        raise ValueError(f"sign_exchange_allreduce needs a power-of-two world, got {world}")

    _reduce_scatter_halving(buffers, transport)

    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    sign_bufs: List[torch.Tensor] = []
    for rank, buf in enumerate(buffers):
        signs = torch.empty(numel, dtype=torch.int8, device=buf.device)
        lo, hi = chunk * rank, min(numel, chunk * (rank + 1))
        signs[lo:hi] = torch.sign(buf[lo:hi]).to(torch.int8)
        sign_bufs.append(signs)

    # fp32 transport on int8 buffers = plain device copies, lossless
    _allgather_doubling(sign_bufs, "fp32")

    for buf, signs in zip(buffers, sign_bufs):
        buf.copy_(signs)  # int8 -> fp32: -1.0 / 0.0 / 1.0


# hook-written statistics that are safe to merge across mirror copies:
# associative reductions reproduce the serial totals up to fp32 summation order
_MERGEABLE_STATS = {
    "imatrix": "sum",  # module.imatrix += sum(x^2) per shard
    "act_max": "max",  # element-wise running max
}


def _merge_mirror_stats(home: torch.nn.Module, mirrors: List[torch.nn.Module]) -> None:
    """Fold hook-written statistics from mirror copies back into the home.

    Each mirror forwarded a disjoint sample shard; its hooks accumulated
    into mirror-local module attrs which die with the mirror. The stats we
    know how to merge (see ``_MERGEABLE_STATS``) are associative, so the
    merged home totals equal the serial pass's totals up to fp32 summation
    order (~1e-7 relative) -- far below any quantization-relevant scale.
    """
    home_mods = dict(home.named_modules())
    for mirror in mirrors:
        if mirror is None:
            continue
        for name, m_mod in mirror.named_modules():
            h_mod = home_mods.get(name)
            if h_mod is None:
                continue
            for attr, how in _MERGEABLE_STATS.items():
                m_val = getattr(m_mod, attr, None)
                if not torch.is_tensor(m_val):
                    continue
                h_val = getattr(h_mod, attr, None)
                if h_val is None:
                    setattr(h_mod, attr, m_val.to(h_mod.weight.device if hasattr(h_mod, "weight") else m_val.device))
                    continue
                m_val = m_val.to(h_val.device, h_val.dtype)
                if how == "sum":
                    setattr(h_mod, attr, h_val + m_val)
                else:  # max
                    setattr(h_mod, attr, torch.max(h_val, m_val))


_pool_move_warned: set = set()
_coll_mirror_setup_logged: set = set()


def expect_pool_local(pieces, device, site: str) -> None:
    """Warn (once per site) when pool pieces are NOT on ``device``.

    The distributed-pool contract says DDP shard reads are device-local; the
    defensive ``.to(device)`` at those sites would silently paper over a
    placement bug (or a demoted block paying per-batch copies). This makes
    any actual cross-device engagement visible: count, devices found, site.
    """
    device = torch.device(device)
    stray = [t for t in pieces if t.device != device]
    if stray and site not in _pool_move_warned:
        _pool_move_warned.add(site)
        found = sorted({str(t.device) for t in stray})
        logger.warning(
            "[tune-ddp] %s: %d/%d pool pieces are not on %s (found on %s) -- cross-device copies engaged; "
            "expected zero under the distributed pool",
            site,
            len(stray),
            len(pieces),
            device,
            found,
        )


def distribute_pool(pool: List[torch.Tensor], devices: List[torch.device]) -> None:
    """Scatter a per-sample calibration pool across ``devices`` (in place).

    Device r owns samples [r*shard, (r+1)*shard) -- the same boundaries the
    DDP tune shards and the sharded collection use -- so shard-local reads
    (ref_r build, _select_batch for shard r, per-replica collections) never
    cross devices. Pieces already on their target device are untouched
    (idempotent; a pool produced by the previous block's sharded collection
    on the same group costs nothing). Pools smaller than / indivisible by
    the world are left alone (serial consumers handle them via move-on-demand
    cats).
    """
    n = len(pool)
    world = len(devices)
    if world < 2 or n < world or n % world != 0:
        return
    shard = n // world
    for r, dev in enumerate(devices):
        dev = torch.device(dev)
        for i in range(r * shard, (r + 1) * shard):
            if pool[i].device != dev:
                pool[i] = pool[i].to(dev)


def sharded_nograd_forward(
    runner,
    block,
    inputs,
    input_others: dict,
    out_device: torch.device,
    devices: List[torch.device],
    sample_count: Optional[int] = None,
    merge_stats: bool = False,
    max_devices: int = 0,
):
    """Parallelize a no-grad collection forward across ``devices``.

    ``max_devices > 0`` truncates the shard set to the first K devices
    (shards grow accordingly): forward hooks force dynamo graph breaks,
    leaving the compiled runner as python-bound eager sections that
    GIL-convoy beyond a handful of threads -- hook-carrying passes are
    capped by the caller while hookless passes shard wide.

    The collection passes (reference outputs, quantized-output cascade) are
    plain forwards over the whole sample pool on one GPU while the mirrors
    idle. Here the pool is split into equal disjoint shards; an ephemeral
    copy of the block on each device forwards its shard in a parallel thread;
    per-shard outputs are parked on ``out_device`` and concatenated in order.
    Bit-identical to the serial pass: rows are sample-independent and the
    module copies carry identical weights. Falls back to the serial runner
    call when the pool is not divisible or fewer than two devices are given.
    Mirrors are dropped (freed) afterwards. Returned pieces stay on the
    device that computed them (distributed pool); consumers either read
    shard-locally or use device-safe cats.
    """
    world = len(devices)
    n = sample_count if sample_count is not None else (len(inputs) if isinstance(inputs, list) else 0)
    _vram_devs = []
    try:
        import torch.cuda as _tc

        if _tc.is_available():
            _seen = set()
            for _pd in [block] if not isinstance(block, (list, tuple)) else list(block):
                for _pp in _pd.parameters():
                    if _pp.device.type == "cuda":
                        _seen.add(_pp.device)
            _vram_devs = sorted(_seen, key=lambda d: d.index or 0)
            if _vram_devs:
                logger.info(
                    "[tune-vram] collect start, standing reserved: %s",
                    ", ".join(f"{_d.index}:{_tc.memory_reserved(_d) / 2**30:.2f}G" for _d in _vram_devs),
                )
                for _d in _vram_devs:
                    _tc.reset_peak_memory_stats(_d)
    except Exception:  # noqa: BLE001 - diagnostic only
        _vram_devs = []
    if max_devices and 0 < max_devices < world:
        logger.info(
            "[tune-ddp] sharded collect: capping concurrent shards at %d of %d device(s) (hook pass)",
            max_devices,
            world,
        )
        devices = list(devices)[:max_devices]
        world = max_devices
    if world < 2 or n < world or n % world != 0:
        return runner(block, inputs, input_others, cache_device=out_device)
    shard = n // world
    shards = [list(range(r * shard, (r + 1) * shard)) for r in range(world)]
    # the input pool typically arrives pooled on the primary device; without
    # re-placement every shard pulls its pieces cross-device from that ONE
    # source -- measured at world=8: uniformly 2.3-2.8 s of shard-forward
    # time versus 0.6-0.8 s for the same pass reading an already-distributed
    # pool. Distribute first (idempotent, same layout the tune engagement
    # enforces later); serial consumers afterwards re-gather to home.
    # Tensor pools only: the helper's contract also admits opaque pools.
    if (
        isinstance(inputs, list)
        and inputs
        and all(isinstance(t, torch.Tensor) for t in inputs)
        and (sample_count is None or sample_count == len(inputs))
    ):
        distribute_pool(inputs, devices)
    import time as _time

    _t_mirrors = _time.perf_counter()
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
    global _coll_mirror_setup_logged
    if world > 1 and "_coll" not in _coll_mirror_setup_logged:
        _coll_mirror_setup_logged.add("_coll")
        logger.info(
            "[tune-ddp] collection mirror setup: %.0f ms per pass (world=%d) -- included in ref_collect/post_collect",
            (_time.perf_counter() - _t_mirrors) * 1000,
            world,
        )
    _t_setup_ms = (_time.perf_counter() - _t_mirrors) * 1000
    parts: List = [None] * world
    fwd_walls = [0.0] * world  # per-thread stores at distinct indices: race-free
    evs: List = [None] * world

    def _run(r):
        rep = reps[r]
        dev_r = _block_device(rep)
        t_r = _time.perf_counter()
        ev = None
        if dev_r.type == "cuda":
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        # outputs stay ON the replica that computed them: the per-sample list
        # returned below is a distributed pool (device r owns shard r's rows)
        if dev_r.type == "cuda":
            with torch.cuda.device(dev_r):
                parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)
        else:
            parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)
        if ev is not None:
            ev[1].record()
            evs[r] = ev
        fwd_walls[r] = (_time.perf_counter() - t_r) * 1000

    # spawn path, not the ReplicaGroup pool: this bare helper instance has no
    # lifecycle and would leak pool workers (collection passes are twice per
    # block -- the spawn cost is irrelevant here)
    run_threaded_spawn([lambda r=r: _run(r) for r in range(world)])
    # per-shard GPU times: sync the participating devices (their outputs are
    # consumed right after anyway -- the first downstream read syncs too)
    fwd_gpu = [0.0] * world
    for r, dev in enumerate(devices):
        if evs[r] is not None:
            try:
                torch.cuda.synchronize(dev)
                fwd_gpu[r] = evs[r][0].elapsed_time(evs[r][1])
            except RuntimeError:  # a broken pair must never kill the pass
                fwd_gpu[r] = float("nan")
    _t_split = _time.perf_counter()
    # Return the SERIAL structure: a per-sample list ([1, S, H] pieces from
    # split_outputs), NOT one flat/batched tensor -- downstream consumers cat
    # per-sample refs along dim 0 (tune loss) and iterate the list for the
    # cascade inputs, so a tensor here changes every per-sample shape.
    pieces: List[torch.Tensor] = []
    split_outputs = getattr(runner, "split_outputs", None)
    for part in parts:
        pieces.extend(split_outputs(part) if split_outputs else torch.split(part, 1, dim=0))
    # threaded shard calls each stamped a PARTIAL last_output_dict (their own
    # shard's rows); the serial text path leaves it unset so callers fall back
    # to the returned list -- clear the residue to keep that contract.
    runner.last_output_dict = None
    _t_merge = _time.perf_counter()
    if merge_stats:
        _merge_mirror_stats(block, mirrors)
    mirrors.clear()  # drop mirror refs; the caching allocator reclaims them
    reps.clear()
    # steady-state breakdown for the collection passes: setup = ephemeral
    # mirror deepcopy/relocate (per pass!), fwd = threaded shard forwards
    # (wall includes enqueue, gpu is the device-measured chain incl. the
    # per-thread output parking), split/merge = host assembly. Logged for
    # EVERY pass: the first-call-per-block compile signature (uniformly
    # ~2.3-2.8 s fwd at world=8 vs 0.6-0.8 s on the second pass) needs the
    # per-block view to separate warmup from steady state.
    _walls = sorted(fwd_walls)
    _gpus = sorted(fwd_gpu)
    if _vram_devs:
        try:
            import torch.cuda as _tc

            logger.info(
                "[tune-vram] collect done, %s",
                ", ".join(
                    f"{_d.index}:peak {_tc.max_memory_reserved(_d) / 2**30:.2f}G"
                    f"/now {_tc.memory_reserved(_d) / 2**30:.2f}G"
                    for _d in _vram_devs
                ),
            )
        except Exception:  # noqa: BLE001 - diagnostic only
            pass
    logger.info(
        "[tune-ddp] sharded collect breakdown: world=%d n=%d setup=%.0fms "
        "fwd_wall[min/med/max]=%.0f/%.0f/%.0fms fwd_gpu[min/med/max]=%.0f/%.0f/%.0fms "
        "split=%.0fms merge=%.0fms",
        world,
        n,
        _t_setup_ms,
        _walls[0],
        _walls[len(_walls) // 2],
        _walls[-1],
        _gpus[0],
        _gpus[len(_gpus) // 2],
        _gpus[-1],
        (_t_merge - _t_split) * 1000,
        (_time.perf_counter() - _t_merge) * 1000,
    )
    return pieces


_STAGED_MISS_LOGGED = False
_MIRROR_PATH_LOGGED = False


def _repair_replica(home: torch.nn.Module, replica: torch.nn.Module, dev: torch.device) -> None:
    """Make a ``torch.nn.parallel.replicate`` output safe to TUNE.

    ``replicate`` shallow-copies every module: parameters/buffers are
    broadcast per device (chunked, no 2x source spike), but every plain
    ``__dict__`` attribute stays SHARED with home, and the broadcast params
    land as non-leaf non-Parameter attributes (``_former_parameters``) --
    replicas are not trainable modules as-is. The repair walk re-leafs the
    broadcast tensors into fresh ``nn.Parameter``s, re-keys wrapper aliasing
    dicts (``m.params``) to those NEW objects -- otherwise mirror optimizers
    would collect HOME's parameters and silently tune the home -- and clones
    tensor-valued plain attrs (weight_min/max, caches) plus shared mutable
    containers onto the replica device.
    """
    home_mods = dict(home.named_modules())
    # break whole-dict aliasing from shallow copies before walking (children
    # were already re-pointed by replicate; only the container is shared)
    replica._modules = dict(replica._modules)
    replica._parameters = dict(replica._parameters)
    replica._buffers = dict(replica._buffers)
    for name, rmod in replica.named_modules():
        hmod = home_mods.get(name)
        if hmod is None:  # pragma: no cover - replica cannot grow modules
            continue
        former = getattr(rmod, "_former_parameters", None) or {}
        # ── parameters: re-leaf the broadcast copies ──────────────────────
        new_params = {}
        for key, hparam in hmod._parameters.items():
            if hparam is None:
                # None-registered parameter (e.g. Linear(bias=False) keeps
                # ``bias: None``): preserve the registration -- dropping the
                # key breaks attribute lookup on the replica (wrapper forward
                # reads orig_layer.bias)
                new_params[key] = None
                continue
            src = former.get(key)
            if not torch.is_tensor(src):
                cand = rmod._parameters.get(key)
                src = cand if cand is not None and cand is not hparam else None
            if not torch.is_tensor(src):
                src = hparam.detach().to(dev)  # last resort: direct copy
            new_params[key] = torch.nn.Parameter(src.detach(), requires_grad=hparam.requires_grad)
        rmod._parameters = new_params
        for key in new_params:
            rmod.__dict__.pop(key, None)  # drop the non-leaf plain-attr shadow
        if hasattr(rmod, "_former_parameters"):
            try:
                delattr(rmod, "_former_parameters")
            except AttributeError:  # pragma: no cover
                pass
        # ── buffers: adopt the broadcast copy wherever it landed ─────────
        # (rebind a FRESH dict: a shallow copy may share the dict object
        # itself, and in-place writes would corrupt the home module)
        new_buffers = dict(rmod._buffers)
        for key, hbuf in hmod._buffers.items():
            if hbuf is None:
                new_buffers[key] = None  # keep the None registration
                continue
            src = new_buffers.get(key)
            if src is None or src is hbuf:
                src = rmod.__dict__.get(key)
                if not torch.is_tensor(src) or src is hbuf:
                    src = hbuf.to(dev)
                    if src is hbuf:  # .to() is a no-op on the same device
                        src = src.clone()
            new_buffers[key] = src
            rmod.__dict__.pop(key, None)
        rmod._buffers = new_buffers
        # ── plain attrs: clone tensors, deepcopy shared containers ───────
        for attr, val in list(rmod.__dict__.items()):
            if attr.startswith("_") or attr in ("training", "T_destination"):
                continue  # torch-internal bookkeeping / immutable scalars
            hval = hmod.__dict__.get(attr)
            if torch.is_tensor(val) and val is hval:
                setattr(rmod, attr, hval.detach().clone().to(dev))
            elif isinstance(val, (list, dict, set)) and val is hval:
                setattr(rmod, attr, _copy_attr_value(hval))
        # ── wrapper aliasing dict: re-key to the NEW parameters ──────────
        if hasattr(hmod, "orig_layer") and isinstance(getattr(hmod, "params", None), dict):
            fixed = {}
            for k, v in hmod.params.items():
                if isinstance(v, torch.nn.Parameter) and k in rmod._parameters:
                    fixed[k] = rmod._parameters[k]
                elif torch.is_tensor(v):
                    fixed[k] = v.detach().clone().to(dev)
                elif isinstance(v, (list, dict, set)):
                    fixed[k] = _copy_attr_value(v)
                else:
                    fixed[k] = v
            rmod.params = fixed


_PEER_ACCESS_LOGGED = False


def _copy_attr_value(value):
    """Deep-copy a shared attr value, cloning non-leaf tensors by value.

    ``copy.deepcopy`` refuses non-leaf tensors; any that appear inside shared
    containers are cloned detached instead. Leaf tensors and Parameters keep
    the plain deepcopy semantics, so with the ordinary per-parameter wrapper
    layout this is behaviour-identical to copy.deepcopy.
    """
    if torch.is_tensor(value):
        if value.is_leaf:
            return copy.deepcopy(value)
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy_attr_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_attr_value(v) for v in value]
    if isinstance(value, set):
        return set(value)  # tensors are unhashable -- scalar members copy fine
    return copy.deepcopy(value)


_GRAPH_CAPTURE_LOCK = threading.Lock()


class SharedHandle:
    """Deepcopy-transparent holder for process-wide handles (Events, Threads).

    Blocks get handles attached as attributes and are later ``copy.deepcopy``
    -ed (collection mirrors, replica builds); pickling a lock/thread raises.
    Copies keep the SAME underlying object -- exactly the sharing semantics
    these handles need.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


def _cuda_graphs_supported() -> bool:
    """CUDA graphs need a CUDA build; CPU runs stay eager."""
    return torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")


def cuda_graphs_engage(replica_group, *, inputs_are_list=True, is_diffusion=False, has_valid_token_mask=False) -> bool:
    """Whether the DDP tune loop may capture per-replica CUDA graphs.

    Quiet refusals are environment properties only (serial path, no CUDA,
    dict/diffusion inputs, valid-token-mask loss). Streaming mode does NOT
    refuse: background pack/ready-transform threads simply DEFER the capture
    (the loop waits for them at iteration barriers, since global-mode
    capture needs the process CUDA-quiet). Everything failing during
    prepare/capture/replay is a real fault and raises (see
    :class:`GraphedReplicaStep`).
    """
    if replica_group is None:
        return False
    if not (inputs_are_list and not is_diffusion and not has_valid_token_mask):
        return False
    return _cuda_graphs_supported()


def serial_graphs_engage(
    replica_group,
    *,
    device_is_cuda: bool = True,
    inputs_are_list: bool = True,
    is_diffusion: bool = False,
    has_valid_token_mask: bool = False,
    is_lfq_block: bool = False,
    loss_on_tune_device: bool = True,
    fence_free: bool = True,
    accumulate_one: bool = True,
) -> bool:
    """Whether the serial tune loop may capture a whole-iteration CUDA graph.

    Quiet refusals are environment properties only (DDP path, no CUDA,
    dict/diffusion inputs, valid-token-mask loss, LFQ block, loss on another
    device, fence-bound prepare, gradient accumulation). Streaming /
    background threads do NOT refuse -- the capture barrier defers until the
    process is CUDA-quiet. Everything failing during prepare/capture/replay is
    a real fault and raises (see :class:`GraphedReplicaStep`). The env gate
    (``AR_TUNE_CUDA_GRAPHS``) is the caller's concern, as is the step-capture
    escalation (scaler/momentum/weight-decay/dynamic-gap/VRAM).
    """
    if replica_group is not None:
        return False
    if not (
        device_is_cuda
        and inputs_are_list
        and not is_diffusion
        and not has_valid_token_mask
        and not is_lfq_block
        and loss_on_tune_device
        and fence_free
        and accumulate_one
    ):
        return False
    return _cuda_graphs_supported()


def _copy_into_static(dst, src) -> None:
    """Refresh a static buffer tree from a freshly gathered one, leaf-wise.

    Tensors are ``copy_``-ed in place (addresses must stay stable for CUDA
    graph replay); identical objects (shared caches such as rotary tables)
    are skipped; non-tensor leaves must already be equal. A shape or dtype
    mismatch raises -- replay would read a wrong-shaped buffer, so the run
    halts instead of silently training on garbage.
    """
    if isinstance(dst, torch.Tensor):
        if dst is src:
            return
        if not isinstance(src, torch.Tensor):
            raise RuntimeError(f"static-buffer leaf type drift: {type(dst)} vs {type(src)}")
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise RuntimeError(
                f"static-buffer shape/dtype drift: {tuple(dst.shape)}/{dst.dtype} vs " f"{tuple(src.shape)}/{src.dtype}"
            )
        dst.copy_(src, non_blocking=True)
        return
    if isinstance(dst, dict):
        for k in dst:
            if k not in src:
                raise RuntimeError(f"static-buffer key vanished: {k!r}")
            _copy_into_static(dst[k], src[k])
        extra = set(src) - set(dst)
        if extra:
            raise RuntimeError(f"static-buffer gained key(s): {sorted(extra)!r}")
        return
    if isinstance(dst, (tuple, list)):
        if len(dst) != len(src):
            raise RuntimeError(f"static-buffer length drift: {len(dst)} vs {len(src)}")
        for d, v in zip(dst, src):
            _copy_into_static(d, v)
        return
    if dst != src:
        raise RuntimeError(f"static-buffer non-tensor leaf changed: {dst!r} vs {src!r}")


class GraphedReplicaStep:
    """One replica's tune step (prepare eager + compute in a CUDA graph).

    Split contract:
    - ``prepare(...)`` runs eagerly EVERY iteration: gathers this
      iteration's samples into the replica's STATIC device buffers (inputs,
      kwargs, fp reference). Shapes must repeat; a drift raises (halt).
    - ``compute()`` is the pure-GPU region -- forward + loss + backward on
      the static buffers, grads parked (``zero_grad(set_to_none=False)``)
      so backward accumulates into the addresses baked at capture.
    - the first ``warmup_iters`` ``__call__``-s run prepare + compute
      eagerly; ``capture_now()`` then records the graph (from the MAIN
      thread at a point where the pool workers are parked -- torch.cuda.graph
      forbids concurrent CUDA work process-wide during capture) and
      immediately replays once so that iteration's work executes; later
      ``__call__``-s just prepare + replay. The returned loss is a CLONE
      of the static output tensor (its buffer is rewritten by each replay).

    Any prepare/capture/replay failure is logged and re-raised: the run
    halts instead of silently degrading -- set AR_TUNE_CUDA_GRAPHS=0 to
    disable graphs.
    """

    def __init__(
        self,
        prepare,
        compute,
        *,
        warmup_iters: int = 2,
        name: str = "",
        device=None,
        graph_factory=None,
        capture_ctx=None,
    ):
        self.prepare = prepare
        self.compute = compute
        self.warmup_iters = warmup_iters
        self.name = name or "replica"
        self.device = device
        self._graph = None
        self._static_loss = None
        self._calls = 0
        self._graph_factory = graph_factory if graph_factory is not None else lambda: torch.cuda.CUDAGraph()
        # explicit per-device capture stream: torch.cuda.graph() without
        # stream= uses a process-wide singleton bound to whichever device
        # captured first -- later replicas on other devices would fail
        self._capture_stream = (
            torch.cuda.Stream(device=device) if device is not None and _cuda_graphs_supported() else None
        )
        self._capture_ctx = (
            capture_ctx if capture_ctx is not None else lambda g: torch.cuda.graph(g, stream=self._capture_stream)
        )

    def _run_eager(self):
        """Uncaptured fwd+loss+bwd, on the capture stream when CUDA.

        prepare() copies run on the default stream: order them in (side waits
        default), run compute under the side stream so the autograd nodes it
        creates match the capture stream, then order the default stream after
        it (the exchange/step that follow read the grads).
        """
        if self._capture_stream is None:
            return self.compute()
        cur = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(cur)
        with torch.cuda.device(self.device), torch.cuda.stream(self._capture_stream):
            loss = self.compute()
        cur.wait_stream(self._capture_stream)
        return loss

    def capture_now(self):
        """Record the graph now (main thread, workers parked) and replay once.

        The process-wide capture lock is held anyway: torch.cuda.graph uses
        a singleton capture stream, so captures must serialize even if a
        future caller forgets the parked-workers contract.
        """
        try:
            with _GRAPH_CAPTURE_LOCK:
                with torch.cuda.device(self.device) if self.device is not None else contextlib.nullcontext():
                    graph = self._graph_factory()
                    with self._capture_ctx(graph):
                        self._static_loss = self.compute()
                    # capture only RECORDS: execute this iteration's work
                    graph.replay()
                    self._graph = graph
        except Exception as exc:
            logger.error(
                "[tune-ddp] cuda graph capture failed for %s (%s): set AR_TUNE_CUDA_GRAPHS=0 to disable",
                self.name,
                exc,
            )
            raise
        return self._static_loss.detach().clone()

    def prepare_only(self, *args, **kwargs):
        """Run just the eager prepare (paced dispatch round 1). Failures halt."""
        self._prepare_calls = getattr(self, "_prepare_calls", 0) + 1
        try:
            self.prepare(*args, **kwargs)
        except Exception as exc:
            logger.error(
                "[tune-ddp] cuda-graph prepare failed for %s (%s): set AR_TUNE_CUDA_GRAPHS=0 to disable",
                self.name,
                exc,
            )
            raise

    def replay_only(self):
        """Replay the captured step and clone the static loss (round 2)."""
        if self._graph is None:
            raise RuntimeError(f"replay_only() called before capture for {self.name}")
        try:
            self._graph.replay()
        except Exception as exc:
            logger.error(
                "[tune-ddp] cuda-graph replay failed for %s (%s): set AR_TUNE_CUDA_GRAPHS=0 to disable",
                self.name,
                exc,
            )
            raise
        return self._static_loss.detach().clone()

    def __call__(self, *args, **kwargs):
        self._prepare_calls = getattr(self, "_prepare_calls", 0) + 1
        try:
            self.prepare(*args, **kwargs)
        except Exception as exc:
            logger.error(
                "[tune-ddp] cuda-graph prepare failed for %s (%s): set AR_TUNE_CUDA_GRAPHS=0 to disable",
                self.name,
                exc,
            )
            raise
        if self._graph is None:
            self._calls += 1
            # uncaptured iterations run eagerly: the first warmup_iters by
            # design, later ones while the capture barrier DEFERS (background
            # pack/transform threads still issue CUDA -- global-mode capture
            # needs the process quiet, so capture waits for them to finish).
            # Eager iterations run on the CAPTURE stream: AccumulateGrad nodes
            # must be created on the same stream that later captures, or the
            # engine's cross-stream sync makes the legacy stream depend on the
            # capturing stream (cudaErrorStreamCaptureImplicit) -- the
            # make_graphed_callables pattern.
            try:
                return self._run_eager().detach()
            except Exception as exc:
                logger.error(
                    "[tune-ddp] cuda-graph warm-up compute failed for %s (%s): " "set AR_TUNE_CUDA_GRAPHS=0 to disable",
                    self.name,
                    exc,
                )
                raise
        try:
            self._graph.replay()
        except Exception as exc:
            logger.error(
                "[tune-ddp] cuda graph replay failed for %s (%s): set AR_TUNE_CUDA_GRAPHS=0 to disable",
                self.name,
                exc,
            )
            raise
        return self._static_loss.detach().clone()


def _plan_buckets(params: List[torch.nn.Parameter], num_buckets: int) -> List[List[torch.nn.Parameter]]:
    """Split a flat param list into <=num_buckets contiguous, param-aligned buckets."""
    total = sum(p.numel() for p in params)
    target = max(1, total // max(1, num_buckets))
    buckets: List[List[torch.nn.Parameter]] = []
    cur: List[torch.nn.Parameter] = []
    cur_n = 0
    for p in params:
        cur.append(p)
        cur_n += p.numel()
        if cur_n >= target and len(buckets) < num_buckets - 1:
            buckets.append(cur)
            cur, cur_n = [], 0
    if cur:
        buckets.append(cur)
    return buckets


class BucketExchangeSession:
    """Overlapped gradient exchange: buckets ship as their backprops complete.

    One daemon thread is the SINGLE issuer of all cross-device copies (the
    removed hook-fan-out overlap destroyed the P2P fast path by issuing from
    whichever autograd thread fired). Per-replica post-accumulate-grad hooks
    fire during EAGER backward only (autograd host callbacks do not run on
    CUDA-graph replay -- the caller must gate overlap off when graphs are
    active), mark their bucket ready and record a CUDA event on the stream
    that wrote the grads; the thread waits
    for all replicas' events, then runs the existing sign-exchange algorithm
    on just that bucket's slice (bit-identical per element: the halving tree
    of each element is unchanged by bucketing). Grads are written back in
    place, so the end-of-iteration step is unchanged.

    Any exchange error halts: join_and_check() re-raises (no silent latch).
    """

    def __init__(self, params_per_replica, transport: str, exchange_fn=None, num_buckets: int = 12):
        self.world = len(params_per_replica)
        if self.world < 2:
            raise ValueError("BucketExchangeSession needs world >= 2")
        self.params_per_replica = params_per_replica
        if any(len(ps) != len(params_per_replica[0]) for ps in params_per_replica):
            raise ValueError("BucketExchangeSession needs identical param layouts across replicas")
        self.transport = transport
        # buckets are INDEX RANGES over the flat param list: each replica owns
        # its own Parameter objects at the same positions (mirrors)
        self.bucket_ranges = []
        # FIXED per-bucket flat layouts derived from the PARAM SHAPES (never
        # from which grads happen to exist at exchange time): every replica's
        # exchange buffer has the SAME length, with zero at positions whose
        # param has no grad on that replica. (Building the buffer by cat-ing
        # the not-None grads made lengths DIVERGE across replicas when grad
        # presence differed -- the halving reduce-scatter then sliced two
        # different-sized buffers and crashed with a size-mismatch.)
        self.bucket_layouts: List[List[tuple]] = []
        self.bucket_sizes: List[int] = []
        start = 0
        for bucket in _plan_buckets(params_per_replica[0], num_buckets):
            self.bucket_ranges.append((start, start + len(bucket)))
            off, layout = 0, []
            for p in bucket:
                layout.append((off, p.numel()))
                off += p.numel()
            self.bucket_layouts.append(layout)
            self.bucket_sizes.append(off)
            start += len(bucket)
        self._flat_bufs: List[List[Optional[torch.Tensor]]] = [
            [None] * len(self.bucket_ranges) for _ in range(self.world)
        ]
        self._exchange_fn = exchange_fn if exchange_fn is not None else self._default_exchange
        self._cv = threading.Condition(threading.Lock())
        self._ready = [0] * self.num_buckets
        self._processed: set = set()
        self._error: Optional[BaseException] = None
        self._iter_active = False
        self._thread: Optional[threading.Thread] = None
        self._handles: list = []
        self._events: dict = {}  # (replica, bucket) -> torch.cuda.Event on CUDA

    def _bucket_params(self, r: int, b: int) -> List[torch.nn.Parameter]:
        lo, hi = self.bucket_ranges[b]
        return self.params_per_replica[r][lo:hi]

    @property
    def num_buckets(self) -> int:
        return len(self.bucket_ranges)

    # ── arming ────────────────────────────────────────────────────────────
    def arm(self):
        """Register one completion hook per (replica, bucket). Call once per block."""
        for r in range(self.world):
            for b in range(self.num_buckets):
                # backward completes last-module-first: the bucket's EARLIEST
                # param is the last to finish, so it marks bucket completion
                sentinel = self._bucket_params(r, b)[0]
                self._handles.append(sentinel.register_post_accumulate_grad_hook(self._make_hook(r, b, sentinel)))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, r, b, sentinel):
        def hook(_param):
            if not self._iter_active or self._error is not None:
                return
            try:
                if torch.cuda.is_available() and sentinel.grad is not None and sentinel.grad.is_cuda:
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream(sentinel.grad.device))
                    self._events[(r, b)] = ev
            except Exception:  # noqa: BLE001 - event recording must not kill bwd
                pass
            with self._cv:
                self._ready[b] += 1
                self._cv.notify_all()

        return hook

    # ── per-iteration lifecycle ───────────────────────────────────────────
    def begin_iter(self):
        with self._cv:
            self._ready = [0] * self.num_buckets
            self._processed = set()
            self._error = None
            self._iter_active = True
            self._events = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="tune-bucket-exchange")
        self._thread.start()

    def join_and_check(self, timeout: Optional[float] = 300.0):
        assert self._thread is not None
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "[tune-ddp] bucket exchange stalled: ready=%s processed=%s -- a bucket never completed "
                "(grad-less or frozen sentinel param?); set AR_TUNE_EXCH_OVERLAP=0 to fall back "
                "to the monolithic exchange",
                self._ready,
                sorted(self._processed),
            )
            raise TimeoutError("bucket exchange did not finish (join timeout)")
        err = self._error
        with self._cv:
            self._iter_active = False
        if err is not None:
            raise err

    # ── worker ────────────────────────────────────────────────────────────
    def _run(self):
        try:
            n = self.num_buckets
            while len(self._processed) < n:
                with self._cv:
                    self._cv.wait_for(
                        lambda: self._error is not None
                        or any(self._ready[b] >= self.world for b in range(n) if b not in self._processed)
                    )
                    if self._error is not None:
                        return
                    pick = next(b for b in range(n) if b not in self._processed and self._ready[b] >= self.world)
                    self._processed.add(pick)  # claim before releasing the lock
                self._exchange_fn(pick)
                with self._cv:
                    self._cv.notify_all()
        except BaseException as exc:  # noqa: BLE001 - halt, never latch silently
            self._error = exc
            with self._cv:
                self._cv.notify_all()

    def _default_exchange(self, b: int):
        buckets = [self._bucket_params(r, b) for r in range(self.world)]
        # order the copies after the grad writes (events recorded at hook time
        # on whichever stream ran that replica's backward)
        for r in range(self.world):
            ev = self._events.get((r, b))
            if ev is not None:
                lo, _hi = self.bucket_ranges[b]
                dev = self.params_per_replica[r][lo].device
                if dev.type == "cuda":
                    torch.cuda.current_stream(dev).wait_event(ev)
        bufs = []
        for r, bucket in enumerate(buckets):
            buf = self._flat_bufs[r][b]
            if buf is None:
                dev = bucket[0].device if bucket else self.params_per_replica[r][self.bucket_ranges[b][0]].device
                buf = torch.zeros(self.bucket_sizes[b], dtype=torch.float32, device=dev)
                self._flat_bufs[r][b] = buf
            buf.zero_()  # no-grad positions contribute zero on this replica
            for p, (off, n) in zip(bucket, self.bucket_layouts[b]):
                if p.grad is not None:
                    buf[off : off + n].copy_(p.grad.detach().reshape(-1))
            bufs.append(buf)
        sign_exchange_allreduce(bufs, transport=self.transport)
        for buf, bucket in zip(bufs, buckets):
            for p, (off, n) in zip(bucket, self.bucket_layouts[b]):
                if p.grad is not None:
                    p.grad.copy_(buf[off : off + n].view_as(p.grad))


class HostLeafPinner:
    """Pin host-side tensor leaves that REPEAT by identity, pass fresh ones through.

    Prepare-time refresh copies of constant CPU leaves (shared caches, static
    metadata) go through pageable H2D every iteration; a pinned source makes
    them async staging-free copies. Only leaves seen AGAIN as the same object
    get pinned: one-shot tensors (freshly cat-ed per-sample masks) never
    repeat, so they pass through untouched -- pinning them would grow the
    cache unboundedly, and caching by id() alone could hand back a previous
    iteration's data after garbage collection recycles the address. CUDA
    tensor leaves pass through untouched.
    """

    def __init__(self):
        # id(src) -> (src, pinned|None); the strong src reference pins the id,
        # first-sighting entries carry pinned=None until pin_repeats() runs
        self._cache: dict = {}

    def wrap(self, tree):
        if isinstance(tree, torch.Tensor):
            if tree.device.type != "cpu" or not torch.cuda.is_available():
                return tree
            entry = self._cache.get(id(tree))
            if entry is not None:
                src, pinned = entry
                return pinned if (pinned is not None and src is tree) else tree
            self._cache[id(tree)] = (tree, None)  # remember first sighting
            return tree
        if isinstance(tree, dict):
            return {k: self.wrap(v) for k, v in tree.items()}
        if isinstance(tree, (tuple, list)):
            return type(tree)(self.wrap(v) for v in tree)
        return tree

    def pin_repeats(self):
        """Promote first-sightings to pinned copies (call after gather #1).

        Constant leaves are exactly the ones the next wrap() will see again
        as the same object; fresh one-shot tensors never re-appear, so their
        remember-entries simply go unused (and die with the pinner).
        """
        if not torch.cuda.is_available():
            return
        for key, entry in list(self._cache.items()):
            src, pinned = entry
            if pinned is None:
                self._cache[key] = (src, src.pin_memory())


def run_proactive_paced_steps(group, steps, current_shards, next_shards, out_losses, worker_ms, prepared):
    """Proactive pipeline: replay the current step, then prepare the NEXT one.

    ``prepared`` says whether ``current_shards`` was already staged at the
    previous iteration's tail (True -> replay-round only); False -> the first
    steady iteration, which runs the full paced prepare+replay. After the
    replay round returns, iteration i+1's prepare is enqueued while the GPU
    still executes iteration i -- the static-buffer refresh is a
    write-after-read ordered by the per-device stream FIFO.
    """
    if not prepared:
        run_paced_replica_steps(group, steps, current_shards, out_losses, worker_ms)
    else:
        world = len(steps)

        def _replay(r):
            t0 = time.perf_counter()
            out_losses[r] = steps[r].replay_only()
            if worker_ms is not None:
                worker_ms[r].append((time.perf_counter() - t0) * 1000.0)

        group.run_threaded([lambda r=r: _replay(r) for r in range(world)])
    if next_shards is not None:
        group.run_threaded([lambda r=r: steps[r].prepare_only(next_shards[r]) for r in range(len(steps))])


def accumulate_serial_loss(acc, loss, num_elm: float = 1.0):
    """Accumulate one serial-batch loss term on device (fp64, left-to-right).

    Mirrors the legacy ``total_loss += loss.item() / num_elm`` Python float
    sum exactly (same order, same fp64 precision) while keeping the host off
    the loss->backward critical path: read the accumulated total once AFTER
    the batch loop's backward enqueues.
    """
    term = loss.detach().double() / num_elm
    return term if acc is None else acc + term


def make_serial_graphed_step(
    block,
    block_fwd,
    active_inputs,
    input_others,
    fp_outputs,
    loss_fn,
    device,
    fence_free: bool = True,
    warmup_iters: int = 2,
    graph_factory=None,
    capture_ctx=None,
    step_fn=None,
):
    """Serial twin of the DDP graphed replica step (T8b).

    ``prepare(row)`` gathers the iteration's samples into SINGLE-DEVICE static
    buffers (first call materializes, later calls ``_copy_into_static``-refresh
    -- addresses must stay stable for CUDA graph replay); ``compute()`` is the
    pure-GPU region (forward + loss + backward on the static buffers). The
    returned :class:`GraphedReplicaStep` handles warmup/capture/replay; the
    optimizer step stays EAGER (extended into the graph by T8c).

    ``loss_fn(pred, ref)`` must not consume sample indices: the engaged serial
    path has ``valid_token_mask=None`` so the indices argument of the loss is
    never read.
    """
    st = {"inputs": None, "others": None, "ref": None}
    pinner = HostLeafPinner()
    _pinned_once = False
    dev_is_cuda = getattr(device, "type", "") == "cuda"
    dev_ctx = torch.cuda.device(device) if dev_is_cuda else contextlib.nullcontext()

    def _to_dev(x):
        if isinstance(x, torch.Tensor) and x.device != device:
            return x.to(device)
        if isinstance(x, dict):
            return {k: _to_dev(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return type(x)(_to_dev(v) for v in x)
        return x

    def prepare(row):
        nonlocal _pinned_once
        with dev_ctx:
            expect_pool_local([fp_outputs[j] for j in row], device, "serial-ref")
            ref = torch.cat([fp_outputs[j].to(device) for j in row], dim=0)
            bins, bother = [], []
            if fence_free:
                # host-int batches: no device indices tensor (pageable H2D)
                # and no per-element .item() fences in select_batch
                for i in range(0, len(row), block_fwd.batch_size):
                    bi = row[i : i + block_fwd.batch_size]
                    _in, _oth = block_fwd.select_batch(active_inputs, input_others, bi)
                    bins.append(_in)
                    bother.append(pinner.wrap(_oth))
                if not _pinned_once:
                    _pinned_once = True
                    pinner.pin_repeats()
            else:
                indices = torch.tensor(row, dtype=torch.long, device=device)
                for i in range(0, len(indices), block_fwd.batch_size):
                    bi = indices[i : i + block_fwd.batch_size]
                    _in, _oth = block_fwd.select_batch(active_inputs, input_others, bi)
                    bins.append(_in)
                    bother.append(_oth)
            if st["ref"] is None:
                # shallow-copy the dict nodes: the runner pops keys from the
                # kwargs it receives; the static structure must stay pristine.
                # Stage every leaf on the device: to_device() inside the
                # captured region must no-op (unpinned H2D during capture is
                # illegal); later prepares refresh cross-device eagerly.
                st["ref"], st["inputs"] = _to_dev(ref), [_to_dev(b) for b in bins]
                st["others"] = [_to_dev(dict(o)) for o in bother]
                return
            _copy_into_static(st["ref"], ref)
            _copy_into_static(st["inputs"], bins)
            _copy_into_static(st["others"], bother)

    def compute():
        with dev_ctx:
            outs = []
            for _in, _oth in zip(st["inputs"], st["others"]):
                # dict() per call: the runner pops keys from the kwargs it
                # receives (same tensor leaves -- the graph-baked addresses
                # are untouched)
                raw = block_fwd._forward_one_batch(block, _in, dict(_oth))
                out = block_fwd._normalize_output(raw, block)
                outs.append(out.to(device) if out.device != device else out)
            pred = outs[0] if len(outs) == 1 else torch.cat(outs, dim=block_fwd.batch_dim)
            loss = loss_fn(pred, st["ref"])
            loss.backward()
            if step_fn is not None:
                # T8c: the optimizer step runs INSIDE the captured region --
                # warmup iterations execute it eagerly with the same
                # device-lr tensors the replay will read
                step_fn()
            return loss

    return GraphedReplicaStep(
        prepare,
        compute,
        warmup_iters=warmup_iters,
        name="serial",
        device=device if dev_is_cuda else None,
        graph_factory=graph_factory,
        capture_ctx=capture_ctx,
    )


def sign_step_foreach_ok(optimizer) -> bool:
    """Whether the optimizer qualifies for the foreach pure-sign fast path.

    Mirrors SignSGD's engaged configuration EXACTLY: a SignSGD instance (not a
    different optimizer class/subclass with its own step semantics) whose
    defaults AND every per-group override keep momentum 0, weight_decay 0,
    maximize off (maximize would flip the update direction), nesterov off.
    Checked on the optimizer's own state so home and mirror optimizers
    qualify identically.
    """
    from auto_round import envs as _envs
    from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

    if not _envs.AR_TUNE_FOREACH_STEP:
        return False  # bisect/rollback lever: legacy single-tensor loop
    if not isinstance(optimizer, SignSGD):
        return False
    d = getattr(optimizer, "defaults", {}) or {}
    if d.get("maximize", False) or d.get("nesterov", False):
        return False
    if float(d.get("momentum", 0) or 0) != 0.0 or float(d.get("weight_decay", 0) or 0) != 0.0:
        return False
    for group in optimizer.param_groups:
        if group.get("maximize", d.get("maximize", False)) or group.get("nesterov", d.get("nesterov", False)):
            return False
        if float(group.get("momentum", d.get("momentum", 0)) or 0) != 0.0:
            return False
        if float(group.get("weight_decay", d.get("weight_decay", 0)) or 0) != 0.0:
            return False
    return True


def _sign_step_foreach(optimizer, free_grads: bool = False) -> None:
    """Pure-sign SignSGD step via foreach ops (bit-identical, ~3 kernels/group).

    Same math as SignSGD's always-used ``_single_tensor_sgd`` for the engaged
    configuration: ``param.add_(sign(grad), alpha=-lr)`` then a parked zero
    (``set_to_none=False`` semantics). Params with ``grad=None`` are skipped,
    exactly like the optimizer. The 1195-launch single-tensor loop collapses
    to ~3 foreach kernels per param group. ``free_grads=True`` releases the
    grad tensors after the update (host-side, matching the legacy
    ``set_to_none=True`` between-iterations behavior for non-graph callers).
    """
    with torch.no_grad():
        # the wrapped optimizer.step() would set this; the foreach fast path
        # replaces that call, so set it here -- LRScheduler.step() otherwise
        # fires a false-positive "before optimizer.step()" warning
        optimizer._opt_called = True  # noqa: B010  (torch's own protocol)
        for group in optimizer.param_groups:
            pairs = [(p, p.grad) for p in group["params"] if p.grad is not None]
            if not pairs:
                continue
            params = [p for p, _ in pairs]
            grads = [g for _, g in pairs]
            signs = torch._foreach_sign(grads)
            torch._foreach_add_(params, signs, alpha=-float(group["lr"]))
            if free_grads:
                for p in params:
                    p.grad = None
            else:
                torch._foreach_zero_(grads)


def serial_step_vram_ok(needed_bytes, free_fn, margin: int = 1 << 30) -> bool:
    """Whole-iteration staging VRAM guard (T8c).

    The pre-step parameter staging costs one full tuning-param set; engage
    whole-iteration capture only when the device holds it plus a margin.
    ``free_fn=None`` (no CUDA context) always allows -- CPU tests stage freely.
    """
    if free_fn is None:
        return True
    free, _total = free_fn()
    return needed_bytes + margin <= free


def make_serial_step_fn(optimizer, device):
    """Capture-safe whole-iteration step (T8c) for the serial tune loop.

    Returns ``(step_fn, pre_by_id)``:

    - ``step_fn`` must run AFTER backward (inside the captured region, or
      eagerly during warmup): it first stages every optimizer parameter into
      its PRE-STEP twin (the best-params snapshot must pair loss L_i with the
      pre-step iterate P_i -- a post-replay live read would see P_{i+1}), then
      applies the SignSGD update ``param.add_(sign(grad), alpha=-lr)`` with the
      lr read from a per-group 0-dim DEVICE tensor (``group['_ar_neg_lr_dev']``,
      param dtype -- a baked python-float lr would freeze the schedule), and
      zeroes grads in place (parked addresses, ``set_to_none=False`` semantics).
      Mirrors SignSGD's engaged configuration (momentum=0, weight_decay=0)
      bit-for-bit: the single fp64->dtype rounding of ``-lr`` matches the
      in-kernel alpha cast.
    - ``pre_by_id`` maps ``id(param)`` -> staging twin for
      :func:`collect_best_params_pre`.
    """
    pre_step = []
    pre_by_id = {}
    for group in optimizer.param_groups:
        params = [p for p in group["params"] if p.requires_grad]
        if not params:
            continue
        dtype = params[0].dtype
        if any(p.dtype != dtype for p in params):
            raise ValueError("mixed dtypes inside a param group: device-lr twin cannot stay bit-exact")
        group["_ar_neg_lr_dev"] = torch.full((), -float(group["lr"]), dtype=dtype, device=device)
        for p in params:
            pre = torch.empty_like(p)
            pre.copy_(p)  # staging starts at the current iterate
            pre_step.append((p, pre))
            pre_by_id[id(p)] = pre

    def step_fn():
        with torch.no_grad():  # leaf in-place updates (SignSGD step decorator)
            optimizer._opt_called = True  # noqa: B010  (see _sign_step_foreach)
            for p, pre in pre_step:
                pre.copy_(p)
            for group in optimizer.param_groups:
                lr_dev = group.get("_ar_neg_lr_dev")
                if lr_dev is None:
                    continue
                pairs = [(p, p.grad) for p in group["params"] if p.grad is not None]
                if not pairs:
                    continue
                params = [p for p, _ in pairs]
                grads = [g for _, g in pairs]
                signs = torch._foreach_sign(grads)
                torch._foreach_mul_(signs, lr_dev)
                torch._foreach_add_(params, signs)
                torch._foreach_zero_(grads)

    return step_fn, pre_by_id


def collect_best_params_pre(block, cache_device, pre_by_id):
    """:func:`collect_best_params` reading PRE-STEP staging twins (T8c).

    Identical walk; each tuning parameter is read from its pre-step twin so the
    snapshot pairs loss L_i with iterate P_i even though the optimizer step ran
    inside the replay. Frozen (non-optimizer) params fall through to the live
    tensor -- they never step, so live == pre.
    """
    params = {}
    if hasattr(block, "orig_layer"):
        for key in block.params.keys():
            src = pre_by_id.get(id(block.params[key]), block.params[key])
            params[key] = src.data.to(cache_device, copy=True)
    else:
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                params[n] = {}
                for key in m.params.keys():
                    src = pre_by_id.get(id(m.params[key]), m.params[key])
                    params[n][key] = src.data.to(cache_device, copy=True)
    return params


def run_paced_replica_steps(group, steps, shards, out_losses, worker_ms=None):
    """Two-round graphed dispatch: all prepares (pool latch), then all replays.

    The pool's per-round completion event is the barrier: replays launch within
    microseconds of each other once every prepare has finished, restoring the
    lockstep the eager host enqueue used to provide. Falls back nowhere -- the
    caller gates this on every step being captured.
    """
    world = len(steps)
    group.run_threaded([lambda r=r: steps[r].prepare_only(shards[r]) for r in range(world)])

    def _replay(r):
        t0 = time.perf_counter()
        out_losses[r] = steps[r].replay_only()
        if worker_ms is not None:
            worker_ms[r].append((time.perf_counter() - t0) * 1000.0)

    group.run_threaded([lambda r=r: _replay(r) for r in range(world)])


def run_threaded_spawn(fns: Sequence) -> None:
    """Run one callable per item in parallel SPAWNED threads (legacy path).

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


class ReplicaThreadPool:
    """Persistent per-replica workers replacing per-iteration thread spawns.

    The tune loop calls ``run_threaded`` twice per iteration (forward shards
    + mirror optimizer steps); spawning and joining ``world`` fresh threads
    each call costs ~1-2 ms per thread of setup/teardown plus GIL churn --
    a measurable slice of the per-iteration host gap. The pool keeps one
    daemon worker per replica alive for the block's lifetime, fed through
    per-worker queues; each round's completion is signalled by per-worker
    events, preserving the spawn path's semantics exactly: every callable
    runs to completion, and the first failure BY WORKER INDEX is re-raised
    in the calling thread. Workers are daemon threads so a pool abandoned
    by a crash can never hang process exit; ``shutdown()`` joins them in
    the normal flow (ReplicaGroup.teardown).
    """

    def __init__(self, n_workers: int) -> None:
        self.n = n_workers
        self._queues: List["queue.Queue"] = [queue.Queue() for _ in range(n_workers)]
        self._errors: List[Optional[BaseException]] = [None] * n_workers
        self._events: List[threading.Event] = []
        self._threads: List[threading.Thread] = []
        for i in range(n_workers):
            t = threading.Thread(target=self._worker, args=(i,), name=f"tune-ddp-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self, idx: int) -> None:
        for fn in iter(self._queues[idx].get, None):  # None = poison pill
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced by run()
                self._errors[idx] = exc
            finally:
                self._events[idx].set()

    def run(self, fns: Sequence) -> None:
        """Execute one round: fns[i] runs on worker i; raise first-by-index error."""
        if len(fns) != self.n:
            raise ValueError(f"wrong count: pool has {self.n} worker(s), got {len(fns)} callable(s)")
        # fresh round state before anything is enqueued (workers are idle
        # between rounds -- run() only returns after every event of the
        # previous round was set)
        self._errors = [None] * self.n
        self._events = [threading.Event() for _ in range(self.n)]
        for i, fn in enumerate(fns):
            self._queues[i].put(fn)
        for ev in self._events:
            ev.wait()
        failed = [i for i, exc in enumerate(self._errors) if exc is not None]
        if failed:
            raise self._errors[min(failed)]

    def shutdown(self) -> None:
        for q in self._queues:
            q.put(None)
        for t in self._threads:
            t.join(timeout=30)


class _AsyncBestTracker:
    """Fence-free best-params selection: device compare, pinned flag, host poll.

    stage() enqueues ONLY GPU work: the world-sum loss into ``all_losses[i]``
    (fp64, summed left-to-right exactly like the legacy host float sum), the
    strict-less compare against the running device minimum (updated by
    ``torch.where`` in the same stream -- equality keeps the older best, the
    Python ``<`` tie rule), and the boolean into a PINNED host byte guarded
    by a CUDA event. poll() -- once per iteration -- resolves the oldest
    READY flags without blocking and promotes by pointer swap over the
    staged (pre-step) snapshots; a pending cap of 2 bounds snapshot VRAM
    (the rare forced wait resolves the oldest flag). drain() performs the
    one legal bounded wait at loop end and publishes init/best values.

    Selection semantics are identical to immediate selection: same
    (loss, pre-step params) pairs, same argmin, same first-wins ties.
    """

    def __init__(self, iters: int, device: torch.device, world: int = 1):
        self.iters = iters
        self.device = device
        self.world = world
        self._cuda = device.type == "cuda" and torch.cuda.is_available()
        self.all_losses = torch.zeros(iters, dtype=torch.float64, device=device)
        self.best = torch.full((), float("inf"), dtype=torch.float64, device=device)
        self._flags = torch.zeros(iters, dtype=torch.uint8, pin_memory=self._cuda)
        self._events = [torch.cuda.Event() if self._cuda else None for _ in range(iters)]
        # per-replica staging slots on the loss device: replica losses live on
        # their own devices, cross-device adds are illegal -- gather first
        self._slots = [torch.zeros((), dtype=torch.float64, device=device) for _ in range(world)]
        self._pending = []  # FIFO of (iter, snapshot); capped at 1 (parity with the old delayed mode)
        self.best_params = None
        self.best_iter = None
        self.init_loss = None
        self.best_loss = None
        self._just_promoted = False

    # ── per-iteration enqueue (no host sync) ──────────────────────────────
    def stage(self, snapshot, losses, i: int):
        if not losses:
            return
        for r, l in enumerate(losses[: self.world]):
            # cross-device copy_ into the loss-device slot (casts to fp64);
            # subsequent ops on this device's stream are ordered after it
            self._slots[r].copy_(l)
        loss = self._slots[0]
        for extra in self._slots[1 : len(losses[: self.world])]:
            loss = loss + extra
        self.all_losses[i].copy_(loss)
        flag = loss < self.best
        self.best = torch.where(flag, loss, self.best)
        if self._cuda:
            self._flags[i].copy_(flag.to(torch.uint8), non_blocking=True)
            self._events[i].record()
        else:
            self._flags[i] = flag.to(torch.uint8).item()  # CPU: synchronous semantics
        while self._pending:  # cap 1: same snapshot VRAM as the old delayed mode
            self._resolve_oldest(wait=True)
        self._pending.append((i, snapshot))

    # ── non-blocking resolution ───────────────────────────────────────────
    def _ready(self, i: int) -> bool:
        return self._events[i].query() if self._cuda else True

    def _resolve_oldest(self, wait: bool):
        i, snap = self._pending[0]
        if wait and self._cuda:
            self._events[i].synchronize()
        elif not self._ready(i):
            return
        self._pending.pop(0)
        if bool(self._flags[i].item()):  # pinned byte: landed before the event
            self.best_params = snap
            self.best_iter = i
            self._just_promoted = True

    def poll(self):
        """Resolve ready flags (never blocks); call once per iteration."""
        self._just_promoted = False
        while self._pending and self._ready(self._pending[0][0]):
            self._resolve_oldest(wait=False)

    @property
    def last_promoted(self) -> bool:
        return self._just_promoted

    # ── loop-end drain (the one legal wait) ───────────────────────────────
    def drain(self):
        while self._pending:
            self._resolve_oldest(wait=True)
        if self.iters > 0:
            self.init_loss = float(self.all_losses[0])
        if self.best_iter is not None:
            self.best_loss = float(self.best)


def _template_signature(home: torch.nn.Module, world: int, devices, extra=()) -> tuple:
    """Signature of a block's module template for cross-block graph reuse.

    Two blocks share a template iff their module structure and every
    parameter/buffer shape+dtype match -- exactly the condition under which a
    captured replica graph (baked tensor addresses, static input buffers) can
    be replayed for the other block after syncing values into the persistent
    mirror modules. Devices participate: the cached mirrors must live on the
    same plan. ``extra`` carries everything value-shaping that no shape
    encodes: the quant config (recipe, bits, data_type, sym, asym_search,
    minmax flag) and the calibration geometry (nsamples, batch, pool shapes) --
    a mixed-bits or config-changed run must never false-hit a cached template.
    """

    def _qcfg(m):
        # per-module quant config: mixed-bit layouts give different templates
        return (
            getattr(m, "bits", None),
            getattr(m, "group_size", None),
            getattr(m, "data_type", None),
            getattr(m, "sym", None),
        )

    mods = tuple((name, type(m).__name__, _qcfg(m)) for name, m in home.named_modules())
    params = tuple((tuple(p.shape), str(p.dtype)) for p in home.parameters())
    buffers = tuple((tuple(b.shape), str(b.dtype)) for b in home.buffers())
    return (mods, params, buffers, world, tuple(str(d) for d in devices), tuple(extra))


# Cross-block cache of layer-module TEMPLATES for graphs: entry holds the
# persistent mirror modules, the per-replica GraphedReplicaStep objects (with
# their CAPTURED graphs) and the static buffer dicts -- all address-stable, so
# a same-template block syncs its values in and replays instead of rebuilding
# mirrors and re-capturing. LRU-bounded by AR_TUNE_GRAPH_TEMPLATE_CACHE
# (default 2; 0 disables -- qwen3.5's 3:1 linear:full interleave needs 2).
_GRAPH_TEMPLATE_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()


def sync_module_values(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """One-way address-stable value sync between same-template modules.

    Real block -> persistent replicas (block start) and staging home -> real
    block (loop end). See :func:`sync_mirror_from_home`.
    """
    sync_mirror_from_home(src, dst)


def sync_mirror_from_home(home: torch.nn.Module, mirror: torch.nn.Module) -> None:
    """Copy a new block's state into a persistent mirror (template hit).

    Values only -- addresses must stay stable for the captured graph: params
    and buffers are ``copy_``-ed in place (cross-device D2D when needed), the
    wrapper ``params`` dicts keep referencing the same Parameters, and parked
    grads are ZEROED in place (never set to None: the replayed backward writes
    into the baked grad addresses).
    """
    with torch.no_grad():
        home_params = dict(home.named_parameters())
        for name, p in mirror.named_parameters():
            src = home_params.get(name)
            if src is None or src.shape != p.shape or src.dtype != p.dtype:
                raise RuntimeError(f"template drift at parameter {name!r} -- refusing to reuse the cached mirror")
            p.data.copy_(src.data)  # copy_ handles cross-device
            if p.grad is not None:
                p.grad.zero_()
        home_bufs = dict(home.named_buffers())
        for name, b in mirror.named_buffers():
            src = home_bufs.get(name)
            if src is None or src.shape != b.shape or src.dtype != b.dtype:
                raise RuntimeError(f"template drift at buffer {name!r} -- refusing to reuse the cached mirror")
            b.copy_(src)
        # plain-tensor attrs the forward/anchor reads (imatrix feeds the
        # recipe init-search and the dq in-forward search; its storage is
        # baked into the cached graph) -- sync IN PLACE
        home_mods = dict(home.named_modules())
        for name, mm in mirror.named_modules():
            hm = home_mods.get(name)
            if hm is None:
                continue
            for attr in ("weight_min", "weight_max"):
                # the quant grids MUST arrive with the fresh block: when the
                # recipe anchor bails (no recipe / ineligible layout) nothing
                # else refreshes them, and stale per-group grids from the
                # PREVIOUS block would silently quantize this one
                src = getattr(hm, attr, None)
                dst = getattr(mm, attr, None)
                if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
                    dst.copy_(src)
            for attr in ("imatrix", "act_max"):
                src = getattr(hm, attr, None)
                dst = getattr(mm, attr, None)
                if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
                    dst.copy_(src)
            # the deferred-anchor flag must say PENDING like a fresh mirror,
            # or the sharded anchor no-ops and stale grids get broadcast
            if getattr(hm, "_recipe_anchor_deferred", False):
                mm._recipe_anchor_deferred = True


def _shell_bind_home(cached_home: torch.nn.Module, block: torch.nn.Module) -> None:
    """Rebind the REAL block's tensors as views of the cached home set (T12).

    Inverted ownership: the cached home set owns the persistent storages the
    captured graphs read; the real block becomes a zero-storage shell. After
    the cached set is value-refreshed from the fresh block (sync_mirror_from_home
    on a hit, deepcopy on a miss), every downstream reader (unwrapper/pack/
    refit) that walks the REAL block sees the tuned values through views by
    construction -- the loop-end sync-back pass disappears, and the duplicate
    set on the home GPU (~2 GB on 27B-class blocks) is freed at rebind.

    Rebinding is safe HERE because the real block carries no captured graphs
    (they live on the cached replicas); views hold references to their
    storages, so an eviction that drops the entry cannot dangle a live shell.
    """
    cached_mods = dict(cached_home.named_modules())
    # frozen-margin layout parity FIRST: the anchor pinned (unregistered) the
    # min/max margin params on the CACHED set while the real block still
    # carries them registered -- pin the real block's too, or its
    # named_parameters walk below would see params the cached set no longer
    # has. The pin is a value-neutral layout op (unregister + constant 1.0):
    # no views are created before validation completes, so a drift raise
    # still leaves no partially VIEW-bound block behind.
    with torch.no_grad():
        for name, dm in block.named_modules():
            cm = cached_mods.get(name)
            if cm is None:
                continue
            if getattr(cm, "_tune_recipe_frozen_margins", False) and hasattr(dm, "_pin_margins_frozen"):
                dm._pin_margins_frozen()
    cached_params = dict(cached_home.named_parameters())
    cached_bufs = dict(cached_home.named_buffers())
    # validate EVERYTHING before creating any views: drift must raise before
    # a partially rebound block (some params views, some not) is left behind
    for name, p in block.named_parameters():
        src = cached_params.get(name)
        if src is None or src.shape != p.shape or src.dtype != p.dtype:
            raise RuntimeError(f"template drift at parameter {name!r} -- refusing to shell-bind the block")
    for name, b in block.named_buffers():
        src = cached_bufs.get(name)
        if src is None or src.shape != b.shape or src.dtype != b.dtype:
            raise RuntimeError(f"template drift at buffer {name!r} -- refusing to shell-bind the block")
    for name, dm in block.named_modules():
        cm = cached_mods.get(name)
        if cm is None:
            continue
        for attr in ("weight_min", "weight_max", "imatrix", "act_max"):
            src_t, dst_t = getattr(cm, attr, None), getattr(dm, attr, None)
            if isinstance(src_t, torch.Tensor) and isinstance(dst_t, torch.Tensor) and src_t.shape != dst_t.shape:
                raise RuntimeError(f"template drift at {name!r}.{attr} -- refusing to shell-bind the block")
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.data = cached_params[name].data  # view: storage shared, Parameter object untouched
        for name, b in block.named_buffers():
            b.data = cached_bufs[name].data
        for name, dm in block.named_modules():
            cm = cached_mods.get(name)
            if cm is None:
                continue
            # plain-tensor attrs the anchor/unwrapper/pack read -- bind by reference
            for attr in ("weight_min", "weight_max", "imatrix", "act_max"):
                src_t, dst_t = getattr(cm, attr, None), getattr(dm, attr, None)
                if isinstance(src_t, torch.Tensor) and isinstance(dst_t, torch.Tensor):
                    setattr(dm, attr, src_t)
            if getattr(cm, "_tune_recipe", None) is not None:
                dm._tune_recipe = cm._tune_recipe
            if getattr(cm, "_tune_recipe_frozen_margins", False):
                dm._tune_recipe_frozen_margins = True
            dm._recipe_anchor_deferred = False  # grids arrive via the views


def evict_template_cache_for_free(min_free_bytes: int, devices=None, only_devices=None) -> None:
    """Evict LRU template-cache entries until the devices hold enough free VRAM.

    A MISS block builds a fresh replica set while cached entries from other
    templates stay resident -- on 24 GB cards the second template's build
    OOMs (observed: 23.24/23.58 GB on cuda:0 during the first full-attention
    block with the linear entry cached). Evicted entries drop their module/
    graph references; empty_cache on the affected devices returns the
    reserved segments so the check reflects reality.

    ``only_devices`` restricts eviction to entries whose replicas live on
    those devices: with ping-pong groups, an entry pinned to the OTHER
    group's GPUs does not compete with this build for VRAM and must survive
    (min_free_bytes is a PER-DEVICE requirement -- see _free()).
    """
    if not _GRAPH_TEMPLATE_CACHE:
        return
    if devices is None:
        devices = (
            [torch.device("cuda", i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        )
    devs = [d for d in devices if getattr(d, "type", "") == "cuda"]
    only = {str(d) for d in only_devices} if only_devices is not None else None

    def _free():
        return min(torch.cuda.mem_get_info(d)[0] for d in devs) if devs else 1 << 62

    def _entry_devices(entry):
        out = set()
        for rep in entry.get("replicas", []):
            p = next(rep.parameters(), None)
            if p is not None:
                out.add(str(p.device))
        return out

    freed_devs = set()
    while _GRAPH_TEMPLATE_CACHE and _free() < min_free_bytes:
        _key, entry = None, None
        for k in _GRAPH_TEMPLATE_CACHE.keys():  # oldest first
            e = _GRAPH_TEMPLATE_CACHE[k]
            if only is None or (_entry_devices(e) & only):
                _key, entry = k, e
                break
        if _key is None:
            break  # no eligible entry left -- cannot buy more VRAM on these devices
        _GRAPH_TEMPLATE_CACHE.pop(_key)
        freed_devs |= _entry_devices(entry)
        entry.clear()  # drop refs: mirrors, staging, graphs, statics
        logger.info(
            "[tune-ddp] graph template cache: evicted an entry for VRAM (free target %.2f GiB)",
            min_free_bytes / 2**30,
        )
    for d in freed_devs:
        try:
            dev = torch.device(d)
            torch.cuda.empty_cache(0 if dev.index is None else dev.index)
        except Exception:  # noqa: BLE001 - best-effort release
            pass


def template_cache_size() -> int:
    from auto_round import envs

    return int(getattr(envs, "AR_TUNE_GRAPH_TEMPLATE_CACHE", 2) or 0)


def _trim_template_cache() -> None:
    """Enforce the template-cache cap PER DDP GROUP.

    Captured graphs pin replicas on one group's physical GPUs: an entry is
    only ever reusable within its own group. A global LRU lets one group's
    template diversity (a 3rd template -- MTP/dense variants -- landing
    mostly on one group) evict the OTHER group's working set, so each group
    keeps its own newest ``template_cache_size()`` entries. With a single
    group this is identical to the old global cap.
    """
    cap = template_cache_size()
    if cap <= 0:
        return
    counts: dict = {}
    for key in list(_GRAPH_TEMPLATE_CACHE.keys())[::-1]:  # newest first
        group = _GRAPH_TEMPLATE_CACHE[key].get("group", ())
        counts[group] = counts.get(group, 0) + 1
        if counts[group] > cap:
            _GRAPH_TEMPLATE_CACHE.pop(key)
            logger.info(
                "[tune-ddp] graph template cache: evicted LRU entry for group %s (per-group cap %d)",
                list(group)[:1] or "<unknown>",
                cap,
            )


class ReplicaGroup:
    """Persistent mirrors of a wrapped block for the iteration loop."""

    def __init__(
        self, block, plan: DDPPlan, grad_transport: str = "fp32", staged_source=None, mirrors=None, home_replica=None
    ) -> None:
        global _PEER_ACCESS_LOGGED
        self.plan = plan
        self.grad_transport = grad_transport
        self.home = block  # the REAL block (sync source / copy-back target)
        self.tune_home = home_replica if home_replica is not None else block  # module the loop tunes
        self.mirrors: List[torch.nn.Module] = []
        self.adopted = 0
        self.template_reused = mirrors is not None or home_replica is not None
        if not _PEER_ACCESS_LOGGED and any(d.type == "cuda" for d in plan.devices):
            _PEER_ACCESS_LOGGED = True
            _pairs = enable_peer_access(plan.devices)
            if _pairs:
                logger.info("[tune-ddp] P2P peer access enabled for %d pair(s)", len(_pairs))
            else:
                logger.info(
                    "[tune-ddp] P2P peer access NOT enabled (no accessible pairs); "
                    "cross-device copies may stage through host memory"
                )
        if mirrors is not None:
            # template cache hit: adopt the persistent mirror modules (values
            # are synced by the caller BEFORE the recipe anchor re-pins the
            # tuning grids); their captured graphs stay alive
            self.mirrors = list(mirrors)
        else:
            for dev in plan.devices[1:]:  # plan.devices[0] is the home by construction
                mirror = self._make_mirror(block, dev, staged_source)
                _relocate_params(mirror, dev)
                self.mirrors.append(mirror)
        self.replicas = [self.tune_home] + self.mirrors
        self.world = len(self.replicas)
        # persistent replica worker pool (built lazily on first run_threaded
        # when AR_TUNE_DDP_THREAD_POOL is on; None keeps spawn-per-call)
        self._pool = None
        logger.info(
            "[tune-ddp] world=%d devices=%s adopted=%d/%d",
            self.world,
            [str(d) for d in plan.devices],
            self.adopted,
            max(0, self.world - 1),
        )

    def _make_mirror(self, block, dev, staged_source):
        """Mirror the block onto ``dev``, adopting prefetched weights when sane.

        Fast path (CUDA): ``torch.nn.parallel.replicate`` + a repair walk --
        parameters/buffers are broadcast per device in coalesced chunks (no
        whole-block 2x duplication on the home GPU, faster than a Python
        deepcopy), then :func:`_repair_replica` re-leafs the broadcast
        tensors into trainable Parameters and clones the attrs replicate
        shares with home. With staged copies + pristine weights, the big
        checkpoint weights are swapped to empty for the broadcast and
        re-materialized from the device-local staged copies (the cross-GPU
        traffic then carries only the small tuning state). Falls back to the
        plain deepcopy paths on CPU or any replicate failure (mirror
        correctness > speed).
        """
        global _STAGED_MISS_LOGGED, _MIRROR_PATH_LOGGED
        staged = None
        prefix = ""
        if staged_source is not None:
            streamer, prefix = staged_source
            try:
                staged = streamer.staged_replica_tensors(prefix, dev)
            except Exception as e:  # noqa: BLE001 - fall back to replication
                logger.debug("[tune-ddp] staged_replica_tensors(%s, %s) raised %r", prefix, dev, e)
            if staged is None:
                if not _STAGED_MISS_LOGGED:
                    _STAGED_MISS_LOGGED = True
                    status = getattr(streamer, "staged_replica_status", lambda _p: [])(prefix)
                    logger.info(
                        "[tune-ddp] staged-replica adoption miss for %s on %s "
                        "(devices with complete replicas: %s); using %s instead",
                        prefix,
                        dev,
                        status or "none",
                        "deepcopy" if not _MIRROR_PATH_LOGGED else "mirror build",
                    )
        adopt = bool(staged) and getattr(block, "_stream_weights_pristine", False)
        if dev.type == "cuda":
            try:
                mirror = self._replicate_mirror(block, dev, staged if adopt else None, prefix)
                if adopt:
                    self.adopted += 1
                if not _MIRROR_PATH_LOGGED:
                    _MIRROR_PATH_LOGGED = True
                    logger.info("[tune-ddp] mirror path: replicate+repair engaged (first mirror on %s)", dev)
                return mirror
            except Exception as e:  # noqa: BLE001 - deepcopy fallback below
                logger.info("[tune-ddp] replicate mirror onto %s failed (%r); using deepcopy", dev, e)
        if not adopt:
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

    @staticmethod
    def _replicate_mirror(block, dev, staged, prefix):
        """Build one mirror via replicate+repair (+ optional staged adoption)."""
        from torch.nn.parallel import replicate as _torch_replicate

        swap = []
        try:
            if staged is not None:
                # adoption: the big checkpoint weights ride as empty tensors
                # so the broadcast carries only the tuning state; the home
                # pointers are restored right after (the saved references
                # keep the storages alive)
                for _n, p in block.named_parameters():
                    if ".orig_layer." in _n:
                        swap.append((p, p.data))
                        p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            # devices[0] must be the tensors' SOURCE device (broadcast goes
            # FROM it; replicas[0] reuses the originals -- no home duplicate)
            _src = next(block.parameters()).device
            if _src == dev:  # pragma: no cover - mirrors never share the home
                raise RuntimeError("mirror device equals the home device")
            replica = _torch_replicate(block, [_src, dev])[1]
        finally:
            for p, data in swap:
                p.data = data
        _repair_replica(block, replica, dev)
        if staged is not None:
            for name, param in replica.named_parameters():
                if ".orig_layer." not in name:
                    continue
                key = (
                    (prefix + "." + name.replace(".orig_layer.", ".")) if prefix else name.replace(".orig_layer.", ".")
                )
                if key in staged:
                    param.data = staged[key].to(param.device)
        return replica

    def round_params(self) -> List[List[torch.nn.Parameter]]:
        out = []
        for rep in self.replicas:
            ps = []
            for _n, m in rep.named_modules():
                if hasattr(m, "orig_layer") and "v" in getattr(m, "params", {}):
                    ps.append(m.params["v"])
            out.append(ps)
        return out

    def sync_grads(
        self, params_per_replica: List[List[torch.nn.Parameter]], prof=None, sign_exchange: bool = False
    ) -> None:
        """All-reduce v-gradients so every replica holds the identical average.

        ``sign_exchange=True`` (caller-gated to pure sign-SGD, i.e. no
        momentum) exchanges int8 signs instead of averaged values -- bitwise
        identical across ranks, at least as faithful to the fp32 mean, and
        the all-gather wire shrinks 4x (see sign_exchange_allreduce).
        ``prof`` (optional tune profiler) splits the work into bufprep /
        exchange / writeback stages so the profile line shows where the
        allreduce time actually goes.
        """
        from auto_round import envs
        from auto_round.utils.tune_profile import stage as _stage

        with _stage(prof, "bufprep"):
            bufs = _param_grad_buffers(params_per_replica)
        if any(b is None for b in bufs):
            # a replica without gradients means the collected params are not
            # the ones the forward/backward touched -- the tune would silently
            # degrade to single-shard updates
            logger.warning(
                "[tune-ddp] sync_grads skipped: %d/%d replica(s) have no gradients on their collected params",
                sum(1 for b in bufs if b is None),
                len(bufs),
            )
            return
        with _stage(prof, "exchange"):
            if use_one_shot(self.world, self.grad_transport):
                global _ONESHOT_LOGGED
                if not _ONESHOT_LOGGED:
                    _ONESHOT_LOGGED = True
                    logger.info(
                        "[tune-ddp] one-shot allreduce engaged: world=%d transport=%s", self.world, self.grad_transport
                    )
                one_shot_allreduce(bufs, scale=1.0 / self.world, transport=self.grad_transport, prof=prof)
            else:
                if sign_exchange and envs.AR_TUNE_DDP_SIGN_EXCHANGE:
                    global _SIGN_LOGGED
                    if not _SIGN_LOGGED:
                        _SIGN_LOGGED = True
                        logger.info(
                            "[tune-ddp] sign-cast exchange engaged: world=%d transport=%s (int8 sign allgather)",
                            self.world,
                            self.grad_transport,
                        )
                    sign_exchange_allreduce(bufs, transport=self.grad_transport)
                else:
                    halving_doubling_allreduce(bufs, scale=1.0 / self.world, transport=self.grad_transport)
        with _stage(prof, "writeback"):
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

        Uses the persistent worker pool when AR_TUNE_DDP_THREAD_POOL is on
        (default): the tune loop calls this twice per iteration, and pool
        reuse removes the per-call thread spawn/join overhead. Falls back to
        ``run_threaded_spawn`` when the pool is disabled, not yet built, or
        the callable count does not match the pool width.

        Exceptions propagate: the first failure (by thread index) is re-raised
        in the joining thread -- a swallowed worker failure would otherwise
        leave missing shard losses/grads and corrupt the step silently.
        """
        pool = getattr(self, "_pool", None)
        if pool is None:
            from auto_round import envs

            if envs.AR_TUNE_DDP_THREAD_POOL:
                # sized by the replica width, not the first call's fn count:
                # narrower callers (e.g. a padded-in-later stage) must not
                # permanently pin the pool to a wrong size
                self._pool = pool = ReplicaThreadPool(getattr(self, "world", len(fns)))
        if pool is not None and len(fns) == pool.n:
            pool.run(fns)
        else:
            run_threaded_spawn(fns)

    def teardown(self) -> None:
        pool, self._pool = getattr(self, "_pool", None), None
        if pool is not None:
            pool.shutdown()
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
