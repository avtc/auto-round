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
import queue
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
_OVERLAP_LOGGED = False


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
                setattr(rmod, attr, copy.deepcopy(hval))
        # ── wrapper aliasing dict: re-key to the NEW parameters ──────────
        if hasattr(hmod, "orig_layer") and isinstance(getattr(hmod, "params", None), dict):
            fixed = {}
            for k, v in hmod.params.items():
                if isinstance(v, torch.nn.Parameter) and k in rmod._parameters:
                    fixed[k] = rmod._parameters[k]
                elif torch.is_tensor(v):
                    fixed[k] = v.detach().clone().to(dev)
                elif isinstance(v, (list, dict, set)):
                    fixed[k] = copy.deepcopy(v)
                else:
                    fixed[k] = v
            rmod.params = fixed


_PEER_ACCESS_LOGGED = False


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


class _DelayedBestTracker:
    """Delayed-loss best-params selection for the DDP tune loop.

    The per-iteration host wait on ``loss.item()`` drains the whole GPU
    chain of the iteration that just ran; deferring the read to the next
    iteration overlaps that drain with the freshly enqueued forward instead
    of stalling the pipeline between iterations. Selection semantics are
    IDENTICAL to immediate selection -- the same (loss, pre-step params)
    pairs are compared with the same strict-less rule; the comparison and
    the ``.item()`` read simply happen one iteration later on the host.

    Snapshots alternate between two slots. ``resolve()`` promotes the
    resolved slot's snapshot by IDENTITY; when the alternation is about to
    overwrite the still-best slot, that snapshot is first cloned out
    (device-local copy of the nested dict) -- zero copies while improvements
    keep arriving.
    """

    def __init__(self) -> None:
        self._slots: List[Optional[dict]] = [None, None]
        self._pending: Optional[Tuple[int, Sequence, int]] = None  # (slot, losses, iter)
        self._next_slot = 0
        self.best_loss: Optional[float] = None
        self.best_params: Optional[dict] = None
        self.best_iter: Optional[int] = None
        self.last_promoted = False

    def reset(self) -> None:
        """Drop pending iteration, ring slots, and promotion (grid re-swap)."""
        self.__init__()

    @staticmethod
    def _copy_snapshot(snapshot: dict) -> dict:
        return {
            mod: {key: (val.clone() if torch.is_tensor(val) else val) for key, val in entry.items()}
            for mod, entry in snapshot.items()
        }

    def stage(self, snapshot: Optional[dict], losses: Sequence, iter_index: int) -> None:
        """Park iteration ``iter_index``'s pre-step snapshot and loss tensors."""
        slot = self._next_slot % 2
        self._next_slot += 1
        if self.best_params is not None and self._slots[slot] is self.best_params:
            self.best_params = self._copy_snapshot(self.best_params)
        self._slots[slot] = snapshot
        self._pending = (slot, losses, iter_index)

    def resolve(self) -> Optional[float]:
        """Drain the parked iteration: return its loss, promote if best.

        Returns ``None`` when nothing is pending (loop start / post-drain).
        ``snapshot=None`` iterations (loss-delay-only mode) never promote.
        """
        if self._pending is None:
            return None
        slot, losses, iter_index = self._pending
        self._pending = None
        self.last_promoted = False
        loss = float(sum(l.item() for l in losses if l is not None))
        snapshot = self._slots[slot]
        if snapshot is not None and (self.best_loss is None or loss < self.best_loss):
            self.best_loss = loss
            self.best_params = snapshot
            self.best_iter = iter_index
            self.last_promoted = True
        return loss


class ReplicaGroup:
    """Persistent mirrors of a wrapped block for the iteration loop."""

    def __init__(self, block, plan: DDPPlan, grad_transport: str = "fp32", staged_source=None) -> None:
        global _PEER_ACCESS_LOGGED
        self.plan = plan
        self.grad_transport = grad_transport
        self.home = block
        self.mirrors: List[torch.nn.Module] = []
        self.adopted = 0
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
        for dev in plan.devices[1:]:  # plan.devices[0] is the home by construction
            mirror = self._make_mirror(block, dev, staged_source)
            _relocate_params(mirror, dev)
            self.mirrors.append(mirror)
        self.replicas = [block] + self.mirrors
        self.world = len(self.replicas)
        # overlapped-exchange state (built lazily on first sync_grads; None
        # keeps the classic sequential exchange)
        self._overlap = None
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

    def _init_overlap(self, params_per_replica) -> bool:
        """Build hook + staging state for the overlapped gradient exchange.

        For every replica r and bucket k (one bucket per collected v-param):
        a ``register_post_accumulate_grad_hook`` fires mid-backward, encodes
        the bucket on replica r's SIDE stream and copies the payload into
        preallocated staging slots on every device d (including a self-slot
        at [d==r], so every rank later sums the identical transport-rounded
        bits). Copies into device d go onto d's side stream -- never the
        default stream, which is busy running backward: this is what lets
        the wire work overlap the remaining layers instead of queueing
        behind the whole drain.

        Stream ordering contract:
        - side_r.wait_stream(default_r) inside the hook waits exactly up to
          this bucket's accumulation (an event, not the whole backward);
        - cross-device copy_ event-syncs the SOURCE device's current stream
          (side_r, where the encode ran) onto the destination side stream;
        - sync_grads' tail makes every default stream wait its side stream
          before the canonical reduce, and the optimizer step then runs on
          the default stream -- fully ordered, nothing asynchronous leaks.
        """
        from auto_round import envs

        if envs.AR_DISABLE_OVERLAP_EXCHANGE:
            return False
        if self.world < 2 or not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            return False
        devs = []
        for params in params_per_replica:
            if not params:
                return False
            devs.append(params[0].device)
        if any(d.type != "cuda" for d in devs):
            return False
        n_buckets = min(len(params) for params in params_per_replica)
        if n_buckets < 1:
            return False
        wire_dt = {"int8": torch.int8, "bf16": torch.bfloat16}.get(self.grad_transport, torch.float32)
        side = [torch.cuda.Stream(device=d) for d in devs]
        # staging[d][r][k]: replica r's encoded bucket k as received on device d
        staging = [[None] * self.world for _ in range(self.world)]
        metas = [[None] * self.world for _ in range(self.world)]
        for d in range(self.world):
            for r in range(self.world):
                staging[d][r] = [torch.empty(p.numel(), dtype=wire_dt, device=devs[d]) for p in params_per_replica[r]]
                metas[d][r] = [torch.empty((), dtype=torch.float32, device=devs[d]) for _ in range(n_buckets)]
        acc = [
            torch.empty(max(p.numel() for p in params), dtype=torch.float32, device=d)
            for d, params in zip(devs, params_per_replica)
        ]
        hooks = []
        for r in range(self.world):
            for k in range(n_buckets):
                hooks.append(
                    params_per_replica[r][k].register_post_accumulate_grad_hook(
                        self._make_bucket_hook(r, k, devs, side, staging, metas)
                    )
                )
        self._overlap = {
            "devs": devs,
            "side": side,
            "staging": staging,
            "metas": metas,
            "acc": acc,
            "hooks": hooks,
            "n_buckets": n_buckets,
        }
        global _OVERLAP_LOGGED
        if not _OVERLAP_LOGGED:
            _OVERLAP_LOGGED = True
            logger.info(
                "[tune-ddp] overlapped gradient exchange engaged: world=%d transport=%s buckets=%d",
                self.world,
                self.grad_transport,
                n_buckets,
            )
        return True

    def _make_bucket_hook(self, r, k, devs, side, staging, metas):
        transport = self.grad_transport
        world = self.world
        dev_r = devs[r]

        def hook(param):
            # post-accumulate hooks receive the PARAM (its .grad is populated);
            # encode the gradient, flattened to match the [numel] staging slots
            # (a 2-D param payload would hit copy_ broadcasting, not a flat copy)
            g = param.grad
            if g is None:  # defensive: nothing accumulated for this bucket
                return
            cur = torch.cuda.current_stream(dev_r)
            with torch.cuda.device(dev_r):
                side[r].wait_stream(cur)
                with torch.cuda.stream(side[r]):
                    payload, meta = _encode_transport(g.detach().reshape(-1), transport)
                    for d in range(world):
                        with torch.cuda.device(devs[d]), torch.cuda.stream(side[d]):
                            staging[d][r][k].copy_(payload, non_blocking=True)
                            if meta is not None:
                                metas[d][r][k].copy_(meta, non_blocking=True)

        return hook

    def _finish_overlap(self, params_per_replica, prof) -> None:
        """Wait the side streams and run the canonical per-bucket reduction."""
        from auto_round.utils.tune_profile import stage as _stage

        ov = self._overlap
        # ov_wait: how long the home default stream sits waiting the side
        # streams AFTER draining its own backward -- the true straggler+side
        # lag. ov_reduce: the canonical reduction kernels themselves.
        with _stage(prof, "ov_wait"):
            for d, dev in enumerate(ov["devs"]):
                with torch.cuda.device(dev):
                    torch.cuda.current_stream(dev).wait_stream(ov["side"][d])
        with _stage(prof, "ov_reduce"):
            for k in range(ov["n_buckets"]):
                for d, dev in enumerate(ov["devs"]):
                    p = params_per_replica[d][k]
                    if p.grad is None:
                        # a replica without gradients means the backward never
                        # touched this bucket -- its staging slot holds the
                        # PREVIOUS iteration's payload and the tune would
                        # silently degrade; same loud contract as the
                        # sequential path
                        logger.warning(
                            "[tune-ddp] overlap exchange: replica %d bucket %d has no gradient; "
                            "left its local (un-averaged) value in place",
                            d,
                            k,
                        )
                        continue
                    with torch.cuda.device(dev):
                        total = _canonical_bucket_sum(
                            [row[k] for row in ov["staging"][d]],
                            [row[k] for row in ov["metas"][d]],
                            self.grad_transport,
                            torch.float32,
                        )
                        acc = ov["acc"][d][: p.numel()]
                        acc.copy_(total)
                        acc.mul_(1.0 / self.world)
                        p.grad.copy_(acc.view_as(p.grad))
            return

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

        if self._overlap is not None or self._init_overlap(params_per_replica):
            self._finish_overlap(params_per_replica, prof)
            return
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
        ov, self._overlap = self._overlap, None
        if ov is not None:
            for h in ov["hooks"]:
                h.remove()
        pool, self._pool = getattr(self, "_pool", None), None
        if pool is not None:
            pool.shutdown()
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
