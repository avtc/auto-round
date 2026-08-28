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


def _reset_mirror_stats(mirrors: List[torch.nn.Module]) -> None:
    """Zero hook-written stats on mirrors that will be REUSED (cached).

    Ephemeral mirrors die after their pass, so merged stats vanish with them.
    A cached mirror survives into the next pass -- without a reset its stale
    stats would be merged AGAIN (double counting).
    """
    for mirror in mirrors:
        if mirror is None:
            continue
        for _n, m_mod in mirror.named_modules():
            for attr in _MERGEABLE_STATS:
                m_val = getattr(m_mod, attr, None)
                if torch.is_tensor(m_val):
                    setattr(m_mod, attr, torch.zeros_like(m_val))


def sharded_nograd_forward(
    runner,
    block,
    inputs,
    input_others: dict,
    out_device: torch.device,
    devices: List[torch.device],
    sample_count: Optional[int] = None,
    merge_stats: bool = False,
    mirror_cache: Optional[dict] = None,
):
    """Parallelize a no-grad collection forward across ``devices``.

    The collection passes (reference outputs, quantized-output cascade) are
    plain forwards over the whole sample pool on one GPU while the mirrors
    idle. Here the pool is split into equal disjoint shards; a copy of the
    block on each device forwards its shard in a parallel thread; per-shard
    outputs are parked on ``out_device`` and concatenated in order.
    Bit-identical to the serial pass: rows are sample-independent and the
    module copies carry identical weights. Falls back to the serial runner
    call when the pool is not divisible or fewer than two devices are given.

    ``mirror_cache`` (caller-owned {device: module} dict) keeps the mirrors
    ALIVE across calls -- sequential collection passes over the same block
    state (fp-input hooks, q-input hooks) build them once instead of per
    pass. Mirrors are built via replicate+repair on CUDA (deepcopy fallback);
    merged hook stats are reset on cached mirrors so a later pass cannot
    double-count. Without a cache, mirrors are dropped (freed) afterwards.
    Returned pieces stay on the device that computed them (distributed pool);
    consumers either read shard-locally or use device-safe cats.
    """
    world = len(devices)
    n = sample_count if sample_count is not None else (len(inputs) if isinstance(inputs, list) else 0)
    if world < 2 or n < world or n % world != 0:
        return runner(block, inputs, input_others, cache_device=out_device)
    shard = n // world
    shards = [list(range(r * shard, (r + 1) * shard)) for r in range(world)]
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
            m = mirror_cache.get(dev) if mirror_cache is not None else None
            if m is None:
                try:  # replicate+repair: coalesced broadcast, no 2x home spike
                    m = ReplicaGroup._replicate_mirror(block, dev, None, "")
                except Exception:  # noqa: BLE001 - CPU / replicate failure
                    m = copy.deepcopy(block).to(dev)
                _relocate_params(m, dev)
                if mirror_cache is not None:
                    mirror_cache[dev] = m
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
    parts: List = [None] * world

    def _run(r):
        rep = reps[r]
        dev_r = _block_device(rep)
        # outputs stay ON the replica that computed them: the per-sample list
        # returned below is a distributed pool (device r owns shard r's rows)
        if dev_r.type == "cuda":
            with torch.cuda.device(dev_r):
                parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)
        else:
            parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)

    runner_group = ReplicaGroup.__new__(ReplicaGroup)  # reuse only the thread runner
    runner_group.run_threaded([lambda r=r: _run(r) for r in range(world)])
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
    if merge_stats:
        _merge_mirror_stats(block, mirrors)
        if mirror_cache is not None:
            _reset_mirror_stats(mirrors)
    if mirror_cache is None:
        mirrors.clear()  # drop mirror refs; the caching allocator reclaims them
        reps.clear()
    return pieces


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
        staged = None
        prefix = ""
        if staged_source is not None:
            streamer, prefix = staged_source
            try:
                staged = streamer.staged_replica_tensors(prefix, dev)
            except Exception:
                staged = None
        adopt = bool(staged) and getattr(block, "_stream_weights_pristine", False)
        if dev.type == "cuda":
            try:
                mirror = self._replicate_mirror(block, dev, staged if adopt else None, prefix)
                if adopt:
                    self.adopted += 1
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

    def sync_tuning_params(self) -> None:
        """Copy the home's post-tune wrapper state onto every mirror.

        After the tune the home may have diverged from the mirrors: the
        best-iteration parameter restore and post-tune passes (e.g. scale
        refit) write home-side tuning params. Mirrors reused for the
        post-tune sharded collection must reflect the FINAL state. Weights
        are untouched by the tune and stay identical; only wrapper params
        (``_parameters``, the ``params`` dict) and small plain-attr tensors
        (scale/zp caches) are copied.
        """
        home_mods = dict(self.home.named_modules())
        for mirror in self.mirrors:
            for name, m_mod in mirror.named_modules():
                h_mod = home_mods.get(name)
                if h_mod is None or not hasattr(h_mod, "orig_layer"):
                    continue
                for key, hp in h_mod._parameters.items():
                    if hp is None:
                        continue
                    mp = m_mod._parameters.get(key)
                    if mp is not None and mp.shape == hp.shape:
                        mp.data.copy_(hp.data.to(mp.device))
                for key, hv in getattr(h_mod, "params", {}).items():
                    if not torch.is_tensor(hv):
                        continue
                    mv = getattr(m_mod, "params", {}).get(key)
                    if torch.is_tensor(mv) and mv.shape == hv.shape:
                        mv.data.copy_(hv.data.to(mv.device))
                for attr, hv in h_mod.__dict__.items():
                    if attr.startswith("_") or not torch.is_tensor(hv):
                        continue
                    mv = m_mod.__dict__.get(attr)
                    if torch.is_tensor(mv) and mv.shape == hv.shape:
                        mv.data.copy_(hv.data.to(mv.device))

    def teardown(self) -> None:
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
