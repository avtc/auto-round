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

"""Per-chunk device placement for block calibration data (``AR_CALIBRATION_DATA_DEVICE``).

Policy (user rulings, Sep-13):

* primary-first: when the block input/output pools fit the primary cache device
  (free minus the estimated forward working set minus a flat 0.5 GiB allocator
  reserve), every chunk stays there -- byte-identical to today's behavior, zero
  peer traffic. The working-set estimate reuses the same per-block accounting
  the streaming branch's block placement used (``estimate_tuning_block_mem``:
  ``layer_activation_memory + additional_memory``), computed from each block's
  own module tree, never from the previous block.
* otherwise shard: chunks spread over the candidate devices proportionally to
  free capacity (the primary included), and are consumed chunk-wise by the
  existing per-batch ``.to(compute_device)`` in the block forward path, so only
  the at-rest placement changes.
* ``cpu`` mode parks the pools on host RAM explicitly (forwards still run on
  the GPUs) -- the pool-scoped equivalent of ``low_gpu_mem_usage`` without its
  other side effects. ``auto`` never falls back to CPU silently: when sharding
  cannot hold the pools either, ``None`` is returned so the caller keeps
  today's behavior and the genuine OOM fires (with the census diagnostics).

``AR_CALIBRATION_DATA_DEVICE``: ``auto`` (default) | ``off`` | ``cpu`` |
explicit csv (``cuda:1,cuda:2``). Pinned by the same-name CLI argument.
"""

from typing import Callable, List, Optional, Sequence

import torch

from auto_round.logger import logger


class PoolPlacement:
    """Deterministic per-chunk device plan for one calibration output pool."""

    def __init__(self, devices: Sequence[str], capacities: Sequence[int], n_chunks: int):
        self.devices = [str(d) for d in devices]
        self.capacities = [int(c) for c in capacities]
        self.plan = _spread_plan(self.devices, self.capacities, n_chunks)

    def device_for_index(self, i: int) -> str:
        """Device for output chunk ``i`` (deterministic, wraps for over-long pools)."""
        return self.plan[i % len(self.plan)]

    def counts(self) -> dict:
        """Chunks per device (diagnostics)."""
        out: dict = {}
        for d in self.plan:
            out[d] = out.get(d, 0) + 1
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"PoolPlacement({self.counts()})"


def _spread_plan(devices: Sequence[str], capacities: Sequence[int], n_chunks: int) -> List[str]:
    """Largest-remainder proportional split, interleaved across devices.

    Interleaving (round-robin order of the assigned slots) keeps consecutive
    chunks on different devices so transient per-chunk peaks never stack on a
    single peer.
    """
    total = sum(capacities)
    if total <= 0 or n_chunks <= 0 or not devices:
        return list(devices)
    quotas = [c * n_chunks / total for c in capacities]
    counts = [int(q) for q in quotas]
    remainders = sorted(range(len(devices)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in range(n_chunks - sum(counts)):
        counts[remainders[i % len(remainders)]] += 1
    # interleave: one slot per device in turn, skipping exhausted devices
    plan: List[str] = []
    cursors = [0] * len(devices)
    while any(cursors[i] < counts[i] for i in range(len(devices))):
        for i in range(len(devices)):
            if cursors[i] < counts[i]:
                plan.append(devices[i])
                cursors[i] += 1
    return plan


def _bytes_by_device(obj) -> tuple:
    """(device_str -> bytes, total bytes) for tensor leaves of a nested pool object."""
    per_device: dict = {}

    def _walk(o):
        if isinstance(o, torch.Tensor):
            key = str(o.device)
            per_device[key] = per_device.get(key, 0) + int(o.numel()) * o.element_size()
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(obj)
    return per_device, sum(per_device.values())


def _short_device_key(dev: str) -> str:
    """'cuda:0' -> '0', 'cpu' -> 'cpu' (memory-monitor grammar)."""
    d = str(dev)
    if d.startswith("cuda:"):
        return d.split(":", 1)[1]
    return d


def calib_data_line(inputs, aux, plan, outputs_bytes: int, n_chunks: int, primary: str) -> str:
    """One-line calibration-data summary in the memory-monitor format:

    ``'input': 8.12GB, 'output': 8.12GB, 'aux': 0.12GB, 'per_device': {'0': 2.03GB, '1': 2.03GB, 'cpu': 0.12GB}``

    ``inputs`` is the list of live input pool objects (fp and q chains merged --
    the dual chain is upstream's qon architecture; during collection both pools
    coexist, which is the peak the placement accounts for). Input/aux devices
    are ground truth (walked from the tensors); output bytes are this block's
    planned placement (the plan, or all-primary under today's behavior).
    Per-device totals combine parked + planned.
    """
    per_device: dict = {}
    input_total = 0
    for obj in inputs:
        by_dev, total = _bytes_by_device(obj)
        input_total += total
        for d, b in by_dev.items():
            per_device[d] = per_device.get(d, 0) + b
    by_dev, aux_total = _bytes_by_device(aux)
    for d, b in by_dev.items():
        per_device[d] = per_device.get(d, 0) + b
    if outputs_bytes > 0:
        if plan is not None:
            per_chunk = outputs_bytes / max(n_chunks, 1)
            for dev, cnt in plan.counts().items():
                per_device[dev] = per_device.get(dev, 0) + int(cnt * per_chunk)
        else:
            per_device[str(primary)] = per_device.get(str(primary), 0) + outputs_bytes
    devs = ", ".join(
        f"'{_short_device_key(d)}': {b / 2**30:.2f}GB" for d, b in sorted(per_device.items(), key=lambda kv: -kv[1])
    )
    if plan is None:
        place = "none->primary"
    elif len(plan.devices) == 1:
        place = "single"
    else:
        place = "spread"
    return (
        f"'input': {input_total / 2**30:.2f}GB, 'output': {outputs_bytes / 2**30:.2f}GB, "
        f"'aux': {aux_total / 2**30:.2f}GB, 'plan': '{place}', 'per_device': {{{devs}}}"
    )


def _tensor_bytes(obj) -> int:
    """Total bytes of tensor leaves in a nested list/tuple/dict pool object."""
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _pool_chunk_count(obj) -> int:
    """Number of per-sample chunks in a pool object (length of first list found)."""
    if isinstance(obj, (list, tuple)):
        tensor_like = [v for v in obj if isinstance(v, torch.Tensor)]
        if tensor_like:
            return len(obj)
        for v in obj:
            n = _pool_chunk_count(v)
            if n:
                return n
        return 0
    if isinstance(obj, dict):
        for v in obj.values():
            n = _pool_chunk_count(v)
            if n:
                return n
    return 0


def _move_pool_to(obj, target: str):
    """Recursively move a pool's tensors onto ``target`` in place (skip locals)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(target) if str(obj.device) != target else obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _move_pool_to(obj[i], target)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_pool_to(v, target) for v in obj)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _move_pool_to(obj[k], target)
        return obj
    return obj


def consolidate_pool_onto(objs, target: str, block, batch_size: int, reserved_bytes: int = 0) -> str:
    """Consolidate the incoming calibration pools onto the compute device.

    Returns ``'local'`` (already on target), ``'consolidated'`` (moved in one
    bulk pass), or ``'spread'`` (the pools do not fit next to the block's
    working set -- consumers keep fetching batch-by-batch instead). This is the
    fits-home rung of the ladder: fit-primary / fit-home -> single device,
    oversized pool -> spread + per-batch gather.
    """
    if not str(target).startswith("cuda"):
        return "spread"
    from auto_round.algorithms.quantization.search_shard import _probe_usable_bytes

    total = sum(_tensor_bytes(o) for o in objs if o is not None)
    if total <= 0:
        return "local"
    free = _probe_usable_bytes(target)
    if free is None:
        return "spread"
    need = placement_need_bytes(block, objs[0], batch_size)
    # ``reserved_bytes``: output pools that will share the target this block.
    # The first GPU validation showed the inputs-only budget was optimistic --
    # consolidating 4GiB of inputs onto the busiest device while the outputs
    # were also headed there produced exactly the OOM the policy exists to
    # prevent. Consolidate only when BOTH fit beside the working set.
    if free - need - reserved_bytes < total:
        return "spread"
    devices = set()

    def _collect(o):
        if isinstance(o, torch.Tensor):
            devices.add(str(o.device))
        elif isinstance(o, dict):
            for v in o.values():
                _collect(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _collect(v)

    for o in objs:
        if o is not None:
            _collect(o)
    if devices <= {target}:
        return "local"
    for o in objs:
        if o is not None:
            _move_pool_to(o, target)
    return "consolidated"


def resolve_pool_placement(
    pool_bytes: int,
    n_chunks: int,
    primary: str,
    need_bytes: int,
    candidate_devices: Sequence[str],
    free_probe: Callable[[str], Optional[int]],
    mode: str = "auto",
) -> Optional[PoolPlacement]:
    """Decide per-chunk output placement for a calibration pool.

    ``need_bytes`` is the primary's own forward working-set demand (estimated
    per block via :func:`placement_need_bytes`); the pool must fit in
    ``free(primary) - need_bytes`` to stay primary-resident.

    Returns ``None`` when the caller should keep today's behavior (policy off,
    CPU-parked lane, or sharding cannot hold the pool -- the genuine OOM then
    fires loudly per the no-silent-CPU ruling).
    """
    mode = (mode or "auto").strip().lower()
    if mode == "off":
        return None
    if mode == "cpu":
        return PoolPlacement(["cpu"], [1], n_chunks)  # explicit host-RAM parking
    if str(primary).startswith("cpu"):
        logger.debug("[calib-data-device] resolve: none (cpu primary)")
        return None  # low_gpu_mem_usage (or a CPU lane) owns placement here
    forced = None
    if mode not in ("", "auto"):
        forced = [d.strip() for d in mode.split(",") if d.strip()]

    candidates: List[str] = []
    if forced is not None:
        candidates = forced
    else:
        seen = {str(primary)}
        candidates = [str(primary)]
        for d in candidate_devices:
            key = str(d)
            if key.startswith("cpu") or key in seen:
                continue
            seen.add(key)
            candidates.append(key)

    probed = [(d, free_probe(d)) for d in candidates]
    usable = [(d, f) for d, f in probed if f is not None and f > 0]
    if not usable:
        logger.debug("[calib-data-device] resolve: none (no usable device among %s)", candidates)
        return None

    primary_free = dict(usable).get(str(primary), 0)
    if primary_free - need_bytes >= pool_bytes:
        # primary-first: identical to today's behavior, zero peer traffic
        return PoolPlacement([str(primary)], [max(primary_free - need_bytes, 1)], n_chunks)

    total = sum(f for _, f in usable)
    if total - need_bytes < pool_bytes:
        logger.debug(
            "[calib-data-device] resolve: none (insufficient: total %.2fGiB - need %.2fGiB < pool %.2fGiB)",
            total / 2**30,
            need_bytes / 2**30,
            pool_bytes / 2**30,
        )
        return None  # sharding cannot help either: keep today's behavior, real OOM fires

    # capacity-aware sharding: subtract the working-set need from the primary only
    devices = [d for d, _ in usable]
    capacities = [max(f - need_bytes, 1) if d == str(primary) else f for d, f in usable]
    return PoolPlacement(devices, capacities, n_chunks)


_RESERVE_BYTES = int(0.5 * 2**30)  # flat allocator reserve (24 GiB-class cards)


def placement_need_bytes(block, pool, batch_size: int) -> int:
    """Primary working-set need: the streaming branch's per-block accounting.

    Ports the card-0 accounting of ``set_auto_device_map_for_block_with_tuning``
    (mapped-placement era): ``layer_activation_memory + additional_memory`` from
    ``estimate_tuning_block_mem``, computed from THIS block's module tree (never
    the previous block's measurements), plus a flat 0.5 GiB allocator reserve.
    """
    if block is None:
        return _RESERVE_BYTES
    try:
        from auto_round.utils.device import estimate_tuning_block_mem

        _, layer_activation_memory, _, additional_memory = estimate_tuning_block_mem(block, pool, batch_size)
        return int((layer_activation_memory + additional_memory) * 2**30) + _RESERVE_BYTES
    except Exception as e:  # pragma: no cover - placement must never break quantization
        logger.warning(
            "[calib-data-device] working-set estimate failed (%s); using %.1fGiB reserve", e, _RESERVE_BYTES / 2**30
        )
        return _RESERVE_BYTES


def resolve_placement_for_pool(
    pool,
    chains: int,
    primary: str,
    candidate_devices: Sequence[str],
    block=None,
    batch_size: int = 8,
    mode: str = "auto",
) -> Optional[PoolPlacement]:
    """Resolve placement from a live pool object (orchestrator entry point).

    ``primary`` must be the lane's actual cache device: a CPU primary
    (``low_gpu_mem_usage``) deactivates the policy so the two mechanisms never
    fight. ``chains`` doubles the byte demand when a second (quantized-input)
    pool of the same size will also be produced for the block. ``block`` feeds
    the per-block working-set estimate.
    """
    from auto_round.algorithms.quantization.search_shard import _probe_usable_bytes

    pool_bytes = _tensor_bytes(pool) * max(int(chains), 1)
    n_chunks = _pool_chunk_count(pool)
    if n_chunks <= 0 or pool_bytes <= 0:
        return None
    try:
        plan = resolve_pool_placement(
            pool_bytes,
            n_chunks,
            primary,
            placement_need_bytes(block, pool, batch_size),
            candidate_devices,
            _probe_usable_bytes,
            mode=mode,
        )
    except Exception as e:  # pragma: no cover - placement must never break quantization
        logger.warning("[calib-data-device] placement resolve failed (%s); keeping single-device behavior", e)
        return None
    return plan
