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

"""Per-chunk output placement for block calibration pools (``AR_POOL_SHARD``).

Policy (user rulings, Sep-13):

* primary-first: when the pool fits the primary cache device (free minus a
  working-set margin), every chunk stays there -- byte-identical to today's
  behavior, zero peer traffic.
* otherwise shard: chunks are spread over the candidate devices proportionally
  to free capacity (the primary included), and consumed chunk-wise by the
  existing per-batch ``.to(compute_device)`` in the block forward path.
* never silently fall back to CPU: CPU parking is ``low_gpu_mem_usage``'s job.
  When sharding cannot help either, ``None`` is returned so the caller keeps
  today's behavior and the genuine OOM fires (with the census diagnostics).

``AR_POOL_SHARD``: ``auto`` (default) | ``off`` | explicit csv (``cuda:1,cuda:2``).
``AR_POOL_SHARD_MARGIN_GB``: primary headroom reserved for the forward working
set + fragmentation (default 6 GiB, census-calibrated: ~4.6 GiB linear-loop
working set + ~2.3 GiB reserved-unallocated observed at the block-2 wall).
"""

from typing import Callable, List, Optional, Sequence

import torch

from auto_round import envs
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


def _tensor_bytes(obj) -> int:
    """Total bytes of tensor leaves in a nested list/tuple/dict pool object."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
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


def resolve_pool_placement(
    pool_bytes: int,
    n_chunks: int,
    primary: str,
    margin_bytes: int,
    candidate_devices: Sequence[str],
    free_probe: Callable[[str], Optional[int]],
) -> Optional[PoolPlacement]:
    """Decide per-chunk output placement for a calibration pool.

    Returns ``None`` when the caller should keep today's behavior (policy off,
    CPU-parked lane, or sharding cannot hold the pool -- the genuine OOM then
    fires loudly per the no-silent-CPU ruling).
    """
    mode = (envs.AR_POOL_SHARD or "auto").strip().lower()
    if mode == "off":
        return None
    if str(primary).startswith("cpu"):
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
        return None

    primary_free = dict(usable).get(str(primary), 0)
    if primary_free - margin_bytes >= pool_bytes:
        # primary-first: identical to today's behavior, zero peer traffic
        return PoolPlacement([str(primary)], [max(primary_free - margin_bytes, 1)], n_chunks)

    total = sum(f for _, f in usable)
    if total - margin_bytes < pool_bytes:
        return None  # sharding cannot help either: keep today's behavior, real OOM fires

    # capacity-aware sharding: subtract the working-set margin from the primary only
    devices = [d for d, _ in usable]
    capacities = [max(f - margin_bytes, 1) if d == str(primary) else f for d, f in usable]
    return PoolPlacement(devices, capacities, n_chunks)


def pool_shard_margin_bytes() -> int:
    """Primary-headroom margin for the forward working set + fragmentation."""
    try:
        gb = float(getattr(envs, "AR_POOL_SHARD_MARGIN_GB", None) or 6.0)
    except (TypeError, ValueError):
        gb = 6.0
    return int(gb * 2**30)


_SHARD_ENGAGED_LOGGED = set()


def _log_engaged_once(plan: PoolPlacement, pool_bytes: int, primary: str) -> None:
    key = repr(plan)
    if key in _SHARD_ENGAGED_LOGGED:
        return
    _SHARD_ENGAGED_LOGGED.add(key)
    logger.info(
        "[pool-shard] output pool (%.2fGiB x%d chunks) does not fit %s; spreading %s",
        pool_bytes / 2**30,
        len(plan.plan),
        primary,
        plan.counts(),
    )


def resolve_placement_for_pool(
    pool, chains: int, primary: str, candidate_devices: Sequence[str]
) -> Optional[PoolPlacement]:
    """Resolve placement from a live pool object (orchestrator entry point).

    ``primary`` must be the lane's actual cache device: a CPU primary
    (``low_gpu_mem_usage``) deactivates the policy so the two mechanisms never
    fight. ``chains`` doubles the byte demand when a second (quantized-input)
    pool of the same size will also be produced for the block.
    """
    from auto_round.algorithms.quantization.search_shard import _probe_usable_bytes

    pool_bytes = _tensor_bytes(pool) * max(int(chains), 1)
    n_chunks = _pool_chunk_count(pool)
    if n_chunks <= 0 or pool_bytes <= 0:
        return None
    try:
        free_probe = _probe_usable_bytes
        plan = resolve_pool_placement(
            pool_bytes,
            n_chunks,
            primary,
            pool_shard_margin_bytes(),
            candidate_devices,
            free_probe,
        )
    except Exception:  # pragma: no cover - placement must never break quantization
        return None
    if plan is not None and len(plan.devices) > 1:
        _log_engaged_once(plan, pool_bytes, primary)
    return plan
