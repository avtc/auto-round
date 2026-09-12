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

"""Run weight-local quantization searches in parallel, one worker per weight device.

Some quantization searches only read the module's own weight (plus small
per-module statistics such as the imatrix) and never touch calibration
activations. When the block's weights are sharded across several GPUs such
searches can run directly on the device that already hosts each weight: no
copies, no mirrors, and the devices naturally share the work. This module
provides the generic grouping/execution helpers used by the wrapper and RTN
search loops.

Searches that depend on activations (e.g. the AWQ clip search) are not
weight-local and must stay on their serial paths.
"""

import threading
import time
from collections import OrderedDict

import torch

import auto_round.envs as envs
from auto_round.logger import logger


def group_items_by_device(items, device_of, none_key="uncategorized"):
    """Group ``(index, item)`` pairs by ``device_of(item)`` preserving input order.

    Args:
        items: Iterable of items to group.
        device_of: Callable mapping an item to a device key (str or torch.device).
        none_key: Bucket label for items whose device is ``None``.

    Returns:
        ``OrderedDict[device_key, list[(index, item)]]`` with devices in first-seen
        order and items in original order within each device.
    """
    groups = OrderedDict()
    for idx, item in enumerate(items):
        key = device_of(item)
        if key is None:
            key = none_key
        groups.setdefault(key, []).append((idx, item))
    return groups


def run_items_by_device(groups, fn, use_cuda_ctx=True):
    """Run ``fn(item)`` for every grouped item, one worker thread per device group.

    A single group is executed inline on the calling thread. Any exception raised
    by ``fn`` is re-raised on the calling thread after all workers joined
    (fail-visible: search failures are never silently swallowed).

    Args:
        groups: Output of :func:`group_items_by_device`.
        fn: Callable executed as ``fn(index, item)``; must be safe to run
            concurrently across groups (items within one group run serially on
            their worker).
        use_cuda_ctx: Wrap each cuda worker in ``torch.cuda.device`` so ops land
            on the weight device even when the ambient current device differs.
    """
    if len(groups) <= 1:
        for _idx, item in next(iter(groups.values()), []):
            fn(_idx, item)
        return

    first_error = []
    error_lock = threading.Lock()
    threads = []

    def _worker(device_key, indexed_items):
        try:
            key = str(device_key)
            need_ctx = use_cuda_ctx and torch.device(key).type == "cuda" and torch.cuda.is_available()
            if need_ctx:
                with torch.cuda.device(torch.device(key)):
                    for _idx, item in indexed_items:
                        fn(_idx, item)
            else:
                for _idx, item in indexed_items:
                    fn(_idx, item)
        except BaseException as exc:  # noqa: B036 - re-raised below, never swallowed
            with error_lock:
                if not first_error:
                    first_error.append(exc)

    for device_key, indexed_items in groups.items():
        if not indexed_items:
            continue
        t = threading.Thread(target=_worker, args=(device_key, indexed_items), name=f"search-shard-{device_key}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if first_error:
        raise first_error[0]


def shard_eligible(device_keys):
    """Return True when the device mix justifies threading the search loop.

    Sharding engages only for multi-device (cuda-containing) mixes: on a single
    device the serial loop is already optimal and stays bit-for-bit unchanged.
    """
    keys = {str(k) for k in device_keys}
    if len(keys) < 2:
        return False
    for k in keys:
        try:
            if torch.device(k).type == "cuda":
                return True
        except (ValueError, RuntimeError):  # unrecognized labels are simply not cuda
            continue
    return False


_ENGAGED_LOGGED = set()


def log_engaged_once(label):
    """Log the sharding engagement once per process (INFO, no counters).

    Detailed per-block counters are intentionally not emitted here; they belong
    to the perf-counter infrastructure of the parallel-tuning work.
    """
    if label in _ENGAGED_LOGGED:
        return
    _ENGAGED_LOGGED.add(label)
    logger.info("[search-shard] %s: running weight-local searches with one worker per device", label)


def shard_disabled_by_env():
    """Kill switch for the per-device search sharding."""
    return bool(envs.AR_DISABLE_SEARCH_SHARD)


def wrap_shard_enabled():
    """Opt-in gate for threading the wrapper-time searches.

    Measured on a 300B MoE block spanning 8 GPUs, threading the per-module
    wrapper searches made the wrap phase SLOWER (+32 s per block). The
    serialization mechanism is not yet profiled; torch.compile is NOT a wrap-
    phase factor (compile_func only lazily binds the callable -- actual
    compilation happens at each wrapper's first tune-loop call), so the
    leading unmeasured candidates are GIL contention on the many short
    python-dominated searches and allocator/dispatch contention. The
    optimized-RTN iters=0 searches (fewer, much longer searches) measured
    2.7x faster when threaded and stay enabled by default.
    """
    return bool(envs.AR_ENABLE_WRAP_SEARCH_SHARD)


def _wrap_batch_device_of(inputs):
    return str(inputs[0].device)


def _fn_key(fn):
    """Stable, safe key term for a resolved search callable.

    Plain functions key by identity location (same function = same behavior);
    ``functools.partial`` objects key by their visible captures; anything else
    (e.g. a per-call closure, whose captures are not introspectable) keys by
    ``repr`` so it never merges with a lookalike. Merging two different searches
    into one stacked batch would silently apply the wrong math to one side, so
    unknown callables always stay separate.
    """
    import functools

    if isinstance(fn, functools.partial):
        kwargs = tuple(sorted(fn.keywords.items(), key=lambda kv: str(kv[0])))
        try:
            qualname = fn.func.__qualname__
            module = fn.func.__module__
        except AttributeError:  # pragma: no cover - exotic callables
            return repr(fn)
        return ("partial", module, qualname, fn.args, kwargs)
    if callable(fn):
        try:
            return ("fn", fn.__module__, fn.__qualname__)
        except AttributeError:  # pragma: no cover - exotic callables
            return repr(fn)
    return repr(fn)


def _wrap_batch_key(inputs):
    weight, _data_type, bits, _imatrix, thresh, search_fn = inputs
    return (
        str(weight.device),
        tuple(weight.shape),
        str(weight.dtype),
        bits,
        float(thresh),
        _fn_key(search_fn),
    )


def _probe_usable_bytes(device_key):
    """Corrected free bytes on a cuda device (raw free + reserved-but-unallocated)."""
    try:
        dev = torch.device(str(device_key))
        if dev.type != "cuda" or dev.index is None or not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(dev.index)
        free += torch.cuda.memory_reserved(dev.index) - torch.cuda.memory_allocated(dev.index)
        return max(free, 0)
    except (ValueError, RuntimeError, AttributeError):
        return None


_WRAP_BATCH_MAX_ELEMS = 2**28  # ~1 GiB fp32 stacked weights per batched call (matches the NeUQI expert batching)


def _wrap_batch_max_elems():
    """Element budget per stacked batch; AR_WRAP_SEARCH_BATCH_GB overrides in GiB of fp32 weights."""
    try:
        gb = float(envs.AR_WRAP_SEARCH_BATCH_GB)
    except (TypeError, ValueError):
        gb = None
    if gb is not None and gb > 0:
        return max(int(gb * 2**30 // 4), 1)
    return _WRAP_BATCH_MAX_ELEMS


def _batch_cap(group, device_key, max_batch):
    """Modules per stacked batch: explicit cap > free-VRAM probe > 64, capped by the element budget."""
    if max_batch is not None:
        return max(1, max_batch)
    inputs0 = group[0]._deferred_search_inputs
    elements_per_module = inputs0[0].numel() + (inputs0[3].numel() if inputs0[3] is not None else 0)
    # the search is bandwidth-bound: batches beyond ~1 GiB of stacked weights move
    # the same total bytes, so the fixed element budget only lowers transient VRAM
    elem_cap = max(1, _wrap_batch_max_elems() // max(elements_per_module, 1))
    probe_cap = 64
    usable = _probe_usable_bytes(device_key)
    if usable is not None:
        per_module_bytes = elements_per_module * 4 * 4  # fp32 working set incl. temporaries
        probe_cap = max(1, min(1024, usable // 2 // max(per_module_bytes, 1)))
    return max(1, min(probe_cap, elem_cap))


def run_batched_wrap_search(deferred_wrappers, max_batch=None, batch_vram_budget=4 * 2**30):
    """Run deferred weight-local wrap searches on stacked same-shape batches.

    Wrappers stage ``(weight_reshape, data_type, bits, imatrix, q_scale_thresh,
    search_fn)`` tuples in ``_deferred_search_inputs``, where ``search_fn`` is the
    exact callable the per-module path would have invoked (resolved at wrap time,
    so future dispatch changes -- e.g. alternative optimized searches -- travel
    with the module automatically). Modules whose staged key
    ``(device, shape, weight dtype, bits, threshold, search_fn)`` matches are
    stacked along a leading dim and searched with ONE ``search_fn`` call: the
    per-row math is unchanged (row-independent reductions over the last dim), so
    results are bit-identical to the per-module path while python/launch overhead
    drops by the batch factor. Batches are capped by ``max_batch`` and by a VRAM
    budget on the group's device; groups on different devices run on one worker
    thread per device; singleton groups take the identical per-module call.

    Returns True when the inputs were consumed; False when sharding is disabled
    by AR_DISABLE_SEARCH_SHARD (the caller then runs the searches per module).
    """
    del batch_vram_budget  # budget derived per device in _batch_cap
    if shard_disabled_by_env():
        return False
    if not deferred_wrappers:
        return False

    device_groups = OrderedDict()
    for w in deferred_wrappers:
        inputs = w._deferred_search_inputs
        if inputs is None:
            raise RuntimeError(
                f"{type(w).__name__} was queued for batched wrap search without staged inputs; "
                "the defer_search flag never reached its search init"
            )
        dev = _wrap_batch_device_of(inputs)
        device_groups.setdefault(dev, []).append(w)

    stats = {dev: {"modules": 0, "batches": 0, "singletons": 0} for dev in device_groups}

    def _run_one(wrapper):
        inputs = wrapper._deferred_search_inputs
        if inputs is None:
            return
        weight, _data_type, bits, imatrix, _thresh, search_fn = inputs
        wrapper.finalize_batched_search(search_fn(weight, bits, imatrix))

    def _run_device(device_key, wrappers):
        _t0 = time.perf_counter()
        by_key = OrderedDict()
        for w in wrappers:
            by_key.setdefault(_wrap_batch_key(w._deferred_search_inputs), []).append(w)
        for _key, group in by_key.items():
            if len(group) < 2:
                _run_one(group[0])
                stats[device_key]["singletons"] += 1
                continue
            cap = _batch_cap(group, device_key, max_batch)
            for start in range(0, len(group), cap):
                chunk = group[start : start + cap]
                inputs0 = chunk[0]._deferred_search_inputs
                _bits = inputs0[2]
                search_fn = inputs0[5]
                stacked_w = torch.stack([w._deferred_search_inputs[0] for w in chunk])
                stacked_im = torch.stack([w._deferred_search_inputs[3] for w in chunk])
                try:
                    results = search_fn(stacked_w, _bits, stacked_im)
                except torch.OutOfMemoryError:
                    logger.warning(
                        "[search-shard] stacked wrap search OOM (%d modules); finishing this chunk "
                        "per-module (shrink batches with AR_WRAP_SEARCH_BATCH_GB or disable with "
                        "AR_DISABLE_SEARCH_SHARD=1)",
                        len(chunk),
                    )
                    for w in chunk:
                        w._run_deferred_search_now()
                    stats[device_key]["singletons"] += len(chunk)
                    continue
                for w, res in zip(chunk, results):
                    w.finalize_batched_search(res)
                stats[device_key]["batches"] += 1
        stats[device_key]["modules"] = len(wrappers)
        stats[device_key]["wall"] = time.perf_counter() - _t0

    if len(device_groups) > 1:
        keyed = group_items_by_device(
            list(device_groups.values()), device_of=lambda ws: _wrap_batch_device_of(ws[0]._deferred_search_inputs)
        )
        run_items_by_device(
            keyed, lambda _idx, ws: _run_device(_wrap_batch_device_of(ws[0]._deferred_search_inputs), ws)
        )
    else:
        for dev, ws in device_groups.items():
            _run_device(dev, ws)
    _t_total = sum(st.get("wall", 0.0) for st in stats.values())
    _n_mod = sum(st["modules"] for st in stats.values())
    _n_batch = sum(st["batches"] for st in stats.values())
    _n_single = sum(st["singletons"] for st in stats.values())
    per_device = " ".join(f"{dev}={st.get('wall', 0.0):.2f}s" for dev, st in stats.items())
    logger.debug(
        "[search-shard] wrap search: %d modules, %d batch + %d singleton search calls, "
        "%.2fs device-wall (%.2fs summed) [%s]",
        _n_mod,
        _n_batch,
        _n_single,
        max(st.get("wall", 0.0) for st in stats.values()),
        _t_total,
        per_device,
    )
    return True
