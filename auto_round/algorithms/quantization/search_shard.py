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
