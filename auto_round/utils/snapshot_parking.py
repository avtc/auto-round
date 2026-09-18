# Copyright (c) 2025 Intel Corporation
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

"""Probe-based parking for best-params snapshots of huge tuning layers.

A huge layer's best-params snapshot is as large as its rounding parameter
(fp32 ``value``), and the tune loop re-copies it on every improving
iteration. Parking it on the host unconditionally costs a multi-second
device-to-host copy per improving iteration, so this module picks the
cheapest device that provably has room:

1. the layer's own device, when half its free pool covers the snapshot —
   the same half-free-pool convention ``WrapperLinear.row_block_bounds``
   uses for its row windows, so the two consumers share one budget rule;
2. otherwise an idle CUDA peer with room — a peer-to-peer copy replaces
   the host round trip and the home device keeps its full window budget;
3. otherwise the host, exactly as before.

Selection is fail-safe: any probe failure or non-CUDA home falls back to
the host path, which is the previously shipped behavior.

Merge note: this module is deliberately self-contained and does not touch
``compressors/utils.py``, where the multi-GPU improvements branch keeps its
own snapshot helpers; whichever branch lands second keeps or unifies the
two without shared hunks.
"""

import weakref
from typing import Optional

import torch

from auto_round.logger import logger

# The home device shares its free pool with the row windows of the row-blocked
# tune path; both consumers follow the same "use half the free pool" rule.
_HOME_POOL_FRACTION = 0.5
# An idle peer carries a CUDA context and allocator fragmentation; leave this
# fraction of its free memory untouched when parking there.
_PEER_HEADROOM_FRACTION = 0.1

_announced: "weakref.WeakSet" = weakref.WeakSet()


def _free_bytes(device: torch.device) -> Optional[int]:
    """Free bytes on a CUDA device, or ``None`` when the probe fails."""
    try:
        free, _total = torch.cuda.mem_get_info(device)  # pylint: disable=c-extension-no-member
        return int(free)
    except Exception:  # pylint: disable=broad-except  # pragma: no cover - exotic devices
        return None


def _cuda_device_count() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:  # pylint: disable=broad-except  # pragma: no cover
        return 0


def snapshot_bytes(wrapper: torch.nn.Module) -> int:
    """Total bytes a best-params snapshot of ``wrapper`` duplicates (real tensor bytes)."""
    params = getattr(wrapper, "params", None)
    if not params:
        return 0
    return int(sum(t.numel() * t.element_size() for t in params.values()))


def _wrapper_home_device(wrapper: torch.nn.Module) -> Optional[torch.device]:
    home = getattr(wrapper, "device", None)
    if isinstance(home, torch.device):
        return home
    params = getattr(wrapper, "params", None)
    if params:
        for tensor in params.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
    return None


def select_snapshot_device(wrapper: torch.nn.Module) -> torch.device:
    """Cheapest device that provably fits the snapshot; host fallback.

    Returns the wrapper's home device, an idle CUDA peer, or the host, in
    that order, using live free-memory probes. Never raises.
    """
    need = snapshot_bytes(wrapper)
    home = _wrapper_home_device(wrapper)
    if need <= 0 or home is None or home.type != "cuda":
        return torch.device("cpu")

    free = _free_bytes(home)
    if free is not None and free * _HOME_POOL_FRACTION >= need:
        return home

    for index in range(_cuda_device_count()):
        candidate = torch.device("cuda", index)
        if candidate == home:
            continue
        peer_free = _free_bytes(candidate)
        if peer_free is not None and peer_free * (1.0 - _PEER_HEADROOM_FRACTION) >= need:
            if wrapper not in _announced:
                _announced.add(wrapper)
                logger.info(
                    "[snapshot] parking the best-params snapshot (%.2f GiB) on idle peer %s; "
                    "the home device %s keeps its full row-window budget",
                    need / 2**30,
                    candidate,
                    home,
                )
            return candidate

    return torch.device("cpu")
