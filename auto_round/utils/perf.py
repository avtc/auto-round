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
"""Opt-in per-block phase timers, gated by ``AR_PERF_COUNTERS``.

Enabled, each quantized block emits one summary line::

    [perf] block=model.layers.10 total: 88s, pack: 10s, tune: 70s, read: 500ms, write: 2.5s, other: 5.0s

Collection points (add new ones only with a comment saying where they wrap):

- ``read``   -- orchestrator block loop: offloader reload + materialize + dtype/device placement
- ``tune``   -- orchestrator block loop: block pipeline (calibration, search, SignRound tuning)
- ``pack``   -- ``immediate_pack_block``: batched and per-module packing of the block
- ``write``  -- orchestrator block loop: immediate shard write (incl. resume flush)
- ``other``  -- residual: total block wall time minus the phases above

Durations use one decimal digit below 10 and none above, with an adaptive
unit (``ms`` / ``s`` / ``m``). Lines are emitted through ``tqdm.write`` so
they never garble an active progress bar (plain log fallback).
"""

import logging
import time
from contextlib import contextmanager

__all__ = ["PerfCounters", "perf", "format_duration"]

logger = logging.getLogger(__name__)

_PHASE_ORDER = ("pack", "tune", "read", "write")


def format_duration(seconds: float) -> str:
    """Format a duration as ``500ms`` / ``2.5s`` / ``88s`` / ``1.5m``.

    One decimal digit when the displayed value is below 10, none when it is
    10 or above; the unit scales ms -> s -> m (minutes only from 10 min up,
    so typical block phases read as 70s / 88s, not 1.2m / 1.5m).
    """
    if seconds < 1.0:
        value, unit = seconds * 1000.0, "ms"
    elif seconds < 600.0:
        value, unit = seconds, "s"
    else:
        value, unit = seconds / 60.0, "m"
    if value >= 10.0:
        return f"{value:.0f}{unit}"
    return f"{value:.1f}{unit}"


class PerfCounters:
    """Accumulates wall-clock phase times for the block being processed.

    All methods are no-ops (and ``phase`` yields without timing) while
    ``AR_PERF_COUNTERS`` is unset, so the disabled path costs one boolean
    check per call site.
    """

    def __init__(self):
        self._times: dict[str, float] = {}
        self._block_start: float | None = None
        self._block_label: str = ""

    @staticmethod
    def _enabled() -> bool:
        from auto_round import envs

        return bool(envs.AR_PERF_COUNTERS)

    @staticmethod
    def _sync():
        """Drain async CUDA work so phase boundaries are honest."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover - diagnostics must never crash a run
            pass

    @contextmanager
    def phase(self, name: str):
        if not self._enabled():
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._times[name] = self._times.get(name, 0.0) + elapsed

    def start_block(self, label: str):
        if not self._enabled():
            return
        self._sync()
        self._times = {}
        self._block_label = label
        self._block_start = time.perf_counter()

    def finish_block(self):
        if self._block_start is None:
            return
        self._sync()
        total = time.perf_counter() - self._block_start
        self._block_start = None
        self._emit(self._summary_line(total))

    def _summary_line(self, total: float) -> str:
        parts = [f"total: {format_duration(total)}"]
        for name in _PHASE_ORDER:
            elapsed = self._times.get(name, 0.0)
            if elapsed > 0.0:
                parts.append(f"{name}: {format_duration(elapsed)}")
        other = max(total - sum(self._times.values()), 0.0)
        parts.append(f"other: {format_duration(other)}")
        return f"[perf] block={self._block_label} " + ", ".join(parts)

    @staticmethod
    def _emit(message: str):
        """Print above any active tqdm bar instead of garbling its redraw."""
        try:
            from tqdm import tqdm

            tqdm.write(message)
        except Exception:  # pragma: no cover - fallback when tqdm misbehaves
            logger.info(message)


#: Process-wide singleton the orchestrator and packers record into.
perf = PerfCounters()
