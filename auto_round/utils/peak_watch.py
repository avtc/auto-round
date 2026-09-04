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
"""Peak-RSS watcher: attribute the process high-water mark to a phase.

The memory inventory samples fire between phases and structurally miss the
during-block transient peak; this watcher samples RSS on a short interval
and, whenever a new maximum is set, records the current phase label plus a
top-regions snapshot. With glibc's ``mmap_threshold`` lowered (e.g.
``GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072``) every large
allocation is its own anonymous mapping whose size equals the allocation
size, so the captured snapshot names the live big allocations directly.
"""

import threading
import time

from auto_round.logger import logger


class PeakWatcher:
    """Best-effort peak-RSS attribution; must never break the run."""

    def __init__(self, interval_s: float = 0.02, top_k: int = 8):
        self._interval = interval_s
        self._top_k = top_k
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.phase = "init"
        self.peak_rss_gb = 0.0
        self.peak_phase = "init"
        self.peak_regions = ""
        self.peak_at_s = 0.0
        self._t0 = time.perf_counter()

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ar-peak-watch", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase

    # -- internals -----------------------------------------------------------

    def _run(self):
        import psutil

        proc = psutil.Process()
        while not self._stop.wait(self._interval):
            try:
                rss_gb = proc.memory_info().rss / 2**30
            except Exception:  # noqa: BLE001  diagnostics only
                continue
            with self._lock:
                if rss_gb <= self.peak_rss_gb + 1e-6:
                    continue
                self.peak_rss_gb = rss_gb
                self.peak_phase = self.phase
                self.peak_at_s = time.perf_counter() - self._t0
                self.peak_regions = self._snapshot_regions(proc)

    def _snapshot_regions(self, proc) -> str:
        try:
            regions = sorted(proc.memory_maps(grouped=False), key=lambda m: -m.rss)[: self._top_k]
            return "; ".join(f"{m.rss / 2**30:.2f}G {m.path or '[anon]'}" for m in regions if m.rss > 0)
        except Exception:  # noqa: BLE001  diagnostics only
            return ""

    # -- reporting -----------------------------------------------------------

    def log(self, tag: str):
        """One INFO line with the current peak attribution."""
        if self.peak_rss_gb <= 0:
            return
        logger.info(
            "[stream-mem] %s peak: %.2fG @ phase=%s (t+%.1fs) | regions: %s",
            tag,
            self.peak_rss_gb,
            self.peak_phase,
            self.peak_at_s,
            self.peak_regions or "n/a",
        )

    def reset_run_max(self):
        """Forget the max so the next block reports its own peak."""
        with self._lock:
            self.peak_rss_gb = 0.0
            self.peak_phase = self.phase
            self.peak_regions = ""
            self.peak_at_s = 0.0
