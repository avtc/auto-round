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
import time

import pytest

from auto_round.utils.peak_watch import PeakWatcher


class TestPeakWatcher:
    def test_peak_attributed_to_phase_and_logged(self, caplog):
        w = PeakWatcher(interval_s=0.005)
        w.start()
        w.set_phase("search")
        _ballast = bytearray(300 * 2**20)  # 300MB -> new max in that phase
        time.sleep(0.1)
        w.stop()
        del _ballast
        assert w.peak_rss_gb > 0.2
        assert w.peak_phase == "search"
        import logging

        from auto_round.logger import logger as _proj_logger

        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record)

        _handler = _Cap()
        _proj_logger.addHandler(_handler)
        try:
            w.log("block 0")
        finally:
            _proj_logger.removeHandler(_handler)
        assert any("phase=search" in r.getMessage() for r in records)

    def test_phase_update_visible_while_running(self):
        w = PeakWatcher(interval_s=0.005)
        w.start()
        w.set_phase("load")
        time.sleep(0.03)
        w.set_phase("pack")
        w.stop()
        assert w.phase == "pack"

    def test_reset_run_max(self):
        w = PeakWatcher()
        w.peak_rss_gb = 5.0
        w.peak_phase = "old"
        w.reset_run_max()
        assert w.peak_rss_gb == 0.0
        assert w.log("x") is None  # no peak recorded -> no log line, no raise

    def test_stop_is_idempotent(self):
        w = PeakWatcher()
        w.start()
        w.stop()
        w.stop()  # must not raise


@pytest.mark.parametrize("unused", [1])
def test_module_import_clean(unused):
    import auto_round.utils.peak_watch as m

    assert m.PeakWatcher is not None


class TestRunnerAliasRelease:
    def test_composer_source_releases_last_output_dict(self):
        # the runner's alias must be cleared after the reference forward so
        # auxiliary outputs do not span the whole search/pack/write phase
        import inspect

        from auto_round.algorithms import composer

        src = inspect.getsource(composer)
        anchor = 'reference_next_input = getattr(block_forward_fn, "last_output_dict", None) or reference_output'
        assert src.count(anchor) == 1
        idx = src.splitlines().index([ln for ln in src.splitlines() if anchor in ln][0])
        window = "\n".join(src.splitlines()[idx : idx + 9])
        assert "last_output_dict = None" in window
