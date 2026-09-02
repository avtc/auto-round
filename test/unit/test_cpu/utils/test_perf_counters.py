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
"""AR_PERF_COUNTERS phase timers: formatting, accumulation, emission."""

import io

import pytest

from auto_round.utils.perf import PerfCounters, format_duration


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.5, "500ms"),
        (0.0025, "2.5ms"),
        (0.01, "10ms"),
        (0.999, "999ms"),
        (1.0, "1.0s"),
        (2.5, "2.5s"),
        (5.0, "5.0s"),
        (10.0, "10s"),
        (70.0, "70s"),
        (88.0, "88s"),
        (90.0, "90s"),
        (599.0, "599s"),
        (600.0, "10m"),
        (660.0, "11m"),
    ],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


class TestPerfCounters:
    def test_disabled_records_and_emits_nothing(self, monkeypatch, capsys):
        monkeypatch.delenv("AR_PERF_COUNTERS", raising=False)
        counters = PerfCounters()
        with counters.phase("pack"):
            pass
        counters.start_block("b0")
        counters.finish_block()
        assert counters._times == {}
        assert capsys.readouterr().out == ""

    def test_summary_line_phases_and_other(self, monkeypatch):
        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        counters = PerfCounters()
        counters._block_label = "b"
        counters._times = {"pack": 10.0, "tune": 70.0, "read": 0.5, "write": 2.5}
        line = counters._summary_line(total=88.0)
        assert line == "[perf] block=b total: 88s, pack: 10s, tune: 70s, read: 500ms, write: 2.5s, other: 5.0s"

    def test_summary_line_skips_unrecorded_phases(self, monkeypatch):
        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        counters = PerfCounters()
        counters._block_label = "b"
        counters._times = {"pack": 0.25}
        line = counters._summary_line(total=1.0)
        assert line == "[perf] block=b total: 1.0s, pack: 250ms, other: 750ms"

    def test_phase_accumulates_across_calls(self, monkeypatch):
        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        counters = PerfCounters()
        with counters.phase("pack"):
            pass
        with counters.phase("pack"):
            pass
        assert counters._times["pack"] >= 0.0
        assert len(counters._times) == 1

    def test_finish_block_emits_via_stdout(self, monkeypatch, capsys):
        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        counters = PerfCounters()
        counters.start_block("blk")
        counters._times = {"pack": 3.0}
        counters.finish_block()
        out = capsys.readouterr().out
        assert "[perf] block=blk" in out
        assert "pack: 3.0s" in out
        assert "other:" in out

    def test_finish_block_without_start_is_noop(self, monkeypatch, capsys):
        monkeypatch.setenv("AR_PERF_COUNTERS", "1")
        counters = PerfCounters()
        counters.finish_block()
        assert capsys.readouterr().out == ""
