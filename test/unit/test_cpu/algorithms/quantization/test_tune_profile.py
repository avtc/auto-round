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
"""Tests for the env-gated tuning-loop profiler (AR_TUNE_PROFILE)."""

import time

import pytest

from auto_round.utils.tune_profile import TuneProfiler, make_tune_profiler, stage


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AR_TUNE_PROFILE", raising=False)
    assert make_tune_profiler() is None


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("AR_TUNE_PROFILE", "1")
    prof = make_tune_profiler()
    assert isinstance(prof, TuneProfiler)
    assert prof.debug is False
    monkeypatch.setenv("AR_TUNE_PROFILE", "true")
    assert isinstance(make_tune_profiler(), TuneProfiler)


def test_level_2_enables_debug_placement_line(monkeypatch, caplog):
    monkeypatch.setenv("AR_TUNE_PROFILE", "2")
    prof = make_tune_profiler()
    assert isinstance(prof, TuneProfiler) and prof.debug is True
    ar_logger = _capture_autoround_logs(caplog)
    try:
        prof.log_placement(
            device="cuda:1",
            loss_device="cuda:1",
            cache_device="cpu",
            inputs=[("cpu", (8, 2048, 4096)), ("cuda:0", (8, 2048, 4096))],
            fp_outputs=[("cuda:0", (8, 2048, 4096))],
            nsamples=128,
        )
    finally:
        ar_logger.removeHandler(caplog.handler)
    line = next(rec.message for rec in caplog.records if "[tune-profile-placement]" in rec.message)
    assert "cuda:1" in line and "cpu" in line and "nsamples=128" in line


def test_stage_with_none_profiler_is_nullcontext():
    with stage(None, "fwd"):
        pass  # must not raise


def test_host_accumulation_and_counts():
    prof = TuneProfiler(device="cpu")
    with prof.stage("fwd"):
        time.sleep(0.002)
    with prof.stage("fwd"):
        time.sleep(0.002)
    with prof.stage("snapshot"):
        time.sleep(0.001)
    assert prof.counts["fwd"] == 2
    assert prof.counts["snapshot"] == 1
    assert prof.host["fwd"] >= 0.004
    assert prof.host["snapshot"] >= 0.001


def test_stage_counts_even_on_exception():
    prof = TuneProfiler(device="cpu")
    with pytest.raises(RuntimeError):
        with prof.stage("bwd"):
            raise RuntimeError("boom")
    assert prof.counts["bwd"] == 1


def _capture_autoround_logs(caplog):
    """caplog can't see the autoround logger (propagate=False); attach its handler."""
    import logging

    ar_logger = logging.getLogger("autoround")
    ar_logger.addHandler(caplog.handler)
    ar_logger.setLevel("INFO")
    return ar_logger


def test_summary_log_line(caplog, monkeypatch):
    monkeypatch.setenv("AR_TUNE_PROFILE", "1")
    prof = make_tune_profiler()
    with prof.stage("fwd"):
        pass
    with prof.stage("bwd"):
        pass
    ar_logger = _capture_autoround_logs(caplog)
    try:
        prof.log_summary(block_name="layers.3", iters_done=10, wall=0.5)
    finally:
        ar_logger.removeHandler(caplog.handler)
    assert any(
        "[tune-profile]" in rec.message and "layers.3" in rec.message and "fwd" in rec.message for rec in caplog.records
    )


def test_summary_zero_iters_guard():
    prof = TuneProfiler(device="cpu")
    prof.log_summary(block_name="b", iters_done=0, wall=0.0)  # must not raise


def test_cuda_device_without_availability_uses_host_only(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    prof = TuneProfiler(device="cuda")
    assert prof._use_events is False
    with prof.stage("fwd"):
        pass
    assert prof.counts["fwd"] == 1


def test_summary_reports_host_and_wall_consistently(caplog, monkeypatch):
    monkeypatch.setenv("AR_TUNE_PROFILE", "1")
    prof = make_tune_profiler()
    with prof.stage("ref_h2d"):
        pass
    with prof.stage("step"):
        pass
    ar_logger = _capture_autoround_logs(caplog)
    try:
        prof.log_summary(block_name="x", iters_done=4, wall=0.08)
    finally:
        ar_logger.removeHandler(caplog.handler)
    line = next(rec.message for rec in caplog.records if "[tune-profile]" in rec.message)
    assert "ms/iter" in line
    assert "iters=4" in line
