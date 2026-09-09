# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""AR_MEM_COUNTERS gates the DEBUG VRAM census; the WARNING (OOM) census stays unconditional."""

import logging

from auto_round.logger import logger as _lg


def _fake_cuda(monkeypatch):
    import torch

    calls = {"mem_get_info": 0}

    def _mem_get_info(dev2):
        calls["mem_get_info"] += 1
        return (0, 0)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev2: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev2: 0)
    # the project logger does not propagate; surface it for caplog
    from auto_round.logger import logger as _root_logger

    monkeypatch.setattr(_root_logger, "propagate", True)
    return calls


def test_debug_census_silent_without_mem_counters(monkeypatch, caplog):
    from auto_round.utils.device import log_cuda_memory_census

    monkeypatch.delenv("AR_MEM_COUNTERS", raising=False)
    calls = _fake_cuda(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        log_cuda_memory_census("entry lm_head", "cuda:0")
    assert not any("[vram]" in r.message for r in caplog.records)
    # gated out before the allocator is even queried (the walk is not free)
    assert calls["mem_get_info"] == 0


def test_debug_census_logs_with_mem_counters(monkeypatch, caplog):
    from auto_round.utils.device import log_cuda_memory_census

    monkeypatch.setenv("AR_MEM_COUNTERS", "1")
    calls = _fake_cuda(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=_lg.name):
        log_cuda_memory_census("entry lm_head", "cuda:0")
    assert any("[vram]" in r.message for r in caplog.records)
    assert calls["mem_get_info"] >= 1


def test_warning_census_stays_unconditional(monkeypatch, caplog):
    from auto_round.utils.device import log_cuda_memory_census

    monkeypatch.delenv("AR_MEM_COUNTERS", raising=False)
    calls = _fake_cuda(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=_lg.name):
        log_cuda_memory_census("[tune-oom] blk", "cuda:0", log_level="warning")
    assert any("[vram]" in r.message for r in caplog.records)
    assert calls["mem_get_info"] >= 1


def test_no_device_is_a_noop_without_cuda(monkeypatch, caplog):
    import torch

    from auto_round.utils.device import log_cuda_memory_census

    monkeypatch.delenv("AR_MEM_COUNTERS", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # documented no-op without CUDA: must not raise (torch.device(str(None))
    # would) and must not log
    log_cuda_memory_census("entry lm_head")
    assert not any("[vram]" in r.message for r in caplog.records)
