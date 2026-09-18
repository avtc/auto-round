# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Snapshot routing decisions log once per change (INFO/WARN), DEBUG on repeats."""

from types import SimpleNamespace

from auto_round.compressors.utils import _snapshot_route_log_


class TestSnapshotRouteLog:
    def test_info_once_then_debug(self, caplog):
        import logging

        from auto_round.logger import logger as ar_logger

        block = SimpleNamespace()
        with caplog.at_level(logging.DEBUG, logger="autoround"):
            ar_logger.addHandler(caplog.handler)
            try:
                _snapshot_route_log_(block, "peer:cuda:1", "[snapshot] cloning %.2fGiB to idle peer %s", 1.61, "cuda:1")
                _snapshot_route_log_(block, "peer:cuda:1", "[snapshot] cloning %.2fGiB to idle peer %s", 1.61, "cuda:1")
                _snapshot_route_log_(block, "host", "[snapshot] parking on host", warn_first=True)
                _snapshot_route_log_(block, "host", "[snapshot] parking on host", warn_first=True)
            finally:
                ar_logger.removeHandler(caplog.handler)
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(infos) == 1  # first peer announcement only
        assert len(warns) == 1  # first host fallback only
        assert len(debugs) == 2  # both repeats
