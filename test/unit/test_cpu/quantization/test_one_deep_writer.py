# coding=utf-8
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the one-deep background writer used by the shard write/flush."""

import time
import unittest

from auto_round.compressors.orchestrator import _OneDeepWriter


class TestOneDeepWriter(unittest.TestCase):
    def test_dispatch_runs_and_join_blocks(self):
        w = _OneDeepWriter()
        done = []
        w.dispatch(lambda: done.append(1))
        w.join()
        self.assertEqual(done, [1])
        self.assertIsNone(w._t)

    def test_one_deep_previous_joined_before_next_starts(self):
        w = _OneDeepWriter()
        order = []

        def slow():
            order.append("slow-start")
            time.sleep(0.2)
            order.append("slow-end")

        def fast():
            order.append("fast-start")

        w.dispatch(slow)
        w.dispatch(fast)  # must join slow first
        w.join()
        self.assertEqual(order[0], "slow-start")
        self.assertEqual(order[1], "slow-end")
        self.assertEqual(order[2], "fast-start")

    def test_exception_reraised_on_join(self):
        w = _OneDeepWriter()

        def boom():
            raise RuntimeError("writer failed")

        w.dispatch(boom)
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            w.join()
        # subsequent dispatches still work (state cleared)
        ran = []
        w.dispatch(lambda: ran.append(1))
        w.join()
        self.assertEqual(ran, [1])

    def test_join_without_dispatch_is_noop(self):
        w = _OneDeepWriter()
        w.join()  # must not raise
