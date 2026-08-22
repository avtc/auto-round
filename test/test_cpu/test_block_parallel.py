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

"""Tests for auto_round.compressors.block_parallel (pure helpers)."""

import os
import unittest

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402


class TestSplitSpans(unittest.TestCase):
    def test_single_group_even_split(self):
        spans = block_parallel.split_spans([64], 4)
        self.assertEqual(spans, [[(0, 0, 16)], [(0, 16, 32)], [(0, 32, 48)], [(0, 48, 64)]])

    def test_spans_never_cross_groups(self):
        # three groups of 3, 2 workers -> every span stays inside one group
        spans = block_parallel.split_spans([3, 3, 3], 2)
        flat = [s for worker in spans for s in worker]
        sizes = [3, 3, 3]
        for gi, s, e in flat:
            self.assertLessEqual(e, sizes[gi])
            self.assertGreaterEqual(s, 0)
            self.assertGreater(e - s, 0)
        # complete, ordered, non-overlapping coverage: spans tile every group
        covered = sum(e - s for gi, s, e in flat)
        self.assertEqual(covered, sum(sizes))
        for gi in range(3):
            intervals = sorted((s, e) for g, s, e in flat if g == gi)
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], sizes[gi])
            for (_s1, e1), (s2, _e2) in zip(intervals, intervals[1:]):
                self.assertEqual(e1, s2)
        counts = [sum(e - s for _, s, e in w) for w in spans]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_many_single_block_groups_balanced(self):
        spans = block_parallel.split_spans([1] * 8, 4)
        counts = [sum(e - s for _, s, e in worker) for worker in spans]
        self.assertEqual(sum(counts), 8)
        self.assertEqual(max(counts) - min(counts), 0)

    def test_more_workers_than_blocks(self):
        spans = block_parallel.split_spans([3], 8)
        self.assertEqual(len(spans), 3)
        self.assertEqual(sorted(s for w in spans for _, s, e in w for _ in [0]), [0, 1, 2])

    def test_empty(self):
        self.assertEqual(block_parallel.split_spans([], 4), [])


class TestSpansJson(unittest.TestCase):
    def test_roundtrip(self):
        spans = [[(0, 0, 16), (3, 1, 2)], [(0, 16, 32)]]
        self.assertEqual(block_parallel.spans_from_json(block_parallel.spans_to_json(spans)), spans)


class TestWorkerCommand(unittest.TestCase):
    def test_py_script_uses_executable(self):
        cmd = block_parallel.worker_command(["script.py", "--a", "1"])
        self.assertEqual(cmd[0].endswith("python") or "python" in cmd[0], True)
        self.assertEqual(cmd[1:], ["script.py", "--a", "1"])

    def test_console_script_passthrough(self):
        cmd = block_parallel.worker_command(["/usr/bin/auto-round", "--a", "1"])
        self.assertEqual(cmd, ["/usr/bin/auto-round", "--a", "1"])

    def test_device_map_collapsed_for_pinned_worker(self):
        argv = ["auto-round", "--device_map", "0,1,2,3", "--iters", "500"]
        self.assertEqual(
            block_parallel.worker_command(argv, device=2),
            ["auto-round", "--device_map", "0", "--iters", "500"],
        )

    def test_device_map_equals_form_collapsed(self):
        argv = ["auto-round", "--device_map=0,1", "--iters", "500"]
        self.assertEqual(
            block_parallel.worker_command(argv, device=1),
            ["auto-round", "--device_map=0", "--iters", "500"],
        )


if __name__ == "__main__":
    unittest.main()
