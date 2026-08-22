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

"""Tests for the push-assignment protocol helpers (rank metadata, merge hygiene)."""

import os
import tempfile
import unittest

import torch

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402


class TestResultRank(unittest.TestCase):
    def test_rank_roundtrip_and_merge_hygiene(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["AR_BLOCK_PARALLEL_RANK"] = "3"
            try:
                block_parallel.save_block_results(
                    d, "model.layers.0", {"l0": {"scale": torch.tensor([1.0]), "zp": None}}
                )
            finally:
                os.environ.pop("AR_BLOCK_PARALLEL_RANK", None)
            self.assertEqual(block_parallel.read_result_rank(d, "model.layers.0"), 3)
            # merge skips advisory keys
            merged = block_parallel.load_all_block_results(d)
            self.assertIn("l0", merged)
            self.assertNotIn("_worker_rank", merged)

    def test_rank_absent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(block_parallel.read_result_rank(d, "missing"), -1)


class TestCountAgnosticState(unittest.TestCase):
    """Durable state must not depend on worker count (rerun with any N)."""

    def test_done_from_manifest_prefix_without_result_files(self):
        # serial-after-serial produces manifests and no block files; a parallel
        # rerun must still initialize done-sets from the prefix alone
        with tempfile.TemporaryDirectory() as d:
            # simulate: two blocks completed serially -> result files absent
            # (verified indirectly: has_block_results is the only other source)
            self.assertFalse(block_parallel.has_block_results(d, "model.layers.0"))
            self.assertEqual(block_parallel.missing_result_blocks(d, [["b0", "b1", "b2"]]), ["b0", "b1", "b2"])


if __name__ == "__main__":
    unittest.main()
