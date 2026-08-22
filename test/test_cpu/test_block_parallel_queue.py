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

"""Tests for the block-parallel dispatch queue (file-based job protocol)."""

import os
import tempfile
import unittest

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402


class TestQueueProtocol(unittest.TestCase):
    def test_write_claim_consume(self):
        with tempfile.TemporaryDirectory() as d:
            q = block_parallel.queue_dir(d)
            block_parallel.write_job(q, seq=1, job_type="tune", block_name="model.layers.0", group=0, index=0)
            block_parallel.write_job(q, seq=2, job_type="tune", block_name="model.layers.1", group=0, index=1)
            job = block_parallel.claim_next_job(q)
            self.assertEqual(job["seq"], 1)
            self.assertEqual(job["block_name"], "model.layers.0")
            # claimed job is not claimable again
            job2 = block_parallel.claim_next_job(q)
            self.assertEqual(job2["seq"], 2)
            self.assertEqual(block_parallel.claim_next_job(q), None)

    def test_failure_marks_job(self):
        with tempfile.TemporaryDirectory() as d:
            q = block_parallel.queue_dir(d)
            block_parallel.write_job(q, seq=1, job_type="tune", block_name="b0", group=0, index=0)
            job = block_parallel.claim_next_job(q)
            block_parallel.job_failed(q, job, "boom")
            self.assertEqual(block_parallel.claim_next_job(q), None)  # not silently re-claimed
            failed = block_parallel.list_failed(q)
            self.assertEqual(len(failed), 1)

    def test_stop_file(self):
        with tempfile.TemporaryDirectory() as d:
            q = block_parallel.queue_dir(d)
            self.assertFalse(block_parallel.read_stop(q))
            block_parallel.write_stop(q)
            self.assertTrue(block_parallel.read_stop(q))

    def test_produce_job_fields(self):
        with tempfile.TemporaryDirectory() as d:
            q = block_parallel.queue_dir(d)
            block_parallel.write_job(q, seq=1, job_type="produce", group=0, start=0, end=64)
            job = block_parallel.claim_next_job(q)
            self.assertEqual(job["job_type"], "produce")
            self.assertEqual((job["start"], job["end"]), (0, 64))

    def test_list_pending_after_claim(self):
        with tempfile.TemporaryDirectory() as d:
            q = block_parallel.queue_dir(d)
            block_parallel.write_job(q, seq=1, job_type="tune", block_name="b0", group=0, index=0)
            block_parallel.write_job(q, seq=2, job_type="tune", block_name="b1", group=0, index=1)
            self.assertEqual(len(block_parallel.list_pending(q)), 2)
            block_parallel.claim_next_job(q)
            self.assertEqual(len(block_parallel.list_pending(q)), 1)


if __name__ == "__main__":
    unittest.main()
