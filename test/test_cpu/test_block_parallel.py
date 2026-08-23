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

import torch

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402


class TestWorkerCommand(unittest.TestCase):
    def test_py_script_uses_executable(self):
        cmd = block_parallel.worker_command(["script.py", "--a", "1"])
        self.assertEqual(cmd[0].endswith("python") or "python" in cmd[0], True)
        self.assertEqual(cmd[1:], ["script.py", "--a", "1"])

    def test_console_script_passthrough(self):
        cmd = block_parallel.worker_command(["/usr/bin/auto-round", "--a", "1"])
        self.assertEqual(cmd, ["/usr/bin/auto-round", "--a", "1"])

    def test_module_launch_reproduces_m(self):
        argv = ["/home/ubuntu/git/auto-round/auto_round/__main__.py", "--iters", "500"]
        self.assertEqual(
            block_parallel.worker_command(argv),
            [block_parallel.sys.executable, "-m", "auto_round", "--iters", "500"],
        )

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


class TestGuardReasons(unittest.TestCase):
    """Guard returns None (ok), 'env' (flag off), or a reason (flag on, blocked)."""

    BASE = dict(
        need_quanted_input=False,
        super_group_size=None,
        nblocks=1,
        n_block_groups=1,
        is_immediate_packing=True,
        is_immediate_saving=True,
        n_blocks_total=64,
        argv=["auto-round", "--a"],
    )

    def setUp(self):
        os.environ["AR_ENABLE_BLOCK_PARALLEL_TUNING"] = "1"
        os.environ.pop("AR_BLOCK_PARALLEL_WORKER", None)
        os.environ.pop("AR_RESUME_DIR", None)

    def tearDown(self):
        os.environ.pop("AR_ENABLE_BLOCK_PARALLEL_TUNING", None)

    def test_enabled_returns_none(self):
        self.assertIsNone(block_parallel.block_parallel_tuning_enabled(**self.BASE))

    def test_flag_off_returns_env(self):
        os.environ["AR_ENABLE_BLOCK_PARALLEL_TUNING"] = "0"
        self.assertEqual(block_parallel.block_parallel_tuning_enabled(**self.BASE), "env")

    def test_quanted_input_reason_mentions_flag(self):
        reason = block_parallel.block_parallel_tuning_enabled(**{**self.BASE, "need_quanted_input": True})
        self.assertIn("--no-enable_quanted_input", reason)

    def test_worker_reason(self):
        os.environ["AR_BLOCK_PARALLEL_WORKER"] = "1"
        self.assertIn("worker", block_parallel.block_parallel_tuning_enabled(**self.BASE))

    def test_api_usage_reason(self):
        self.assertIn("command line", block_parallel.block_parallel_tuning_enabled(**{**self.BASE, "argv": None}))


class TestDeleteChainState(unittest.TestCase):
    def test_removes_and_tolerates_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            block_parallel.save_chain_state(d, 0, 5, {"h": torch.zeros(2)})
            self.assertTrue(block_parallel.chain_state_exists(d, 0, 5))
            block_parallel.delete_chain_state(d, 0, 5)
            self.assertFalse(block_parallel.chain_state_exists(d, 0, 5))
            # missing file is a no-op, not an error (double prune, crash leftovers)
            block_parallel.delete_chain_state(d, 0, 5)
            block_parallel.delete_chain_state(d, 3, 7)


class TestOrchestratorMethodRoster(unittest.TestCase):
    """The parent parallel path calls these on CompressionOrchestrator; a
    call-site-without-def slipped through once (method lost in a refactor,
    surfaced only on the server at Packing time) and killed a finished run."""

    def test_parent_path_methods_exist(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        for name in (
            "_maybe_block_parallel_tune",
            "_apply_tuned_results",
            "_collect_tuned_layers",
            "_dump_parallel_worker_results",
            "_serve_block_queue",
            "_manifest_frontier",
            "_ff_one_block",
            "_quantize_blocks",
        ):
            self.assertTrue(hasattr(CompressionOrchestrator, name), name)


if __name__ == "__main__":
    unittest.main()


class TestBlockResultsComplete(unittest.TestCase):
    def test_empty_file_is_not_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            block_parallel.save_block_results(d, "blk", {})  # empty payload + _worker_rank
            self.assertTrue(block_parallel.has_block_results(d, "blk"))
            self.assertFalse(block_parallel.block_results_complete(d, "blk"))
            block_parallel.save_block_results(d, "blk2", {"l": {"scale": torch.ones(2), "zp": None}})
            self.assertTrue(block_parallel.block_results_complete(d, "blk2"))
            self.assertFalse(block_parallel.block_results_complete(d, "missing"))
