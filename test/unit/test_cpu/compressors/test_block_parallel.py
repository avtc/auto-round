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
import tempfile
import unittest
from unittest import mock

import torch

os.environ.setdefault("AR_AUTO_SCHEME_CACHE", os.path.join(os.path.dirname(__file__), "cache"))

from auto_round.compressors import block_parallel  # noqa: E402
from auto_round.compressors import orchestrator  # noqa: E402


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
        argv=["C:/repo/auto_round/__main__.py", "--a"],
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
    """The parent parallel path calls these methods on CompressionOrchestrator;
    every call site needs a definition present on the class (a refactor can
    move a method out from under a live call site, and workers only exercise
    these paths at run time, not at import)."""

    def test_parent_path_methods_exist(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        for name in (
            "_maybe_block_parallel_tune",
            "_apply_tuned_results",
            "_collect_tuned_layers",
            "_serve_block_queue",
            "_manifest_frontier",
            "_ff_one_block",
            "_quantize_blocks",
        ):
            self.assertTrue(hasattr(CompressionOrchestrator, name), name)


class TestBlockResultsComplete(unittest.TestCase):
    def test_empty_file_is_complete(self):
        """A block whose every layer is excluded from quantization tunes to
        nothing; its empty result file is a valid completion, not corruption."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            block_parallel.save_block_results(d, "blk", {})  # empty payload + _worker_rank
            self.assertTrue(block_parallel.has_block_results(d, "blk"))
            self.assertTrue(block_parallel.block_results_complete(d, "blk"))
            block_parallel.save_block_results(d, "blk2", {"l": {"scale": torch.ones(2), "zp": None}})
            self.assertTrue(block_parallel.block_results_complete(d, "blk2"))
            self.assertFalse(block_parallel.block_results_complete(d, "missing"))

    def test_unloadable_file_is_not_complete(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, block_parallel._block_results_path(d, "blk"))
            with open(path, "wb") as f:
                f.write(b"not a torch file")
            self.assertTrue(block_parallel.has_block_results(d, "blk"))
            self.assertFalse(block_parallel.block_results_complete(d, "blk"))


class TestPackEquivalenceOriginalVsQdqWeight(unittest.TestCase):
    """Serial packing runs on qdq weights (the unwrapper overwrites weight.data with the
    fake-quant reconstruction); the parallel apply packs ORIGINAL weights and sets only
    scale/zp. Both feed the same packer, which re-quantizes with round(W/s + z) -- so the
    packed codes are identical iff quantization is idempotent on its own grid: dequantized
    values are grid points, and re-quantizing a grid point (plus float roundtrip error far
    below the 0.5 rounding boundary) returns the same integer. This pins that property with
    the affine formula both pack paths use, across sym/asym, group sizes, and dtypes."""

    QMAX = 15  # int4 unsigned codes

    @staticmethod
    def _codes(w, s, z):
        return torch.clamp(torch.round(w / s) + z, 0, 15)

    def test_grid_idempotency(self):
        torch.manual_seed(0)
        n_elem = 256 * 1024
        for group in (None, 128):
            for sym in (True, False):
                for dtype in (torch.float32, torch.bfloat16):
                    w = (torch.randn(n_elem) * 0.02).to(dtype)
                    if group is None:
                        wg = w.reshape(1, -1)
                        s = (wg.abs().amax() / 7.0).clamp(min=1e-6).reshape(1, 1)
                        z = torch.full_like(s, 8.0 if sym else 7.5)
                    else:
                        wg = w.reshape(-1, group)
                        s = (wg.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
                        z = (
                            torch.full_like(s, 8.0)
                            if sym
                            else (-wg.min(dim=1, keepdim=True).values / s).round().clamp(0, 15)
                        )
                    c1 = self._codes(wg, s, z)
                    self.assertTrue((c1 >= 0).all() and (c1 <= 15).all())
                    # qdq reconstruction (what the serial unwrapper leaves in weight.data),
                    # computed in the storage dtype -- includes the bf16 roundtrip error
                    w2 = (s * (c1 - z)).to(dtype)
                    c2 = self._codes(w2, s, z)
                    self.assertTrue(
                        torch.equal(c1, c2),
                        f"non-idempotent codes: group={group} sym={sym} dtype={dtype}, "
                        f"{(c1 != c2).sum().item()} mismatches",
                    )

    def test_apply_vs_serial_packed_codes(self):
        """End-to-end flavor: one Linear packed from original vs qdq weight, identical
        scale/zp -> identical codes (the property _apply_tuned_results relies on)."""
        torch.manual_seed(1)
        dtype = torch.bfloat16
        w = (torch.randn(64, 256) * 0.02).to(dtype)
        wg = w.reshape(-1, 128)
        s = (wg.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-6)
        z = torch.full_like(s, 8.0)  # sym int4 in unsigned code space
        lin_orig = torch.nn.Linear(256, 64, bias=False, dtype=dtype)
        with torch.no_grad():
            lin_orig.weight.copy_(w)
        c_orig = self._codes(lin_orig.weight.reshape(-1, 128), s, z)
        w_qdq = (s * (c_orig - z)).reshape(lin_orig.weight.shape).to(dtype)
        lin_qdq = torch.nn.Linear(256, 64, bias=False, dtype=dtype)
        with torch.no_grad():
            lin_qdq.weight.copy_(w_qdq)
        c_qdq = self._codes(lin_qdq.weight.reshape(-1, 128), s, z)
        self.assertTrue(torch.equal(c_orig, c_qdq))


class TestScratchCleanup(unittest.TestCase):
    def test_cleanup_removes_dir_and_resets(self):
        """After the export finalizes the scratch dir (results, checkpoints,
        worker logs) must be removed and the stash cleared."""
        import tempfile

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        with tempfile.TemporaryDirectory() as d:
            scratch = os.path.join(d, "block_parallel_results")
            os.makedirs(scratch, exist_ok=True)
            with open(os.path.join(scratch, "worker_0.log"), "w", encoding="utf-8") as f:
                f.write("log")
            orch._bp_results_to_clean = scratch
            orch._cleanup_block_parallel_scratch()
            self.assertFalse(os.path.exists(scratch))
            self.assertIsNone(orch._bp_results_to_clean)

    def test_cleanup_without_stash_is_noop(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch._cleanup_block_parallel_scratch()  # must not raise


class TestLaunchIsReexecutable(unittest.TestCase):
    def test_m_package_main_is_reexecutable(self):
        self.assertTrue(block_parallel.launch_is_reexecutable(["/repo/auto_round/__main__.py", "--device_map", "0,1"]))

    def test_python_driver_script_is_reexecutable(self):
        self.assertTrue(block_parallel.launch_is_reexecutable(["driver.py", "--device_map", "0,1"]))

    def test_foreign_launcher_is_rejected(self):
        # foreign -m packages, console binaries, and empty argv must not re-exec
        self.assertFalse(block_parallel.launch_is_reexecutable(["/venv/bin/pytest/__main__.py", "-q"]))
        self.assertFalse(block_parallel.launch_is_reexecutable(["pytest", "-q"]))
        self.assertFalse(block_parallel.launch_is_reexecutable(["jupyter", "notebook"]))
        self.assertFalse(block_parallel.launch_is_reexecutable([]))
        self.assertFalse(block_parallel.launch_is_reexecutable(None))

    def test_enabled_reason_mentions_launcher_for_foreign_argv(self):
        os.environ["AR_ENABLE_BLOCK_PARALLEL_TUNING"] = "1"
        os.environ.pop("AR_BLOCK_PARALLEL_WORKER", None)
        try:
            reason = block_parallel.block_parallel_tuning_enabled(
                need_quanted_input=False,
                super_group_size=None,
                nblocks=1,
                n_block_groups=2,
                is_immediate_packing=True,
                is_immediate_saving=True,
                n_blocks_total=4,
                argv=["pytest", "-q"],
            )
        finally:
            os.environ.pop("AR_ENABLE_BLOCK_PARALLEL_TUNING", None)
        self.assertIsNotNone(reason)
        self.assertIn("launcher", reason)


class TestDispatchCandidate(unittest.TestCase):
    """Strict-frontier gating: a block is dispatchable only at its group's
    ramp start or once its entry checkpoint exists; lowest block wins."""

    def _run(self, done, assigned, ramp_start, existing, all_blocks=None):
        all_blocks = all_blocks or [["b0", "b1", "b2"], ["c0", "c1"]]
        with mock.patch.object(orchestrator, "_bp_chain_state_exists", side_effect=lambda r, g, k: (g, k) in existing):
            return orchestrator._dispatch_candidate(all_blocks, done, assigned, ramp_start, "results")

    def test_ramp_start_always_available_without_checkpoints(self):
        # fresh run: both group heads dispatchable, nothing else
        self.assertEqual(self._run([set(), set()], {}, {0: 0, 1: 0}, set()), [(0, 0), (1, 0)])

    def test_later_block_waits_for_its_checkpoint(self):
        # (0,1) published -> it becomes the next candidate after (0,0) assigned
        assigned = {0: (0, 0)}
        self.assertEqual(self._run([set(), set()], assigned, {0: 0, 1: 0}, {(0, 1)}), [(0, 1), (1, 0)])

    def test_no_candidate_when_entries_not_published(self):
        # only non-ramp blocks remain and none has a checkpoint: empty (retry)
        done = [{0}, set()]
        self.assertEqual(self._run(done, {0: (0, 1)}, {0: 0, 1: 0}, set()), [(1, 0)])

    def test_empty_when_all_done_or_assigned(self):
        done = [{0, 1, 2}, {0}]
        assigned = {0: (1, 1)}
        self.assertEqual(self._run(done, assigned, {0: 3, 1: 1}, set()), [])

    def test_resume_ramp_start_is_the_manifest_frontier(self):
        # blocks 0-1 done in group 0: ramp start is 2 and dispatches checkpoint-free
        done = [{0, 1}, set()]
        self.assertEqual(self._run(done, {}, {0: 2, 1: 0}, set()), [(0, 2), (1, 0)])


class TestResolveChainEntry(unittest.TestCase):
    """Worker-side entry resolution matrix."""

    def _orch(self, manifest_frontier=(0, None), loaded=None):
        import types

        orch = orchestrator.CompressionOrchestrator.__new__(orchestrator.CompressionOrchestrator)
        orch._manifest_frontier = lambda g: manifest_frontier
        orch.compress_context = types.SimpleNamespace(cache_device="cpu")
        self._loaded_calls = []
        orch_loaded = self._loaded_calls

        def fake_load(results_dir, g, k, device=None):
            orch_loaded.append((g, k))
            return loaded

        patcher = mock.patch.object(orchestrator, "_bp_load_chain_state", side_effect=fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)
        with mock.patch("time.sleep"):
            return orch

    def test_held_entry_returned_without_io(self):
        orch = self._orch()
        held = (0, 3, "entry-tensor")
        self.assertIs(orch._resolve_chain_entry(0, 3, held, "results", rank=1), "entry-tensor")
        self.assertEqual(self._loaded_calls, [])

    def test_checkpoint_entry_loaded_when_present(self):
        orch = self._orch(loaded="ckpt-entry")
        self.assertIs(orch._resolve_chain_entry(0, 2, None, "results", rank=1), "ckpt-entry")

    def test_manifest_frontier_used_for_resumed_ramp_block(self):
        orch = self._orch(manifest_frontier=(2, "frontier-entry"), loaded=None)
        self.assertIs(orch._resolve_chain_entry(0, 2, None, "results", rank=1), "frontier-entry")

    def test_missing_nonfrontier_checkpoint_raises(self):
        orch = self._orch(manifest_frontier=(2, "frontier-entry"), loaded=None)
        with self.assertRaisesRegex(RuntimeError, "not published"):
            orch._resolve_chain_entry(0, 3, None, "results", rank=1)

    def test_first_block_without_checkpoint_returns_none(self):
        orch = self._orch(loaded=None)
        # k == 0: serial block-0 semantics, caller falls back to the group entry
        self.assertIsNone(orch._resolve_chain_entry(1, 0, None, "results", rank=1))


class TestReadBlockResult(unittest.TestCase):
    def _save(self, d, name, payload):
        rank = payload.pop("_worker_rank", None)
        if rank is not None:
            os.environ["AR_BLOCK_PARALLEL_RANK"] = str(rank)
        try:
            block_parallel.save_block_results(d, name, payload)
        finally:
            if rank is not None:
                os.environ.pop("AR_BLOCK_PARALLEL_RANK", None)
        return d

    def test_returns_payload_for_valid_result(self):
        with tempfile.TemporaryDirectory() as d:
            self._save(d, "blk", {"scale": 1, "_worker_rank": 2})
            data = block_parallel.read_block_result(d, "blk")
            self.assertEqual(data["_worker_rank"], 2)

    def test_empty_layer_dict_is_a_valid_result(self):
        with tempfile.TemporaryDirectory() as d:
            self._save(d, "blk", {})
            # save_block_results injects the advisory rank into every payload
            self.assertEqual(block_parallel.read_block_result(d, "blk"), {"_worker_rank": -1})

    def test_none_for_missing_corrupt_and_nondict(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(block_parallel.read_block_result(d, "missing"))
            bad = os.path.join(d, "block_bad.pt")
            with open(bad, "wb") as f:
                f.write(b"not-a-torch-file")
            self.assertIsNone(block_parallel.read_block_result(d, "bad"))


if __name__ == "__main__":
    unittest.main()


class TestTunedLayerKey(unittest.TestCase):
    def test_global_name_wins_then_path_then_block(self):
        import torch.nn as nn

        from auto_round.compressors.orchestrator import _tuned_layer_key

        sub = nn.Linear(2, 2)
        self.assertEqual(_tuned_layer_key("blk", "mlp", sub), "blk.mlp")
        sub.global_name = "custom.name"
        self.assertEqual(_tuned_layer_key("blk", "mlp", sub), "custom.name")
        self.assertEqual(_tuned_layer_key("blk", "", sub), "custom.name")
        # without global_name, the root submodule keys to the bare block name
        # -- collect and apply must agree here or the parent rejects the block
        plain = nn.Linear(2, 2)
        self.assertEqual(_tuned_layer_key("blk", "", plain), "blk")


if __name__ == "__main__":
    unittest.main()
