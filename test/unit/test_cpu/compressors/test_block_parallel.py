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


def _RealRTNConfig():
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    return RTNConfig(bits=4, group_size=32, data_type="int")


class TestImmediatePackingWithOutsideBlockQlayer(unittest.TestCase):
    """Quant layers outside blocks (e.g. lm_head W8) must no longer disable
    immediate packing on the data-driven path: the outside-block loop runs
    after the block loop and packs/saves at export (proven by the streaming
    flow with the same configuration). The old rule also made BPT impossible
    and forced env-var hy3 runs into a full-model reload at final assembly."""

    def _adjust(self, need_calib, has_qlayer_outside_block, inplace=True):
        from types import SimpleNamespace

        from auto_round.compressors.base import BaseCompressor

        fmt = SimpleNamespace(
            is_fake=lambda: False,
            is_gguf=lambda: False,
            is_supported_immediate_packing=lambda: True,
            is_supported_immediate_saving=lambda: True,
        )
        ctx = SimpleNamespace(is_immediate_packing=False, is_immediate_saving=False, low_cpu_mem_usage=True)
        stub = SimpleNamespace(
            formats=[fmt],
            inplace=inplace,
            has_qlayer_outside_block=property(lambda self: has_qlayer_outside_block),
            need_calib=need_calib,
            compress_context=ctx,
            model_context=SimpleNamespace(
                model=SimpleNamespace(
                    __class__=type("Qwen2ForCausalLM", (), {}),
                    _tied_weight_keys={},
                ),
                is_mllm=False,
            ),
            quantize_config=_RealRTNConfig(),
            disable_opt_rtn=False,
            output_dir="out",
        )
        stub._ensure_shard_writer = lambda: None
        BaseCompressor._adjust_immediate_packing_and_saving(stub)
        return ctx

    def test_data_driven_outside_block_qlayer_keeps_immediate_packing(self):
        ctx = self._adjust(need_calib=True, has_qlayer_outside_block=True)
        assert ctx.is_immediate_packing is True, "outside-block qlayers must not disable immediate packing"
        assert ctx.is_immediate_saving is True

    def test_data_driven_outside_block_qlayer_keeps_inplace(self):
        """The phase-4 method itself must not flip inplace=False for the same
        configuration _adjust_immediate_packing_and_saving handles: outside-block
        qlayers + need_calib (data-driven AutoScheme run with a pinned lm_head).
        inplace=False starves the packing=True assignment and blocks BPT with
        'requires immediate packing and saving'."""
        from types import SimpleNamespace

        from auto_round.compressors.base import BaseCompressor

        fmt = SimpleNamespace(
            is_fake=lambda: False,
            is_gguf=lambda: False,
            is_supported_immediate_packing=lambda: True,
            is_supported_immediate_saving=lambda: True,
        )
        ctx = SimpleNamespace(is_immediate_packing=False, is_immediate_saving=False, low_cpu_mem_usage=True)
        stub = SimpleNamespace(
            formats=[fmt],
            inplace=True,
            has_qlayer_outside_block=property(lambda self: True),
            need_calib=True,
            compress_context=ctx,
            model_context=SimpleNamespace(
                model=SimpleNamespace(
                    __class__=type("Qwen2ForCausalLM", (), {}),
                    _tied_weight_keys={},
                ),
                is_mllm=False,
            ),
            quantize_config=_RealRTNConfig(),
            disable_opt_rtn=False,
            output_dir="out",
        )
        stub._ensure_shard_writer = lambda: None
        stub._offloader = SimpleNamespace(reset=lambda: None)
        stub._finalize_torch_compile = lambda: None
        stub.enable_torch_compile = False
        stub._adjust_immediate_packing_and_saving = lambda: BaseCompressor._adjust_immediate_packing_and_saving(stub)
        import auto_round.compressors.base as base_mod

        base_mod.set_non_auto_device_map(stub.model_context.model, None)
        BaseCompressor._hardware_setup(stub)
        assert stub.inplace is True, "data-driven outside-block qlayers must not disable inplace"
        assert ctx.is_immediate_packing is True
        assert ctx.is_immediate_saving is True

    def test_no_qlayer_outside_block_still_packs(self):
        ctx = self._adjust(need_calib=True, has_qlayer_outside_block=False)
        assert ctx.is_immediate_packing is True


class TestBPChainPruneGate(unittest.TestCase):
    """Regression: adjacent blocks 2 and 3 finished in the same poll tick;
    the results scan pruned chain_g0_b3.pt on block 3's completion, and the
    in-order manifest commit for block 2 -- which hard-links exactly that
    file -- then died with FileNotFoundError. chain_g{k} may only be pruned
    once the manifest frontier has consumed it (block k-1 committed)."""

    def test_not_pruned_before_manifest_consumes_it(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        # the exact crash state: manifest at block 2, block 3 completed
        assert CompressionOrchestrator._bp_prune_chain_entry(3, resumable=True, resume_index=2) is False

    def test_pruned_after_manifest_passes(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        # frontier committed through block 3 -> chain_g0_b3 was linked at the
        # commit of block 2 and again consumed; safe to prune
        assert CompressionOrchestrator._bp_prune_chain_entry(3, resumable=True, resume_index=4) is True
        # boundary: committing block 2 (frontier becomes 3) links chain_g0_b3
        assert CompressionOrchestrator._bp_prune_chain_entry(3, resumable=True, resume_index=3) is True

    def test_pruned_immediately_when_not_resumable(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        # no manifest exists: chain files only serve assignment-time loads
        assert CompressionOrchestrator._bp_prune_chain_entry(3, resumable=False, resume_index=0) is True


class TestBPRequiredVRAMEstimate(unittest.TestCase):
    """The per-worker VRAM requirement must match what the worker executes.

    Ground truth from the code paths (see orchestrator._bp_required_vram_bytes
    docstring): zero-shot (iters=0) holds the block once with chunk-capped
    search transients and batch_size-chunked forwards; iterative tuning
    (iters>0) additionally materializes fp32 rounding params + gradients for
    every wrapped layer of the block at once.
    """

    GiB = 1024**3

    def _hy3_block_bytes(self):
        # largest hy3 block (MTP layer): ~3.95B params bf16
        return int(3.95e9 * 2)

    def test_zero_shot_hy3_block_fits_a_3090(self):
        """Regression: the hy3 A/B run (iters=0) was refused with a 61.6 GiB
        estimate built from nsamples-sized fp32 activations and a x3 block
        multiplier; the real working set is ~2x the block."""
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        req = CompressionOrchestrator._bp_required_vram_bytes(
            self._hy3_block_bytes(), iters=0, batch_size=8, seqlen=2048, hidden=4096
        )
        assert req < 24 * self.GiB, f"hy3 zero-shot worker must fit a 3090, got {req / self.GiB:.1f} GiB"
        assert req > self._hy3_block_bytes(), "block weights must be included"
        # must not scale with nsamples (not even a parameter) and the search
        # transient is chunk-capped, not a multiple of the block
        assert req < self._hy3_block_bytes() * 2 + 6 * self.GiB

    def test_iterative_hy3_block_is_correctly_refused(self):
        """iters>0 wraps every layer with fp32 value params + grads
        (numel*4 each) -> ~5x the bf16 block: > any 3090 for a 3.95B MoE block."""
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        req = CompressionOrchestrator._bp_required_vram_bytes(
            self._hy3_block_bytes(), iters=20, batch_size=8, seqlen=2048, hidden=4096
        )
        assert req >= 5 * self._hy3_block_bytes(), "fp32 rounding params + grads must dominate"
        assert req > 24 * self.GiB, "a 3.95B-param MoE block cannot be tuned iteratively on one 3090"

    def test_iterative_27b_dense_block_fits(self):
        """0.42B-param dense block (the proven iters20/iters500 runs) must stay
        eligible at iters>0."""
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        block = int(0.42e9 * 2)
        req = CompressionOrchestrator._bp_required_vram_bytes(block, iters=200, batch_size=8, seqlen=2048, hidden=5120)
        assert req < 24 * self.GiB

    def test_activation_term_scales_with_batch_size(self):
        """Forwards are chunked by batch_size (BlockForwardRunner), so the GPU
        activation term follows the batch, never the full nsamples set."""
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        base = CompressionOrchestrator._bp_required_vram_bytes(
            self._hy3_block_bytes(), iters=0, batch_size=8, seqlen=2048, hidden=4096
        )
        doubled_batch = CompressionOrchestrator._bp_required_vram_bytes(
            self._hy3_block_bytes(), iters=0, batch_size=16, seqlen=2048, hidden=4096
        )
        assert doubled_batch > base
        doubled_seq = CompressionOrchestrator._bp_required_vram_bytes(
            self._hy3_block_bytes(), iters=0, batch_size=8, seqlen=4096, hidden=4096
        )
        assert doubled_seq > base

    def test_search_transient_is_chunk_capped(self):
        """Even a hypothetical 100 GiB block gets a bounded search transient at
        iters=0 (the expert search chunks at ~2^28 elements per call)."""
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        huge = 100 * self.GiB
        req = CompressionOrchestrator._bp_required_vram_bytes(huge, iters=0, batch_size=8, seqlen=2048, hidden=4096)
        # 2.5 GiB search cap + act transients + 2 GiB margin = ~6.5 GiB; a
        # block-proportional multiplier (the old x3) would add 200 GiB here
        assert req < huge + 7 * self.GiB, "zero-shot transient must be capped, not block-proportional"


class TestGuardReasons(unittest.TestCase):
    """Guard returns None (ok), 'disabled' (off), or a reason (on, blocked)."""

    BASE = dict(
        enabled=True,
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
        os.environ.pop("AR_BLOCK_PARALLEL_WORKER", None)
        os.environ.pop("AR_RESUME_DIR", None)

    def test_enabled_returns_none(self):
        self.assertIsNone(block_parallel.block_parallel_tuning_enabled(**self.BASE))

    def test_disabled_returns_disabled(self):
        self.assertEqual(block_parallel.block_parallel_tuning_enabled(**{**self.BASE, "enabled": False}), "disabled")

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
        os.environ.pop("AR_BLOCK_PARALLEL_WORKER", None)
        reason = block_parallel.block_parallel_tuning_enabled(
            enabled=True,
            need_quanted_input=False,
            super_group_size=None,
            nblocks=1,
            n_block_groups=2,
            is_immediate_packing=True,
            is_immediate_saving=True,
            n_blocks_total=4,
            argv=["pytest", "-q"],
        )
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


class TestAssertNoCpuOffload:
    """Data-driven runs fail fast when accelerate CPU-offloaded part of the model."""

    def test_raises_on_cpu_offload(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(
            model=SimpleNamespace(hf_device_map={"model.layers.0": "cuda:0", "model.layers.31": "cpu"})
        )
        import pytest

        with pytest.raises(RuntimeError, match="full GPU residency"):
            orch._assert_no_cpu_offload()

    def test_passes_when_fully_resident(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(
            model=SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.31": 1})
        )
        orch._assert_no_cpu_offload()  # must not raise

    def test_passes_without_device_map(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(model=SimpleNamespace())
        orch._assert_no_cpu_offload()  # must not raise


class TestStripStaleDeviceHooks:
    """Streamed blocks lose accelerate hooks that remember pre-streaming devices."""

    def test_strips_hooked_modules_only(self, monkeypatch):
        import torch

        import auto_round.compressors.utils as cu

        removed = []

        def fake_remove(mod):
            removed.append(mod)
            if hasattr(mod, "_hf_hook"):
                del mod._hf_hook

        import accelerate.hooks  # noqa: F401  ensure import path exists

        monkeypatch.setattr("accelerate.hooks.remove_hook_from_module", fake_remove)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        block[0]._hf_hook = object()  # only the first module is hooked
        n = cu.strip_stale_device_hooks_(block)
        assert n == 1
        assert removed == [block[0]]
        assert not hasattr(block[0], "_hf_hook")

    def test_noop_without_hooks(self):
        import torch

        from auto_round.compressors.utils import strip_stale_device_hooks_

        block = torch.nn.Linear(4, 4)
        assert strip_stale_device_hooks_(block) == 0


class TestInputOthersAlwaysMoved:
    """block_forward moves kwargs even when hidden states are already placed."""

    def test_to_device_recurses_pe_tuples(self):
        import torch

        from auto_round.utils.model import to_device

        cos, sin = torch.ones(2), torch.ones(2)
        others = {"position_embeddings": [(cos, sin)], "attention_mask": [torch.ones(2)]}
        moved = to_device(others, torch.device("cpu"))
        # identity for on-device tensors - masters untouched, structure intact
        assert moved["position_embeddings"][0][0] is cos


class TestRehomeBlock:
    """Shared/setup modules reachable from a streamed block move onto its home."""

    def test_moves_foreign_tensor_leaves_meta(self):
        import torch

        from auto_round.compressors.utils import rehome_block_

        block = torch.nn.Linear(4, 4)  # cpu params
        block.register_buffer("meta_tbl", torch.ones(4, device="meta"))
        moved = rehome_block_(block, torch.device("meta"))
        assert moved == 2  # weight + bias moved cpu -> meta
        assert block.weight.device.type == "meta"
        assert block.meta_tbl.device.type == "meta"  # was already there, untouched

    def test_noop_when_placed(self):
        import torch

        from auto_round.compressors.utils import rehome_block_

        block = torch.nn.Linear(4, 4)
        assert rehome_block_(block, torch.device("cpu")) == 0
        assert block.weight.device.type == "cpu"


class TestStreamMemInventory:
    """AR_STREAM_MEM_INVENTORY: bucket names + graceful no-CUDA behavior."""

    def test_mem_bucket_names(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        b = CompressionOrchestrator._mem_bucket
        assert b("model.language_model.layers.12.self_attn.q_proj.weight") == "block:12"
        assert b("model.embed_tokens.weight") == "embeddings"
        assert b("lm_head.weight") == "nonblock:lm_head"

    def test_inventory_noop_without_cuda(self):
        from types import SimpleNamespace

        import torch

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model = torch.nn.Linear(4, 4)  # cpu-only box: loop body skips cuda tensors
        orch._log_device_inventory({"fp_inputs": [torch.ones(2)]}, "test")  # must not raise


class TestAutoStagingPrimaryFit:
    """auto staging adds the primary only when the largest block fits free VRAM."""

    def test_largest_block_bytes_from_meta(self):
        import torch

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [torch.nn.ModuleDict({"w": torch.nn.Linear(4, 4, device="meta")}) for _ in range(2)]
                )
                self.blocks[1].big = torch.nn.Linear(16, 16, device="meta")

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model = FakeModel()
        got = orch._largest_block_bytes()
        want = (16 * 16 + 16) * 4 + (4 * 4 + 4) * 4  # big block weight+bias dominates
        assert abs(got - want) < 1, (got, want)

    def test_primary_fit_none_without_cuda(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        assert orch._primary_fits_largest_block(torch.device("cpu")) is None
