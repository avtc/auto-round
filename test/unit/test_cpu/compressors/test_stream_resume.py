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
"""AR_RESUME_DIR support for the --stream_quantization (zero-shot) loop.

The per-group manifests are the shared currency: any execution mode (serial,
block-parallel, streaming) continues where another stopped. These tests pin
the three streaming-side pieces: the chain jump to the saved frontier entry,
the pending-offset arithmetic, and the done-block handler (pure skip vs
applying a BPT worker result).
"""

import torch

from auto_round.compressors.orchestrator import CompressionOrchestrator
from auto_round.utils.resume import ResumeState


class _QOffComposer:
    def need_quanted_input(self):
        return False


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dec0 = _ToyBlock()
        self.dec1 = _ToyBlock()


class TestStreamResumePendingOffset:
    def test_first_group_partial(self):
        states = [ResumeState.__new__(ResumeState)]
        states[0].completed_blocks = ["b0", "b1"]
        assert CompressionOrchestrator._stream_resume_pending_offset([["b0", "b1", "b2"]], states) == 2

    def test_all_done_returns_none(self):
        states = [ResumeState.__new__(ResumeState)]
        states[0].completed_blocks = ["b0", "b1"]
        assert CompressionOrchestrator._stream_resume_pending_offset([["b0", "b1"]], states) is None

    def test_multi_group_spillover(self):
        s0 = ResumeState.__new__(ResumeState)
        s0.completed_blocks = ["a", "b"]
        s1 = ResumeState.__new__(ResumeState)
        s1.completed_blocks = ["c"]
        off = CompressionOrchestrator._stream_resume_pending_offset([["a", "b"], ["c", "d"]], [s0, s1])
        assert off == 3  # first pending = "d" (flat index 3)


class TestStreamResumeJumpChain:
    def _orch(self):
        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        # alg_composer is a read-only property over _alg_composer (base.py)
        object.__setattr__(orch, "_alg_composer", _QOffComposer())
        return orch

    def test_jumps_to_deepest_entry(self, tmp_path):
        orch = self._orch()
        rs0 = ResumeState(str(tmp_path / "g0"), "sig", ["b0", "b1"])
        rs0.mark_block_done("b0", None, torch.ones(2, 2))
        rs0.mark_block_done("b1", None, torch.zeros(2, 2))  # deepest entry wins
        calib = {"fp_inputs": "stale", "input_others": {}}
        orch._stream_resume_jump_chain(calib, [rs0])
        assert torch.equal(calib["fp_inputs"], torch.zeros(2, 2))
        assert "q_inputs" not in calib  # qoff: q chain not restored

    def test_no_progress_is_noop(self, tmp_path):
        orch = self._orch()
        rs0 = ResumeState(str(tmp_path / "g0"), "sig", ["b0"])
        calib = {"fp_inputs": "stale", "input_others": {}}
        orch._stream_resume_jump_chain(calib, [rs0])
        assert calib["fp_inputs"] == "stale"

    def test_none_calib_state_safe(self, tmp_path):
        orch = self._orch()
        rs0 = ResumeState(str(tmp_path / "g0"), "sig", ["b0", "b1"])
        rs0.mark_block_done("b0", None, torch.ones(2, 2))
        orch._stream_resume_jump_chain(None, [rs0])  # weight-only streaming: no crash


class TestStreamResumeDoneBlock:
    def _orch(self, model):
        from types import SimpleNamespace

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model = model
        orch.device = "cpu"
        orch.shard_writer = None
        orch.nblocks = 1
        orch.compress_context = SimpleNamespace(
            is_immediate_packing=False, is_immediate_saving=False, low_cpu_mem_usage=False
        )
        return orch

    def test_pure_skip_when_no_result(self):
        model = _ToyModel()
        orch = self._orch(model)
        used = orch._stream_resume_done_block(
            "dec0",
            results_dir=None,
            streamer=None,
            stage_devices=None,
            stream_block_idx=0,
            pbar=None,
            tied_weights_layers=[],
        )
        assert used is False

    def test_applies_bpt_result(self, tmp_path):
        from auto_round.compressors import block_parallel as bp

        model = _ToyModel()
        orch = self._orch(model)
        scale = torch.full((4, 1), 0.5)
        result_path = bp.block_result_path(str(tmp_path), "dec0")
        import os

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        torch.save({"dec0.linear": {"scale": scale, "zp": 0}}, result_path)

        used = orch._stream_resume_done_block(
            "dec0",
            results_dir=str(tmp_path),
            streamer=None,
            stage_devices=None,
            stream_block_idx=0,
            pbar=None,
            tied_weights_layers=[],
        )
        assert used is True
        assert torch.equal(model.dec0.linear.scale, scale)
        assert model.dec0.linear.zp == 0

    def test_corrupt_result_fails_fast(self, tmp_path):
        from auto_round.compressors import block_parallel as bp

        import os

        model = _ToyModel()
        orch = self._orch(model)
        result_path = bp.block_result_path(str(tmp_path), "dec0")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "wb") as f:
            f.write(b"not a torch file")

        import pytest

        with pytest.raises(RuntimeError, match="unreadable"):
            orch._stream_resume_done_block(
                "dec0",
                results_dir=str(tmp_path),
                streamer=None,
                stage_devices=None,
                stream_block_idx=0,
                pbar=None,
                tied_weights_layers=[],
            )

    def test_key_mismatch_fails_fast(self, tmp_path):
        from auto_round.compressors import block_parallel as bp

        import os

        model = _ToyModel()
        orch = self._orch(model)
        result_path = bp.block_result_path(str(tmp_path), "dec0")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        torch.save({"totally.unrelated": {"scale": torch.ones(1), "zp": 0}}, result_path)

        import pytest

        with pytest.raises(RuntimeError, match="matched no layers"):
            orch._stream_resume_done_block(
                "dec0",
                results_dir=str(tmp_path),
                streamer=None,
                stage_devices=None,
                stream_block_idx=0,
                pbar=None,
                tied_weights_layers=[],
            )


class TestBPTResumeFromStreaming:
    """The BPT parent skips streaming-packed blocks (cross-mode resume)."""

    def _orch(self, model):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import _tuned_layer_key

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(model=model, amp_dtype=None)
        orch.compress_context = SimpleNamespace(
            is_immediate_packing=False, is_immediate_saving=False, low_cpu_mem_usage=False
        )
        orch.shard_writer = None
        orch.layer_config = {}
        return orch

    def test_apply_skips_manifest_done_blocks(self):
        model = _ToyModel()
        orch = self._orch(model)
        from auto_round.compressors.orchestrator import _tuned_layer_key

        scale = torch.full((4, 1), 0.25)
        tuned = {_tuned_layer_key("dec1", "linear", model.dec1.linear): {"scale": scale, "zp": 0}}

        # dec0 is streaming-packed (skipped): no result for it, no scale set,
        # and the expected-coverage validation must not flag its layers
        orch._apply_tuned_results([["dec0", "dec1"]], tuned, skip_blocks={"dec0"})

        assert torch.equal(model.dec1.linear.scale, scale)
        assert not hasattr(model.dec0.linear, "scale")

    def test_apply_raises_on_uncovered_pending_block(self):
        import pytest

        model = _ToyModel()
        # a to-pack layer: bits attr makes check_to_quantized True
        model.dec0.linear.bits = 4
        orch = self._orch(model)

        with pytest.raises(RuntimeError, match="missing from worker"):
            orch._apply_tuned_results([["dec0"]], {})
