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
"""Background ready-transforms (default ON; AR_DISABLE_BG_READY_TRANSFORMS opts out).

While block N tunes on its ping-pong group, the next block is early-loaded
on the idle group's home device and its weight-only layer-wise transforms
run there off the critical path. Tests cover the streamer wait helpers, the
composer runner (lock + no-op semantics) and the orchestrator worker's
success/failure contract.
"""

import threading

import torch
from torch import nn

from auto_round.compressors.orchestrator import CompressionOrchestrator
from auto_round.utils.checkpoint_streamer import CheckpointStreamer


class _StubTransform:
    def __init__(self):
        self.calls = []

    def rotate_layer(self, layer, layer_idx=None, **kwargs):
        self.calls.append((layer, layer_idx))


class _FakeStreamer:
    """Minimal streamer double for the background worker."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.waits = []
        self.released = []
        self.fail_wait = False

    def prefix_bytes(self, prefix):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def wait_until_staged(self, prefix, timeout=None):
        self.waits.append(prefix)
        return not self.fail_wait

    def load_module_(self, block, prefix, device=None):
        # meta-safe leaf swap, mirroring CheckpointStreamer._assign_leaf_
        by_short = dict(block.named_parameters(recurse=True))
        by_short.update(dict(block.named_buffers(recurse=True)))
        for name, tensor in self.tensors.items():
            short = name[len(prefix) + 1 :] if prefix else name
            if short in by_short and isinstance(by_short[short], nn.Parameter):
                parent = block
                parts = short.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                parent._parameters[parts[-1]] = nn.Parameter(tensor.clone(), requires_grad=False)

    def release_replicas(self, prefix):
        self.released.append(prefix)


class TestBgReadyEnv:
    def test_default_on_and_opt_out(self, monkeypatch):
        from auto_round import envs

        monkeypatch.delenv("AR_DISABLE_BG_READY_TRANSFORMS", raising=False)
        assert envs.AR_DISABLE_BG_READY_TRANSFORMS is False
        monkeypatch.setenv("AR_DISABLE_BG_READY_TRANSFORMS", "1")
        assert envs.AR_DISABLE_BG_READY_TRANSFORMS is True


class TestComposerReadyRunner:
    def _composer(self, transforms):
        from auto_round.algorithms.composer import AlgorithmComposer

        comp = AlgorithmComposer.__new__(AlgorithmComposer)
        comp._rotation_transforms = transforms
        comp._ready_transform_lock = threading.Lock()
        return comp

    def _ctx(self, block, name="blk"):
        from auto_round.algorithms.composer import BlockContext

        return BlockContext(model=None, block_names=[name], block_name=name, block_index=0)

    def test_runner_applies_transforms_under_lock(self):
        stub = _StubTransform()
        comp = self._composer([stub])
        seen = {}
        orig = stub.rotate_layer

        def _probe(layer, layer_idx=None, **kw):
            seen["locked"] = comp._ready_transform_lock.acquire(blocking=False)
            if seen["locked"]:
                comp._ready_transform_lock.release()
            orig(layer, layer_idx=layer_idx, **kw)

        stub.rotate_layer = _probe
        block = nn.Linear(4, 4)
        comp.run_ready_transforms(block, self._ctx(block))
        assert len(stub.calls) == 1 and stub.calls[0][0] is block
        assert seen["locked"] is False  # the runner held the lock throughout

    def test_runner_noop_without_transforms(self):
        comp = self._composer([])
        comp.run_ready_transforms(nn.Linear(4, 4), self._ctx(nn.Linear(4, 4)))
        # no raise, nothing to assert beyond reaching here


class TestBgReadyWorker:
    def _orchestrator(self, model, transforms):
        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model = model

        class _Composer:
            has_layerwise_rotation = True

            def run_ready_transforms(self, block, ctx):
                for t in transforms:
                    t.rotate_layer(block, layer_idx=ctx.block_index)

        orch._alg_composer = _Composer()
        return orch

    def test_worker_loads_and_transforms_next_block(self):
        model = nn.Sequential()
        blk = nn.Linear(4, 4)
        model.add_module("blk", blk)
        raw = {"blk.weight": torch.randn(4, 4), "blk.bias": torch.randn(4)}
        streamer = _FakeStreamer(raw)
        stub = _StubTransform()
        orch = self._orchestrator(model, [stub])
        # simulate the streaming state: block leaves are meta
        model.blk._parameters["weight"] = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
        model.blk._parameters["bias"] = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)

        t = orch._start_bg_ready_transform("blk", torch.device("cpu"), streamer)
        t.join(timeout=30)
        assert not t.is_alive()
        assert getattr(model.blk, "_bg_ready_done", False) is True
        assert streamer.waits == ["blk"] and streamer.released == ["blk"]
        assert model.blk.weight.data.device.type == "cpu"
        torch.testing.assert_close(model.blk.weight.data, raw["blk.weight"])
        assert len(stub.calls) == 1 and stub.calls[0][0] is model.blk

    def test_worker_failure_marks_block_and_recovers_in_main_path(self):
        model = nn.Sequential()
        blk = nn.Linear(4, 4)
        model.add_module("blk", blk)
        streamer = _FakeStreamer({"blk.weight": torch.randn(4, 4), "blk.bias": torch.randn(4)})
        streamer.fail_wait = True  # staging never arrives
        orch = self._orchestrator(model, [_StubTransform()])

        t = orch._start_bg_ready_transform("blk", torch.device("cpu"), streamer)
        t.join(timeout=30)
        assert not t.is_alive()
        assert getattr(blk, "_bg_ready_done", False) is False
        assert getattr(blk, "_bg_ready_failed", None)
        assert streamer.released == []  # replicas kept for the main path


class TestStreamerWaitHelpers:
    def test_wait_true_without_prefetch_thread(self, tmp_path):
        st = CheckpointStreamer.__new__(CheckpointStreamer)
        st._prefetch_thread = None
        assert st.wait_until_staged("blk") is True

    def test_wait_false_when_stopped(self):
        st = CheckpointStreamer.__new__(CheckpointStreamer)
        st._prefetch_thread = threading.Thread(target=lambda: None)
        st._prefetch_stop = True
        st._prefetch_err = None
        st._prefetch_staged = []
        st._prefetch_cond = threading.Condition()
        assert st.wait_until_staged("blk", timeout=0.1) is False

    def test_prefix_bytes_counts_tensors(self, tmp_path, monkeypatch):
        import json
        import os

        from safetensors.torch import save_file

        tensors = {"blk.a.weight": torch.zeros(8, 8, dtype=torch.bfloat16)}
        save_file(tensors, os.path.join(tmp_path, "model.safetensors"))
        with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in tensors}}, f)
        st = CheckpointStreamer(str(tmp_path))
        assert st.prefix_bytes("blk.a") == 8 * 8 * 2


class TestPrefetchReplicaEligible:
    def _composer(self, rotation, preprocessors):
        from types import SimpleNamespace

        return SimpleNamespace(has_layerwise_rotation=rotation, preprocessors=preprocessors)

    def test_plain_ddp_eligible(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        assert CompressionOrchestrator._prefetch_replica_eligible(4, ["cuda:0", "cuda:4"], self._composer(False, []))

    def test_rotation_or_preprocessors_skip_fanout(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        # rotation active: staged raw copies can never be adopted
        assert not CompressionOrchestrator._prefetch_replica_eligible(4, ["cuda:0", "cuda:4"], self._composer(True, []))
        # preprocessor (AWQ) mutates weights in-loop: adoption equally dead
        assert not CompressionOrchestrator._prefetch_replica_eligible(
            4, ["cuda:0", "cuda:4"], self._composer(False, [_StubTransform()])
        )

    def test_world_or_devices_gates(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        assert not CompressionOrchestrator._prefetch_replica_eligible(1, ["cuda:0"], self._composer(False, []))
        assert not CompressionOrchestrator._prefetch_replica_eligible(4, [], self._composer(False, []))
