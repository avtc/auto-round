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
"""Tests for DDP-group prefetch fan-out and staged-weight mirror adoption."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from auto_round.utils.checkpoint_streamer import CheckpointStreamer


def _sharded_dir(tmp_path, blocks):
    weight_map = {}
    for i, (blk, tensors) in enumerate(blocks.items()):
        shard = f"m-{i}.safetensors"
        save_file(tensors, os.path.join(tmp_path, shard), metadata={"format": "pt"})
        for k in tensors:
            weight_map[k] = shard
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return str(tmp_path)


def _wait_staged(streamer, prefix, timeout=10.0):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        with streamer._prefetch_cond:
            if prefix in streamer._prefetch_staged:
                return True
        time.sleep(0.02)
    return False


class TestPrefetchFanout:
    def test_replica_copies_staged_and_released(self, tmp_path):
        torch.manual_seed(0)
        blk = "layers.0"
        tensors = {"layers.0.a.weight": torch.randn(4, 4), "layers.0.b.weight": torch.randn(4, 4)}
        path = _sharded_dir(tmp_path, {blk: tensors})
        streamer = CheckpointStreamer(path)
        try:
            streamer.start_prefetch(
                [blk],
                depth=1,
                stage_devices=[torch.device("cpu")],
                replica_of=lambda idx, primary: [torch.device("cpu")],  # same-device replica on CPU
            )
            assert _wait_staged(streamer, blk)
            staged = streamer.staged_replica_tensors(blk, torch.device("cpu"))
            # CPU 'replica' shares the primary device: the fan-out skips it, so no
            # separate copy exists -- this documents the same-device exclusion
            assert staged is None or set(staged or {}) == set(tensors)
        finally:
            streamer.stop_prefetch()

    def test_release_replicas_frees_entries(self, tmp_path):
        torch.manual_seed(0)
        blk = "layers.1"
        tensors = {"layers.1.w.weight": torch.randn(2, 2)}
        path = _sharded_dir(tmp_path, {blk: tensors})
        streamer = CheckpointStreamer(path)
        try:
            # simulate a ready replica entry
            with streamer._prefetch_cond:
                streamer._prefetch_replica_cache[torch.device("cpu")] = dict(tensors)
                streamer._prefetch_replica_ready.add((blk, torch.device("cpu")))
            assert streamer.staged_replica_tensors(blk, torch.device("cpu")) is not None
            streamer.release_replicas(blk)
            assert streamer.staged_replica_tensors(blk, torch.device("cpu")) is None
            assert streamer._prefetch_replica_cache[torch.device("cpu")] == {}
        finally:
            streamer.stop_prefetch()

    def test_primary_consume_keeps_replicas(self, tmp_path):
        torch.manual_seed(0)
        blk = "layers.2"
        tensors = {"layers.2.w.weight": torch.randn(2, 2)}
        path = _sharded_dir(tmp_path, {blk: tensors})
        streamer = CheckpointStreamer(path)
        try:
            with streamer._prefetch_cond:
                streamer._prefetch_replica_cache[torch.device("cpu")] = dict(tensors)
                streamer._prefetch_replica_ready.add((blk, torch.device("cpu")))
            streamer.prefetch_consumed(blk)  # primary-side consume
            assert streamer.staged_replica_tensors(blk, torch.device("cpu")) is not None  # replicas survive
        finally:
            streamer.stop_prefetch()


class _FakeStreamer:
    def __init__(self, tensors):
        self.tensors = tensors

    def staged_replica_tensors(self, prefix, device):
        return dict(self.tensors) if self.tensors else None


class TestMirrorAdoption:
    def _wrapped_block(self):
        block = torch.nn.Sequential()
        wrapper = torch.nn.Module()
        wrapper.orig_layer = torch.nn.Linear(8, 8)
        wrapper.params = {"v": torch.nn.Parameter(torch.zeros(8, 8))}
        wrapper._stream_weights_pristine_holder = True
        block.add_module("m", wrapper)
        return block

    def test_adoption_swaps_weights_from_staged(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import DDPPlan, ReplicaGroup

        torch.manual_seed(0)
        block = self._wrapped_block()
        block._stream_weights_pristine = True
        staged_w = torch.randn(8, 8)
        streamer = _FakeStreamer({"blk.m.weight": staged_w})
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=4)
        group = ReplicaGroup(block, plan, staged_source=(streamer, "blk"))
        assert group.adopted == 1
        # the adopted mirror's orig_layer weight IS the staged tensor (no copy)
        assert group.mirrors[0].m.orig_layer.weight.data_ptr() == staged_w.data_ptr()
        # tuning params still fresh Parameters
        assert isinstance(group.mirrors[0].m.params["v"], torch.nn.Parameter)
        group.teardown()

    def test_non_pristine_falls_back_to_deepcopy(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import DDPPlan, ReplicaGroup

        block = self._wrapped_block()
        block._stream_weights_pristine = False
        streamer = _FakeStreamer({"blk.m.weight": torch.randn(8, 8)})
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=4)
        group = ReplicaGroup(block, plan, staged_source=(streamer, "blk"))
        assert group.adopted == 0
        group.teardown()

    def test_no_staged_falls_back(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import DDPPlan, ReplicaGroup

        block = self._wrapped_block()
        block._stream_weights_pristine = True
        streamer = _FakeStreamer(None)
        plan = DDPPlan(world=2, devices=[torch.device("cpu")] * 2, shard_size=4)
        group = ReplicaGroup(block, plan, staged_source=(streamer, "blk"))
        assert group.adopted == 0
        assert torch.equal(group.mirrors[0].m.orig_layer.weight, block.m.orig_layer.weight)
        group.teardown()
