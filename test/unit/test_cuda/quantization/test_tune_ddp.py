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
"""CUDA tests for single-process data parallelism (AR_TUNE_DDP_WORLD).

Run on a multi-GPU machine: cross-device halving-doubling all-reduce and
ReplicaGroup gradient sync must match the arithmetic mean exactly (fp32).
"""

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.data_parallel import (
    ReplicaGroup,
    DDPPlan,
    halving_doubling_allreduce,
)

needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 CUDA GPUs")


@needs_two_gpus
@pytest.mark.parametrize("world", [2])
def test_cross_device_allreduce_matches_mean(world):
    torch.manual_seed(0)
    devs = [torch.device("cuda", i) for i in range(world)]
    bufs = [torch.randn(1000, device=d) for d in devs]
    expected = torch.stack([b.cpu() for b in bufs]).sum(0) / world
    halving_doubling_allreduce(bufs, scale=1.0 / world)
    for b in bufs:
        assert torch.allclose(b.cpu(), expected, rtol=1e-5, atol=1e-6)


@needs_two_gpus
def test_cross_device_allreduce_ranks_identical():
    torch.manual_seed(1)
    bufs = [torch.randn(512, device=torch.device("cuda", i)) for i in range(2)]
    halving_doubling_allreduce(bufs, scale=0.5)
    assert torch.equal(bufs[0].cpu(), bufs[1].cpu())


@needs_two_gpus
def test_replica_group_sync_grads_two_devices():
    torch.manual_seed(2)
    block = torch.nn.Sequential()
    wrapper = torch.nn.Module()
    wrapper.orig_layer = torch.nn.Linear(16, 16)
    wrapper.params = {"v": torch.nn.Parameter(torch.zeros(16, 16))}
    block.add_module("m", wrapper)
    block.to("cuda:0")

    plan = DDPPlan(world=2, devices=[torch.device("cuda:0"), torch.device("cuda:1")], shard_size=4)
    group = ReplicaGroup(block, plan)
    assert group.mirrors[0].m.params["v"].device == torch.device("cuda:1")

    home_p = block.m.params["v"]
    mirror_p = group.mirrors[0].m.params["v"]
    home_p.grad = torch.full_like(home_p, 3.0)
    mirror_p.grad = torch.full_like(mirror_p, 5.0)
    group.sync_grads([[home_p], [mirror_p]])
    assert torch.allclose(home_p.grad, torch.full_like(home_p, 4.0))
    assert torch.allclose(mirror_p.grad, torch.full_like(mirror_p, 4.0))
    group.teardown()
    torch.cuda.synchronize()


@needs_two_gpus
def test_threaded_replica_execution():
    """Two replicas compute on their own devices concurrently via threads."""
    tensors = [torch.randn(64, 64, device=torch.device("cuda", i)) for i in range(2)]
    outs = [None, None]

    def work(r):
        outs[r] = tensors[r] @ tensors[r]

    block = torch.nn.Linear(2, 2).to("cuda:0")  # arbitrary home module for the group API
    plan = DDPPlan(world=2, devices=[torch.device("cuda:0"), torch.device("cuda:1")], shard_size=4)
    group = ReplicaGroup(block, plan)
    group.run_threaded([lambda r=r: work(r) for r in range(2)])
    for r in range(2):
        assert torch.allclose(outs[r], tensors[r] @ tensors[r], atol=1e-4)
    group.teardown()
