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
"""Persistent collection mirrors: one mirror build per block state.

sharded_nograd_forward accepts a caller-owned mirror_cache so sequential
collection passes over the same block state (fp/q hook passes; the post-tune
quantized pass reusing the tune's ReplicaGroup) build mirrors once instead
of per pass. Tests cover cache reuse, stat-reset between cached passes, and
the home->mirror tuning-param sync.
"""

import torch
from torch import nn

from auto_round.algorithms.quantization.sign_round import data_parallel as dp


class _Runner:
    """Minimal block_forward double: cats selected inputs, records indices."""

    def __init__(self):
        self.calls = []

    def __call__(self, block, inputs, input_others, indices=None, cache_device=None):
        self.calls.append(list(indices))
        sel = torch.cat([inputs[i] for i in indices], dim=0)
        return sel + 1.0

    def split_outputs(self, out):
        return list(torch.split(out, 1, dim=0))


class TestMirrorCacheReuse:
    def _block(self):
        blk = nn.Sequential()
        blk.add_module("lin", nn.Linear(4, 4))
        return blk

    def test_cache_builds_mirrors_once_across_passes(self, monkeypatch):
        blk = self._block()
        pool = [torch.randn(1, 4) for _ in range(4)]
        devices = [torch.device("cpu"), torch.device("cpu")]  # world=2
        runner = _Runner()
        builds = []
        # pretend the home lives elsewhere so both slots take the mirror path
        monkeypatch.setattr(dp, "_block_device", lambda b: torch.device("cuda:99"))

        import contextlib

        monkeypatch.setattr(torch.cuda, "device", lambda dev: contextlib.nullcontext())

        orig = dp.ReplicaGroup._replicate_mirror

        def _spy(block, dev, staged, prefix):
            builds.append(dev)
            return orig(block, dev, staged, prefix)  # raises on CPU -> deepcopy fallback

        monkeypatch.setattr(dp.ReplicaGroup, "_replicate_mirror", staticmethod(_spy))
        cache = {}
        out1 = dp.sharded_nograd_forward(runner, blk, pool, {}, torch.device("cpu"), devices, mirror_cache=cache)
        out2 = dp.sharded_nograd_forward(runner, blk, pool, {}, torch.device("cpu"), devices, mirror_cache=cache)
        assert len(out1) == len(out2) == 4
        torch.testing.assert_close(out1[0], out2[0])  # same block state -> identical
        # both slots key the same cpu device: ONE build serves both, and the
        # second pass reuses it (no rebuild)
        assert len(builds) == 1 and len(cache) == 1
        # without a cache every pass rebuilds every slot
        builds.clear()
        dp.sharded_nograd_forward(runner, blk, pool, {}, torch.device("cpu"), devices)
        dp.sharded_nograd_forward(runner, blk, pool, {}, torch.device("cpu"), devices)
        assert len(builds) == 4


class TestResetMirrorStats:
    def test_cached_mirrors_stats_zeroed(self):
        blk = nn.Linear(4, 4)
        blk.imatrix = torch.ones(4) * 5.0
        blk.act_max = torch.ones(4) * 3.0
        dp._reset_mirror_stats([blk])
        torch.testing.assert_close(blk.imatrix, torch.zeros(4))
        torch.testing.assert_close(blk.act_max, torch.zeros(4))


class TestSyncTuningParams:
    def _group(self):
        class _Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                layer = nn.Linear(4, 4, bias=False)
                layer.bits = 4
                self.orig_layer = layer
                self.v = nn.Parameter(torch.zeros(4))
                self.params = {"v": self.v, "extra": torch.zeros(4)}
                self.scale = torch.zeros(4)

        home = _Wrap()
        mirror = _Wrap()
        for p, q in zip(home.parameters(), mirror.parameters()):
            q.data.copy_(p.data)

        class _Plan:
            devices = [torch.device("cpu"), torch.device("cpu")]

        group = dp.ReplicaGroup.__new__(dp.ReplicaGroup)
        group.home = home
        group.mirrors = [mirror]
        group.replicas = [home, mirror]
        group.plan = _Plan()
        return home, mirror, group

    def test_home_final_state_reaches_mirror(self):
        home, mirror, group = self._group()
        # post-tune divergence: best-iter restore + refit write home-side
        with torch.no_grad():
            home.v.fill_(7.0)
            home.params["extra"].fill_(3.0)
            home.scale.fill_(9.0)
        group.sync_tuning_params()
        assert float(mirror.v.data[0]) == 7.0
        assert float(mirror.params["extra"][0]) == 3.0
        assert float(mirror.scale[0]) == 9.0
        # params dict aliasing kept coherent: same object as the parameter
        assert mirror.params["v"] is mirror.v
