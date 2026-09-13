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

"""Unit tests for AR_CALIBRATION_DATA_DEVICE calibration-data placement (policy + routing).

No CUDA required: free-memory probing and device candidates are injected, and the
BlockForwardRunner integration uses meta/cpu device targets to observe routing.
"""

import unittest
from unittest import mock

import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.utils import pool_placement as pp


def _fake_probe(free_map):
    return lambda d: free_map.get(d)


class TestSpreadPlan(unittest.TestCase):
    def test_proportional_interleaved(self):
        plan = pp._spread_plan(["a", "b"], [6, 4], 10)
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan.count("a"), 6)
        self.assertEqual(plan.count("b"), 4)
        # interleaved: no two consecutive chunks on the same device when both remain
        self.assertNotEqual(plan[0], plan[1])

    def test_zero_capacity_falls_back_to_devices(self):
        plan = pp._spread_plan(["a"], [0], 5)
        self.assertEqual(plan, ["a"])

    def test_single_device(self):
        plan = pp._spread_plan(["a"], [100], 5)
        self.assertEqual(plan, ["a"] * 5)


class TestPoolPlacement(unittest.TestCase):
    def test_counts_and_wrap(self):
        p = pp.PoolPlacement(["a", "b"], [5, 5], 4)
        self.assertEqual(p.counts(), {"a": 2, "b": 2})
        # index beyond plan wraps deterministically
        self.assertEqual(p.device_for_index(4), p.device_for_index(0))


class TestResolvePoolPlacement(unittest.TestCase):
    GB = 2**30

    def _resolve(self, mode="auto", free=None, pool=1 * GB, margin=2 * GB, primary="cuda:0", candidates=None):
        free = free if free is not None else {"cuda:0": 20 * self.GB}
        candidates = candidates if candidates is not None else ["cuda:0", "cuda:1", "cuda:2"]
        with mock.patch.object(pp.envs, "AR_CALIBRATION_DATA_DEVICE", mode):
            return pp.resolve_pool_placement(pool, 128, primary, margin, candidates, _fake_probe(free))

    def test_off_env_returns_none(self):
        self.assertIsNone(self._resolve(mode="off"))

    def test_cpu_mode_parks_on_host(self):
        plan = self._resolve(mode="cpu")
        self.assertEqual(plan.devices, ["cpu"])
        self.assertEqual(plan.counts(), {"cpu": 128})

    def test_need_estimator_floor_and_growth(self):
        floor = pp.placement_need_bytes(None, None, 8)
        self.assertGreaterEqual(floor, int(0.5 * 2**30))
        with mock.patch(
            "auto_round.utils.device.estimate_tuning_block_mem",
            return_value=({}, 3.0, 4.0, 5.0),
        ):
            need = pp.placement_need_bytes(object(), [], 8)
        # layer_activation(3.0) + additional(5.0) GiB + 0.5 GiB reserve
        self.assertEqual(need, int(8.5 * 2**30))

    def test_cpu_primary_returns_none(self):
        self.assertIsNone(self._resolve(primary="cpu"))

    def test_fits_primary_stays_primary(self):
        plan = self._resolve(pool=4 * self.GB, free={"cuda:0": 20 * self.GB})
        self.assertEqual(plan.devices, ["cuda:0"])
        self.assertEqual(plan.counts(), {"cuda:0": 128})

    def test_margin_blocks_primary_placement(self):
        # 8 GiB pool, 8 free, 2 GiB margin -> only 6 usable on primary: not enough,
        # peers absorb the rest while the margin still protects the primary
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertEqual(sum(plan.counts().values()), 128)
        self.assertLessEqual(plan.counts().get("cuda:0", 0), 64)

    def test_shards_proportionally(self):
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 30 * self.GB},
        )
        counts = plan.counts()
        self.assertGreater(counts["cuda:2"], counts["cuda:1"])
        self.assertEqual(sum(counts.values()), 128)

    def test_insufficient_capacity_returns_none(self):
        plan = self._resolve(
            pool=100 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertIsNone(plan)  # no silent CPU fallback: the real OOM fires

    def test_forced_csv_overrides_candidates(self):
        plan = self._resolve(
            mode="cuda:7,cuda:3",
            free={"cuda:7": 10 * self.GB, "cuda:3": 10 * self.GB},
        )
        self.assertEqual(set(plan.devices), {"cuda:7", "cuda:3"})

    def test_probe_none_skips_device(self):
        plan = self._resolve(
            pool=8 * self.GB,
            free={"cuda:0": 8 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": None},
        )
        self.assertNotIn("cuda:2", plan.counts())


class TestPoolBytes(unittest.TestCase):
    def test_tensor_bytes_nested(self):
        t = torch.zeros(2048, 4096, dtype=torch.float32)
        pool = {"hidden_states": [t, t], "mask": t}
        self.assertEqual(pp._tensor_bytes(pool), 3 * t.numel() * 4)

    def test_chunk_count(self):
        pool = {"hidden_states": [torch.zeros(1) for _ in range(7)]}
        self.assertEqual(pp._pool_chunk_count(pool), 7)


class TestRunnerRouting(unittest.TestCase):
    def _runner(self):
        r = BlockForwardRunner(batch_dim=0, batch_size=2, device="cpu", cache_device="cpu", enable_torch_compile=False)
        r.block_forward = lambda block, hidden, others, amp, amp_dtype, dev, x: hidden
        return r

    def test_placement_routes_per_sample_outputs(self):
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {})
        types = [o.device.type for o in outs]
        self.assertEqual(types, ["meta", "cpu", "meta", "cpu"])

    def test_no_placement_keeps_cache_device(self):
        r = self._runner()
        self.assertIsNone(getattr(r, "pool_placement", None))
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {})
        self.assertTrue(all(o.device.type == "cpu" for o in outs))

    def test_indices_mode_ignores_placement(self):
        # indices-mode concatenates outputs: mixed devices would break the cat,
        # so placement must not apply there
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        out = r.forward(object(), inputs, {}, indices=torch.tensor([0, 1]))
        self.assertEqual(out.device.type, "cpu")

    def test_explicit_cache_device_call_overrides_placement(self):
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {}, cache_device="cpu")
        self.assertTrue(all(o.device.type == "cpu" for o in outs))


if __name__ == "__main__":
    unittest.main()
