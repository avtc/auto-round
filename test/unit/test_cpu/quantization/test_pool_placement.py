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

"""Unit tests for --calibration_data_device calibration-data placement (policy + routing).

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

    def _resolve(self, mode="auto", free=None, pool=1 * GB, need=2 * GB, primary="cuda:0", candidates=None):
        free = free if free is not None else {"cuda:0": 20 * self.GB}
        candidates = candidates if candidates is not None else ["cuda:0", "cuda:1", "cuda:2"]
        return pp.resolve_pool_placement(pool, 128, primary, need, candidates, _fake_probe(free), mode=mode)

    def test_off_env_returns_none(self):
        self.assertIsNone(self._resolve(mode="off"))

    def test_cpu_mode_parks_on_host(self):
        plan = self._resolve(mode="cpu")
        self.assertEqual(plan.devices, ["cpu"])
        self.assertEqual(plan.counts(), {"cpu": 128})

    def test_need_estimator_first_principles(self):
        """Need = flat working allowance + reserve; iters>0 adds 14B/param of
        tuning state for candidate-homed parameters only. Pool bytes are NOT
        part of the need (the gates count them via their own terms)."""
        reserve = int(0.5 * 2**30)
        working = int(3 * 2**30)
        pool = torch.zeros(64, dtype=torch.float32)  # 256 B -- size must not matter
        big_pool = torch.zeros(1 << 20, dtype=torch.float32)
        # no block: working + reserve only; pool size must not change the need
        need0 = pp.placement_need_bytes(None, pool, 8)
        self.assertEqual(need0, working + reserve)
        self.assertEqual(pp.placement_need_bytes(None, big_pool, 8), need0)
        # iters=0 with a block: same (no tuning state at zero-shot)
        need0b = pp.placement_need_bytes(object(), pool, 8, iters=0, primary="cuda:0")
        self.assertEqual(need0b, need0)
        # iters>0: +14 B per candidate-homed parameter; peers' params ignored
        m = torch.nn.Linear(4, 4)  # 16 weights + 4 bias = 20 params, all on CPU
        dev = str(next(m.parameters()).device)
        need1 = pp.placement_need_bytes(m, pool, 8, iters=20, primary=dev)
        self.assertEqual(need1, need0 + 20 * 14)
        # primary mismatch: parameters are not homed there -> no state charge
        need2 = pp.placement_need_bytes(m, pool, 8, iters=20, primary="cuda:7")
        self.assertEqual(need2, need0)

    def test_consumer_retargets_single_away_from_cache_primary(self):
        """iters>0: the tune consumer (not the cache primary) is the
        single-device target; the primary demotes to a plain peer."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 20 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
            consumer="cuda:1",
        )
        self.assertEqual(plan.devices, ["cuda:1"])  # 12 - 3.5 >= 8 fits on the consumer

    def test_consumer_needs_headroom_like_the_primary_did(self):
        """The consumer is charged the working-set need: no room -> spread
        (with the consumer need-charged, the primary NOT charged)."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(10 * self.GB),  # consumer cannot host the pool beside its need
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe({"cuda:0": 20 * self.GB, "cuda:1": 12 * self.GB, "cuda:2": 11 * self.GB}),
            consumer="cuda:1",
        )
        counts = plan.counts()
        # need-floor: the consumer's capacity floors at ~zero, so it gets the
        # smallest share while the primary (now an uncharged peer) leads
        self.assertLess(counts.get("cuda:1", 0), counts.get("cuda:0", 0))
        self.assertLessEqual(counts.get("cuda:1", 0), counts.get("cuda:2", 0))

    def test_occupied_incoming_pool_blocks_fake_single_fit(self):
        """The single-device gate must count pool bytes already resident on the
        candidate -- without it the plan alternates single/spread across blocks
        (a single plan loads the device, the next resolve sees less free)."""
        base = {"cuda:0": 20 * self.GB, "cuda:1": 13 * self.GB, "cuda:2": 11 * self.GB}
        need = int(3.5 * self.GB)
        pool = 8 * self.GB
        free_fit = 13 * self.GB  # 13 - 3.5 >= 8: would look like a fit...
        with_occupied = pp.resolve_pool_placement(
            pool, 128, "cuda:0", need, ["cuda:0", "cuda:1", "cuda:2"], _fake_probe(base), consumer="cuda:1"
        )
        self.assertEqual(with_occupied.devices, ["cuda:1"])
        # ...until the incoming input pool (already on the consumer) is charged
        blocked = pp.resolve_pool_placement(
            pool,
            128,
            "cuda:0",
            need,
            ["cuda:0", "cuda:1", "cuda:2"],
            _fake_probe(base),
            consumer="cuda:1",
            occupied_bytes=free_fit - 0,  # charge the resident 13 GiB fully
        )
        # 13 - 3.5 - 13 < 8: single vetoed, consumer capacity floored
        counts = blocked.counts()
        self.assertLessEqual(counts.get("cuda:1", 0), counts.get("cuda:2", 0))
        self.assertLess(counts.get("cuda:1", 0), counts.get("cuda:0", 0))

    def test_degenerate_consumer_falls_back_to_primary(self):
        """consumer == primary or non-cuda consumer: primary-first as before."""
        plan = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1"],
            _fake_probe({"cuda:0": 12 * self.GB, "cuda:1": 12 * self.GB}),
            consumer="cuda:0",  # same as primary
        )
        self.assertEqual(plan.devices, ["cuda:0"])
        plan2 = pp.resolve_pool_placement(
            8 * self.GB,
            128,
            "cuda:0",
            int(3.5 * self.GB),
            ["cuda:0", "cuda:1"],
            _fake_probe({"cuda:0": 12 * self.GB, "cuda:1": 12 * self.GB}),
            consumer="cpu",  # non-cuda: ignored
        )
        self.assertEqual(plan2.devices, ["cuda:0"])

    def test_huge_moe_need_still_shards_onto_peers(self):
        """Regression: the need constrains the primary alone, never the fleet.

        A MoE block's need runs to hundreds of GiB under the x7 expert
        multiplier; charging it against total free vetoed every shard plan
        ('insufficient: total - need < pool'), parking the pools on the
        busiest device and OOM-ing the next block.
        """
        plan = self._resolve(
            pool=4 * self.GB,
            need=342 * self.GB,
            free={"cuda:0": 2 * self.GB, "cuda:1": 10 * self.GB, "cuda:2": 10 * self.GB},
        )
        self.assertIsNotNone(plan)
        counts = plan.counts()
        self.assertEqual(counts.get("cuda:0", 0), 0)  # need-floor leaves no chunks on the primary
        self.assertGreater(counts.get("cuda:1", 0), 0)
        self.assertGreater(counts.get("cuda:2", 0), 0)
        self.assertEqual(sum(counts.values()), 128)

    def test_zero_peer_capacity_returns_none(self):
        plan = self._resolve(
            pool=4 * self.GB,
            need=2 * self.GB,
            free={"cuda:0": 1 * self.GB, "cuda:1": 1 * self.GB},
        )
        self.assertIsNone(plan)

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


class TestCalibDataLine(unittest.TestCase):
    def test_bytes_by_device_counts_referenced_leaves(self):
        a = torch.zeros(10)  # 40B
        b = torch.zeros(6)  # 24B
        per_dev, total = pp._bytes_by_device({"h": [a, b], "m": a})
        self.assertEqual(per_dev, {"cpu": 104})  # 'm' references a again
        self.assertEqual(total, 104)

    def test_monitor_grammar(self):
        fp = [torch.zeros(1, 64, 64) for _ in range(4)]  # 64KiB
        q = [torch.zeros(1, 64, 64) for _ in range(4)]
        plan = pp.PoolPlacement(["cpu", "cpu"], [1, 1], 8)
        line = pp.calib_data_line([fp, q], None, plan, 64 * 64 * 4 * 8, 8, "cpu")
        self.assertIn("'input': 0.00GB", line)
        self.assertIn("'output': 0.00GB", line)
        self.assertIn("'aux': 0.00GB", line)
        self.assertIn("'per_device': {'cpu':", line)
        self.assertNotIn("cuda", line)

    def test_short_device_keys(self):
        self.assertEqual(pp._short_device_key("cuda:3"), "3")
        self.assertEqual(pp._short_device_key("cpu"), "cpu")

    def test_plan_devices_short_keys_and_fallback(self):
        plan = pp.PoolPlacement(["cuda:0", "cuda:1"], [1, 1], 2)
        line = pp.calib_data_line([], None, plan, 4 * 2**30, 2, "cuda:0")
        self.assertIn("'0':", line)
        self.assertIn("'1':", line)
        # no plan -> outputs land on the primary
        line2 = pp.calib_data_line([], None, None, 2**30, 1, "cuda:0")
        self.assertIn("'0': 1.00GB", line2)


class TestConsolidate(unittest.TestCase):
    class _OnDevice(torch.Tensor):
        # real tensor (bytes/isinstance work) with an overridden device view
        @property
        def device(self):
            return torch.device(self._fake_device)

    def _tensor_on(self, dev):
        t = torch.zeros(2).as_subclass(self._OnDevice)
        t._fake_device = dev
        return t

    def _ctx(self, free_gib, need_gib):
        return (
            mock.patch.object(pp, "placement_need_bytes", return_value=int(need_gib * 2**30)),
            mock.patch(
                "auto_round.utils.device.probe_usable_bytes",
                return_value=int(free_gib * 2**30),
            ),
        )

    def test_local_when_already_on_target(self):
        pool = [self._tensor_on("cuda:0") for _ in range(3)]
        with self._ctx(10, 0)[0], self._ctx(10, 0)[1]:
            with mock.patch.object(pp, "_move_pool_to") as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "local")
        mv.assert_not_called()

    def test_consolidated_moves_in_place(self):
        pool = [self._tensor_on("cuda:1"), self._tensor_on("cuda:2")]
        with self._ctx(10, 0)[0], self._ctx(10, 0)[1]:
            with mock.patch.object(pp, "_move_pool_to") as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "consolidated")
        mv.assert_any_call(pool, "cuda:0")

    def test_spread_when_not_fitting(self):
        pool = [self._tensor_on("cuda:1")]
        with self._ctx(10, 9)[0], self._ctx(10, 9)[1]:
            with mock.patch.object(pp, "_tensor_bytes", return_value=5 * 2**30), mock.patch.object(
                pp, "_move_pool_to"
            ) as mv:
                self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "spread")
        mv.assert_not_called()

    def test_consolidation_reachable_under_first_principles_need(self):
        """Regression: the old 2x-pool need made attach-time consolidation
        mathematically unreachable (need ~= 2x pool + reserve always exceeded
        any free next to the pools it was gating). With pools counted by the
        gate's own terms, a genuine fit consolidates again."""
        GBi = 2**30
        pool = [torch.zeros(1) for _ in range(4)]
        # 4 GiB of pool tensors, 8 GiB free on the target cuda:0,
        # need 3.5 GiB (working + reserve)
        with mock.patch.object(pp, "_tensor_bytes", return_value=4 * GBi), mock.patch.object(
            pp, "_move_pool_to"
        ) as mv, mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=8 * GBi), mock.patch.object(
            pp, "placement_need_bytes", return_value=int(3.5 * GBi)
        ):
            self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "consolidated")
        mv.assert_called()
        # genuinely tight: free - need < pool -> spread, never moved
        with mock.patch.object(pp, "_tensor_bytes", return_value=4 * GBi), mock.patch.object(
            pp, "_move_pool_to"
        ) as mv2, mock.patch(
            "auto_round.utils.device.probe_usable_bytes", return_value=int(7 * GBi)
        ), mock.patch.object(
            pp, "placement_need_bytes", return_value=int(3.5 * GBi)
        ):
            self.assertEqual(pp.consolidate_pool_onto([pool], "cuda:0", object(), 8), "spread")
        mv2.assert_not_called()

    def test_non_cuda_target_spread(self):
        self.assertEqual(pp.consolidate_pool_onto([[torch.zeros(1)]], "cpu", object(), 8), "spread")

    def test_empty_is_local(self):
        self.assertEqual(pp.consolidate_pool_onto([None, []], "cuda:0", object(), 8), "local")


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

    def test_mixed_device_batch_gathers_before_cat(self):
        # regression: _select_batch cat'd per-sample tensors on their (mixed)
        # park devices before the compute-device move -> torch.cat crash.
        # Duck-typed tensors (a CPU box has no second real device; meta cannot
        # be copied out of).
        r = self._runner()

        class _FakeT:
            def __init__(self, dev):
                self.device = torch.device(dev)
                self.moved_to = None

            def to(self, target):
                self.moved_to = target
                return torch.zeros(2, 1)

        mixed = [_FakeT("cpu"), _FakeT("meta"), _FakeT("cpu")]
        out = r._gather_same_device(mixed, torch.device("cpu"))
        self.assertTrue(all(t.moved_to == torch.device("cpu") for t in mixed))
        self.assertTrue(all(o.device.type == "cpu" for o in out))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a real second device")
    def test_mixed_device_batch_select_integration(self):
        r = self._runner()
        r.device = "cuda:0"
        inputs = [torch.zeros(2, 1, device="cpu")] * 2 + [torch.zeros(2, 1, device="cuda:0")] * 2
        sel = r._select_batch(inputs, {"m": [torch.zeros(2, 1) for _ in range(4)]}, torch.tensor([0, 1, 2, 3]))
        self.assertEqual(sel[0].device.type, "cuda")
        self.assertEqual(sel[0].shape[0], 4)

    def test_uniform_device_batch_untouched(self):
        r = self._runner()
        a = torch.zeros(2, 1)
        inputs = [a, a.clone(), a.clone(), a.clone()]
        out = r._gather_same_device(inputs, torch.device("meta"))
        self.assertIs(out[0], a)  # same objects, no copies

    def test_explicit_cache_device_call_overrides_placement(self):
        r = self._runner()
        r.pool_placement = pp.PoolPlacement(["meta", "cpu"], [1, 1], 4)
        inputs = [torch.zeros(1, 1) for _ in range(4)]
        outs = r.forward(object(), inputs, {}, cache_device="cpu")
        self.assertTrue(all(o.device.type == "cpu" for o in outs))


if __name__ == "__main__":
    unittest.main()
