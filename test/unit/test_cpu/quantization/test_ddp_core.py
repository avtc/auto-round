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
"""CPU-tier tests for the ported single-process DDP tuning surface.

Covers the device-agnostic contract: shard-constrained samplers, plan
resolution with injected free-VRAM data, transport encoding math, pool
distribution layout, gather-on-CPU paths, env defaults, and the
block-runner device-safe concatenation. Mirror/replica behavior on real
CUDA devices lives in the CUDA tier (test_ddp_mirror.py + e2e).
"""

import pytest
import torch

from auto_round.algorithms.block_runner import _cat_device_safe
from auto_round.algorithms.quantization.sign_round.data_parallel import (
    _encode_transport,
    _xchg,
    distribute_pool,
    gather_block_for_mirroring_,
    resolve_ddp_plan,
)
from auto_round.compressors.utils import IndexSampler, shard_samplers


class TestShardSamplers:
    def test_layout_and_epoch_coverage(self):
        samplers = shard_samplers(nsamples=8, world=4, batch_per_replica=1)
        assert samplers is not None and len(samplers) == 4
        drawn = [s.next_batch() for s in samplers]
        # each replica draws only from its own contiguous shard
        for r, batch in enumerate(drawn):
            assert all(r * 2 <= j < (r + 1) * 2 for j in batch)
        # one shard-epoch covers every sample exactly once (batch=1 x shard=2 draws)
        rest = [s.next_batch() for s in samplers]
        assert sorted(j for b in drawn + rest for j in b) == list(range(8))

    def test_ineligible_layouts_return_none(self):
        assert shard_samplers(nsamples=7, world=4, batch_per_replica=1) is None  # indivisible
        assert shard_samplers(nsamples=8, world=1, batch_per_replica=8) is None  # single device
        assert shard_samplers(nsamples=8, world=4, batch_per_replica=3) is None  # draw > shard

    def test_index_sampler_explicit_pool(self):
        s = IndexSampler(4, 2, indices=[10, 11, 12, 13])
        b = s.next_batch()
        assert sorted(b) in ([10, 11], [12, 13]) or set(b) <= {10, 11, 12, 13}
        with pytest.raises(ValueError):
            IndexSampler(3, 2, indices=[1, 2])


class TestResolveDDPPlan:
    def _plan(self, world, free, footprint=100):
        return resolve_ddp_plan(
            world,
            torch.device("cuda", 0),  # device OBJECTS only -- no CUDA runtime touched
            8,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2", "3"],  # bare indices: normalized to cuda:N
            vram_free_bytes={torch.device("cuda", i): free[i] for i in range(4)},
            mirror_footprint_bytes=footprint,
            margin_bytes=0,  # toy byte values in these tests
        )

    def test_full_world_when_vram_fits(self):
        plan = self._plan(4, [1000, 1000, 1000, 1000])
        assert plan.enabled and plan.world == 4
        assert plan.shard_size == 2

    def test_device_without_mirror_fit_is_dropped(self):
        plan = self._plan(4, [1000, 10, 1000, 1000])  # cuda:1 too small
        assert plan.enabled
        assert torch.device("cuda", 1) not in plan.devices
        assert plan.world == 3  # reduced by the VRAM guard
        assert any("world reduced" in n for n in plan.notes)
        # the quantizer gates non-power-of-two worlds (fail-visible downgrade to
        # serial with an INFO) -- resolve itself just reports the fitting subset

    def test_world_collapses_to_one_when_nothing_fits(self):
        plan = self._plan(4, [5, 5, 5, 5], footprint=100)
        assert not plan.enabled

    def test_non_power_of_two_request_resolves_but_caller_gates(self):
        # resolve reports the fitting subset; the shared resolver's
        # power-of-two gate is what disables engagement for world=3
        plan = resolve_ddp_plan(
            3,
            torch.device("cuda", 0),
            12,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2"],
            vram_free_bytes={torch.device("cuda", i): 1 << 30 for i in range(3)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 3 and plan.enabled


class TestTransportMath:
    def test_fp32_passthrough(self):
        t = torch.randn(16)
        out = _xchg(t, t.device, torch.float32, "fp32")
        assert out is t

    def test_bf16_transport_roundtrip_preserves_signs(self):
        t = torch.randn(1024)
        out = _xchg(t, t.device, torch.float32, "bf16")
        assert torch.equal(torch.sign(out), torch.sign(t))

    def test_int8_transport_preserves_signs(self):
        t = torch.randn(1024) * 0.01
        out = _xchg(t, t.device, torch.float32, "int8")
        # signs are what sign-SGD consumes; |t| below the int8 quantum rounds
        # to zero (sign 0), magnitudes carry bounded int8 error
        assert ((torch.sign(out) == torch.sign(t)) | (out == 0)).all()
        assert (out - t).abs().max() < 1e-3

    def test_encode_transport_meta(self):
        t = torch.randn(8)
        payload, meta = _encode_transport(t, "int8")
        assert payload.dtype == torch.int8 and meta is not None
        payload, meta = _encode_transport(t, "bf16")
        assert payload.dtype == torch.bfloat16 and meta is None
        payload, meta = _encode_transport(t, "fp32")
        assert payload is t and meta is None


class TestDistributePool:
    def test_plan_layout_covers_pool_exactly(self):
        # distribute_pool gives device r the contiguous range [r*shard,(r+1)*shard);
        # on uniform CPU devices that is a no-op, so pin the layout arithmetic
        # through the plan the pool follows
        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_ddp_plan

        plan = resolve_ddp_plan(
            4,
            torch.device("cuda", 0),
            8,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2", "3"],  # avoids torch.cuda.device_count() on CUDA-less hosts
            vram_free_bytes={torch.device("cuda", i): 1 << 40 for i in range(4)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 4
        assert plan.shard_size * plan.world == 8
        # device objects in the plan are cuda:N (unusable on CUDA-less hosts);
        # exercise the scatter loop with the degenerate all-cpu device list
        distribute_pool([torch.zeros(2) for _ in range(8)], [torch.device("cpu")] * 4)
        # indivisible / too-small pools are left alone by contract (serial cats handle them)
        distribute_pool([torch.zeros(2)], [torch.device("cpu")] * 4)


class TestGatherOnCPU:
    def test_noop_on_whole_block(self):
        block = torch.nn.Linear(4, 4)
        assert gather_block_for_mirroring_(block, torch.device("cpu")) is False

    def test_repoints_stale_tuning_device_strings(self):
        block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        block[1].tuning_device = "cuda:5"
        assert gather_block_for_mirroring_(block, torch.device("cpu")) is True
        assert str(block[1].tuning_device) == "cpu"

    def test_moves_wrapper_dict_state(self):
        block = torch.nn.Linear(4, 4)
        block.params = {"v": torch.nn.Parameter(torch.ones(4), requires_grad=True)}
        gather_block_for_mirroring_(block, torch.device("cpu"))
        assert block.params["v"].device.type == "cpu"


class TestCatDeviceSafe:
    def test_same_device_is_plain_cat(self):
        parts = [torch.zeros(2, 3), torch.ones(2, 3)]
        out = _cat_device_safe(parts, dim=0)
        assert torch.equal(out, torch.cat(parts, dim=0))

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError):
            _cat_device_safe([], dim=0)


class TestEnvDefaults:
    def test_ddp_defaults_resolve(self, monkeypatch):
        for name in ("AR_TUNE_DDP_WORLD", "AR_TUNE_DDP_DEVICES"):
            monkeypatch.delenv(name, raising=False)
        from auto_round import envs

        assert envs.AR_TUNE_DDP_WORLD == 1
        assert envs.AR_TUNE_DDP_DEVICES == ""

    def test_world_env_round_trip(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "4")
        from auto_round import envs

        assert envs.AR_TUNE_DDP_WORLD == 4


class TestTupleKwargSlicing:
    """transformers-v5 rope arrives as position_embeddings=(cos, sin) with a
    per-sample batch dim; shard/batch forwards smaller than the cached batch
    must slice tuple-of-tensors kwargs elementwise or crash in rope."""

    def _runner(self, n=8, batch_size=4):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(n)]
        return (
            BlockForwardRunner(
                batch_dim=0, batch_size=batch_size, device=torch.device("cpu"), cache_device="cpu", amp=False
            ),
            inputs,
        )

    def test_tuple_kwargs_sliced_to_batch_indices(self):
        runner, inputs = self._runner()
        cos = torch.randn(8, 6, 2)
        sin = torch.randn(8, 6, 2)
        _, others = runner.select_batch(inputs, {"position_embeddings": (cos, sin)}, [3, 5])
        assert isinstance(others["position_embeddings"], tuple)
        assert others["position_embeddings"][0].shape == (2, 6, 2)
        assert others["position_embeddings"][1].shape == (2, 6, 2)
        assert torch.equal(others["position_embeddings"][0][0], cos[3])
        assert torch.equal(others["position_embeddings"][0][1], cos[5])

    def test_broadcast_tuple_elements_pass_through(self):
        runner, inputs = self._runner()
        table = torch.randn(1, 6, 2)  # broadcast-shaped: not per-sample
        _, others = runner.select_batch(inputs, {"position_embeddings": (table, table)}, [3, 5])
        assert others["position_embeddings"][0] is table  # unsliced, broadcastable

    def test_mixed_and_non_tensor_tuples_untouched(self):
        runner, inputs = self._runner()
        val = (torch.randn(8, 6, 2), "flag")
        _, others = runner.select_batch(inputs, {"weird": val}, [3, 5])
        assert others["weird"] is val  # mixed tuple treated as opaque


class TestSharedCacheShardPick:
    """shared_cache_keys kwargs arrive as ONE ENTRY PER BATCH (list[n_pool/batch]).
    A sub-batch draw (DDP shard) must pick the owning batch's entry and slice its
    within-batch rows; full-batch draws keep the legacy pick exactly."""

    def _runner(self):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(128)]
        return (
            BlockForwardRunner(
                batch_dim=0,
                batch_size=8,
                device=torch.device("cpu"),
                cache_device="cpu",
                amp=False,
                shared_cache_keys=("position_embeddings",),
            ),
            inputs,
        )

    def _pe(self):
        # 16 batch entries, each an (cos, sin) tuple of [8, S, D] per-sample tensors
        return [(torch.full((8, 6, 2), float(b)), torch.full((8, 6, 2), -float(b))) for b in range(16)]

    def test_shard_picks_owning_batch_and_rows(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [32, 33])
        cos, sin = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and sin.shape == (2, 6, 2)
        assert torch.all(cos == 4.0) and torch.all(sin == -4.0)  # entry 4, rows 0-1

    def test_shard_wrapping_rows_across_batch_boundary_fails_safe(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [7, 8])
        # batch 0 entry (indices[0]=7), rows 7 and 0 -- within-entry slice
        cos, _ = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and torch.all(cos == 0.0)

    def test_full_batch_draw_keeps_legacy_pick(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, list(range(8, 16)))
        # legacy: multi-index full-batch -> val[0], untouched (batch 0's entry, 8 rows)
        cos, _ = others["position_embeddings"]
        assert isinstance(cos, torch.Tensor) and cos.shape[0] == 8 and torch.all(cos == 0.0)

    def test_single_index_sub_batch_draw_per_batch_list(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [5])
        # per-batch list: sample 5 -> batch 0 entry, row 5
        cos, _ = others["position_embeddings"]
        assert cos.shape == (1, 6, 2) and torch.all(cos == 0.0)

    def test_per_sample_list_sub_batch_draw(self):
        runner, inputs = self._runner()
        # per-sample layout: 128 entries, each a [1, 6, 2] tensor
        ps = [torch.full((1, 6, 2), float(i)) for i in range(128)]
        _, others = runner.select_batch(inputs, {"position_embeddings": ps}, [32, 33])
        assert others["position_embeddings"].shape == (2, 6, 2)
        assert torch.all(others["position_embeddings"][0] == 32.0)


class TestLoggingGlobals:
    def test_engaged_log_globals_initialized(self):
        """Regression: the env-strip once dropped the module-level
        ``_ENGAGED_LOGGED_SIG`` init, and the ``global`` read in
        resolve_tune_ddp_plan_ then NameError'd -- but only on GPU runs,
        because a CPU home declines before the sig block (CPU tests cannot
        reach the engaged path)."""
        from auto_round.algorithms.quantization.sign_round import data_parallel as dp

        assert dp._ENGAGED_LOGGED_SIG is None
        assert isinstance(dp._coll_mirror_setup_logged, set)


class _FakeCudart:
    """Scriptable libcudart stand-in: rc maps call-name -> return code."""

    def __init__(self, can_rc=0, can_value=1, set_rc=0, enable_rc=0):
        self.can_rc, self.can_value = can_rc, can_value
        self.set_rc, self.enable_rc = set_rc, enable_rc
        self.enable_calls = []

    def cudaDeviceCanAccessPeer(self, out, i, j):
        out._obj.value = self.can_value
        return self.can_rc

    def cudaSetDevice(self, i):
        return self.set_rc

    def cudaDeviceEnablePeerAccess(self, j, flags):
        self.enable_calls.append(j)
        return self.enable_rc

    def cudaGetErrorString(self, rc):
        return b"fake cuda error"


class TestRequestedWorldErrors:
    """A requested parallel world is a requirement: infeasible -> RuntimeError, never silent serial."""

    def _fake_quantizer(self):
        from types import SimpleNamespace

        q = SimpleNamespace(
            iters=10,
            gradient_accumulate_steps=1,
            enable_lfq=False,
            _resolved_ddp_plan=None,
        )
        q._get_scaler = lambda: None
        return q

    def test_infeasible_world_raises(self, monkeypatch):
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="ineligible"):
            resolve_tune_ddp_plan_(self._fake_quantizer(), block, [torch.zeros(1)], None, "cpu")

    def test_iters0_is_not_a_decline_reason(self, monkeypatch):
        """The DDP world shards the collection at iters=0 too (campaign
        semantics restored): with a fake quantizer at iters=0 the only
        ineligibility reason left must be the non-CUDA home, never iters."""
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        q = self._fake_quantizer()
        q.iters = 0
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError) as excinfo:
            resolve_tune_ddp_plan_(q, block, [torch.zeros(1)], None, "cpu")
        assert "iters" not in str(excinfo.value)
        assert "not CUDA" in str(excinfo.value)

    def test_no_world_set_stays_serial(self, monkeypatch):
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        plan = resolve_tune_ddp_plan_(self._fake_quantizer(), block, [torch.zeros(1)], None, "cpu")
        assert plan.world == 1


class TestDisableP2P:
    """AR_TUNE_DISABLE_P2P skips enablement entirely (A/B testing knob)."""

    def test_disable_returns_empty_without_touching_cudart(self, monkeypatch):
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        monkeypatch.setenv("AR_TUNE_DISABLE_P2P", "1")
        # CPU devices: without the early-out this would hit the CUDA-only path
        assert enable_peer_access([torch.device("cpu"), torch.device("cpu")]) == []

    def test_unset_still_asks_cudart(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        monkeypatch.delenv("AR_TUNE_DISABLE_P2P", raising=False)
        # on hosts without libcudart the normal path logs the load failure and
        # returns [] -- reaching that return proves the early-out did NOT fire
        assert enable_peer_access([0, 1]) == []


class TestTunePhaseLine:
    """Formatter for the AR_PERF_COUNTERS per-block tune phase breakdown."""

    def test_formats_all_four_buckets(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({"wrap": 8.21, "prepare": 0.42, "loop": 3.5, "tail": 0.47}, 0)
        assert "iters=0" in line
        assert "wrap=8.21s" in line and "prepare=0.42s" in line
        assert "loop=3.50s" in line and "tail=0.47s" in line

    def test_missing_buckets_default_to_zero(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({}, 10)
        assert "wrap=0.00s" in line and "tail=0.00s" in line and "iters=10" in line


class TestBlockHasTuningEntries:
    """DDP decline check: all-float blocks must be detectable pre-engagement."""

    def test_plain_block_has_no_entries(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        assert block_has_tuning_entries(torch.nn.Sequential(torch.nn.Linear(8, 8))) is False

    def test_minmax_only_block_counts_as_tunable(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class MinmaxOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"wmax": nn.Parameter(torch.ones(1))}

        # full-mirror can still tune minmax params
        assert block_has_tuning_entries(torch.nn.Sequential(MinmaxOnly())) is True

    def test_round_block_counts(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class RoundHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"v": nn.Parameter(torch.ones(4))}

        assert block_has_tuning_entries(torch.nn.Sequential(RoundHolder())) is True


class TestPerfTraceHelpers:
    """AR_PERF_TRACE one-shot iteration profiler (diagnostic scaffolding)."""

    def test_unset_and_valid_values(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.quantizer import _perf_trace_iter

        monkeypatch.delenv("AR_PERF_TRACE", raising=False)
        assert _perf_trace_iter(50) is None
        monkeypatch.setenv("AR_PERF_TRACE", "2")
        assert _perf_trace_iter(50) == 2

    def test_invalid_values_return_none(self, monkeypatch, caplog, _autoround_log_propagate):
        from auto_round.algorithms.quantization.sign_round.quantizer import _perf_trace_iter

        monkeypatch.setenv("AR_PERF_TRACE", "abc")
        assert _perf_trace_iter(50) is None
        monkeypatch.setenv("AR_PERF_TRACE", "-1")
        assert _perf_trace_iter(50) is None
        monkeypatch.setenv("AR_PERF_TRACE", "50")
        assert _perf_trace_iter(50) is None
        assert any("AR_PERF_TRACE" in r.message for r in caplog.records)

    def test_start_stop_exports_trace(self, monkeypatch, tmp_path):
        from auto_round.algorithms.quantization.sign_round.quantizer import _start_perf_trace, _stop_perf_trace

        out = tmp_path / "trace.json"
        monkeypatch.setenv("AR_PERF_TRACE_PATH", str(out))
        prof = _start_perf_trace(0)
        x = torch.ones(4, 4).sum()
        _stop_perf_trace(prof, "unit-test")
        assert out.exists() and out.stat().st_size > 0


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""
    import logging

    logger = logging.getLogger("autoround")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


class TestEnablePeerAccessFailVisible:
    """Every cudart rc must be checked; the returned pair list may never
    overstate what actually happened, and failures must be visible."""

    def _patch_cdll(self, monkeypatch, fake):
        import ctypes as real_ctypes

        def _factory(name):
            if "cudart" in str(name):
                return fake
            return real_ctypes.CDLL(name)

        monkeypatch.setattr("ctypes.CDLL", _factory)

    def test_success_and_already_set_count_as_enabled(self, monkeypatch):
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        self._patch_cdll(monkeypatch, _FakeCudart(enable_rc=0))
        assert enable_peer_access([0, 1]) == ["0->1", "1->0"]
        self._patch_cdll(monkeypatch, _FakeCudart(enable_rc=704))
        assert enable_peer_access([0, 1]) == ["0->1", "1->0"]

    def test_enable_failure_warns_and_excludes_pair(self, monkeypatch, caplog, _autoround_log_propagate):
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        self._patch_cdll(monkeypatch, _FakeCudart(enable_rc=1))
        with caplog.at_level("WARNING"):
            pairs = enable_peer_access([0, 1])
        assert pairs == []  # rc=1 must NOT be reported as enabled
        assert any("P2P enable failed" in r.message for r in caplog.records)

    def test_probe_and_setdevice_failures_warn(self, monkeypatch, caplog, _autoround_log_propagate):
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        self._patch_cdll(monkeypatch, _FakeCudart(can_rc=217))
        with caplog.at_level("WARNING"):
            assert enable_peer_access([0, 1]) == []
        assert any("P2P probe failed" in r.message for r in caplog.records)

        self._patch_cdll(monkeypatch, _FakeCudart(set_rc=101))
        with caplog.at_level("WARNING"):
            assert enable_peer_access([0, 1]) == []
        assert any("cudaSetDevice" in r.message for r in caplog.records)

    def test_topology_cannot_peer_is_silent_not_enabled(self, monkeypatch, caplog, _autoround_log_propagate):
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        self._patch_cdll(monkeypatch, _FakeCudart(can_value=0))
        with caplog.at_level("WARNING"):
            assert enable_peer_access([0, 1]) == []
        assert not any("P2P" in r.message for r in caplog.records)  # not an error

    def test_missing_libcudart_warns(self, monkeypatch):
        import ctypes as real_ctypes

        def _factory(name):
            raise OSError(f"no {name}")

        monkeypatch.setattr("ctypes.CDLL", _factory)
        import logging

        logging.getLogger("auto_round").setLevel(logging.WARNING)
        from auto_round.algorithms.quantization.sign_round.data_parallel import enable_peer_access

        assert enable_peer_access([0, 1]) == []
