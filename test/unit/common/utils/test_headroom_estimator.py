# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schema-aware staging search headroom (replaces the fixed 4 GiB constant).

The streamer reserves headroom beyond a staged block's weights for the tuning
pass that runs beside it; the estimator derives the requirement from the run's
tuning profile instead of the historical one-size constant.
"""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

GIB = 1024**3

BASE = 2**28 * 4 + 3 * 2**25 * 4 + int(0.5 * GIB)  # qdq chunk + search temps + system allowance
FALLBACK = 4 * GIB


class TestEstimator:
    def test_iters0_dense(self):
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        assert estimate_search_headroom_bytes(iters=0, block_bytes=512 * GIB) == BASE

    def test_iters0_moe_adds_routing_buffer(self):
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        routing = 16384 * 8 * 4096 * 4  # tokens x top_k x hidden x fp32
        assert estimate_search_headroom_bytes(iters=0, moe_routing_bytes=routing) == BASE + routing

    def test_iters0_moe_anchor_matches_empirical_constant(self):
        """The profile the 4 GiB constant was calibrated on (MoE, chunked
        forward of 8x2048 tokens, top_k 8, hidden 4096) must land within ~10%
        of it, so switching to the estimator does not loosen the threshold
        the empirical runs were tuned against."""
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        routing = 16384 * 8 * 4096 * 4
        est = estimate_search_headroom_bytes(iters=0, moe_routing_bytes=routing)
        assert 0.95 * FALLBACK <= est <= 1.10 * FALLBACK

    def test_iters_gt0_scales_with_block(self):
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        routing = GIB
        assert estimate_search_headroom_bytes(iters=200, block_bytes=2 * GIB, moe_routing_bytes=routing) == (
            BASE + routing + 3 * 2 * GIB
        )

    def test_unknown_inputs_fall_back(self):
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        assert estimate_search_headroom_bytes() == FALLBACK
        assert estimate_search_headroom_bytes(iters=None, block_bytes=GIB) == FALLBACK
        # iters>0 without block bytes: the dominant term is unknown
        assert estimate_search_headroom_bytes(iters=10, block_bytes=None) == FALLBACK

    def test_ddp_world_shards_only_data_terms(self):
        """Data-parallel replicas split the calibration rows (routing buffer
        shrinks per device) but replicate the tuning params (never shrink)."""
        from auto_round.utils.checkpoint_streamer import estimate_search_headroom_bytes

        routing = 2 * GIB
        block = GIB
        full = estimate_search_headroom_bytes(iters=20, block_bytes=block, moe_routing_bytes=routing)
        half = estimate_search_headroom_bytes(iters=20, block_bytes=block, moe_routing_bytes=routing, ddp_world_size=2)
        assert half == full - routing // 2
        # inert by default: world 1 is exactly the serial estimate
        assert (
            estimate_search_headroom_bytes(iters=20, block_bytes=block, moe_routing_bytes=routing, ddp_world_size=1)
            == full
        )


class TestStreamerHeadroom:
    def _streamer(self, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        save_file({"blk.a.weight": torch.ones(2, 2)}, os.path.join(tmp_path, "model.safetensors"))
        return CheckpointStreamer(str(tmp_path))

    def test_profile_or_default(self, tmp_path):
        streamer = self._streamer(tmp_path)
        # no profile supplied: historical class default
        streamer._tuning_iters = None
        streamer._moe_routing_bytes = None
        assert streamer._headroom_for(2 * GIB) == FALLBACK
        # zero-shot dense profile
        streamer._tuning_iters = 0
        streamer._moe_routing_bytes = None
        assert streamer._headroom_for(2 * GIB) == BASE
        # tuning profile scales with the staged block
        streamer._tuning_iters = 50
        assert streamer._headroom_for(2 * GIB) == BASE + 3 * 2 * GIB
        assert streamer._headroom_for(4 * GIB) == BASE + 3 * 4 * GIB

    def test_start_prefetch_persists_profile(self, tmp_path):
        streamer = self._streamer(tmp_path)
        streamer.start_prefetch(["blk.a"], depth=1, stage_devices=None, tuning_iters=0, moe_routing_bytes=1024)
        streamer.stop_prefetch()
        assert streamer._tuning_iters == 0
        assert streamer._moe_routing_bytes == 1024

    def test_orchestrator_passes_profile_to_start_prefetch(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        i = src.index("start_prefetch(")
        window = src[i : i + 400]
        assert "tuning_iters=" in window and "moe_routing_bytes=" in window
        assert "self._tuning_headroom_profile()" in src


class TestTuningHeadroomProfile:
    """Profile derivation from quantizer configs + model config."""

    def _profile(self, quantizers, cfg, batch=8, seqlen=2048):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        stub = SimpleNamespace(
            alg_composer=SimpleNamespace(block_quantizer=quantizers),
            model_context=SimpleNamespace(config=cfg),
            calibration_context=SimpleNamespace(batch_size=batch, seqlen=seqlen),
        )
        return CompressionOrchestrator._tuning_headroom_profile(stub)

    def test_dense_zero_shot(self):
        assert self._profile(SimpleNamespace(iters=0), SimpleNamespace(hidden_size=4096)) == (0, None)

    def test_iters_max_across_configs(self):
        quants = [SimpleNamespace(iters=0), SimpleNamespace(iters=200)]
        cfg = SimpleNamespace(hidden_size=4096)
        assert self._profile(quants, cfg) == (200, None)

    def test_config_without_iters_field_counts_as_zero_shot(self):
        cfg = SimpleNamespace(hidden_size=4096)
        assert self._profile(SimpleNamespace(), cfg) == (0, None)

    def test_moe_routing_bytes(self):
        cfg = SimpleNamespace(hidden_size=4096, text_config=None, model_type="qwen3_moe", num_experts_per_tok=8)
        iters, routing = self._profile(SimpleNamespace(iters=0), cfg)
        assert iters == 0
        assert routing == 8 * 2048 * 8 * 4096 * 4

    def test_vl_moe_reads_text_config(self):
        text_cfg = SimpleNamespace(hidden_size=4096, num_experts_per_tok=8)
        cfg = SimpleNamespace(model_type="ernie4_5_vl_moe", text_config=text_cfg)
        iters, routing = self._profile(SimpleNamespace(iters=0), cfg)
        assert iters == 0
        assert routing == 8 * 2048 * 8 * 4096 * 4

    def test_dbrx_spelling(self):
        cfg = SimpleNamespace(hidden_size=4096, text_config=None, model_type="dbrx", moe_top_k=4)
        iters, routing = self._profile(SimpleNamespace(iters=0), cfg)
        assert routing == 8 * 2048 * 4 * 4096 * 4

    def test_moe_unknown_shape_keeps_default(self):
        # MoE per config string but no derivable top_k/hidden/batch
        assert self._profile(SimpleNamespace(iters=0), SimpleNamespace(model_type="some_moe"), batch=None) == (
            None,
            None,
        )


class TestBoundedStagingWait:
    """The staging wait must be visible and bounded: log once (including the
    previously-silent no-rescue case), force a second host-RAM slot after the
    force timeout, raise after the raise timeout."""

    def _streamer(self, tmp_path):
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        save_file({"blk.a.weight": torch.ones(2, 2)}, os.path.join(tmp_path, "model.safetensors"))
        return CheckpointStreamer(str(tmp_path))

    def _prime(self, streamer, monkeypatch, devices, cpu_staged):
        import auto_round.utils.checkpoint_streamer as cs

        streamer._prefetch_stage_devices = devices
        streamer._prefetch_stop = False
        streamer._prefetch_cpu_staged = cpu_staged
        monkeypatch.setattr(cs, "pick_stage_device", lambda *a, **k: None)

    def test_regular_rescue_unchanged(self, tmp_path, monkeypatch):
        s = self._streamer(tmp_path)
        self._prime(s, monkeypatch, [torch.device("cuda", 0)], cpu_staged=0)
        s._staging_force_slot_after_s = 1e9
        s._staging_raise_after_s = 1e9
        assert s._wait_for_stage_device(0, "blk") == torch.device("cpu")
        assert s._prefetch_cpu_staged == 1

    def test_forced_second_slot_after_timeout(self, tmp_path, monkeypatch):
        s = self._streamer(tmp_path)
        self._prime(s, monkeypatch, [torch.device("cuda", 0)], cpu_staged=1)
        s._staging_force_slot_after_s = 0.0  # deadline already passed
        s._staging_raise_after_s = 1e9
        assert s._wait_for_stage_device(0, "blk") == torch.device("cpu")
        assert s._prefetch_cpu_staged == 2  # bounded: exactly one forced slot

    def test_raises_when_both_slots_taken(self, tmp_path, monkeypatch):
        s = self._streamer(tmp_path)
        self._prime(s, monkeypatch, [torch.device("cuda", 0)], cpu_staged=2)
        s._staging_force_slot_after_s = 0.0
        s._staging_raise_after_s = 0.0
        with pytest.raises(RuntimeError, match="staging wait"):
            s._wait_for_stage_device(0, "blk")

    def test_wait_logs_in_rescue_disallowed_case(self, tmp_path, monkeypatch):
        """No CUDA staging device: previously this loop logged NOTHING. The
        project logger does not propagate, so record calls on a stub."""
        import auto_round.utils.checkpoint_streamer as cs

        seen = []

        class _Log:
            def info(self, *a, **k):
                seen.append(str(a))

            def warning(self, *a, **k):
                seen.append(str(a))

        s = self._streamer(tmp_path)
        self._prime(s, monkeypatch, [], cpu_staged=0)
        s._staging_force_slot_after_s = 1e9
        s._staging_raise_after_s = 0.0
        monkeypatch.setattr(cs, "logger", _Log())
        with pytest.raises(RuntimeError, match="no staging devices"):
            s._wait_for_stage_device(0, "blk")
        assert any("no host-RAM rescue" in msg for msg in seen)


class TestMtpZeroShotFallback:
    """AR_MTP_ZERO_SHOT=1 lets a tuning run quantize pinned checkpoint-only
    groups with the closed-form search instead of leaving them verbatim."""

    def _run(self, iters, monkeypatch, env="0"):
        import torch.nn as nn

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        monkeypatch.setenv("AR_MTP_ZERO_SHOT", env)
        model = nn.Module()
        outer = nn.Module()
        layers = nn.Module()
        blk = nn.Module()
        blk.a = nn.Linear(8, 8, bias=False)
        layers.add_module("0", blk)
        outer.add_module("layers", layers)
        model.add_module("model", outer)
        tensors = ["model.layers.0.a.weight", "mtp.fc.weight", "mtp.norm.weight"]

        class Streamer:
            tensor_names = tensors

            def names_under(self, prefix):
                return [n for n in tensors if n.startswith(prefix + ".")]

            def tensor_meta(self, name):
                return {"mtp.fc.weight": ((16, 32), "BF16"), "mtp.norm.weight": ((32,), "BF16")}.get(name)

            def resolve_checkpoint_name(self, name):
                return name if name in tensors else None

        from types import MethodType

        stub = SimpleNamespace(
            alg_composer=SimpleNamespace(block_quantizer=[SimpleNamespace(iters=iters)]),
            model=model,
            layer_config={"mtp.fc": {"bits": 8, "data_type": "int"}},
            regex_config={},
        )
        stub._checkpoint_only_groups_ = MethodType(CompressionOrchestrator._checkpoint_only_groups_, stub)
        stub._pin_entry_for = MethodType(CompressionOrchestrator._pin_entry_for, stub)
        stub._analyze_checkpoint_only_group_ = MethodType(CompressionOrchestrator._analyze_checkpoint_only_group_, stub)
        claimed, tree_groups = CompressionOrchestrator._materialize_pinned_checkpoint_only_blocks_(stub, Streamer(), [])
        assert tree_groups == []
        return claimed, model

    def test_tuning_run_defaults_to_verbatim(self, monkeypatch):
        claimed, model = self._run(10, monkeypatch)
        assert claimed == set()
        assert not hasattr(model, "mtp")

    def test_env_forces_zero_shot_materialization(self, monkeypatch):
        claimed, model = self._run(10, monkeypatch, env="1")
        assert "mtp.fc.weight" in claimed
        lin = model.mtp.fc
        assert lin.bits == 8 and lin.global_name == "mtp.fc"
