# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Block-path best-params snapshot placement: one ladder for every lane.

The outside-block lane decides snapshot placement via select_snapshot_device
(fits-beside-floor -> idle peer -> host). The block path cloned blindly to
cache_device - the predictor-tree tune died at iter 1 because the clone fit
but took the room the next backward needed. snapshot_best_params gives the
block path the same ladder: host under low_gpu_mem, beside-weights local
copies when the activation floor leaves room, idle peer, host fallback."""

import torch
import torch.nn as nn

from auto_round.compressors.utils import snapshot_best_params


class _Wrap(nn.Module):
    """Minimal wrapper-shaped module: params dict with a fp32 value."""

    def __init__(self, out_f, in_f, device="cpu"):
        super().__init__()
        self.orig_layer = nn.Linear(in_f, out_f, bias=False)
        self.params = {"value": torch.zeros(out_f, in_f, dtype=torch.float32, device=device)}
        self.device = torch.device(device)


class _FakeAR:
    """Device-manager backend stub: the ladder routes through get_ar_device."""

    def __init__(self, free_by_index, count):
        self._free_by_index, self._count = free_by_index, count

    def is_available(self):
        return True

    @property
    def device_count(self):
        return self._count

    def mem_get_info(self, index=0):
        return self._free_by_index.get(index, 0), 1 << 40


def _cuda_stub(monkeypatch, free_bytes_by_index, device_count=1, dev_type="cuda"):
    """Stub the accelerator introspection the ladder consults; CPU-only tests."""
    import auto_round.utils.device_manager as dm_mod

    fake = _FakeAR(free_bytes_by_index, device_count)
    monkeypatch.setattr(dm_mod, "get_ar_device", lambda t: fake)


class TestSnapshotBestParams:
    def test_cpu_cache_device_keeps_host_snapshot(self):
        blk = _Wrap(8, 4)
        out = snapshot_best_params(blk, "cpu")
        assert out["value"].device.type == "cpu"

    def test_no_floor_and_free_gpu_keeps_local_copy(self, monkeypatch):
        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 8 << 30})
        out = snapshot_best_params(blk, "cuda:0")
        # beside-weights semantics: the copy stays on the parameter device
        assert out["value"].device.type == blk.params["value"].device.type

    def test_floor_exceeded_without_peer_parks_on_host(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20}, device_count=1)  # ~no free VRAM
        warned = []
        monkeypatch.setattr(utils_mod.logger, "warning", lambda *a, **k: warned.append(a), raising=False)
        out = snapshot_best_params(blk, "cuda:0", act_floor_bytes=1 << 30)
        assert out["value"].device.type == "cpu"  # host fallback, never raises
        assert warned  # loud, not silent

    def test_floor_exceeded_with_idle_peer_gathers_to_peer(self, monkeypatch):
        """Assert the DECISION (CPU-only build cannot perform the copy)."""
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        # home cuda:0 nearly full; idle cuda:1 has headroom
        _cuda_stub(monkeypatch, {0: 1 << 20, 1: 16 << 30}, device_count=2)
        calls = []
        monkeypatch.setattr(utils_mod, "collect_best_params", lambda block, dev: calls.append(dev) or {"value": None})
        snapshot_best_params(blk, "cuda:0", act_floor_bytes=1 << 30)
        assert calls and calls[0] == torch.device("cuda", 1)  # idle peer, not host


class TestNonCudaAccelerators:
    def test_xpu_home_floor_exceeded_parks_on_host(self, monkeypatch):
        import auto_round.compressors.utils as utils_mod

        blk = _Wrap(8, 4)
        _cuda_stub(monkeypatch, {0: 1 << 20}, device_count=1, dev_type="xpu")
        warned = []
        monkeypatch.setattr(utils_mod.logger, "warning", lambda *a, **k: warned.append(a), raising=False)
        out = snapshot_best_params(blk, "xpu:0", act_floor_bytes=1 << 30)
        assert out["value"].device.type == "cpu"
        assert warned


class TestActFloorHelper:
    def test_floor_is_estimator_backed_and_positive(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        blk = _Wrap(8, 4)
        rows = [torch.randn(1, 16, 4) for _ in range(2)]
        floor = SignRoundQuantizer._snapshot_act_floor_(blk, rows, 2)
        assert floor is not None and floor > 0

    def test_floor_is_none_on_unestimatable_input(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        assert SignRoundQuantizer._snapshot_act_floor_(None, None, 0) is None
