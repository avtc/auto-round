# Copyright (c) 2025 Intel Corporation
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

"""Tests for probe-based best-params snapshot parking (utils/snapshot_parking.py)."""

import logging

import pytest
import torch

from auto_round.utils import snapshot_parking


class _FakeWrapper(torch.nn.Module):
    """Minimal stand-in for a WrapperLinear: a params dict plus a home device."""

    def __init__(self, n_elems=1000, home="cpu"):
        super().__init__()
        self.device = torch.device(home)
        self.params = {
            "value": torch.zeros(n_elems, dtype=torch.float32),
            "min_scale": torch.ones(4, dtype=torch.float32),
            "max_scale": torch.ones(4, dtype=torch.float32),
        }


@pytest.fixture()
def probes(monkeypatch):
    """Route the module's device probes to a dict of free bytes per device."""

    class _Probes:
        def __init__(self):
            self.free = {}
            self.count = 0

        def install(self):
            monkeypatch.setattr(snapshot_parking, "_free_bytes", lambda dev: self.free.get(dev, None))
            monkeypatch.setattr(snapshot_parking, "_cuda_device_count", lambda: self.count)

    p = _Probes()
    p.install()
    return p


class TestSnapshotBytes:
    def test_counts_real_tensor_bytes(self):
        w = _FakeWrapper(n_elems=1000)
        # value: 1000 * 4 bytes, min_scale/max_scale: 4 * 4 bytes each
        assert snapshot_parking.snapshot_bytes(w) == (1000 + 4 + 4) * 4

    def test_no_params_is_zero(self):
        w = _FakeWrapper()
        w.params = {}
        assert snapshot_parking.snapshot_bytes(w) == 0


class TestSelectSnapshotDevice:
    def test_home_when_half_free_pool_covers(self, probes):
        probes.free = {torch.device("cuda:0"): 10 * 2**30}
        probes.count = 1
        w = _FakeWrapper(n_elems=64 * 2**20, home="cuda:0")  # ~256 MiB snapshot
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cuda:0")

    def test_peer_when_home_pool_too_small(self, probes):
        probes.free = {
            torch.device("cuda:0"): 2**30,  # half = 512 MiB < snapshot
            torch.device("cuda:1"): 10 * 2**30,
        }
        probes.count = 2
        w = _FakeWrapper(n_elems=256 * 2**20, home="cuda:0")
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cuda:1")

    def test_peer_needs_headroom(self, probes):
        need = 256 * 2**20 * 4
        probes.free = {
            torch.device("cuda:0"): 2**30,
            # exactly `need` free: the 10% headroom must decline it
            torch.device("cuda:1"): need,
        }
        probes.count = 2
        w = _FakeWrapper(n_elems=256 * 2**20, home="cuda:0")
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cpu")

    def test_host_when_everything_is_tight(self, probes):
        probes.free = {
            torch.device("cuda:0"): 2**30,
            torch.device("cuda:1"): 2**30,
        }
        probes.count = 2
        w = _FakeWrapper(n_elems=256 * 2**20, home="cuda:0")
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cpu")

    def test_non_cuda_home_returns_host(self, probes):
        probes.free = {torch.device("cpu"): 10 * 2**30}
        probes.count = 0
        w = _FakeWrapper(n_elems=64 * 2**20, home="cpu")
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cpu")

    def test_probe_failure_falls_back_to_host(self, probes):
        probes.free = {}  # every probe returns None
        probes.count = 4
        w = _FakeWrapper(n_elems=64 * 2**20, home="cuda:0")
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cpu")

    def test_home_device_read_from_params_when_attr_missing(self, probes):
        probes.free = {torch.device("cuda:0"): 10 * 2**30}
        probes.count = 1
        w = _FakeWrapper(n_elems=64 * 2**20, home="cuda:0")
        del w.device  # device attr gone; fall back to param tensors
        w.params["value"] = w.params["value"].to("cpu")
        # params now live on cpu -> home is cpu -> host result, no crash
        assert snapshot_parking.select_snapshot_device(w) == torch.device("cpu")

    def test_peer_selection_logged_once(self, probes):
        probes.free = {
            torch.device("cuda:0"): 2**30,
            torch.device("cuda:1"): 10 * 2**30,
        }
        probes.count = 2
        w = _FakeWrapper(n_elems=256 * 2**20, home="cuda:0")
        snapshot_parking._announced.clear()

        # the autoround logger does not propagate, so capture via a direct handler
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture(level=logging.INFO)
        library_logger = snapshot_parking.logger
        library_logger.addHandler(handler)
        try:
            snapshot_parking.select_snapshot_device(w)
            snapshot_parking.select_snapshot_device(w)
        finally:
            library_logger.removeHandler(handler)
        peer_lines = [r for r in captured if "parking the best-params snapshot" in r.getMessage()]
        assert len(peer_lines) == 1, "peer parking must be announced once per wrapper"
        assert "cuda:1" in peer_lines[0].getMessage()


class TestBestParamDeviceIntegration:
    """The quantizer-level hook keeps small snapshots and None wrappers unchanged."""

    @pytest.fixture()
    def quant(self):
        from types import MethodType, SimpleNamespace

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        q = SimpleNamespace(compress_context=SimpleNamespace(cache_device=torch.device("cuda:7")))
        q._best_param_device = MethodType(SignRoundQuantizer._best_param_device, q)
        return q

    def test_small_snapshots_keep_cache_device(self, quant):
        assert quant._best_param_device(2**20) == torch.device("cuda:7")

    def test_huge_snapshot_without_wrapper_parks_on_host(self, quant):
        assert quant._best_param_device(2**28) == torch.device("cpu")

    def test_huge_snapshot_with_wrapper_uses_probe_ladder(self, quant, probes):
        probes.free = {
            torch.device("cuda:0"): 2**30,
            torch.device("cuda:1"): 10 * 2**30,
        }
        probes.count = 2
        w = _FakeWrapper(n_elems=256 * 2**20, home="cuda:0")
        assert quant._best_param_device(2**28, w) == torch.device("cuda:1")
