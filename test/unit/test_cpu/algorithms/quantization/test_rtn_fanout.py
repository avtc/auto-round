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
"""Optimized-RTN tuning fan-out: per-module GPU scheduling and device override."""

from unittest import mock

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer


class _RecordingQuantizer:
    """Stands in for the quantizer: records (device, order) per call."""

    def __init__(self, config):
        self.config = config
        self.calls = []

    # borrow the real scheduling method unbound
    _quantize_targets = OptimizedRTNQuantizer._quantize_targets

    def quantize_layer_outside_block(self, m, device_override=None):
        self.calls.append((m, device_override))


def _targets(n=10):
    mods = []
    for i in range(n):
        m = nn.Linear(8, 8, bias=False)
        m.global_name = f"layers.0.mlp.{i}"
        mods.append(m)
    return mods


class TestQuantizeTargetsFanout:
    def test_serial_when_single_gpu(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=None))
        with mock.patch("torch.cuda.device_count", return_value=1):
            q._quantize_targets(_targets())
        assert all(dev is None for _, dev in q.calls)
        assert len(q.calls) == 10

    def test_serial_forced_by_config(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=False))
        with mock.patch("torch.cuda.device_count", return_value=8):
            q._quantize_targets(_targets())
        assert all(dev is None for _, dev in q.calls)

    def test_round_robin_across_gpus(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=None))
        with mock.patch("torch.cuda.device_count", return_value=4):
            q._quantize_targets(_targets())
        assert len(q.calls) == 10
        # deterministic round-robin assignment by submission index; call
        # completion order varies with thread scheduling, so assert the
        # module -> device mapping rather than the call order
        by_name = {m.global_name: dev for m, dev in q.calls}
        for i in range(10):
            assert by_name[f"layers.0.mlp.{i}"] == f"cuda:{i % 4}"

    def test_round_robin_respects_worker_cap(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=True))
        mods = _targets(3)
        with mock.patch("torch.cuda.device_count", return_value=8):
            q._quantize_targets(mods)
        assert [dev for _, dev in q.calls] == ["cuda:0", "cuda:1", "cuda:2"]

    def test_empty_targets_noop(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=None))
        q._quantize_targets([])
        assert q.calls == []
