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
"""Optimized-RTN tuning fan-out and expert batching: scheduling + equivalence."""

from unittest import mock

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer


def _mk_expert(parent, idx, proj="gate_proj", out=16, inn=32, imatrix=False):
    m = nn.Linear(inn, out, bias=False)
    m.global_name = f"{parent}.experts.{idx}.{proj}"
    m.bits, m.sym, m.group_size, m.data_type = 4, False, 32, "int"
    m.scale_dtype = torch.float16
    if imatrix:
        m.imatrix = torch.rand(inn) + 0.1
    return m


class _RecordingQuantizer:
    """Stands in for the quantizer: records (device, order) per module call."""

    def __init__(self, config, batch=False):
        self.config = config
        self.calls = []
        self.batches = []
        self.model_context = mock.MagicMock(is_moe_model=False)
        if not batch:
            self._split_expert_batches = lambda targets, n_jobs_hint=0: ([], targets)
        else:
            self._quantize_expert_batch = self._record_expert_batch

    # borrow the real methods unbound
    _quantize_targets = OptimizedRTNQuantizer._quantize_targets
    _expert_search_active = OptimizedRTNQuantizer._expert_search_active
    _split_expert_batches = OptimizedRTNQuantizer._split_expert_batches
    _quantize_expert_batch = OptimizedRTNQuantizer._quantize_expert_batch

    def quantize_layer_outside_block(self, m, device_override=None):
        self.calls.append((m, device_override))

    def _record_expert_batch(self, mods, device):
        self.batches.append((mods, device))
        return True

    _quantize_expert_batch = OptimizedRTNQuantizer._quantize_expert_batch


def _dense_targets(n=10):
    mods = []
    for i in range(n):
        m = nn.Linear(8, 8, bias=False)
        m.global_name = f"layers.0.mlp.linear.{i}"
        mods.append(m)
    return mods


class TestQuantizeTargetsFanout:
    def test_serial_when_single_gpu(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=True))
        with mock.patch("torch.cuda.device_count", return_value=1):
            q._quantize_targets(_dense_targets())
        assert all(dev is None for _, dev in q.calls)
        assert len(q.calls) == 10

    def test_serial_forced_by_config(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=False))
        with mock.patch("torch.cuda.device_count", return_value=8):
            q._quantize_targets(_dense_targets())
        assert all(dev is None for _, dev in q.calls)

    def test_round_robin_across_gpus(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=True))
        with mock.patch("torch.cuda.device_count", return_value=4):
            q._quantize_targets(_dense_targets())
        assert len(q.calls) == 10
        # deterministic round-robin assignment by submission index; call
        # completion order varies with thread scheduling, so assert the
        # module -> device mapping rather than the call order
        by_name = {m.global_name: dev for m, dev in q.calls}
        for i in range(10):
            assert by_name[f"layers.0.mlp.linear.{i}"] == f"cuda:{i % 4}"

    def test_round_robin_respects_worker_cap(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=True))
        mods = _dense_targets(3)
        with mock.patch("torch.cuda.device_count", return_value=8):
            q._quantize_targets(mods)
        assert sorted(dev for _, dev in q.calls) == ["cuda:0", "cuda:1", "cuda:2"]

    def test_empty_targets_noop(self):
        q = _RecordingQuantizer(RTNConfig(parallel_tuning=True))
        q._quantize_targets([])
        assert q.calls == []


class TestExpertBatching:
    def _experts(self, n=4, imatrix=False):
        return [_mk_expert("layers.1.mlp", i, imatrix=imatrix) for i in range(n)]

    def test_split_groups_same_shape(self):
        q = _RecordingQuantizer(RTNConfig(), batch=True)
        targets = self._experts(4) + _dense_targets(3)
        batches, singles = q._split_expert_batches(targets)
        assert len(batches) == 1 and len(batches[0]) == 4
        assert len(singles) == 3
        # different projection names and shapes form separate groups
        targets = self._experts(2) + self._experts(2, imatrix=False)
        targets += [_mk_expert("layers.1.mlp", 0, proj="up_proj") for _ in range(2)]
        targets += [_mk_expert("layers.2.mlp", i, out=8) for i in range(2)]
        batches, singles = q._split_expert_batches(targets)
        # gate@l1 (x4, identical keys merge), up@l1 (x2), gate@l2 (x2, shape differs)
        sizes = sorted(len(b) for b in batches)
        assert sizes == [2, 2, 4]

    def test_job_hint_splits_groups_to_fill_gpus(self):
        """n_jobs_hint splits same-shape groups into ~equal chunks so the
        fan-out fills every GPU; hint=0 keeps monolithic groups."""
        q = _RecordingQuantizer(RTNConfig(), batch=True)
        targets = self._experts(6) + [_mk_expert("layers.1.mlp", i, proj="up_proj") for i in range(6)]

        batches, singles = q._split_expert_batches(targets, n_jobs_hint=0)
        assert sorted(len(b) for b in batches) == [6, 6]

        # 8-GPU hint over 2 groups -> 4 chunks-per-group target; ceil-split of
        # 6 by 2 yields 3 chunks of 2 per group -> 6 jobs of 2
        batches, singles = q._split_expert_batches(targets, n_jobs_hint=8)
        assert len(batches) == 6
        sizes = sorted(len(b) for b in batches)
        assert sizes == [2, 2, 2, 2, 2, 2]
        # membership preserved exactly once across chunks
        chunked = [m for b in batches for m in b]
        assert sorted(id(m) for m in chunked) == sorted(id(m) for m in targets)
        assert not singles

    def test_batches_are_scheduled_as_single_jobs(self, caplog):
        import logging

        # expert batching only pays off when the NeUQI search actually runs,
        # so the gate requires the explicit opt-in
        q = _RecordingQuantizer(RTNConfig(asym_search="neuqi", parallel_tuning=True), batch=True)
        ar_logger = logging.getLogger("autoround")
        ar_logger.addHandler(caplog.handler)  # autoround logger has propagate=False
        try:
            with mock.patch("torch.cuda.device_count", return_value=2):
                q._quantize_targets(self._experts(4) + _dense_targets(2))
        finally:
            ar_logger.removeHandler(caplog.handler)
        # 2-GPU hint splits the 4-expert group into 2 chunks of 2 (one job each)
        assert len(q.batches) == 2 and sorted(len(b[0]) for b in q.batches) == [2, 2]
        assert len(q.calls) == 2  # dense modules as singles
        # the fan-out timing summary must be logged
        assert any("fan-out done in" in r.getMessage() for r in caplog.records)

    def test_default_auto_skips_expert_batching(self):
        """Without --enable_neuqi the plain min/max initializer needs no search,
        so experts are quantized as singles and no batch job is scheduled."""
        q = _RecordingQuantizer(RTNConfig(), batch=True)
        with mock.patch("torch.cuda.device_count", return_value=2):
            q._quantize_targets(self._experts(4))
        assert q.batches == []
        assert len(q.calls) == 4

    def test_batched_matches_per_module_search_no_imatrix(self):
        """imatrix-free (uniform weight) batching skips materializing a ones
        tensor and must stay bit-identical to the per-module search."""
        torch.manual_seed(11)
        mods = self._experts(3, imatrix=False)
        assert not any(hasattr(m, "imatrix") for m in mods)
        copies = [_mk_expert("layers.1.mlp", i, imatrix=False) for i in range(3)]
        for m, c in zip(mods, copies):
            with torch.no_grad():
                c.weight.copy_(m.weight)

        q = _RecordingQuantizer(RTNConfig())
        assert q._quantize_expert_batch(mods, "cpu") is True

        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym

        for m, c in zip(mods, copies):
            qdq, scale, zp = quant_tensor_opt_rtn_asym(
                c.weight.data, bits=c.bits, group_size=c.group_size, v=0.0, imatrix=None
            )
            assert torch.allclose(m.weight.data, qdq, atol=1e-6), m.global_name
            assert torch.allclose(m.scale.float(), scale.reshape(m.weight.shape[0], -1).float(), atol=1e-6)
            assert torch.allclose(m.zp.float(), zp.reshape(m.weight.shape[0], -1).float(), atol=1e-6)

    def test_batched_matches_per_module_search(self):
        """Stacked expert search must reproduce per-module results exactly."""
        torch.manual_seed(7)
        mods = self._experts(3, imatrix=True)
        copies = [_mk_expert("layers.1.mlp", i, imatrix=True) for i in range(3)]
        for m, c in zip(mods, copies):
            with torch.no_grad():
                c.weight.copy_(m.weight)
            c.imatrix = m.imatrix.clone()

        # batch=False keeps the REAL _quantize_expert_batch on the stub
        q = _RecordingQuantizer(RTNConfig())
        assert q._quantize_expert_batch(mods, "cpu") is True

        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym

        for m, c in zip(mods, copies):
            qdq, scale, zp = quant_tensor_opt_rtn_asym(
                c.weight.data, bits=c.bits, group_size=c.group_size, v=0.0, imatrix=c.imatrix
            )
            assert torch.allclose(m.weight.data, qdq, atol=1e-6), m.global_name
            assert torch.allclose(m.scale.float(), scale.reshape(m.weight.shape[0], -1).float(), atol=1e-6)
            assert torch.allclose(m.zp.float(), zp.reshape(m.weight.shape[0], -1).float(), atol=1e-6)
