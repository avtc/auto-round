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
"""Tests for per-block calibration-state re-homing in the tuning loop."""

from unittest import mock

import torch

from auto_round.algorithms.quantization.sign_round.quantizer import _rehome_calibration_state


def test_cpu_device_is_noop():
    lst = [torch.randn(2, 2), torch.randn(2, 2)]
    orig = [t.data_ptr() for t in lst]
    moved = _rehome_calibration_state(lst, [torch.randn(2, 2)], torch.device("cpu"))
    assert moved == 0
    assert [t.data_ptr() for t in lst] == orig


def test_cuda_guard_skips_when_vram_insufficient(monkeypatch):
    lst = [torch.randn(2, 2)]  # 16 bytes needed, but free will be tiny
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (0, 1 << 30), raising=False)
    moved = _rehome_calibration_state(lst, [], torch.device("cuda:1"))
    assert moved == 0
    assert lst[0].device.type == "cpu"  # untouched


def test_cuda_rehome_moves_when_room(monkeypatch):
    lst = [torch.randn(2, 2), torch.randn(2, 2)]
    refs = [torch.randn(2, 2)]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (8 << 30, 16 << 30), raising=False)
    with mock.patch.object(torch.Tensor, "to", return_value=None) as to_mock:
        moved = _rehome_calibration_state(lst, refs, torch.device("cuda:1"))
    assert moved == 3  # both lists walked: 2 inputs + 1 fp_output row
    assert to_mock.call_count == 3


def test_all_local_tensors_move_nothing(monkeypatch):
    # tensors already on the target device: needed == 0, no mem_get_info call at all
    lst = [torch.randn(2, 2)]
    calls = []

    def _fail(dev):
        calls.append(dev)
        raise AssertionError("mem_get_info must not be called when nothing needs moving")

    monkeypatch.setattr(torch.cuda, "mem_get_info", _fail, raising=False)
    with mock.patch.object(torch.Tensor, "device", new_callable=mock.PropertyMock) as dev_mock:
        dev_mock.return_value = torch.device("cuda:1")
        moved = _rehome_calibration_state(lst, [], torch.device("cuda:1"))
    assert moved == 0
    assert calls == []


def test_non_list_inputs_ignored():
    moved = _rehome_calibration_state({"a": torch.randn(2)}, None, torch.device("cpu"))
    assert moved == 0


def test_stream_out_device_none_without_stamp():
    import torch.nn as nn

    from auto_round.algorithms.composer import _stream_out_device

    assert _stream_out_device(nn.Linear(2, 2)) is None


def test_stream_out_device_follows_stamp():
    import torch.nn as nn

    from auto_round.algorithms.composer import _stream_out_device

    block = nn.Linear(2, 2)
    block._stream_home_device = torch.device("cuda:1")
    assert _stream_out_device(block) == torch.device("cuda:1")
