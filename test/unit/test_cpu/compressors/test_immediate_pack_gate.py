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
"""The batched AutoRound-format packer must not engage for other format families.

``auto_round:llm_compressor`` begins with the ``auto_round`` prefix but packs
through the compressed-tensors ``pack_layer`` -- routing its same-shape module
groups through the AutoRound qweight/qzeros/scales batched packer silently
produces a mixed-format checkpoint (CT-packed modules alongside GPTQ-packed
ones) that no single consumer can load.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from auto_round.compressors.utils import _format_supports_batched_pack
from auto_round.export.export_to_autoround.export import pack_layers_batched


def test_gate_rejects_llm_compressor_family():
    stub = SimpleNamespace(output_format="auto_round:llm_compressor", format_name="llm_compressor")
    assert not _format_supports_batched_pack(stub)


def test_gate_rejects_llm_compressor_without_format_name():
    stub = SimpleNamespace(output_format="auto_round:llm_compressor")
    assert not _format_supports_batched_pack(stub)


def test_gate_accepts_autoround_family():
    assert _format_supports_batched_pack(SimpleNamespace(output_format="auto_round", format_name="auto_round"))
    assert _format_supports_batched_pack(
        SimpleNamespace(output_format="auto_round:exllamav2", format_name="auto_round")
    )


def test_gate_still_rejects_fp_and_fake_variants():
    for fmt in ("auto_round:nv_fp4", "auto_round:mxfp8", "auto_round:fake"):
        assert not _format_supports_batched_pack(SimpleNamespace(output_format=fmt, format_name="auto_round"))


def test_batched_packer_refuses_llm_compressor_backend():
    m = nn.ModuleDict({"a": nn.Linear(64, 8, bias=False)})
    m["a"].bits, m["a"].group_size, m["a"].sym = 4, 32, False
    m["a"].act_bits, m["a"].data_type = 16, "int"
    m["a"].scale = torch.rand(8, 2) * 0.01 + 0.001
    m["a"].zp = torch.zeros(8, 2)
    with pytest.raises(ValueError, match="llm_compressor"):
        pack_layers_batched(["a"], m, backend="auto_round:llm_compressor")
