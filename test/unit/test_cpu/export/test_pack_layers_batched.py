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
"""Batched expert packing must be bitwise-identical to per-module pack_layer."""

import pytest
import torch
from torch import nn

from auto_round.export.export_to_autoround.export import pack_layer, pack_layers_batched
from auto_round.utils.model import get_module


class _MoEish(nn.Module):
    def __init__(self, n_experts=4, out=8, inp=32, bias=False):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(inp, out, bias=bias) for _ in range(n_experts)])

    def arm(self, bits=4, group_size=32, sym=True, zp_tensor=False, seed=0):
        g = torch.Generator().manual_seed(seed)
        for e in self.experts:
            with torch.no_grad():
                e.weight.copy_(torch.randn(e.weight.shape, generator=g))
                if e.bias is not None:
                    e.bias.copy_(torch.randn(e.bias.shape, generator=g) * 0.1)
                e.bits = bits
                e.group_size = group_size
                e.sym = sym
                e.act_bits = 16
                groups = e.weight.shape[1] // group_size
                e.scale = torch.rand(e.weight.shape[0], groups, generator=g) * 0.01 + 0.001
                if sym:
                    e.zp = 0
                elif zp_tensor:
                    e.zp = torch.randint(0, 8, (e.weight.shape[0], groups), generator=g)
                else:
                    e.zp = 4
        return self


def _packed(m, name):
    mod = get_module(m, name)
    assert type(mod).__name__ == "QuantLinear", f"{name} not packed"
    return mod


@pytest.mark.parametrize("sym,zp_tensor,bias", [(True, False, False), (False, True, False), (False, False, True)])
def test_batched_matches_per_module(sym, zp_tensor, bias):
    a = _MoEish(bias=bias).arm(sym=sym, zp_tensor=zp_tensor, seed=7)
    b = _MoEish(bias=bias).arm(sym=sym, zp_tensor=zp_tensor, seed=7)
    names = [f"experts.{i}" for i in range(len(a.experts))]

    for n in names:
        pack_layer(n, a, backend="auto_round:exllamav2", device="cpu")
    packed, leftover = pack_layers_batched(names, b, backend="auto_round:exllamav2", device="cpu")

    assert packed == names
    assert leftover == []
    for i, n in enumerate(names):
        pa, pb = _packed(a, n), _packed(b, n)
        torch.testing.assert_close(pb.qweight, pa.qweight, msg=f"qweight mismatch expert {i}")
        torch.testing.assert_close(pb.qzeros, pa.qzeros, msg=f"qzeros mismatch expert {i}")
        torch.testing.assert_close(pb.scales, pa.scales, msg=f"scales mismatch expert {i}")
        if bias:
            torch.testing.assert_close(pb.bias, pa.bias, msg=f"bias mismatch expert {i}")


def test_odd_layers_routed_to_leftover():
    a = _MoEish(n_experts=2).arm()
    a.experts[1].act_bits = 8  # qact path: per-module only
    names = ["experts.0", "experts.1"]
    packed, leftover = pack_layers_batched(names, a, backend="auto_round:exllamav2", device="cpu")
    assert packed == ["experts.0"] or packed == []  # singleton group also goes to leftover
    assert "experts.1" in leftover


def test_meta_weights_skipped():
    a = _MoEish(n_experts=2).arm()
    a.experts[1].to("meta")
    packed, leftover = pack_layers_batched(["experts.0", "experts.1"], a, backend="auto_round:exllamav2", device="cpu")
    assert "experts.1" not in packed and "experts.1" not in leftover  # silently skipped (already packed semantics)


def test_chunking_stays_bitwise_identical():
    a = _MoEish(n_experts=4).arm(seed=3)
    b = _MoEish(n_experts=4).arm(seed=3)
    names = [f"experts.{i}" for i in range(4)]
    for n in names:
        pack_layer(n, a, backend="auto_round:exllamav2", device="cpu")
    # force one expert per chunk
    packed, _ = pack_layers_batched(names, b, backend="auto_round:exllamav2", device="cpu", max_batch_bytes=1)
    assert packed == names
    for i, n in enumerate(names):
        torch.testing.assert_close(_packed(b, n).qweight, _packed(a, n).qweight)
