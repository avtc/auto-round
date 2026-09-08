# Copyright (c) 2024 Intel Corporation
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

import pytest
import torch

from auto_round.utils.stream_placement import (
    _atomic_groups,
    is_placement_template,
    parse_device_template,
    placement_device_of,
    resolve_block_placement,
)


class _Leaf(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n))


class _MoEBlock(torch.nn.Module):
    """Mimics a decoder block: attention leaves + N experts + shared MLP."""

    def __init__(self, n_experts=4):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({"q_proj": _Leaf(8), "k_proj": _Leaf(4), "v_proj": _Leaf(4)})
        self.moe = torch.nn.ModuleDict(
            {
                "experts": torch.nn.ModuleList(
                    [torch.nn.ModuleDict({"gate_proj": _Leaf(2), "down_proj": _Leaf(2)}) for _ in range(n_experts)]
                )
            }
        )
        self.mlp = torch.nn.ModuleDict({"gate_proj": _Leaf(8), "up_proj": _Leaf(8), "down_proj": _Leaf(8)})


class TestIsPlacementTemplate:
    def test_device_list_is_not_a_template(self):
        assert not is_placement_template("0,1")
        assert not is_placement_template("cuda:0")  # single device, no comma
        assert not is_placement_template("0,1,2,3")

    def test_colon_pairs_are_templates(self):
        assert is_placement_template("q_proj:0,mlp.*:1")
        assert is_placement_template("self_attn.*:cuda:0")

    def test_garbage(self):
        assert not is_placement_template(None)
        assert not is_placement_template("")
        assert not is_placement_template("auto")


class TestParseTemplate:
    def test_parse(self):
        spec = parse_device_template("q_proj:0, mlp.* : 1")
        assert spec == {"q_proj": "0", "mlp.*": "1"}

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="module:device"):
            parse_device_template("q_proj,mlp:1")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_device_template(" , ")


class TestAtomicGroups:
    def test_experts_stay_whole(self):
        leaves = [f"moe.experts.{i}.{p}" for i in range(3) for p in ("gate_proj", "down_proj")] + ["mlp.gate_proj"]
        groups = _atomic_groups(leaves)
        keys = [k for k, _ in groups]
        assert "moe.experts.0" in keys and "moe.experts.1" in keys and "moe.experts.2" in keys
        by_key = dict(groups)
        assert by_key["moe.experts.0"] == ["moe.experts.0.gate_proj", "moe.experts.0.down_proj"]
        assert by_key["mlp.gate_proj"] == ["mlp.gate_proj"]


class TestResolvePlacement:
    def test_regex_and_exact_template(self):
        block = _MoEBlock()
        placement = resolve_block_placement(
            block, "self_attn.q_proj:0,self_attn.k_proj:0,self_attn.v_proj:0,moe.*:1,mlp.*:2", fallback="cpu"
        )
        assert placement["self_attn.q_proj"] == torch.device("cuda", 0)
        assert placement["moe.experts.0.gate_proj"] == torch.device("cuda", 1)
        assert placement["mlp.gate_proj"] == torch.device("cuda", 2)
        # every leaf is resolved (matched or fallback), none missing
        assert set(placement) == {n for n, m in block.named_modules() if not list(m.children())}

    def test_fallback_for_unmatched(self):
        block = _MoEBlock()
        placement = resolve_block_placement(block, "mlp.*:1", fallback="cpu")
        assert placement["self_attn.q_proj"] == torch.device("cpu")
        assert placement["mlp.gate_proj"] == torch.device("cuda", 1)

    def test_device_list_balances_contiguously(self):
        block = _MoEBlock()
        placement = resolve_block_placement(block, "0,1", fallback="cpu")
        devices = {d for d in placement.values()}
        assert torch.device("cuda", 0) in devices
        assert torch.device("cuda", 1) in devices
        # contiguity: a whole expert shares one device
        assert placement["moe.experts.2.gate_proj"] == placement["moe.experts.2.down_proj"]
        seen = [
            placement["moe.experts.0.gate_proj"],
            placement["moe.experts.1.gate_proj"],
            placement["moe.experts.2.gate_proj"],
            placement["moe.experts.3.gate_proj"],
        ]
        # a contiguous partition over 4 experts crosses at most once
        switches = sum(1 for a, b in zip(seen, seen[1:]) if a != b)
        assert switches <= 1

    def test_single_device_list(self):
        block = _MoEBlock()
        placement = resolve_block_placement(block, "0", fallback="cpu")
        assert set(placement.values()) == {torch.device("cuda", 0)}

    def test_invalid_pattern_raises(self):
        block = _MoEBlock()
        with pytest.raises(ValueError, match="invalid device_map pattern"):
            resolve_block_placement(block, "[q_proj:0", fallback="cpu")


class TestPlacementDeviceOf:
    def test_strip_leaf_attr(self):
        placement = {"self_attn.q_proj": torch.device("cuda", 1)}
        assert placement_device_of(placement, "self_attn.q_proj.weight", "cuda:0") == "cuda:1"
        # names outside the placement scope keep the fallback
        assert placement_device_of(placement, "model.layers.0.self_attn.q_proj.weight", "cuda:0") == "cuda:0"

    def test_unmatched_falls_back(self):
        assert placement_device_of({}, "x.weight", "cuda:0") == "cuda:0"
        assert placement_device_of({"a.b": torch.device("cuda", 2)}, "c.d.bias", "cuda:0") == "cuda:0"
