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
    FlowProbe,
    _atomic_groups,
    block_signature,
    is_placement_template,
    parse_device_template,
    partition_flow_order,
    placement_device_of,
    resolve_block_placement,
)


class _Leaf(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x + 0.0 * self.weight.sum()


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

    def test_shared_group_stays_whole_under_device_list(self):
        block = _MoEBlock()
        # q/k/v as a shared-quantization group: must never straddle devices
        placement = resolve_block_placement(
            block, "0,1", fallback="cpu", shared_leaf_groups=[["q_proj", "k_proj", "v_proj"]]
        )
        devs = {
            placement["self_attn.q_proj"],
            placement["self_attn.k_proj"],
            placement["self_attn.v_proj"],
        }
        assert len(devs) == 1, f"shared group straddles: {devs}"
        # balance still holds across the two devices overall
        assert len({d for d in placement.values()}) == 2

    def test_shared_group_straddle_warns_under_template(self, monkeypatch):
        import auto_round.utils.stream_placement as sp

        warned = []
        monkeypatch.setattr(sp.logger, "warning", lambda msg, *a, **k: warned.append(msg % a if a else msg))
        block = _MoEBlock()
        resolve_block_placement(
            block,
            "self_attn.q_proj:0,self_attn.k_proj:1,self_attn.v_proj:0",
            fallback="cpu",
            shared_leaf_groups=[["q_proj", "k_proj", "v_proj"]],
        )
        assert any("straddles devices" in w for w in warned)

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


class _SwappedExec(torch.nn.Module):
    """Registration order (mlp first) deliberately differs from execution."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.ModuleDict({"gate_proj": _Leaf(8), "down_proj": _Leaf(8)})
        self.attn = torch.nn.ModuleDict({"q_proj": _Leaf(4), "v_proj": _Leaf(4)})
        # registration: mlp.*, attn.*; execution below: attn first

    def forward(self, x):
        x = self.attn["q_proj"](x)
        x = self.attn["v_proj"](x)
        x = self.mlp["gate_proj"](x)
        x = self.mlp["down_proj"](x)
        return x


class TestBlockSignature:
    def test_digit_indices_are_stripped(self):
        a, b = _MoEBlock(n_experts=4), _MoEBlock(n_experts=4)
        assert block_signature(a) == block_signature(b)

    def test_different_layouts_differ(self):
        assert block_signature(_MoEBlock()) != block_signature(_SwappedExec())

    def test_different_expert_counts_differ(self):
        assert block_signature(_MoEBlock(n_experts=3)) != block_signature(_MoEBlock(n_experts=4))


class TestPartitionFlowOrder:
    def test_atoms_balance_freely(self):
        # 4 expert atoms of 4 bytes each + 2 loose leaves of 1 byte; 2 devices.
        # Atom sizes dominate: balance wants 2+2 experts per device, NOT
        # 0-1 on dev0 and 2-3 on dev1 bound to flow order.
        units = []
        for i in range(4):
            units.append(([f"moe.experts.{i}.gate_proj", f"moe.experts.{i}.down_proj"], 4, 100, True))
        units.append((["attn.q_proj"], 1, 10, False))
        units.append((["attn.v_proj"], 1, 10, False))
        placement = partition_flow_order(units, [torch.device("cpu", 0), torch.device("cpu", 1)])
        load = {}
        for names, b, _i, _a in units:
            dev = placement[names[0]]
            load[dev] = load.get(dev, 0) + b
        # expert atoms whole on one device each
        for i in range(4):
            assert placement[f"moe.experts.{i}.gate_proj"] == placement[f"moe.experts.{i}.down_proj"]
        loads = sorted(load.values())
        assert loads[1] - loads[0] <= 4, f"unbalanced: {loads}"

    def test_contiguous_units_cut_at_cheapest_boundary(self):
        # 3 loose units, cut cost = next unit input bytes; the cheap boundary
        # sits between u1 and u2. With 2 devices, the cut must land there.
        units = [
            (["u0"], 5, 0, False),
            (["u1"], 5, 1000, False),  # boundary u0|u1 expensive
            (["u2"], 5, 1, False),  # boundary u1|u2 cheap
        ]
        placement = partition_flow_order(units, [torch.device("cpu", 0), torch.device("cpu", 1)])
        assert placement["u0"] == placement["u1"]
        assert placement["u1"] != placement["u2"]

    def test_single_device_places_everything(self):
        units = [(["a"], 1, 0, False), (["b"], 1, 0, True)]
        placement = partition_flow_order(units, [torch.device("cpu")])
        assert set(placement.values()) == {torch.device("cpu")}


class TestFlowProbe:
    def test_records_execution_order_not_registration(self):
        block = _SwappedExec()
        recorded = []

        def on_complete(records):
            recorded.extend(records)

        FlowProbe(block, on_complete)
        x = torch.zeros(1, 4)
        block(x)
        order = [name for name, _b in recorded]
        assert order == ["attn.q_proj", "attn.v_proj", "mlp.gate_proj", "mlp.down_proj"]

    def test_hooks_are_one_shot(self):
        block = _SwappedExec()
        calls = []
        FlowProbe(block, lambda records: calls.append(list(records)))
        x = torch.zeros(1, 4)
        block(x)
        block(x)  # second forward must not re-record
        assert len(calls) == 1

    def test_input_bytes_captured(self):
        block = _SwappedExec()
        recorded = []
        FlowProbe(block, lambda records: recorded.extend(records))
        block(torch.zeros(2, 4))
        assert all(b > 0 for _n, b in recorded)
