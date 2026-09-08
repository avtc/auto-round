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


class TestContainerParamPlacement:
    """Container-direct params (e.g. GDN ``linear_attn.A_log``) must not
    fall to the fallback: the probe, the tune entry and staging all key off
    the first parameter's device."""

    def _gdn_block(self):
        class GDN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A_log = torch.nn.Parameter(torch.zeros(2))
                self.in_proj = torch.nn.Linear(4, 8)
                self.conv = torch.nn.Conv1d(8, 8, 3, groups=8)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = GDN()
                self.mlp = torch.nn.Linear(4, 4)

        return Layer()

    def test_device_list_placement_covers_container_params(self):
        block = self._gdn_block()
        placement = resolve_block_placement(block, "1,2", torch.device("cpu"))
        assert placement["linear_attn"] == placement["linear_attn.in_proj"]
        # staging resolves the container's direct tensors through the module key
        from auto_round.utils.stream_placement import placement_device_of

        assert placement_device_of(placement, "linear_attn.A_log", "cpu:9") == str(placement["linear_attn"])

    def test_template_placement_covers_container_params(self):
        block = self._gdn_block()
        placement = resolve_block_placement(block, "linear_attn.*:1,mlp:2", torch.device("cpu"))
        assert placement["linear_attn"] == placement["linear_attn.in_proj"]
        assert str(placement["mlp"]) == "cuda:2"

    def test_first_placement_device_is_in_map(self):
        block = self._gdn_block()
        placement = resolve_block_placement(block, "1,2", torch.device("cpu"))
        from auto_round.utils.stream_placement import first_placement_device

        assert str(first_placement_device(placement, "cpu:9")) == "cuda:1"
        assert first_placement_device({}, "cpu:9") == "cpu:9"

    def test_staging_miss_warns_and_uses_fallback(self, capfd):
        from auto_round.utils.stream_placement import placement_device_of

        placement = {"mlp": "cuda:1"}
        got = placement_device_of(placement, "zz_probe_only.rope.weight", "cuda:1")
        assert got == "cuda:1"
        # project loggers do not propagate: assert on the emitted line
        out = capfd.readouterr()
        assert "zz_probe_only.rope.weight" in out.err and "matches no placement entry" in out.err
        # latched: a second miss of the same leaf does not repeat the warning
        placement_device_of(placement, "zz_probe_only.rope.bias", "cuda:1")
        assert capfd.readouterr().err.count("zz_probe_only") == 0


class TestRehomeMapped:
    def test_container_entry_moves_only_direct_tensors(self):
        from auto_round.compressors.utils import rehome_block_mapped_

        calls = {}

        class Rec(torch.nn.Module):
            def __init__(self, key):
                super().__init__()
                self.key = key
                calls[key] = []
                self.w = torch.nn.Parameter(torch.zeros(1))

            def _apply(self, fn, recurse=True):
                calls[self.key].append(recurse)
                return self

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leafmod = Rec("leaf")
                box = Rec("box")
                box.kid = Rec("kid")
                self.box = box

        parent = Parent()
        rehome_block_mapped_(parent, {"leafmod": torch.device("cpu"), "box": torch.device("cpu")}, torch.device("cpu"))
        # placement loop: leaves recurse, containers touch direct tensors only
        assert calls["leaf"][0] is True
        assert calls["box"][0] is False
        # the _rest sweep afterwards calls every child once more (the stub's
        # box._apply does not descend into kid, so kid stays at zero)
        assert len(calls["leaf"]) == 2 and len(calls["box"]) == 2 and calls["kid"] == []

    def test_unmatched_tensor_warns_with_name(self, capfd):
        from auto_round.compressors.utils import rehome_block_mapped_

        class Orphan(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.orphan_bias = torch.nn.Parameter(torch.zeros(2))

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.placed = torch.nn.Linear(2, 2)
                self.orphan = Orphan()

        parent = Parent()
        moved = rehome_block_mapped_(parent, {"placed": torch.device("cpu")}, torch.device("cpu"))
        out = capfd.readouterr()
        assert "orphan.orphan_bias" in out.err and "matched no placement entry" in out.err
        # already on the cpu default: counted as a miss, not a move
        assert moved == 0


class TestSharedExpertAtomicity:
    def test_shared_experts_and_shared_mlp_stay_whole(self):
        from auto_round.utils.stream_placement import _atomic_groups

        leaves = [
            "self_attn.q_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
            "mlp.experts.0.gate_proj",
            "mlp.experts.0.up_proj",
            "mlp.down_proj",
            "shared_mlp.gate_proj",
            "shared_mlp.up_proj",
        ]
        groups = dict(_atomic_groups(leaves))
        assert groups["mlp.shared_experts"] == [
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ]
        assert groups["shared_mlp"] == ["shared_mlp.gate_proj", "shared_mlp.up_proj"]
        # unrelated dense mlp leaves stay independent units
        assert groups["mlp.down_proj"] == ["mlp.down_proj"]

    def test_device_list_keeps_shared_experts_on_one_device(self):
        import torch

        class Sh(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(4, 4)
                self.up_proj = torch.nn.Linear(4, 4)
                self.down_proj = torch.nn.Linear(4, 4)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_experts = Sh()
                self.down_proj = torch.nn.Linear(4, 4)

        placement = resolve_block_placement(M(), "1,2", torch.device("cpu"))
        devs = {placement[k] for k in placement if k.startswith("shared_experts")}
        assert len(devs) == 1, placement


class TestMappedEngagementPredicate:
    def _orch(self, prefetch, prefetch_map):
        from auto_round.compressors import orchestrator as orch

        o = object.__new__(orch.CompressionOrchestrator)
        o.stream_prefetch = prefetch
        o.stream_prefetch_device_map = prefetch_map
        return o

    def _set_map(self, monkeypatch, device_map):
        import auto_round.compressors.orchestrator as orch

        class _DM:
            pass

        dm = _DM()
        dm.device_map = device_map
        monkeypatch.setattr(orch, "device_manager", dm)

    def test_plain_list_prefetch_off_engages_mapped(self, monkeypatch):
        self._set_map(monkeypatch, "0,1,2,3")
        assert self._orch("off", None)._stream_mapped_enabled() is True

    def test_plain_list_prefetch_auto_keeps_rotation(self, monkeypatch):
        self._set_map(monkeypatch, "0,1,2,3")
        assert self._orch("auto", None)._stream_mapped_enabled() is False

    def test_plain_list_prefetch_on_keeps_rotation(self, monkeypatch):
        self._set_map(monkeypatch, "0,1")
        assert self._orch("on", None)._stream_mapped_enabled() is False

    def test_single_device_prefetch_off_stays_single_home(self, monkeypatch):
        self._set_map(monkeypatch, "0")
        assert self._orch("off", None)._stream_mapped_enabled() is False

    def test_template_and_prefetch_map_engage(self, monkeypatch):
        self._set_map(monkeypatch, "q_proj:0,mlp:1")
        assert self._orch("off", None)._stream_mapped_enabled() is True
        self._set_map(monkeypatch, "0,1")
        assert self._orch("auto", "2,3")._stream_mapped_enabled() is True


class TestPlacementDeviceOfPrefixFallback:
    """Direct-parameter containers (custom archs keep sibling tensors as
    direct params of one module) resolve through the parent chain."""

    def test_direct_param_container_hits_parent(self):
        from auto_round.utils.stream_placement import placement_device_of

        placement = {"mlp.shared_mlp": "cuda:1", "mlp": "cuda:2"}
        assert placement_device_of(placement, "mlp.shared_mlp.gate_proj.weight", "cpu") == "cuda:1"
        assert placement_device_of(placement, "mlp.shared_mlp.up_proj.weight", "cpu") == "cuda:1"
        # container-level buffer reaches the container key
        assert placement_device_of(placement, "mlp.e_score_correction_bias", "cpu") == "cuda:2"

    def test_total_miss_still_warns_and_falls_back(self, capfd):
        from auto_round.utils.stream_placement import placement_device_of

        got = placement_device_of({"mlp": "cuda:1"}, "zz_direct_only.rope.weight", "cuda:1")
        assert got == "cuda:1"
        out = capfd.readouterr()
        assert "zz_direct_only.rope.weight" in out.err and "matches no placement entry" in out.err


class TestSharedLayersEntryRouting:
    def test_shared_layers_routed_to_compressor(self):
        from auto_round.autoround import _ENTRY_KWARG_OWNERS

        assert _ENTRY_KWARG_OWNERS.get("shared_layers") == "compressor"


class TestTuneStatePricing:
    def test_linear_costs_four_times_norm(self):
        import torch

        from auto_round.utils.stream_placement import _leaf_tune_state_bytes

        lin = torch.nn.Linear(100, 100)  # 10k params, bf16-sized by dtype of params (fp32 here)
        norm = torch.nn.LayerNorm(100)  # 200 params
        lin_b = sum(p.numel() * p.element_size() for p in lin.parameters())
        assert _leaf_tune_state_bytes(lin) == 7 * lin_b
        assert _leaf_tune_state_bytes(norm) == sum(p.numel() * p.element_size() for p in norm.parameters())

    def test_partition_balances_resident_cost_with_activation(self):
        from auto_round.utils.stream_placement import partition_flow_order

        devs = ["d0", "d1"]
        # unit A: tiny params but a HUGE input activation (the mlp-container
        # case: full-hidden copy resides on its device); unit B: big state, no input
        units = [(["a"], 1, 1000, False), (["b"], 500, 0, False), (["c"], 500, 0, False)]
        place = partition_flow_order(units, devs)
        # 'a' (resident 1001) must not share its device with the bulk of b+c
        dev_a = place["a"]
        other = [n for n, d in place.items() if d == dev_a]
        assert sum(1 for n in other if n in ("b", "c")) <= 1  # at most one heavy unit alongside

    def test_reserve_pushes_atoms_off_reserved_device(self):
        from auto_round.utils.stream_placement import partition_flow_order

        devs = ["d0", "d1", "d2", "d3"]
        units = [([f"e{i}"], 100, 1, True) for i in range(40)]  # uniform atoms
        place = partition_flow_order(units, devs, reserves={"d0": 250})
        per = {d: sum(1 for n, dd in place.items() if dd == d) for d in devs}
        assert per["d0"] < per["d1"] and per["d0"] < per["d3"]  # pushed off the reserved device
        others = [per["d1"], per["d2"], per["d3"]]
        assert max(others) - min(others) <= 1  # indivisible atoms: +-1
        # effective loads (atoms + reserve) stay balanced within one atom
        eff = [per[d] * 100 + (250 if d == "d0" else 0) for d in devs]
        assert max(eff) - min(eff) <= 100


class TestFlowProbeRoutedDetection:
    def _probe_block(self):
        import torch
        import torch.nn as nn

        class Routed(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(16, 16)  # any child: makes it a container

            def forward(self, hidden, top_k_index, top_k_weights):
                return hidden

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Routed()

            def forward(self, h):
                idx = torch.zeros(4, 8, dtype=torch.long)
                w = torch.zeros(4, 8)
                return self.experts(h, idx, w)

        return Blk()

    def test_routed_container_records_marker_with_topk_and_bytes(self):
        import torch

        from auto_round.utils.stream_placement import FlowProbe

        got = []
        blk = self._probe_block()
        probe = FlowProbe(blk, lambda records: got.extend(records))
        blk(torch.randn(4, 16))
        routed = [r for r in got if str(r[0]).startswith("__routed__:")]
        assert routed, f"no routed marker in {got}"
        name, top_k, hidden_bytes = routed[0]
        assert name == "__routed__:experts"
        assert top_k == 8
        assert hidden_bytes == 4 * 16 * 4  # fp32 default dtype numel x elemsize

    def test_plain_container_records_no_marker(self):
        import torch
        import torch.nn as nn

        from auto_round.utils.stream_placement import FlowProbe

        class Plain(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(8, 8)

            def forward(self, x):
                return self.inner(x)

        got = []
        blk = Plain()
        FlowProbe(blk, lambda records: got.extend(records))
        blk(torch.randn(4, 8))
        assert not [r for r in got if str(r[0]).startswith("__routed__:")]
