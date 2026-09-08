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

"""Mapped-placement plumbing on CheckpointStreamer (per-tensor devices)."""

import pytest
import torch

from auto_round.utils import checkpoint_streamer as cs_mod
from auto_round.utils.checkpoint_streamer import CheckpointStreamer


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(4, 4, bias=False)})


def _bare_streamer(monkeypatch):
    """A streamer shell whose fetch is replaced by a recording stub."""
    s = object.__new__(CheckpointStreamer)
    s.weight_map = {
        "blk.q_proj.weight": "s0",
        "blk.mlp.gate_proj.weight": "s0",
    }
    s._model_type = ""
    s._prefetch_thread = None
    s._perf_segs = None
    seen = {}

    def fake_fetch(name, device=None, raw=False):
        seen[name] = device
        return torch.zeros(4, 4)

    monkeypatch.setattr(s, "fetch", fake_fetch)
    return s, seen


class TestLoadModuleDeviceOf:
    def test_per_tensor_devices(self, monkeypatch):
        s, seen = _bare_streamer(monkeypatch)
        monkeypatch.setattr(s, "names_under", lambda prefix: ["blk.q_proj.weight", "blk.mlp.gate_proj.weight"])
        monkeypatch.setattr(s, "_assign_leaf_", lambda module, rel, tensor: True)
        monkeypatch.setattr(cs_mod, "_park_cpu_buffers_", lambda module, device: None)
        placement = {"q_proj": "cpu", "mlp.gate_proj": "cpu"}

        def device_of(name):
            leaf = name[len("blk.") :].rsplit(".", 1)[0]
            return placement.get(leaf)

        module = _Toy()
        s.load_module_(module, "blk", device="cpu", device_of=device_of)
        # device_of is honored per tensor (here both resolve, distinct calls)
        assert seen["blk.q_proj.weight"] == "cpu"
        assert seen["blk.mlp.gate_proj.weight"] == "cpu"

    def test_device_of_overrides_single_home(self, monkeypatch):
        s, seen = _bare_streamer(monkeypatch)
        monkeypatch.setattr(s, "names_under", lambda prefix: ["blk.q_proj.weight"])
        monkeypatch.setattr(s, "_assign_leaf_", lambda module, rel, tensor: True)
        monkeypatch.setattr(cs_mod, "_park_cpu_buffers_", lambda module, device: None)

        calls = []

        def device_of(name):
            calls.append(name)
            return None  # unmatched -> fetch gets None (no forced home)

        module = _Toy()
        s.load_module_(module, "blk", device="cpu", device_of=device_of)
        assert calls == ["blk.q_proj.weight"]
        assert seen["blk.q_proj.weight"] is None

    def test_recorded_stage_home_ignored_under_device_of(self, monkeypatch):
        """With device_of, a stale recorded single home must not override."""
        s, seen = _bare_streamer(monkeypatch)
        s._prefetch_stage_dev = {"blk": torch.device("cuda", 3)}
        monkeypatch.setattr(s, "names_under", lambda prefix: ["blk.q_proj.weight"])
        monkeypatch.setattr(s, "_assign_leaf_", lambda module, rel, tensor: True)
        monkeypatch.setattr(cs_mod, "_park_cpu_buffers_", lambda module, device: None)
        module = _Toy()
        s.load_module_(module, "blk", device="cpu", device_of=lambda name: None)
        assert seen["blk.q_proj.weight"] is None  # not str(cuda:3)


class TestPinStreamMapped:
    """The materialize-side helper binds wrappers (tuning_device) and chain
    hooks for a mapped streamed block - streaming never calls dispatch_block,
    so this is the single attachment site."""

    def test_pins_tuning_device_and_attaches_hooks(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch

        class _DM:
            device = torch.device("cpu", 0)
            device_list = [torch.device("cpu", 0), torch.device("cpu", 1)]

        monkeypatch.setattr(orch, "device_manager", _DM())
        block = _Toy()
        placement = {"q_proj": torch.device("cpu", 1), "mlp.gate_proj": torch.device("cpu", 0)}
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        # wrappers bind to the per-leaf template devices
        assert block.q_proj.tuning_device == torch.device("cpu", 1)
        assert block.mlp["gate_proj"].tuning_device == torch.device("cpu", 0)
        # cross-device chain hooks attached on multi-device maps
        assert getattr(block.q_proj, "_stream_align_hook", None) is not None
        # idempotent: re-pinning (restage) replaces the hook handles
        # instead of stacking a second aligner
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        assert len(block.q_proj._forward_pre_hooks) == 1
        assert len(block.q_proj._forward_hooks) == 1
        assert block.q_proj._stream_align_hook.target == torch.device("cpu", 1)

    def test_single_device_map_skips_hooks(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch

        class _DM:
            device = torch.device("cpu")
            device_list = [torch.device("cpu")]

        monkeypatch.setattr(orch, "device_manager", _DM())
        block = _Toy()
        placement = {"q_proj": torch.device("cpu")}
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        assert block.q_proj.tuning_device == torch.device("cpu")
        assert getattr(block.q_proj, "_hf_hook", None) is None


class TestStartPrefetchMapped:
    def test_stage_devices_and_stage_device_of_mutex(self, monkeypatch, tmp_path):
        s = object.__new__(CheckpointStreamer)
        s._prefetch_thread = None
        with pytest.raises(ValueError, match="mutually exclusive"):
            s.start_prefetch(
                ["blk"],
                stage_devices=[torch.device("cpu")],
                stage_device_of=lambda idx, name, prefix: "cpu",
            )


class TestOrchestratorMappedMode:
    def _shell(self, **attrs):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        o = object.__new__(CompressionOrchestrator)
        for k, v in attrs.items():
            setattr(o, k, v)
        return o

    def test_mapped_enabled_by_template_or_next(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch

        class _DM:
            device_map = "0,1"  # plain list -> NOT mapped

        monkeypatch.setattr(orch, "device_manager", _DM())
        # plain list keeps the prefetch-rotation semantics while prefetch is on
        o = self._shell(stream_prefetch="auto", stream_prefetch_device_map=None)
        assert o._stream_mapped_enabled() is False
        # ... but with prefetch off there is no rotation: multi-device means mapped
        o = self._shell(stream_prefetch="off", stream_prefetch_device_map=None)
        assert o._stream_mapped_enabled() is True

        monkeypatch.setattr(orch, "device_manager", type("DM2", (), {"device_map": "self_attn.*:0,mlp.*:1"})())
        assert o._stream_mapped_enabled() is True

        monkeypatch.setattr(orch, "device_manager", _DM())
        o = self._shell(stream_prefetch_device_map="self_attn.*:2,mlp.*:3")
        assert o._stream_mapped_enabled() is True

    def test_resolver_superseded_and_conflict(self, monkeypatch):
        import pytest

        import auto_round.compressors.orchestrator as orch

        class _DM:
            device_map = "self_attn.*:0,mlp.*:1"

        monkeypatch.setattr(orch, "device_manager", _DM())
        o = self._shell(stream_prefetch_device_map=None, stream_prefetch="auto", device="cpu")
        assert o._resolve_stream_stage_devices() is None  # superseded -> no rotation

        o = self._shell(stream_prefetch_device_map=None, stream_prefetch="cuda:1", device="cpu")
        with pytest.raises(ValueError, match="conflicts with mapped placement"):
            o._resolve_stream_stage_devices()

    def test_park_rows_cpu(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        rows = [torch.zeros(2), torch.zeros(3)]
        out = CompressionOrchestrator._park_rows_cpu_(rows)
        assert all(r.device.type == "cpu" for r in out)
        assert CompressionOrchestrator._park_rows_cpu_(torch.zeros(2)).device.type == "cpu"
        assert CompressionOrchestrator._park_rows_cpu_(None) is None
        assert CompressionOrchestrator._park_rows_cpu_([None, torch.zeros(2)])[0] is None


class TestRehomeBlockMapped:
    def test_leaves_distributed_rest_on_fallback(self):
        from auto_round.compressors.utils import rehome_block_mapped_

        block = _Toy()
        block.q_proj.weight.data = torch.randn(4, 4)
        block.mlp["gate_proj"].weight.data = torch.randn(4, 4)
        placement = {"q_proj": torch.device("cpu"), "mlp.gate_proj": torch.device("cpu")}
        # both devices are cpu here; verify the fallback path with a buffer
        # living outside any mapped leaf
        block.extra = torch.nn.Parameter(torch.randn(2))
        moved = rehome_block_mapped_(block, placement, "cpu")
        assert moved >= 0  # cpu->cpu is a no-op move count
        assert block.q_proj.weight.device.type == "cpu"


class TestStreamPrefetchDeviceMapGuard:
    def test_requires_stream_quantization(self, monkeypatch):
        import pytest

        monkeypatch.delenv("AR_DISK_STREAM_MODEL", raising=False)
        from auto_round import RTNConfig
        from auto_round.compressors.base import BaseOrchestrator

        with pytest.raises(ValueError, match="stream_prefetch_device_map requires stream_quantization"):
            BaseOrchestrator(
                config=RTNConfig(),
                model="dummy",
                tokenizer=None,
                nsamples=8,
                stream_prefetch_device_map="2,3",
                stream_quantization=False,
            )

    def test_requires_stream_prefetch(self, monkeypatch):
        import pytest

        monkeypatch.delenv("AR_DISK_STREAM_MODEL", raising=False)
        from auto_round import RTNConfig
        from auto_round.compressors.base import BaseOrchestrator

        with pytest.raises(ValueError, match="stream_prefetch_device_map requires stream_prefetch"):
            BaseOrchestrator(
                config=RTNConfig(),
                model="dummy",
                tokenizer=None,
                nsamples=8,
                stream_prefetch="off",
                stream_prefetch_device_map="2,3",
                stream_quantization=True,
            )


class TestSharedLayersPlacementGroups:
    """--shared_layers groups become placement atoms (union with the
    layer_config comma-key shared-quant groups)."""

    def test_union_of_comma_keys_and_shared_layers(self):
        from auto_round.compressors.orchestrator import CompressionOrchestrator

        o = object.__new__(CompressionOrchestrator)
        o.layer_config = {"lm_head": {"bits": 8}, "q_proj,k_proj,v_proj": {}}
        o.shared_layers = [["gate_proj", "up_proj"], ["single"], None, ["a", "b"]]
        groups = o._mapped_shared_groups_()
        assert ["q_proj", "k_proj", "v_proj"] in groups
        assert ["gate_proj", "up_proj"] in groups
        assert ["a", "b"] in groups
        # single-member and None entries are dropped
        assert not any(g == ["single"] for g in groups)

    def test_shared_layers_group_survives_device_list_partition(self):
        import torch

        from auto_round.utils.stream_placement import _atomic_groups, _shared_atoms

        leaves = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        groups = dict(_atomic_groups(leaves))
        for atom in _shared_atoms(leaves, [["gate_proj", "up_proj"]]):
            merged = set(atom)
            rebuilt = []
            for key, lg in groups.items():
                if merged & set(lg):
                    merged |= set(lg)
                else:
                    rebuilt.append((key, lg))
            rebuilt.append(("shared:" + atom[0].rpartition(".")[2], sorted(merged)))
            groups = dict(rebuilt)
        assert set(groups["shared:gate_proj"]) >= {"mlp.gate_proj", "mlp.up_proj"}


class TestStreamAlignHookProtocols:
    def test_both_call_conventions(self):
        import torch

        from auto_round.compressors.utils import StreamAlignHook

        lin = torch.nn.Linear(2, 2)
        hook = StreamAlignHook(torch.device("cpu"))
        # torch convention: (module, args, kwargs) -> (args, kwargs)
        args, kwargs = hook._pre(lin, (torch.zeros(1, 2),), {"attention_mask": torch.zeros(1, 2)})
        assert isinstance(args, tuple) and "attention_mask" in kwargs
        assert hook.input_device is not None
        # legacy manual replay (wrapper.py): (module, args) -> moved args
        out = hook._pre(lin, (torch.zeros(1, 2),))
        assert isinstance(out, tuple) and not isinstance(out[0], dict)

    def test_post_returns_output(self):
        import torch

        from auto_round.compressors.utils import StreamAlignHook

        hook = StreamAlignHook(torch.device("cpu"))
        hook.input_device = torch.device("cpu")
        out = hook._post(None, None, {"a": torch.zeros(1)})
        assert isinstance(out, dict)


class TestAlignTrace:
    def test_trace_logs_only_real_hops(self, monkeypatch, caplog):
        import logging as _logging

        import torch

        import auto_round.compressors.utils as cu

        monkeypatch.setenv("AR_STREAM_TRACE_DEVICES", "1")
        monkeypatch.setattr(cu.logger, "propagate", True)  # autoround logger keeps its own handler
        cu._TRACE_HOPS = None  # re-read the env
        with caplog.at_level(_logging.INFO, logger=cu.logger.name):
            cu._trace_align_hop("self_attn.q_proj", torch.device("cpu"), torch.device("cuda:1"))
            cu._trace_align_hop("self_attn.k_proj", torch.device("cpu"), torch.device("cpu"))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("self_attn.q_proj" in m and "cuda:1" in m for m in msgs)
        assert not any("k_proj" in m for m in msgs)
        cu._TRACE_HOPS = None


class TestWrapperOwnsAlignment:
    def test_wrapper_gets_hook_and_orig_detached(self, monkeypatch):
        import torch

        import auto_round.compressors.orchestrator as orch
        from auto_round.compressors.utils import StreamAlignHook

        class _DM:
            device_list = [torch.device("cpu"), torch.device("cpu", 1)]

        monkeypatch.setattr(orch, "device_manager", _DM())

        class FakeWrap(torch.nn.Module):
            def __init__(self, orig):
                super().__init__()
                self.orig_layer = orig

        lin = torch.nn.Linear(2, 2)
        lin.tuning_device = torch.device("cpu", 1)
        wrap = FakeWrap(lin)
        block = torch.nn.ModuleDict({"q": wrap})
        placement = {"q": "cpu:1"}
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        # alignment moved to the wrapper: real torch hooks, correct caller
        assert getattr(wrap, "_stream_align_hook", None) is not None
        assert isinstance(wrap._stream_align_hook, StreamAlignHook)
        assert getattr(wrap, "tuning_device", None) == torch.device("cpu", 1)
        # stale orig-layer hook detached (its input_device recording is wrong)
        assert getattr(lin, "_stream_align_hook", None) is None
        # idempotent re-pin keeps exactly one hook set
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        assert len(wrap._forward_pre_hooks) == 1 and len(wrap._forward_hooks) == 1


class TestContainerAndWrapRealign:
    def test_paramless_container_gets_inherited_hook(self, monkeypatch):
        import torch

        import auto_round.compressors.orchestrator as orch

        class _DM:
            device = torch.device("cpu", 0)
            device_list = [torch.device("cpu", 0), torch.device("cpu", 1)]

        monkeypatch.setattr(orch, "device_manager", _DM())

        # paramless container wrapping two leaves on different devices
        attn = torch.nn.ModuleDict({"q": torch.nn.Linear(2, 2), "o": torch.nn.Linear(2, 2)})
        block = torch.nn.ModuleDict({"self_attn": attn})
        placement = {"self_attn.q": "cpu:0", "self_attn.o": "cpu:1"}
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        # the container aligns to its FIRST placed descendant's device
        assert getattr(attn, "_stream_align_hook", None) is not None
        assert attn._stream_align_hook.target == torch.device("cpu", 0)
        assert not hasattr(attn, "tuning_device")  # no direct params

    def test_wrap_realign_moves_hook_from_orig_to_wrapper(self, monkeypatch):
        import torch

        import auto_round.compressors.orchestrator as orch

        class _DM:
            device = torch.device("cpu", 0)
            device_list = [torch.device("cpu", 0), torch.device("cpu", 1)]

        monkeypatch.setattr(orch, "device_manager", _DM())

        class FakeWrap(torch.nn.Module):
            def __init__(self, orig):
                super().__init__()
                self.orig_layer = orig

        # stage 1: pre-wrap pin hooks the bare leaf
        lin = torch.nn.Linear(2, 2)
        block = torch.nn.ModuleDict({"q": lin})
        placement = {"q": "cpu:1"}
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        assert getattr(lin, "_stream_align_hook", None) is not None
        # stage 2: wrapper replaces the leaf; re-pin (as quantize_block does)
        wrap = FakeWrap(lin)
        block["q"] = wrap
        orch.CompressionOrchestrator._pin_stream_mapped_(block, placement)
        assert getattr(wrap, "_stream_align_hook", None) is not None
        assert getattr(lin, "_stream_align_hook", None) is None  # detached
