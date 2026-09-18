# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the predictor-tree orchestrator wiring (fake model + stub composer)."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from auto_round.compressors.orchestrator import CompressionOrchestrator
from auto_round.compressors.predictor_tree import list_checkpoint_tensors

HID = 8


class _Norm(nn.Module):
    def __init__(self, size=HID, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return x * self.weight


class _DecLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = _Norm()
        self.fc1 = nn.Linear(HID, HID, bias=False)
        self.fc2 = nn.Linear(HID, HID, bias=False)

    def forward(self, x, **kwargs):
        return self.fc2(torch.nn.functional.relu(self.fc1(self.norm1(x))))


class _MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enorm = _Norm()
        self.hnorm = _Norm()
        self.eh_proj = nn.Linear(2 * HID, HID, bias=False)
        self.layer = _DecLayer()
        self.final_norm = _Norm()


class _Body(nn.Module):
    """Decoder blocks + final norm + lm_head; forward feeds the norm (hook target)."""

    def __init__(self, vocab=16):
        super().__init__()
        self.blocks = nn.ModuleList([_DecLayer() for _ in range(2)])
        self.norm = _Norm()
        self.lm_head = nn.Linear(HID, vocab, bias=False)
        self.mtp = None
        self.embed = nn.Embedding(vocab, HID)

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))

    def get_input_embeddings(self):
        return self.embed


def _write_ckpt(tmp_path):
    mtp = _MTP()
    fc = nn.Linear(HID, 16, bias=False)  # vocab head pinned like mtp.fc
    tensors = {f"mtp.{n}": p.detach().clone() for n, p in mtp.named_parameters()}
    tensors["mtp.fc.weight"] = fc.weight.detach().clone()
    from safetensors.torch import save_file

    save_file(tensors, os.path.join(tmp_path, "model.safetensors"))
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_map": {n: "model.safetensors" for n in tensors}}, f)
    return mtp, fc


_ORCH_METHODS = (
    "_discover_final_norm_",
    "_prepare_predictor_tuning_",
    "_pins_for_tree_",
    "_schema_pins_for_tree_",
    "_stamp_pin_",
    "_attach_pinned_tree_",
    "_detach_tree_",
    "_tune_tree_heads_",
    "_chain_hidden_rows",
    "_snapshot_predictor_aux_",
    "_tune_predictor_trees_",
    "_tune_predictor_trees_impl_",
    "_tune_one_predictor_tree_",
)


def _orch(model, tmp_path, layer_config=None, iters=10, composer=None):
    o = SimpleNamespace()
    for name in _ORCH_METHODS:
        if name == "_chain_hidden_rows":  # staticmethod: bind directly, no __get__
            setattr(o, name, getattr(CompressionOrchestrator, name))
        else:
            setattr(o, name, getattr(CompressionOrchestrator, name).__get__(o))
    if not hasattr(model, "config"):
        model.config = SimpleNamespace(name_or_path=str(tmp_path), hidden_size=HID, text_config=None)
    o.model_context = SimpleNamespace(model=model, amp_dtype=torch.float32)
    o.layer_config = layer_config if layer_config is not None else {}
    o.alg_composer = composer or SimpleNamespace(block_quantizer=[SimpleNamespace(iters=iters)])
    o.formats = []
    o.calibration_context = SimpleNamespace(batch_size=2, nsamples=2)
    o.compress_context = SimpleNamespace(is_immediate_packing=False)
    o._preprocess_block_inputs = lambda inputs: (
        inputs.get("input_ids"),
        {k: v for k, v in inputs.items() if k != "input_ids"},
    )
    return o


ALL_BLOCKS = [["blocks.0"], ["blocks.1"]]


class TestDiscoverFinalNorm:
    def test_finds_the_block_external_norm(self, tmp_path):
        o = _orch(_Body(), tmp_path)
        assert o._discover_final_norm_(ALL_BLOCKS) == "norm"

    def test_none_when_norm_inside_blocks(self, tmp_path):
        o = _orch(_Body(), tmp_path)
        assert o._discover_final_norm_([["blocks.0"], ["blocks.1"], ["norm"]]) is None


class TestPinsForTree:
    def test_regex_and_exact(self, tmp_path):
        o = _orch(_Body(), tmp_path, layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}})
        pins = o._pins_for_tree_(["mtp.layer.fc1.weight", "mtp.fc.weight"])
        assert "mtp.*" in pins


class TestPreparePredictor:
    def test_plan_without_pins_or_hooks(self, tmp_path):
        _write_ckpt(tmp_path)
        model = _Body()
        o = _orch(model, tmp_path)  # no pins: the tree follows the schema now
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is not None
        assert o._predictor_plan_["norm"] == "norm"
        assert "mtp" in o._predictor_plan_["roots"]
        # no capture hook is installed anywhere: tails come from the chain
        assert getattr(model.norm, "_forward_pre_hooks", {}) == {}
        assert getattr(model, "_forward_pre_hooks", {}) == {}

    def test_noop_without_local_dir(self, tmp_path):
        o = _orch(_Body(), tmp_path)
        o.model_context.model.config.name_or_path = "hf-org/model-id"
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is None

    def test_plan_at_zero_iters_too(self, tmp_path):
        # iters=0 aligns the tree with the run's search path instead of the
        # export-time WOQ round-trip
        _write_ckpt(tmp_path)
        o = _orch(_Body(), tmp_path, iters=0)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is not None

    def test_plan_for_gguf_formats_too(self, tmp_path):
        # tuned MTP under gguf is the proven campaign configuration (the GGUF
        # twins carry quantized nextn); the tree tunes regardless of format
        _write_ckpt(tmp_path)
        o = _orch(_Body(), tmp_path)
        o.formats = [SimpleNamespace(is_gguf=lambda: True)]
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is not None


class TestTunePredictorTrees:
    def test_full_stage_with_stub_composer(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()
        model(torch.randint(0, 16, (2, 5)))  # populate any caches nothing depends on here

        calls = {"block": [], "outside": []}

        class _Composer:
            block_quantizer = [SimpleNamespace(iters=10)]

            def dispatch_block(self, block, input_ids, input_others):
                return block

            def compress_block(self, shell, fp, io, block_ctx=None, q_inputs=None, input_ids=None):
                # run the tree forward once so the test exercises the bound predictor
                out = shell(fp[0], _predictor_e=io["_predictor_e"][0])
                calls["block"].append(
                    {
                        "fp": len(fp),
                        "io_keys": sorted(k for k in io if k != "_predictor_e"),
                        "e_rows": len(io["_predictor_e"]),
                        "q": q_inputs is None,
                        "out_shape": tuple(out.shape[-2:]),
                    }
                )
                return fp, fp  # (new_q_output, reference_output)

            def compress_layer_outside_block(self, mod, fp_inputs=None, q_inputs=None, input_ids=None):
                calls["outside"].append((mod.global_name, tuple(fp_inputs[0].shape)))

        o = _orch(
            model,
            tmp_path,
            layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}},
            composer=_Composer(),
        )
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        # the block loop's stored chain tail: (q rows, fp rows), raw pre-norm
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        token_ids = [torch.randint(0, 16, (1, 5)) for _ in range(2)]
        o._predictor_tree_aux_ = {"attention_mask": [torch.ones(1, 5) for _ in range(2)]}
        o._tune_predictor_trees_(token_ids)

        assert len(calls["block"]) == 1
        b = calls["block"][0]
        assert b["fp"] == 2 and b["e_rows"] == 2 and b["q"] is False  # chain tail carries q rows
        assert b["out_shape"] == (5, HID)
        assert "attention_mask" in b["io_keys"]
        # the vocab head tuned layer-wise on the tree outputs
        assert [c[0] for c in calls["outside"]] == ["mtp.fc"]
        # pins stamped into layer_config under checkpoint names
        assert "mtp.eh_proj" in o.layer_config
        assert "mtp.fc" in o.layer_config
        assert getattr(model.get_submodule("mtp.eh_proj"), "bits", None) == 4
        # the raw chain tail is consumed and released by the stage
        assert o._lm_head_chain_tail_ is None


class TestTreeFallbacks:
    def test_no_pins_tunes_via_schema(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()
        calls = {"block": [], "outside": []}

        class _Composer:
            block_quantizer = [
                SimpleNamespace(iters=10, config=SimpleNamespace(bits=4, group_size=-1, sym=True, data_type="int"))
            ]

            def dispatch_block(self, block, input_ids, input_others):
                return block

            def compress_block(self, shell, fp, io, **k):
                calls["block"].append(1)
                return fp, fp  # (new_q_output, reference_output)

            def compress_layer_outside_block(self, *a, **k):
                calls["outside"].append(1)

        o = _orch(
            model, tmp_path, layer_config={"model.*": {"bits": 4, "group_size": -1, "sym": True}}, composer=_Composer()
        )
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        o._predictor_tree_aux_ = {}
        o._tune_predictor_trees_([torch.randint(0, 16, (1, 5)) for _ in range(2)])
        # the tree joined as an extra block with SCHEMA pins (no explicit pins)
        assert len(calls["block"]) == 1
        assert getattr(model.get_submodule("mtp.eh_proj"), "bits", None) == 4

    def test_build_failure_degrades_to_export_path(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()
        calls = {"block": []}

        class _Composer:
            block_quantizer = [
                SimpleNamespace(iters=10, config=SimpleNamespace(bits=4, group_size=-1, sym=True, data_type="int"))
            ]

            def dispatch_block(self, block, input_ids, input_others):
                return block

            def compress_block(self, *a, **k):
                calls["block"].append(1)

            def compress_layer_outside_block(self, *a, **k):
                pass

        o = _orch(
            model,
            tmp_path,
            layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}},
            composer=_Composer(),
        )
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)

        import auto_round.compressors.orchestrator as orch_mod

        def _boom(model, source_dir, ckpt, info, sibling):
            raise RuntimeError("synthetic build failure")

        monkeypatch.setattr(orch_mod, "build_predictor_tree", _boom)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        o._predictor_tree_aux_ = getattr(o, "_predictor_tree_aux_", {})
        o._tune_predictor_trees_([torch.randint(0, 16, (1, 5)) for _ in range(2)])
        assert calls["block"] == []


class TestImmediatePackingPath:
    def test_tree_modules_packed_when_immediate(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()
        packed = []

        class _Composer:
            block_quantizer = [SimpleNamespace(iters=10)]

            def dispatch_block(self, block, input_ids, input_others):
                return block

            def compress_block(self, shell, fp, io, block_ctx=None, q_inputs=None, input_ids=None):
                return fp, fp

            def compress_layer_outside_block(self, *a, **k):
                pass

        o = _orch(
            model,
            tmp_path,
            layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}},
            composer=_Composer(),
        )
        o.compress_context.is_immediate_packing = True
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)

        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "immediate_pack", lambda name, cfg: packed.append(name))
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        o._predictor_tree_aux_ = getattr(o, "_predictor_tree_aux_", {})
        o._tune_predictor_trees_([torch.randint(0, 16, (1, 5)) for _ in range(2)])
        assert "mtp.fc" in packed  # head packed inside _tune_tree_heads_
        assert any(name.startswith("mtp.") for name in packed)  # tree Linears packed


class TestCacheOverrideHelpers:
    def test_no_early_stop_override_left(self):
        # the sentinel override is retired: predictor tails come from the
        # stored chain tail, so the collection walk keeps its fast path
        src = (
            open(CompressionOrchestrator.__module__.replace(".", "/") + ".py", encoding="utf-8").read()
            if False
            else open("auto_round/compressors/orchestrator.py", encoding="utf-8").read()
        )
        assert "_predictor_overrides_last_cache_" not in src

    def test_aux_snapshot_excludes_both_row_spellings(self, tmp_path):
        o = _orch(_Body(), tmp_path)
        rows = [torch.randn(1, 5, HID) for _ in range(2)]
        masks = [torch.ones(1, 5) for _ in range(2)]
        # the cache entry before the rename: primary rows under "hidden_states"
        entry = {"hidden_states": rows, "attention_mask": masks}
        o._snapshot_predictor_aux_({"blocks.1": entry}, ["blocks.0", "blocks.1"])
        assert set(o._predictor_tree_aux_.keys()) == {"attention_mask"}
        assert o._predictor_tree_aux_["attention_mask"] is masks
        # after the rename the same exclusion must hold
        entry2 = {"input_ids": rows, "attention_mask": masks}
        o._snapshot_predictor_aux_({"blocks.1": entry2}, ["blocks.0", "blocks.1"])
        assert set(o._predictor_tree_aux_.keys()) == {"attention_mask"}


class TestTreeTuneContainment:
    def test_tune_failure_leaves_export_path_and_releases(self, tmp_path, monkeypatch):
        """A crash anywhere in the tree tune (here: materialize) must degrade
        loudly to the export path and release the tree - the run continues
        after every block is already tuned and streamed."""
        _write_ckpt(tmp_path)
        model = _Body()
        calls = {"detach": 0}
        monkeypatch.setattr(
            CompressionOrchestrator,
            "_detach_tree_",
            lambda self, group: calls.__setitem__("detach", calls["detach"] + 1),
        )
        import auto_round.compressors.orchestrator as orch_mod

        def _boom(*a, **k):
            raise RuntimeError("synthetic tune failure")

        monkeypatch.setattr(orch_mod, "materialize_model_", _boom)
        o = _orch(
            model,
            tmp_path,
            layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}},
            composer=SimpleNamespace(
                block_quantizer=[SimpleNamespace(iters=10)],
                dispatch_block=lambda block, input_ids, input_others: block,
                compress_block=lambda *a, **k: None,
            ),
        )
        o._detach_tree_ = lambda group: calls.__setitem__("detach", calls["detach"] + 1)
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        o._predictor_tree_aux_ = {}
        o._tune_predictor_trees_([torch.randint(0, 16, (1, 5)) for _ in range(2)])
        assert calls["detach"] == 1  # tree released for the export pass
        assert o._lm_head_chain_tail_ is None  # tail released in the finally


class TestLazyRefs:
    def test_refs_resolve_live_through_wrappers(self, tmp_path):
        from auto_round.compressors.predictor_tree import bind_predictor_forward

        _write_ckpt(tmp_path)
        model = _Body()
        o = _orch(model, tmp_path, layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}})
        o._prepare_predictor_tuning_(ALL_BLOCKS)

        import auto_round.compressors.predictor_tree as pt

        ckpt = pt.list_checkpoint_tensors(str(tmp_path))
        info = pt.analyze_predictor_group(ckpt, "mtp", HID)
        _, sibling = pt.pick_sibling_layer(model, ckpt, info, ALL_BLOCKS)
        pt.build_predictor_tree(model, str(tmp_path), ckpt, info, sibling)
        shell = model.get_submodule("mtp")
        # PATH refs + model: resolve live at forward time
        bind_predictor_forward(
            shell,
            {
                "norm_e": info["norm_e"][: -len(".weight")],
                "norm_h": info["norm_h"][: -len(".weight")],
                "fc": info["fc"][: -len(".weight")],
                "layer": info["layer_root"],
                "final_norm": info["final_norm"][: -len(".weight")],
            },
            model=model,
        )
        h = torch.randn(1, 5, HID)
        e = torch.randn(1, 5, HID)
        out1 = shell(h, _predictor_e=e)

        # replace the mixer with a wrapper whose forward is visibly different;
        # the tree forward must pick the REPLACEMENT up (lazy resolution)
        doubled = torch.nn.Linear(2 * HID, HID, bias=False)
        with torch.no_grad():
            doubled.weight.copy_(model.get_submodule("mtp.eh_proj").weight * 2.0)
        model.get_submodule("mtp").eh_proj = doubled
        out2 = shell(h, _predictor_e=e)
        assert not torch.allclose(out1, out2), "lazy refs must resolve the wrapped module live"


class TestRegexConfigPins:
    def test_pins_from_regex_config_activate_the_plan(self, tmp_path):
        """The resolver parks checkpoint-only pins in regex_config, not layer_config."""
        _write_ckpt(tmp_path)
        model = _Body()
        # what the resolver actually produces for safetensor-only targets
        o = _orch(model, tmp_path, layer_config={})
        o.regex_config = {"mtp.*": {"bits": 4, "group_size": -1, "sym": True}}
        o._ORCH_BIND = None
        from types import MethodType as _MT

        from auto_round.compressors.orchestrator import CompressionOrchestrator as _CO

        # rebind _pins_for_tree_ and _prepare_predictor_tuning_ against this instance
        o._pins_for_tree_ = _MT(_CO._pins_for_tree_, o)
        o._prepare_predictor_tuning_ = _MT(_CO._prepare_predictor_tuning_, o)
        o._predictor_plan_ = None
        o._predictor_fp_tail_ = None
        o._predictor_q_tail_ = None
        o._predictor_tree_aux_ = None
        o._predictor_hook_ = None
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is not None, "regex_config pins must activate the predictor plan"


class TestDetachOnNoPinnedLinear:
    def test_tree_detached_when_pins_match_nothing(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()

        class _Composer:
            block_quantizer = [SimpleNamespace(iters=10)]

            def dispatch_block(self, block, input_ids, input_others):
                return block

            def compress_block(self, *a, **k):
                raise AssertionError("must not tune when no Linear matched")

            def compress_layer_outside_block(self, *a, **k):
                pass

        o = _orch(
            model,
            tmp_path,
            # pins match tensors (activates the plan) but no tree Linear is
            # named fc_only - the group must detach and the run must survive
            layer_config={"mtp.fc.weight": {"bits": 4, "group_size": -1, "sym": True}},
            composer=_Composer(),
        )
        monkeypatch.setattr("auto_round.compressors.orchestrator.get_block_names", lambda m: ALL_BLOCKS)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        _rows = lambda: [torch.randn(1, 5, HID) for _ in range(2)]  # noqa: E731
        o._lm_head_chain_tail_ = (_rows(), _rows())
        o._predictor_tree_aux_ = getattr(o, "_predictor_tree_aux_", {})
        o._tune_predictor_trees_([torch.randint(0, 16, (1, 5)) for _ in range(2)])
        assert "mtp" not in [n for n, _ in model.named_modules()], "tree must detach when no Linear matched"
