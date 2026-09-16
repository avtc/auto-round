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
    "_begin_predictor_capture_",
    "_finish_predictor_capture_",
    "_begin_predictor_q_capture_",
    "_pins_for_tree_",
    "_stamp_pin_",
    "_tune_predictor_trees_",
)


def _orch(model, tmp_path, layer_config=None, iters=10, composer=None):
    o = SimpleNamespace()
    for name in _ORCH_METHODS:
        setattr(o, name, getattr(CompressionOrchestrator, name).__get__(o))
    if not hasattr(model, "config"):
        model.config = SimpleNamespace(name_or_path=str(tmp_path), hidden_size=HID, text_config=None)
    o.model_context = SimpleNamespace(model=model)
    o.layer_config = layer_config if layer_config is not None else {}
    o.alg_composer = composer or SimpleNamespace(block_quantizer=[SimpleNamespace(iters=iters)])
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
    def test_plan_and_hook_when_pinned(self, tmp_path):
        _write_ckpt(tmp_path)
        model = _Body()
        o = _orch(model, tmp_path, layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}})
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is not None
        assert o._predictor_plan_["norm"] == "norm"
        assert "mtp" in o._predictor_plan_["roots"]
        # forward through the model captures the fp tail
        model(torch.randint(0, 16, (2, 5)))
        o._finish_predictor_capture_("fp")
        assert o._predictor_fp_tail_ is not None and len(o._predictor_fp_tail_) == 2
        assert o._predictor_fp_tail_[0].shape == (1, 5, HID)
        # hook removed after finish
        handles = getattr(model.norm, "_forward_pre_hooks", {})
        assert len(handles) == 0

    def test_noop_without_local_dir(self, tmp_path):
        o = _orch(_Body(), tmp_path)
        o.model_context.model.config.name_or_path = "hf-org/model-id"
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is None

    def test_noop_at_zero_iters(self, tmp_path):
        _write_ckpt(tmp_path)
        o = _orch(_Body(), tmp_path, layer_config={"mtp.*": {"bits": 4, "group_size": -1, "sym": True}}, iters=0)
        o._prepare_predictor_tuning_(ALL_BLOCKS)
        assert o._predictor_plan_ is None


class TestTunePredictorTrees:
    def test_full_stage_with_stub_composer(self, tmp_path, monkeypatch):
        _write_ckpt(tmp_path)
        model = _Body()
        model(torch.randint(0, 16, (2, 5)))  # populate any caches nothing depends on here

        calls = {"block": [], "outside": []}

        class _Composer:
            block_quantizer = [SimpleNamespace(iters=10)]

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
        model(torch.randint(0, 16, (2, 5)))
        o._finish_predictor_capture_("fp")
        token_ids = [torch.randint(0, 16, (1, 5)) for _ in range(2)]
        o._predictor_tree_aux_ = {"attention_mask": [torch.ones(1, 5) for _ in range(2)]}
        o._tune_predictor_trees_(token_ids)

        assert len(calls["block"]) == 1
        b = calls["block"][0]
        assert b["fp"] == 2 and b["e_rows"] == 2 and b["q"] is True
        assert b["out_shape"] == (5, HID)
        assert "attention_mask" in b["io_keys"]
        # the vocab head tuned layer-wise on the tree outputs
        assert [c[0] for c in calls["outside"]] == ["mtp.fc"]
        # pins stamped into layer_config under checkpoint names
        assert "mtp.eh_proj" in o.layer_config
        assert "mtp.fc" in o.layer_config
        assert getattr(model.get_submodule("mtp.eh_proj"), "bits", None) == 4
