# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for predictor-tree (MTP) materialization (tiny fakes, no model downloads)."""

import json
import os

import pytest
import torch
import torch.nn as nn

from auto_round.compressors.predictor_tree import (
    analyze_predictor_group,
    bind_predictor_forward,
    build_predictor_tree,
    checkpoint_only_roots,
    ensure_module_path,
    list_checkpoint_tensors,
    pick_sibling_layer,
    predictor_forward,
    synthesize_predictor_e,
)

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
    """Two decoder blocks + final norm; blocks are siblings for the tree layer."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_DecLayer() for _ in range(2)])
        self.norm_f = _Norm()
        self.mtp = None  # attached later by the tree builder


def _write_ckpt(tmp_path, mtp: _MTP, prefix="mtp"):
    tensors = {f"{prefix}.{n}": p.detach().clone() for n, p in mtp.named_parameters()}
    from safetensors.torch import save_file

    fn = os.path.join(tmp_path, "model.safetensors")
    save_file(tensors, fn)
    idx = {"weight_map": {n: "model.safetensors" for n in tensors}}
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f)
    return fn


class TestSynthesizePredictorE:
    def test_token_ids_shift_by_one(self):
        ids = torch.tensor([[10, 11, 12]])
        embed = nn.Embedding(20, 4)
        e = synthesize_predictor_e(ids, embed=embed)
        assert e.shape == (1, 3, 4)
        assert torch.allclose(e[0, 0], embed.weight[11])
        assert torch.allclose(e[0, 1], embed.weight[12])
        assert torch.allclose(e[0, 2], embed.weight[12])  # final position repeats

    def test_negative_ignore_index_never_looks_up(self):
        # calibration caches mark ignored positions (incl. every last
        # position) with -100; clamping must keep the embedding lookup valid
        ids = torch.tensor([[10, 11, -100]])
        embed = nn.Embedding(20, 4)
        e = synthesize_predictor_e(ids, embed=embed)
        assert e.shape == (1, 3, 4)
        assert torch.allclose(e[0, 2], embed.weight[0])

    def test_embedded_rows_shift(self):
        rows = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        e = synthesize_predictor_e(rows)
        assert torch.allclose(e[0, 0], rows[0, 1])
        assert torch.allclose(e[0, 2], rows[0, 2])


class TestPredictorForward:
    def test_runs_like_a_block(self):
        mtp = _MTP()
        bind_predictor_forward(
            mtp,
            {
                "norm_e": mtp.enorm,
                "norm_h": mtp.hnorm,
                "fc": mtp.eh_proj,
                "layer": mtp.layer,
                "final_norm": mtp.final_norm,
            },
        )
        h = torch.randn(2, 5, HID)
        e = torch.randn(2, 5, HID)
        out = mtp(h, _predictor_e=e)
        assert out.shape == (2, 5, HID)

    def test_per_row_list_input_is_concatenated(self):
        mtp = _MTP()
        bind_predictor_forward(
            mtp,
            {
                "norm_e": mtp.enorm,
                "norm_h": mtp.hnorm,
                "fc": mtp.eh_proj,
                "layer": mtp.layer,
                "final_norm": mtp.final_norm,
            },
        )
        h = torch.randn(2, 5, HID)
        e_rows = [torch.randn(1, 5, HID) for _ in range(2)]
        out = mtp(h, _predictor_e=e_rows)
        assert out.shape == (2, 5, HID)


class TestEnsureModulePath:
    def test_creates_intermediate_shells(self):
        m = nn.Module()
        parent = ensure_module_path(m, "a.b.c")  # returns the parent for leaf "c"
        parent.add_module("c", nn.Linear(2, 2))
        assert m.a.b.c is not None

    def test_replaces_stale_none_attribute(self):
        m = _Body()  # carries a plain mtp=None attribute
        parent = ensure_module_path(m, "mtp.layer")
        parent.add_module("layer", nn.Linear(2, 2))
        assert m.mtp.layer is not None


class TestCheckpointReading:
    def test_lists_and_roots(self, tmp_path):
        mtp = _MTP()
        _write_ckpt(tmp_path, mtp)
        body = _Body()
        tensors = list_checkpoint_tensors(str(tmp_path))
        assert "mtp.eh_proj.weight" in tensors
        assert tensors["mtp.eh_proj.weight"][0] == (HID, 2 * HID)
        roots = checkpoint_only_roots(tensors, body)
        assert roots == ["mtp"]


class TestAnalyzePredictorGroup:
    def test_roles_from_shapes_and_names(self, tmp_path):
        mtp = _MTP()
        _write_ckpt(tmp_path, mtp)
        tensors = list_checkpoint_tensors(str(tmp_path))
        info = analyze_predictor_group(tensors, "mtp", HID)
        assert info is not None
        assert info["fc"] == "mtp.eh_proj.weight"
        assert info["norm_e"] == "mtp.enorm.weight"
        assert info["norm_h"] == "mtp.hnorm.weight"
        assert info["final_norm"] == "mtp.final_norm.weight"
        assert info["layer_root"] == "mtp.layer"

    def test_rejects_non_predictor_group(self, tmp_path):
        from safetensors.torch import save_file

        save_file({"foo.weight": torch.randn(4, 4)}, os.path.join(tmp_path, "x.safetensors"))
        tensors = list_checkpoint_tensors(str(tmp_path))
        assert analyze_predictor_group(tensors, "foo", HID) is None


class TestBuildPredictorTree:
    def _setup(self, tmp_path):
        mtp = _MTP()
        _write_ckpt(tmp_path, mtp)
        body = _Body()
        tensors = list_checkpoint_tensors(str(tmp_path))
        info = analyze_predictor_group(tensors, "mtp", HID)
        all_blocks = [["blocks.0"], ["blocks.1"]]
        return mtp, body, tensors, info, all_blocks

    def test_sibling_pick(self, tmp_path):
        _, body, tensors, info, all_blocks = self._setup(tmp_path)
        picked = pick_sibling_layer(body, tensors, info, all_blocks)
        assert picked is not None
        name, mod = picked
        assert name.startswith("blocks.")
        assert isinstance(mod, _DecLayer)

    def test_tree_builds_and_matches_reference(self, tmp_path):
        mtp_ref, body, tensors, info, all_blocks = self._setup(tmp_path)
        name, sibling = pick_sibling_layer(body, tensors, info, all_blocks)
        claimed = build_predictor_tree(body, str(tmp_path), tensors, info, sibling)
        assert any("mtp.eh_proj.weight" in c for c in claimed)
        tree = body.get_submodule("mtp")
        # weights loaded from checkpoint, not the sibling copy
        assert torch.allclose(tree.eh_proj.weight, mtp_ref.eh_proj.weight.detach())
        assert torch.allclose(tree.layer.fc1.weight, mtp_ref.layer.fc1.weight.detach())
        assert torch.allclose(tree.enorm.weight, mtp_ref.enorm.weight.detach())
        bind_predictor_forward(
            tree,
            {
                "norm_e": tree.enorm,
                "norm_h": tree.hnorm,
                "fc": tree.eh_proj,
                "layer": tree.layer,
                "final_norm": tree.final_norm,
            },
        )
        h = torch.randn(2, 6, HID)
        e = torch.randn(2, 6, HID)
        out = tree(h, _predictor_e=e)
        with torch.no_grad():
            ref = mtp_ref.final_norm(
                mtp_ref.layer(mtp_ref.eh_proj(torch.cat([mtp_ref.enorm(e), mtp_ref.hnorm(h)], -1)))
            )
        assert torch.allclose(out, ref, atol=1e-5)
