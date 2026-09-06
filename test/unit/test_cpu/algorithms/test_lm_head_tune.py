# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""lm_head tuning inputs from the calibration chain tail (streaming, iters>0).

At iters>0 the streaming zero-shot pass tunes lm_head with the chain's final
hidden states, exactly like the data-driven path tunes outside-block layers.
``_lm_head_tune_inputs_`` derives the per-sample row lists (unwrapping
dict-shaped chain tails) or returns ``None`` to keep the closed-form search.
"""

import inspect
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from auto_round.compressors.orchestrator import CompressionOrchestrator


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class _NoNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 8)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 8)


class _TrailingTree(nn.Module):
    """lm_head followed by an attached checkpoint-only placeholder subtree -
    the module-order "last leaf" lands inside the placeholder, not on lm_head."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)
        self.mtp = nn.Module()
        self.mtp.pre_fc_norm_hidden = nn.LayerNorm(4)


class _BackboneWrapper(nn.Module):
    """Text backbone + vision tower wrapper: the vision norms sit between the
    last text block and lm_head and share the 1D-weight (and even width)
    signature, so the final-norm scan must be restricted to the backbone."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Block()])
        self.model.language_model.norm = nn.LayerNorm(4)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


def _orch(iters, remain, model=None):
    orch = SimpleNamespace(model=model or _Tiny())
    orch._max_tune_iters = lambda: iters
    for name in ("_lm_head_tune_inputs_", "_final_norm_module_", "_resolve_lm_head_name_"):
        setattr(orch, name, MethodType(getattr(CompressionOrchestrator, name), orch))
    # staticmethods: attach the plain function (instance attrs never bind self)
    orch._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows
    orch._block_backbone_prefix_ = CompressionOrchestrator._block_backbone_prefix_
    return orch


def _rows(n=2):
    return [torch.randn(1, 5, 4) for _ in range(n)]


class TestLmHeadNameResolution:
    """lm_head resolves from the quantization plan, never from module order."""

    def test_trailing_placeholder_tree_still_resolves_lm_head(self):
        orch = _orch(200, ["lm_head"], model=_TrailingTree())
        assert orch._resolve_lm_head_name_(["lm_head"]) == "lm_head"

    def test_trailing_placeholder_tree_tunes_end_to_end(self):
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        orch = _orch(200, ["lm_head"], model=_TrailingTree())
        out = orch._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is not None  # no silent closed-form fallback

    def test_unpinned_lm_head_resolves_to_none(self, capfd):
        assert _orch(200, ["norm"])._resolve_lm_head_name_(["norm"]) is None
        assert "warning" not in capfd.readouterr().err.lower()

    def test_multiple_candidates_warn(self, capfd):
        names = ["lm_head", "decoder.lm_head"]
        orch = _orch(200, names)
        assert orch._resolve_lm_head_name_(names) == "lm_head"
        assert "multiple lm_head candidates" in capfd.readouterr().err

    def test_substring_fallback_resolves_prefixed_heads(self):
        orch = _orch(200, ["model.lm_head_proj"])
        assert orch._resolve_lm_head_name_(["model.lm_head_proj"]) == "model.lm_head_proj"


class _FlatLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class TestFinalNormDiscovery:
    def test_wrapper_vision_norms_are_not_picked(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.language_model.layers.0"]])
        orch = _orch(200, ["lm_head"], model=_BackboneWrapper())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.language_model.norm"
        assert mod is orch.model.model.language_model.norm

    def test_flat_model_norm_still_found(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.layers.0"]])
        orch = _orch(200, ["lm_head"], model=_FlatLlama())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.norm"
        assert mod is orch.model.model.norm


class TestLmHeadTuneInputs:
    def test_zero_shot_run_keeps_closed_form(self, capfd):
        out = _orch(0, ["lm_head"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["lm_head"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_unpinned_lm_head_untouched(self, capfd):
        out = _orch(200, ["norm"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["norm"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_pinned_tuning_run_returns_post_norm_rows(self):
        rows, qrows, ids = _rows(3), _rows(3), [torch.randint(0, 8, (1, 5)) for _ in range(3)]
        state = {"fp_inputs": rows, "q_inputs": qrows, "token_ids": ids}
        orch = _orch(200, ["lm_head"])
        fp, q, tok = orch._lm_head_tune_inputs_(state, ["lm_head"])
        norm = orch.model.norm
        for got, src_rows in ((fp, rows), (q, qrows)):
            assert tok is ids
            for r, s in zip(got, src_rows):
                torch.testing.assert_close(r, norm(s.to(norm.weight.dtype)), msg="rows must be post-final-norm")
        assert fp is not rows  # fresh list: the tune loop reassigns entries in place

    def test_missing_final_norm_falls_back(self, capfd):
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        out = _orch(200, ["lm_head"], model=_NoNorm())._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is None
        assert "cannot locate the final norm" in capfd.readouterr().err

    def test_meta_final_norm_without_streamer_falls_back(self, capfd):
        orch = _orch(200, ["lm_head"])
        with torch.device("meta"):
            orch.model.norm = nn.LayerNorm(4)
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        out = orch._lm_head_tune_inputs_(state, ["lm_head"], streamer=None)
        assert out is None
        assert "final norm is still meta" in capfd.readouterr().err

    def test_dict_chain_tail_unwraps_hidden_states(self):
        rows, qrows, ids = _rows(2), _rows(2), [torch.randint(0, 8, (1, 5)) for _ in range(2)]
        state = {
            "fp_inputs": {"hidden_states": rows, "prev_topk_indices": torch.zeros(2)},
            "q_inputs": {"hidden_states": qrows, "prev_topk_indices": torch.zeros(2)},
            "token_ids": ids,
        }
        orch = _orch(200, ["lm_head"])
        fp, q, _ = orch._lm_head_tune_inputs_(state, ["lm_head"])
        norm = orch.model.norm
        for got, src_rows in ((fp, rows), (q, qrows)):
            for r, s in zip(got, src_rows):
                torch.testing.assert_close(r, norm(s.to(norm.weight.dtype)))

    def test_missing_chain_state_warns_and_falls_back(self, capfd):
        out = _orch(200, ["lm_head"])._lm_head_tune_inputs_({}, ["lm_head"])
        assert out is None
        assert "falls back" in capfd.readouterr().err

    def test_malformed_rows_warn_and_fall_back(self, capfd):
        state = {"fp_inputs": torch.zeros(2), "token_ids": [torch.zeros(1, 5)]}
        out = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is None
        assert "falls back" in capfd.readouterr().err

    def test_mis_shaped_q_rows_tune_on_fp_inputs(self, capfd):
        state = {"fp_inputs": _rows(3), "q_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 3}
        fp, q, _ = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert fp is not None and q is None
        assert "enable_quanted_input" in capfd.readouterr().err


class TestLmHeadTuneWiring:
    """Contract pins for the outside-block loop call site."""

    def test_zero_shot_loop_wires_tune_kwargs(self):
        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "self._lm_head_tune_inputs_(" in src
        i = src.index("compress_layer_outside_block(")
        window = src[i : i + 400]
        assert "**tune_kwargs" in window
        assert '"fp_inputs": _fp_rows' in src
        assert '"input_ids": _token_ids' in src

    def test_helper_only_runs_under_streaming(self):
        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "if streamer is not None else None" in src
