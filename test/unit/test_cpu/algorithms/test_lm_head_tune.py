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


def _orch(iters, remain):
    orch = SimpleNamespace(model=_Tiny())
    orch._max_tune_iters = lambda: iters
    orch._lm_head_tune_inputs_ = MethodType(CompressionOrchestrator._lm_head_tune_inputs_, orch)
    return orch


def _rows(n=2):
    return [torch.randn(1, 5, 4) for _ in range(n)]


class TestLmHeadTuneInputs:
    def test_zero_shot_run_keeps_closed_form(self, capfd):
        out = _orch(0, ["lm_head"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["lm_head"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_unpinned_lm_head_untouched(self, capfd):
        out = _orch(200, ["norm"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["norm"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_pinned_tuning_run_returns_rows(self):
        rows, qrows, ids = _rows(3), _rows(3), [torch.randint(0, 8, (1, 5)) for _ in range(3)]
        state = {"fp_inputs": rows, "q_inputs": qrows, "token_ids": ids}
        out = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is not None
        fp, q, tok = out
        assert fp == rows and q == qrows and tok == ids
        assert fp is not rows  # fresh list: the tune loop reassigns entries in place

    def test_dict_chain_tail_unwraps_hidden_states(self):
        rows, qrows, ids = _rows(2), _rows(2), [torch.randint(0, 8, (1, 5)) for _ in range(2)]
        state = {
            "fp_inputs": {"hidden_states": rows, "prev_topk_indices": torch.zeros(2)},
            "q_inputs": {"hidden_states": qrows, "prev_topk_indices": torch.zeros(2)},
            "token_ids": ids,
        }
        fp, q, _ = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert fp == rows and q == qrows

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
