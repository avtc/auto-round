# Copyright (c) 2025 Intel Corporation
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

"""Tests for lm_head chain-tail inputs (block-loop chain output -> final norm -> lm_head).

Covers: row extraction from list/dict chain states, lm_head name resolution,
the tail derivation (norm applied, width sanity, closed-form fallbacks), the
early-stop override gate, and the outside-block lane consuming tail inputs
without issuing capture passes.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

import auto_round.compressors.orchestrator as orch
from auto_round.compressors.orchestrator import CompressionOrchestrator


class _TinyModel(nn.Module):
    def __init__(self, hidden=8, vocab=16):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = nn.RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.hf_device_map = {}


def _orchestrator_like(model):
    o = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        _tail_fed_layers_=[],
        _lm_head_chain_tail_=None,
        _lm_head_norm_name_=None,
        _predictor_plan_=None,
    )
    o._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows  # staticmethod: bind directly
    for name in ("_resolve_lm_head_name_", "_lm_head_tail_inputs_", "_discover_final_norm_", "_attach_tail_imatrix_"):
        setattr(o, name, MethodType(getattr(CompressionOrchestrator, name), o))
    o._predictor_overrides_last_cache_ = MethodType(CompressionOrchestrator._predictor_overrides_last_cache_, o)
    return o


class TestChainHiddenRows:
    def test_plain_list_passthrough(self):
        rows = [torch.zeros(1, 4) for _ in range(3)]
        assert CompressionOrchestrator._chain_hidden_rows(rows) is rows

    def test_dict_with_hidden_states(self):
        rows = [torch.zeros(1, 4) for _ in range(2)]
        assert CompressionOrchestrator._chain_hidden_rows({"hidden_states": rows}) is rows

    def test_structured_output_takes_hidden_states(self):
        rows = [torch.zeros(1, 4)]
        state = {"hidden_states": rows, "conv": [torch.zeros(1)]}
        assert CompressionOrchestrator._chain_hidden_rows(state) is rows

    def test_nested_dict_under_hidden_states(self):
        rows = [torch.zeros(1, 4)]
        assert CompressionOrchestrator._chain_hidden_rows({"hidden_states": {"inner": rows}}) is rows


class TestResolveLmHeadName:
    def setup_method(self):
        self.o = _orchestrator_like(_TinyModel())

    def test_exact_leaf_match(self):
        assert self.o._resolve_lm_head_name_(["lm_head"]) == "lm_head"

    def test_dotted_leaf_match(self):
        assert self.o._resolve_lm_head_name_(["model.lm_head", "other"]) == "model.lm_head"

    def test_substring_fallback(self):
        assert self.o._resolve_lm_head_name_(["proj.lm_head_w8"]) == "proj.lm_head_w8"

    def test_none_when_absent(self):
        assert self.o._resolve_lm_head_name_(["embed_tokens"]) is None
        assert self.o._resolve_lm_head_name_([]) is None


class TestLmHeadTailInputs:
    def setup_method(self):
        self.model = _TinyModel()
        self.o = _orchestrator_like(self.model)
        self.o._lm_head_norm_name_ = "model.norm"

    def _rows(self, scale=1.0):
        return [torch.randn(1, 5, 8) * scale for _ in range(3)]

    def test_norm_applied_to_fp_and_q_rows(self):
        fp, q = self._rows(), self._rows(2.0)
        self.o._lm_head_chain_tail_ = (q, fp)
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None
        norm = self.model.model.norm
        with torch.no_grad():
            expected_fp = [norm(r) for r in fp]
            expected_q = [norm(r) for r in q]
        for got, exp in zip(out[0], expected_fp):
            assert torch.allclose(got, exp, atol=1e-6)
        for got, exp in zip(out[1], expected_q):
            assert torch.allclose(got, exp, atol=1e-6)

    def test_norm_runs_on_weight_device_rows_return_to_row_device(self):
        """Cross-device contract: the row handed to the norm sits on the norm's
        weight device, the returned row lands on the row's original device.
        Chain-tail rows park on the cache device (host) while the norm's weight
        can be resident elsewhere - mixing them crashes RMSNorm."""
        fp, q = self._rows(), self._rows(2.0)
        self.o._lm_head_chain_tail_ = (q, fp)
        norm = self.model.model.norm
        seen_devices = []
        orig_forward = norm.forward

        def recording_forward(x):
            seen_devices.append(x.device)
            return orig_forward(x)

        norm.forward = recording_forward
        try:
            out = self.o._lm_head_tail_inputs_("lm_head")
        finally:
            norm.forward = orig_forward
        assert out is not None
        wdev = norm.weight.device
        assert all(d == wdev for d in seen_devices), f"norm saw rows on {seen_devices}, weight on {wdev}"
        assert all(r.device == fp[0].device for r in out[0])
        assert all(r.device == q[0].device for r in out[1])

    def test_missing_tail_falls_back(self):
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_bad_row_format_falls_back(self):
        self.o._lm_head_chain_tail_ = (None, "not-rows")
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_width_mismatch_falls_back(self):
        self.model.model.norm = nn.RMSNorm(4)  # wrong width vs lm_head.in_features=8
        self.o._lm_head_chain_tail_ = (self._rows(), self._rows())
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_missing_norm_falls_back(self):
        self.o._lm_head_norm_name_ = None
        self.o._lm_head_chain_tail_ = (self._rows(), self._rows())
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_malformed_q_rows_degrade_to_fp_only(self):
        self.o._lm_head_chain_tail_ = (None, self._rows())
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None and out[1] is None

    def test_structured_chain_state(self):
        fp = {"hidden_states": self._rows()}
        q = {"hidden_states": self._rows(2.0)}
        self.o._lm_head_chain_tail_ = (q, fp)
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None and len(out[0]) == 3


class TestTailModeGates:
    def test_override_false_without_predictor(self):
        o = _orchestrator_like(_TinyModel())
        assert o._predictor_overrides_last_cache_() is False

    def test_override_false_with_tail_fed_layers_only(self):
        # the single-block-target early-stop must stay active for tail mode:
        # lm_head statistics come from the tail rows, not from the walk
        o = _orchestrator_like(_TinyModel())
        o._tail_fed_layers_ = ["lm_head"]
        assert o._predictor_overrides_last_cache_() is False

    def test_override_true_with_predictor_plan(self):
        o = _orchestrator_like(_TinyModel())
        o._predictor_plan_ = {"roots": ["mtp"]}
        assert o._predictor_overrides_last_cache_() is True


class TestTailImatrix:
    """The fp-input imatrix must equal what the quantizer hook would accumulate."""

    def _rows(self):
        torch.manual_seed(0)
        return [torch.randn(1, 7, 8) for _ in range(3)]

    def test_imatrix_matches_hook_math(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        rows = self._rows()
        o._attach_tail_imatrix_("lm_head", rows)
        expected = None
        n = 0
        for row in rows:
            flat = row.reshape(-1, row.shape[-1]).to(torch.float32)
            sq = torch.sum(flat.pow(2), dim=0)
            expected = sq if expected is None else expected + sq
            n += flat.shape[0]
        assert hasattr(model.lm_head, "imatrix")
        assert torch.allclose(model.lm_head.imatrix, expected, atol=1e-5)
        assert model.lm_head.imatrix_cnt == n

    def test_imatrix_never_overwrites_existing(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        model.lm_head.imatrix = torch.ones(8)
        o._attach_tail_imatrix_("lm_head", self._rows())
        assert torch.equal(model.lm_head.imatrix, torch.ones(8))

    def test_imatrix_skipped_without_rows(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        o._attach_tail_imatrix_("lm_head", [])
        assert not hasattr(model.lm_head, "imatrix")


class TestLaneConsumesTailInputs:
    """The outside-block lane feeds lm_head from the tail and skips capture passes."""

    def _run_lane(self, monkeypatch):
        model = _TinyModel()
        o = _orchestrator_like(model)
        o._tail_fed_layers_ = ["lm_head"]
        o._lm_head_norm_name_ = "model.norm"
        fp, q = [torch.randn(1, 5, 8) for _ in range(2)], [torch.randn(1, 5, 8) for _ in range(2)]
        o._lm_head_chain_tail_ = (q, fp)

        captured_calls = []

        class _Composer:
            def need_quanted_input(self):
                return True

            def compress_layer_outside_block(self, layer, fp_inputs=None, q_inputs=None, **kw):
                captured_calls.append((fp_inputs, q_inputs))

        o.alg_composer = _Composer()
        o.compress_context = SimpleNamespace(is_immediate_packing=False, is_immediate_saving=False, cache_device="cpu")
        o.calibration_context = SimpleNamespace(nsamples=2)
        o.formats = []
        o.act_bits = 16
        o.act_dynamic = True

        cache_calls = []

        def _no_cache_data(*args, **kwargs):
            cache_calls.append((args, kwargs))
            return {}

        o.cache_data = _no_cache_data
        o.model = model
        monkeypatch.setattr(orch, "memory_monitor", SimpleNamespace(update=lambda: None, log_summary=lambda: None))
        o._tune_predictor_trees_ = lambda token_ids: None  # SimpleNamespace: instance attr only

        lane = MethodType(orch.CompressionOrchestrator._quantize_layers_outside_blocks, o)
        lane(["lm_head"], {}, token_ids=None)
        return captured_calls, cache_calls, o

    def test_lane_passes_tail_rows_and_skips_capture(self, monkeypatch):
        captured_calls, cache_calls, o = self._run_lane(monkeypatch)
        assert len(captured_calls) == 1
        fp_used, q_used = captured_calls[0]
        assert fp_used is not None and q_used is not None
        norm = o.model_context.model.model.norm
        with torch.no_grad():
            exp_fp = [norm(r) for r in o._lm_head_chain_tail_[1]] if o._lm_head_chain_tail_ else None
        if exp_fp is not None:
            for got, exp in zip(fp_used, exp_fp):
                assert torch.allclose(got, exp, atol=1e-6)
        # the tail rows were consumed and released; no capture pass was issued at all
        assert o._lm_head_chain_tail_ is None
        assert cache_calls == []

    def test_lane_attaches_tail_imatrix(self, monkeypatch):
        captured_calls, cache_calls, o = self._run_lane(monkeypatch)
        lm = o.model_context.model.lm_head
        assert hasattr(lm, "imatrix") and lm.imatrix.numel() == lm.in_features
        # parity with the hook math over the exact rows the lane received
        norm = o.model_context.model.model.norm
        rows = captured_calls[0][0]
        expected = None
        for row in rows:
            flat = row.reshape(-1, row.shape[-1]).to(torch.float32)
            sq = torch.sum(flat.pow(2), dim=0)
            expected = sq if expected is None else expected + sq
        assert torch.allclose(lm.imatrix, expected, atol=1e-5)
        assert lm.imatrix_cnt == sum(r.numel() // r.shape[-1] for r in rows)
