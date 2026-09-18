# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact VRAM-requirement calculation for block tunes (schema-derived)."""

import torch
import torch.nn as nn

from auto_round.utils.tune_memory import (
    _scheme_signature,
    exact_tune_bytes,
    predict_block_tune_peak,
    probe_saved_ratio,
)


def _stamped(out_f, in_f, bits=4, g=128, sym=True, data_type="int"):
    m = nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
    m.bits, m.group_size, m.sym, m.data_type = bits, g, sym, data_type
    m.super_bits = m.super_group_size = None
    m.scale_dtype = None
    return m


class _TinyWrapper(nn.Module):
    """Minimal wrapper stand-in: params dict with an fp32 value."""

    def __init__(self, layer, enable_minmax_tuning=True, enable_torch_compile=False, device=None):
        super().__init__()
        self.orig_layer = layer
        n = layer.weight.numel()
        self.params = {"value": torch.zeros(n // layer.group_size, layer.group_size, dtype=torch.float32)}

    def forward(self, x):
        # depend on the value so the graph saves a value-shaped operand,
        # like the real wrapper's fake-quant math does
        w_eff = self.orig_layer.weight.t().float() * self.params["value"].view(self.orig_layer.weight.t().shape)
        return x @ w_eff.to(x.dtype)


class TestExactBytes:
    def test_arithmetic_matches_shapes(self):
        m = _stamped(256, 512)  # 131072 elems, g128 -> 1024 groups
        b = exact_tune_bytes(m, enable_minmax_tuning=True)
        assert b["weight"] == 131072 * 2  # bf16
        assert b["value"] == 131072 * 4
        assert b["minmax"] == 1024 * 4 * 2
        assert b["grads"] == b["value"] + b["minmax"]
        assert b["snapshot"] == b["value"] + b["minmax"]

    def test_minmax_off_drops_scales(self):
        m = _stamped(64, 128)
        b = exact_tune_bytes(m, enable_minmax_tuning=False)
        assert b["minmax"] == 0 and b["snapshot"] == b["value"]

    def test_per_tensor_group_layout(self):
        m = _stamped(64, 128, g=0)
        b = exact_tune_bytes(m, enable_minmax_tuning=True)
        assert b["minmax"] == 1 * 4 * 2  # one per-tensor group


class TestProbe:
    def test_ratio_positive_and_cached(self):
        m = _stamped(256, 512)
        r1 = probe_saved_ratio(_TinyWrapper, m, torch.device("cpu"), enable_minmax_tuning=True)
        assert r1["fwd"] is not None and r1["fwd"] > 0
        r2 = probe_saved_ratio(_TinyWrapper, m, torch.device("cpu"), enable_minmax_tuning=True)
        assert r1 is r2  # same signature -> cached object

    def test_signature_tracks_scheme(self):
        a = _scheme_signature(_stamped(8, 8, sym=True))
        b = _scheme_signature(_stamped(8, 8, sym=False))
        assert a != b


class TestPredict:
    def test_total_is_sum_of_terms(self):
        layers = [_stamped(256, 512), _stamped(128, 512)]
        r = probe_saved_ratio(_TinyWrapper, layers[0], torch.device("cpu"), enable_minmax_tuning=True)
        out = predict_block_tune_peak(
            layers,
            torch.device("cpu"),
            enable_minmax_tuning=True,
            wrapper_cls=_TinyWrapper,
            allocated_now=1000,
            act_est_bytes=2000,
            frag_budget_bytes=3000,
        )
        fixed = sum(exact_tune_bytes(m, True)[k] for m in layers for k in ("weight", "value", "minmax", "grads"))
        snap = sum(exact_tune_bytes(m, True)["snapshot"] for m in layers)
        saved = sum(int(exact_tune_bytes(m, True)["value"] * r["fwd"]) for m in layers)
        assert out["total"] == 1000 + fixed + snap + saved + 2000 + 3000

    def test_host_snapshot_excluded_and_unknown_saved_is_none(self):
        m = _stamped(64, 128)
        out = predict_block_tune_peak(
            [m],
            torch.device("cpu"),
            enable_minmax_tuning=True,
            wrapper_cls=None,
            snapshot_on_host=True,
        )
        assert out["snapshot"] == 0
        assert out["saved"] is None and out["total"] is None  # unknown, not zero
