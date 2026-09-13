# coding=utf-8
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-device grouped experts plans and the align-hook exemption."""

import unittest

import torch
from torch import nn

from auto_round.modeling.fused_moe.grouped_experts import (
    _build_plan,
    _hooks_are_alignment_only,
    _projection_is_supported,
    _run_routes,
)


def _linear(out=4, inp=4):
    lin = nn.Linear(inp, out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn_like(lin.weight))
    return lin


def _expert(gate=True, seed=0):
    torch.manual_seed(seed)
    mod = nn.Module()
    mod.up_proj = _linear()
    mod.down_proj = _linear()
    if gate:
        mod.gate_proj = _linear()
    return mod


class _Experts(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.n = n
        for i in range(n):
            setattr(self, str(i), _expert(seed=i))
        self.act_fn = nn.functional.silu

    def forward(self, x, idx, w):
        return _run_routes(self, x, idx, w, self.n)


class TestHookExemption(unittest.TestCase):
    def test_no_hooks_is_alignment_only(self):
        self.assertTrue(_hooks_are_alignment_only(_linear()))

    def test_plain_hook_is_not_exempt(self):
        lin = _linear()
        lin.register_forward_hook(lambda m, i, o: o)
        self.assertFalse(_hooks_are_alignment_only(lin))
        self.assertFalse(_projection_is_supported(lin))  # calibration hooks must fire

    def test_align_hook_without_offload_is_exempt(self):
        from accelerate.hooks import AlignDevicesHook

        lin = _linear()
        lin.register_forward_pre_hook(AlignDevicesHook(execution_device=lin.weight.device))
        self.assertTrue(_hooks_are_alignment_only(lin))
        self.assertTrue(_projection_is_supported(lin))

    def test_align_hook_with_offload_is_not_exempt(self):
        from accelerate.hooks import AlignDevicesHook

        lin = _linear()
        hook = AlignDevicesHook(execution_device=lin.weight.device, offload=True)
        lin.register_forward_pre_hook(hook)
        self.assertFalse(_hooks_are_alignment_only(lin))
        self.assertFalse(_projection_is_supported(lin))


class TestBuildPlanPerSubset(unittest.TestCase):
    def test_single_device_subset_builds(self):
        experts = _Experts(2)
        plan = _build_plan(experts, [0, 1])
        self.assertIsNotNone(plan)
        self.assertEqual(plan.device, torch.device("cpu"))

    def test_mixed_bits_within_subset_falls_back(self):
        experts = _Experts(2)
        experts.up_proj = None  # break slot resolution -> container validation fails
        try:
            experts.up_proj = None
        except AttributeError:
            pass
        # remove slot attribute entirely via a container without it
        container = nn.Module()
        container.not_a_slot = _linear()
        setattr(experts, "0", container)
        self.assertIsNone(_build_plan(experts, [0]))


class TestRunRoutesSingleGroupCPU(unittest.TestCase):
    def test_matches_reference(self):
        torch.manual_seed(7)
        experts = _Experts(4)
        n_tokens, n_experts, top_k, hidden = 12, 4, 2, 4
        x = torch.randn(n_tokens, hidden)
        idx = torch.randint(0, n_experts, (n_tokens, top_k))
        w = torch.rand(n_tokens, top_k)

        out = experts(x, idx, w)

        ref = torch.zeros_like(x)
        for t in range(n_tokens):
            for k in range(top_k):
                e = getattr(experts, str(int(idx[t, k])))
                h = nn.functional.silu(e.gate_proj(x[t])) * e.up_proj(x[t])
                ref[t] += w[t, k] * e.down_proj(h)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@unittest.skipIf(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    "needs >=2 CUDA devices",
)
class TestRunRoutesMultiDeviceCUDA(unittest.TestCase):
    def test_two_device_groups_match_reference(self):
        torch.manual_seed(11)
        experts = _Experts(4).to("meta")
        # experts 0,1 on cuda:0; 2,3 on cuda:1 (weights move explicitly)
        for i in range(4):
            e = getattr(experts, str(i))
            dev = f"cuda:{i // 2}"
            for slot in ("up_proj", "down_proj", "gate_proj"):
                getattr(e, slot).to(dev)
        n_tokens, n_experts, top_k, hidden = 64, 4, 2, 4
        x = torch.randn(n_tokens, hidden, device="cuda:2")  # input on a third device
        idx = torch.randint(0, n_experts, (n_tokens, top_k), device="cuda:2")
        w = torch.rand(n_tokens, top_k, device="cuda:2")

        out = experts(x, idx, w)
        self.assertEqual(out.device, x.device)

        # reference on cuda:2 by explicit per-expert moves
        ref = torch.zeros_like(x)
        for t in range(n_tokens):
            for k in range(top_k):
                e = getattr(experts, str(int(idx[t, k])))
                dev = next(p.device for p in e.up_proj.parameters())
                xt = x[t].to(dev)
                h = nn.functional.silu(e.gate_proj(xt)) * e.up_proj(xt)
                ref[t] += (w[t, k] * e.down_proj(h).to(x.device)).to(x.device)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
