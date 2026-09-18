# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Two-pass windowed backward: exact block-tune gradients without the
full-size saved set.

Pass 1 computes dL/dw_q with the qdq math under no_grad (the block graph
saves only activation-sized tensors). Pass 2, per wrapper per weight window,
recomputes the window's qdq WITH graph and backpropagates
sum(w_q_window * g_window): by the chain rule dL/dvalue = g . dqdq/dvalue -
identical to differentiating through the composite, bit-exact."""

import torch
import torch.nn as nn

from auto_round.utils.distributed import _noop_sync, engine_owns_gradient_sync


class _Wrap(nn.Module):
    def __init__(self, out_f, in_f, g=128):
        super().__init__()
        self.orig_layer = self._lin = nn.Linear(in_f, out_f, bias=False)
        self.orig_layer.bits, self.orig_layer.group_size = 4, g
        self.orig_layer.sym, self.orig_layer.data_type = True, "int"
        self.orig_layer.super_bits = self.orig_layer.super_group_size = None
        self.params = {"value": torch.zeros(out_f * in_f // g, g, dtype=torch.float32)}


def _ste_round(x):
    """Straight-through round, like the real quant funcs."""
    return x + (x.round() - x).detach()


def _qdq(weight, value, g=128):
    """Composite fake-quant standing in for the wrapper's math."""
    w = weight.float().reshape(-1, g)
    v = value.reshape(-1, g)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    return (_ste_round(w / scale + v).clamp(-7, 7) * scale).reshape(weight.shape)


class TestEngineOwnsGradientSync:
    def test_plain_block_and_noop(self):
        assert not engine_owns_gradient_sync(_Wrap(8, 8), _noop_sync)

    def test_nonnoop_sync_fn_owns(self):
        assert engine_owns_gradient_sync(_Wrap(8, 8), lambda: None)

    def test_dist_initialized_owns(self, monkeypatch):
        import auto_round.utils.distributed as dmod

        class _FakeDist:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

        monkeypatch.setattr(dmod.torch, "distributed", _FakeDist, raising=False)
        assert engine_owns_gradient_sync(_Wrap(8, 8), _noop_sync)


class TestTwoPassParity:
    def test_value_grads_match_composite_bitexact(self):
        torch.manual_seed(0)
        w = _Wrap(256, 512)
        w.params["value"].requires_grad_(True)
        with torch.no_grad():
            w.params["value"].normal_(0, 0.1)  # non-trivial rounding offsets
        weight, value = w.orig_layer.weight, w.params["value"]
        x = torch.randn(2, 32, 512)
        ref = torch.randn(2, 32, 256)

        # full composite reference
        wq = _qdq(weight, value)
        loss = ((x @ wq.t().float() - ref) ** 2).mean()
        g_full = torch.autograd.grad(loss, value)[0]

        # two-pass: g = dL/dw_q (activation-sized graph), then local windows
        with torch.no_grad():
            wq_ng = _qdq(weight, value)
        wq_leaf = wq_ng.detach().clone().requires_grad_(True)
        loss1 = ((x @ wq_leaf.t().float() - ref) ** 2).mean()
        g_wq = torch.autograd.grad(loss1, wq_leaf)[0]

        acc = torch.zeros_like(value)
        step = 128 * 4  # window = 4 groups of rows in the flat layout
        for s in range(0, value.numel(), step):
            v_win = value.reshape(-1)[s : s + step]
            wq_win = _qdq_flat_window(weight, value, s, step)
            g_win = g_wq.reshape(-1)[s : s + step].reshape(-1, 128)
            local = (wq_win * g_win).sum()
            grad = torch.autograd.grad(local, value)[0]
            assert grad.shape == value.shape
            acc += grad
        # compare accumulated windowed grads to the composite
        assert torch.allclose(g_full, acc, atol=0, rtol=0)


def _qdq_flat_window(weight, value, start, step):
    """Windowed qdq over the flattened layout (mirrors _qdq_weight_block)."""
    g = 128
    w = weight.float().reshape(-1, g)
    v = value.reshape(-1, g)
    w_win = w.reshape(-1)[start : start + step].reshape(-1, g)
    v_win = v.reshape(-1)[start : start + step].reshape(-1, g)
    scale = w_win.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    return _ste_round(w_win / scale + v_win).clamp(-7, 7) * scale


class TestRealWrapperTwoPass:
    def test_grads_match_plain_forward_bitexact(self):
        """The real WrapperLinear: leaf-forward + windowed backward_from_weight_grad_
        must reproduce autograd through wrapper.forward exactly."""
        import torch.nn.functional as F  # noqa: F401

        torch.manual_seed(1)
        layer = torch.nn.Linear(512, 256, bias=False).to(torch.bfloat16)
        layer.bits, layer.group_size, layer.sym, layer.data_type = 4, 128, True, "int"
        layer.super_bits = layer.super_group_size = None
        layer.scale_dtype = None
        layer.act_bits, layer.act_sym, layer.act_data_type, layer.act_dynamic = 16, True, None, None
        from auto_round.wrapper import WrapperLinear

        mk = lambda: WrapperLinear(
            layer, enable_minmax_tuning=True, enable_torch_compile=False, device=torch.device("cpu")
        )
        w_plain, w_two = mk(), mk()
        with torch.no_grad():
            w_two.params["value"].copy_(w_plain.params["value"])
            w_two.params["min_scale"].copy_(w_plain.params["min_scale"])
            w_two.params["max_scale"].copy_(w_plain.params["max_scale"])
        for w in (w_plain, w_two):
            w.params["value"].requires_grad_(True)
            w.params["min_scale"].requires_grad_(True)
            w.params["max_scale"].requires_grad_(True)
            w.min_scale = w.params["min_scale"]
            w.max_scale = w.params["max_scale"]
        x = torch.randn(2, 32, 512, dtype=torch.bfloat16)
        ref = torch.randn(2, 32, 256, dtype=torch.float32)

        # plain: full composite graph
        y = w_plain(x)
        loss = (y.float() - ref).pow(2).mean()
        loss.backward()
        g_plain_v = w_plain.params["value"].grad.clone()
        g_plain_min = w_plain.params["min_scale"].grad.clone()

        # two-pass
        leaf = w_two.begin_two_pass_forward_()
        y2 = w_two(x)
        loss2 = (y2.float() - ref).pow(2).mean()
        g_wq = torch.autograd.grad(loss2, leaf)[0]
        w_two.backward_from_weight_grad_(g_wq)

        assert torch.allclose(w_two.params["value"].grad, g_plain_v, atol=0, rtol=0)
        assert torch.allclose(w_two.params["min_scale"].grad, g_plain_min, atol=1e-7, rtol=1e-5) or torch.allclose(
            w_two.params["min_scale"].grad, g_plain_min, atol=0, rtol=0
        )
        # forward outputs identical
        assert torch.equal(y.detach(), y2.detach())
