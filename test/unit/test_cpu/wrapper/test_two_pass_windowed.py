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


class TestStepParity:
    def _mk_wrapper(self, seed, in_f=512, out_f=256):
        torch.manual_seed(seed)
        layer = torch.nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
        layer.bits, layer.group_size, layer.sym, layer.data_type = 4, 128, True, "int"
        layer.super_bits = layer.super_group_size = None
        layer.scale_dtype = None
        layer.act_bits, layer.act_sym, layer.act_data_type, layer.act_dynamic = 16, True, None, None
        from auto_round.wrapper import WrapperLinear

        w = WrapperLinear(layer, enable_minmax_tuning=True, enable_torch_compile=False, device=torch.device("cpu"))
        for key in ("value", "min_scale", "max_scale"):
            w.params[key].requires_grad_(True)
        w.min_scale = w.params["min_scale"]
        w.max_scale = w.params["max_scale"]
        return w

    def test_step_matches_plain_backward_bitexact(self):
        from types import SimpleNamespace

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        w1, w2 = self._mk_wrapper(2), self._mk_wrapper(3, 256, 128)
        x = torch.randn(2, 16, 512, dtype=torch.bfloat16)
        ref = torch.randn(2, 16, 128, dtype=torch.float32)

        def block_fwd(a, b, inp):
            return b(a(inp))

        # plain composite
        y = block_fwd(w1, w2, x)
        loss = (y.float() - ref).pow(2).mean()
        loss.backward()
        plain = {n: w.params["value"].grad.clone() for n, w in (("w1", w1), ("w2", w2))}

        for w in (w1, w2):
            w.params["value"].grad = None

        # engaged: leaves + the quantizer step (chained wrappers = block shape)
        leaves = [w.begin_two_pass_forward_() for w in (w1, w2)]
        y2 = block_fwd(w1, w2, x)
        loss2 = (y2.float() - ref).pow(2).mean()
        SignRoundQuantizer._two_pass_backward_step_(SimpleNamespace(), [w1, w2], [], loss2, leaves)

        assert torch.equal(y.detach(), y2.detach())
        assert torch.equal(w1.params["value"].grad, plain["w1"])
        assert torch.equal(w2.params["value"].grad, plain["w2"])

    def test_quantize_block_rules_chunked_backward_by_env(self):
        import inspect

        from auto_round.algorithms.quantization.sign_round import quantizer as qmod

        src = inspect.getsource(qmod.SignRoundQuantizer.quantize_block)
        assert "AR_TUNE_CHUNKED_BACKWARD" in src or "_resolve_chunked_mode_" in src  # env rules the mode
        assert "_two_pass_backward_step_" in src  # engaged replacement
        assert "_scale_loss_and_backward(scaler, loss)" in src  # plain path retained
        assert "_is_oom_" in src  # auto switches on out-of-memory only


class TestChunkedBackwardMode:
    def test_env_tri_state_and_default(self, monkeypatch):
        from auto_round import envs as envs_mod

        monkeypatch.delenv("AR_TUNE_CHUNKED_BACKWARD", raising=False)
        assert envs_mod.AR_TUNE_CHUNKED_BACKWARD == "auto"
        for val in ("1", "0", "auto", " AUTO "):
            monkeypatch.setenv("AR_TUNE_CHUNKED_BACKWARD", val)
            assert envs_mod.AR_TUNE_CHUNKED_BACKWARD == val.strip().lower()
        monkeypatch.setenv("AR_TUNE_CHUNKED_BACKWARD", "yes")
        try:
            envs_mod.AR_TUNE_CHUNKED_BACKWARD
        except ValueError as e:
            assert "AR_TUNE_CHUNKED_BACKWARD" in str(e)
        else:
            raise AssertionError("invalid value must fail loudly")

    def test_is_oom(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        assert SignRoundQuantizer._is_oom_(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        assert not SignRoundQuantizer._is_oom_(RuntimeError("shape mismatch"))
        try:
            raise torch.cuda.OutOfMemoryError  # type: ignore[attr-defined]
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            assert SignRoundQuantizer._is_oom_(e)

    def test_forced_mode_declines_loudly_under_scaler(self, caplog):
        import logging
        from types import SimpleNamespace

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
        from auto_round.logger import logger as ar_logger

        fake = SimpleNamespace(_tune_sync_fn=None)
        mode, why = SignRoundQuantizer._resolve_chunked_mode_(fake, None, scaler=object())
        assert mode == "0" and why == "scaler active"

        with caplog.at_level(logging.DEBUG, logger="autoround"):
            ar_logger.addHandler(caplog.handler)
            try:
                fake2 = SimpleNamespace(_tune_sync_fn=None)
                monkey_env = {"AR_TUNE_CHUNKED_BACKWARD": "1"}
                import os

                old = os.environ.get("AR_TUNE_CHUNKED_BACKWARD")
                os.environ["AR_TUNE_CHUNKED_BACKWARD"] = "1"
                try:
                    mode2, _ = SignRoundQuantizer._resolve_chunked_mode_(fake2, None, scaler=object())
                finally:
                    if old is None:
                        os.environ.pop("AR_TUNE_CHUNKED_BACKWARD", None)
                    else:
                        os.environ["AR_TUNE_CHUNKED_BACKWARD"] = old
            finally:
                ar_logger.removeHandler(caplog.handler)
        assert mode2 == "0"
        assert any("AR_TUNE_CHUNKED_BACKWARD=1 ignored" in r.message for r in caplog.records)


class TestStableLeaf:
    def test_leaf_identity_stable_content_refreshes(self):
        w = self._w() if hasattr(self, "_w") else None
        import torch.nn as nn

        from auto_round.wrapper import WrapperLinear

        torch.manual_seed(7)
        layer = nn.Linear(64, 32, bias=False).to(torch.bfloat16)
        layer.bits, layer.group_size, layer.sym, layer.data_type = 4, 128, True, "int"
        layer.super_bits = layer.super_group_size = None
        layer.scale_dtype = None
        layer.act_bits, layer.act_sym, layer.act_data_type, layer.act_dynamic = 16, True, None, None
        w = WrapperLinear(layer, enable_minmax_tuning=True, enable_torch_compile=False, device=torch.device("cpu"))
        for k in ("value", "min_scale", "max_scale"):
            w.params[k].requires_grad_(True)
        w.min_scale = w.params["min_scale"]
        w.max_scale = w.params["max_scale"]

        leaf1 = w.begin_two_pass_forward_()
        leaf2 = w.begin_two_pass_forward_()
        # same tensor object across iterations: dynamo guards on the wrapper
        # attribute stay valid instead of recompiling per iteration
        assert leaf1 is leaf2
        before = leaf1.detach().clone()

        with torch.no_grad():
            w.params["value"].add_(0.5)
        leaf3 = w.begin_two_pass_forward_()
        assert leaf3 is leaf1
        assert not torch.equal(leaf3.detach(), before)  # content follows the params

        w.end_two_pass_forward_()
        assert w._two_pass_leaf_weight is None  # released for pack
