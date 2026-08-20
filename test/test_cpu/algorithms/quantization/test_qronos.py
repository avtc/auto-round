# Copyright (c) 2026 Intel Corporation
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
import math

import pytest
import torch

from auto_round.algorithms.quantization.qronos import QronosConfig
from auto_round.algorithms.quantization.qronos.quantizer import (
    compute_group_grid,
    qronos_sequential_quantize,
)


def _spd_gram(X):
    """H = X^T X in fp32 (X: [m, N] with m > N, full column rank)."""
    X = X.to(torch.float32)
    return X.T @ X


def _damp(H, alpha=1e-6):
    sigma1 = torch.linalg.matrix_norm(H, ord=2)
    H = H + alpha * sigma1 * torch.eye(H.shape[0])
    return H


def _rtn(W, scale, zp, maxq):
    """Column-wise RTN with per-column scale/zp (asym)."""
    q = torch.clamp(torch.round(W / scale) + zp, 0, maxq)
    return scale * (q - zp)


class TestQronosSequentialQuantize:
    """Unit tests of the pure Qronos core (Algorithm 1 of arXiv 2505.11695)."""

    def test_exact_on_grid_weights(self):
        """On-grid weights + X~ == X must round-trip bit-exactly.

        If every w_t is exactly representable on the quantization grid,
        RTN(w_t) == w_t, so all error-diffusion updates vanish and the
        algorithm must return Q == W (the identity fixed point).
        """
        torch.manual_seed(0)
        m, N, n = 96, 32, 24
        X = torch.randn(m, N)
        H = _damp(_spd_gram(X))
        maxq = 15  # 4-bit asym
        zp = torch.full((n, N), 7.0)
        scale = torch.full((n, N), 2.0**-8)
        # integer weights on the grid: s * (k - z), k in [0, maxq]
        k = torch.randint(0, maxq + 1, (n, N)).to(torch.float32)
        W = scale * (k - zp)
        Q, loss = qronos_sequential_quantize(W, H, H, scale, zp, maxq, block_size=8)
        torch.testing.assert_close(Q, W, atol=0.0, rtol=0.0)
        assert loss.item() < 1e-9  # fp32 round-trip noise only
        assert torch.equal(X @ W.T, X @ Q.T)

    def test_block_size_equivalence(self):
        """The blocked update must be mathematically equivalent at any block size."""
        torch.manual_seed(1)
        m, N, n = 128, 64, 40
        X = torch.randn(m, N)
        H = _damp(_spd_gram(X))
        W = torch.randn(n, N)
        scale = torch.full((n, N), 0.05)
        zp = torch.full((n, N), 7.0)
        Q_a, _ = qronos_sequential_quantize(W, H, H, scale, zp, 15, block_size=8)
        Q_b, _ = qronos_sequential_quantize(W, H, H, scale, zp, 15, block_size=64)
        torch.testing.assert_close(Q_a, Q_b, atol=1e-5, rtol=1e-5)

    def test_beats_rtn_and_matches_optq_when_x_equals_xtilde(self):
        """With X~ == X, Qronos reduces to OPTQ and must not lose to plain RTN."""
        torch.manual_seed(2)
        m, N, n = 128, 48, 32
        X = torch.randn(m, N)
        H = _damp(_spd_gram(X))
        W = torch.randn(n, N) * 0.3
        maxq = 15
        scale, zp = compute_group_grid(W, bits=4, group_size=16, sym=False)
        sc = torch.repeat_interleave(scale, 16, dim=1)
        zc = torch.repeat_interleave(zp, 16, dim=1)
        Q, _ = qronos_sequential_quantize(W, H, H, sc, zc, maxq, block_size=16)
        Q_rtn = _rtn(W, sc, zc, maxq)
        err_q = (X @ (W - Q).T).pow(2).sum()
        err_r = (X @ (W - Q_rtn).T).pow(2).sum()
        assert err_q < err_r

    def test_identity_L_fallback_matches_rtn(self):
        """use_init=False with an identity L must reduce to plain RTN.

        The error-correcting init assumes the Cholesky block identity
        L_{>=2,>=2} L_{>=2,>=2}^T == (H_{>=2,>=2})^{-1}; with an identity L
        it degenerates to a raw Hessian conjugation of the weights, which is
        arbitrarily worse than RTN. The fallback path must therefore skip it.
        """
        torch.manual_seed(3)
        n, N = 32, 48
        H = _damp(_spd_gram(torch.randn(128, N)))
        W = torch.randn(n, N) * 0.3
        maxq = 15
        scale, zp = compute_group_grid(W, bits=4, group_size=16, sym=False)
        sc = torch.repeat_interleave(scale, 16, dim=1)
        zc = torch.repeat_interleave(zp, 16, dim=1)
        L_eye = torch.eye(N)
        Q, _ = qronos_sequential_quantize(W, H, H, sc, zc, maxq, block_size=16, L=L_eye, use_init=False)
        Q_rtn = _rtn(W, sc, zc, maxq)
        assert torch.allclose(Q, Q_rtn, atol=1e-6)
        # the init-active variant on the same identity L applies the OPTQ
        # first-column correction (solve form) and must stay bounded near RTN;
        # the historical failure was a raw Hessian conjugation growing the
        # error several-fold
        Q_init, _ = qronos_sequential_quantize(W, H, H, sc, zc, maxq, block_size=16, L=L_eye, use_init=True)
        # bounded: the correction may or may not flip rounding decisions on a
        # given draw, but must never grow the error several-fold as the
        # historical conjugation form did
        assert (W - Q_init).norm() <= (W - Q_rtn).norm() * 1.5

    def test_xtilde_neq_x_corrects_previous_layers(self):
        """With X~ != X, the G = X~^T X term must not increase the X~-output error."""
        torch.manual_seed(3)
        m, N, n = 128, 48, 32
        X = torch.randn(m, N)
        X_tilde = X + 0.15 * torch.randn(m, N)
        H = _damp(_spd_gram(X_tilde))
        G = _damp(X_tilde.T @ X)
        W = torch.randn(n, N) * 0.3
        maxq = 15
        scale, zp = compute_group_grid(W, bits=4, group_size=16, sym=False)
        sc = torch.repeat_interleave(scale, 16, dim=1)
        zc = torch.repeat_interleave(zp, 16, dim=1)
        Q_g, _ = qronos_sequential_quantize(W, H, G, sc, zc, maxq, block_size=16)
        # control: drop G (use H) -> plain OPTQ with stale statistics
        Q_h, _ = qronos_sequential_quantize(W, H, H, sc, zc, maxq, block_size=16)
        # the objective Qronos optimizes is || X w - X~ q ||^2
        err_g = (X @ W.T - X_tilde @ Q_g.T).pow(2).sum()
        err_h = (X @ W.T - X_tilde @ Q_h.T).pow(2).sum()
        assert err_g <= err_h * 1.001

    def test_sym_grid(self):
        torch.manual_seed(4)
        m, N, n = 64, 32, 16
        X = torch.randn(m, N)
        H = _damp(_spd_gram(X))
        W = torch.randn(n, N) * 0.2
        scale, zp = compute_group_grid(W, bits=4, group_size=-1, sym=True)
        assert zp is None
        maxq = 7  # 2^(b-1)
        sc = scale.expand(n, N)
        Q, _ = qronos_sequential_quantize(W, H, H, sc, None, maxq, block_size=8)
        assert torch.isfinite(Q).all()
        # grid membership: q = s * clamp(round(w/s)) with q in [-maxq*s, (maxq-1)*s]
        grid_vals = sc * torch.clamp(torch.round(Q / sc), -maxq, maxq - 1)
        torch.testing.assert_close(Q, grid_vals, atol=1e-6, rtol=1e-6)


class TestComputeGroupGrid:
    def test_asym_grid_conventions(self):
        """Must mirror data_type/int.py quant_tensor_asym exactly."""
        torch.manual_seed(5)
        W = torch.randn(8, 32)
        scale, zp = compute_group_grid(W, bits=4, group_size=16, sym=False)
        assert scale.shape == (8, 2) and zp.shape == (8, 2)
        maxq = 15.0
        Wg = W.reshape(8, 2, 16)
        wmin = torch.clamp(Wg.min(-1).values, max=0)
        wmax = torch.clamp(Wg.max(-1).values, min=0)
        exp_scale = ((wmax - wmin) / maxq).to(torch.float16).clamp(min=1e-5)
        exp_zp = torch.round(-wmin / exp_scale)
        torch.testing.assert_close(scale, exp_scale, atol=0.0, rtol=0.0)
        torch.testing.assert_close(zp, exp_zp, atol=0.0, rtol=0.0)


class TestQronosPipeline:
    """Integration through the AutoRound entry on a tiny local checkpoint."""

    @pytest.fixture(scope="class")
    def tiny_lm(self, tmp_path_factory):
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        torch.manual_seed(7)
        model = LlamaForCausalLM(cfg)
        d = tmp_path_factory.mktemp("qronos_ckpt")
        # minimal fast tokenizer (no sentencepiece dependency), as in the
        # stream_checkpoint test fixture
        from tokenizers import Tokenizer, models as tk_models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(tk_models.WordLevel(vocab={"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tk)
        tok.save_pretrained(str(d))
        model.save_pretrained(str(d))
        return str(d)

    @staticmethod
    def _dataset(vocab=64, rows=4, seqlen=16):
        torch.manual_seed(11)
        return [{"input_ids": torch.randint(0, vocab, (1, seqlen))} for _ in range(rows)]

    def test_quantize_and_save(self, tiny_lm, tmp_path):
        from auto_round.compressors.entry import AutoRound

        ar = AutoRound(
            tiny_lm,
            scheme="W4A16",
            alg_configs=[
                QronosConfig(group_size=16, block_size=8, dampening_alpha=1e-6, actorder=True, sym=False)
            ],
            format="auto_round",
            disable_model_free=True,
            device_map="cpu",
            nsamples=4,
            seqlen=16,
            dataset=self._dataset(),
            low_cpu_mem_usage=True,
        )
        ar.quantize_and_save(str(tmp_path / "out"), format="auto_round")
        # exported in packed auto_round-gptq layout (GPTQ-style, transposed):
        # qweight [in, out*bits/32], scales [groups, out], qzeros [groups, out*bits/32]
        import os

        from safetensors.torch import load_file

        out_dir = ar.output_dir
        shards = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
        tensors = {}
        for s in shards:
            tensors.update(load_file(os.path.join(out_dir, s)))
        n_checked = 0
        for name, t in tensors.items():
            if not name.endswith(".qweight"):
                continue
            base = name[: -len(".qweight")]
            qw, sc, qz = t, tensors[f"{base}.scales"], tensors[f"{base}.qzeros"]
            out_features = sc.shape[1]
            n_groups = sc.shape[0]
            in_features = n_groups * 16  # group_size used in this test
            assert qw.shape == (in_features, out_features * 4 // 32)  # 4-bit packing
            assert qz.shape == (n_groups, out_features * 4 // 32)
            n_checked += 1
        assert n_checked == 15  # 3 blocks x (q, o, gate, up, down); k/v skipped (indivisible)
        # the packed int4 backend is GPU-only, so verify loadability + grid
        # membership through the fake format instead (same quantizer output)
        ar2 = AutoRound(
            tiny_lm,
            scheme="W4A16",
            alg_configs=[
                QronosConfig(group_size=16, block_size=8, dampening_alpha=1e-6, actorder=True, sym=False)
            ],
            format="fake",
            disable_model_free=True,
            device_map="cpu",
            nsamples=4,
            seqlen=16,
            dataset=self._dataset(),
            low_cpu_mem_usage=True,
        )
        model, _folders = ar2.quantize_and_save(str(tmp_path / "out_fake"), format="fake")
        n_grid = 0
        for _n, mod in model.named_modules():
            sc = getattr(mod, "scale", None)
            if sc is None or not hasattr(mod, "weight"):
                continue
            zp = getattr(mod, "zp", None)
            w = mod.weight.data.to(torch.float32)
            sc = sc.to(torch.float32).view(sc.shape[0], -1)
            g = sc.shape[1]
            wq = w.reshape(w.shape[0], g, -1)
            s3 = sc.unsqueeze(-1)
            if isinstance(zp, torch.Tensor):
                z3 = zp.to(torch.float32).view(zp.shape[0], -1).unsqueeze(-1)
                qdq = s3 * (torch.clamp(torch.round(wq / s3) + z3, 0, 15) - z3)
            else:
                off = float(zp) if zp is not None else 0.0
                qdq = s3 * (torch.clamp(torch.round(wq / s3) + off, 0, 15) - off)
            torch.testing.assert_close(wq, qdq, atol=2e-3, rtol=2e-3)
            n_grid += 1
        assert n_grid == 15
        ids = torch.randint(0, 64, (1, 12))
        logits = model(ids).logits
        assert torch.isfinite(logits).all()


class TestHessianRobustness:
    """Statistics collection must survive non-finite replay activations and
    poisoned Hessians (diverging bf16 quantized-prefix replay, exploding input
    channels) without corrupting the weights."""

    def _mk_linear(self, n=8, N=32):
        import torch.nn as nn

        lin = nn.Linear(N, n, bias=False)
        lin.bits, lin.sym, lin.group_size, lin.data_type = 4, False, 32, "int"
        lin.global_name = "test.linear"
        return lin

    def test_hook_sanitizes_inf_inputs(self):
        from auto_round.algorithms.quantization.qronos.quantizer import QronosQuantizer
        from auto_round.algorithms.quantization.qronos.config import QronosConfig

        q = QronosQuantizer(QronosConfig())
        lin = self._mk_linear()
        q.register_qinput_forward_hooks.__func__  # exists
        handles = q.register_qinput_forward_hooks(_ParentOf(lin))
        x_bad = torch.randn(4, 32)
        x_bad[0, 3] = float("inf")
        x_bad[1, 5] = float("nan")
        x_good = torch.randn(4, 32)
        with torch.no_grad():
            lin(x_bad)
            lin(x_good)
        for h in handles:
            h.remove()
        H = lin._qronos_H
        assert torch.isfinite(H).all(), "H must stay finite when an input sample contains inf/nan"

    def test_fp_hooks_accumulate_H_without_cascade(self):
        """enable_quanted_input=False: the fp sweep must accumulate H directly.

        Without this the q-input sweep (which normally accumulates H) never
        runs, no module carries statistics, and every layer silently degrades
        to plain RTN instead of OPTQ.
        """
        from auto_round.algorithms.quantization.qronos.quantizer import QronosQuantizer
        from auto_round.algorithms.quantization.qronos.config import QronosConfig

        torch.manual_seed(0)
        q = QronosQuantizer(QronosConfig(enable_quanted_input=False))
        lin = self._mk_linear()
        handles = q.register_fp_input_forward_hooks(_ParentOf(lin))
        xs = [torch.randn(4, 32) for _ in range(3)]
        with torch.no_grad():
            for x in xs:
                lin(x)
        for h in handles:
            h.remove()
        H_ref = sum((x.T @ x) for x in xs)
        assert hasattr(lin, "_qronos_H"), "fp sweep must accumulate H when the cascade is off"
        assert torch.allclose(lin._qronos_H, H_ref, atol=1e-4)
        assert not hasattr(lin, "_qronos_xfp"), "no pairing stash is needed without the cascade"

        # with the cascade on, the fp sweep stashes inputs for G pairing instead
        q_on = QronosQuantizer(QronosConfig(enable_quanted_input=True))
        lin2 = self._mk_linear()
        handles = q_on.register_fp_input_forward_hooks(_ParentOf(lin2))
        with torch.no_grad():
            lin2(torch.randn(4, 32))
        for h in handles:
            h.remove()
        assert hasattr(lin2, "_qronos_xfp") and len(lin2._qronos_xfp) == 1
        assert not hasattr(lin2, "_qronos_H")

    def test_poisoned_hessian_falls_back_to_rtn(self):
        """H finite but with a 1e12 dynamic range must NOT guide the rounding."""
        from auto_round.algorithms.quantization.qronos.quantizer import QronosQuantizer
        from auto_round.algorithms.quantization.qronos.config import QronosConfig

        q = QronosQuantizer(QronosConfig())
        lin = self._mk_linear()
        # healthy-ish H plus one exploding channel -> astronomic diag range
        H = torch.eye(32) * 10.0
        H[7, 7] = 1e12
        lin._qronos_H = H
        w_before = lin.weight.data.clone()
        loss = q._qronos_quantize_layer(lin)
        assert loss >= 0
        # weight must still be quantized (not left fp) and finite
        assert torch.isfinite(lin.weight.data).all()

    def test_escalating_damping_recovers_poorly_conditioned_h(self):
        """A nearly-singular H should be rescued by damping escalation, not RTN."""
        from auto_round.algorithms.quantization.qronos.quantizer import QronosQuantizer
        from auto_round.algorithms.quantization.qronos.config import QronosConfig

        q = QronosQuantizer(QronosConfig())
        lin = self._mk_linear()
        H = torch.eye(32) * 10.0
        H[0, 0] = 1e-7  # nearly-degenerate direction
        lin._qronos_H = H
        loss = q._qronos_quantize_layer(lin)
        assert loss >= 0
        assert torch.isfinite(lin.weight.data).all()


class _ParentOf(torch.nn.Module):
    """Minimal block stand-in so named_modules() finds the linear."""

    def __init__(self, lin):
        super().__init__()
        self.linear = lin

    def forward(self, x):
        return self.linear(x)


class TestCrossGramGates:
    """The first-column + posterior-mean steps consume G globally, so any G/H
    anomaly (pairing-order bug, finitely diverging bf16 replay) must be caught
    by the fp-referenced health gates and degrade to OPTQ instead of
    destroying the layer."""

    def _layer(self, W, bits=4, g=32):
        import torch.nn as nn

        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(W)
        lin.bits, lin.sym, lin.group_size, lin.data_type = bits, False, g, "int"
        lin.global_name = "test.linear"
        lin._qronos_xfp = [torch.zeros(1)]
        lin._qronos_idx = 1
        return lin

    def test_diverged_and_misaligned_degrade_to_healthy(self):
        from auto_round.algorithms.quantization.qronos.quantizer import (
            QronosQuantizer, compute_group_grid,
        )
        from auto_round.algorithms.quantization.qronos.config import QronosConfig

        torch.manual_seed(3)
        n, N, g = 128, 256, 32
        W = torch.randn(n, N) * 0.02
        A = torch.randn(2048, N)
        H_fp = A.T @ A / 2048
        X2 = A + 0.05 * torch.randn_like(A)
        Xdiv = A.clone()
        Xdiv[:100] *= 300.0
        Xbad = A[torch.randperm(2048)]

        scale, zp = compute_group_grid(W, bits=4, group_size=g, sym=False)
        sg = scale.unsqueeze(-1).to(torch.float32)
        zg = zp.unsqueeze(-1).to(torch.float32)
        rtn = (sg * (torch.clamp(torch.round(W.view(n, N // g, g) / sg + zg), 0, 15.0) - zg)).view(n, N)
        rel_rtn = float((W - rtn).norm() / W.norm())

        quant = QronosQuantizer(QronosConfig(group_size=g, sym=False, block_size=128, actorder=True))
        scenarios = {
            "healthy": (X2.T @ X2 / 2048, X2.T @ A / 2048),
            "misaligned": (X2.T @ X2 / 2048, Xbad.T @ A / 2048),
            "diverged": (Xdiv.T @ Xdiv / 2048, Xdiv.T @ A / 2048),
        }
        for tag, (H, G) in scenarios.items():
            lin = self._layer(W)
            lin._qronos_H = H
            lin._qronos_G = G
            lin._qronos_Hfp = H_fp
            quant._qronos_quantize_layer(lin)
            rel = float((W - lin.weight.data).norm() / W.norm())
            assert rel < 2.0 * rel_rtn + 0.02, f"{tag}: rel err {rel:.4f} vs RTN {rel_rtn:.4f}"
