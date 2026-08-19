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
import copy

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.cafeq import CafeQConfig, CafeQTransform

HIDDEN = 64
HEAD_DIM = 16
N_V_HEADS = 4
N_LAYERS = 2


class MultiHeadAttn(nn.Module):
    """GQA-shaped attention: N_V_HEADS v-heads, scalar row-weighted mixing.

    The row-wise mixing (diag-attention) commutes with any right-multiplication
    on the head axis - the exactness property CafeQ's fold relies on.
    """

    def __init__(self, dim, head_dim, n_heads):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # per-head [seq, seq] diagonal attention: row-wise scaling of v
        seq = x.shape[1]
        scores = (q * k).mean(-1, keepdim=True) / (q.shape[-1] ** 0.5)
        a = torch.softmax(scores, dim=1)
        vh = v.reshape(v.shape[0], seq, self.n_heads, self.head_dim)
        out = (a.unsqueeze(-1) * vh).reshape(v.shape[0], seq, -1)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(dim)
        self.self_attn = MultiHeadAttn(dim, HEAD_DIM, N_V_HEADS)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(x)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, HIDDEN)
        self.layers = nn.ModuleList([Block(HIDDEN) for _ in range(N_LAYERS)])
        self.norm = nn.RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, 32, bias=False)

    def forward(self, ids):
        return self.lm_head(self.norm(self.layers[1](self.layers[0](self.embed_tokens(ids)))))


def _rand_init(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layernorm" in n or n.endswith("norm.weight"):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn_like(p) * (0.2 if "mlp" in n else 1.0))


class TestCafeQ:
    def _model(self):
        torch.manual_seed(3)
        model = MiniModel()
        _rand_init(model)
        model.eval()
        return model

    def test_exactness(self):
        model = self._model()
        ids = torch.randint(0, 32, (2, 10))
        with torch.no_grad():
            before = model(ids)
        cfg = CafeQConfig(head_dim=HEAD_DIM, iters=200, lr=1e-2, seed=7)
        CafeQTransform(cfg).apply_to_model(model)
        with torch.no_grad():
            after = model(ids)
        torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)

    def test_pqe_and_ranges_improve(self):
        """The learned per-head M must cut the V/O paired quantization error."""
        model = self._model()
        W_v = [l.self_attn.v_proj.weight.detach().clone() for l in model.layers]
        W_o = [l.self_attn.o_proj.weight.detach().clone() for l in model.layers]

        def pqe(wv, wo, hd=N_V_HEADS * HEAD_DIM):
            # stacked product per head; RTN-quantized (4-bit asym, per-channel)
            err = 0.0
            for v, o in zip(wv, wo):
                prod = o @ v
                q = lambda t: _rtn(t, 15)
                err += (q(o) @ q(v) - prod).pow(2).sum().sqrt()
            return err.item()

        before_pqe = pqe(W_v, W_o)
        before_max = max(
            max(w.abs().max().item() for w in W_v), max(w.abs().max().item() for w in W_o)
        )
        CafeQTransform(CafeQConfig(head_dim=HEAD_DIM, iters=500, lr=1e-2, seed=7)).apply_to_model(model)
        W_v2 = [l.self_attn.v_proj.weight.detach() for l in model.layers]
        W_o2 = [l.self_attn.o_proj.weight.detach() for l in model.layers]
        after_pqe = pqe(W_v2, W_o2)
        after_max = max(max(w.abs().max().item() for w in W_v2), max(w.abs().max().item() for w in W_o2))
        assert after_pqe < before_pqe * 0.98
        assert after_max < before_max

    def test_determinism(self):
        ids = torch.randint(0, 32, (1, 8))
        outs = []
        for _ in range(2):
            model = self._model()
            CafeQTransform(CafeQConfig(head_dim=HEAD_DIM, iters=50, seed=7)).apply_to_model(model)
            with torch.no_grad():
                outs.append(model(ids))
        torch.testing.assert_close(outs[0], outs[1], atol=0.0, rtol=0.0)

    def test_no_full_attn_raises(self):
        model = nn.Sequential(nn.Linear(HIDDEN, HIDDEN))
        with pytest.raises(ValueError, match="no.*attention"):
            CafeQTransform(CafeQConfig(iters=10)).apply_to_model(model)

    def test_head_dim_unresolvable_raises(self):
        model = self._model()
        with pytest.raises(ValueError, match="head_dim not resolvable"):
            CafeQTransform(CafeQConfig(iters=10)).apply_to_model(model)

    def test_m_is_orthogonal(self):
        model = self._model()
        t = CafeQTransform(CafeQConfig(head_dim=HEAD_DIM, iters=100, seed=5))
        t.apply_to_model(model)
        for m in t.learned_matrices.values():
            eye = torch.eye(m.shape[0])
            torch.testing.assert_close(m @ m.T, eye, atol=1e-4, rtol=1e-4)


def _rtn(t, maxq):
    t = t.to(torch.float32)
    wmin = torch.clamp(t.min(-1).values, max=0).unsqueeze(-1)
    wmax = torch.clamp(t.max(-1).values, min=0).unsqueeze(-1)
    s = ((wmax - wmin) / maxq).clamp(min=1e-8)
    return s * torch.clamp(torch.round(t / s + (-wmin / s)), 0, maxq) - s * (-wmin / s)
