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
"""CafeQ learned paired V/O transform (clean-room, arXiv 2511.19705, Sec. 4.2).

Learns a per-layer orthogonal ``M`` (head_dim x head_dim) on the value-head
axis and folds it offline into v_proj / o_proj. Exactness: attention consumes
``v`` via row-wise per-head mixing, and ``(A_h V_h) M = A_h (V_h M)`` by
associativity, so ``W_v[h] <- M^T W_v[h]`` / ``W_o[:, h] <- W_o[:, h] M``
preserve the function bit-for-bit with zero online ops (stock-vLLM safe).

The transform is optimized on the paper's paired LogSumExp proxy over
per-channel max-magnitudes of ``U_h = (M^T W_v[h])^T`` and
``V_h = M^T W_o[:, h]^T`` - a smooth maximum of the quantization ranges,
whose minimizer shrinks the paired quantization error of the V*O product.
Calibration-free: weight statistics only.

Not ported from the paper (by design):
- Sec. 4.1 single-matrix learned block-diagonal M: needs an ONLINE ``M^{-1}``
  on the layer output (per-layer rotations do not fold through the residual
  stream) - no stock-vLLM path. The offline-fusable global variant of that
  idea is covered by BlockHadamard (fixed) and PeRQ (permuted).
- Algorithm 1 paired adaptive rounding: delegated to the terminal quantizer
  (OptRTN/NeUQI/Qronos) per our stack's separation of concerns.
"""

from __future__ import annotations


import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.cafeq.config import CafeQConfig
from auto_round.algorithms.transforms.spinquant.rotation_utils import iter_transformer_layers

from auto_round.logger import logger


def _random_orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform-ish random orthogonal matrix via QR of a Gaussian."""
    g = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(g)
    # fix signs so the distribution is unbiased (Q with positive diagonal R)
    d = torch.diagonal(r).sign()
    return q * d.unsqueeze(0)


def cayley(A: torch.Tensor) -> torch.Tensor:
    """Orthogonal matrix (I - A)(I + A)^{-1} for skew-symmetric A."""
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(I + A, I - A)


def inverse_cayley(M: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric A with cayley(A) = M (for orthogonal M without -1 eigs)."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return torch.linalg.solve(I + M, I - M)


class CafeQTransform(BaseRotation):
    """Learned per-head V/O rotation, offline-fused (calibration-free)."""

    def __init__(self, config: CafeQConfig) -> None:
        super().__init__(config)
        self.learned_matrices: dict = {}
        self._lse_history: dict = {}

    # ── model introspection ────────────────────────────────────────────────
    @staticmethod
    def _full_attn_layers(model) -> list:
        out = []
        for layer in iter_transformer_layers(model):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            v = getattr(attn, "v_proj", None)
            o = getattr(attn, "o_proj", None)
            if isinstance(v, nn.Linear) and isinstance(o, nn.Linear):
                out.append((layer, v, o))
        return out

    @staticmethod
    def _check_not_gated(attn_v, attn_o, attn_q) -> None:
        """Refuse archs whose attention output is gated before o_proj.

        Qwen3.5/Next-style attention packs [query | gate] into q_proj
        (out = 2 x o.in) and multiplies the o_proj INPUT by sigmoid(gate)
        elementwise. An input-dependent channelwise gate does not commute
        with a non-diagonal M, so the fold would NOT be exact.
        """
        if attn_q.out_features == 2 * attn_o.in_features:
            raise ValueError(
                "[CafeQ] this architecture gates the attention output before o_proj "
                "(q_proj packs [query | gate]); a non-diagonal V/O transform cannot "
                "fold exactly. Pass allow_gated=True to proceed anyway (inexact)."
            )

    @staticmethod
    def _resolve_head_dim(model, layers) -> int:
        cfg = getattr(model, "config", None)
        for candidate in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "language_model", None)):
            if candidate is None:
                continue
            for attr in ("v_head_dim", "head_dim"):
                val = getattr(candidate, attr, None)
                if isinstance(val, int) and val > 0:
                    return val
        raise ValueError(
            "[CafeQ] head_dim not resolvable from the model config - pass CafeQConfig(head_dim=...) explicitly"
        )

    # ── learning ───────────────────────────────────────────────────────────
    def _lse_proxy(self, W_v: torch.Tensor, W_o: torch.Tensor, M: torch.Tensor, hd: int) -> torch.Tensor:
        """Paired LogSumExp proxy (arXiv 2511.19705 Sec. 4.2).

        Per v-head h (shared M): U_h = (M^T W_v[h])^T with channel maxes over
        the head axis (input channels); V_j = M^T W_o[:, j]^T for every o input
        block j consuming head h (GQA replication) with channel maxes over the
        head axis (output channels). Smooth max (temperature t) over all.
        """
        n_vh = W_v.shape[0] // hd
        group = W_o.shape[1] // W_v.shape[0]  # o input blocks per v-head
        t = self.config.lse_temp
        chans = []
        for h in range(n_vh):
            U = (M.T @ W_v[h * hd : (h + 1) * hd]).T  # [in, hd]
            chans.append(U.abs().amax(dim=1))  # per input channel, max over head axis
            for j in range(h * group, (h + 1) * group):
                V = M.T @ W_o[:, j * hd : (j + 1) * hd].T  # [hd, out]
                chans.append(V.abs().amax(dim=0))  # per output channel, max over head axis
        allc = torch.cat(chans)
        return (t * allc).logsumexp(0) / t

    def _learn(self, W_v: torch.Tensor, W_o: torch.Tensor, hd: int):
        cfg = self.config
        gen = torch.Generator(device=W_v.device.type).manual_seed(cfg.seed)
        M0 = _random_orthogonal(hd, gen).to(W_v.dtype)
        with torch.no_grad():
            init_loss = self._lse_proxy(W_v, W_o, M0, hd).item()
        A0 = inverse_cayley(M0)
        P = (A0 / 2).clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=cfg.lr)
        for _ in range(cfg.iters):
            opt.zero_grad(set_to_none=True)
            M = cayley(P - P.T)
            loss = self._lse_proxy(W_v, W_o, M, hd)
            loss.backward()
            opt.step()
        with torch.no_grad():
            M = cayley(P - P.T)
            final_loss = self._lse_proxy(W_v, W_o, M, hd).item()
        return M.detach(), init_loss, final_loss

    # ── folding ────────────────────────────────────────────────────────────
    @staticmethod
    def _fold(v: nn.Linear, o: nn.Linear, M: torch.Tensor, hd: int) -> None:
        n_vh = v.weight.shape[0] // hd
        group = o.weight.shape[1] // v.weight.shape[0]  # GQA: o input blocks per v-head
        wv, wo = v.weight.data, o.weight.data
        for h in range(n_vh):
            wv[h * hd : (h + 1) * hd, :] = M.T @ wv[h * hd : (h + 1) * hd, :]
            for j in range(h * group, (h + 1) * group):
                wo[:, j * hd : (j + 1) * hd] = wo[:, j * hd : (j + 1) * hd] @ M  # M^{-T} == M
        if v.bias is not None:
            for h in range(n_vh):
                v.bias.data[h * hd : (h + 1) * hd] = M.T @ v.bias.data[h * hd : (h + 1) * hd]

    # ── entry point ────────────────────────────────────────────────────────
    def apply_to_model(self, model: nn.Module, **kwargs) -> nn.Module:
        cfg = self.config
        layers = self._full_attn_layers(model)
        if not layers:
            raise ValueError(
                "[CafeQ] no full-attention layers with v_proj/o_proj found - nothing to transform"
            )
        hd = cfg.head_dim or self._resolve_head_dim(model, layers)
        total_before, total_after = 0.0, 0.0
        for layer, v, o in layers:
            q = getattr(getattr(layer, "self_attn", None), "q_proj", None)
            if q is not None and not cfg.allow_gated:
                self._check_not_gated(v, o, q)
            if v.weight.shape[0] % hd != 0 or o.weight.shape[1] % hd != 0 or (
                o.weight.shape[1] % v.weight.shape[0] != 0
            ):
                raise ValueError(
                    f"[CafeQ] head_dim {hd} incompatible with v_proj {tuple(v.weight.shape)} / "
                    f"o_proj {tuple(o.weight.shape)}"
                )
            W_v = v.weight.detach().to(torch.float32)
            W_o = o.weight.detach().to(torch.float32)
            M, l0, l1 = self._learn(W_v, W_o, hd)
            self.learned_matrices[getattr(layer, "global_name", id(layer))] = M.to(v.weight.dtype)
            self._lse_history[getattr(layer, "global_name", id(layer))] = (l0, l1)
            self._fold(v, o, M.to(v.weight.dtype), hd)
            total_before += l0
            total_after += l1
        logger.info(
            "[CafeQ] learned per-head V/O transform (head_dim=%d, iters=%d, lr=%.0e): "
            "%d layers, paired LogSumExp proxy %.4f -> %.4f",
            hd,
            cfg.iters,
            cfg.lr,
            len(layers),
            total_before / max(len(layers), 1),
            total_after / max(len(layers), 1),
        )
        return model
