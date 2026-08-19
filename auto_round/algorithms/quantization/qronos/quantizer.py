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
import torch

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.qronos.config import QronosConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
from auto_round.utils import check_to_quantized

__all__ = ["QronosQuantizer", "compute_group_grid", "qronos_sequential_quantize", "spectral_norm_estimate"]

_SCALE_DTYPE = torch.float16
_Q_SCALE_THRESH = 1e-5


def spectral_norm_estimate(H: torch.Tensor, iters: int = 24) -> torch.Tensor:
    """Largest eigenvalue of a symmetric PSD matrix via power iteration.

    Avoids the O(N^3) SVD of ``torch.linalg.matrix_norm(H, 2)``; a handful of
    matvecs is enough accuracy for dampening (``lambda = alpha * sigma_1``).
    """
    v = torch.randn(H.shape[0], device=H.device, dtype=H.dtype)
    for _ in range(iters):
        v = H @ v
        v = v / v.norm().clamp(min=1e-12)
    return (v @ (H @ v)).abs().clamp(min=1e-12)


def compute_group_grid(
    W: torch.Tensor, bits: int, group_size: int, sym: bool
) -> tuple:
    """Per-group min-max grid on the fp32 weight ``W`` [n, N].

    Mirrors ``auto_round/data_type/int.py::quant_tensor_asym`` (and ``_sym``)
    conventions exactly - including the ``scale_dtype`` cast BEFORE the
    ``q_scale_thresh`` clamp and the half-to-even zero-point rounding - so the
    export path re-quantizes to the same grid the rounding loop used.

    Returns ``(scale, zp)`` of shape [n, n_groups] (``zp is None`` for sym).
    Supports group_size > 0 (pads the last partial group) and -1 (per-channel).
    """
    n, N = W.shape
    g = N if (group_size is None or group_size == -1) else int(group_size)
    if g == 0:
        raise ValueError("Qronos does not support per-tensor (group_size=0) quantization")
    if g < 0:
        raise ValueError(f"invalid group_size {group_size}")
    pad = (-N) % g
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    Wg = Wp.reshape(n, -1, g)
    wmin = torch.clamp(Wg.min(dim=-1).values, max=0)
    wmax = torch.clamp(Wg.max(dim=-1).values, min=0)
    if sym:
        maxq = float(2 ** (bits - 1))
        max_v = (2 * (wmax < -wmin).int() - 1) * torch.maximum(wmax, -wmin)
        scale = (max_v / maxq).to(_SCALE_DTYPE)
        scale = torch.where(scale < 0, torch.clamp(scale, max=-_Q_SCALE_THRESH), torch.clamp(scale, min=_Q_SCALE_THRESH))
        return scale, None
    maxq = float(2**bits) - 1
    scale = ((wmax - wmin) / maxq).to(_SCALE_DTYPE)
    scale = torch.clamp(scale, min=_Q_SCALE_THRESH)
    zp = torch.round(-wmin / scale)
    return scale, zp


def _quant_col(w: torch.Tensor, s: torch.Tensor, z, maxq: float) -> torch.Tensor:
    """Quantize one weight column to the per-column grid (RTN rounding step)."""
    if z is None:
        return s * torch.clamp(torch.round(w / s), -maxq, maxq - 1)
    return s * (torch.clamp(torch.round(w / s) + z, 0, maxq) - z)


def qronos_sequential_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    G: torch.Tensor,
    scale: torch.Tensor,
    zp,
    maxq: float,
    block_size: int = 128,
    L: torch.Tensor = None,
) -> tuple:
    """The Qronos rounding loop (Algorithm 1, arXiv 2505.11695).

    Args:
        W: fp32 weight in PERMUTED column space, [n, N].
        H: dampened Hessian ``X~^T X~`` in the same permuted space, [N, N].
        G: cross-Gram ``X~^T X`` in the same permuted space, [N, N] (``H``
            when the quantized-input cascade is unavailable).
        scale / zp: per-column grid parameters, [n, N] (zp ``None`` for sym).
        maxq: grid bound (``2^bits - 1`` asym, ``2^(bits-1)`` sym).
        block_size: blocked-update granularity (exact at any value).
        L: precomputed lower Cholesky factor of ``H^{-1}`` (optional).

    Returns:
        ``(Q, loss)``: quantize-dequantized weight [n, N] and the per-layer
        trace-loss proxy ``0.5 * sum (w - q)^2 / L_tt^2``.
    """
    n, N = W.shape
    W = W.clone()
    if L is None:
        # H^{-1} = L L^T via chol -> inverse -> chol (numerically the most
        # stable route; mirrors llm-compressor's GPTQ pipeline).
        L = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)))
    losses = torch.zeros(n, device=W.device, dtype=W.dtype)

    # ── special first column (error-correcting initialization) ─────────────
    # q_1 = Q((G_{1,>=1} w - H_{1,>=2} w_{>=2}) / H_11)          [Prop. E.2]
    num = W @ G[0, :] - W[:, 1:] @ H[0, 1:]
    q0 = _quant_col(num / H[0, 0], scale[:, 0], None if zp is None else zp[:, 0], maxq)
    W[:, 0] = q0
    losses += (num / H[0, 0] - q0) ** 2 / (L[0, 0] ** 2)
    if N > 1:
        # w_{>=2}^{(1)} = (H_{>=2,>=2})^{-1} (G_{>=2,>=1} w - H_{>=2,1} q_1)  [Lemma G.3:
        # (H_{>=2,>=2})^{-1} == L_{>=2,>=2} L_{>=2,>=2}^T for L = chol(H^{-1}),
        # so the factor is APPLIED, not solved against]
        C = L[1:, 1:]
        M = W @ G[1:, :].T - q0.unsqueeze(1) * H[1:, 0].unsqueeze(0)
        W[:, 1:] = (M @ C) @ C.T

        # ── blocked sequential loop, t = 2..N (OPTQ-style diffusion) ──────
        for i1 in range(1, N, block_size):
            i2 = min(i1 + block_size, N)
            W1 = W[:, i1:i2].clone()
            Q1 = torch.empty_like(W1)
            Err1 = torch.empty_like(W1)
            for k in range(i2 - i1):
                t = i1 + k
                w = W1[:, k]
                z = None if zp is None else zp[:, t]
                q = _quant_col(w, scale[:, t], z, maxq)
                Q1[:, k] = q
                e = (w - q) / L[t, t]
                Err1[:, k] = e
                losses += (w - q) ** 2 / (L[t, t] ** 2)
                if k + 1 < i2 - i1:
                    W1[:, k + 1 :] -= e.unsqueeze(1) * L[t + 1 : i2, t].unsqueeze(0)
            W[:, i1:i2] = Q1
            if i2 < N:
                # lower-triangular L: cross-block entries live in L[i2:, i1:i2]
                W[:, i2:] -= Err1 @ L[i2:, i1:i2].T
    return W, losses.sum() / 2

def _sanitise_input(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Zero out non-finite activations so a diverging bf16 replay cannot poison H/G.

    The quantized-input sweep replays calibration data through the already
    quantized prefix; on chaotic architectures (hybrid linear-attention) some
    blocks diverge to inf/nan in bf16. Accumulating ``x.T @ x`` on such inputs
    poisons the Hessian (observed 2026-08-19: cholesky failures across ~1/3 of
    blocks, trace-loss proxies 1e11-1e13 vs ~5e4 on healthy blocks). Dropping
    only the non-finite entries keeps the remaining tokens of the sample.
    """
    if torch.isfinite(x).all():
        return x
    n_bad = int((~torch.isfinite(x)).sum())
    logger.warning_once(
        "[Qronos] %s: %d non-finite input entries zeroed before statistics accumulation "
        "(diverging bf16 replay); the affected tokens no longer contribute to H/G.",
        getattr(module, "global_name", type(module).__name__),
        n_bad,
    )
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


_POISONED_DIAG_SPREAD = 1e12


@register_pipeline_member(QronosConfig)
class QronosQuantizer(BaseQuantizer):
    """Data-driven sequential Hessian rounding with cross-layer correction."""

    def __init__(self, config: QronosConfig) -> None:
        BaseQuantizer.__init__(self, config)

    def can_compile_block_forward(self):
        return False

    # ── statistics collection ──────────────────────────────────────────────
    @staticmethod
    def _extract_input(args):
        x = args[0] if isinstance(args, (tuple, list)) else args
        x = x[0] if isinstance(x, (tuple, list)) else x
        return x

    def register_fp_input_forward_hooks(self, block: torch.nn.Module) -> list:
        """FP reference forward: stash per-module inputs on CPU for pairing.

        The stashed tensors pair index-wise with the quantized-input forward
        (both sweeps iterate the same cached samples in the same order), which
        is what the cross-Gram ``G = X~^T X`` accumulation requires.
        """
        handles = []

        def make_hook():
            def hook(module, args, output):
                x = self._extract_input(args).detach()
                x = x.reshape(-1, x.shape[-1]).to(torch.float32)
                x = _sanitise_input(module, x)
                if not hasattr(module, "_qronos_xfp"):
                    module._qronos_xfp = []
                module._qronos_xfp.append(x.to("cpu"))

            return hook

        for _name, m in block.named_modules():
            if check_to_quantized(m):
                handles.append(m.register_forward_hook(make_hook()))
        return handles

    def register_qinput_forward_hooks(self, block: torch.nn.Module) -> list:
        """Quantized-input forward: accumulate ``H = X~^T X~`` and ``G = X~^T X``."""

        def make_hook():
            def hook(module, args, output):
                x = self._extract_input(args).detach().to(torch.float32)
                x = x.reshape(-1, x.shape[-1])
                x = _sanitise_input(module, x)
                N = x.shape[1]
                H = getattr(module, "_qronos_H", None)
                if H is None:
                    H = torch.zeros(N, N, dtype=torch.float32, device=x.device)
                H += x.T @ x
                module._qronos_H = H
                xfp = getattr(module, "_qronos_xfp", None)
                if xfp:
                    idx = getattr(module, "_qronos_idx", 0)
                    if idx < len(xfp):
                        G = getattr(module, "_qronos_G", None)
                        if G is None:
                            G = torch.zeros(N, N, dtype=torch.float32, device=x.device)
                        G += x.T @ xfp[idx].to(x.device, torch.float32)
                        module._qronos_G = G
                        module._qronos_idx = idx + 1

            return hook

        handles = []
        for _name, m in block.named_modules():
            if check_to_quantized(m):
                handles.append(m.register_forward_hook(make_hook()))
        return handles

    # ── quantization ───────────────────────────────────────────────────────
    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        valid_token_mask=None,
        **kwargs,
    ) -> dict:
        """Apply Qronos rounding to every target Linear in the block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (unused - the
                statistics are collected by the fp/q hook pair).
            input_others: Auxiliary kwargs passed to the block forward (unused).
            fp_outputs: FP reference outputs of the block (unused).
            q_inputs: Quantized inputs from the previous block (consumed by the
                hook registered for the quantized-input forward).
            block_ctx: Per-block pipeline context (BlockContext).
            valid_token_mask: unused (weights-only algorithm).
            **kwargs: Reserved for forward-compatibility.

        Returns:
            dict: Empty dict - Qronos has no tunable parameters to track.
        """
        total_loss, n_layers = 0.0, 0
        for _name, m in block.named_modules():
            if not check_to_quantized(m):
                continue
            if not hasattr(m, "_qronos_H"):
                logger.warning("[Qronos] no calibration statistics on %s - falling back to RTN", _name)
                self._quantize_layer_via_rtn(m, disable_opt_rtn=True)
                continue
            total_loss += self._qronos_quantize_layer(m)
            n_layers += 1
        logger.info(
            "[Qronos] block %s: %d layers, mean trace-loss proxy %.6f",
            getattr(block_ctx, "block_name", "?"),
            n_layers,
            total_loss / max(n_layers, 1),
        )
        return {}

    def _qronos_quantize_layer(self, m: torch.nn.Module) -> float:
        cfg = self.config
        H = m._qronos_H.to(torch.float32)
        if not torch.isfinite(H).all():
            logger.warning_once(
                "[Qronos] %s: Hessian contains non-finite entries - falling back to RTN",
                getattr(m, "global_name", "?"),
            )
            self._quantize_layer_via_rtn(m, disable_opt_rtn=True)
            return 0.0
        _diag_live = torch.diag(H)
        _diag_live = _diag_live[_diag_live > 0]
        if _diag_live.numel() and float(_diag_live.max()) / float(_diag_live.median()) > _POISONED_DIAG_SPREAD:
            logger.warning_once(
                "[Qronos] %s: Hessian diag spread exceeds %.0e (one or more exploding input "
                "channels) - guidance would be garbage, falling back to RTN",
                getattr(m, "global_name", "?"),
                _POISONED_DIAG_SPREAD,
            )
            self._quantize_layer_via_rtn(m, disable_opt_rtn=True)
            return 0.0
        G = getattr(m, "_qronos_G", None)
        if G is not None:
            G = G.to(torch.float32)
            n_pair = getattr(m, "_qronos_idx", 0)
            n_fp = len(getattr(m, "_qronos_xfp", []) or [])
            if n_pair != n_fp:
                logger.warning(
                    "[Qronos] %s: fp/quantized forward misaligned (%d vs %d batches) - dropping G, "
                    "reducing this layer to OPTQ",
                    getattr(m, "global_name", "?"),
                    n_pair,
                    n_fp,
                )
                G = None
        else:
            n_fp = len(getattr(m, "_qronos_xfp", []) or [])
            if n_fp:
                logger.warning(
                    "[Qronos] %s: no quantized-input pass collected - reducing this layer to OPTQ",
                    getattr(m, "global_name", "?"),
                )

        weight = m.weight.data
        conv1d = weight.dim() == 2 and (m.__class__.__name__ == "Conv1D")
        W = weight.detach().to(torch.float32)
        if conv1d:  # transformers Conv1D stores [in, out]; make it [out, in]
            W = W.t().contiguous()
        n, N = W.shape

        bits = getattr(m, "bits", None) or self.config.bits
        sym = getattr(m, "sym", None)
        sym = self.config.sym if sym is None else sym
        g = getattr(m, "group_size", None)
        g = self.config.group_size if g is None else g
        data_type = getattr(m, "data_type", None) or self.config.data_type or "int"
        if data_type != "int":
            raise ValueError(f"[Qronos] only data_type='int' is supported, got {data_type!r}")

        # dead columns (never-observed channels): standard GPTQ treatment
        dead = torch.diag(H) == 0
        if dead.any():
            W[:, dead] = 0
            H = H.clone()
            H[dead, dead] = 1

        # activation ordering by descending diag(H) (paper convention)
        if cfg.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
        else:
            perm = torch.arange(N, device=H.device)

        H_p = H[perm][:, perm].contiguous()
        G_p = (G if G is not None else H)[perm][:, perm].contiguous()

        # dampening: lambda = alpha * sigma_1 (bounds the condition number);
        # escalate x10 on inversion failure before giving up on Hessian guidance
        lam = cfg.dampening_alpha * spectral_norm_estimate(H_p)
        L = None
        for _attempt in range(3):
            H_d = H_p.clone()
            H_d.diagonal().add_(lam)
            try:
                L = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H_d)))
                break
            except Exception:  # pylint: disable=broad-except
                lam *= 10.0
        if L is None:
            logger.warning(
                "[Qronos] %s: Hessian inversion failed (damping escalated x1000) - "
                "falling back to plain RTN ordering",
                getattr(m, "global_name", "?"),
            )
            L = torch.eye(N, dtype=H_p.dtype, device=H_p.device)

        # grid from the ORIGINAL (dead-zeroed) weight - identical to RTN
        scale, zp = compute_group_grid(W, bits=bits, group_size=g, sym=sym)
        maxq = float(2 ** (bits - 1)) if sym else float(2**bits) - 1
        # expand to per-column and follow the permutation; group membership is
        # defined by ORIGINAL column position, so no g_idx is needed at export
        g_eff = N if (g is None or g == -1) else g
        col_group = perm // g_eff
        scale_c = scale[:, col_group].to(torch.float32)
        zp_c = None if zp is None else zp[:, col_group].to(torch.float32)

        Q_p, loss = qronos_sequential_quantize(
            W[:, perm].contiguous(), H_p, G_p, scale_c, zp_c, maxq, cfg.block_size, L=L
        )
        Q = torch.empty_like(W)
        Q[:, perm] = Q_p

        out = Q.to(weight.dtype)
        if conv1d:
            out = out.t().contiguous()
        weight.copy_(out)
        m.scale = scale.to("cpu")
        if zp is not None:
            m.zp = zp.to("cpu")
        else:
            # mirrors quant_tensor_sym's third return: the offset that shifts
            # the signed sym range into [0, 2^b - 1] for the int packer
            m.zp = int(maxq)

        for attr in ("_qronos_H", "_qronos_G", "_qronos_xfp", "_qronos_idx"):
            if hasattr(m, attr):
                delattr(m, attr)
        return loss.item()
