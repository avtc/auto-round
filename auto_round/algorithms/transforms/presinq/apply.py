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
"""Pre-SINQ: function-preserving Sinkhorn-normalised weight folding.

Port of the Pre-SINQ reparameterisation from the SINQ project
(https://github.com/huawei-csl/SINQ, Apache-2.0, arXiv 2509.22944) with
auto-round adaptations:

* float64 folds with broadcast scaling (no ``diag`` matmuls), row-chunked;
* the (1+weight) RMSNorm convention (qwen3_next / qwen3_5) is handled via the
  runtime probe reused from the SpinQuant transform;
* hybrid linear-attention layers (``linear_attn.in_proj_qkv/z/a/b``) fold on
  the input side only — the delta-rule state and internal norm do not commute
  with per-channel scaling, so ``out_proj`` is never touched;
* MoE: per-block up<->down folds (experts + shared experts) plus one scale
  vector for the shared post-attention norm pooled over all blocks; every
  consumer of that norm (routers, shared-expert gates) absorbs ``1/t``;
* MLA-style attention (``kv_a_proj_with_mqa`` / ``q_a_proj``) is folded as a
  norm consumer; unknown Linear consumers of a folded norm abort that layer's
  fold with a warning (conservative: never silently break the function).

All folds are exactly function-preserving; the transform leaves no runtime
artifacts, so checkpoints quantized afterwards export unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.presinq.config import PRE_SINQ_ATTRIBUTION, PreSINQConfig
from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    _LINEAR_ATTN_INPUT_PROJS,
    _is_one_plus_weight_norm,
    get_mlp_module,
    get_proj,
    get_router_linears,
    iter_transformer_layers,
)
from auto_round.logger import logger

__all__ = ["PreSINQRotation"]

_ROW_CHUNK = 8192  # cap fp64 temporaries regardless of matrix width

#: Direct-Linear children of self_attn that consume the post-input_layernorm
#: hidden state (MLA down-projections included).
_ATTN_INPUT_CONSUMERS = ("q_proj", "k_proj", "v_proj", "q_a_proj", "kv_a_proj_with_mqa")
#: Direct-Linear children of self_attn that do NOT consume that hidden state
#: (attention outputs / per-head secondary projections).
_ATTN_NON_CONSUMERS = ("o_proj", "q_b_proj", "kv_b_proj")


def _scale_columns_(weight: nn.Parameter, scale: torch.Tensor) -> None:
    """In-place ``weight <- weight * scale[None, :]`` in float64 (chunked)."""
    t = scale.to(weight.device, torch.float64)
    w = weight.data
    for s in range(0, w.shape[0], _ROW_CHUNK):
        block = w[s : s + _ROW_CHUNK].to(torch.float64) * t.view(1, -1)
        w[s : s + _ROW_CHUNK] = block.to(w.dtype)


def _scale_rows_(weight: nn.Parameter, scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
    """In-place ``weight <- weight * scale[:, None]`` (and its bias) in float64."""
    t = scale.to(weight.device, torch.float64)
    w = weight.data
    for s in range(0, w.shape[0], _ROW_CHUNK):
        block = w[s : s + _ROW_CHUNK].to(torch.float64) * t.view(-1, 1)
        w[s : s + _ROW_CHUNK] = block.to(w.dtype)
    if bias is not None:
        bias.data.copy_((bias.data.to(torch.float64) * t).to(bias.dtype))


def _fold_into_norm_(norm: nn.Module, scale: torch.Tensor) -> None:
    """Fold ``scale`` into the norm's effective gamma.

    Standard RMSNorm (``out = normed * w``): ``w <- w * scale``.
    (1+weight) RMSNorm (``out = normed * (1 + w)``): ``w <- (1 + w) * scale - 1``.
    """
    w = norm.weight.data
    t = scale.to(w.device, torch.float64)
    w64 = w.to(torch.float64)
    if _is_one_plus_weight_norm(norm):
        w.copy_(((1.0 + w64) * t).sub_(1.0).to(w.dtype))
    else:
        w.copy_((w64 * t).to(w.dtype))


def _linears(mod: nn.Module, names) -> list[nn.Linear]:
    out = []
    for name in names:
        lin = getattr(mod, name, None)
        if isinstance(lin, nn.Linear):
            out.append(lin)
    return out


def _norm_is_foldable(norm: Optional[nn.Module]) -> bool:
    if norm is None or not hasattr(norm, "weight"):
        return False
    if getattr(norm, "bias", None) is not None:
        logger.warning_once(
            "[PreSINQ] skipping a norm with bias (only gamma-scaling is exact); layer left unchanged."
        )
        return False
    return True


def _moe_blocks(mlp: nn.Module, experts: nn.ModuleList) -> list[nn.Module]:
    """Expert-like blocks consuming the post-attention norm: experts + shared experts."""
    blocks = list(experts)
    shared = getattr(mlp, "shared_expert", None)
    if isinstance(shared, nn.Module):
        blocks.append(shared)
    shared_list = getattr(mlp, "shared_experts", None)
    if isinstance(shared_list, nn.ModuleList):
        blocks.extend(shared_list)
    return blocks


@BaseRotation.register("presinq")
class PreSINQRotation(BaseRotation):
    """Pre-SINQ transform registered under ``"presinq"``.

    Runs ``n_repeat`` whole-model passes over the text backbone, folding
    Sinkhorn-normalised column scales between adjacent weights. See the
    package docstring for the fold patterns and their exactness arguments.
    """

    def __init__(self, config: PreSINQConfig) -> None:
        super().__init__(config)
        self._stats_log_t = 0.0
        self._stats_n = 0

    @property
    def supports_layerwise(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def apply_to_model(
        self,
        model: nn.Module,
        data_type: str = "mx_fp",
        **kwargs,
    ) -> nn.Module:
        cfg = self.config
        logger.info(PRE_SINQ_ATTRIBUTION)
        logger.info(
            "[PreSINQ] Starting (group_size=%d, n_iter=%d, n_repeat=%d, normalize_outproj=%s)",
            cfg.group_size,
            cfg.n_iter,
            cfg.n_repeat,
            cfg.normalize_outproj,
        )
        n_layers = 0
        for _rep in range(cfg.n_repeat):
            n_layers = self._fold_pass(model)
        if n_layers == 0:
            logger.warning("[PreSINQ] no transformer layers found — model left unchanged.")
        mean_log_t = self._stats_log_t / max(self._stats_n, 1)
        logger.info(
            "[PreSINQ] Done: %d layers folded per pass, %d passes, mean |log t| = %.4f.",
            n_layers,
            cfg.n_repeat,
            mean_log_t,
        )
        return model

    # ------------------------------------------------------------------
    # Per-pass logic
    # ------------------------------------------------------------------
    def _fold_pass(self, model: nn.Module) -> int:
        cfg = self.config
        n = 0
        for layer in iter_transformer_layers(model):
            self._fold_attention_inputs(layer)
            self._fold_mlp(layer)
            if cfg.normalize_outproj:
                self._fold_v_o(layer)
            n += 1
        return n

    def _track(self, t: torch.Tensor) -> None:
        self._stats_log_t += float(t.to(torch.float64).log().abs().sum())
        self._stats_n += t.numel()

    def _fold_attention_inputs(self, layer: nn.Module) -> None:
        """input_layernorm -> attention input projections (MLA included)."""
        norm = getattr(layer, "input_layernorm", None)
        if not _norm_is_foldable(norm):
            return
        attn = getattr(layer, "self_attn", None)
        linear_attn = getattr(layer, "linear_attn", None) if attn is None else None
        if attn is not None:
            for name, child in attn.named_children():
                if (
                    isinstance(child, nn.Linear)
                    and name not in _ATTN_INPUT_CONSUMERS
                    and name not in _ATTN_NON_CONSUMERS
                ):
                    logger.warning_once(
                        "[PreSINQ] unknown input_layernorm consumer %r — skipping this layer's "
                        "attention fold (known consumers: %s).",
                        name,
                        _ATTN_INPUT_CONSUMERS,
                    )
                    return
            consumers = _linears(attn, _ATTN_INPUT_CONSUMERS)
        elif isinstance(linear_attn, nn.Module):
            consumers = _linears(linear_attn, _LINEAR_ATTN_INPUT_PROJS)
        else:
            return
        if not consumers:
            return
        t = column_scales([c.weight for c in consumers], self.config.group_size, self.config.n_iter)
        self._track(t)
        _fold_into_norm_(norm, t)
        for lin in consumers:
            _scale_columns_(lin.weight, t.reciprocal())

    def _fold_mlp(self, layer: nn.Module) -> None:
        mlp = get_mlp_module(layer)
        if mlp is None:
            return
        experts = getattr(mlp, "experts", None)
        if isinstance(experts, nn.ModuleList) and len(experts) > 0:
            self._fold_moe(mlp, experts, layer)
        else:
            self._fold_dense_mlp(mlp, layer)

    def _fold_dense_mlp(self, mlp: nn.Module, layer: nn.Module) -> None:
        gate, up, down = get_proj(mlp, "gate"), get_proj(mlp, "up"), get_proj(mlp, "down")
        if gate is not None and gate is up:
            logger.warning_once(
                "[PreSINQ] fused gate_up projection detected — MLP fold skipped "
                "(fused halves cannot be located portably)."
            )
            return
        if up is not None and down is not None:
            # SwiGLU up<->down: silu(g) * (u*t) == (silu(g) * u) * t
            t = column_scales([down.weight], self.config.group_size, self.config.n_iter)
            self._track(t)
            _scale_columns_(down.weight, t.reciprocal())
            _scale_rows_(up.weight, t, bias=up.bias)
        norm = getattr(layer, "post_attention_layernorm", None)
        if _norm_is_foldable(norm) and gate is not None and up is not None:
            t = column_scales([gate.weight, up.weight], self.config.group_size, self.config.n_iter)
            self._track(t)
            _fold_into_norm_(norm, t)
            _scale_columns_(gate.weight, t.reciprocal())
            _scale_columns_(up.weight, t.reciprocal())

    def _fold_moe(self, mlp: nn.Module, experts: nn.ModuleList, layer: nn.Module) -> None:
        blocks = _moe_blocks(mlp, experts)
        # Per-block up<->down folds (independent scales per block).
        for blk in blocks:
            gate, up, down = get_proj(blk, "gate"), get_proj(blk, "up"), get_proj(blk, "down")
            if gate is not None and gate is up:
                continue  # fused gate_up inside a block — skip that block
            if up is None or down is None:
                continue
            t = column_scales([down.weight], self.config.group_size, self.config.n_iter)
            self._track(t)
            _scale_columns_(down.weight, t.reciprocal())
            _scale_rows_(up.weight, t, bias=up.bias)
        # One shared scale vector for the post-attention norm, pooled over all
        # blocks' gate+up (a single norm can only carry one vector).
        norm = getattr(layer, "post_attention_layernorm", None)
        if not _norm_is_foldable(norm):
            return
        pooled: list[torch.Tensor] = []
        for blk in blocks:
            gate, up = get_proj(blk, "gate"), get_proj(blk, "up")
            if gate is not None and gate is not up:
                pooled.append(gate.weight)
            if up is not None:
                pooled.append(up.weight)
        if not pooled:
            return
        t = column_scales(pooled, self.config.group_size, self.config.n_iter)
        self._track(t)
        _fold_into_norm_(norm, t)
        # The norm feeds the blocks AND the router(s)/shared-expert gates;
        # every consumer of the norm output absorbs 1/t. (The upstream
        # DeepSeek-Lite script skips the router — inexact; we don't.)
        # NOTE: get_router_linears already covers gate/router/shared_expert_gate.
        for router in get_router_linears(mlp):
            _scale_columns_(router.weight, t.reciprocal())
        for blk in blocks:
            gate, up = get_proj(blk, "gate"), get_proj(blk, "up")
            if gate is not None and gate is not up:
                _scale_columns_(gate.weight, t.reciprocal())
            if up is not None:
                _scale_columns_(up.weight, t.reciprocal())

    def _fold_v_o(self, layer: nn.Module) -> None:
        """Exact v<->o fold, GQA-aware.

        ``o_proj`` column ``j = h*hd + d`` (q-head ``h``, dim ``d``) reads value
        channel ``r = (h // g)*hd + d`` of ``v_proj`` (``g`` = q-heads per
        kv-head). Per-column sinkhorn scales are pooled (median) per shared
        value channel; ``v`` rows/bias scale by ``s``, ``o`` columns by
        ``1/s``. Exact for any positive ``s``: softmax(q*k) is invariant to
        value-channel scaling.

        ``head_dim`` is taken from the attention module (``head_dim`` or
        ``num_key_value_heads``); when it cannot be inferred the fold is
        skipped rather than guessed.
        """
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return
        v, o = getattr(attn, "v_proj", None), getattr(attn, "o_proj", None)
        if not isinstance(v, nn.Linear) or not isinstance(o, nn.Linear):
            return
        v_rows, o_cols = v.weight.shape[0], o.weight.shape[1]
        if o_cols % v_rows != 0:
            logger.warning_once(
                "[PreSINQ] normalize_outproj: o cols (%d) not divisible by v rows (%d); skipping.",
                o_cols,
                v_rows,
            )
            return
        g = o_cols // v_rows  # q-heads per kv-head (1 for MHA)
        hd = getattr(attn, "head_dim", None)
        if not isinstance(hd, int) or hd <= 0:
            n_kv = getattr(attn, "num_key_value_heads", None)
            hd = v_rows // n_kv if isinstance(n_kv, int) and n_kv > 0 else 0
        if hd <= 0 or v_rows % hd != 0 or (v_rows // hd) * g * hd != o_cols:
            logger.warning_once(
                "[PreSINQ] normalize_outproj: cannot infer head_dim for the exact GQA "
                "grouping; skipping this layer's v<->o fold."
            )
            return
        kv = v_rows // hd
        t = column_scales([o.weight], self.config.group_size, self.config.n_iter)  # per o column
        t_heads = t.view(kv * g, hd)  # rows = q-heads
        s = t_heads.view(kv, g, hd).median(dim=1).values  # [kv, hd] pooled per value channel
        s_flat = s.reshape(-1)  # aligned with v rows (r = b*hd + d)
        s_cols = s.repeat_interleave(g, dim=0).reshape(-1)  # aligned with o columns
        self._track(s_flat)
        _scale_columns_(o.weight, s_cols.reciprocal())
        _scale_rows_(v.weight, s_flat, bias=v.bias)
