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

* **single-writeback shadow arithmetic**: folds accumulate float64 scale
  vectors per weight and write back to the original dtype exactly once after
  all passes, so repeated passes do not compound bf16 rounding noise;
* sinkhorn scale computation runs batched (all CPU cores) and on the GPU
  when the weights already live there;
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

import time

import torch
import torch.nn as nn
import tqdm

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
import threading

from auto_round.logger import logger

__all__ = ["PreSINQRotation"]

#: Direct-Linear children of self_attn that consume the post-input_layernorm
#: hidden state (MLA down-projections included).
_ATTN_INPUT_CONSUMERS = ("q_proj", "k_proj", "v_proj", "q_a_proj", "kv_a_proj_with_mqa")
#: Direct-Linear children of self_attn that do NOT consume that hidden state
#: (attention outputs / per-head secondary projections).
_ATTN_NON_CONSUMERS = ("o_proj", "q_b_proj", "kv_b_proj")


class _WeightState:
    """Cumulative float64 scales for one weight (lazy, device-local)."""

    __slots__ = ("col", "row", "bias", "one_plus")

    def __init__(self):
        self.col: torch.Tensor | None = None  # [in_features]
        self.row: torch.Tensor | None = None  # [out_features]
        self.bias: nn.Parameter | None = None  # bias shares the row scale
        self.one_plus: bool | None = None  # norm convention (norm weights only)


def _linears(mod: nn.Module, names) -> list[nn.Linear]:
    out = []
    for name in names:
        lin = getattr(mod, name, None)
        if isinstance(lin, nn.Linear):
            out.append(lin)
    return out


def _norm_is_foldable(norm) -> bool:
    if norm is None or not hasattr(norm, "weight"):
        return False
    if getattr(norm, "bias", None) is not None:
        logger.warning_once("[PreSINQ] skipping a norm with bias (only gamma-scaling is exact); layer left unchanged.")
        return False
    return True


def _moe_blocks(mlp: nn.Module, experts: nn.ModuleList) -> list[nn.Module]:
    """Expert-like blocks consuming the post-attention norm: experts + shared experts."""
    blocks = list(experts)
    for name in ("shared_expert", "shared_experts", "shared_mlp"):
        shared = getattr(mlp, name, None)
        if isinstance(shared, nn.Module) and not isinstance(shared, nn.ModuleList):
            blocks.append(shared)
            continue
        if isinstance(shared, nn.ModuleList):
            blocks.extend(shared)
    return blocks


@BaseRotation.register("presinq")
class PreSINQRotation(BaseRotation):
    """Pre-SINQ transform registered under ``"presinq"``.

    Runs ``n_repeat`` whole-model passes over the text backbone, folding
    Sinkhorn-normalised column scales between adjacent weights. Scales
    accumulate in float64 and are written back to the weights' dtype exactly
    once at the end of the final pass. See the package docstring for the fold
    patterns and their exactness arguments.
    """

    def __init__(self, config: PreSINQConfig) -> None:
        super().__init__(config)
        self._states: dict = {}  # Parameter -> _WeightState (identity keys)
        self._state_lock = threading.Lock()  # fan-out folds touch _states/_stats concurrently
        self._stats_log_t = 0.0
        self._stats_n = 0

    @property
    def supports_layerwise(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Layer-wise protocol (models too large for full residency)
    # ------------------------------------------------------------------
    # Valid here because every Pre-SINQ fold is strictly layer-local (norm <->
    # its consumers, up<->down, v<->o, MoE blocks/routers all live inside one
    # decoder layer) and function-preserving on its own — unlike offline
    # SpinQuant R1, which rotates the shared inter-layer residual stream and
    # therefore cannot be applied per-layer.
    def prepare_layerwise(
        self,
        model: nn.Module,
        data_type: str = "mx_fp",
        **kwargs,
    ) -> "PreSINQRotation":
        """Init per-run state; no weights are touched (base-class contract)."""
        self._stats_log_t, self._stats_n = 0.0, 0
        self._states.clear()
        cfg = self.config
        logger.info(
            "[PreSINQ] Layer-wise mode armed (group_size=%d, n_iter=%d, n_repeat=%d, "
            "normalize_outproj=%s); folds will be applied per loaded layer.",
            cfg.group_size,
            cfg.n_iter,
            cfg.n_repeat,
            cfg.normalize_outproj,
        )
        return self

    def rotate_layer(self, layer: nn.Module, layer_idx: int, **kwargs) -> None:
        """Fold one decoder layer (n_repeat local passes) and flush its scales.

        Runs before reference-output collection, so the block quantizer sees
        the folded weights. Refolding an already-folded layer remains
        function-preserving (any positive scales are exact). Per-layer state
        is flushed here, keeping peak bookkeeping at one layer's worth.
        """
        cfg = self.config
        for _ in range(cfg.n_repeat):
            self._fold_attention_inputs(layer)
            self._fold_mlp(layer)
            if cfg.normalize_outproj:
                self._fold_v_o(layer)
        self._finalize()
        self._states.clear()

    def finalize_layerwise(self, model: nn.Module) -> None:
        """Post-loop summary; nothing to clean up (no global state)."""
        mean_log_t = self._stats_log_t / max(self._stats_n, 1)
        logger.info("[PreSINQ] Layer-wise pass complete (all layers), mean |log t| = %.4f.", mean_log_t)

    # ------------------------------------------------------------------
    # Shadow-scale registry
    # ------------------------------------------------------------------
    def _state(self, weight: nn.Parameter) -> _WeightState:
        with self._state_lock:
            return self._state_locked(weight)

    def _state_locked(self, weight: nn.Parameter) -> _WeightState:
        st = self._states.get(weight)
        if st is None:
            st = _WeightState()
            self._states[weight] = st
        return st

    def _effective(self, weight: nn.Parameter, dtype=torch.float32) -> torch.Tensor:
        """Current effective weight = original * cumulative scales (fp64 math)."""
        st = self._state(weight)
        w = weight.data.to(torch.float64)
        if st.col is not None:
            w = w * st.col.view(1, -1)
        if st.row is not None:
            w = w * st.row.view(-1, 1)
        return w.to(dtype)

    def _apply_col(self, weight: nn.Parameter, t: torch.Tensor) -> None:
        st = self._state(weight)
        t = t.to(weight.device, torch.float64)
        st.col = t if st.col is None else st.col * t

    def _apply_row(self, weight: nn.Parameter, t: torch.Tensor, bias: nn.Parameter = None) -> None:
        st = self._state(weight)
        t = t.to(weight.device, torch.float64)
        st.row = t if st.row is None else st.row * t
        if bias is not None:
            st.bias = bias

    def _apply_norm_fold(self, norm: nn.Module, t: torch.Tensor) -> None:
        st = self._state(norm.weight)
        if st.one_plus is None:
            st.one_plus = _is_one_plus_weight_norm(norm)
        t = t.to(norm.weight.device, torch.float64)
        st.col = t if st.col is None else st.col * t

    def _finalize(self) -> None:
        """Single dtype write-back of all accumulated scales."""
        for weight, st in self._states.items():
            if st.one_plus is None:
                # regular weight matrix: col scales columns, row scales rows (+bias)
                if st.col is None and st.row is None:
                    continue
                w = weight.data.to(torch.float64)
                if st.col is not None:
                    w = w * st.col.view(1, -1)
                if st.row is not None:
                    w = w * st.row.view(-1, 1)
                weight.data.copy_(w.to(weight.dtype))
                if st.bias is not None and st.row is not None:
                    st.bias.data.copy_((st.bias.data.to(torch.float64) * st.row).to(st.bias.dtype))
            else:
                # norm weight: 1-D gamma; st.col is elementwise
                if st.col is None:
                    continue
                w64 = weight.data.to(torch.float64)
                if st.one_plus:
                    gamma = (1.0 + w64) * st.col
                    weight.data.copy_(gamma.sub_(1.0).to(weight.dtype))
                else:
                    weight.data.copy_((w64 * st.col).to(weight.dtype))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def apply_to_model(
        self,
        model: nn.Module,
        data_type: str = "mx_fp",
        use_tqdm: bool = True,
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
        t_start = time.time()
        try:
            for _rep in range(cfg.n_repeat):
                n_layers = self._fold_pass(model, _rep + 1, use_tqdm)
            self._finalize()
        finally:
            self._states.clear()
        if n_layers == 0:
            logger.warning("[PreSINQ] no transformer layers found — model left unchanged.")
        mean_log_t = self._stats_log_t / max(self._stats_n, 1)
        logger.info(
            "[PreSINQ] Done: %d layers x %d passes in %.1fs, mean |log t| = %.4f (single fp64 writeback).",
            n_layers,
            cfg.n_repeat,
            time.time() - t_start,
            mean_log_t,
        )
        return model

    # ------------------------------------------------------------------
    # Per-pass logic
    # ------------------------------------------------------------------
    def _fold_pass(self, model: nn.Module, pass_idx: int, use_tqdm: bool = True) -> int:
        cfg = self.config
        layers = list(iter_transformer_layers(model))
        t0 = time.time()
        n = 0
        desc = f"[PreSINQ] pass {pass_idx}/{cfg.n_repeat}"
        for layer in tqdm.tqdm(layers, desc=desc, disable=not use_tqdm):
            self._fold_attention_inputs(layer)
            self._fold_mlp(layer)
            if cfg.normalize_outproj:
                self._fold_v_o(layer)
            n += 1
        logger.info("[PreSINQ] pass %d/%d done: %d layers in %.1fs", pass_idx, cfg.n_repeat, n, time.time() - t0)
        return n

    def _track(self, t: torch.Tensor) -> None:
        s = float(t.to(torch.float64).log().abs().sum())
        with self._state_lock:
            self._stats_log_t += s
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
        t = column_scales([self._effective(c.weight) for c in consumers], self.config.group_size, self.config.n_iter)
        self._track(t)
        self._apply_norm_fold(norm, t)
        for lin in consumers:
            self._apply_col(lin.weight, t.reciprocal())

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
            t = column_scales([self._effective(down.weight)], self.config.group_size, self.config.n_iter)
            self._track(t)
            self._apply_col(down.weight, t.reciprocal())
            self._apply_row(up.weight, t, bias=up.bias)
        norm = getattr(layer, "post_attention_layernorm", None)
        if _norm_is_foldable(norm) and gate is not None and up is not None:
            t = column_scales(
                [self._effective(gate.weight), self._effective(up.weight)], self.config.group_size, self.config.n_iter
            )
            self._track(t)
            self._apply_norm_fold(norm, t)
            self._apply_col(gate.weight, t.reciprocal())
            self._apply_col(up.weight, t.reciprocal())

    def _fold_moe(self, mlp: nn.Module, experts: nn.ModuleList, layer: nn.Module) -> None:
        blocks = _moe_blocks(mlp, experts)
        # Per-block up<->down folds (independent scales per block): each
        # block's sinkhorn is independent, so the loop fans out over GPUs when
        # several are visible (MoE layers carry hundreds of small experts).
        self._fold_expert_blocks(blocks)
        # One shared scale vector for the post-attention norm, pooled over all
        # blocks' gate+up (a single norm can only carry one vector).
        norm = getattr(layer, "post_attention_layernorm", None)
        if not _norm_is_foldable(norm):
            return
        pooled = []
        for blk in blocks:
            gate, up = get_proj(blk, "gate"), get_proj(blk, "up")
            if gate is not None and gate is not up:
                pooled.append(self._effective(gate.weight))
            if up is not None:
                pooled.append(self._effective(up.weight))
        if not pooled:
            return
        t = column_scales(pooled, self.config.group_size, self.config.n_iter)
        self._track(t)
        self._apply_norm_fold(norm, t)
        # The norm feeds the blocks AND the router(s)/shared-expert gates;
        # every consumer of the norm output absorbs 1/t. (The upstream
        # DeepSeek-Lite script skips the router — inexact; we don't.)
        # NOTE: get_router_linears already covers gate/router/shared_expert_gate.
        for router in get_router_linears(mlp):
            self._apply_col(router.weight, t.reciprocal())
        for blk in blocks:
            gate, up = get_proj(blk, "gate"), get_proj(blk, "up")
            if gate is not None and gate is not up:
                self._apply_col(gate.weight, t.reciprocal())
            if up is not None:
                self._apply_col(up.weight, t.reciprocal())

    def _fold_expert_blocks(self, blocks: list) -> None:
        """Fold every expert block's up<->down pair, in parallel over GPUs.

        On models with many experts per layer (hundreds of small projections)
        the per-block sinkhorn dominates the fold time; the blocks are
        independent, so they are distributed round-robin across CUDA devices.
        Per-block results are identical to the serial loop.
        """
        devices = self._fold_devices(len(blocks))
        if devices is None:
            for blk in blocks:
                self._fold_one_expert_block(blk)
            return

        from concurrent.futures import ThreadPoolExecutor

        logger.info(
            "[PreSINQ] expert fold fan-out: %d blocks across %d GPUs (round-robin).",
            len(blocks),
            len(devices),
        )
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(self._fold_one_expert_block, blk, device=devices[i % len(devices)])
                for i, blk in enumerate(blocks)
            ]
            for f in futures:
                f.result()  # propagate the first worker exception

    def _fold_devices(self, n_blocks: int):
        """Worker devices for the expert fold: CUDA round-robin or None (serial)."""
        parallel_folds = getattr(self.config, "parallel_folds", None)
        n_gpu = torch.cuda.device_count()
        if parallel_folds is None:
            parallel_folds = n_gpu > 1
        if not parallel_folds or n_gpu <= 1 or n_blocks < 4:
            return None
        return [f"cuda:{i}" for i in range(min(n_gpu, n_blocks))]

    def _fold_one_expert_block(self, blk: nn.Module, device=None) -> None:
        gate, up, down = get_proj(blk, "gate"), get_proj(blk, "up"), get_proj(blk, "down")
        if gate is not None and gate is up:
            return  # fused gate_up inside a block — skip that block
        if up is None or down is None:
            return
        if device is not None:
            # compute the sinkhorn on the worker's device; the scale vector is
            # tiny, so moving it back before the writebacks is free
            t = column_scales([self._effective(down.weight).to(device)], self.config.group_size, self.config.n_iter).to(
                down.weight.device
            )
        else:
            t = column_scales([self._effective(down.weight)], self.config.group_size, self.config.n_iter)
        self._track(t)
        self._apply_col(down.weight, t.reciprocal())
        self._apply_row(up.weight, t, bias=up.bias)

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
        t = column_scales([self._effective(o.weight)], self.config.group_size, self.config.n_iter)  # per o column
        t_heads = t.view(kv * g, hd)  # rows = q-heads
        s = t_heads.view(kv, g, hd).median(dim=1).values  # [kv, hd] pooled per value channel
        s_flat = s.reshape(-1)  # aligned with v rows (r = b*hd + d)
        s_cols = s.repeat_interleave(g, dim=0).reshape(-1)  # aligned with o columns
        self._track(s_flat)
        self._apply_col(o.weight, s_cols.reciprocal())
        self._apply_row(v.weight, s_flat, bias=v.bias)
