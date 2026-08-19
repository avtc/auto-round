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
"""Offline-fused block-Hadamard rotation.

Folds a deterministic block-diagonal randomized Hadamard matrix into the
model weights so that any standard quantizer afterwards sees approximately
rotation-invariant (flatter) weight distributions. Exactly
function-preserving:

1. Every RMSNorm gamma is folded into its consumers (pure norms remain -
   RMSNorm(R x) = R RMSNorm(x) for orthogonal R).
2. Every residual-stream *consumer* absorbs the rotation on input channels
   (``W_new = W @ R``), every *producer* on output channels
   (``W_new = R @ W``); ``R`` is symmetric orthogonal
   (``R = D @ H @ D`` per block), so both sides use the same matrix.

Covers standard attention (q/k/v/o), GatedDeltaNet linear attention
(``in_proj_qkv/z/a/b`` + ``out_proj``), dense and MoE MLPs (experts, shared
experts, routers), embedding, and lm_head.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.block_hadamard.config import BlockHadamardConfig
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    _LINEAR_ATTN_INPUT_PROJS,
    _is_one_plus_weight_norm,
    dedupe_modules,
    get_hadamard_K,
    get_mlp_module,
    get_proj,
    get_router_linears,
    iter_layer_mlp_blocks,
    iter_transformer_layers,
    rotate_in_channels_,
    rotate_out_channels_,
)

logger = logging.getLogger(__name__)

_HEAD_PATHS = ("lm_head", "model.lm_head", "model.language_model.lm_head", "language_model.lm_head")
_EMBED_PATHS = (
    "model.embed_tokens",
    "embed_tokens",
    "model.language_model.embed_tokens",
    "language_model.embed_tokens",
    "model.model.embed_tokens",
)
_FINAL_NORM_PATHS = (
    "model.norm",
    "norm",
    "model.language_model.norm",
    "language_model.norm",
    "model.model.norm",
)


def build_block_rotation(
    block_size: int,
    seed: int = 42,
    randomized: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the symmetric orthogonal block rotation ``R = D @ H / sqrt(n) @ D``.

    ``H`` is the (unnormalized) Hadamard matrix of size *block_size* (Sylvester
    construction for powers of two, known matrices otherwise, via
    :func:`get_hadamard_K`). The random sign diagonal ``D`` is drawn from a
    seeded generator, making the transform deterministic and reproducible.
    """
    H, _K = get_hadamard_K(block_size)
    if H.shape[0] != block_size:
        raise ValueError(f"get_hadamard_K returned {H.shape[0]} for block_size={block_size}")
    H = H.to(device=device, dtype=torch.float64) / math.sqrt(block_size)
    if not randomized:
        return H
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    signs = (torch.randint(0, 2, (block_size,), generator=gen, dtype=torch.int64) * 2 - 1).to(
        device=H.device, dtype=torch.float64
    )
    return signs.unsqueeze(1) * H * signs.unsqueeze(0)


@BaseRotation.register("block_hadamard")
class BlockHadamardRotation(BaseRotation):
    """Block-Hadamard rotation registered under ``"block_hadamard"``.

    Applied once at model level (before calibration caching / the block
    loop). Not layer-wise capable in this revision: the rotation couples all
    residual-stream consumers, so the whole weight set must be transformed
    together (streamed application is planned separately).
    """

    def __init__(self, config: BlockHadamardConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config(cls, config) -> "BlockHadamardRotation":
        if isinstance(config, dict):
            import dataclasses

            valid_fields = {f.name for f in dataclasses.fields(BlockHadamardConfig)}
            filtered = {k: v for k, v in config.items() if k != "algorithm" and k in valid_fields}
            config = BlockHadamardConfig(**filtered)
        return cls(config)

    # ------------------------------------------------------------------
    # Model plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(model: nn.Module, paths: tuple) -> Optional[nn.Module]:
        for path in paths:
            obj = model
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                return obj
        return None

    @classmethod
    def _get_embed(cls, model) -> Optional[nn.Embedding]:
        mod = cls._resolve(model, _EMBED_PATHS)
        return mod if isinstance(mod, nn.Embedding) else None

    @classmethod
    def _get_lm_head(cls, model) -> Optional[nn.Linear]:
        mod = cls._resolve(model, _HEAD_PATHS)
        return mod if isinstance(mod, nn.Linear) else None

    @classmethod
    def _get_final_norm(cls, model) -> Optional[nn.Module]:
        return cls._resolve(model, _FINAL_NORM_PATHS)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, model, layers, embed, lm_head) -> int:
        cfg = self.config
        problems = []
        if embed is None:
            problems.append("no embed_tokens found")
        else:
            hidden = embed.weight.shape[-1]
            if hidden % cfg.block_size != 0:
                problems.append(f"hidden={hidden} not divisible by block_size={cfg.block_size}")
        if lm_head is not None and embed is not None:
            if lm_head.weight.data_ptr() == embed.weight.data_ptr():
                problems.append(
                    "lm_head.weight is tied to embed_tokens.weight; a tied head cannot "
                    "absorb both the embedding-side and head-side rotations"
                )
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                missing = [
                    p for p in ("q_proj", "k_proj", "v_proj", "o_proj") if not isinstance(getattr(attn, p, None), nn.Linear)
                ]
                if missing:
                    problems.append(f"layer {idx}: attention lacks {missing}")
            else:
                linear_attn = getattr(layer, "linear_attn", None)
                if not isinstance(linear_attn, nn.Module):
                    problems.append(f"layer {idx}: no self_attn/linear_attn")
                else:
                    has_in = any(
                        isinstance(getattr(linear_attn, p, None), nn.Linear) for p in _LINEAR_ATTN_INPUT_PROJS
                    )
                    has_out = isinstance(getattr(linear_attn, "out_proj", None), nn.Linear)
                    if not (has_in and has_out):
                        problems.append(f"layer {idx}: linear_attn lacks in_proj_*/out_proj linears")
            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, norm_name, None)
                if norm is not None and not hasattr(norm, "weight"):
                    problems.append(f"layer {idx}: {norm_name} has no weight parameter")
                if norm is not None and getattr(norm, "bias", None) is not None:
                    problems.append(f"layer {idx}: {norm_name} has a bias - not rotation-equivariant")
        if problems:
            raise ValueError(
                "[BlockHadamard] cannot apply offline block rotation to this model - "
                "the residual-stream rotation cannot be kept equivalent:\n  - "
                + "\n  - ".join(problems[:10])
                + ("\n  ..." if len(problems) > 10 else "")
            )
        return embed.weight.shape[-1] if embed is not None else 0

    # ------------------------------------------------------------------
    # Gamma folding (norms become pure before rotation)
    # ------------------------------------------------------------------
    @staticmethod
    def _gamma_and_reset(norm: nn.Module):
        """Effective per-channel gain + reset value for either norm convention."""
        w = norm.weight.data
        if _is_one_plus_weight_norm(norm):
            return w.to(torch.float64) + 1.0, 0.0
        return w.to(torch.float64), 1.0

    def _fold_linear_in(self, proj: Optional[nn.Linear], gamma_f64) -> bool:
        if not isinstance(proj, nn.Linear):
            return False
        with torch.no_grad():
            w = proj.weight.data.to(torch.float64)
            proj.weight.data = (w * gamma_f64.view(1, -1)).to(proj.weight.dtype)
        return True

    def _fold_norm_gammas(self, model, layers, lm_head) -> None:
        for layer in layers:
            # input_layernorm -> q/k/v or linear-attention input projections
            norm = getattr(layer, "input_layernorm", None)
            if norm is not None and hasattr(norm, "weight"):
                gamma, reset = self._gamma_and_reset(norm)
                fused = False
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    for proj_name in ("q_proj", "k_proj", "v_proj"):
                        fused = self._fold_linear_in(getattr(attn, proj_name, None), gamma) or fused
                else:
                    linear_attn = getattr(layer, "linear_attn", None)
                    if isinstance(linear_attn, nn.Module):
                        for proj_name in _LINEAR_ATTN_INPUT_PROJS:
                            fused = self._fold_linear_in(getattr(linear_attn, proj_name, None), gamma) or fused
                if fused:
                    norm.weight.data.fill_(reset)

            # post_attention_layernorm -> gate/up (experts, shared) + routers
            norm = getattr(layer, "post_attention_layernorm", None)
            if norm is not None and hasattr(norm, "weight"):
                gamma, reset = self._gamma_and_reset(norm)
                consumers = []
                for block, _kind in iter_layer_mlp_blocks(layer):
                    for proj_kind in ("gate", "up"):
                        proj = get_proj(block, proj_kind)
                        if proj is not None:
                            consumers.append(proj)
                mlp = get_mlp_module(layer)
                if mlp is not None:
                    consumers.extend(get_router_linears(mlp))
                consumers = dedupe_modules(consumers)
                fused = False
                for proj in consumers:
                    # NOTE: fold ALL consumers - `any(...)` would short-circuit
                    # and silently skip experts/routers after the first match.
                    fused = self._fold_linear_in(proj, gamma) or fused
                if fused:
                    norm.weight.data.fill_(reset)

        # final norm -> lm_head
        final_norm = self._get_final_norm(model)
        if final_norm is not None and hasattr(final_norm, "weight") and isinstance(lm_head, nn.Linear):
            gamma, reset = self._gamma_and_reset(final_norm)
            if self._fold_linear_in(lm_head, gamma):
                final_norm.weight.data.fill_(reset)

    # ------------------------------------------------------------------
    # Rotation fusion
    # ------------------------------------------------------------------
    def _rotate_model(self, model, layers, embed, lm_head, R: torch.Tensor, perm=None) -> int:
        """Fuse the orthogonal transform ``Q = blockdiag(R) @ P`` into weights.

        *perm* (optional list, ``P x = x[perm]``) is applied first via cheap
        index ops; the block rotation follows through the standard helpers.
        Consumers absorb ``W <- W @ Q.T`` (``W[:, perm]`` then ``W @ R.T``),
        producers ``W <- Q @ W`` (``W[perm, :]`` then ``R @ W``), the
        embedding rows likewise. For the symmetric no-permutation case this
        is byte-identical to the plain block-Hadamard fusion.
        """
        rotated: set = set()
        cfg = self.config

        device = embed.weight.device
        R_local = R.to(device)
        with torch.no_grad():
            W = embed.weight.data
            if perm is not None:
                W = W[:, perm]
            W_f64 = W.to(torch.float64)
            w_blocks = W_f64.reshape(W.shape[0], -1, cfg.block_size)
            embed.weight.data = (w_blocks @ R_local.T).reshape(W.shape).to(embed.weight.dtype)

        def _permute_in(proj: nn.Linear) -> bool:
            if proj in rotated:  # dedup BEFORE permuting (helper would skip later)
                return False
            if perm is not None:
                proj.weight.data = proj.weight.data[:, perm]
            return True

        def _permute_out(proj: nn.Linear) -> bool:
            if proj in rotated:
                return False
            if perm is not None:
                proj.weight.data = proj.weight.data[perm, :]
                if proj.bias is not None:
                    proj.bias.data = proj.bias.data[perm]
            return True

        for layer in layers:
            layer_device = next(layer.parameters()).device
            R_l = R.to(layer_device)

            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        _permute_in(proj)
                        rotate_in_channels_(proj, R_in=R_l, rotated_modules=rotated)
                o_proj = getattr(attn, "o_proj", None)
                if isinstance(o_proj, nn.Linear):
                    _permute_out(o_proj)
                    rotate_out_channels_(o_proj, R_out=R_l, rotated_modules=rotated)

            linear_attn = getattr(layer, "linear_attn", None) if attn is None else None
            if isinstance(linear_attn, nn.Module):
                for proj_name in _LINEAR_ATTN_INPUT_PROJS:
                    proj = getattr(linear_attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        _permute_in(proj)
                        rotate_in_channels_(proj, R_in=R_l, rotated_modules=rotated)
                out_proj = getattr(linear_attn, "out_proj", None)
                if isinstance(out_proj, nn.Linear):
                    _permute_out(out_proj)
                    rotate_out_channels_(out_proj, R_out=R_l, rotated_modules=rotated)

            for block, _kind in iter_layer_mlp_blocks(layer):
                gate = get_proj(block, "gate")
                if gate is not None:
                    _permute_in(gate)
                    rotate_in_channels_(gate, R_in=R_l, rotated_modules=rotated)
                up = get_proj(block, "up")
                if up is not None:
                    _permute_in(up)
                    rotate_in_channels_(up, R_in=R_l, rotated_modules=rotated)
                down = get_proj(block, "down")
                if down is not None:
                    _permute_out(down)
                    rotate_out_channels_(down, R_out=R_l, rotated_modules=rotated)

            mlp = get_mlp_module(layer)
            if mlp is not None:
                for router in get_router_linears(mlp):
                    _permute_in(router)
                    rotate_in_channels_(router, R_in=R_l, rotated_modules=rotated)

        if isinstance(lm_head, nn.Linear):
            _permute_in(lm_head)
            rotate_in_channels_(lm_head, R_in=R.to(lm_head.weight.device), rotated_modules=rotated)

        return len(rotated) + 1  # +1 for the embedding

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def apply_to_model(self, model: nn.Module, **kwargs) -> nn.Module:
        cfg = self.config
        if cfg.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {cfg.block_size}")

        embed = self._get_embed(model)
        lm_head = self._get_lm_head(model)
        layers = list(iter_transformer_layers(model))
        if not layers:
            raise ValueError("[BlockHadamard] no decoder layers found - unsupported model structure")
        self._validate(model, layers, embed, lm_head)

        R = build_block_rotation(cfg.block_size, seed=cfg.seed, randomized=cfg.randomized)
        logger.info(
            "[BlockHadamard] applying offline-fused block-diagonal Hadamard "
            "(block_size=%d, randomized=%s, seed=%d) to %d layers.",
            cfg.block_size,
            cfg.randomized,
            cfg.seed,
            len(layers),
        )

        self._fold_norm_gammas(model, layers, lm_head)
        n_rotated = self._rotate_model(model, layers, embed, lm_head, R)

        self._warn_unrotated_stream_consumers(layers, embed, rotated_hint_hidden=embed.weight.shape[-1])
        logger.info(
            "[BlockHadamard] done: %d linear modules rotated (+ embedding); model remains "
            "function-preserving, exports stay stock.",
            n_rotated,
        )
        return model

    def _warn_unrotated_stream_consumers(self, layers, embed, rotated_hint_hidden: int) -> None:
        """Best-effort safety net: flag unrotated linears reading the residual stream.

        Any ``nn.Linear`` inside a decoder layer whose input width equals the
        hidden size but which was not covered by the known projection names
        could silently break the rotation equivalence on new architectures.
        Routers and known projections are always rotated, so they never appear.
        """
        hidden = embed.weight.shape[-1] if embed is not None else rotated_hint_hidden
        warned = 0
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            known = set()
            if attn is not None:
                known.update(id(getattr(attn, p, None)) for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
            else:
                la = getattr(layer, "linear_attn", None)
                if isinstance(la, nn.Module):
                    known.update(id(getattr(la, p, None)) for p in _LINEAR_ATTN_INPUT_PROJS)
                    known.add(id(getattr(la, "out_proj", None)))
            for block, _kind in iter_layer_mlp_blocks(layer):
                for kind in ("gate", "up", "down"):
                    proj = get_proj(block, kind)
                    if proj is not None:
                        known.add(id(proj))
            mlp = get_mlp_module(layer)
            if mlp is not None:
                known.update(id(r) for r in get_router_linears(mlp))
            for name, mod in layer.named_modules():
                if not isinstance(mod, nn.Linear) or id(mod) in known:
                    continue
                if mod.weight.shape[-1] == hidden:
                    logger.warning(
                        "[BlockHadamard] layer %d: linear %r reads the hidden stream but is not a "
                        "known projection - verify the rotation stays exact on this architecture.",
                        idx,
                        name,
                    )
                    warned += 1
        if warned:
            logger.warning(
                "[BlockHadamard] %d unrecognised hidden-stream linears detected; treat the "
                "function-preserving guarantee as unverified for this model.",
                warned,
            )
