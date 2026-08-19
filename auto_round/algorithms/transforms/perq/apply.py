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
"""PeRQ transform: MassDiff permutation + block-Hadamard offline fusion.

Reuses the block-Hadamard fusion machinery (gamma folding, per-module
in/out-channel rotation, validation) from
:class:`~auto_round.algorithms.transforms.block_hadamard.BlockHadamardRotation`;
only the mass vector and the permutation stage are PeRQ-specific.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.block_hadamard.apply import (
    BlockHadamardRotation,
    build_block_rotation,
    resolve_auto_block_size,
)
from auto_round.algorithms.transforms.perq.config import PeRQConfig
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    _LINEAR_ATTN_INPUT_PROJS,
    get_mlp_module,
    get_proj,
    get_router_linears,
    iter_layer_mlp_blocks,
    iter_transformer_layers,
)

logger = logging.getLogger(__name__)


def massdiff_permutation(mass: torch.Tensor, block_size: int) -> List[int]:
    """Greedy MassDiff assignment (PeRQ default calibration algorithm).

    Channels are processed in descending mass order and each is assigned to
    the block with the least accumulated mass (capacity *block_size*), which
    equalises the per-block l1 mass - the quantity that governs worst-case
    post-rotation outliers. Deterministic; ties resolved by channel index.

    Args:
        mass: per-channel mass vector (non-negative), length divisible by
            *block_size*.
        block_size: Hadamard block size / block capacity.

    Returns:
        Permutation ``perm`` such that ``P x = x[perm]`` places every channel
        into its assigned block (blocks emitted in index order, each keeping
        its descending-mass insertion order).
    """
    n = mass.numel()
    if n % block_size != 0:
        raise ValueError(f"mass length {n} not divisible by block_size {block_size}")
    n_blocks = n // block_size
    mass_f64 = mass.to(torch.float64)
    order = torch.argsort(mass_f64, descending=True, stable=True).tolist()

    block_mass = [0.0] * n_blocks
    block_fill = [0] * n_blocks
    block_contents: List[List[int]] = [[] for _ in range(n_blocks)]

    for ch in order:
        # pick the least-mass block with remaining capacity (ties: lowest index)
        best = -1
        best_mass = None
        for b in range(n_blocks):
            if block_fill[b] >= block_size:
                continue
            m = block_mass[b]
            if best_mass is None or m < best_mass - 1e-12 or (abs(m - best_mass) <= 1e-12 and b < best):
                best = b
                best_mass = m
        block_contents[best].append(ch)
        block_mass[best] += float(mass_f64[ch])
        block_fill[best] += 1

    perm: List[int] = []
    for contents in block_contents:
        perm.extend(contents)
    return perm


@BaseRotation.register("perq")
class PeRQRotation(BlockHadamardRotation):
    """PeRQ rotation registered under ``"perq"``.

    One global ``Q = blockdiag(H_b) @ P`` fused offline; inherits validation,
    gamma folding and the fusion loop from ``BlockHadamardRotation``.
    """

    def __init__(self, config: PeRQConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config(cls, config) -> "PeRQRotation":
        if isinstance(config, dict):
            import dataclasses

            valid_fields = {f.name for f in dataclasses.fields(PeRQConfig)}
            filtered = {k: v for k, v in config.items() if k != "algorithm" and k in valid_fields}
            config = PeRQConfig(**filtered)
        return cls(config)

    # ------------------------------------------------------------------
    # Mass vector (phase 1: weight statistics)
    # ------------------------------------------------------------------
    def _collect_consumer_columns(self, layers, lm_head):
        """Yield ``(module, is_input_side)`` for every hidden-stream consumer."""
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        yield proj
            else:
                linear_attn = getattr(layer, "linear_attn", None)
                if isinstance(linear_attn, nn.Module):
                    for proj_name in _LINEAR_ATTN_INPUT_PROJS:
                        proj = getattr(linear_attn, proj_name, None)
                        if isinstance(proj, nn.Linear):
                            yield proj
            for block, _kind in iter_layer_mlp_blocks(layer):
                for kind in ("gate", "up"):
                    proj = get_proj(block, kind)
                    if proj is not None:
                        yield proj
            mlp = get_mlp_module(layer)
            if mlp is not None:
                for router in get_router_linears(mlp):
                    yield router
        if isinstance(lm_head, nn.Linear):
            yield lm_head

    def _mass_weight_proxy(self, layers, lm_head) -> torch.Tensor:
        """Per-channel mass from consumer weight columns (calibration-free).

        For every hidden-stream consumer ``W``, the squared column norms
        ``sum_r W[r, c]^2`` are accumulated in float64 and square-rooted at
        the end: a scale-free aggregate of how strongly channel *c* drives
        the following computation. Consumers with duplicate identity (MoE
        router aliasing) are counted once.
        """
        acc = None
        seen = set()
        for proj in self._collect_consumer_columns(layers, lm_head):
            if id(proj) in seen:
                continue
            seen.add(id(proj))
            cols = proj.weight.data.to(torch.float64).pow(2).sum(dim=0)
            acc = cols if acc is None else acc + cols
        if acc is None:
            raise ValueError("[PeRQ] no stream consumers found for the weight mass proxy")
        return acc.sqrt()

    def _mass_acts(self, model, layers, lm_head) -> torch.Tensor:
        """Per-channel mass from activation statistics in an imatrix dump.

        Loads ``{"imatrix": {module_name: E[x^2] per input channel}}`` (as
        produced by ``dump_imatrix.py``) and aggregates the per-channel RMS
        ``sqrt(E[x^2])`` over every hidden-stream consumer found in the dump.
        All consumers observe the same (norm-scaled) residual stream, so the
        aggregate is a sound global mass vector. Module names are matched by
        exact name or unique suffix (VLM wrappers rename the backbone with a
        ``language_model.`` prefix that canonical dump keys lack).
        """
        import os

        cfg = self.config
        if not cfg.imatrix_path:
            raise ValueError('[PeRQ] mass="acts" requires imatrix_path pointing at an imatrix dump')
        if not os.path.isfile(cfg.imatrix_path):
            raise FileNotFoundError(f"[PeRQ] imatrix dump not found: {cfg.imatrix_path}")
        payload = torch.load(cfg.imatrix_path, map_location="cpu")
        stats = payload.get("imatrix", payload) if isinstance(payload, dict) else None
        if not isinstance(stats, dict) or not stats:
            raise ValueError(f"[PeRQ] {cfg.imatrix_path} does not contain an imatrix dict")

        names = {id(mod): name for name, mod in model.named_modules()}
        keys = list(stats.keys())

        def match(name: str):
            if name and name in stats:
                return name
            cands = [k for k in keys if k and name and (name.endswith(k) or k.endswith(name))]
            if len(cands) == 1:
                return cands[0]
            return None  # missing or ambiguous

        acc = None
        n_found, n_missing = 0, 0
        seen = set()
        for proj in self._collect_consumer_columns(layers, lm_head):
            if id(proj) in seen:
                continue
            seen.add(id(proj))
            key = match(names.get(id(proj), ""))
            if key is None:
                n_missing += 1
                continue
            vec = stats[key]
            if isinstance(vec, (tuple, list)) and len(vec) == 2:  # (sums, counts) fallback
                sums, cnt = vec
                vec = sums / cnt.clamp(min=1)
            vec = vec.to(torch.float64)
            if acc is None:
                acc = vec.clamp(min=0).sqrt()
            else:
                acc = acc + vec.clamp(min=0).sqrt()
            n_found += 1
        if n_found == 0:
            raise ValueError(
                f"[PeRQ] no stream consumers matched the imatrix dump "
                f"({cfg.imatrix_path}); check module naming"
            )
        if n_missing:
            logger.warning(
                "[PeRQ] %d/%d stream consumers missing from the imatrix dump "
                "(aggregated from the remaining %d).",
                n_missing,
                n_found + n_missing,
                n_found,
            )
        return acc / n_found

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def apply_to_model(self, model: nn.Module, **kwargs) -> nn.Module:
        cfg = self.config
        if cfg.mass not in ("weight", "none", "acts"):
            raise ValueError(f"unknown mass source {cfg.mass!r} (expected 'weight', 'acts' or 'none')")

        embed = self._get_embed(model)
        lm_head = self._get_lm_head(model)
        layers = list(iter_transformer_layers(model))
        if not layers:
            raise ValueError("[PeRQ] no decoder layers found - unsupported model structure")
        if embed is not None and cfg.block_size == 0:
            cfg.block_size = resolve_auto_block_size(embed.weight.shape[-1])
            logger.info(
                "[PeRQ] block_size=0 (auto) resolved to %d for hidden=%d.",
                cfg.block_size,
                embed.weight.shape[-1],
            )
        if cfg.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {cfg.block_size}")
        self._validate(model, layers, embed, lm_head)
        hidden = embed.weight.shape[-1]

        perm: Optional[List[int]] = None
        if cfg.mass != "none":
            if cfg.mass == "acts":
                mass = self._mass_acts(model, layers, lm_head)
            else:
                mass = self._mass_weight_proxy(layers, lm_head)
            perm = massdiff_permutation(mass, cfg.block_size)
            block_masses = mass.to(torch.float64)[torch.tensor(perm)].reshape(-1, cfg.block_size).sum(dim=1)
            logger.info(
                "[PeRQ] MassDiff permutation from %s statistics: %d channels / %d blocks, "
                "block l1 mass min/mean/max = %.4f/%.4f/%.4f (balance ratio %.3f).",
                "activation" if cfg.mass == "acts" else "weight",
                hidden,
                hidden // cfg.block_size,
                block_masses.min().item(),
                block_masses.mean().item(),
                block_masses.max().item(),
                (block_masses.max() / block_masses.mean()).item(),
            )

        R = build_block_rotation(cfg.block_size, seed=cfg.seed, randomized=cfg.randomized)
        logger.info(
            "[PeRQ] applying offline-fused permutation + block-Hadamard "
            "(block_size=%d, permutation=%s, randomized=%s, seed=%d) to %d layers.",
            cfg.block_size,
            "on" if perm is not None else "off",
            cfg.randomized,
            cfg.seed,
            len(layers),
        )

        self._fold_norm_gammas(model, layers, lm_head)
        n_rotated = self._rotate_model(model, layers, embed, lm_head, R, perm=perm)

        logger.info(
            "[PeRQ] done: %d linear modules rotated (+ embedding); model remains "
            "function-preserving, exports stay stock.",
            n_rotated,
        )
        return model
