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
"""Sinkhorn-normalized scaling for Pre-SINQ.

This module is a port of the ``sinkhorn_log`` routine from the SINQ project
(https://github.com/huawei-csl/SINQ, Apache-2.0, arXiv 2509.22944) adapted for
auto-round: float64 math for exactness, no vmap (plain loop over column tiles)
and device-agnostic execution. The algorithm and its hyperparameter semantics
are kept faithful to the original.

The routine iteratively rescales the rows and columns of a matrix to balance
their standard deviations and returns, alongside the scaled matrix, the
best-so-far column scale vector ``mu1`` and row scale vector ``mu2``.
Pre-SINQ consumes only the *column* scales (see ``fold.py``).
"""

from __future__ import annotations

import torch

__all__ = ["sinkhorn_log"]


def sinkhorn_log(
    matrix: torch.Tensor,
    order: int = 8,
    clip_min: float = 1e-3,
    clip_max: float = 1e3,
    eps: float = 1e-6,
    stop_on_increasing_imbalance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sinkhorn-style std balancing; returns ``(scaled, mu1_cols, mu2_rows)``.

    Faithful float64 port of SINQ's ``sinkhorn_log``: the dual variables whose
    scaled matrix achieved the minimal imbalance (max-std / min-std ratio) are
    tracked and returned even if the iteration later diverges.

    Args:
        matrix: 2D weight (block) tensor, any float dtype, any device.
        order: number of iterations (Pre-SINQ upstream default: 4).
        clip_min/clip_max: clamps for the std ratios used in the updates.
        eps: numerical slack for the target scale.
        stop_on_increasing_imbalance: freeze updates once imbalance rises.

    Returns:
        ``(scaled, mu1, mu2)`` where ``scaled = matrix / mu1 / mu2``,
        ``mu1`` is ``[1, L]`` (column scales) and ``mu2`` is ``[K, 1]`` (row
        scales), both float64 on the input device.
    """
    dtype = torch.float64
    m = matrix.to(dtype)
    dev = m.device
    measure = torch.std

    def imbalance(mat: torch.Tensor) -> torch.Tensor:
        s1, s2 = measure(mat, 1), measure(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min

    imb_min = torch.tensor(float("inf"), dtype=dtype, device=dev)
    gate = torch.tensor(0.0, dtype=dtype, device=dev)

    tgt_small = (
        torch.minimum(
            m.std(1).clamp(clip_min, clip_max).min(),
            m.std(0).clamp(clip_min, clip_max).min(),
        )
        + eps
    )

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

    # Candidate for step k=0 (identity scaling)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()
    imb_min = torch.minimum(imb_min, imbalance(m))

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()
        ib = imbalance(cur)

        better = (ib <= imb_min).to(dtype)
        imb_min = torch.min(imb_min, ib)
        mu1_star = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clip(gate + rising, max=1.0)

        g = 1.0 - gate

        std_r = measure(cur, 1).clamp(clip_min, clip_max)
        std_c = measure(cur, 0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + (sal_col * g)).clip(-0.3, 10.0)
        log_mu2 = (log_mu2 + (sal_row * g)).clip(-0.3, 10.0)

    scaled = m / mu1_star / mu2_star
    return scaled, mu1_star, mu2_star


def column_scales(
    weights: list[torch.Tensor],
    group_size: int,
    n_iter: int,
) -> torch.Tensor:
    """Per-column Sinkhorn scales for the concatenated consumers of one input.

    Mirrors SINQ's ``get_sink_scale``: concatenate the weight matrices of all
    consumers along rows, split columns into tiles of ``group_size`` (adjusted
    to a divisor of the input width), run :func:`sinkhorn_log` per tile with
    ``n_iter`` iterations and median-normalise the concatenated column scales.

    Computation concatenates in float32 on the CPU (weights are moved; the
    copies are freed immediately) and each tile is upcast to float64 inside
    :func:`sinkhorn_log`, so peak memory stays near a single fp32 copy of the
    concatenated matrix plus one fp64 tile. The returned vector is float64
    CPU.

    Args:
        weights: list of 2D weight tensors sharing the same input width.
        group_size: nominal tile width (adjusted to a divisor).
        n_iter: sinkhorn iterations per tile.

    Returns:
        float64 CPU tensor of shape ``[input_width]``.
    """
    ws = [w.detach().to("cpu", torch.float32) for w in weights]
    del weights
    W = torch.cat(ws, dim=0)
    del ws
    width = W.shape[1]

    block = _find_block(width, group_size)
    n_tiles = width // block

    tiles = []
    for i in range(n_tiles):
        _, mu1, _ = sinkhorn_log(W[:, i * block : (i + 1) * block], order=n_iter)
        tiles.append(mu1.reshape(-1))
    t = torch.cat(tiles)
    return t / t.median()


def _find_block(width: int, block: int) -> int:
    """Return the largest divisor of ``width`` closest to ``block`` (>=1)."""
    if block <= 0 or width % block == 0:
        return max(block, 1)
    for i in range(block):
        up, down = block + i, block - i
        if up <= width and width % up == 0:
            return up
        if down >= 1 and width % down == 0:
            return down
    return 1
