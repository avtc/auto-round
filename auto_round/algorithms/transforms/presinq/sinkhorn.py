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
"""Sinkhorn-normalised scaling for Pre-SINQ.

Port of the ``sinkhorn_log`` routine from the SINQ project
(https://github.com/huawei-csl/SINQ, Apache-2.0, arXiv 2509.22944) adapted
for auto-round:

* float64 math for exactness;
* **batched over leading dimensions**: column tiles are processed as a
  ``[n_tiles, H, block]`` batch (in groups), so a single op set engages all
  CPU cores via torch's intra-op parallelism — or runs on the GPU when the
  weights already live there (device is inherited from the inputs);
* the algorithm and its hyperparameter semantics are faithful to the
  original scalar-per-matrix routine.

The routine iteratively rescales the rows and columns of a matrix to balance
their standard deviations and returns, alongside the scaled matrix, the
best-so-far column scale vector ``mu1`` and row scale vector ``mu2``.
Pre-SINQ consumes only the *column* scales (see ``apply.py``).
"""

from __future__ import annotations

import torch

__all__ = ["sinkhorn_log", "column_scales"]

_TILE_GROUP = 16  # tiles processed per batched call (bounds fp64 temporaries)


def sinkhorn_log(
    matrix: torch.Tensor,
    order: int = 8,
    clip_min: float = 1e-3,
    clip_max: float = 1e3,
    eps: float = 1e-6,
    stop_on_increasing_imbalance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sinkhorn-style std balancing; returns ``(scaled, mu1_cols, mu2_rows)``.

    Faithful float64 port of SINQ's ``sinkhorn_log``, batched over leading
    dimensions: ``matrix`` is ``[..., K, L]``; for a plain 2D input the
    results match the original routine exactly. The dual variables whose
    scaled matrix achieved the minimal imbalance (max-std / min-std ratio)
    are tracked and returned even if the iteration later diverges.

    Returns:
        ``(scaled, mu1, mu2)`` where ``scaled = matrix / mu1 / mu2``,
        ``mu1`` is ``[..., L]`` (column scales) and ``mu2`` is
        ``[..., K, 1]`` (row scales), float64 on the input device.
    """
    dtype = torch.float64
    m = matrix.to(dtype)

    def imbalance(mat: torch.Tensor) -> torch.Tensor:
        s1 = mat.std(dim=-1)  # [..., K] per-row std
        s2 = mat.std(dim=-2)  # [..., L] per-col std
        s_min = torch.minimum(s1.amin(dim=-1), s2.amin(dim=-1)).clamp_min(1e-12)
        s_max = torch.maximum(s1.amax(dim=-1), s2.amax(dim=-1))
        return s_max / s_min  # [...]

    imb_min = torch.full(m.shape[:-2], float("inf"), dtype=dtype, device=m.device)

    tgt_small = (
        torch.minimum(
            m.std(-1).clamp(clip_min, clip_max).amin(-1),
            m.std(-2).clamp(clip_min, clip_max).amin(-1),
        )
        + eps
    )  # [...]

    log_mu1 = torch.zeros(*m.shape[:-2], m.shape[-1], dtype=dtype, device=m.device)
    log_mu2 = torch.zeros(*m.shape[:-2], m.shape[-2], 1, dtype=dtype, device=m.device)

    # Candidate for step k=0 (identity scaling)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()
    imb_min = torch.minimum(imb_min, imbalance(m))

    gate = torch.zeros_like(imb_min)

    for _ in range(order):
        cur = (m / log_mu1.exp().unsqueeze(-2)) / log_mu2.exp()
        ib = imbalance(cur)

        better = ib <= imb_min  # [...]
        imb_min = torch.minimum(imb_min, ib)
        mu1_star = torch.where(better.unsqueeze(-1), log_mu1.exp(), mu1_star)
        mu2_star = torch.where(better.unsqueeze(-1).unsqueeze(-1), log_mu2.exp(), mu2_star)

        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clip(gate + rising, max=1.0)

        g = 1.0 - gate  # [...]

        std_r = cur.std(dim=-1).clamp(clip_min, clip_max)  # [..., K]
        std_c = cur.std(dim=-2).clamp(clip_min, clip_max)  # [..., L]

        sal_col = (std_c / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()  # [..., L]
        sal_row = (std_r / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()  # [..., K]

        log_mu1 = (log_mu1 + sal_col * g.unsqueeze(-1)).clip(-0.3, 10.0)
        log_mu2 = (log_mu2 + sal_row.unsqueeze(-1) * g.unsqueeze(-1).unsqueeze(-1)).clip(-0.3, 10.0)

    scaled = m / mu1_star.unsqueeze(-2) / mu2_star
    return scaled, mu1_star, mu2_star


def column_scales(
    matrices: list[torch.Tensor],
    group_size: int,
    n_iter: int,
) -> torch.Tensor:
    """Per-column Sinkhorn scales for the concatenated consumers of one input.

    Mirrors SINQ's ``get_sink_scale``: concatenate the (effective) weight
    matrices of all consumers along rows, split columns into tiles of
    ``group_size`` (adjusted to a divisor of the input width), run a batched
    :func:`sinkhorn_log` over tile groups and median-normalise the
    concatenated column scales.

    Device policy: when every input matrix lives on the same device it is
    used as-is (GPU weights -> GPU computation); otherwise everything is
    moved to the CPU. The result is float64 on that device.

    Args:
        matrices: list of 2D weight tensors sharing the same input width.
        group_size: nominal tile width (adjusted to a divisor).
        n_iter: sinkhorn iterations per tile.

    Returns:
        float64 tensor of shape ``[input_width]`` on the compute device.
    """
    devices = {m.device for m in matrices}
    device = next(iter(devices)) if len(devices) == 1 else torch.device("cpu")
    ws = [m.detach().to(device, torch.float32) for m in matrices]
    W = torch.cat(ws, dim=0)
    del ws
    H, width = W.shape

    block = _find_block(width, group_size)
    n_tiles = width // block
    Wt = W.view(H, n_tiles, block).permute(1, 0, 2)  # [n_tiles, H, block]
    del W

    parts = []
    for i in range(0, n_tiles, _TILE_GROUP):
        batch = Wt[i : i + _TILE_GROUP].contiguous()
        _, mu1, _ = sinkhorn_log(batch, order=n_iter)
        parts.append(mu1.reshape(-1, block))
    t = torch.cat(parts).reshape(-1)
    return t / t.median()


def _find_block(width: int, block: int) -> int:
    """Return the divisor of ``width`` closest to ``block`` (>=1)."""
    if block <= 0 or width % block == 0:
        return max(block, 1)
    for i in range(block):
        up, down = block + i, block - i
        if up <= width and width % up == 0:
            return up
        if down >= 1 and width % down == 0:
            return down
    return 1
