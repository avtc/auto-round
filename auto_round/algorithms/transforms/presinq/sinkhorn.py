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
* **std-reuse**: each iteration's row/column stds are computed once and
  reused for both the imbalance measure and the clamped update targets
  (the original port recomputed them), and the scaled-matrix
  materialization is skipped when only the scales are needed — both changes
  are bit-exact and roughly halve the sinkhorn's time (~1.8x measured);
* on non-CPU devices the loop runs as a fused ``torch.compile`` graph by
  default (``AR_PRESINQ_BACKEND``), agreeing with the eager loop to ~1e-15
  in fp64, with a permanent fallback to the eager loop on any failure;
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

from auto_round import envs
from auto_round.logger import logger

__all__ = ["sinkhorn_log", "column_scales"]

_TILE_GROUP = 16  # tiles processed per batched call (bounds fp64 temporaries)


def _wants_compile(device_type: str) -> bool:
    """Whether the sinkhorn loop should run compiled, per ``AR_PRESINQ_BACKEND``."""
    backend = envs.AR_PRESINQ_BACKEND
    if backend == "compile":
        return True
    if backend == "auto":
        return device_type != "cpu"  # fused kernels & fewer launches help most on GPU
    return False  # "eager" (and any unrecognized value): the eager reference loop


# Compiled-loop state: lazily built compiled callable plus a permanent
# fallback latch (any compile/compile-runtime failure reverts to the eager
# loop, whose results are bit-exact with the previous release).
_compiled_body = None
_compiled_broken = False


def _get_compiled_body():
    """Build (once) the compiled sinkhorn loop; ``None`` when compilation fails."""
    global _compiled_body, _compiled_broken
    if _compiled_broken or _compiled_body is not None:
        return _compiled_body
    try:
        from auto_round.utils.device import _bump_dynamo_cache_limit

        _bump_dynamo_cache_limit(32)  # a few tile-group shapes per model
        _compiled_body = torch.compile(_sinkhorn_body, dynamic=False)
    except Exception as e:  # pragma: no cover - depends on host toolchain
        _compiled_broken = True
        logger.warning("[PreSINQ] sinkhorn torch.compile unavailable (%s); using the eager loop", e)
    return _compiled_body


def _sinkhorn_body(m, tgt_small, order, clip_min, clip_max, stop_on_increasing_imbalance):
    """The sinkhorn iteration loop (compile-friendly: no data-dependent control).

    Same candidates and updates as the eager loop in :func:`sinkhorn_log`, with
    two compile-specific hardenings (the eager loop needs neither):

    * the best-so-far tracker starts from ``+inf``, so iteration 0 always
      captures the identity candidate — eager compares it against the k=0
      imbalance with an exact ``==``, which compiled reduction-order noise
      (``cur == m`` when the log-mus are zero) would flip catastrophically via
      the gate freeze;
    * the tracker/gate comparisons use a relative slack of 1e-12: compiled
      reduction order perturbs the imbalance by ~1e-16, and an exact-equality
      tie in eager (trajectories revisit a previous minimum) must resolve the
      same way (the later candidate wins, as eager's ``<=`` does) or the
      diverged trajectory can end on a materially worse candidate. The slack
      is ~1000x above the noise and 10+ orders below any meaningful imbalance
      gap, so non-tie decisions are unaffected.
    """
    shape = m.shape
    imb_min = torch.full(m.shape[:-2], float("inf"), dtype=m.dtype, device=m.device)
    log_mu1 = torch.zeros(*shape[:-2], shape[-1], dtype=m.dtype, device=m.device)
    log_mu2 = torch.zeros(*shape[:-2], shape[-2], 1, dtype=m.dtype, device=m.device)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()
    gate = torch.zeros_like(imb_min)
    for _ in range(order):
        mu1e = log_mu1.exp()
        mu2e = log_mu2.exp()
        cur = (m / mu1e.unsqueeze(-2)) / mu2e
        s1 = cur.std(dim=-1)
        s2 = cur.std(dim=-2)
        s_min = torch.minimum(s1.amin(dim=-1), s2.amin(dim=-1)).clamp_min(1e-12)
        s_max = torch.maximum(s1.amax(dim=-1), s2.amax(dim=-1))
        ib = s_max / s_min
        better = ib <= imb_min * (1.0 + 1e-12)
        imb_min = torch.minimum(imb_min, ib)
        mu1_star = torch.where(better.unsqueeze(-1), mu1e, mu1_star)
        mu2_star = torch.where(better.unsqueeze(-1).unsqueeze(-1), mu2e, mu2_star)
        if stop_on_increasing_imbalance:
            rising = ib > imb_min * (1.0 + 1e-12)
            gate = torch.clip(gate + rising.to(m.dtype), max=1.0)
        g = 1.0 - gate
        sal_col = (s2.clamp(clip_min, clip_max) / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()
        sal_row = (s1.clamp(clip_min, clip_max) / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()
        log_mu1 = (log_mu1 + sal_col * g.unsqueeze(-1)).clip(-0.3, 10.0)
        log_mu2 = (log_mu2 + sal_row.unsqueeze(-1) * g.unsqueeze(-1).unsqueeze(-1)).clip(-0.3, 10.0)
    return mu1_star, mu2_star


def sinkhorn_log(
    matrix: torch.Tensor,
    order: int = 8,
    clip_min: float = 1e-3,
    clip_max: float = 1e3,
    eps: float = 1e-6,
    stop_on_increasing_imbalance: bool = True,
    want_scaled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sinkhorn-style std balancing; returns ``(scaled, mu1_cols, mu2_rows)``.

    Faithful float64 port of SINQ's ``sinkhorn_log``, batched over leading
    dimensions: ``matrix`` is ``[..., K, L]``; for a plain 2D input the
    results match the original routine exactly (bit-for-bit on CPU eager).
    The dual variables whose scaled matrix achieved the minimal imbalance
    (max-std / min-std ratio) are tracked and returned even if the iteration
    later diverges.

    Per iteration the row/column stds are computed once and reused for both
    the imbalance measure and the clamped update targets, and
    ``want_scaled=False`` skips the final scaled-matrix materialization when
    only the scales are needed (both are bit-exact optimizations over the
    original port). On non-CPU devices the loop can additionally run as a
    fused ``torch.compile`` graph (``AR_PRESINQ_BACKEND``), agreeing with the
    eager loop to ~1e-15 in fp64.

    Returns:
        ``(scaled, mu1, mu2)`` where ``scaled = matrix / mu1 / mu2``
        (``None`` when ``want_scaled=False``), ``mu1`` is ``[..., L]`` (column
        scales) and ``mu2`` is ``[..., K, 1]`` (row scales), float64 on the
        input device.
    """
    dtype = torch.float64
    m = matrix.to(dtype)

    def imbalance_from(s1, s2):
        s_min = torch.minimum(s1.amin(dim=-1), s2.amin(dim=-1)).clamp_min(1e-12)
        s_max = torch.maximum(s1.amax(dim=-1), s2.amax(dim=-1))
        return s_max / s_min

    # Row/column stds computed once per matrix and reused for both the k=0
    # imbalance candidate and the clamped target (the original port computed
    # the same stds three separate times).
    s1 = m.std(dim=-1)
    s2 = m.std(dim=-2)
    imb_min = imbalance_from(s1, s2)
    tgt_small = torch.minimum(s1.clamp(clip_min, clip_max).amin(-1), s2.clamp(clip_min, clip_max).amin(-1)) + eps

    body = _get_compiled_body() if _wants_compile(m.device.type) else None
    if body is not None:
        try:
            mu1_star, mu2_star = body(m, tgt_small, order, clip_min, clip_max, stop_on_increasing_imbalance)
            if want_scaled:
                return m / mu1_star.unsqueeze(-2) / mu2_star, mu1_star, mu2_star
            return None, mu1_star, mu2_star
        except Exception as e:
            global _compiled_broken
            _compiled_broken = True
            if m.is_cuda:
                torch.cuda.empty_cache()
            logger.warning("[PreSINQ] compiled sinkhorn failed (%s); using the eager loop", e)

    log_mu1 = torch.zeros(*m.shape[:-2], m.shape[-1], dtype=dtype, device=m.device)
    log_mu2 = torch.zeros(*m.shape[:-2], m.shape[-2], 1, dtype=dtype, device=m.device)

    # Candidate for step k=0 (identity scaling)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()

    gate = torch.zeros_like(imb_min)

    for _ in range(order):
        mu1e = log_mu1.exp()
        mu2e = log_mu2.exp()
        cur = (m / mu1e.unsqueeze(-2)) / mu2e
        s1 = cur.std(dim=-1)  # [..., K] per-row std
        s2 = cur.std(dim=-2)  # [..., L] per-col std
        ib = imbalance_from(s1, s2)

        better = ib <= imb_min  # [...]
        imb_min = torch.minimum(imb_min, ib)
        mu1_star = torch.where(better.unsqueeze(-1), mu1e, mu1_star)
        mu2_star = torch.where(better.unsqueeze(-1).unsqueeze(-1), mu2e, mu2_star)

        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clip(gate + rising, max=1.0)

        g = 1.0 - gate  # [...]

        std_r = s1.clamp(clip_min, clip_max)  # reuse the stds computed for the imbalance
        std_c = s2.clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()  # [..., L]
        sal_row = (std_r / tgt_small.unsqueeze(-1)).clamp(0.7, 2.0).log()  # [..., K]

        log_mu1 = (log_mu1 + sal_col * g.unsqueeze(-1)).clip(-0.3, 10.0)
        log_mu2 = (log_mu2 + sal_row.unsqueeze(-1) * g.unsqueeze(-1).unsqueeze(-1)).clip(-0.3, 10.0)

    if not want_scaled:
        return None, mu1_star, mu2_star
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

    Device policy: when every input matrix lives on the same non-CPU device it
    is used as-is (GPU weights -> GPU computation). CPU-resident or mixed
    inputs prefer a CUDA device when available (the sinkhorn batches are
    small, ~0.3 GiB fp64); the CPU is only used as a last resort.
    The result is float64 on the compute device.

    Args:
        matrices: list of 2D weight tensors sharing the same input width.
        group_size: nominal tile width (adjusted to a divisor).
        n_iter: sinkhorn iterations per tile.

    Returns:
        float64 tensor of shape ``[input_width]`` on the compute device.
    """
    device = _select_device(matrices)
    _log_device_once(device)
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
        _, mu1, _ = sinkhorn_log(batch, order=n_iter, want_scaled=False)
        parts.append(mu1.reshape(-1, block))
    t = torch.cat(parts).reshape(-1)
    return t / t.median()


def _select_device(matrices: list[torch.Tensor]) -> torch.device:
    """GPU when possible: same non-CPU device > CUDA > CPU."""
    devices = {m.device for m in matrices}
    if len(devices) == 1 and next(iter(devices)).type != "cpu":
        return next(iter(devices))
    if torch.cuda.is_available():
        # CPU-resident or mixed weights: compute on the GPU anyway - the
        # batched fp64 sinkhorn is ~50x faster there and batches are small.
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _log_device_once(device: torch.device) -> None:
    """Announce the scale-compute device once per process."""
    global _LOGGED_DEVICE
    if _LOGGED_DEVICE != str(device):
        _LOGGED_DEVICE = str(device)
        from auto_round.logger import logger

        logger.info("[PreSINQ] sinkhorn scale computation device: %s", device)


_LOGGED_DEVICE = ""


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
