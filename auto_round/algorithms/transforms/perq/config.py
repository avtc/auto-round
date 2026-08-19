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
"""PeRQ configuration."""

from __future__ import annotations

from dataclasses import dataclass

from auto_round.algorithms.transforms.base import BaseRotationConfig

PE_RQ_ATTRIBUTION = (
    "PeRQ: MassDiff channel permutation balancing per-block mass, followed by "
    "block-Hadamard rotation (arXiv 2601.22347; MassDiff algorithm clean-room "
    "from the paper - the reference implementation lives in brevitas, Apache-2.0)."
)

__all__ = ["PeRQConfig", "PE_RQ_ATTRIBUTION"]


@dataclass
class PeRQConfig(BaseRotationConfig):
    # Weight-statistics-only transform: never requires calibration data.
    # (Phase 2 will add an activation-statistics mass source behind the same
    #  config; the data-free policy keeps need_calib=False either way.)
    need_calib = False

    """Configuration for the PeRQ offline permutation + block-Hadamard fusion.

    Applies one GLOBAL orthogonal transform ``Q = blockdiag(H_b) @ P`` to the
    residual stream, fused offline into the weights: ``P`` is a channel
    permutation chosen by MassDiff so every size-*block_size* block carries
    approximately equal l1 mass (worst-case post-rotation outliers are
    governed by the maximum per-block mass), and ``H_b`` is a seeded
    randomized Hadamard per block. Exactly function-preserving; exported
    checkpoints are plain weights loadable in unmodified inference engines.

    A single global permutation is required for offline exactness (a per-layer
    permutation cannot be folded through the residual stream). Phase 1
    estimates the per-channel mass from weight statistics (aggregated over
    every hidden-stream consumer); phase 2 will optionally replace it with
    activation ``mean|x|`` statistics.

    Args:
        algorithm: registry key, always ``"perq"``.
        block_size: Hadamard block dimension; MassDiff balances mass over
            blocks of this size. ``0`` (default) = AUTO: the largest
            power-of-two divisor of the hidden width, strictly below full
            width (same rule as BlockHadamardConfig). Evidence (PeRQ Table 5):
            MassDiff beats no-permute at EVERY block size and quality trends
            upward with b up to - but NOT including - the degenerate
            full-width block; manual values in the 256-1024 band are the
            recommended sweep points on 5120-wide models.
        seed: seed for the randomized Hadamard sign diagonal.
        randomized: use ``D @ H @ D`` per block; False uses plain Hadamard.
        mass: mass-vector source. ``"weight"`` (default, calibration-free)
            aggregates per-column weight magnitudes of all stream consumers;
            ``"none"`` disables the permutation (block-Hadamard only).
    """

    algorithm: str = "perq"
    block_size: int = 0
    seed: int = 42
    randomized: bool = True
    mass: str = "weight"
