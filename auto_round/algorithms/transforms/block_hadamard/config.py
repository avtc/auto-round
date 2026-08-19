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
"""Block-Hadamard rotation configuration."""

from __future__ import annotations

from dataclasses import dataclass

from auto_round.algorithms.transforms.base import BaseRotationConfig

BLOCK_HADAMARD_ATTRIBUTION = (
    "Block-diagonal randomized Hadamard rotation with offline weight fusion "
    "(QuaRot-class transform, arXiv 2404.00456; block-diagonal variant as used "
    "by PeRQ, arXiv 2601.22347)."
)

__all__ = ["BlockHadamardConfig", "BLOCK_HADAMARD_ATTRIBUTION"]


@dataclass
class BlockHadamardConfig(BaseRotationConfig):
    # Weight-statistics-only transform: never requires calibration data.
    # (Transform configs default to need_calib=True in _needs_calibration_data,
    #  which would force the data-driven path - wrong for this transform.)
    need_calib = False

    """Configuration for the offline-fused block-Hadamard rotation.

    Folds a deterministic block-diagonal Hadamard matrix (per block
    ``R = D @ H @ D`` with a seeded random sign diagonal) into the model
    weights. Norm gammas are folded into consumers first, then every
    residual-stream consumer absorbs the rotation on its input channels and
    every producer on its output channels - exactly function-preserving, no
    runtime artifacts, stock export formats.

    Hybrid-attention architectures (qwen3_next / qwen3_5 GatedDeltaNet
    linear-attention layers) are covered natively via their ``in_proj_*`` /
    ``out_proj`` chains; no environment flag is required.

    Full-model (resident) mode only in this revision: the transform runs once
    over all weights before calibration caching. Layer-wise / streamed
    application is planned but not yet wired.

    Args:
        algorithm: registry key, always ``"block_hadamard"``.
        block_size: Hadamard block dimension (power of two, or a known
            Hadamard size). The rotation is applied block-diagonally to every
            weight dimension that is a multiple of ``block_size``.
            ``0`` (default) = AUTO: the largest power-of-two divisor of the
            hidden width, strictly below full width (e.g. 1024 for hidden
            5120, 2048 for hidden 4096). PeRQ Table 5: quality trends upward
            with block size up to - but NOT including - the degenerate
            full-width block (no blocks left to balance; quality collapses:
            L3-8B RTN 10.2 @2048 vs 38.0 @Full).
        seed: seed for the random sign diagonal (reproducibility).
        randomized: use ``D @ H @ D`` (randomized Hadamard, spreads sign
            structure); False uses the plain normalized Hadamard.
    """

    algorithm: str = "block_hadamard"
    block_size: int = 0
    seed: int = 42
    randomized: bool = True
