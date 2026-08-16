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
"""Pre-SINQ configuration."""

from __future__ import annotations

from dataclasses import dataclass

from auto_round.algorithms.transforms.base import BaseRotationConfig

PRE_SINQ_ATTRIBUTION = (
    "Pre-SINQ reparameterisation ported from the SINQ project "
    "(https://github.com/huawei-csl/SINQ, Apache-2.0, arXiv 2509.22944)."
)

__all__ = ["PreSINQConfig", "PRE_SINQ_ATTRIBUTION"]


@dataclass
class PreSINQConfig(BaseRotationConfig):
    # Weight-statistics-only transform: never requires calibration data.
    # (Transform configs default to need_calib=True in _needs_calibration_data,
    #  which would force the data-driven path - wrong for Pre-SINQ.)
    need_calib = False

    """Configuration for the Pre-SINQ function-preserving weight fold.

    Pre-SINQ computes Sinkhorn-normalised *column* scales from weight
    statistics only (calibration-free) and absorbs them into neighbouring
    weights so that any standard single-scale quantizer (RTN / OptRTN /
    GPTQ) afterwards sees better-balanced weight matrices. The transform is
    exactly function-preserving and leaves no runtime artifacts, so exported
    checkpoints load in unmodified inference engines.

    Hyperparameters use upstream's defaults; the upstream calibration-based
    hyperparameter sweep is deliberately not ported (data-free policy).

    Args:
        algorithm: registry key, always ``"presinq"``.
        group_size: tile width (input channels) for per-block sinkhorn
            scales; automatically adjusted down to a divisor of the layer's
            input width. Independent of the quantizer's group size.
        n_iter: sinkhorn iterations per tile (upstream pre-SINQ default 4).
        n_repeat: whole-model folding passes (fixed-point iteration;
            upstream default 3).
        normalize_outproj: additionally fold per-column scales between
            ``v_proj`` and ``o_proj``. Exact for multi-head attention
            (heads == kv heads); skipped with a warning for GQA.
    """

    algorithm: str = "presinq"
    group_size: int = 64
    n_iter: int = 4
    n_repeat: int = 3
    normalize_outproj: bool = False
