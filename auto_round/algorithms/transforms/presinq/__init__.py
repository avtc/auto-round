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
"""Pre-SINQ transform: calibration-free, function-preserving Sinkhorn folding.

Usage (with the RTN/OptRTN block quantizer)::

    from auto_round import AutoRound
    from auto_round.algorithms.transforms.presinq import PreSINQConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    ar = AutoRound(model, alg_configs=[PreSINQConfig(),
                                       RTNConfig(group_size=32, disable_opt_rtn=False)])

The transform folds Sinkhorn-normalised column scales into neighbouring
weights (norm gammas, SwiGLU up<->down, optionally v<->o) so any standard
quantizer afterwards sees better-balanced matrices. Exactly function
preserving; no runtime artifacts; exported checkpoints are plain weights.
"""

from auto_round.algorithms.transforms.presinq.config import PRE_SINQ_ATTRIBUTION, PreSINQConfig
from auto_round.algorithms.transforms.presinq.apply import PreSINQRotation
from auto_round.algorithms.transforms.presinq.sinkhorn import column_scales, sinkhorn_log

__all__ = ["PRE_SINQ_ATTRIBUTION", "PreSINQConfig", "PreSINQRotation", "column_scales", "sinkhorn_log"]
