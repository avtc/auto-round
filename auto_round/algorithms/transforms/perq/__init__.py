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
"""PeRQ: MassDiff permutation + block-Hadamard rotation (offline, opt-in).

Usage (with the RTN/OptRTN block quantizer)::

    from auto_round import AutoRound
    from auto_round.algorithms.transforms.perq import PeRQConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    ar = AutoRound(model, alg_configs=[PeRQConfig(block_size=64),
                                       RTNConfig(group_size=32, disable_opt_rtn=False)])

Phase 1 (this implementation) is fully calibration-free: the per-channel
mass vector is estimated from weight statistics. The permutation is a pure
channel reorder; together with the block-Hadamard it forms one orthogonal
transform fused offline into the weights - exactly function-preserving,
stock export formats.
"""

from auto_round.algorithms.transforms.perq.config import PE_RQ_ATTRIBUTION, PeRQConfig
from auto_round.algorithms.transforms.perq.apply import PeRQRotation, massdiff_permutation

__all__ = ["PE_RQ_ATTRIBUTION", "PeRQConfig", "PeRQRotation", "massdiff_permutation"]