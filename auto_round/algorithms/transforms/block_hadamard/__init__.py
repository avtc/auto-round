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
"""Offline-fused block-Hadamard rotation (calibration-free).

Usage (with the RTN/OptRTN block quantizer)::

    from auto_round import AutoRound
    from auto_round.algorithms.transforms.block_hadamard import BlockHadamardConfig
    from auto_round.algorithms.quantization.rtn.config import RTNConfig

    ar = AutoRound(model, alg_configs=[BlockHadamardConfig(block_size=64),
                                       RTNConfig(group_size=32, disable_opt_rtn=False)])

The transform folds a deterministic block-diagonal randomized Hadamard matrix
into the model weights (QuaRot-class offline fusion). It is exactly
function-preserving, covers hybrid-attention architectures (qwen3_next /
qwen3_5 GatedDeltaNet linear-attention layers) natively, and leaves no
runtime artifacts: exported checkpoints are plain weights that load in
unmodified inference engines.
"""

from auto_round.algorithms.transforms.block_hadamard.config import (
    BLOCK_HADAMARD_ATTRIBUTION,
    BlockHadamardConfig,
)
from auto_round.algorithms.transforms.block_hadamard.apply import BlockHadamardRotation, build_block_rotation

__all__ = [
    "BLOCK_HADAMARD_ATTRIBUTION",
    "BlockHadamardConfig",
    "BlockHadamardRotation",
    "build_block_rotation",
]
