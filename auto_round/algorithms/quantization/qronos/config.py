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
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.logger import logger


class QronosConfig(QuantizationConfig):
    """Qronos sequential Hessian rounding (clean-room, arXiv 2505.11695).

    Qronos quantizes a layer's weight column-by-column with error diffusion,
    like OPTQ/GPTQ, but additionally corrects the error accumulated by
    PREVIOUSLY quantized blocks: it accumulates two statistics per layer from
    the calibration forwards,

    - ``H = X~^T X~``  - the Hessian of the *quantized-model* inputs, and
    - ``G = X~^T X``   - the cross-Gram against the *fp-model* inputs,

    and uses both in an exact-correction initialization of the sequential
    recursion (Algorithm 1). With ``X~ == X`` (no quantized-input cascade) the
    update provably reduces to OPTQ.

    The grid (scale / zero-point) is plain per-group min-max, identical to
    RTN, so the rounding algorithm is isolated as the only variable.
    """

    need_calib = True

    def __init__(
        self,
        *,
        block_size: int = 128,
        dampening_alpha: float = 1e-6,
        actorder: bool = True,
        enable_quanted_input: bool = True,
        enable_init: bool = True,
        **kwargs,
    ) -> None:
        """Initialize a Qronos configuration.

        Args:
            block_size: Number of weight columns processed per blocked update
                (memory/compute granularity only - mathematically exact at any
                value). Default 128.
            dampening_alpha: Hessian dampening ``lambda = alpha * sigma_1(H)``
                (sigma_1 = largest eigenvalue), bounding the condition number
                at ``alpha^-1``. The paper uses 1e-6 for weight-only
                quantization. Default 1e-6.
            actorder: Quantize columns in descending order of ``diag(H)``
                (activation ordering, as in GPTQ). The result is written back
                in the original column order, so no ``g_idx`` remapping is
                needed at export. Default True.
            enable_quanted_input: Feed each block the outputs of the previously
                quantized blocks (the cascade that makes ``X~ != X`` - the
                source of Qronos' edge over OPTQ). Default True.
            enable_init: Apply the error-correcting first-column
                initialization (Algorithm 1). With ``enable_quanted_input=False``
                the init reduces exactly to the OPTQ first-column update;
                disabling it yields plain sequential RTN with error diffusion
                from column two onward. Default True.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig (bits, group_size, sym, data_type, ...).
        """
        super().__init__(**kwargs)
        self.block_size = block_size
        self.dampening_alpha = dampening_alpha
        self.actorder = actorder
        self.enable_quanted_input = enable_quanted_input
        self.enable_init = enable_init
        if not self.enable_quanted_input:
            logger.warning_once(
                "QronosConfig(enable_quanted_input=False): without the quantized-input cascade, "
                "X~ == X and Qronos reduces exactly to OPTQ/GPTQ."
            )
