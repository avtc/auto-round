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
from auto_round.algorithms.transforms.base import BaseRotationConfig


class CafeQConfig(BaseRotationConfig):
    """CafeQ learned paired V/O transform (clean-room, arXiv 2511.19705).

    Learns a per-layer orthogonal transform ``M`` on the value head axis and
    folds it offline into ``v_proj`` / ``o_proj``:

    - ``W_v[h] <- M^T @ W_v[h]`` (per v-head output block)
    - ``W_o[:, h] <- W_o[:, h] @ M`` (matching input block; ``M^{-T} == M``)

    Exact for any attention whose value mixing is row-wise per head (all real
    attentions: ``(A_h V_h) M_h == A_h (V_h M_h)``), so the fold is
    function-preserving with no online ops - deployable on stock vLLM.

    ``M`` is optimized with Adam on a Cayley parametrization (orthogonal by
    construction) against the paper's paired LogSumExp proxy: a smooth max
    over per-channel max-magnitudes of both transformed matrices, which
    drives down the quantization ranges (and hence the paired quantization
    error of the V*O product). Calibration-free: weights only.

    Of the paper's three mechanisms, the single-matrix learned block-diagonal
    (Sec. 4.1) is intentionally NOT ported: it requires an online ``M^{-1}``
    on the layer output, which cannot be folded through the residual stream
    and has no stock-vLLM deployment path. The paired adaptive rounding
    (Algorithm 1) is delegated to the terminal quantizer (OptRTN/NeUQI/Qronos).
    """

    algorithm: str = "cafeq"
    head_dim: int = None
    iters: int = 3000
    lr: float = 1e-2
    lse_temp: float = 5.0
    seed: int = 42

    def __init__(
        self,
        *,
        head_dim: int = None,
        iters: int = 3000,
        lr: float = 1e-2,
        lse_temp: float = 5.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Initialize a CafeQ configuration.

        Args:
            head_dim: value head dimension. ``None`` (default) resolves from
                the model config chain (``head_dim`` / ``v_head_dim``, with
                VLM ``text_config`` nesting); must be provided when the model
                carries no resolvable config.
            iters: Adam iterations per layer (tiny hd x hd problems converge
                fast; the paper sweeps up to 100k for full-matrix M).
            lr: Adam learning rate (paper sweep sweet spot 1e-3..1e-2).
            lse_temp: temperature ``t`` of the LogSumExp smooth-max (the
                paper found the loss insensitive to ``t``).
            seed: init seed for the random-orthogonal starting point.
            **kwargs: Reserved.
        """
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.iters = iters
        self.lr = lr
        self.lse_temp = lse_temp
        self.seed = seed
