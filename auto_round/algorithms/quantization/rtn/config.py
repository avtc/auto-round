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


class RTNConfig(QuantizationConfig):
    need_calib = False

    def __init__(
        self,
        *,
        disable_opt_rtn: bool = None,
        asym_search: bool = None,
        **kwargs,
    ) -> None:
        """Initialize an RTN configuration.

        Args:
            disable_opt_rtn: Whether to disable the optimized RTN path.
                ``None`` keeps the default heuristic, True forces plain
                RTN, and False forces the optimized implementation.
            asym_search: Explicit control over the asymmetric joint
                (scale, integer zero-point) search (NeUQI, arXiv 2505.17595)
                used by the optimized-RTN path when ``sym=False``:

                * ``None`` (default) - auto: engage whenever the optimized
                  path runs (``disable_opt_rtn`` False, ``iters == 0``).
                * ``True`` - force it; raises ``ValueError`` at construction
                  when the config cannot engage it (``sym=True`` or explicit
                  ``disable_opt_rtn=True``). Engagement additionally requires
                  the zero-shot pipeline (``iters == 0``), which is inherent
                  to the optimized-RTN dispatch.
                * ``False`` - plain min/max asymmetric quantization even on
                  the optimized path (per-config equivalent of
                  ``AR_DISABLE_NEUQI=1``).

                The symmetric path (``sym=True``) is never affected. The
                ``AR_DISABLE_NEUQI`` env var still wins as a kill-switch.
                Grid sizes are tunable via ``AR_NEUQI_COARSE``/``AR_NEUQI_FINE``.
            **kwargs: Common quantization arguments forwarded to
                QuantizationConfig, such as bits, group_size, sym,
                data_type, and activation quantization fields.
        """
        # pop before super().__init__ so it doesn't leak into QuantizationConfig as an unknown kwarg
        enable_opt_rtn = kwargs.pop("enable_opt_rtn", None)
        super().__init__(**kwargs)

        if enable_opt_rtn:
            disable_opt_rtn = False
        self.orig_disable_opt_rtn = disable_opt_rtn

        if disable_opt_rtn is None:  # TODO wenhuach move to AR entry
            if self.bits and self.bits >= 8 and self.act_bits and self.act_bits >= 8 and self.data_type == "int":
                logger.warning("`disable_opt_rtn` is turned on for W8A16/W8A8 quantization to improve efficiency.")
                disable_opt_rtn = True
        if disable_opt_rtn is None:
            logger.info(
                "`enable_opt_rtn` is turned on, set `--disable_opt_rtn` for higher speed at the cost of accuracy."
            )
            disable_opt_rtn = False
        self.disable_opt_rtn = disable_opt_rtn

        if asym_search is True:
            if self.sym:
                raise ValueError(
                    "asym_search=True requires sym=False: the joint (scale, zero-point) search is an "
                    "asymmetric-path algorithm; the symmetric path keeps its own grid search."
                )
            if disable_opt_rtn:
                raise ValueError(
                    "asym_search=True requires the optimized-RTN path: it replaces the plain min/max "
                    "initialization of that path. Set disable_opt_rtn=False (or leave it unset)."
                )
        self.asym_search = asym_search


class OptimizedRTNConfig(RTNConfig):
    need_calib = True
