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

from auto_round import envs
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.logger import logger


class RTNConfig(QuantizationConfig):
    need_calib = False

    def __init__(
        self,
        *,
        disable_opt_rtn: bool = None,
        asym_search: str = "auto",
        parallel_tuning: bool = None,
        batch_expert_tuning: bool = True,
        **kwargs,
    ) -> None:
        """Initialize an RTN configuration.

        Args:
            disable_opt_rtn: Whether to disable the optimized RTN path.
                ``None`` keeps the default heuristic, True forces plain
                RTN, and False forces the optimized implementation.
            batch_expert_tuning: Quantize same-shape MoE expert projections
                in single stacked search calls instead of one call per module.
                Row-independent search: results identical to per-module tuning;
                removes per-module overhead on models with hundreds of small
                experts per layer. Default True.
            parallel_tuning: Fan per-module tuning (scale/zp search, NeUQI
                joint search) out over all available GPUs. ``None`` (default)
                enables it when more than one CUDA device is visible; ``False``
                forces the serial single-device path. Per-module results are
                identical either way.
            asym_search: How the per-group scale and integer zero point are
                initialized on the asymmetric zero-shot optimized-RTN path
                (the joint NeUQI search is opt-in):

                * ``"auto"`` (default) and ``"minmax"``: plain min/max
                  initialization, even on the optimized path.
                * ``"neuqi"``: opt into the joint (scale, integer zero-point)
                  search (arXiv 2505.17595). Requires ``sym=False`` and the
                  optimized path (``disable_opt_rtn`` not forced True).

                Scope: asymmetric zero-shot RTN only. Other quantizers
                (SignRound at ``iters > 0``, Qronos, AWQ) perform their own
                parameter search and ignore this field; the symmetric path
                uses its own scale grid search. The search grid sizes are
                tunable via ``AR_NEUQI_COARSE`` and ``AR_NEUQI_FINE``.
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

        if parallel_tuning is None and envs.AR_DISABLE_TUNING_FANOUT:
            # AR_DISABLE_TUNING_FANOUT flips the AUTO default to serial; an
            # explicit parallel_tuning kwarg still wins (no silent overrides)
            parallel_tuning = False
        self.parallel_tuning = parallel_tuning
        self.batch_expert_tuning = batch_expert_tuning
        valid_asym_search = ("auto", "neuqi", "minmax")
        if asym_search not in valid_asym_search:
            raise ValueError(f"asym_search={asym_search!r} is not one of {valid_asym_search}.")
        if asym_search != "auto" and self.sym:
            raise ValueError(
                f"asym_search={asym_search!r} applies to the asymmetric path only (sym=False); the "
                "symmetric path uses its own scale grid search."
            )
        if asym_search == "neuqi" and disable_opt_rtn:
            raise ValueError(
                'asym_search="neuqi" requires the optimized-RTN path: it replaces the plain min/max '
                "initialization of that path. Set disable_opt_rtn=False (or leave it unset)."
            )
        self.asym_search = asym_search


class OptimizedRTNConfig(RTNConfig):
    need_calib = True
