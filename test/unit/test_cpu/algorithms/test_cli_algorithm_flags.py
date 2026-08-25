# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI wiring for --enable_neuqi and --enable_presinq: parsed flags must reach
the built algorithm configs (RTNConfig.asym_search, PreSINQConfig composition)."""

import unittest

from auto_round.cli.algorithms import AlgorithmHandler
from auto_round.cli.parser import build_quantize_parser


def _parse(extra):
    parser = build_quantize_parser()
    return parser.parse_args(["dummy-model", *extra])


def _build(args):
    return AlgorithmHandler.build_configs(args, common_kwargs={})


class TestEnableNeuqiFlag(unittest.TestCase):
    def test_default_off_keeps_auto(self):
        args = _parse(["--iters", "0"])
        configs = _build(args)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        rtn = [c for c in configs if isinstance(c, RTNConfig)]
        self.assertTrue(rtn, "an RTN config must always be built for iters=0")
        self.assertEqual(rtn[0].asym_search, "auto")

    def test_flag_sets_neuqi(self):
        args = _parse(["--enable_neuqi", "--asym", "--enable_opt_rtn", "--iters", "0"])
        configs = _build(args)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        rtn = [c for c in configs if isinstance(c, RTNConfig)]
        self.assertTrue(rtn)
        self.assertEqual(rtn[0].asym_search, "neuqi")


class TestEnablePreSINQFlag(unittest.TestCase):
    def test_flag_composes_presinq_config(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--enable_presinq", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)]
        self.assertEqual(len(presinq), 1, "--enable_presinq must add exactly one PreSINQConfig")
        # locked defaults pass through untouched
        self.assertEqual(presinq[0].n_iter, 4)
        self.assertEqual(presinq[0].n_repeat, 3)

    def test_algorithm_list_composes_presinq(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--algorithm", "presinq,rtn", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)]
        self.assertEqual(len(presinq), 1)

    def test_no_flag_no_presinq(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--iters", "0"])
        configs = _build(args)
        self.assertEqual([c for c in configs if isinstance(c, PreSINQConfig)], [])

    def test_hyperparameter_overrides_flow_into_config(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--enable_presinq", "--presinq_group_size", "128", "--presinq_n_repeat", "5", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)][0]
        self.assertEqual(presinq.group_size, 128)
        self.assertEqual(presinq.n_repeat, 5)
