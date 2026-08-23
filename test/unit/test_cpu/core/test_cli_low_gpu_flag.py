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
# See the License for the specific language governing permissions and limitations
# under the License.

"""CLI flag semantics for --low_gpu_mem_usage (enable/disable pair)."""

import os
import unittest

from auto_round.cli.parser import build_quantize_parser


def _parse(extra):
    parser = build_quantize_parser()
    return parser.parse_args(["dummy-model", *extra])


class TestLowGpuMemUsageFlag(unittest.TestCase):
    def test_default_is_off_matching_api_default(self):
        args = _parse([])
        self.assertFalse(args.low_gpu_mem_usage)

    def test_flag_enables(self):
        self.assertTrue(_parse(["--low_gpu_mem_usage"]).low_gpu_mem_usage)

    def test_no_and_disable_aliases_turn_it_off(self):
        self.assertFalse(_parse(["--no-low_gpu_mem_usage"]).low_gpu_mem_usage)
        self.assertFalse(_parse(["--disable_low_gpu_mem_usage"]).low_gpu_mem_usage)

    def test_enable_and_disable_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            _parse(["--low_gpu_mem_usage", "--no-low_gpu_mem_usage"])


if __name__ == "__main__":
    unittest.main()


class TestEnableBlockParallelTuningFlag(unittest.TestCase):
    def test_flag_parses_default_off(self):
        self.assertFalse(_parse([]).enable_block_parallel_tuning)
        self.assertTrue(_parse(["--enable_block_parallel_tuning"]).enable_block_parallel_tuning)

    def test_flag_flows_into_entry_kwargs(self):
        from auto_round.cli.main import _build_entry_base_kwargs

        def kw(args):
            return _build_entry_base_kwargs(args, low_cpu_mem_usage=True, enable_torch_compile=None, layer_config=None)

        self.assertFalse(kw(_parse([]))["enable_block_parallel_tuning"])
        self.assertTrue(kw(_parse(["--enable_block_parallel_tuning"]))["enable_block_parallel_tuning"])


if __name__ == "__main__":
    unittest.main()
