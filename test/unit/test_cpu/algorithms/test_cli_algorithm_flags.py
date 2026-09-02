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

"""CLI wiring for streaming flags: parsed flags must reach the compressor kwargs."""

import unittest

from auto_round.cli.algorithms import AlgorithmHandler
from auto_round.cli.parser import build_quantize_parser


def _parse(extra):
    parser = build_quantize_parser()
    return parser.parse_args(["dummy-model", *extra])


def _build(args):
    return AlgorithmHandler.build_configs(args, common_kwargs={})




class TestStreamingCliFlags(unittest.TestCase):
    """Streaming/layerwise CLI flags must reach the AutoRound compressor kwargs."""

    @staticmethod
    def _compressor_kwargs(extra):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        return _build_entry_compressor_kwargs(_parse(extra))

    def test_streaming_flags_reach_kwargs(self):
        kwargs = self._compressor_kwargs(
            [
                "--layerwise_rotation",
                "--stream_quantization",
                "--iters",
                "0",
            ]
        )
        self.assertIs(kwargs["layerwise_rotation"], True)
        self.assertIs(kwargs["stream_quantization"], True)
        # unset prefetch -> off
        self.assertEqual(kwargs["stream_prefetch"], 0)
        self.assertIsNone(kwargs["stream_prefetch_devices"])

    def test_prefetch_auto(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "auto"])
        self.assertIsNone(kwargs["stream_prefetch"], "auto derives depth at loop start")
        self.assertEqual(kwargs["stream_prefetch_devices"], "auto")

    def test_prefetch_cpu(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "cpu"])
        self.assertIsNone(kwargs["stream_prefetch"])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cpu"])

    def test_prefetch_gpu_list_ints(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "1,2"])
        self.assertIsNone(kwargs["stream_prefetch"])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cuda:1", "cuda:2"])

    def test_prefetch_gpu_list_names(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", " cuda:3 , cuda:5 "])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cuda:3", "cuda:5"])

    def test_prefetch_off_synonyms(self):
        for value in ("off", "", "0"):
            kwargs = self._compressor_kwargs(["--stream_prefetch", value] if value else [])
            self.assertEqual(kwargs["stream_prefetch"], 0)
            self.assertIsNone(kwargs["stream_prefetch_devices"])

    def test_prefetch_empty_list_rejected(self):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        args = _parse(["--stream_prefetch", " , "])
        with self.assertRaises(ValueError):
            _build_entry_compressor_kwargs(args)

    def test_defaults_are_api_neutral(self):
        kwargs = self._compressor_kwargs([])
        self.assertIsNone(kwargs["layerwise_rotation"], "unset must auto-resolve, not force off")
        self.assertIs(kwargs["stream_quantization"], False)
        self.assertEqual(kwargs["stream_prefetch"], 0)
        self.assertIsNone(kwargs["stream_prefetch_devices"])


