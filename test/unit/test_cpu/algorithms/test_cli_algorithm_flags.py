# Copyright (c) 2026 Intel Corporation
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
# ==============================================================================


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
    """Streaming CLI flags must reach the AutoRound compressor kwargs."""

    @staticmethod
    def _compressor_kwargs(extra):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        return _build_entry_compressor_kwargs(_parse(extra))

    def test_streaming_flags_reach_kwargs(self):
        kwargs = self._compressor_kwargs(["--stream_quantization", "--iters", "0"])
        self.assertIs(kwargs["stream_quantization"], True)
        # unset prefetch -> off; no separate devices kwarg exists
        self.assertEqual(kwargs["stream_prefetch"], "off")
        self.assertNotIn("stream_prefetch_devices", kwargs)
        # layerwise_rotation stays unset (auto-resolves under streaming)
        self.assertNotIn("layerwise_rotation", kwargs)

    def test_prefetch_auto(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "auto"])
        self.assertEqual(kwargs["stream_prefetch"], "auto")

    def test_prefetch_cpu(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "cpu"])
        self.assertEqual(kwargs["stream_prefetch"], "cpu")

    def test_prefetch_single_device_int(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "1"])
        self.assertEqual(kwargs["stream_prefetch"], "cuda:1")

    def test_prefetch_single_device_name(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", " cuda:3 "])
        self.assertEqual(kwargs["stream_prefetch"], "cuda:3")

    def test_prefetch_device_list_rejected(self):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        args = _parse(["--stream_prefetch", "1,2"])
        with self.assertRaises(ValueError):
            _build_entry_compressor_kwargs(args)

    def test_prefetch_off_synonyms(self):
        for value in ("off", "", "0"):
            kwargs = self._compressor_kwargs(["--stream_prefetch", value] if value else [])
            self.assertEqual(kwargs["stream_prefetch"], "off")

    def test_layerwise_rotation_flag_removed(self):
        # the flag is gone: rotation mode auto-resolves under streaming
        with self.assertRaises(SystemExit):
            _parse(["--layerwise_rotation"])

    def test_defaults_are_api_neutral(self):
        kwargs = self._compressor_kwargs([])
        self.assertIs(kwargs["stream_quantization"], False)
        self.assertEqual(kwargs["stream_prefetch"], "off")
