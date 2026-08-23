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

"""Block-name resolution for the Qwen3.5/3.6 omni family (model_type ``qwen3_5``)."""

import unittest
from unittest import mock


def _make_qwen3_5_mock(n_layers=4, n_vision_blocks=3, with_audio=False):
    model = mock.MagicMock()
    model.config = mock.MagicMock()
    model.config.model_type = "qwen3_5"
    model.config.architectures = ["Qwen3_5ForConditionalGeneration"]

    language_model = mock.MagicMock()
    language_model.layers = [mock.MagicMock() for _ in range(n_layers)]
    visual = mock.MagicMock()
    visual.blocks = [mock.MagicMock() for _ in range(n_vision_blocks)]
    inner = mock.MagicMock()
    inner.language_model = language_model
    inner.visual = visual
    if with_audio:
        inner.audio_tower = mock.MagicMock()  # must never join the block list
    model.model = inner
    return model


class TestQwen3_5Registry(unittest.TestCase):
    def test_registered_in_special_multimodal_block(self):
        from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

        self.assertIn("qwen3_5", SPECIAL_MULTIMODAL_BLOCK)

    def test_resolve_model_type_falls_back_to_model_type(self):
        from auto_round.utils.model import resolve_model_type

        model = _make_qwen3_5_mock()
        self.assertEqual(resolve_model_type(model), "qwen3_5")


class TestQwen3_5BlockDetection(unittest.TestCase):
    def test_default_quantizes_text_decoder_only(self):
        from auto_round.special_model_handler import _get_qwen3_5_multimodal_block

        model = _make_qwen3_5_mock(n_layers=4, n_vision_blocks=3)
        blocks = _get_qwen3_5_multimodal_block(model)
        self.assertEqual(blocks, [[f"model.language_model.layers.{i}" for i in range(4)]])

    def test_quant_vision_includes_vision_tower_before_text(self):
        from auto_round.special_model_handler import _get_qwen3_5_multimodal_block

        model = _make_qwen3_5_mock(n_layers=4, n_vision_blocks=3)
        blocks = _get_qwen3_5_multimodal_block(model, quant_vision=True)
        self.assertEqual(blocks[0], [f"model.visual.blocks.{i}" for i in range(3)])
        self.assertEqual(blocks[1], [f"model.language_model.layers.{i}" for i in range(4)])

    def test_audio_tower_never_included(self):
        from auto_round.special_model_handler import _get_qwen3_5_multimodal_block

        model = _make_qwen3_5_mock(n_layers=2, n_vision_blocks=2, with_audio=True)
        for quant_vision in (False, True):
            blocks = _get_qwen3_5_multimodal_block(model, quant_vision=quant_vision)
            flat = [b for group in blocks for b in group]
            self.assertFalse(any("audio" in b for b in flat))


if __name__ == "__main__":
    unittest.main()
