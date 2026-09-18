# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Raw token-id caching must happen on BOTH calib drivers.

The standard (text) calib always cached the -100-masked ids under
``inputs["input_ids"]``; the multimodal driver forwarded the batches for the
block hooks but skipped the caching, so VLM runs lost the e-side input for
predictor-tree tuning and the valid-token-mask derivation. The helper is
shared now - these tests pin its contract from both call sites' perspective.
"""

from types import SimpleNamespace

import torch

from auto_round.calibration.llm import LLMCalibrator


def _calibrator(pad_token_id=None):
    c = LLMCalibrator.__new__(LLMCalibrator)
    c.inputs = {}
    c.tokenizer = SimpleNamespace(pad_token_id=pad_token_id) if pad_token_id is not None else None
    return c


class TestCacheRawInputIds:
    def test_marks_last_position_and_splits_per_sample(self):
        c = _calibrator()
        ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
        c._cache_raw_input_ids_(ids)
        cached = c.inputs["input_ids"]
        assert len(cached) == 2
        assert cached[0].shape == (1, 4)
        assert cached[0][0, -1].item() == -100  # no next-token target
        assert cached[0][0, :-1].tolist() == [5, 6, 7]
        assert cached[1][0, :-1].tolist() == [9, 10, 11]
        # the source tensor is left untouched
        assert ids[0, -1].item() == 8

    def test_pad_tokens_marked_when_pad_id_known(self):
        c = _calibrator(pad_token_id=0)
        ids = torch.tensor([[5, 0, 7, 0]])
        c._cache_raw_input_ids_(ids)
        assert c.inputs["input_ids"][0][0].tolist() == [5, -100, 7, -100]

    def test_trailing_repeat_heuristic_when_no_pad_id(self):
        c = _calibrator()
        ids = torch.tensor([[3, 3, 3, 3]])
        c._cache_raw_input_ids_(ids)
        assert c.inputs["input_ids"][0][0].tolist() == [-100, -100, -100, -100]

    def test_caches_land_on_host(self):
        c = _calibrator()
        c._cache_raw_input_ids_(torch.tensor([[1, 2, 3]]))
        assert c.inputs["input_ids"][0].device.type == "cpu"
