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
"""Token-embedding selection for the streaming calibration chain.

VL-capable checkpoints (qwen3_5) carry a small positional Embedding in the
vision tower that iterates BEFORE the token embedding; the selector must not
take the first checkpoint-backed Embedding it sees.
"""

from types import SimpleNamespace

import torch
from torch import nn

from auto_round.utils.streaming_calibration import _check_ids_in_vocab, _find_embedding


class _Streamer:
    def __init__(self, *names):
        self.weight_map = {f"{n}.weight" for n in names}


class _Vision(nn.Module):
    def __init__(self, pos_rows):
        super().__init__()
        self.pos_emb = nn.Embedding(pos_rows, 8)


class _TextModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, 8)


class _VLTiny(nn.Module):
    """Vision tower (small positional table) BEFORE the token embedding."""

    def __init__(self, pos_rows=64, vocab=500):
        super().__init__()
        self.visual = _Vision(pos_rows)
        self.model = _TextModel(vocab)
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab))


def test_prefers_config_vocab_match_over_first_embedding():
    m = _VLTiny()
    streamer = _Streamer("visual.pos_emb", "model.embed_tokens")
    name, mod = _find_embedding(m, streamer)
    assert name == "model.embed_tokens"
    assert mod.num_embeddings == 500


def test_largest_embedding_when_config_silent():
    m = _VLTiny()
    m.config = SimpleNamespace()  # no vocab_size anywhere
    streamer = _Streamer("visual.pos_emb", "model.embed_tokens")
    name, mod = _find_embedding(m, streamer)
    assert name == "model.embed_tokens" and mod.num_embeddings == 500


def test_guard_passes_with_correct_vocab():
    rows = [torch.tensor([499, 0, 17])]
    _check_ids_in_vocab(rows, 500)  # must not raise
    try:
        _check_ids_in_vocab(rows, 64)
        raise SystemError("should have raised")
    except ValueError:
        pass
