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

import pytest
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
    with pytest.raises(ValueError, match="different model's tokenizer"):
        _check_ids_in_vocab(rows, 64)


# ---------------------------------------------------------------------------
# Text-rotary selection (VL towers carry an incompatible vision rotary first)
# ---------------------------------------------------------------------------


class _VisionRotary(nn.Module):
    def forward(self, x, dim):  # vision convention: (x, dim) -- incompatible
        del x
        return dim, dim


class _TextRotary(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = 8
        self.inv_freq = None  # filled below on a real device

    def forward(self, x, position_ids):  # text convention
        d = self.inv_freq.numel() * 2
        freqs = torch.einsum("i,j->ij", position_ids.float().reshape(-1), self.inv_freq.float())
        cos = freqs.cos().unsqueeze(0)[..., : d // 2]
        sin = freqs.sin().unsqueeze(0)[..., : d // 2]
        return cos, sin


def _make_text_rotary():
    r = _TextRotary()
    d = 16
    r.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    return r


class _VLTinyRotary(nn.Module):
    """Vision rotary BEFORE the text rotary in named_modules (qwen3_5 shape)."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace()  # replaced below; keeps attr-chain honest
        self.visual = SimpleNamespace(rotary_emb=_VisionRotary())
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.rotary_emb = _make_text_rotary()


def test_find_model_rotary_prefers_language_model_attr():
    import torch.nn as nn

    from auto_round.utils.streaming_calibration import _find_model_rotary

    m = _VLTinyRotary()
    # plain-object 'model' attr breaks nn traversal; rebuild cleanly
    m = nn.Module()
    m.visual = nn.Module()
    m.visual.rotary_emb = _VisionRotary()
    m.language_model = nn.Module()
    m.language_model.model = nn.Module()
    m.language_model.model.rotary_emb = _make_text_rotary()
    cfg = SimpleNamespace(text_config=SimpleNamespace(rope_theta=10000.0))
    rotary = _find_model_rotary(m, cfg, torch.device("cpu"))
    assert isinstance(rotary, _TextRotary), "must resolve the text rotary via .language_model.model"


def test_find_model_rotary_scan_skips_vision_and_requires_position_ids():
    from auto_round.utils.streaming_calibration import _find_model_rotary

    m = torch.nn.Module()
    m.visual = torch.nn.Module()
    m.visual.deep = torch.nn.Module()
    m.visual.deep.rotary_emb = _VisionRotary()  # first in named_modules
    m.text_side = torch.nn.Module()
    m.text_side.rotary_emb = _make_text_rotary()
    cfg = SimpleNamespace()
    rotary = _find_model_rotary(m, cfg, torch.device("cpu"))
    assert isinstance(rotary, _TextRotary), "scan must deprioritize vision + prefer position_ids signature"


def test_find_model_rotary_callable_position_embeddings():
    """The selected rotary must satisfy the text-block call shape (x, pos)."""
    from auto_round.utils.streaming_calibration import _find_model_rotary

    m = torch.nn.Module()
    m.visual = torch.nn.Module()
    m.visual.rotary_emb = _VisionRotary()
    m.language_model = torch.nn.Module()
    m.language_model.model = torch.nn.Module()
    m.language_model.model.rotary_emb = _make_text_rotary()
    cfg = SimpleNamespace(text_config=SimpleNamespace(rope_theta=10000.0))
    rotary = _find_model_rotary(m, cfg, torch.device("cpu"))
    pos = torch.arange(4, device="cpu").unsqueeze(0)
    cos, sin = rotary(torch.zeros(1, 4), pos)
    assert cos.shape[-1] == 8 and sin.shape[-1] == 8


class TestParkCpuBuffers:
    """Non-checkpoint CPU buffers (rotary tables) move to the load device."""

    def test_cpu_target_is_noop(self):
        import torch

        from auto_round.utils.checkpoint_streamer import _park_cpu_buffers_

        m = torch.nn.Linear(4, 4)
        m.register_buffer("tbl", torch.ones(4))
        _park_cpu_buffers_(m, "cpu")
        assert m.tbl.device.type == "cpu"

    def test_meta_buffers_left_alone(self):
        import torch

        from auto_round.utils.checkpoint_streamer import _park_cpu_buffers_

        m = torch.nn.Linear(4, 4, device="meta")
        m.register_buffer("tbl", torch.ones(4, device="meta"))
        _park_cpu_buffers_(m, "cuda:0") if torch.cuda.is_available() else _park_cpu_buffers_(m, "cpu")
        assert m.tbl.device.type == "meta"
