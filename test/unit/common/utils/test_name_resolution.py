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
"""Checkpoint-name resolution for conversion aliases (export-side lookups)."""

import re

from auto_round.utils import checkpoint_streamer as cs_mod
from auto_round.utils.checkpoint_streamer import CheckpointStreamer, reverse_name_map

# registry-style rewrites: checkpoint-side patterns, PLAIN module-side targets
_TOY_RENAMES = [
    (re.compile(r"model\.layers\.0\.mlp\.router\.gate\.weight"), "model.layers.0.mlp.gate.weight"),
    (re.compile(r"model\.layers\.0\.mlp\.router\.expert_bias"), "model.layers.0.mlp.e_score_correction_bias"),
    (re.compile(r"model\.layers\.0\.mlp\.shared_mlp\."), "model.layers.0.mlp.shared_experts."),
]


def _streamer_with_alias(monkeypatch):
    s = object.__new__(CheckpointStreamer)
    s.weight_map = {
        "model.layers.0.mlp.router.gate.weight": "s0",
        "model.layers.0.mlp.router.expert_bias": "s0",
        "model.layers.0.mlp.shared_mlp.gate_proj.weight": "s0",
        "model.layers.0.self_attn.q_proj.weight": "s0",
    }
    s._model_type = "toy_family"
    monkeypatch.setattr(cs_mod, "_name_rewrites_for", lambda _mt: _TOY_RENAMES)
    return s


class TestReverseNameMap:
    def test_aliases_resolve_to_checkpoint_names(self, monkeypatch):
        s = _streamer_with_alias(monkeypatch)
        assert s.resolve_checkpoint_name("model.layers.0.mlp.gate.weight") == "model.layers.0.mlp.router.gate.weight"
        assert (
            s.resolve_checkpoint_name("model.layers.0.mlp.e_score_correction_bias")
            == "model.layers.0.mlp.router.expert_bias"
        )
        assert (
            s.resolve_checkpoint_name("model.layers.0.mlp.shared_experts.gate_proj.weight")
            == "model.layers.0.mlp.shared_mlp.gate_proj.weight"
        )

    def test_exact_hits_pass_through(self, monkeypatch):
        s = _streamer_with_alias(monkeypatch)
        name = "model.layers.0.self_attn.q_proj.weight"
        assert s.resolve_checkpoint_name(name) == name

    def test_unknown_names_return_none(self, monkeypatch):
        s = _streamer_with_alias(monkeypatch)
        assert s.resolve_checkpoint_name("model.layers.9.mlp.gate.weight") is None

    def test_module_function(self, monkeypatch):
        monkeypatch.setattr(cs_mod, "_name_rewrites_for", lambda _mt: [(re.compile(r"a\.router\.b"), "a.b")])
        assert reverse_name_map("f", {"a.router.b": "s0"}) == {"a.b": "a.router.b"}
