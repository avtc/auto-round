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

"""Primitives for the mocked-prefix continuation of the post-block region.

The lm_head input-capture lane replaces the decoder layers of a loaded model
with pass-through stubs, injects the cached last-block output at the last
layer slot, and records the exact rows the model's own post-block code feeds
the language head. Keeping these primitives side-effect free (no orchestrator
state) makes the smoke gate and the capture pass testable in isolation.
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from auto_round.utils.model import get_module

__all__ = [
    "CaptureHead",
    "PassthroughStub",
    "TailInjector",
    "install_block_stubs_",
    "restore_blocks_",
]


class PassthroughStub(nn.Module):
    """Decoder-layer stand-in that forwards the hidden states untouched.

    ``tuple_arity`` selects the return convention the model's layer loop
    expects: a bare tensor or a single-element tuple (``out[0]`` and
    ``out[-1]`` both resolve to the tensor, which also satisfies loops that
    collect KV caches from ``layer_outputs[-1]``).
    """

    def __init__(self, tuple_arity: bool = True):
        super().__init__()
        self.tuple_arity = tuple_arity

    def forward(self, hidden_states, *args, **kwargs):
        if self.tuple_arity:
            return (hidden_states,)
        return hidden_states


class TailInjector(nn.Module):
    """Last-layer-slot stand-in that returns a cached last-block output.

    The payload is swapped per calibration sample before each mocked pass;
    the forward ignores its inputs entirely and returns the cached rows with
    their original dtype and device preserved.
    """

    def __init__(self, tail: torch.Tensor, tuple_arity: bool = True):
        super().__init__()
        self.tuple_arity = tuple_arity
        self.tail = tail

    def forward(self, *args, **kwargs):
        if self.tuple_arity:
            return (self.tail,)
        return self.tail


class CaptureHead(nn.Module):
    """Language-head stand-in that records its input rows.

    Records the full-sequence input tensor (detached, on the host) for each
    call and returns a tiny dummy ``[batch, 1, out_features]`` zero tensor so
    the head GEMM and the full-vocab logits stay out of the mocked pass;
    post-head wrapper ops (float casts, softcap) run harmlessly on the dummy.
    Collection hooks never fire here: they are attached to the original head
    module, not to this stand-in.
    """

    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.records: List[torch.Tensor] = []

    def forward(self, hidden_states):
        self.records.append(hidden_states.detach().to("cpu"))
        return torch.zeros(
            hidden_states.shape[0],
            1,
            self.out_features,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )


def install_block_stubs_(
    model: nn.Module,
    block_names: List[str],
    tuple_arity: bool = True,
) -> Dict[str, object]:
    """Replace each named block module with a :class:`PassthroughStub`.

    Returns the restore bookkeeping for :func:`restore_blocks_`. The original
    modules are kept alive by the returned mapping, so streaming/meta state
    underneath them survives the temporary swap.
    """
    restore_info: Dict[str, object] = {"slots": [], "tuple_arity": tuple_arity}
    for name in block_names:
        parent_path, _, attr = name.rpartition(".")
        parent = get_module(model, parent_path) if parent_path else model
        if parent is None or not hasattr(parent, attr):
            raise ValueError(f"cannot install a stub for unknown block '{name}'")
        original = getattr(parent, attr)
        setattr(parent, attr, PassthroughStub(tuple_arity=tuple_arity))
        restore_info["slots"].append((parent, attr, original))
    return restore_info


def restore_blocks_(model: nn.Module, restore_info: Union[Dict[str, object], None]) -> None:
    """Put the original block modules back after a mocked pass.

    Restores by ``(parent, attr)`` identity rather than by name so intermediate
    slot changes (the last slot holding a :class:`TailInjector` during the
    capture pass) are irrelevant.
    """
    if not restore_info:
        return
    for parent, attr, original in restore_info["slots"]:
        setattr(parent, attr, original)
