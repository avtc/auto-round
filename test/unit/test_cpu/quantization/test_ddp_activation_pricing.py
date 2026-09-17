# coding=utf-8
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
"""Guard: the tune-DDP activation pricing consumes the real estimator tuple.

Regression test for the ``too many values to unpack (expected 3)`` bug where
the pricing call site unpacked three values while
``estimate_tuning_block_mem`` returns four -- the best-effort ``except`` then
silently skipped activation pricing on every block.
"""

import torch
import torch.nn as nn

from auto_round.utils.device import estimate_tuning_block_mem


class _FakeSelf:
    """Minimal carrier so the helper can be exercised standalone."""

    def __init__(self, world, global_batch, batch_size):
        self._world = world
        self._global_batch = global_batch
        self._batch_size = batch_size


def _pricing_bytes(block, fp_inputs, world, global_batch, batch_size):
    """Replica of the data_parallel.py pricing body (kept in sync by test)."""
    from auto_round.utils.device import estimate_tuning_block_mem as est

    _per_fwd = min(int(batch_size), max(1, global_batch // max(1, world)))
    _layer_mem, _act_mem, _io_mem, _add_mem = est(block, fp_inputs, _per_fwd)
    return int((_act_mem + _io_mem + _add_mem) * 2**30)


def test_estimator_returns_four_values():
    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    inputs = [torch.randn(2, 16, 64)]
    result = estimate_tuning_block_mem(block, inputs, 2)
    assert isinstance(result, tuple) and len(result) == 4, result


def test_pricing_consuming_estimator_does_not_raise():
    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    inputs = [torch.randn(2, 16, 64)]
    total = _pricing_bytes(block, inputs, world=4, global_batch=8, batch_size=2)
    assert total > 0


def test_call_site_unpack_matches_estimator_width():
    """The inline pricing in data_parallel.py must unpack exactly what the estimator returns."""
    import re

    src_path = "auto_round/algorithms/quantization/sign_round/data_parallel.py"
    src = open(src_path, encoding="utf-8").read()
    m = re.search(r"_layer_mem, _act_mem, _io_mem, _add_mem = estimate_tuning_block_mem\(", src)
    assert m is not None, "pricing unpack drifted from the 4-tuple estimator return"

    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    result = estimate_tuning_block_mem(block, [torch.randn(2, 16, 64)], 2)
    assert len(m.group(0).split("=")[0].split(",")) == len(result)
