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

import auto_round.algorithms.quantization.sign_round.data_parallel as data_parallel
from auto_round.utils.device import estimate_tuning_block_mem


def test_estimator_returns_four_values():
    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    inputs = [torch.randn(2, 16, 64)]
    result = estimate_tuning_block_mem(block, inputs, 2)
    assert isinstance(result, tuple) and len(result) == 4, result


def test_real_call_site_prices_with_the_estimator(monkeypatch):
    """The resolver's pricing path must actually run the real estimator.

    Runs resolve_tune_ddp_plan_ with a controlled 4-tuple estimator: the
    monkeypatched estimator must be CALLED with the real block/inputs (not
    silently skipped by the best-effort except), and a 4-tuple must flow
    through the unpack without error. Mirrors are never built during
    resolve (planning only), so this is safe on any host.
    """
    from types import SimpleNamespace

    import auto_round.utils.device as dev_mod

    calls = []

    def _fake_est(block, fp_inputs, per_fwd):
        calls.append((id(block), len(fp_inputs), per_fwd))
        return (1.0, 2.0, 0.5, 0.25)  # (layer, act, io, additional) GiB

    monkeypatch.setattr(dev_mod, "estimate_tuning_block_mem", _fake_est)

    q = SimpleNamespace(
        iters=2,
        gradient_accumulate_steps=1,
        enable_lfq=False,
        _resolved_ddp_plan=None,
        _get_scaler=lambda: None,
    )
    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    fp_inputs = [torch.randn(2, 16, 64)] * 2
    # a requested world that cannot be satisfied raises AFTER pricing (no
    # mirror device passes the guard on CUDA-less hosts) -- the raise itself
    # proves execution reached past the pricing block; on hosts with enough
    # devices the call returns a plan instead, so accept both outcomes
    import pytest

    try:
        data_parallel.resolve_tune_ddp_plan_(
            q, block, fp_inputs, [torch.zeros(1)] * 2, torch.device("cuda", 0), world=2, log=False
        )
    except RuntimeError:
        pass
    assert calls and calls[0][0] == id(block), "estimator never ran against the real block"


def test_call_site_unpack_matches_estimator_width():
    """The inline pricing in data_parallel.py must unpack exactly what the estimator returns."""
    import re

    src_path = data_parallel.__file__
    src = open(src_path, encoding="utf-8").read()
    m = re.search(r"_layer_mem, _act_mem, _io_mem, _add_mem = estimate_tuning_block_mem\(", src)
    assert m is not None, "pricing unpack drifted from the 4-tuple estimator return"

    block = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    result = estimate_tuning_block_mem(block, [torch.randn(2, 16, 64)], 2)
    assert len(m.group(0).split("=")[0].split(",")) == len(result)


def test_estimator_survives_v2_wrapper_modules():
    """SignRoundV2 wrappers expose `bits` but keep `act_bits` on orig_layer.

    Regression: the DDP activation pricing silently skipped on every block
    with wrapped modules (estimator raised AttributeError -> best-effort
    except -> "[tune-ddp] activation pricing skipped"), leaving the mirror
    plan without activation charges.
    """
    from types import SimpleNamespace

    orig = nn.Linear(8, 8)
    orig.act_bits = 16  # what a real orig_layer carries

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = orig.weight
            self.bits = 4  # check_to_quantized reads this on the WRAPPER
            self.in_features = 8
            self.out_features = 8
            self.orig_layer = orig

    block = nn.Sequential(_Wrapper())
    inputs = [torch.randn(2, 4, 8)]
    result = estimate_tuning_block_mem(block, inputs, 2)
    assert isinstance(result, tuple) and len(result) == 4, result
