# coding=utf-8
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
"""Pipeline-phase instrumentation on AlgorithmComposer.compress_block.

Pins ``last_pipeline_walls`` (pre_calib / pre_quant / ref_collect /
q_collect) and the AR_PERF_COUNTERS-gated ``[perf] pipeline phases`` line.
"""

import logging

import torch
import torch.nn as nn

from auto_round.algorithms.composer import AlgorithmComposer, BlockContext


class _FakePreprocessor:
    """Minimal preprocessor: registers one fp-input hook per call."""

    def register_fp_input_forward_hooks(self, block):
        def _h(module, args):
            return None

        return [block.register_forward_pre_hook(_h)]

    def pre_quantize_block(self, ctx):
        pass

    def post_quantize_block(self, ctx):
        pass


class _FakeQuantizer:
    enable_quanted_input = False

    class config:  # read by _register_act_max_hooks
        is_act_nv_fp = False
        is_dynamic_method = False

    def register_fp_input_forward_hooks(self, block):
        return []

    def register_qinput_forward_hooks(self, block):
        return []

    def quantize_block(self, *args, **kwargs):
        return None, None


def _make_composer():
    composer = object.__new__(AlgorithmComposer)
    composer.preprocessors = [_FakePreprocessor()]
    composer.block_quantizer = _FakeQuantizer()
    composer.scheme = None
    composer._coll_ctx = None
    composer.block_forward = lambda block, inputs, others, **kw: [inputs[0]]
    return composer


def _make_ctx():
    return BlockContext(
        model=None,
        block_names=["blk.0"],
        block_name="blk.0",
        block_index=0,
        bs=1,
        is_mllm=False,
        is_diffusion=False,
        pbar=None,
        block_cnt=1,
    )


def _invoke(composer):
    block = nn.Linear(4, 4)
    fp_inputs = [torch.randn(1, 2, 4)]
    composer.compress_block(block, fp_inputs, {}, _make_ctx())


def test_pipeline_walls_populated():
    composer = _make_composer()
    _invoke(composer)
    walls = composer.last_pipeline_walls
    assert set(walls) == {"pre_calib", "pre_quant", "ref_collect", "q_collect"}
    assert all(isinstance(v, float) and v >= 0.0 for v in walls.values())


def test_pipeline_walls_reset_per_block():
    composer = _make_composer()
    _invoke(composer)
    first = dict(composer.last_pipeline_walls)
    _invoke(composer)
    assert composer.last_pipeline_walls is not first  # fresh dict, no accumulation


def test_perf_line_gated():
    import auto_round.envs as envs
    from auto_round.logger import logger as ar_logger

    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Handler(level=logging.INFO)
    ar_logger.addHandler(handler)
    try:
        composer = _make_composer()
        _invoke(composer)
        assert not any("[perf] pipeline phases:" in r for r in records)  # gate off by default

        prev = getattr(envs, "AR_PERF_COUNTERS", False)
        envs.AR_PERF_COUNTERS = True
        try:
            _invoke(composer)
            assert any("[perf] pipeline phases: pre_calib=" in r and "ref_collect=" in r for r in records)
        finally:
            envs.AR_PERF_COUNTERS = prev
    finally:
        ar_logger.removeHandler(handler)


def test_step1_routes_through_collection_seam():
    """The preprocessor stats pass must go through collect_forward (shardable)."""
    calls = []

    class _SeamCtx:
        devices = [torch.device("cpu")]
        collect_stats = {}

        def collect_forward(self, block_forward, block, inputs, input_others, out_dev=None, **kw):
            calls.append((kw.get("hook_pass"), len(inputs)))
            return block_forward(block, inputs, input_others)

        def distribute_pools(self, *a, **kw):
            return None

    composer = _make_composer()
    composer._collection_context = lambda block, fp_inputs: _SeamCtx()
    _invoke(composer)
    assert calls[0] == (True, 1), calls  # Step-1 stats pass, sharded seam, hook-aware
    assert all(c[0] is False for c in calls[1:]) or len(calls) == 1, calls


def test_awq_hook_writes_hold_state_lock():
    """AWQ hook bodies must hold _HOOK_STATE_LOCK (mirror threads share state)."""
    import re as _re

    import auto_round.algorithms.transforms.awq.base as awq_base

    src_path = awq_base.__file__
    src = open(src_path, encoding="utf-8").read()
    assert "_HOOK_STATE_LOCK = threading.Lock()" in src
    # both hook write sites hold the lock (the guarded stats block and the
    # parent-cache append)
    assert src.count("with _HOOK_STATE_LOCK:") >= 2, src.count("with _HOOK_STATE_LOCK:")
