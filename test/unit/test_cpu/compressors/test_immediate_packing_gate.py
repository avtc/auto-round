# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quantized layers outside blocks (lm_head-class) must not disable immediate
packing: the tail lane feeds them from the calibration chain, so blocks can
pack - and shards can stream - progressively exactly as they do without
outside-block layers. (Before the tail lane, the concern was a post-block
capture walk through packed blocks; that walk no longer exists for lm_head,
and GGUF already bypassed the gate.)
"""

from types import MethodType, SimpleNamespace

import auto_round.compressors.base as base_mod
from auto_round.algorithms.quantization.rtn.config import RTNConfig


def _fmt(gguf=False):
    return SimpleNamespace(
        is_gguf=lambda: gguf,
        is_fake=lambda: False,
        is_supported_immediate_packing=lambda: True,
        is_supported_immediate_saving=lambda: True,
    )


def _compressor(outside_block_layers):
    model = type("QwenForCausalLM", (), {"_tied_weight_keys": {}})()
    o = SimpleNamespace(
        formats=[_fmt()],
        inplace=True,
        has_qlayer_outside_block=outside_block_layers,
        need_calib=True,
        disable_opt_rtn=None,
        output_dir="/tmp/does-not-matter",
        shard_writer=None,
        _ensure_shard_writer=lambda self_=None: None,
        quantize_config=RTNConfig(data_type="int"),
        model_context=SimpleNamespace(model=model, is_mllm=False),
        compress_context=SimpleNamespace(
            is_immediate_packing=True,
            is_immediate_saving=True,
            low_cpu_mem_usage=True,
        ),
    )
    o._adjust_immediate_packing_and_saving = MethodType(base_mod.BaseCompressor._adjust_immediate_packing_and_saving, o)
    return o


class TestImmediatePackingWithOutsideBlockLayers:
    def test_outside_block_layers_keep_immediate_packing(self):
        c = _compressor(outside_block_layers=True)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is True
        # low_cpu_mem_usage + packing upgrades to progressive shard writes
        assert c.compress_context.is_immediate_saving is True

    def test_plain_run_unchanged(self):
        c = _compressor(outside_block_layers=False)
        c._adjust_immediate_packing_and_saving()
        assert c.compress_context.is_immediate_packing is True
        assert c.compress_context.is_immediate_saving is True
