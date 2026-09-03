# coding=utf-8
# Apache License 2.0 Copyright (c) 2025 Intel Corporation


"""_adjust_immediate_packing_and_saving interaction rules.

The streaming zero-shot loop requires immediate saving (progressive shard
writes), which requires low_cpu_mem_usage. The legacy downgrade for
outside-block quantized layers with a non-RTN quantizer (SignRound, iters>0)
must not fire under stream_quantization -- it would trip the streaming
immediate-saving guard before the first block.
"""

from types import SimpleNamespace

from auto_round.algorithms.quantization.rtn.config import RTNConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.compressors.base import BaseCompressor


class _Fmt:
    def is_supported_immediate_packing(self):
        return True

    def is_supported_immediate_saving(self):
        return True

    def is_fake(self):
        return False

    def is_gguf(self):
        return False


class _CausalLM:  # name lowercased contains "causallm" -> tied-weight check skipped
    _tied_weight_keys = {}


def _fake_self(stream, quant_cfg, outside=True):
    quant_cfg.data_type = "int"
    return SimpleNamespace(
        formats=[_Fmt()],
        compress_context=SimpleNamespace(low_cpu_mem_usage=True, is_immediate_packing=True, is_immediate_saving=False),
        inplace=True,
        model_context=SimpleNamespace(model=_CausalLM(), is_mllm=False),
        has_qlayer_outside_block=outside,
        quantize_config=quant_cfg,
        stream_quantization=stream,
        output_dir="/tmp/x",
        shard_writer=object(),  # non-None keeps _ensure_shard_writer inert
        model=SimpleNamespace(),
        _ensure_shard_writer=lambda: None,
    )


class TestImmediateSavingRules:
    def test_signround_streaming_keeps_low_cpu_mem_usage(self):
        fake = _fake_self(stream=True, quant_cfg=SignRoundConfig(iters=200))
        BaseCompressor._adjust_immediate_packing_and_saving(fake)
        assert fake.compress_context.low_cpu_mem_usage is True, "streaming must keep low_cpu_mem_usage"
        assert fake.compress_context.is_immediate_saving is True

    def test_signround_non_streaming_still_downgrades(self):
        fake = _fake_self(stream=False, quant_cfg=SignRoundConfig(iters=200))
        BaseCompressor._adjust_immediate_packing_and_saving(fake)
        assert fake.compress_context.low_cpu_mem_usage is False, "legacy downgrade preserved"
        # behavior change vs upstream: per-block immediate packing is NOT
        # disabled anymore (outside-block layers pack at export)
        assert fake.compress_context.is_immediate_packing is True

    def test_rtn_outside_block_keeps_enabled(self):
        fake = _fake_self(stream=False, quant_cfg=RTNConfig())
        BaseCompressor._adjust_immediate_packing_and_saving(fake)
        assert fake.compress_context.low_cpu_mem_usage is True
