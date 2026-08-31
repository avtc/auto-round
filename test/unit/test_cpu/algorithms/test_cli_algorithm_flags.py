# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI wiring for --enable_neuqi and --enable_presinq: parsed flags must reach
the built algorithm configs (RTNConfig.asym_search, PreSINQConfig composition)."""

import os
import unittest

import pytest

from auto_round.cli.algorithms import AlgorithmHandler
from auto_round.cli.parser import build_quantize_parser


def _parse(extra):
    parser = build_quantize_parser()
    return parser.parse_args(["dummy-model", *extra])


def _build(args):
    return AlgorithmHandler.build_configs(args, common_kwargs={})


class TestEnableNeuqiFlag(unittest.TestCase):
    def test_default_off_keeps_auto(self):
        args = _parse(["--iters", "0"])
        configs = _build(args)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        rtn = [c for c in configs if isinstance(c, RTNConfig)]
        self.assertTrue(rtn, "an RTN config must always be built for iters=0")
        self.assertEqual(rtn[0].asym_search, "auto")

    def test_flag_sets_neuqi(self):
        args = _parse(["--enable_neuqi", "--asym", "--enable_opt_rtn", "--iters", "0"])
        configs = _build(args)
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        rtn = [c for c in configs if isinstance(c, RTNConfig)]
        self.assertTrue(rtn)
        self.assertEqual(rtn[0].asym_search, "neuqi")


class TestParallelQuantizationFlag:
    def test_default_off_keeps_serial_and_env_untouched(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '0'])
        (cfg,) = _build(args)
        assert cfg.parallel_tuning is False
        assert "AR_TUNE_DDP_WORLD" not in os.environ

    def test_auto_at_iters0_resolves_ddp_not_fanout(self, monkeypatch):
        """auto at iters=0 resolves the DDP world (sharded collect) instead of
        the per-module fan-out: under --stream_quantization the round-robin
        block staging already spreads the searches across the same GPUs, so
        the fan-out only adds device-to-device contention."""
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '0', '--parallel_quantization', 'auto', '--device_map', '0,1,2,3'])
        (cfg,) = _build(args)
        assert cfg.parallel_tuning is False
        assert cfg.parallel_tuning_workers is None
        assert os.environ["AR_TUNE_DDP_WORLD"] == "4"

    def test_auto_at_iters0_excludes_cpu_devices(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '0', '--parallel_quantization', 'auto', '--device_map', '0,cpu,1'])
        (cfg,) = _build(args)
        assert cfg.parallel_tuning is False
        assert os.environ["AR_TUNE_DDP_WORLD"] == "2"

    def test_auto_at_iters0_single_gpu_stays_off(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '0', '--parallel_quantization', 'auto', '--device_map', '0'])
        (cfg,) = _build(args)
        assert cfg.parallel_tuning is False
        assert "AR_TUNE_DDP_WORLD" not in os.environ

    def test_n_at_iters0_pins_workers(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '0', '--parallel_quantization', '3'])
        (cfg,) = _build(args)
        assert cfg.parallel_tuning
        assert cfg.parallel_tuning_workers == 3

    def test_auto_at_iters_positive_sets_ddp_world_from_device_map(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '20', '--parallel_quantization', 'auto', '--device_map', '0,1,2,3'])
        _build(args)
        assert os.environ["AR_TUNE_DDP_WORLD"] == "4"

    def test_n_at_iters_positive_sets_ddp_world(self, monkeypatch):
        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        args = _parse(['--iters', '20', '--parallel_quantization', '2'])
        _build(args)
        assert os.environ["AR_TUNE_DDP_WORLD"] == "2"

    def test_off_leaves_explicit_env_alone(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "4")
        args = _parse(['--iters', '20'])
        _build(args)
        assert os.environ["AR_TUNE_DDP_WORLD"] == "4"

    def test_conflicting_flag_and_env_raises(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        args = _parse(['--iters', '20', '--parallel_quantization', '4'])
        with pytest.raises(ValueError):
            _build(args)

    def test_parser_rejects_one_and_garbage(self):
        with pytest.raises(SystemExit):
            _parse(['--parallel_quantization', '1'])
        with pytest.raises(SystemExit):
            _parse(['--parallel_quantization', 'bogus'])


class TestEnablePreSINQFlag(unittest.TestCase):
    def test_flag_composes_presinq_config(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--enable_presinq", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)]
        self.assertEqual(len(presinq), 1, "--enable_presinq must add exactly one PreSINQConfig")
        # locked defaults pass through untouched
        self.assertEqual(presinq[0].n_iter, 4)
        self.assertEqual(presinq[0].n_repeat, 3)

    def test_algorithm_list_composes_presinq(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--algorithm", "presinq,rtn", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)]
        self.assertEqual(len(presinq), 1)

    def test_no_flag_no_presinq(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--iters", "0"])
        configs = _build(args)
        self.assertEqual([c for c in configs if isinstance(c, PreSINQConfig)], [])

    def test_hyperparameter_overrides_flow_into_config(self):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        args = _parse(["--enable_presinq", "--presinq_group_size", "128", "--presinq_n_repeat", "5", "--iters", "0"])
        configs = _build(args)
        presinq = [c for c in configs if isinstance(c, PreSINQConfig)][0]
        self.assertEqual(presinq.group_size, 128)
        self.assertEqual(presinq.n_repeat, 5)


class TestStreamingCliFlags(unittest.TestCase):
    """Streaming/layerwise CLI flags must reach the AutoRound compressor kwargs."""

    @staticmethod
    def _compressor_kwargs(extra):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        return _build_entry_compressor_kwargs(_parse(extra))

    def test_streaming_flags_reach_kwargs(self):
        kwargs = self._compressor_kwargs(
            [
                "--layerwise_rotation",
                "--stream_quantization",
                "--iters",
                "0",
            ]
        )
        self.assertIs(kwargs["layerwise_rotation"], True)
        self.assertIs(kwargs["stream_quantization"], True)
        # unset prefetch -> off
        self.assertEqual(kwargs["stream_prefetch"], 0)
        self.assertIsNone(kwargs["stream_prefetch_devices"])

    def test_prefetch_auto(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "auto"])
        self.assertIsNone(kwargs["stream_prefetch"], "auto derives depth at loop start")
        self.assertEqual(kwargs["stream_prefetch_devices"], "auto")

    def test_prefetch_cpu(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "cpu"])
        self.assertIsNone(kwargs["stream_prefetch"])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cpu"])

    def test_prefetch_gpu_list_ints(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", "1,2"])
        self.assertIsNone(kwargs["stream_prefetch"])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cuda:1", "cuda:2"])

    def test_prefetch_gpu_list_names(self):
        kwargs = self._compressor_kwargs(["--stream_prefetch", " cuda:3 , cuda:5 "])
        self.assertEqual(kwargs["stream_prefetch_devices"], ["cuda:3", "cuda:5"])

    def test_prefetch_off_synonyms(self):
        for value in ("off", "", "0"):
            kwargs = self._compressor_kwargs(["--stream_prefetch", value] if value else [])
            self.assertEqual(kwargs["stream_prefetch"], 0)
            self.assertIsNone(kwargs["stream_prefetch_devices"])

    def test_prefetch_empty_list_rejected(self):
        from auto_round.cli.main import _build_entry_compressor_kwargs

        args = _parse(["--stream_prefetch", " , "])
        with self.assertRaises(ValueError):
            _build_entry_compressor_kwargs(args)

    def test_defaults_are_api_neutral(self):
        kwargs = self._compressor_kwargs([])
        self.assertIsNone(kwargs["layerwise_rotation"], "unset must auto-resolve, not force off")
        self.assertIs(kwargs["stream_quantization"], False)
        self.assertEqual(kwargs["stream_prefetch"], 0)
        self.assertIsNone(kwargs["stream_prefetch_devices"])


class TestImatrixEnabledFlag(unittest.TestCase):
    """--imatrix_enabled tri-state: the forced marker overrides the scheme
    rules during config-class selection; auto leaves them in control."""

    @staticmethod
    def _rtn(extra):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        configs = _build(_parse([*extra, "--iters", "0"]))
        return [c for c in configs if isinstance(c, RTNConfig)][0]

    def test_auto_leaves_rules_in_control(self):
        rtn = self._rtn(["--imatrix_enabled", "auto"])
        self.assertNotIn("forced_imatrix", rtn.__dict__, "auto must not force an override")

    def test_true_sets_forced_marker(self):
        rtn = self._rtn(["--imatrix_enabled", "true"])
        self.assertIs(rtn.forced_imatrix, True)

    def test_false_sets_forced_marker(self):
        rtn = self._rtn(["--imatrix_enabled", "false"])
        self.assertIs(rtn.forced_imatrix, False)

    def test_default_is_auto(self):
        rtn = self._rtn([])
        self.assertNotIn("forced_imatrix", rtn.__dict__)

    def test_forced_true_survives_asym_scheme(self):
        """asym resolves imatrix off by rule; the forced marker must stay set so
        config-class selection (autoround.py) turns it back on."""
        rtn = self._rtn(["--imatrix_enabled", "true", "--asym"])
        self.assertIs(rtn.forced_imatrix, True)


class TestDynamicMaxGapFlag(unittest.TestCase):
    def _signround(self, extra):
        args = _parse(["--iters", "50", *extra])
        configs = _build(args)
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        sr = [c for c in configs if isinstance(c, SignRoundConfig)]
        self.assertTrue(sr, "an SignRound config must be built for iters>0")
        return sr[0]

    def test_default_disabled(self):
        self.assertEqual(self._signround([]).dynamic_max_gap, -1)

    def test_flag_reaches_config(self):
        self.assertEqual(self._signround(["--dynamic_max_gap", "15"]).dynamic_max_gap, 15)
