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
"""Tests for the PeRQ transform (MassDiff permutation + block-Hadamard).

Reuses the mini hybrid-model fixtures from ``test_block_hadamard``.
"""

from __future__ import annotations

import pytest
import torch

from test.test_cpu.algorithms.transforms import test_block_hadamard as bh
from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.perq import (
    PeRQConfig,
    PeRQRotation,
    massdiff_permutation,
)
HIDDEN = bh.HIDDEN
BLOCK = 64
VOCAB = bh.VOCAB


class TestMassDiff:
    def test_valid_permutation_deterministic(self):
        mass = torch.rand(HIDDEN, generator=torch.Generator().manual_seed(3))
        p1 = massdiff_permutation(mass, BLOCK)
        p2 = massdiff_permutation(mass, BLOCK)
        assert p1 == p2
        assert sorted(p1) == list(range(HIDDEN))

    def test_balances_extreme_outliers(self):
        # One huge outlier channel: after balancing, the block containing it
        # must have the LOWEST total mass among blocks (all other blocks get
        # two mid-mass channels each).
        mass = torch.tensor([100.0] + [1.0] * 3)
        perm = massdiff_permutation(mass, 2)
        blocks = torch.tensor(perm).reshape(-1, 2)
        sums = mass[blocks].sum(dim=1)
        assert sums[0].item() == 101.0  # outlier block
        assert sums[1].item() == 2.0
        assert sums.argmin().item() == 1  # outlier block ends up lightest

    def test_block_masses_nearly_equal(self):
        mass = torch.rand(256, generator=torch.Generator().manual_seed(5)) * 10
        mass[7] = 100.0  # inject outlier (~28x mean)
        perm = massdiff_permutation(mass, 64)
        sums = mass[torch.tensor(perm)].reshape(-1, 64).sum(dim=1)
        assert (sums.max() / sums.mean()).item() < 1.05

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            massdiff_permutation(torch.ones(10), 4)


class TestPeRQExactness:
    def _run(self, **cfg_overrides):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        model.eval()
        ids = torch.randint(0, VOCAB, (2, 16))
        with torch.no_grad():
            before = model(ids)
        rot = PeRQRotation(PeRQConfig(**{"block_size": BLOCK, **cfg_overrides}))
        rot.apply_to_model(model)
        with torch.no_grad():
            after = model(ids)
        torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)

    def test_weight_mass_default(self):
        self._run()

    def test_no_permutation(self):
        self._run(mass="none")

    def test_smaller_block(self):
        self._run(block_size=32)

    def test_unknown_mass_source(self):
        rot = PeRQRotation(PeRQConfig(mass="acts"))
        with pytest.raises(ValueError, match="imatrix_path"):
            rot.apply_to_model(bh.MiniModel())


class Wrapper(torch.nn.Module):
    """VLM-style wrapper: renames the backbone with a language_model. prefix."""

    def __init__(self, inner):
        super().__init__()
        self.language_model = inner

    def forward(self, ids):
        return self.language_model(ids)


class TestPeRQActsMass:
    def _make_dump(self, model, tmp_path, drop=0, gen=None):
        gen = gen or torch.Generator().manual_seed(11)
        stats = {}
        names = [n for n, _ in model.named_modules()]
        for n in names:
            stats[n] = torch.rand(HIDDEN, generator=gen) + 0.1
        if drop:
            stats = dict(list(stats.items())[: len(stats) - drop])
        path = str(tmp_path / "imatrix.pt")
        torch.save({"imatrix": stats, "counts": {}, "meta": {}}, path)
        return path

    def test_acts_exactness_and_matching(self, tmp_path):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        model.eval()
        # dump uses CANONICAL names (model.layers...) while the wrapped model
        # exposes language_model.model.layers... - suffix matching must resolve
        wrapped = Wrapper(model)
        dump = self._make_dump(model, tmp_path)
        ids = torch.randint(0, VOCAB, (2, 16))
        with torch.no_grad():
            before = wrapped(ids)
        rot = PeRQRotation(PeRQConfig(block_size=64, mass="acts", imatrix_path=dump))
        rot.apply_to_model(wrapped)
        with torch.no_grad():
            after = wrapped(ids)
        torch.testing.assert_close(before, after, atol=2e-4, rtol=2e-4)

    def test_partial_coverage_warns(self, tmp_path):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        dump = self._make_dump(model, tmp_path, drop=3)
        rot = PeRQRotation(PeRQConfig(block_size=64, mass="acts", imatrix_path=dump))
        rot.apply_to_model(model)  # warns, still works

    def test_missing_path(self):
        rot = PeRQRotation(PeRQConfig(mass="acts"))
        with pytest.raises(ValueError, match="imatrix_path"):
            rot.apply_to_model(bh.MiniModel())
        rot2 = PeRQRotation(PeRQConfig(mass="acts", imatrix_path="nope.pt"))
        with pytest.raises(FileNotFoundError):
            rot2.apply_to_model(bh.MiniModel())

    def test_no_match_raises(self, tmp_path):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        path = str(tmp_path / "empty.pt")
        torch.save({"imatrix": {"unrelated.module": torch.rand(HIDDEN)}}, path)
        rot = PeRQRotation(PeRQConfig(mass="acts", imatrix_path=path))
        with pytest.raises(ValueError, match="no stream consumers matched"):
            rot.apply_to_model(model)


class TestPeRQAuto:
    def test_resolves_below_full_width(self):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        rot = PeRQRotation(PeRQConfig())  # block_size=0 -> auto
        rot.apply_to_model(model)
        # mini hidden = 128 (pow2) -> auto steps down once -> 64
        assert rot.config.block_size == 64

    def test_mass_none_same_auto_rule(self):
        torch.manual_seed(0)
        model = bh.MiniModel()
        bh._rand_init(model)
        rot = PeRQRotation(PeRQConfig(mass="none"))
        rot.apply_to_model(model)
        assert rot.config.block_size == 64


class TestPeRQPlumbing:
    def test_registry_and_normalization(self):
        from auto_round.algorithms.transforms import normalize_rotation_config

        assert "perq" in BaseRotation._REGISTRY
        cfg = normalize_rotation_config("perq")
        assert isinstance(cfg, PeRQConfig) and cfg.block_size == 0 and cfg.mass == "weight"
        cfg2 = normalize_rotation_config({"algorithm": "perq", "block_size": 32, "mass": "none"})
        assert isinstance(cfg2, PeRQConfig) and cfg2.block_size == 32
        inst = BaseRotation.from_config(cfg)
        assert isinstance(inst, PeRQRotation)

    def test_composer_family_exclusion(self):
        from auto_round.algorithms.composer import AlgorithmComposer
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.transforms.block_hadamard import BlockHadamardConfig
        from auto_round.algorithms.transforms.presinq import PreSINQConfig

        composer = AlgorithmComposer([PeRQConfig(), RTNConfig()])
        assert len(composer._rotation_configs) == 1

        composer = AlgorithmComposer([PreSINQConfig(), PeRQConfig(), RTNConfig()])
        assert len(composer._rotation_configs) == 2

        with pytest.raises(ValueError, match="mutually exclusive"):
            AlgorithmComposer([PeRQConfig(), BlockHadamardConfig(), RTNConfig()])
