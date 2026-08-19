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
        with pytest.raises(ValueError, match="mass source"):
            rot.apply_to_model(bh.MiniModel())


class TestPeRQPlumbing:
    def test_registry_and_normalization(self):
        from auto_round.algorithms.transforms import normalize_rotation_config

        assert "perq" in BaseRotation._REGISTRY
        cfg = normalize_rotation_config("perq")
        assert isinstance(cfg, PeRQConfig) and cfg.block_size == 64 and cfg.mass == "weight"
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
