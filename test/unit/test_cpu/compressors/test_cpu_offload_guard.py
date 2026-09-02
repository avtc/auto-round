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
"""Fail-fast guard tests for CPU-offloaded data-driven runs."""
import unittest
from unittest import mock

class TestAssertNoCpuOffload:
    """Data-driven runs fail fast when accelerate CPU-offloaded part of the model."""

    def test_raises_on_cpu_offload(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(
            model=SimpleNamespace(hf_device_map={"model.layers.0": "cuda:0", "model.layers.31": "cpu"})
        )
        import pytest

        with pytest.raises(RuntimeError, match="full GPU residency"):
            orch._assert_no_cpu_offload()

    def test_passes_when_fully_resident(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(
            model=SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.31": 1})
        )
        orch._assert_no_cpu_offload()  # must not raise

    def test_passes_without_device_map(self):
        from types import SimpleNamespace

        from auto_round.compressors.orchestrator import CompressionOrchestrator

        orch = CompressionOrchestrator.__new__(CompressionOrchestrator)
        orch.model_context = SimpleNamespace(model=SimpleNamespace())
        orch._assert_no_cpu_offload()  # must not raise
