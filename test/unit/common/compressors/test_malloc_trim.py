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
import torch

import auto_round.envs as envs
from auto_round.compressors.orchestrator import CompressionOrchestrator


class TestMallocTrim:
    def test_env_default_off_and_lazy_read(self, monkeypatch):
        assert envs.AR_STREAM_MALLOC_TRIM is False
        monkeypatch.setenv("AR_STREAM_MALLOC_TRIM", "1")
        assert envs.AR_STREAM_MALLOC_TRIM is True
        monkeypatch.setenv("AR_STREAM_MALLOC_TRIM", "0")
        assert envs.AR_STREAM_MALLOC_TRIM is False

    def test_trim_host_heap_never_raises_and_returns_bool(self):
        # On non-glibc hosts (Windows/macOS) the ctypes load fails and the
        # helper must degrade to a quiet False, never raise.
        result = CompressionOrchestrator._trim_host_heap()
        assert isinstance(result, bool)

    def test_trim_sites_gated_by_env_in_source(self):
        # The two streaming-loop call sites must sit behind the env flag so
        # the default behavior is unchanged.
        import inspect

        src = inspect.getsource(CompressionOrchestrator)
        gated = [ln for ln in src.splitlines() if "_trim_host_heap()" in ln and "def " not in ln]
        assert len(gated) == 2, "expected exactly 2 call sites"
        for ln in gated:
            idx = src.splitlines().index(ln)
            window = "\n".join(src.splitlines()[max(0, idx - 3) : idx + 1])
            assert "AR_STREAM_MALLOC_TRIM" in window, f"call site not env-gated:\n{window}"

    def test_trim_helper_isolated_from_model_state(self):
        # must not require any instance state (called as static from the loop)
        with torch.no_grad():
            t = torch.zeros(1024, 1024)
            CompressionOrchestrator._trim_host_heap()
            t += 1
        assert bool(t.sum() > 0)


class TestStreamMemTop:
    def test_env_parse(self, monkeypatch):
        import auto_round.envs as envs

        assert envs.AR_STREAM_MEM_TOP == 0.0
        monkeypatch.setenv("AR_STREAM_MEM_TOP", "0.25")
        assert abs(envs.AR_STREAM_MEM_TOP - 0.25) < 1e-9
        monkeypatch.setenv("AR_STREAM_MEM_TOP", "not-a-float")
        assert envs.AR_STREAM_MEM_TOP == 0.0


class TestDropFileCache:
    def test_env_parse(self, monkeypatch):
        import auto_round.envs as envs

        assert envs.AR_STREAM_DROP_FILE_CACHE is False
        monkeypatch.setenv("AR_STREAM_DROP_FILE_CACHE", "1")
        assert envs.AR_STREAM_DROP_FILE_CACHE is True

    def test_drop_file_cache_never_raises(self, tmp_path):
        # non-POSIX hosts must silently no-op; a bogus path must not raise
        from auto_round.utils.checkpoint_streamer import CheckpointStreamer

        CheckpointStreamer._drop_file_cache(str(tmp_path / "missing.safetensors"))
