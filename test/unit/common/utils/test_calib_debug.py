# Copyright (c) 2026 Intel Corporation
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

import json
import os

import pytest
import torch

from auto_round.utils.calib_debug import dump_calib_rows, dump_calib_tensor


class TestCalibDebugDump:
    def test_rows_dump_written_when_env_set(self, tmp_path, monkeypatch):
        out = tmp_path / "calib_dbg"
        monkeypatch.setenv("AR_CALIB_DEBUG_DUMP", str(out))
        rows = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])]
        path = dump_calib_rows("stream", rows)
        assert path and os.path.exists(path)
        data = json.load(open(path, encoding="utf-8"))
        assert data["count"] == 2
        assert len(data["rows"]) == 2
        assert all("sha256" in r and r["shape"] == [1, 4] for r in data["rows"])
        # deterministic: same rows -> byte-identical file on a second dump
        path2 = dump_calib_rows("stream", rows)
        assert open(path, "rb").read() == open(path2, "rb").read()

    def test_rows_dump_silent_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AR_CALIB_DEBUG_DUMP", raising=False)
        assert dump_calib_rows("stream", [torch.tensor([[1]])]) == ""
        assert not (tmp_path / "calib_dbg").exists()

    def test_rows_dump_respects_max_rows(self, tmp_path, monkeypatch):
        out = tmp_path / "calib_dbg"
        monkeypatch.setenv("AR_CALIB_DEBUG_DUMP", str(out))
        rows = [torch.tensor([[i, i + 1]]) for i in range(5)]
        path = dump_calib_rows("disk", rows, max_rows=2)
        data = json.load(open(path, encoding="utf-8"))
        assert data["count"] == 5
        assert len(data["rows"]) == 2

    def test_tensor_dump_stats_and_determinism(self, tmp_path, monkeypatch):
        out = tmp_path / "calib_dbg"
        monkeypatch.setenv("AR_CALIB_DEBUG_DUMP", str(out))
        t = torch.randn(2, 8)
        path = dump_calib_tensor("stream", "block0_fp0", t)
        assert path and os.path.exists(path)
        data = json.load(open(path, encoding="utf-8"))
        entry = data["tensor"]
        assert entry["shape"] == [2, 8] and entry["dtype"] == "torch.float32"
        assert entry["norm"] == pytest.approx(float(t.norm()), rel=1e-5)
        again = dump_calib_tensor("stream", "block0_fp0", t)
        assert open(path, "rb").read() == open(again, "rb").read()

    def test_tensor_dump_ignores_non_tensor(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR_CALIB_DEBUG_DUMP", str(tmp_path))
        assert dump_calib_tensor("stream", "junk", None) == ""
        assert dump_calib_tensor("stream", "junk", [1, 2, 3]) == ""
