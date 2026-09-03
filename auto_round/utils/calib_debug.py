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

"""Env-gated calibration diagnostics.

Set ``AR_CALIB_DEBUG_DUMP=<dir>`` to make the calibration funnels of every
execution path (data-driven text, data-driven multimodal-template, and the
streaming chain) record comparable fingerprints of what actually reaches the
model: the token ids of each accepted calibration row and the hidden states
entering the first decoder block. The JSON files are deterministic (no
timestamps) so two runs can be diffed directly to localize where paths
diverge -- e.g. a streamed block-0 loss that differs from the data-driven
block-0 loss despite identical initial weights.
"""

import hashlib
import json
import os
from typing import Any, Optional

import torch

_DUMP_ENV = "AR_CALIB_DEBUG_DUMP"


def _dump_dir() -> Optional[str]:
    path = os.getenv(_DUMP_ENV, None)
    return path if path else None


def _sha256_rows(rows) -> list[dict]:
    entries = []
    for idx, row in enumerate(rows):
        if not isinstance(row, torch.Tensor):
            continue
        flat = row.detach().to(torch.float32).reshape(-1)
        entries.append(
            {
                "row": idx,
                "sha256": hashlib.sha256(row.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                "sha256_f32": hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest(),
                "shape": list(row.shape),
                "dtype": str(row.dtype),
            }
        )
    return entries


def _tensor_stats(tensor) -> dict:
    t = tensor.detach().to(torch.float32).cpu()
    return {
        "sha256_f32": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(t.mean()),
        "std": float(t.std()) if t.numel() > 1 else 0.0,
        "norm": float(t.reshape(-1).norm()),
    }


def dump_calib_rows(tag: str, rows, max_rows: int = 0) -> str:
    """Fingerprint accepted calibration rows (token ids or hidden states).

    Writes ``<dir>/<tag>_rows.json`` with one sha256 per row. No-op unless
    ``AR_CALIB_DEBUG_DUMP`` is set. ``max_rows=0`` records every row.
    """
    directory = _dump_dir()
    if directory is None:
        return ""
    os.makedirs(directory, exist_ok=True)
    selected = rows if max_rows <= 0 else rows[:max_rows]
    payload: dict[str, Any] = {"tag": tag, "count": len(rows), "rows": _sha256_rows(selected)}
    path = os.path.join(directory, f"{tag}_rows.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def dump_calib_tensor(tag: str, name: str, tensor) -> str:
    """Fingerprint one tensor (e.g. the first block's input hidden states).

    Writes ``<dir>/<tag>_<name>.json``. No-op unless the env var is set.
    """
    directory = _dump_dir()
    if directory is None or not isinstance(tensor, torch.Tensor):
        return ""
    os.makedirs(directory, exist_ok=True)
    payload: dict[str, Any] = {"tag": tag, "name": name, "tensor": _tensor_stats(tensor)}
    path = os.path.join(directory, f"{tag}_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
