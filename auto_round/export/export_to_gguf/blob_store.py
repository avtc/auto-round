# Copyright (c) 2025 Intel Corporation
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

"""Blob-shard intermediate for streaming GGUF exports.

Streaming quantization cannot write the GGUF single-file container
progressively, so per-block packing spills the already-quantized ggml tensor
payloads (the exact byte arrays the interactive exporter would hand to
``gguf.GGUFWriter.add_tensor``) into progressive safetensors shards plus a
manifest. A later pure-assembly pass rebuilds the container from those bytes
with zero quantization math, which keeps the tuned quantization state (scales,
zero points, importance matrices) byte-identical to the interactive GGUF path.

The recording writer duck-types ``gguf.GGUFWriter``: tensor payloads go to the
shard store while metadata key/value calls are captured verbatim and replayed
onto a real writer at assembly time.
"""

from __future__ import annotations

import base64
import enum
import json
import os
from collections import OrderedDict
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file as _save_safetensors

from auto_round.logger import logger
from auto_round.utils import LazyImport

gguf = LazyImport("gguf")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_MAX_SHARD_BYTES = 4 << 30  # 4 GiB of blob payload per shard file

# writer methods that only make sense against a real file-backed writer; the
# recorder accepts them as no-ops so conversion code that calls the full
# ``write()`` lifecycle keeps working, and assembly never replays them
_FILE_OP_METHODS = frozenset(
    {"write_header_to_file", "write_kv_data_to_file", "write_tensors_to_file", "close", "flush"}
)

_STORE_SINGLETONS: dict[str, "GgufBlobStore"] = {}


def _encode_value(value: Any) -> Any:
    """JSON-safe encoding for recorded writer-call arguments."""
    if isinstance(value, np.ndarray):
        return {"__np__": [str(value.dtype), list(value.shape), value.ravel().tolist()]}
    if isinstance(value, np.generic):
        return {"__npscalar__": [str(value.dtype), value.item()]}
    if isinstance(value, enum.Enum):
        return {"__enum__": [type(value).__module__, type(value).__qualname__, value.name]}
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__repr__": repr(value)}


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__np__" in value:
            dtype, shape, flat = value["__np__"]
            return np.asarray(flat, dtype=dtype).reshape(shape)
        if "__npscalar__" in value:
            dtype, item = value["__npscalar__"]
            return np.dtype(dtype).type(item)
        if "__enum__" in value:
            import importlib

            module_name, qualname, member = value["__enum__"]
            cls = getattr(importlib.import_module(module_name), qualname.rsplit(".", 1)[-1])
            return cls[member]
        if "__bytes__" in value:
            return base64.b64decode(value["__bytes__"])
        if "__path__" in value:
            return Path(value["__path__"])
        if "__repr__" in value:
            raise ValueError(f"cannot replay recorded argument {value['__repr__']!r}")
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


class _BlobTensorInfo:
    """Minimal stand-in for ``gguf.writer.TensorInfo``.

    Exposes the fields consumer code reads (``shape``) and the metadata the
    parameter counter needs, keyed by tensor name inside ``writer.tensors``.
    """

    __slots__ = ("shape", "n_elements", "dtype", "n_bytes")

    def __init__(self, shape: tuple[int, ...], dtype: str, n_bytes: int):
        self.shape = tuple(shape)
        self.n_elements = int(prod(shape)) if shape else 1
        self.dtype = dtype
        self.n_bytes = n_bytes


class RecordingGgufWriter:
    """Duck-typed ``gguf.GGUFWriter`` that spills tensors to shards.

    ``add_tensor`` payloads (already-quantized uint8 blobs or float16/32
    passthrough arrays) go to the blob store; every other method call is
    recorded with encoded arguments and replayed onto a real writer during
    assembly, so metadata generation code needs no changes.
    """

    def __init__(self, blob_store: "GgufBlobStore", role: str):
        self._store = blob_store
        self._role = role
        # list-of-dicts mirrors gguf.GGUFWriter.tensors for membership checks
        self.tensors: list[dict[str, _BlobTensorInfo]] = [OrderedDict()]
        self.kv_ops: list[dict[str, Any]] = []

    # -- data plane --------------------------------------------------------
    def add_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        raw_shape: tuple[int, ...] | None = None,
        raw_dtype: Any = None,
        tensor_endianess: Any = None,
    ) -> None:
        del tensor_endianess  # single-endianness replay; captured at store level
        data = np.ascontiguousarray(tensor)
        if raw_dtype is not None:
            dtype_name = raw_dtype.name
            if data.dtype == np.uint8:
                shape = list(gguf.quant_shape_from_byte_shape(data.shape, raw_dtype))
            else:
                shape = list(data.shape)
        else:
            dtype_name = {np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}.get(data.dtype)
            if dtype_name is None:
                raise ValueError(
                    f"blob capture for {name!r} needs raw_dtype or a float16/float32 payload, got {data.dtype}"
                )
            shape = list(data.shape)
        if raw_shape is not None:
            shape = list(raw_shape)
        self.tensors[0][name] = _BlobTensorInfo(shape, dtype_name, data.nbytes)
        self._store.add_tensor(self._role, name, data, dtype_name, tuple(shape))

    # -- counters (ported from gguf.GGUFWriter for metadata parity) ---------
    def get_total_parameter_count(self) -> tuple[int, int, int, int]:
        total_params = 0
        shared_params = 0
        expert_params = 0
        expert_sum = 0
        n_expert_tensors = 0
        last_lora_a: tuple[str, _BlobTensorInfo] | None = None

        for name, info in self.tensors[0].items():
            shape = info.shape
            if name.endswith(".lora_a"):
                last_lora_a = (name, info)
                continue
            if name.endswith(".lora_b"):
                if last_lora_a is None or last_lora_a[0] != name[:-1] + "a":
                    logger.warning("can't measure LoRA size correctly, tensor order is unusual")
                    return 0, 0, 0, 0
                shape = (*shape[:-1], last_lora_a[1].shape[-1])
            size = prod(shape)
            if "_exps." in name:
                if len(shape) >= 3:
                    expert_count = shape[-2 if ".bias" in name else -3]
                    expert_params += size // expert_count
                    expert_sum += expert_count
                    n_expert_tensors += 1
                else:
                    shared_params += size
            else:
                shared_params += size
            total_params += size

        expert_count = (expert_sum // n_expert_tensors) if n_expert_tensors > 0 else 0
        if last_lora_a is not None:
            total_params = -total_params
        return total_params, shared_params, expert_params, expert_count

    # -- everything else is recorded for replay ----------------------------
    def __getattr__(self, name: str):
        def _record(*args, **kwargs):
            self.kv_ops.append(
                {
                    "method": name,
                    "args": [_encode_value(a) for a in args],
                    "kwargs": {k: _encode_value(v) for k, v in kwargs.items()},
                }
            )
            return None

        return _record


class GgufBlobStore:
    """Progressive shard writer + manifest for one GGUF export run.

    One store serves both roles of a multimodal export (``text`` and
    ``mmproj``) so a run produces ``model.gguf`` and ``mmproj-model.gguf``
    from the same intermediate directory.
    """

    def __init__(self, root: str | os.PathLike, max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES):
        self.root = Path(root)
        self.blob_dir = self.root / "gguf-blobs"
        self.max_shard_bytes = int(max_shard_bytes)
        self._roles: dict[str, dict[str, Any]] = {}

    # -- lifecycle ----------------------------------------------------------
    @classmethod
    def get_or_create(cls, root: str | os.PathLike) -> "GgufBlobStore":
        key = str(Path(root).resolve())
        store = _STORE_SINGLETONS.get(key)
        if store is None:
            store = cls(root)
            _STORE_SINGLETONS[key] = store
        return store

    @classmethod
    def reset_singletons(cls) -> None:
        _STORE_SINGLETONS.clear()

    def adopt_existing(self) -> int:
        """Load a manifest left by an interrupted run (resume support)."""
        manifest = self._read_manifest()
        adopted = 0
        if manifest is not None:
            for role, state in manifest.get("roles", {}).items():
                entry = self._role_state(role)
                entry["entries"].extend(state.get("tensors", []))
                entry["meta"].update({k: v for k, v in state.items() if k != "tensors"})
                shards = [e["shard"] for e in entry["entries"]]
                entry["shard_counter"] = max((int(s.rsplit("-", 1)[-1].split(".")[0]) for s in shards), default=0)
                adopted += len(entry["entries"])
        return adopted

    # -- per-role state ------------------------------------------------------
    def _role_state(self, role: str) -> dict[str, Any]:
        state = self._roles.get(role)
        if state is None:
            state = {
                "recorder": None,
                "pending": OrderedDict(),
                "entries": [],
                "shard_counter": 0,
                "meta": {},  # arch / endianess / out_path / kv_ops (set at finalize)
            }
            self._roles[role] = state
        return state

    def attach_recorder(self, conversion_instance, role: str) -> RecordingGgufWriter:
        """Replace ``conversion_instance.gguf_writer`` with a recording writer."""
        state = self._role_state(role)
        if state["recorder"] is None:
            state["recorder"] = RecordingGgufWriter(self, role)
        conversion_instance.gguf_writer = state["recorder"]
        return state["recorder"]

    def add_tensor(self, role: str, name: str, data: np.ndarray, dtype_name: str, shape: tuple[int, ...]) -> None:
        state = self._role_state(role)
        state["pending"][name] = {"data": data, "dtype": dtype_name, "shape": list(shape)}

    def flush(self, role: str, reason: str = "block") -> int:
        """Durably write one role's pending blobs; returns tensors written."""
        state = self._role_state(role)
        if not state["pending"]:
            return 0
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        current: dict[str, np.ndarray] = OrderedDict()
        current_bytes = 0
        current_entries: list[dict[str, Any]] = []

        def _emit():
            nonlocal written, current_bytes
            if not current:
                return
            state["shard_counter"] += 1
            shard_name = f"blob-{role}-{state['shard_counter']:05d}.safetensors"
            _save_safetensors(current, str(self.blob_dir / shard_name))
            for entry in current_entries:
                entry["shard"] = shard_name
                state["entries"].append(entry)
            written += len(current)
            current.clear()
            current_entries.clear()
            current_bytes = 0

        for name, blob in state["pending"].items():
            nbytes = int(blob["data"].nbytes)
            if current and current_bytes + nbytes > self.max_shard_bytes:
                _emit()
            current[name] = blob["data"]
            current_bytes += nbytes
            current_entries.append(
                {"name": name, "dtype": blob["dtype"], "shape": blob["shape"], "n_bytes": nbytes, "shard": None}
            )
        _emit()
        state["pending"].clear()
        self._write_manifest()
        if written:
            logger.debug("[gguf-blob] flushed %d tensor(s) for role %s (%s)", written, role, reason)
        return written

    def finalize_role(self, conversion_instance, role: str) -> Path:
        """Capture metadata from a finished conversion instance and flush."""
        recorder = conversion_instance.gguf_writer
        if not isinstance(recorder, RecordingGgufWriter):
            raise TypeError(
                f"expected a RecordingGgufWriter on the {role} conversion instance; the export ran without blob mode"
            )
        state = self._role_state(role)
        state["meta"]["arch"] = gguf.MODEL_ARCH_NAMES[conversion_instance.model_arch]
        state["meta"]["endianess"] = str(getattr(conversion_instance, "endianess", "little"))
        state["meta"]["out_path"] = str(conversion_instance.fname_out)
        state["meta"]["kv_ops"] = recorder.kv_ops
        self.flush(role, reason="finalize")
        self._write_manifest()
        return Path(state["meta"]["out_path"])

    # -- manifest -----------------------------------------------------------
    def _manifest_path(self) -> Path:
        return self.blob_dir / MANIFEST_NAME

    def _read_manifest(self) -> dict[str, Any] | None:
        path = self._manifest_path()
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self) -> None:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        roles = {}
        for role, state in self._roles.items():
            meta = dict(state["meta"])
            if "kv_ops" in meta:
                meta["kv_ops"] = list(meta["kv_ops"])
            roles[role] = {**meta, "tensors": list(state["entries"])}
        manifest = {"version": MANIFEST_VERSION, "roles": roles}
        tmp = self._manifest_path().with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f)
        os.replace(tmp, self._manifest_path())

    # -- assembly -----------------------------------------------------------
    def assemble(self, roles: tuple[str, ...] = ("text", "mmproj"), progress: bool = False) -> list[Path]:
        """Rebuild the GGUF container(s) from the stored blobs (pure assembly)."""
        manifest = self._read_manifest()
        if manifest is None:
            raise FileNotFoundError(f"no blob manifest under {self.blob_dir}")
        outputs = []
        for role in roles:
            state = manifest.get("roles", {}).get(role)
            if not state or not state.get("tensors"):
                continue
            out_path = Path(state["out_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _assemble_role(state, self.blob_dir, out_path, progress=progress)
            outputs.append(out_path)
        if not outputs:
            raise ValueError(f"blob manifest carries no tensors for roles {roles}")
        return outputs


def _coerce_gguf_endian(value) -> Any:
    """Accept the enum, its name, or 'little'/'big' and return ``GGUFEndian``."""
    if isinstance(value, gguf.GGUFEndian):
        return value
    name = str(value).rsplit(".", 1)[-1].lower()
    if name == "little":
        return gguf.GGUFEndian.LITTLE
    if name == "big":
        return gguf.GGUFEndian.BIG
    raise ValueError(f"unsupported endianess {value!r}")


def _assemble_role(state: dict[str, Any], blob_dir: Path, out_path: Path, progress: bool = False) -> None:
    writer = gguf.GGUFWriter(
        path=None, arch=state["arch"], endianess=_coerce_gguf_endian(state.get("endianess", "little"))
    )
    replayed = 0
    for op in state.get("kv_ops", []):
        method = op["method"]
        if method in _FILE_OP_METHODS:
            continue
        args = [_decode_value(a) for a in op["args"]]
        kwargs = {k: _decode_value(v) for k, v in op["kwargs"].items()}
        getattr(writer, method)(*args, **kwargs)
        replayed += 1

    loaded_shard: dict[str, dict[str, np.ndarray]] = {}
    for entry in state["tensors"]:
        shard = entry["shard"]
        tensors = loaded_shard.get(shard)
        if tensors is None:
            from safetensors.numpy import load_file

            tensors = load_file(str(blob_dir / shard))
            loaded_shard[shard] = tensors
        data = tensors[entry["name"]]
        writer.add_tensor(entry["name"], data, raw_dtype=gguf.GGMLQuantizationType[entry["dtype"]])
    logger.info(
        "[gguf-blob] assembling %s: %d tensor(s), %d kv op(s) replayed",
        out_path.name,
        len(state["tensors"]),
        replayed,
    )
    writer.write_header_to_file(path=str(out_path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=progress)
    writer.close()


def assemble_gguf_from_blobs(
    store_root: str | os.PathLike, roles: tuple[str, ...] = ("text", "mmproj"), progress: bool = False
) -> list[Path]:
    """Standalone entry point: assemble GGUF file(s) from a blob-shard run."""
    return GgufBlobStore(store_root).assemble(roles=roles, progress=progress)
