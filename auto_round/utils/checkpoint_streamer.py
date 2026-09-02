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

"""Checkpoint tensor streaming for models that cannot be fully materialized.

Loads a model structure on the ``meta`` device and streams weight tensors
from the checkpoint shards (safetensors or .bin) on demand, one module at a
time. Enables quantization of models far larger than host RAM: peak memory
is one decoder block plus the streamed pass-through tensors.
"""

import json
import os
from typing import Optional

import torch

from auto_round.logger import logger


class CheckpointStreamer:
    """Streams tensors from a HuggingFace checkpoint on demand.

    Supports ``model.safetensors.index.json`` sharded checkpoints, single
    ``*.safetensors`` files, and ``pytorch_model.bin_index.json`` sharded
    pickle checkpoints.

    Args:
        model_path: Local directory containing the checkpoint.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weight_map: dict[str, str] = {}
        self._format: Optional[str] = None  # "safetensors" | "bin"
        self._open_handles: dict[str, object] = {}  # shard path -> safe_open handle
        self._open_order: list[str] = []
        self._max_open = 4

        # Prefetch state: a background reader stages whole module prefixes into
        # host RAM ahead of the consumer; fetch() serves from that cache first.
        self._prefetch_cache: dict[str, torch.Tensor] = {}
        self._prefetch_cond = threading.Condition()
        self._prefetch_remaining: list[str] = []  # prefixes not yet consumed
        self._prefetch_staged: list[str] = []  # staged, awaiting consumption
        self._prefetch_depth = 0
        self._prefetch_stage_devices: Optional[list] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop = False
        self._prefetch_err: Optional[BaseException] = None
        self._prefetch_handles: dict[str, object] = {}  # reader-owned handle pool
        self._prefetch_handle_order: list[str] = []

        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            self._format = "safetensors"
            self._load_index(index)
            return
        single = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single):
            self._format = "safetensors"
            from safetensors import safe_open

            with safe_open(single, framework="pt", device="cpu") as f:
                self.weight_map = {name: "model.safetensors" for name in f.keys()}
            return
        index = os.path.join(model_path, "pytorch_model.bin_index.json")
        if os.path.exists(index):
            self._format = "bin"
            self._load_index(index)
            return
        raise FileNotFoundError(
            f"No streamable checkpoint found in {model_path!r} "
            "(expected model.safetensors.index.json, model.safetensors, or pytorch_model.bin_index.json)."
        )

    def _load_index(self, index_path: str) -> None:
        with open(index_path) as f:
            self.weight_map = json.load(f)["weight_map"]

    @property
    def tensor_names(self) -> list[str]:
        return list(self.weight_map.keys())

    def _shard_path(self, shard_name: str) -> str:
        return os.path.join(self.model_path, shard_name)

    def _get_safe_open(self, shard_name: str):
        """Bounded-cache ``safe_open`` handle per shard (shards are visited in
        mostly-sequential order during block streaming)."""
        if shard_name in self._open_handles:
            return self._open_handles[shard_name]
        from safetensors import safe_open

        handle = safe_open(self._shard_path(shard_name), framework="pt", device="cpu")
        self._open_handles[shard_name] = handle
        self._open_order.append(shard_name)
        while len(self._open_order) > self._max_open:
            old = self._open_order.pop(0)
            h = self._open_handles.pop(old, None)
            if h is not None:
                h.__exit__(None, None, None)
        return handle

    # ── Prefetch ─────────────────────────────────────────────────────────────

    def start_prefetch(
        self, module_prefixes: list, depth: int = 1, stage_devices: Optional[list] = None
    ) -> None:
        """Stream whole module prefixes ahead of the consumer on a background
        thread.

        The reader stages prefixes in visit order and keeps at most ``depth``
        staged-but-unconsumed prefixes; the consumer releases a prefix with
        :meth:`prefetch_consumed`. With ``stage_devices`` (e.g. CUDA devices)
        each prefix is staged directly onto its device, round-robin by prefix
        index, overlapping the checkpoint read with block compute; without it
        prefixes land in host RAM. Memory use is bounded at roughly ``depth``
        prefixes worth of tensors.
        """
        if self._prefetch_thread is not None:
            raise RuntimeError("prefetch already running; call stop_prefetch() first")
        if stage_devices:
            for d in stage_devices:
                if torch.device(d).type == "meta":
                    raise ValueError(
                        f"invalid staging device {d!r}: meta tensors hold no data; "
                        "use a real device (e.g. 'cpu' or 'cuda:k')"
                    )
        self._prefetch_remaining = list(module_prefixes)
        self._prefetch_depth = max(1, int(depth))
        self._prefetch_stop = False
        self._prefetch_err = None
        self._prefetch_stage_devices = (
            [torch.device(d) for d in stage_devices] if stage_devices else None
        )

        def _reader():
            try:
                for idx, prefix in enumerate(module_prefixes):
                    with self._prefetch_cond:
                        while len(self._prefetch_staged) >= self._prefetch_depth and not self._prefetch_stop:
                            self._prefetch_cond.wait()
                    if self._prefetch_stop:
                        return
                    names = self.names_under(prefix)
                    if not names:
                        raise KeyError(f"no checkpoint tensors under prefix {prefix!r}")
                    stage_dev = (
                        self._prefetch_stage_devices[idx % len(self._prefetch_stage_devices)]
                        if self._prefetch_stage_devices
                        else None
                    )
                    for name in names:
                        if self._prefetch_stop:
                            return
                        tensor = self._read_tensor(name, self._prefetch_handles, self._prefetch_handle_order)
                        if stage_dev is not None:
                            tensor = tensor.to(stage_dev)
                        with self._prefetch_cond:
                            self._prefetch_cache[name] = tensor
                    with self._prefetch_cond:
                        self._prefetch_staged.append(prefix)
            except BaseException as e:  # surfaced at the next fetch
                self._prefetch_err = e
            finally:
                self._close_prefetch_handles()

        self._prefetch_thread = threading.Thread(target=_reader, daemon=True, name="ckpt-prefetch")
        self._prefetch_thread.start()

    def prefetch_pending(self) -> bool:
        """Whether the reader has work left (not yet joined/stopped)."""
        thread = self._prefetch_thread
        return thread is not None and thread.is_alive() and self._prefetch_err is None

    def prefetch_error(self) -> Optional[BaseException]:
        return self._prefetch_err

    def prefetch_consumed(self, prefix: str) -> None:
        """Mark a prefix consumed: drop any unconsumed leftovers and release
        the staging slot so the reader can proceed."""
        with self._prefetch_cond:
            for name in [n for n in self._prefetch_cache if n == prefix or n.startswith(prefix + ".")]:
                del self._prefetch_cache[name]
            for lst in (self._prefetch_remaining, self._prefetch_staged):
                if prefix in lst:
                    lst.remove(prefix)
            self._prefetch_cond.notify_all()

    def stop_prefetch(self) -> None:
        """Signal the reader to stop, join it, and clear the staging cache."""
        if self._prefetch_thread is None:
            return
        with self._prefetch_cond:
            self._prefetch_stop = True
            self._prefetch_cond.notify_all()
        self._prefetch_thread.join()
        self._prefetch_thread = None
        with self._prefetch_cond:
            self._prefetch_cache.clear()

    def _prefetch_pop(self, name: str) -> Optional[torch.Tensor]:
        if self._prefetch_err is not None:
            raise RuntimeError(f"checkpoint prefetch failed: {self._prefetch_err!r}") from self._prefetch_err
        if not self._prefetch_cache:
            return None
        with self._prefetch_cond:
            return self._prefetch_cache.pop(name, None)

    def _close_prefetch_handles(self) -> None:
        for handle in self._prefetch_handles.values():
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:  # pragma: no cover - best effort
                    pass
        self._prefetch_handles.clear()
        self._prefetch_handle_order.clear()

    def fetch(self, name: str, device: Optional[str] = None) -> torch.Tensor:
        """Read one tensor from the checkpoint (CPU unless ``device`` given)."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name!r} not present in the checkpoint index.")
        shard = self.weight_map[name]
        if self._format == "safetensors":
            tensor = self._get_safe_open(shard).get_tensor(name)
        else:
            # .bin shards: load (mmap-free) and keep a one-shard cache — shard
            # granularity is coarse for pickle checkpoints, but such shards are
            # also typically written sequentially.
            cache_key = f"__bin__{shard}"
            if cache_key not in self._open_handles:
                try:
                    data = torch.load(self._shard_path(shard), map_location="cpu", weights_only=True)
                except TypeError:  # older torch
                    data = torch.load(self._shard_path(shard), map_location="cpu")  # nosec
                self._open_handles[cache_key] = data
                self._open_order.append(cache_key)
                while len(self._open_order) > self._max_open:
                    old = self._open_order.pop(0)
                    self._open_handles.pop(old, None)
            tensor = self._open_handles[cache_key][name]
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def names_under(self, prefix: str) -> list[str]:
        """Checkpoint tensor names belonging to a module prefix."""
        return [n for n in self.weight_map if n == prefix or n.startswith(prefix + ".")]

    def _assign_leaf_(self, module: torch.nn.Module, rel_name: str, tensor: torch.Tensor) -> bool:
        """Replace a parameter/buffer leaf (meta-safe: swaps ``_parameters`` /
        ``_buffers`` entries instead of ``.data =``, which torch rejects for
        meta leaves)."""
        parts = rel_name.split(".")
        parent = module
        for p in parts[:-1]:
            parent = getattr(parent, p, None)
            if parent is None:
                return False
        leaf = parts[-1]
        if leaf in parent._parameters:
            parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
            return True
        if leaf in parent._buffers:
            parent._buffers[leaf] = tensor
            return True
        return False

    def fetch_into_(self, model: torch.nn.Module, name: str) -> bool:
        """Stream a single tensor by full name straight into ``model``."""
        tensor = self.fetch(name)
        return self._assign_leaf_(model, name, tensor)

    @torch.no_grad()
    def load_module_(self, module: torch.nn.Module, prefix: str, device: Optional[str] = None) -> list[str]:
        """Stream every checkpoint tensor under ``prefix`` into ``module``.

        Tensors replace meta placeholders leaf-by-leaf. Buffers absent from
        the checkpoint (e.g. non-persistent rotary tables) are left untouched.

        Returns the list of loaded tensor names.
        """
        targets = dict(module.named_parameters(recurse=True))
        targets.update(dict(module.named_buffers(recurse=True)))
        by_short = {prefix + ("." if prefix else "") + k: v for k, v in targets.items()}
        loaded = []
        for name in self.names_under(prefix):
            tgt = by_short.get(name, None)
            if tgt is None:
                logger.debug(f"[stream] {name} has no matching parameter/buffer in the module; skipped")
                continue
            tensor = self.fetch(name, device=device)
            if tensor.shape != tgt.shape:
                raise ValueError(
                    f"[stream] shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs module {tuple(tgt.shape)}"
                )
            if not self._assign_leaf_(module, name[len(prefix) + 1:] if prefix else name, tensor):
                raise RuntimeError(f"[stream] failed to assign {name!r} into module")
            loaded.append(name)
        if not loaded:
            raise ValueError(f"[stream] no checkpoint tensors matched module prefix {prefix!r}")
        return loaded

    def close(self) -> None:
        for handle in self._open_handles.values():
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:  # pragma: no cover - best effort
                    pass
        self._open_handles.clear()
        self._open_order.clear()
