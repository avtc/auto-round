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
import threading
import time
from typing import Optional

import torch

from auto_round.logger import logger


# Checkpoint-name fragment -> module-name fragment, applied only when the
# direct spelling misses (never shadows an exact match). Covers checkpoints
# whose block spellings differ from the modeling code: hyv3 stores
# ``mlp.shared_mlp`` / ``mlp.router.gate`` / ``mlp.expert_bias`` while the
# runtime module exposes ``mlp.shared_experts`` / ``mlp.gate`` /
# ``mlp.e_score_correction_bias``.
_CKPT_NAME_REWRITES = (
    (".mlp.shared_mlp.", ".mlp.shared_experts."),
    (".mlp.router.gate.", ".mlp.gate."),
    (".mlp.expert_bias", ".mlp.e_score_correction_bias"),
)


def to_checkpoint_name(name: str) -> str:
    """Map a runtime module-name onto its checkpoint spelling.

    Inverse of the rewrite table above: applies the first matching module-name
    fragment substitution and returns *name* unchanged when nothing matches.
    """
    for ckpt_frag, mod_frag in _CKPT_NAME_REWRITES:
        if mod_frag in name:
            return name.replace(mod_frag, ckpt_frag)
    return name


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def device_free_bytes(device: torch.device) -> Optional[int]:
    """Free bytes on *device*; ``None`` when unknown (non-CUDA devices)."""
    if device.type != "cuda":
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:  # pragma: no cover - platform without a CUDA runtime
        return None


def pick_stage_device(devices: list, index: int, needed_bytes: int, headroom: int) -> Optional[torch.device]:
    """Choose a staging device for prefix *index*, or ``None`` when no CUDA
    device can hold ``needed_bytes`` while keeping ``headroom`` free for the
    tuning fan-out's search batches.

    Walks the round-robin rotation first; callers decide how to react to
    ``None`` (wait for VRAM, or stage on host RAM when explicitly requested).
    """
    for k in range(len(devices)):
        d = torch.device(devices[(index + k) % len(devices)])
        free = device_free_bytes(d)
        if free is None or free >= needed_bytes + headroom:
            return d
    return None


@torch.no_grad()
def _park_cpu_buffers_(module: torch.nn.Module, device) -> None:
    """Move non-checkpoint CPU buffers (e.g. rotary inv_freq tables) to ``device``.

    ``load_module_`` only replaces tensors present in the checkpoint; buffers a
    module computes at init (rotary frequency tables, non-persistent caches) keep
    whatever device the skeleton gave them -- typically CPU. A block forward then
    mixes CUDA weights with CPU cos/sin and dies with a device mismatch at the
    first rope-using (full-attention) block. Already-on-device tensors are left
    untouched (identity mapping, no copies); meta buffers are left for
    materialize_model_ to report.
    """
    target = torch.device(device)
    if target.type == "cpu":
        return

    def _fn(t: torch.Tensor) -> torch.Tensor:
        return t.to(target) if t.device.type == "cpu" else t

    module._apply(_fn)


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
        """Bounded-cache ``safe_open`` handle per shard on the consumer pool
        (shards are visited in mostly-sequential order during block streaming)."""
        return self._get_safe_open_pooled(shard_name, self._open_handles, self._open_order)

    def _get_safe_open_pooled(self, shard_name: str, handles: dict, order: list):
        """Bounded-cache ``safe_open`` handle per shard on a caller-owned pool."""
        if shard_name in handles:
            return handles[shard_name]
        from safetensors import safe_open

        handle = safe_open(self._shard_path(shard_name), framework="pt", device="cpu")
        handles[shard_name] = handle
        order.append(shard_name)
        while len(order) > self._max_open:
            old = order.pop(0)
            h = handles.pop(old, None)
            if h is not None:
                h.__exit__(None, None, None)
        return handle

    # ── Prefetch ─────────────────────────────────────────────────────────────

    # Bytes reserved per staging GPU for the tuning fan-out's share of the
    # active block's search batches (chunked stacks + intermediates).
    _staging_search_headroom = 4 * 1024**3

    def _prefix_bytes_estimate(self, prefix: str) -> int:
        """Exact staged size of the tensors under *prefix*, from shard metadata
        (no tensor data is read)."""
        import math

        total = 0
        for name in self.names_under(prefix):
            shard = self.weight_map[name]
            h = self._get_safe_open_pooled(shard, self._prefetch_handles, self._prefetch_handle_order)
            sl = h.get_slice(name)
            total += math.prod(sl.get_shape()) * _DTYPE_BYTES.get(sl.get_dtype().upper(), 4)
        return total

    def _wait_for_stage_device(self, index: int, prefix: str) -> Optional[torch.device]:
        """Pick a staging device for *prefix*, waiting for VRAM.

        VRAM-first: a device qualifies when its free memory covers the block
        (``needed_bytes``) plus the search headroom for the active layer's
        tuning jobs. When no GPU qualifies, exactly ONE block may wait in host
        RAM (rescue buffer, keeping some I/O overlap alive); further blocks
        wait until VRAM frees or the rescue block is consumed.
        """
        devices = self._prefetch_stage_devices
        headroom = int(getattr(self, "_staging_search_headroom", 0) or 0)
        needed = self._prefix_bytes_estimate(prefix) if devices else 0
        rescue_allowed = any(d.type == "cuda" for d in devices)
        rescue_logged = False
        while not self._prefetch_stop:
            dev = pick_stage_device(devices, index, needed, headroom)
            if dev is not None:
                return dev
            if rescue_allowed:
                with self._prefetch_cond:
                    if self._prefetch_cpu_staged < 1:
                        self._prefetch_cpu_staged += 1
                        if not rescue_logged:
                            logger.info(
                                "[stream] no staging GPU has %d GiB free for a %d GiB block + %d GiB search headroom; "
                                "holding one block in host RAM",
                                (needed + headroom) // (1024**3),
                                needed // (1024**3),
                                headroom // (1024**3),
                            )
                            rescue_logged = True
                        return torch.device("cpu")
            if not rescue_logged and rescue_allowed:
                logger.info(
                    "[stream] staging GPUs below block+headroom watermarks; waiting for VRAM"
                )
                rescue_logged = True
            time.sleep(0.5)
        return None

    def start_prefetch(
        self, module_prefixes: list, depth: int = 1, stage_devices: Optional[list] = None
    ) -> None:
        """Stream whole module prefixes ahead of the consumer on a background
        thread.

        The reader stages prefixes in visit order and keeps at most ``depth``
        staged-but-unconsumed prefixes beyond the current one; the consumer
        releases a prefix with :meth:`prefetch_consumed`. Host RAM use is
        bounded at roughly (depth + 1) prefixes worth of tensors.
        """
        if self._prefetch_thread is not None:
            raise RuntimeError("prefetch already running; call stop_prefetch() first")
        self._prefetch_remaining = list(module_prefixes)
        self._prefetch_depth = max(1, int(depth))
        self._prefetch_stop = False
        self._prefetch_err = None
        self._prefetch_stage_devices = (
            [torch.device(d) for d in stage_devices] if stage_devices else None
        )
        self._prefetch_stage_dev = {}  # prefix -> device it was staged on
        self._prefetch_cpu_staged = 0  # outstanding rescue-buffered prefixes

        def _reader():
            try:
                for prefix in module_prefixes:
                    with self._prefetch_cond:
                        while len(self._prefetch_staged) >= self._prefetch_depth and not self._prefetch_stop:
                            self._prefetch_cond.wait()
                    if self._prefetch_stop:
                        return
                    names = self.names_under(prefix)
                    if not names:
                        raise KeyError(f"no checkpoint tensors under prefix {prefix!r}")
                    stage_dev = None
                    if self._prefetch_stage_devices:
                        stage_dev = self._wait_for_stage_device(idx, prefix)
                    if self._prefetch_stop:
                        return
                    for name in names:
                        if self._prefetch_stop:
                            return
                        tensor = self._read_tensor(name, self._prefetch_handles, self._prefetch_handle_order)
                        with self._prefetch_cond:
                            self._prefetch_cache[name] = tensor
                    with self._prefetch_cond:
                        self._prefetch_staged.append(prefix)
                        if stage_dev is not None:
                            self._prefetch_stage_dev[prefix] = stage_dev
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
        the staging slot (and the host-RAM rescue slot if it held *prefix*).
        """
        with self._prefetch_cond:
            for name in [n for n in self._prefetch_cache if n == prefix or n.startswith(prefix + ".")]:
                del self._prefetch_cache[name]
            for lst in (self._prefetch_remaining, self._prefetch_staged):
                if prefix in lst:
                    lst.remove(prefix)
            dev = self._prefetch_stage_dev.pop(prefix, None)
            if dev is not None and dev.type != "cuda":
                self._prefetch_cpu_staged = max(0, self._prefetch_cpu_staged - 1)
            self._prefetch_cond.notify_all()



    def wait_until_staged(self, prefix: str, timeout: Optional[float] = None) -> bool:
        """Wait until the prefetch reader has fully staged ``prefix``.

        Returns False when the prefix cannot arrive (reader stopped or failed,
        or ``timeout`` seconds elapsed). With no prefetch thread running there
        is nothing to wait for -- :meth:`load_module_` will read from disk.
        """
        if self._prefetch_thread is None:
            return True
        deadline = None if timeout is None else time.time() + timeout
        with self._prefetch_cond:
            while prefix not in self._prefetch_staged:
                if self._prefetch_stop or self._prefetch_err is not None:
                    return False
                if deadline is not None and time.time() >= deadline:
                    return False
                self._prefetch_cond.wait(0.5)
        return True

    def prefix_bytes(self, prefix: str) -> int:
        """Exact staged size (bytes) of the tensors under ``prefix``."""
        return self._prefix_bytes_estimate(prefix)

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
        """Read one tensor (CPU unless ``device`` given); prefetched tensors
        are served from the host-RAM cache."""
        tensor = self._prefetch_pop(name)
        if tensor is None:
            tensor = self._read_tensor(name, self._open_handles, self._open_order)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def _read_tensor(self, name: str, handles: dict, order: list) -> torch.Tensor:
        """Read one tensor through a caller-owned handle pool."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name!r} not present in the checkpoint index.")
        shard = self.weight_map[name]
        if self._format == "safetensors":
            tensor = self._get_safe_open_pooled(shard, handles, order).get_tensor(name)
        else:
            # .bin shards: load (mmap-free) and keep a one-shard cache — shard
            # granularity is coarse for pickle checkpoints, but such shards are
            # also typically written sequentially.
            cache_key = f"__bin__{shard}"
            if cache_key not in handles:
                try:
                    data = torch.load(self._shard_path(shard), map_location="cpu", weights_only=True)
                except TypeError:  # older torch
                    data = torch.load(self._shard_path(shard), map_location="cpu")  # nosec
                handles[cache_key] = data
                order.append(cache_key)
                while len(order) > self._max_open:
                    old = order.pop(0)
                    handles.pop(old, None)
            tensor = handles[cache_key][name]
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
        rewrites_hit = 0
        for name in self.names_under(prefix):
            tgt, mod_name = by_short.get(name), name
            if tgt is None:
                for ckpt_frag, mod_frag in _CKPT_NAME_REWRITES:
                    if ckpt_frag in name:
                        cand = name.replace(ckpt_frag, mod_frag)
                        tgt = by_short.get(cand)
                        if tgt is not None:
                            mod_name = cand
                            rewrites_hit += 1
                            break
            if tgt is None:
                logger.debug(f"[stream] {name} has no matching parameter/buffer in the module; skipped")
                continue
            tensor = self.fetch(name, device=device)
            if tensor.shape != tgt.shape:
                raise ValueError(
                    f"[stream] shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs module {tuple(tgt.shape)}"
                )
            rel = mod_name[len(prefix) + 1 :] if prefix else mod_name
            if not self._assign_leaf_(module, rel, tensor):
                raise RuntimeError(f"[stream] failed to assign {name!r} into module")
            loaded.append(name)
        if rewrites_hit:
            logger.info(f"[stream] {rewrites_hit} checkpoint tensor(s) matched via name rewrite")
        if not loaded:
            raise ValueError(f"[stream] no checkpoint tensors matched module prefix {prefix!r}")
        if device is not None:
            _park_cpu_buffers_(module, device)
        if self._prefetch_thread is not None:
            self.prefetch_consumed(prefix)
        return loaded

    def close(self) -> None:
        self.stop_prefetch()
        for handle in self._open_handles.values():
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:  # pragma: no cover - best effort
                    pass
        self._open_handles.clear()
        self._open_order.clear()
