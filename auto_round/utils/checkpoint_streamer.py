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
import re
import threading
import time
from functools import lru_cache
from typing import Optional

import torch

import auto_round.envs as envs
from auto_round.logger import logger

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


def reverse_name_map(model_type, names) -> dict:
    """Build a module-name -> checkpoint-name map for conversion aliases.

    Applies the per-family checkpoint->module rewrites to every concrete
    checkpoint name and keeps the inversion; export-side lookups that arrive
    with transformers module names can then find their checkpoint source.
    """
    renames = _name_rewrites_for(model_type)
    rev = {}
    if not renames:
        return rev
    for ckpt_name in names:
        for pattern, replacement in renames:
            candidate = pattern.sub(lambda _m, r=replacement: r, ckpt_name)
            if candidate != ckpt_name:
                rev.setdefault(candidate, ckpt_name)
                break
    return rev


@lru_cache(maxsize=None)
def _name_rewrites_for(model_type):
    """Compiled (checkpoint-pattern -> module-replacement) pairs for a family.

    transformers' per-family conversion registry is the single authoritative
    source: pure ``WeightRenaming`` aliases apply (checkpoint-side patterns
    arrive regex-ready, module-side targets plain); fusions/splits
    (``WeightConverter``) are skipped -- they are not name aliases and are
    handled by the MoE replacement machinery. Families absent from the
    registry resolve nothing here; unmatched parameters then fail loudly at
    the materialization assert in the streaming loop.
    """
    if not model_type:
        return ()
    try:
        from transformers.conversion_mapping import WeightRenaming, get_checkpoint_conversion_mapping
    except ImportError:
        return ()
    pairs = []
    for entry in get_checkpoint_conversion_mapping(model_type) or ():
        if not isinstance(entry, WeightRenaming):
            continue
        for source in entry.source_patterns:
            for target in entry.target_patterns:
                pairs.append((re.compile(source), target))
    return tuple(pairs)


class CheckpointStreamer:
    """Streams tensors from a HuggingFace checkpoint on demand.

    Supports ``model.safetensors.index.json`` sharded checkpoints, single
    ``*.safetensors`` files, and ``pytorch_model.bin_index.json`` sharded
    pickle checkpoints.

    Args:
        model_path: Local directory containing the checkpoint.
        load_dtype: Optional target dtype for floating tensors. A fully loaded
            model is converted via ``model.to(amp_dtype)`` during context
            setup; streaming must apply the same policy to every tensor it
            materializes or the streamed run would quantize raw checkpoint
            precision while the normal run quantizes the converted dtype.
    """

    def __init__(self, model_path: str, load_dtype: Optional[torch.dtype] = None):
        self.model_path = model_path
        self.load_dtype = load_dtype
        self.weight_map: dict[str, str] = {}
        self._format: Optional[str] = None  # "safetensors" | "bin"
        self._open_handles: dict[str, object] = {}  # shard path -> safe_open handle
        # shard -> tensor names not yet read; a shard whose set empties is
        # fully consumed and gets closed immediately (its mapping holds
        # every touched page in RSS until unmapped)
        self._shard_unread: dict[str, set] = {}
        self._open_order: list[str] = []
        # Sequential consume-once reads: each shard closes the moment its
        # last tensor is read (_close_if_consumed_), so no pool cap is needed.
        self._model_type = self._read_model_type()

        # Prefetch state: a background reader stages whole module prefixes into
        # host RAM ahead of the consumer; fetch() serves from that cache first.
        self._prefetch_cache: dict[str, torch.Tensor] = {}
        self._prefetch_cond = threading.Condition()
        self._prefetch_remaining: list[str] = []  # prefixes not yet consumed
        self._prefetch_staged: list[str] = []  # staged, awaiting consumption
        self._prefetch_remaining = []
        self._prefetch_enqueued = set()
        self._prefetch_depth = 0
        self._perf_segs = None  # armed per load_module_ under AR_PERF_COUNTERS
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

    def _read_model_type(self) -> Optional[str]:
        """Family of the checkpoint, from ``config.json`` beside the shards.

        Feeds the name-alias layers (transformers' conversion registry plus
        auto-round's fallback registry); ``None`` when no readable config.
        """
        config_path = os.path.join(self.model_path, "config.json")
        try:
            with open(config_path) as f:
                return json.load(f).get("model_type")
        except (OSError, ValueError):
            return None

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
        return handle

    # -- Prefetch -------------------------------------------------------------

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
                logger.info("[stream] staging GPUs below block+headroom watermarks; waiting for VRAM")
                rescue_logged = True
            time.sleep(0.5)
        return None

    def start_prefetch(self, module_prefixes: list, depth: int = 1, stage_devices: Optional[list] = None) -> None:
        """Stream whole module prefixes ahead of the consumer on a background
        thread.

        The reader stages prefixes in visit order and keeps at most ``depth``
        staged-but-unconsumed prefixes beyond the current one; the consumer
        releases a prefix with :meth:`prefetch_consumed`. Host RAM use is
        bounded at roughly (depth + 1) prefixes worth of tensors.
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
        self._prefetch_enqueued = set(module_prefixes)
        self._prefetch_depth = max(1, int(depth))
        self._prefetch_stop = False
        self._prefetch_err = None
        self._prefetch_stage_devices = [torch.device(d) for d in stage_devices] if stage_devices else None
        self._prefetch_stage_dev = {}  # prefix -> device it was staged on
        self._prefetch_cpu_staged = 0  # outstanding rescue-buffered prefixes

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
                    stage_dev = None
                    if self._prefetch_stage_devices:
                        stage_dev = self._wait_for_stage_device(idx, prefix)
                    if self._prefetch_stop:
                        return
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

    def _await_prefetch_prefix(self, prefix: str) -> None:
        """Block until the reader finishes staging *prefix* (or skips past it).

        Without this wait the consumer silently falls back to its own disk
        read while the reader keeps staging the SAME prefix in parallel - two
        full block copies on the staging device, one of them freed only after
        the duplicate lands (observed as a block-sized reserved-but-unallocated
        gap and first-forward OOMs on tightly-fitting devices). Prefixes the
        reader was never asked to stage (embeddings, lm_head, outside-block
        modules) return immediately.
        """
        if self._prefetch_thread is None or self._prefetch_stop:
            return
        with self._prefetch_cond:
            if prefix not in self._prefetch_enqueued:
                return
            while (
                prefix in self._prefetch_remaining
                and prefix not in self._prefetch_staged
                and self._prefetch_err is None
                and not self._prefetch_stop
            ):
                self._prefetch_cond.wait(timeout=5.0)

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

    def fetch(self, name: str, device: Optional[str] = None, raw: bool = False) -> torch.Tensor:
        """Read one tensor (CPU unless ``device`` given); prefetched tensors
        are served from the host-RAM cache. ``raw=True`` skips the resolved
        load-dtype policy (checkpoint-only tensors are passed through
        verbatim, mirroring the normal path where they never become part of
        the model and so are never dtype-converted)."""
        _p = self._perf_segs
        _t = [time.perf_counter()] if _p is not None else None

        def _lap(key: str) -> None:
            if _p is not None:
                now = time.perf_counter()
                _p[key] = _p.get(key, 0.0) + (now - _t[0])
                _t[0] = now

        tensor = self._prefetch_pop(name)
        _lap("pop")
        if tensor is None:
            tensor = self._read_tensor(name, self._open_handles, self._open_order)
            _lap("read")
        if self.load_dtype is not None and not raw and tensor.is_floating_point():
            tensor = tensor.to(self.load_dtype)
            _lap("cast")
        if device is not None:
            tensor = tensor.to(device)
            _lap("copy")
        return tensor

    def _read_tensor(self, name: str, handles: dict, order: list) -> torch.Tensor:
        """Read one tensor through a caller-owned handle pool."""
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name!r} not present in the checkpoint index.")
        shard = self.weight_map[name]
        if self._format == "safetensors":
            tensor = self._get_safe_open_pooled(shard, handles, order).get_tensor(name)
            self._close_if_consumed_(name, shard, handles, order)
        else:
            # .bin shards: load (mmap-free) and keep a one-shard cache - shard
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
            tensor = handles[cache_key][name]
            self._close_if_consumed_(name, shard, handles, order)
        return tensor

    def _close_if_consumed_(self, name: str, shard: str, handles: dict, order: list) -> None:
        """Close a shard the moment its last tensor is read.

        Reads are consume-once for this access pattern, so a fully read
        shard is dead weight: its mapping keeps every touched page
        resident until it is closed. Discarding the just-read tensor and
        releasing the shard here bounds every pool without a size cap.
        Safe against concurrent readers: the set only becomes empty after
        every tensor's discard has landed, and the pool pops are idempotent
        so at most one caller closes.
        """
        unread = self._shard_unread.get(shard)
        if unread is None:
            unread = {n for n, s in self.weight_map.items() if s == shard}
            self._shard_unread[shard] = unread
        unread.discard(name)
        if unread:
            return
        prefetch_handles = getattr(self, "_prefetch_handles", None)
        prefetch_order = getattr(self, "_prefetch_handle_order", None)
        pools = [(handles, order), (self._open_handles, self._open_order)]
        if prefetch_handles is not None:
            pools.append((prefetch_handles, prefetch_order))
        closed = False
        for pool_handles, pool_order in pools:
            for key in (shard, f"__bin__{shard}"):
                handle = pool_handles.pop(key, None)
                if handle is not None:
                    if hasattr(handle, "__exit__"):
                        try:
                            handle.__exit__(None, None, None)
                        except Exception:  # pragma: no cover - best effort
                            pass
                    closed = True
                if key in pool_order:
                    pool_order.remove(key)

    def release_startup_handles_(self) -> None:
        """Close every pooled shard handle.

        Startup reads (embeddings / chain init / resume rebuild) touch shards
        the block loop will never revisit; their mappings would otherwise
        keep multi-GB of touched pages resident for the whole run. Called
        before the prefetch pipeline starts, so no reader can race the close.
        Later reads simply reopen the shards they need.
        """
        for handles, order in (
            (self._open_handles, self._open_order),
            (getattr(self, "_prefetch_handles", None), getattr(self, "_prefetch_handle_order", None)),
        ):
            if not handles:
                continue
            for shard, handle in handles.items():
                if hasattr(handle, "__exit__"):
                    try:
                        handle.__exit__(None, None, None)
                    except Exception:  # pragma: no cover - best effort
                        pass
            handles.clear()
            order.clear()

    def close_shards_not_serving_(self, prefixes) -> None:
        """Close open shards that no upcoming read can touch.

        A shard stays resident while its mapping is open even when its
        remaining unread tensors will never be read in this run (skipped
        resume blocks, or non-text tensors such as vision towers that text
        streaming never materializes). Given the prefixes still planned,
        close every open shard whose unread set does not intersect them;
        later unplanned reads simply reopen the shard.
        """
        planned = set()
        for prefix in prefixes or ():
            planned.update(self.names_under(prefix))
        if not planned:
            return
        dead = []
        for shard, unread in self._shard_unread.items():
            if not unread or planned.isdisjoint(unread):
                dead.append(shard)
        if not dead:
            return
        prefetch_handles = getattr(self, "_prefetch_handles", None)
        prefetch_order = getattr(self, "_prefetch_handle_order", None)
        pools = [(self._open_handles, self._open_order)]
        if prefetch_handles is not None:
            pools.append((prefetch_handles, prefetch_order))
        for shard in dead:
            closed = False
            for pool_handles, pool_order in pools:
                handle = None
                for key in (shard, f"__bin__{shard}"):
                    h = pool_handles.pop(key, None)
                    handle = handle or h
                if handle is not None:
                    if hasattr(handle, "__exit__"):
                        try:
                            handle.__exit__(None, None, None)
                        except Exception:  # pragma: no cover - best effort
                            pass
                    closed = True
                if shard in pool_order:
                    pool_order.remove(shard)
            if closed:
                logger.debug(f"[stream] closed shard {shard}: no planned read remains")

    def close_main_pool_(self) -> None:
        """Close the consumer pool's shard handles (prefetch pool untouched).

        Called after every block's read: touched pages stay resident until
        unmap, so holding a shard open across the blocks it serves accumulates
        their pages until shard end. Closing per block bounds the residency to
        the current block; the reopen cost is a header parse (~ms). The
        prefetch reader keeps its own handles.
        """
        if not self._open_handles:
            return
        for shard, handle in self._open_handles.items():
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:  # pragma: no cover - best effort
                    pass
        self._open_handles.clear()
        self._open_order.clear()

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

    def resolve_checkpoint_name(self, module_name: str) -> Optional[str]:
        """Resolve a module-side tensor name to its checkpoint-side counterpart.

        Inverse of the conversion-registry rewrites: export-side lookups
        (root pass-through, residual hydration) arrive with transformers
        module names while the checkpoint index holds the original names
        (e.g. a MoE router weight stored under a legacy prefix). Exact
        checkpoint hits pass through unchanged.
        """
        if module_name in self.weight_map:
            return module_name
        rev = self.__dict__.get("_module_to_ckpt")
        if rev is None:
            rev = reverse_name_map(self._model_type, self.weight_map)
            self._module_to_ckpt = rev
        return rev.get(module_name)

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
        # AR_PERF_COUNTERS: per-segment fetch accounting for this load only
        perf = {} if envs.AR_PERF_COUNTERS else None
        self._perf_segs = perf
        renames = _name_rewrites_for(self._model_type)
        names = self.names_under(prefix)
        # wait out an in-flight prefetch of this very prefix instead of
        # racing it with a duplicate read (see _await_prefetch_prefix)
        self._await_prefetch_prefix(prefix)
        # file-by-file reads: group the prefix's tensors by shard so each
        # file is opened, fully read and (once consumed) closed before the
        # next one opens, instead of interleaved module-tree order
        names = sorted(names, key=lambda n: self.weight_map[n])
        for name in names:
            tgt, mod_name = by_short.get(name), name
            if tgt is None and renames:
                # checkpoint families whose spellings differ from the modeling
                # code: apply the registry aliases, never shadowing an exact hit
                for pattern, replacement in renames:
                    candidate = pattern.sub(lambda _m, r=replacement: r, name)
                    if candidate != name:
                        tgt = by_short.get(candidate)
                        if tgt is not None:
                            mod_name = candidate
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
            _t_assign = time.perf_counter() if perf is not None else None
            if not self._assign_leaf_(module, rel, tensor):
                raise RuntimeError(f"[stream] failed to assign {name!r} into module")
            if perf is not None:
                perf["assign"] = perf.get("assign", 0.0) + (time.perf_counter() - _t_assign)
            loaded.append(name)
        if not loaded:
            raise ValueError(f"[stream] no checkpoint tensors matched module prefix {prefix!r}")
        if device is not None:
            _park_cpu_buffers_(module, device)
        if self._prefetch_thread is not None:
            self.prefetch_consumed(prefix)
        if perf is not None:
            # segs stay on the instance for the caller's perf rollup
            # (AR_PERF_COUNTERS folds them into its block line)
            perf["tensors"] = len(loaded)
            self._perf_segs = perf
        return loaded

    def close(self) -> None:
        self.stop_prefetch()
        for shard_name, handle in self._open_handles.items():
            if hasattr(handle, "__exit__"):
                try:
                    handle.__exit__(None, None, None)
                except Exception:  # pragma: no cover - best effort
                    pass
        self._open_handles.clear()
        self._open_order.clear()
