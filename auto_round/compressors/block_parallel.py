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

"""Block-parallel tuning support.

The serial block-wise tuning loop chains each block's input from the previous
block's *reference* (original-weights) output, so blocks are independent of each
other's quantization state: any block's input equals the original model's
activation at that depth. This module exploits that by re-exec'ing the same CLI
command in worker processes pinned one per GPU. A strict-frontier relay
feeds each worker the next unassigned block whose entry checkpoint exists;
the assignee extends the chain with one no-grad original-weights forward
(faithful to the serial chain and ~1000x cheaper than tuning) before tuning,
and dumps tuned ``scale``/``zp`` per layer instead of writing shards. The
parent merges the dumps and runs the ordinary immediate-pack/save flow.

Only valid while the quantized-input chain is dormant (``need_quanted_input`` --
GGUF's sequential replay makes blocks order-dependent); see
``block_parallel_tuning_enabled``.
"""

import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

import torch

from auto_round import envs
from auto_round.utils import logger

Span = Tuple[int, int, int]  # (group_idx, start, end) -- end exclusive


def eligible_gpus(required_free_bytes: int, allowed_indices: Optional[List[int]] = None) -> List[int]:
    """CUDA devices with enough free VRAM for one tuning worker.

    ``required_free_bytes`` is derived by the caller from the model (largest
    block's weight bytes times a working-set multiplier plus an activation
    margin) so eligibility adapts to the model instead of a fixed guess.
    Heterogeneous or partially-occupied GPUs filter out naturally.
    """
    if not torch.cuda.is_available():
        return []
    visible = list(range(torch.cuda.device_count()))
    if allowed_indices is not None:
        visible = [i for i in allowed_indices if 0 <= i < torch.cuda.device_count()]
    out = []
    for index in visible:
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:  # noqa: BLE001
            continue
        if free >= required_free_bytes:
            out.append(index)
    return out


def worker_command(argv: Sequence[str], device: Optional[int] = None) -> List[str]:
    """Re-exec command for a worker: same CLI invocation as the parent.

    When ``device`` is given (worker pinned via CUDA_VISIBLE_DEVICES), any
    ``--device_map`` argument is collapsed to ``0`` so multi-GPU maps from the
    parent invocation don't request invisible devices.
    """
    argv = list(argv)
    if argv and str(argv[0]).endswith("__main__.py"):
        # Parent was launched with ``python -m <pkg>``: reproduce -m semantics.
        # Running __main__.py as a plain file puts the package directory (not
        # the repo root) on sys.path, which can shadow the checkout with a
        # stale site-packages copy and break imports.
        from pathlib import PurePath

        package = PurePath(str(argv[0])).parent.name
        argv = [sys.executable, "-m", package] + argv[1:]
    elif argv and str(argv[0]).endswith(".py"):
        argv = [sys.executable] + argv
    if device is not None:
        out, i = [], 0
        while i < len(argv):
            arg = argv[i]
            if arg == "--device_map" and i + 1 < len(argv):
                out.extend(["--device_map", "0"])
                i += 2
                continue
            if arg.startswith("--device_map="):
                out.append("--device_map=0")
            else:
                out.append(arg)
            i += 1
        argv = out
    return argv


def spawn_workers(argv: Sequence[str], gpu_ids: List[int], log_dir: str, extra_env: dict) -> List[subprocess.Popen]:
    """Spawn one assignment-serving worker per GPU, staggered.

    Spawns are staggered by a fixed delay so the per-worker model-build and
    dataset-preprocessing host-RAM transients do not spike together on
    small-RAM hosts. Assignments arrive as lines on each worker's stdin
    (``tune G K`` / ``stop``); results come back as per-block files (durable
    state), never through the pipe. Worker identity (rank) rides in the env.
    """
    import time as _time

    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for rank, gpu in enumerate(gpu_ids):
        if rank:
            _time.sleep(5.0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["AR_BLOCK_PARALLEL_WORKER"] = "1"
        env["AR_BLOCK_PARALLEL_RANK"] = str(rank)
        # AR_RESUME_DIR is deliberately inherited: workers derive their
        # per-block checkpoint dir from it (same user-facing flag as serial).
        env["AR_ENABLE_AUTO_SCHEME_PARALLEL"] = "0"  # single-GPU workers never fan out
        env.update(extra_env)
        log_path = os.path.join(log_dir, f"worker_{rank}.log")
        logger.info("block-parallel tuning: worker %d -> gpu %d (log %s)", rank, gpu, log_path)
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        procs.append(
            subprocess.Popen(  # noqa: S603
                worker_command(argv, device=gpu),
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        )
    return procs


def wait_workers(procs: List[subprocess.Popen]) -> List[int]:
    """Wait for all workers; returns exit codes (nonzero entries logged)."""
    codes = []
    for proc in procs:
        code = proc.wait()
        codes.append(code)
        if code != 0:
            logger.error("block-parallel tuning: worker exited with code %d (see its log)", code)
    return codes


def parallel_results_dir() -> Optional[str]:
    """Directory for per-block tuned results, derived from AR_RESUME_DIR.

    Resume is requested with the same flag as the serial path: when
    AR_RESUME_DIR is set, per-block result files and chain checkpoints are
    written under ``<AR_RESUME_DIR>/block_parallel`` and a rerun picks up every
    block already tuned (regardless of span layout -- files are block-keyed).
    Without AR_RESUME_DIR there is no resumability; the caller uses a fresh
    scratch dir.
    """
    resume_dir = envs.AR_RESUME_DIR
    if not resume_dir:
        return None
    return os.path.join(resume_dir, "block_parallel")


def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "-")


def _block_results_path(results_dir: str, block_name: str) -> str:
    return os.path.join(results_dir, f"block_{_sanitize(block_name)}.pt")


def block_result_path(results_dir: str, block_name: str) -> str:
    """Path of a block's tuned-result file (see ``save_block_results``)."""
    return _block_results_path(results_dir, block_name)


def _chain_state_path(results_dir: str, group: int, block_idx: int) -> str:
    return os.path.join(results_dir, f"chain_g{group}_b{block_idx}.pt")


def save_block_results(results_dir: str, block_name: str, layers: dict) -> None:
    """Atomically persist one block's tuned ``{layer_name: {scale, zp}}``.

    One file per block is the unit of resumption: block tuning is independent
    (inputs chain through original-weights outputs), so any subset of completed
    blocks composes with a rerun that finishes the rest -- with ANY worker
    count, or by the serial path. ``_worker_rank`` is injected as advisory
    metadata (live parent->worker targeting); it never carries durable meaning.
    """
    os.makedirs(results_dir, exist_ok=True)
    payload = dict(layers)
    payload["_worker_rank"] = int(os.getenv("AR_BLOCK_PARALLEL_RANK", "-1"))
    path = _block_results_path(results_dir, block_name)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def has_block_results(results_dir: str, block_name: str) -> bool:
    return os.path.exists(_block_results_path(results_dir, block_name))


def read_block_result(results_dir: str, block_name: str):
    """Load a block's result dict, or ``None`` when it is not usable yet.

    ``None`` covers missing, still-being-written (unloadable), corrupt, and
    non-dict files alike: all mean "not a result" to polling callers.
    An empty layer dict is a valid completion: blocks whose every layer is
    excluded from quantization tune to nothing and legitimately save an empty
    result. Prefer this over a load-then-load-again pair when the caller also
    needs the payload (e.g. the advisory worker rank).
    """
    path = _block_results_path(results_dir, block_name)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  corrupt or half-written file: not a result
        return None
    return data if isinstance(data, dict) else None


def block_results_complete(results_dir: str, block_name: str) -> bool:
    """File exists AND loads as a result dict (see ``read_block_result``).

    Only an unloadable file (corrupt artifact) counts as not-done, so a rerun
    re-tunes the block instead of erroring at pack time. Cheap existence
    checks (polling loops) keep using has_block_results.
    """
    return read_block_result(results_dir, block_name) is not None


def load_all_block_results(results_dir: str) -> dict:
    """Merge every per-block result file into one {layer_name: {scale, zp}} dict."""
    merged: dict = {}
    if not results_dir or not os.path.isdir(results_dir):
        return merged
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("block_") and fname.endswith(".pt")):
            continue
        data = torch.load(os.path.join(results_dir, fname), map_location="cpu", weights_only=True)
        for name, entry in data.items():
            if name.startswith("_"):
                continue  # advisory metadata (_worker_rank), not a layer
            merged[name] = {"scale": entry["scale"], "zp": entry["zp"]}
    return merged


def read_result_rank(results_dir: str, block_name: str) -> int:
    """Advisory worker rank recorded in a block result, or -1."""
    data = read_block_result(results_dir, block_name)
    return int(data.get("_worker_rank", -1)) if data else -1


def missing_result_blocks(results_dir: str, all_blocks: "Sequence[list]") -> "List[str]":
    """Blocks (flattened) with no result file yet."""
    if not results_dir:
        return [block for group in all_blocks for block in group]
    return [block for group in all_blocks for block in group if not has_block_results(results_dir, block)]


def maybe_load_chain_state(results_dir: str, group: int, block_idx: int, device=None):
    """Return the published entry for ``block_idx`` if its checkpoint exists.

    On resume, checkpoints published by a crashed run are still valid (the
    forward is deterministic over original weights), so callers reuse them
    instead of recomputing.
    """
    if not chain_state_exists(results_dir, group, block_idx):
        return None
    return load_chain_state(results_dir, group, block_idx, device=device)


def chain_state_exists(results_dir: str, group: int, block_idx: int) -> bool:
    return os.path.exists(_chain_state_path(results_dir or "", group, block_idx))


def chain_state_path(results_dir: str, group: int, block_idx: int) -> str:
    """Path of a chain-entry checkpoint (see ``save_chain_state``)."""
    return _chain_state_path(results_dir or "", group, block_idx)


def _to_cpu_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_recursive(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu_recursive(v) for k, v in obj.items()}
    return obj


def _to_device_recursive(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device_recursive(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device_recursive(v, device) for k, v in obj.items()}
    return obj


def save_chain_state(results_dir: str, group: int, block_idx: int, hidden) -> None:
    """Checkpoint the chained hidden state entering block ``block_idx``.

    The hidden state is the only live chain variable across blocks (masks and
    positions are re-sourced from the per-block pre-cache every iteration), so
    saving it lets a resumed worker skip already-tuned prefix blocks with no
    replay. ``hidden`` may be a tensor, a per-sample list/tuple, or a dict --
    block replay formats vary by model -- so it is persisted recursively and
    restored with the same structure.
    """
    os.makedirs(results_dir, exist_ok=True)
    payload = {"hidden": _to_cpu_recursive(hidden)}
    # unique temp suffix: concurrent replays of the same prefix (several
    # workers rebuilding the chain from the group entry) publish identical
    # checkpoint names; a shared temp file would race between one worker's
    # torch.save and another's os.replace
    path = _chain_state_path(results_dir, group, block_idx)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def delete_chain_state(results_dir: str, group: int, block_idx: int) -> None:
    """Remove a chain checkpoint.

    Callers must ensure the resume manifest no longer needs it (see
    ``CompressionOrchestrator._bp_prune_chain_entry``): ``chain_g{g}_b{k}``
    is the hard-link source for the commit of block ``k - 1``, so it may
    only be deleted once the manifest frontier has committed through it.
    Missing files are a no-op.
    """
    path = _chain_state_path(results_dir, group, block_idx)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def load_chain_state(results_dir: str, group: int, block_idx: int, device: str):
    """Restore a chained hidden state, or ``None`` when absent."""
    if not results_dir:
        return None
    path = _chain_state_path(results_dir, group, block_idx)
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _to_device_recursive(payload["hidden"], device)


def save_signature(results_dir: str, signature: str) -> None:
    """Persist the run signature the results were produced under."""
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "signature.txt"), "w", encoding="utf-8") as f:
        f.write(signature)


def signature_matches(results_dir: str, signature: str) -> bool:
    """True when stored signature is absent or equal (absent = first run)."""
    path = os.path.join(results_dir, "signature.txt")
    if not os.path.exists(path):
        return True
    with open(path, encoding="utf-8") as f:
        return f.read().strip() == signature.strip()


def shared_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "shared")


def save_shared_nonblocks(model, block_prefixes, path: str) -> int:
    """Persist non-block params/buffers once; workers mmap them (shared pages)."""
    import torch  # noqa: PLC0415

    def _in_block(name):
        return any(name == pref or name.startswith(pref + ".") for pref in block_prefixes)

    payload = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == "meta" or _in_block(name):
            continue
        payload[name] = tensor.detach().to("cpu").clone()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return len(payload)


def apply_shared_nonblocks(model, block_prefixes, path: str) -> int:
    """Load the parent-saved non-block tensors via mmap into ``model``.

    mmap-backed tensors share physical pages across all worker processes
    through the OS page cache, so N workers hold one copy of the embeddings
    (and anything else outside the blocks) instead of N copies.
    """
    import torch  # noqa: PLC0415
    from accelerate.utils import set_module_tensor_to_device  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    module_tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    applied = 0
    for name, value in payload.items():
        current = module_tensors.get(name)
        if current is None or current.device.type == "meta":
            set_module_tensor_to_device(model, name, "cpu", value=value, dtype=value.dtype)
            applied += 1
    return applied


def save_shared_inputs(payload: dict, path: str) -> None:
    """Persist the parent's calibration inputs + synced context snapshot."""
    import torch  # noqa: PLC0415

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_shared_inputs(path: str) -> dict:
    """Mmap-load the shared calibration inputs (tensors share pages).

    ``weights_only=True`` suffices: the payload is tensors and primitives
    (see ``_snapshot_calib_state``)."""
    import torch  # noqa: PLC0415

    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def launch_is_reexecutable(argv: Optional[Sequence[str]]) -> bool:
    """Whether re-running ``argv`` rebuilds this process's quantize entry.

    Workers are spawned by re-executing the parent's command line, so only
    launch forms that rebuild the auto-round entry point may enable parallel
    tuning: the package's own ``-m auto_round`` main script or a Python driver
    script (the re-executed script rebuilds AutoRound and the worker env
    branch routes it into queue-serve mode). Foreign launchers (pytest,
    jupyter, shell wrappers) re-run an unrelated program and are rejected.
    """
    if not argv:
        return False
    argv0 = str(argv[0])
    if argv0.endswith("__main__.py"):
        from pathlib import PurePath

        return PurePath(argv0).parent.name == "auto_round"
    return argv0.endswith(".py")


def block_parallel_tuning_enabled(
    *,
    enabled: bool,
    need_quanted_input: bool,
    super_group_size: Optional[int],
    nblocks: int,
    n_block_groups: int,
    is_immediate_packing: bool,
    is_immediate_saving: bool,
    n_blocks_total: int,
    argv: Optional[Sequence[str]],
) -> Optional[str]:
    """Decide whether block-parallel tuning may run in this process.

    Returns ``None`` when enabled, the string ``"disabled"`` when the feature
    is simply off (normal serial path -- never an error), or a human-readable
    reason when it is ON but a guard blocks execution (the caller should fail
    loudly rather than silently fall back to serial).

    Parallel tuning re-execs the same CLI in worker processes, so it requires
    a reconstructible command line (CLI usage). All conditions that make blocks
    order-dependent or the save path non-mergeable block it.
    """
    if not enabled:
        return "disabled"
    if envs.AR_BLOCK_PARALLEL_WORKER:
        return "worker process (no recursive parallelism)"
    if not launch_is_reexecutable(argv):
        return "no reconstructible command line (API usage or a launcher that cannot re-run auto-round)"
    if need_quanted_input:
        return (
            "quantized-input chain (sequential replay) is active; blocks are order-dependent. "
            "Pass --no-enable_quanted_input for FP-chain calibration"
        )
    if super_group_size is not None:
        return "super_group_size is set"
    if nblocks != 1:
        return f"nblocks={nblocks} != 1"
    if not (is_immediate_packing and is_immediate_saving):
        return "requires immediate packing and saving (low-mem/disk-stream flow)"
    if n_block_groups == 0 or n_blocks_total < 2:
        return "fewer than 2 blocks to tune"
    return None
