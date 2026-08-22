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
command in worker processes, each pinned to one GPU and restricted to a
contiguous span of blocks; workers fast-forward through their span's prefix
with no-grad original-weights forwards (faithful to the serial chain and ~1000x
cheaper than tuning) and dump tuned ``scale``/``zp`` per layer instead of
writing shards. The parent merges the dumps and runs the ordinary
immediate-pack/save flow.

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

def _available_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024**2)
    except OSError:
        pass
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().available / (1024**3)
    except Exception:  # noqa: BLE001
        return None


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
    """Spawn one queue-serving worker per GPU, staggered.

    Spawns are staggered by a fixed delay so the per-worker model-build and
    dataset-preprocessing host-RAM transients do not spike together on
    small-RAM hosts.
    """
    import time as _time

    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for rank, gpu in enumerate(gpu_ids):
        if rank:
            _time.sleep(5.0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # AR_RESUME_DIR is deliberately inherited: workers derive their
        # per-block checkpoint dir from it (same user-facing flag as serial).
        env["AR_ENABLE_AUTO_SCHEME_PARALLEL"] = "0"  # single-GPU workers never fan out
        env.update(extra_env)
        log_path = os.path.join(log_dir, f"worker_{rank}.log")
        logger.info(
            "block-parallel tuning: worker %d -> gpu %d (log %s)", rank, gpu, log_path
        )
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        procs.append(
            subprocess.Popen(  # noqa: S603
                worker_command(argv, device=gpu), env=env, stdout=log_file, stderr=subprocess.STDOUT
            )
        )
    return procs


def log_worker_rss(procs: List[subprocess.Popen], tag: str = "") -> None:
    """Best-effort per-worker RSS log line (Linux /proc; silent elsewhere)."""
    parts = []
    for idx, proc in enumerate(procs):
        if proc.poll() is not None:
            parts.append(f"w{idx}=dead")
            continue
        try:
            with open(f"/proc/{proc.pid}/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts.append(f"w{idx}={int(line.split()[1]) / 1024:.1f}GiB")
                        break
        except OSError:
            return  # non-Linux or proc gone; skip silently
    if parts:
        logger.info("block-parallel worker RSS%s: %s", f" [{tag}]" if tag else "", " ".join(parts))

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


def _chain_state_path(results_dir: str, group: int, block_idx: int) -> str:
    return os.path.join(results_dir, f"chain_g{group}_b{block_idx}.pt")


def save_block_results(results_dir: str, block_name: str, layers: dict) -> None:
    """Atomically persist one block's tuned ``{layer_name: {scale, zp}}``.

    One file per block is the unit of resumption: block tuning is independent
    (inputs chain through original-weights outputs), so any subset of completed
    blocks composes with a rerun that finishes the rest.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = _block_results_path(results_dir, block_name)
    tmp = path + ".tmp"
    torch.save(layers, tmp)
    os.replace(tmp, path)


def has_block_results(results_dir: str, block_name: str) -> bool:
    return os.path.exists(_block_results_path(results_dir, block_name))


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
            merged[name] = {"scale": entry["scale"], "zp": entry["zp"]}
    return merged


def missing_result_blocks(results_dir: str, all_blocks: "Sequence[list]") -> "List[str]":
    """Blocks (flattened) with no result file yet."""
    if not results_dir:
        return [block for group in all_blocks for block in group]
    return [block for group in all_blocks for block in group if not has_block_results(results_dir, block)]


def chain_state_exists(results_dir: str, group: int, block_idx: int) -> bool:
    return os.path.exists(_chain_state_path(results_dir or "", group, block_idx))


def save_chain_state(results_dir: str, group: int, block_idx: int, hidden) -> None:
    """Checkpoint the chained hidden state entering block ``block_idx``.

    The hidden state is the only live chain variable across blocks (masks and
    positions are re-sourced from the per-block pre-cache every iteration), so
    saving it lets a resumed worker skip already-tuned prefix blocks with no
    replay. Non-tensor chain payloads are skipped (nothing to restore).
    """
    if not isinstance(hidden, torch.Tensor):
        return
    os.makedirs(results_dir, exist_ok=True)
    payload = {"hidden": hidden.detach().to("cpu"), "dtype": str(hidden.dtype)}
    tmp = _chain_state_path(results_dir, group, block_idx) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, _chain_state_path(results_dir, group, block_idx))


def load_chain_state(results_dir: str, group: int, block_idx: int, device: str):
    """Restore a chained hidden state, or ``None`` when absent/non-tensor."""
    if not results_dir:
        return None
    path = _chain_state_path(results_dir, group, block_idx)
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    dtype = getattr(torch, payload["dtype"].replace("torch.", ""), None)
    hidden = payload["hidden"]
    if dtype is not None:
        hidden = hidden.to(dtype)
    return hidden.to(device)


def queue_dir(results_dir: str) -> str:
    return os.path.join(results_dir, "queue")


def _job_sort_key(fname: str):
    return int(fname.split("_", 1)[0])


def write_job(qdir: str, seq: int, job_type: str, group: int, **fields) -> None:
    """Write a dispatch job file. ``job_type`` is ``tune`` or ``produce``."""
    os.makedirs(qdir, exist_ok=True)
    payload = {"seq": seq, "job_type": job_type, "group": group}
    payload.update(fields)
    path = os.path.join(qdir, f"{seq:06d}.job")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def claim_next_job(qdir: str):
    """Atomically claim the lowest-seq pending job (rename .job -> .claimed)."""
    if not os.path.isdir(qdir):
        return None
    for fname in sorted(os.listdir(qdir)):
        if not fname.endswith(".job"):
            continue
        path = os.path.join(qdir, fname)
        claimed = path + f".claimed-{os.getpid()}"
        try:
            os.rename(path, claimed)  # atomic claim on POSIX
        except OSError:
            continue  # raced with another worker
        with open(claimed, encoding="utf-8") as f:
            job = json.load(f)
        job["_claimed_path"] = claimed
        return job
    return None


def job_done(qdir: str, job: dict) -> None:
    """Remove a claimed job file after successful completion."""
    try:
        os.remove(job.get("_claimed_path", ""))
    except OSError:
        pass


def job_failed(qdir: str, job: dict, error: str) -> None:
    """Record a failure next to the claim; the parent requeues or reports."""
    path = job.get("_claimed_path", "")
    if not path:
        return
    with open(path + ".failed", "w", encoding="utf-8") as f:
        f.write(error)


def list_failed(qdir: str) -> List[str]:
    if not os.path.isdir(qdir):
        return []
    return [f for f in sorted(os.listdir(qdir)) if f.endswith(".failed")]


def list_pending(qdir: str) -> List[str]:
    if not os.path.isdir(qdir):
        return []
    return [f for f in sorted(os.listdir(qdir)) if f.endswith(".job")]


def write_stop(qdir: str) -> None:
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, "STOP"), "w", encoding="utf-8") as f:
        f.write("stop")


def read_stop(qdir: str) -> bool:
    return os.path.exists(os.path.join(qdir, "STOP"))


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
    applied = 0
    for name, value in payload.items():
        module_tensors = dict(model.named_parameters()) | dict(model.named_buffers())
        current = module_tensors.get(name)
        if current is None or current.device.type == "meta":
            set_module_tensor_to_device(
                model, name, "cpu", value=value, dtype=value.dtype
            )
            applied += 1
    return applied


def save_shared_inputs(payload: dict, path: str) -> None:
    """Persist the parent's calibration inputs + synced context snapshot."""
    import torch  # noqa: PLC0415

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_shared_inputs(path: str) -> dict:
    """Mmap-load the shared calibration inputs (tensors share pages)."""
    import torch  # noqa: PLC0415

    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def block_parallel_tuning_enabled(
    *,
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

    Returns ``None`` when enabled, the string ``"env"`` when the feature flag
    is simply off (normal serial path -- never an error), or a human-readable
    reason when the flag is ON but a guard blocks execution (the caller should
    fail loudly rather than silently fall back to serial).

    Parallel tuning re-execs the same CLI in worker processes, so it requires
    a reconstructible command line (CLI usage). All conditions that make blocks
    order-dependent or the save path non-mergeable block it.
    """
    if not envs.AR_ENABLE_BLOCK_PARALLEL_TUNING:
        return "env"
    if envs.AR_BLOCK_PARALLEL_WORKER:
        return "worker process (no recursive parallelism)"
    if argv is None or len(argv) == 0:
        return "no reconstructible command line (API usage)"
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
