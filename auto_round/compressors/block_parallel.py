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
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

import torch

from auto_round import envs
from auto_round.utils import logger

Span = Tuple[int, int, int]  # (group_idx, start, end) -- end exclusive


def split_spans(group_sizes: Sequence[int], n_workers: int) -> List[List[Span]]:
    """Split blocks into contiguous per-worker spans without crossing groups.

    Blocks are assigned in order; each worker receives a near-equal count. A
    span never crosses a group boundary (each group chains its own inputs from
    its own cached entry point), so a group may contribute several spans.
    """
    total = sum(group_sizes)
    if total == 0 or n_workers <= 0:
        return []
    n_workers = min(n_workers, total)
    per = -(-total // n_workers)
    spans: List[List[Span]] = [[] for _ in range(n_workers)]
    worker, fill = 0, 0
    for group_idx, size in enumerate(group_sizes):
        start = 0
        while start < size:
            if fill >= per and worker < n_workers - 1:
                worker += 1
                fill = 0
            room = per - fill
            if room <= 0:  # last worker absorbs the remainder
                room = size - start
            end = min(size, start + room)
            spans[worker].append((group_idx, start, end))
            fill += end - start
            start = end
    return [s for s in spans if s]


def spans_to_json(spans: List[List[Span]]) -> str:
    return json.dumps([[list(t) for t in worker] for worker in spans])


def spans_from_json(text: str) -> List[List[Span]]:
    return [[(int(g), int(s), int(e)) for g, s, e in worker] for worker in json.loads(text)]


def eligible_gpus(min_free_gb: Optional[float] = None) -> List[int]:
    """CUDA devices with at least ``min_free_gb`` GiB free (heterogeneous-safe)."""
    if not torch.cuda.is_available():
        return []
    if min_free_gb is None:
        min_free_gb = float(os.getenv("AR_BLOCK_PARALLEL_MIN_FREE_GB", "18"))
    out = []
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:  # noqa: BLE001
            continue
        if free / (1024**3) >= min_free_gb:
            out.append(index)
    return out


def worker_command(argv: Sequence[str], device: Optional[int] = None) -> List[str]:
    """Re-exec command for a worker: same CLI invocation as the parent.

    When ``device`` is given (worker pinned via CUDA_VISIBLE_DEVICES), any
    ``--device_map`` argument is collapsed to ``0`` so multi-GPU maps from the
    parent invocation don't request invisible devices.
    """
    argv = list(argv)
    if argv and str(argv[0]).endswith(".py"):
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


def spawn_workers(
    argv: Sequence[str],
    spans: List[List[Span]],
    gpu_ids: List[int],
    results_dir: str,
) -> List[subprocess.Popen]:
    """Spawn one tuning worker per (span-set, gpu). Returns the process list."""
    os.makedirs(results_dir, exist_ok=True)
    procs = []
    for rank, (worker_spans, gpu) in enumerate(zip(spans, gpu_ids)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["AR_BLOCK_PARALLEL_WORKER"] = "1"
        env["AR_BLOCK_PARALLEL_SPANS"] = spans_to_json([worker_spans])
        env["AR_BLOCK_PARALLEL_RESULTS"] = os.path.join(results_dir, f"worker_{rank}.pt")
        env.pop("AR_RESUME_DIR", None)  # per-worker resume state is not supported
        # Never let a worker fan out its own AutoScheme scoring pool if the
        # scheme cache should miss (workers are single-GPU by construction).
        env["AR_ENABLE_AUTO_SCHEME_PARALLEL"] = "0"
        log_path = os.path.join(results_dir, f"worker_{rank}.log")
        logger.info("block-parallel tuning: worker %d -> gpu %d, spans %s (log %s)", rank, gpu, worker_spans, log_path)
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        procs.append(
            subprocess.Popen(  # noqa: S603
                worker_command(argv, device=gpu), env=env, stdout=log_file, stderr=subprocess.STDOUT
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


def load_worker_results(results_dir: str, n_workers: int) -> dict:
    """Merge per-worker tuned {layer_name: {scale, zp}} dumps into one dict."""
    merged: dict = {}
    for rank in range(n_workers):
        path = os.path.join(results_dir, f"worker_{rank}.pt")
        if not os.path.exists(path):
            continue
        data = torch.load(path, map_location="cpu", weights_only=True)
        for name, entry in data.items():
            merged[name] = {"scale": entry["scale"], "zp": entry["zp"]}
    return merged


def dump_worker_results(path: str, results: dict) -> None:
    """Worker-side: persist tuned scale/zp tensors for the parent to merge."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(results, path)


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
) -> bool:
    """Decide whether block-parallel tuning may run in this process.

    Parallel tuning re-execs the same CLI in worker processes, so it requires a
    reconstructible command line (CLI usage). All conditions that make blocks
    order-dependent or the save path non-mergeable disable it.
    """
    if not envs.AR_ENABLE_BLOCK_PARALLEL_TUNING:
        return False
    if envs.AR_BLOCK_PARALLEL_WORKER:
        return False  # never recurse
    if argv is None or len(argv) == 0:
        logger.info("block-parallel tuning disabled: no reconstructible command line (API usage)")
        return False
    if need_quanted_input:
        logger.info("block-parallel tuning disabled: quantized-input chain (sequential replay) is active")
        return False
    if super_group_size is not None:
        logger.info("block-parallel tuning disabled: super_group_size is set")
        return False
    if nblocks != 1:
        logger.info("block-parallel tuning disabled: nblocks=%d != 1", nblocks)
        return False
    if envs.AR_RESUME_DIR:
        logger.info("block-parallel tuning disabled: AR_RESUME_DIR is set")
        return False
    if not (is_immediate_packing and is_immediate_saving):
        logger.info(
            "block-parallel tuning disabled: requires immediate packing and saving (low-mem/disk-stream flow)"
        )
        return False
    if n_block_groups == 0 or n_blocks_total < 2:
        return False
    return True
