# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""OOM diagnostics: tensor census, pluggable anywhere.

Three entry points, cheapest first:

* ``oom_census("context")`` -- a few-line context manager around any
  suspicious frame (including code this repo does not own); prints the
  census and re-raises.
* ``dump_oom_tensor_census_("context")`` -- the bare one-liner for
  existing ``except torch.OutOfMemoryError`` blocks.
* ``install_oom_census_hook()`` -- install once (the CLI does); fires
  for ANY uncaught CUDA OOM on any thread, wherever it escapes.

All are best effort: the census swallows its own failures and never
masks the original error.
"""

from contextlib import contextmanager

import torch

from auto_round.logger import logger


def _group_tensors_by_shape(objs) -> list:
    """(device, dtype, shape) -> [count, bytes] over accelerator tensors; sorted by bytes.

    Covers every non-cpu accelerator (cuda, hpu, xpu, mps, ...): the census
    must name the residents on whichever device ran out.
    """

    def _shape_key(shape) -> tuple:
        try:
            return tuple(int(d) for d in shape)
        except Exception:  # symbolic dims (SymInt) or exotic shapes: stringify
            return (str(tuple(shape)),)

    groups: dict = {}
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type in ("cpu", "meta"):
                continue
            key = (str(obj.device), str(obj.dtype), _shape_key(obj.shape))
            nbytes = obj.numel() * obj.element_size()
        except Exception:  # one unreadable tensor must never kill the census
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = [1, nbytes]
        else:
            g[0] += 1
            g[1] += nbytes
    return sorted(groups.items(), key=lambda kv: -kv[1][1])


def _representatives(groups, objs):
    """One representative tensor per group, drawn from a single scan."""
    wanted = {g[0] for g in groups}  # group keys are (device, dtype, shape)
    seen = {}
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type in ("cpu", "meta"):
                continue
            key = (str(obj.device), str(obj.dtype), _shape_key_public(obj.shape))
        except Exception:
            continue
        if key in wanted and key not in seen:
            seen[key] = obj
    for gk, meta in groups:
        if gk in seen:
            yield seen[gk], (gk, meta)


def _shape_key_public(shape):
    try:
        return tuple(int(d) for d in shape)
    except Exception:
        return (str(tuple(shape)),)


def _describe_referrers(tensor, limit=4):
    """Short descriptions of what holds ``tensor`` (best effort, frames skipped)."""
    import gc as _gc

    out = []
    try:
        for ref in _gc.get_referrers(tensor):
            t = type(ref)
            if t in (dict,):
                keys = [str(k) for k in list(ref.keys())[:4]]
                out.append(f"dict[{','.join(keys)}]" + (f"(len={len(ref)})" if len(keys) < len(ref) else ""))
            elif t in (list, tuple, set):
                out.append(f"{t.__name__}(len={len(ref)})")
            else:
                name = getattr(ref, "__class__", t).__name__
                out.append(name)
            if len(out) >= limit:
                break
    except Exception as e:  # pragma: no cover - diagnostics must not mask the OOM
        out.append(f"<referrer scan failed: {e}>")
    return out


def dump_oom_tensor_census_(context: str = "") -> None:
    """Tensor census at OOM time: per-device allocator state + top tensor groups.

    Names the accumulating residents (a leak shows as one (shape, dtype)
    group growing across blocks). Best effort: never masks the OOM itself.
    """
    import gc

    try:
        try:
            for idx in range(torch.cuda.device_count()):
                logger.error(
                    "[oom] cuda:%s allocated=%.2fGiB reserved=%.2fGiB",
                    idx,
                    torch.cuda.memory_allocated(idx) / 2**30,
                    torch.cuda.memory_reserved(idx) / 2**30,
                )
        except Exception:  # pragma: no cover - allocator stats are cuda-only
            pass
        per_device: dict = {}
        top = _group_tensors_by_shape(gc.get_objects())
        for (_dev, _dt, _shape), (_cnt, _nb) in top:
            per_device[_dev] = per_device.get(_dev, 0) + _nb
        for _dev, _nb in sorted(per_device.items(), key=lambda kv: -kv[1]):
            logger.error("[oom] %s live tensors ≈ %.2fGiB", _dev, _nb / 2**30)
        for (_dev, _dt, _shape), (_cnt, _nb) in top[:8]:
            logger.error("[oom] %s %s %s x%d = %.2fGiB", _dev, _dt, list(_shape), _cnt, _nb / 2**30)
        # name the holders: referrers of one representative tensor per top group
        for rep, (( _dev, _dt, _shape), (_cnt, _nb)) in _representatives(top[:3], gc.get_objects()):
            for desc in _describe_referrers(rep):
                logger.error("[oom]   %s %s held by: %s", _dev, list(_shape), desc)
    except Exception as e:  # pragma: no cover - diagnostics must not mask the OOM
        logger.error("[oom] tensor census failed (%s)", e)


def _is_oom(exc: BaseException) -> bool:
    """torch.OutOfMemoryError (cuda, modern xpu) or message-based OOM (hpu et al.)."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@contextmanager
def oom_census(context: str = ""):
    """Census-on-OOM context manager: plug around any frame, upstream code included.

    .. code-block:: python

        with oom_census("block collection forward"):
            out = block_forward_fn(block, fp_inputs, input_others)
    """
    try:
        yield
    except Exception as exc:
        if _is_oom(exc):
            dump_oom_tensor_census_(context)
        raise


_OOM_HOOK_INSTALLED = False


def install_oom_census_hook() -> bool:
    """Install the last-resort census for uncaught CUDA OOMs (any thread).

    Fires wherever a ``torch.OutOfMemoryError`` escapes to the top of the
    process -- including frames this repo does not own -- printing the tensor
    census before the default exception reporting takes over. Idempotent;
    returns True when it installed the hooks.
    """
    global _OOM_HOOK_INSTALLED
    if _OOM_HOOK_INSTALLED:
        return False
    import sys
    import threading

    prior_sys = sys.excepthook
    prior_the = threading.excepthook

    def _sys_hook(tp, val, tb):
        try:
            if _is_oom(val):
                dump_oom_tensor_census_("uncaught")
        except Exception:  # pragma: no cover - diagnostics must not mask the error
            pass
        prior_sys(tp, val, tb)

    def _the_hook(args):
        try:
            if _is_oom(args.exc_value):
                name = args.thread.name if args.thread is not None else "?"
                dump_oom_tensor_census_(f"uncaught (thread {name})")
        except Exception:  # pragma: no cover - diagnostics must not mask the error
            pass
        prior_the(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _the_hook
    _OOM_HOOK_INSTALLED = True
    return True
