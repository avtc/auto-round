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
    """(device, dtype, shape) -> [count, bytes] over cuda tensors; sorted by bytes."""
    groups: dict = {}
    for obj in objs:
        if isinstance(obj, torch.Tensor) and obj.device.type == "cuda":
            key = (str(obj.device), str(obj.dtype), tuple(obj.shape))
            g = groups.get(key)
            if g is None:
                groups[key] = [1, obj.numel() * obj.element_size()]
            else:
                g[0] += 1
                g[1] += obj.numel() * obj.element_size()
    return sorted(groups.items(), key=lambda kv: -kv[1][1])


def dump_oom_tensor_census_(context: str = "") -> None:
    """Tensor census at OOM time: per-device allocator state + top tensor groups.

    Names the accumulating residents (a leak shows as one (shape, dtype)
    group growing across blocks). Best effort: never masks the OOM itself.
    """
    import gc

    try:
        for idx in range(torch.cuda.device_count()):
            logger.error(
                "[oom] cuda:%s allocated=%.2fGiB reserved=%.2fGiB",
                idx,
                torch.cuda.memory_allocated(idx) / 2**30,
                torch.cuda.memory_reserved(idx) / 2**30,
            )
        top = _group_tensors_by_shape(gc.get_objects())[:8]
        for (_dev, _dt, _shape), (_cnt, _nb) in top:
            logger.error("[oom] %s %s %s x%d = %.2fGiB", _dev, _dt, list(_shape), _cnt, _nb / 2**30)
    except Exception as e:  # pragma: no cover - diagnostics must not mask the OOM
        logger.error("[oom] tensor census failed (%s)", e)


@contextmanager
def oom_census(context: str = ""):
    """Census-on-OOM context manager: plug around any frame, upstream code included.

    .. code-block:: python

        with oom_census("block collection forward"):
            out = block_forward_fn(block, fp_inputs, input_others)
    """
    try:
        yield
    except torch.OutOfMemoryError:
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
            if isinstance(val, torch.OutOfMemoryError):
                dump_oom_tensor_census_("uncaught")
        except Exception:  # pragma: no cover - diagnostics must not mask the error
            pass
        prior_sys(tp, val, tb)

    def _the_hook(args):
        try:
            if isinstance(args.exc_value, torch.OutOfMemoryError):
                name = args.thread.name if args.thread is not None else "?"
                dump_oom_tensor_census_(f"uncaught (thread {name})")
        except Exception:  # pragma: no cover - diagnostics must not mask the error
            pass
        prior_the(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _the_hook
    _OOM_HOOK_INSTALLED = True
    return True
