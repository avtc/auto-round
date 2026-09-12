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

"""Batched zero-shot (iters=0) search driver.

Layers whose staged key -- (device, weight shape/dtype, bits, group size,
symmetry, threshold, resolved quant callable) -- matches are stacked along a
leading dim and quantized with ONE ``weight_quant_func`` call. The quant math
is row-independent, and the per-column imatrix is pre-expanded to each
module's full weight shape before stacking so the quant function's internal
flatten-expand becomes an identity: results are bit-identical to the serial
``unwrapper({})`` path. Write-back goes through the wrapper's ``_apply_qdq``
(the unwrapper's own conventions), so serial and batched outputs can never
drift apart.
"""

from collections import OrderedDict

import torch

from auto_round.algorithms.quantization.search_shard import (
    _batch_cap,
    _fn_key,
    dump_oom_tensor_census_,
    group_items_by_device,
    pick_search_worker_devices,
    run_items_by_device,
)
from auto_round.logger import logger
from auto_round.utils import set_module


def _staged_weight(wrapper):
    """The weight exactly as _qdq_weight would pass it to the quant func."""
    weight = wrapper.orig_layer.weight
    import transformers

    if type(wrapper.orig_layer) == transformers.pytorch_utils.Conv1D:
        weight = weight.t()
    return weight


def _staged_imatrix(wrapper, weight):
    """Per-column imatrix pre-expanded to the full weight shape (or None).

    The quant functions flatten the imatrix and expand it across the whole
    tensor; pre-expanding per module keeps that flatten-expand an identity
    under stacking, which is what makes the batched call elementwise-equal to
    the per-module calls.
    """
    im = getattr(wrapper.orig_layer, "imatrix", None)
    if im is None:
        return None
    im = im.to(weight.device, weight.dtype)
    if im.dim() == 1 and weight.dim() == 2:
        return im.unsqueeze(0).expand_as(weight).contiguous()
    if im.shape == weight.shape:
        return im.contiguous()
    return im


def _staged_key(wrapper, weight, imatrix):
    layer = wrapper.orig_layer
    return (
        str(weight.device),
        tuple(weight.shape),
        str(weight.dtype),
        getattr(layer, "bits", None),
        getattr(layer, "group_size", None),
        getattr(layer, "sym", None),
        float(getattr(wrapper, "q_scale_thresh", 1e-5)),
        imatrix is not None,
        _fn_key(wrapper.weight_quant_func),
    )


@torch.no_grad()
def run_batched_rtn_search(model, staged, max_batch=None):
    """Finish deferred zero-shot wrappers on stacked same-shape batches.

    Args:
        model: The model tree (for set_module re-attachment).
        staged: list of (layer_name, wrapper) with the search deferred.
        max_batch: optional explicit chunk size.

    Returns:
        List of (layer_name, wrapper) that were NOT consumed and must be
        finished per-module by the caller (``wrapper.unwrapper({})``).
    """
    entries = []
    for layer_name, wrapper in staged:
        weight = _staged_weight(wrapper)
        imatrix = _staged_imatrix(wrapper, weight)
        entries.append({"name": layer_name, "w": wrapper, "weight": weight, "im": imatrix})

    device_groups = OrderedDict()
    for e in entries:
        device_groups.setdefault(str(e["weight"].device), []).append(e)

    def _finish_one(e):
        layer = e["w"].unwrapper({})
        set_module(model, e["name"], layer)

    # Offload pass: the searches are weight-local, so chunks may run on any
    # device with headroom (the zero-shot lane leaves every non-home GPU idle).
    # Chunks are assigned round-robin over viable worker devices; the weights
    # and imatrices move to the worker for the stacked call and results write
    # back cross-device through _apply_qdq.
    chunks = []  # (home_device, entries)
    for dev, batch in device_groups.items():
        by_key = OrderedDict()
        for e in batch:
            by_key.setdefault(_staged_key(e["w"], e["weight"], e["im"]), []).append(e)
        for _key, group in by_key.items():
            if len(group) < 2:
                chunks.append((dev, group))
                continue
            cap = _batch_cap(
                [
                    type("S", (), {"_deferred_search_inputs": (e["weight"], None, None, e["im"], None, None)})()
                    for e in group
                ],
                dev,
                None,
            )
            for start in range(0, len(group), cap):
                chunks.append((dev, group[start : start + cap]))

    def _chunk_working_set(chunk):
        e0 = chunk[0]
        per = (e0["weight"].numel() + (e0["im"].numel() if e0["im"] is not None else 0)) * 4 * 4
        return per * len(chunk)

    buckets = OrderedDict()
    rr = 0
    for dev, chunk in chunks:
        workers = pick_search_worker_devices(_chunk_working_set(chunk), home_device=dev)
        worker = workers[rr % len(workers)] if workers else dev
        rr += 1
        buckets.setdefault(str(worker), []).append(chunk)
    if buckets:
        _n_chunk = sum(len(c) for cs in buckets.values() for c in cs)
        logger.debug(
            "[rtn-batch] %d chunks over workers [%s]",
            _n_chunk,
            ", ".join(f"{w}:{len(cs)}" for w, cs in buckets.items()),
        )

    def _run_chunk(chunk):
        w0 = chunk[0]["w"]
        dev = str(chunk[0]["weight"].device)
        worker = str(next(wk for wk, cs in buckets.items() if any(c is chunk for c in cs)))
        if len(chunk) == 1:
            layer = chunk[0]["w"].unwrapper({})
            set_module(model, chunk[0]["name"], layer)
            return
        weights = [e["weight"] for e in chunk]
        ims = [e["im"] for e in chunk]
        if worker != dev:
            weights = [w.to(worker) for w in weights]
            ims = [im.to(worker) if im is not None else None for im in ims]
        stacked_w = torch.stack(weights)
        stacked_im = None
        if ims and ims[0] is not None:
            stacked_im = torch.stack(ims)
        kwargs = w0._quant_call_kwargs(
            torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), imatrix_override=stacked_im
        )
        kwargs = _relocate_tensor_kwargs(kwargs, str(stacked_w.device))
        try:
            qdq, scale, zp = w0.weight_quant_func(stacked_w, **kwargs)
        except torch.OutOfMemoryError:
            logger.warning(
                "[rtn-batch] stacked search OOM (%d modules); finishing this chunk per-module "
                "(shrink batches with AR_WRAP_SEARCH_BATCH_GB or disable with AR_DISABLE_SEARCH_SHARD=1)",
                len(chunk),
            )
            dump_oom_tensor_census_("rtn batched search")
            for e in chunk:
                layer = e["w"].unwrapper({})
                set_module(model, e["name"], layer)
            return
        n = len(chunk)
        scale_parts = _split_leading(scale, n)
        zp_parts = _split_leading(zp, n)
        for i, e in enumerate(chunk):
            e["w"]._apply_qdq(qdq[i], scale_parts[i], zp_parts[i])
            set_module(model, e["name"], e["w"].orig_layer)

    def _run_worker(_worker_key, worker_chunks):
        for chunk in worker_chunks:
            _run_chunk(chunk)

    if len(buckets) > 1:
        keyed = group_items_by_device(list(buckets.values()), device_of=lambda cs: str(cs[0][0]))
        run_items_by_device(keyed, lambda _idx, cs: _run_worker(str(cs[0][0]), cs))
    else:
        for wk, wcs in buckets.items():
            _run_worker(wk, wcs)
    return []


def _relocate_tensor_kwargs(kwargs: dict, device: str) -> dict:
    """Move accelerator-valued kwargs onto the compute device.

    The weight-local searches normally run on the weight's own device, so
    tensor kwargs (e.g. an nv-fp4 ``global_scale``) live wherever the layer
    lives; under worker offload the stacked call runs elsewhere and a
    left-behind operand would fault at the first cross-device op (the mapped-
    lane init_scale crash class). CPU scalars broadcast and are left alone.
    """
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor) and val.device.type != "cpu" and str(val.device) != device:
            kwargs[key] = val.to(device)
    return kwargs


def _split_leading(result, n):
    """Split a quant result along the (possibly flattened) leading batch dim.

    Stacked calls may return per-module rows flattened into the leading dim
    (e.g. scale ``[N*out, 1]``) or keep a batch dim (``[N, out, groups]``);
    both reshape to ``[N, -1, ...]`` identically. Scalars/dicts pass through.
    """
    if not isinstance(result, torch.Tensor):
        return [result] * n
    if result.dim() == 0 or result.numel() == 1:
        return [result] * n
    return result.reshape(n, -1, *result.shape[1:])
