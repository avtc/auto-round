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

    def _run_device(device_key, batch):
        by_key = OrderedDict()
        for e in batch:
            by_key.setdefault(_staged_key(e["w"], e["weight"], e["im"]), []).append(e)
        for _key, group in by_key.items():
            if len(group) < 2:
                _finish_one(group[0])
                continue
            cap = _batch_cap(
                [
                    type("S", (), {"_deferred_search_inputs": (e["weight"], None, None, e["im"], None, None)})()
                    for e in group
                ],
                device_key,
                max_batch,
            )
            for start in range(0, len(group), cap):
                chunk = group[start : start + cap]
                w0 = chunk[0]["w"]
                stacked_w = torch.stack([e["weight"] for e in chunk])
                stacked_im = None
                if chunk[0]["im"] is not None:
                    stacked_im = torch.stack([e["im"] for e in chunk])
                kwargs = w0._quant_call_kwargs(
                    torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), imatrix_override=stacked_im
                )
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
                        _finish_one(e)
                    continue
                n = len(chunk)
                scale_parts = _split_leading(scale, n)
                zp_parts = _split_leading(zp, n)
                for i, e in enumerate(chunk):
                    e["w"]._apply_qdq(qdq[i], scale_parts[i], zp_parts[i])
                    if hasattr(e["w"].orig_layer, "global_name"):
                        pass  # global_name lives on orig_layer already; unwrapper copies it onto itself
                    set_module(model, e["name"], e["w"].orig_layer)

    if len(device_groups) > 1:
        keyed = group_items_by_device(list(device_groups.values()), device_of=lambda es: str(es[0]["weight"].device))
        run_items_by_device(keyed, lambda _idx, es: _run_device(str(es[0]["weight"].device), es))
    else:
        for dev, batch in device_groups.items():
            _run_device(dev, batch)
    return []


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
