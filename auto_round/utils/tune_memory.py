# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact VRAM-requirement calculation for a block tune, per module schema.

Every deterministic term (weights, rounding values, min/max scales, their
gradients, the best-params snapshot) is computed from the module's own
shapes and scheme fields - no multipliers. The one term autograd decides
(saved-for-backward intermediates of the fake-quant math) is schema- and
build-dependent, so it is MEASURERED in-process: a tiny replica wrapper of
the same class and scheme runs one forward (and, on accelerator devices,
one backward under peak-memory stats) under ``saved_tensors_hooks``; the
resulting bytes-per-value-byte ratios scale linearly to the real modules.
Ratios are cached per (wrapper class, scheme signature, torch build,
device type) so each distinct schema pays the probe once per process.
"""

from __future__ import annotations

import torch

from auto_round.logger import logger

_PROBE_CACHE: dict[tuple, dict] = {}


def _scheme_signature(layer) -> tuple:
    """Scheme fields that change the wrapper's quant math (and its saved set)."""
    return (
        getattr(layer, "data_type", "int"),
        getattr(layer, "bits", 4),
        getattr(layer, "group_size", 128),
        bool(getattr(layer, "sym", True)),
        getattr(layer, "super_bits", None),
        getattr(layer, "super_group_size", None),
        str(getattr(layer, "scale_dtype", None)),
    )


def exact_tune_bytes(layer, enable_minmax_tuning: bool) -> dict:
    """Deterministic GPU-resident bytes for one (unwrapped) quantizable layer.

    Covers the steady terms of a tune: the weight in its working dtype, the
    fp32 rounding value, the min/max scale parameters when minmax tuning is
    on, their gradients, and a same-size best-params snapshot. Saved-for-
    backward intermediates are NOT included here - they are probe-measured.
    """
    weight = getattr(layer, "weight", None)
    numel = weight.numel() if weight is not None else 0
    weight_bytes = numel * (weight.element_size() if weight is not None else 2)
    value_bytes = numel * 4  # fp32 rounding parameter
    group_size = getattr(layer, "group_size", 128)
    if isinstance(group_size, int) and 0 < group_size <= numel:
        groups = (numel + group_size - 1) // group_size
    else:  # per-tensor layout (group_size 0)
        groups = 1
    minmax_bytes = groups * 4 * 2 if enable_minmax_tuning else 0  # min_scale + max_scale
    grads_bytes = value_bytes + minmax_bytes
    return {
        "weight": weight_bytes,
        "value": value_bytes,
        "minmax": minmax_bytes,
        "grads": grads_bytes,
        "snapshot": value_bytes + minmax_bytes,
    }


def probe_saved_ratio(
    wrapper_cls, layer, device, enable_minmax_tuning: bool, enable_torch_compile: bool = False
) -> dict:
    """Measure this build's saved/transient bytes per value-byte for one scheme.

    Builds a tiny replica (256x512) stamped with ``layer``'s scheme fields,
    wraps it with ``wrapper_cls`` (the same class the tune will use), and
    runs one forward under ``saved_tensors_hooks``. On CUDA devices a full
    forward+backward under reset/max peak-memory stats additionally yields
    the backward-transient-inclusive ratio. Returns ``{"fwd": r, "peak": r
    or None}``; ``None`` ratios mean "unmeasured" and callers must treat the
    saved term as unknown rather than zero.
    """
    signature = (
        getattr(wrapper_cls, "__qualname__", repr(wrapper_cls)),
        _scheme_signature(layer),
        enable_minmax_tuning,
        bool(enable_torch_compile),
        torch.__version__,
        str(device).split(":")[0],
    )
    if signature in _PROBE_CACHE:
        return _PROBE_CACHE[signature]

    ratios = {"fwd": None, "peak": None}
    try:
        probe_layer = torch.nn.Linear(512, 256, bias=False)
        for attr in (
            "data_type",
            "bits",
            "group_size",
            "sym",
            "super_bits",
            "super_group_size",
            "scale_dtype",
        ):
            if hasattr(layer, attr):
                setattr(probe_layer, attr, getattr(layer, attr))
        probe_layer = probe_layer.to(
            getattr(layer, "weight", probe_layer.weight).dtype if layer.weight is not None else torch.bfloat16
        )
        wrapper = wrapper_cls(
            probe_layer,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_torch_compile=False,  # never compile the probe
            device=torch.device(device),
        )
        value = wrapper.params.get("value")
        value_bytes = value.numel() * value.element_size()
        saved = {}
        from torch.autograd.graph import saved_tensors_hooks

        def _pack(t):
            try:
                saved[t.untyped_storage().data_ptr()] = t.numel() * t.element_size()
            except Exception:  # pylint: disable=broad-except
                pass
            return t

        x = torch.randn(2, 16, 512, dtype=probe_layer.weight.dtype, requires_grad=True)
        with saved_tensors_hooks(_pack, lambda t: t):
            y = wrapper(x)
            loss = y.float().pow(2).mean()
        fwd_saved = sum(n for n in saved.values() if n >= value_bytes)
        ratios["fwd"] = fwd_saved / value_bytes if value_bytes else None
        if isinstance(device, str):
            dev = torch.device(device)
        else:
            dev = device
        if dev.type == "cuda":
            # backward transients on the real backend: peak over fwd+bwd
            # minus the steady param/grad buffers, per value byte
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            before = torch.cuda.memory_allocated(dev)
            y = wrapper(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            torch.cuda.synchronize(dev)
            peak_extra = torch.cuda.max_memory_allocated(dev) - before
            steady = value_bytes + sum(
                t.numel() * t.element_size() for t in wrapper.params.values() if isinstance(t, torch.Tensor)
            )
            steady += x.numel() * x.element_size() + y.numel() * y.element_size()
            ratios["peak"] = max(0.0, (peak_extra - steady) / value_bytes) if value_bytes else None
        del wrapper, probe_layer, x
    except Exception as e:  # pylint: disable=broad-except - probe is advisory
        logger.warning("[tune-mem] saved-ratio probe failed (%s: %s); saved term unknown", type(e).__name__, e)
        ratios = {"fwd": None, "peak": None}
    _PROBE_CACHE[signature] = ratios
    return ratios


def predict_block_tune_peak(
    layers,
    device,
    enable_minmax_tuning: bool,
    wrapper_cls=None,
    allocated_now: int = 0,
    act_est_bytes: int = 0,
    frag_budget_bytes: int = 0,
    snapshot_on_host: bool = False,
    enable_torch_compile: bool = False,
) -> dict:
    """Exact steady terms + probe-measured saved term for a set of layers.

    ``layers`` are the (unwrapped) quantizable modules about to tune. Returns
    a bytes dict: ``total`` is the predicted GPU peak = current allocation +
    per-module terms (saved via the probe ratios, snapshot excluded when the
    ladder parks it on host) + activation estimate + fragmentation budget.
    A ``None`` ratio leaves ``saved`` None and ``total`` None - unknown, not
    zero - so callers gate conservatively instead of trusting a guess.
    """
    per_module = [exact_tune_bytes(layer, enable_minmax_tuning) for layer in layers]
    ratio = None
    if layers and wrapper_cls is not None:
        ratio = probe_saved_ratio(wrapper_cls, layers[0], device, enable_minmax_tuning, enable_torch_compile)
    saved_total = None
    if ratio and ratio.get("peak") is not None:
        saved_total = sum(int(pm["value"] * ratio["peak"]) for pm in per_module)
    elif ratio and ratio.get("fwd") is not None:
        saved_total = sum(int(pm["value"] * ratio["fwd"]) for pm in per_module)
    fixed = sum(pm["weight"] + pm["value"] + pm["minmax"] + pm["grads"] for pm in per_module)
    snapshot = 0 if snapshot_on_host else sum(pm["snapshot"] for pm in per_module)
    out = {
        "modules": len(per_module),
        "fixed": fixed,
        "snapshot": snapshot,
        "saved": saved_total,
        "act": act_est_bytes,
        "frag": frag_budget_bytes,
        "allocated_now": allocated_now,
    }
    if saved_total is None:
        out["total"] = None
    else:
        out["total"] = allocated_now + fixed + snapshot + saved_total + act_est_bytes + frag_budget_bytes
    return out
