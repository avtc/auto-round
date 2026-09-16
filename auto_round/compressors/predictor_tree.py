# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Predictor-tree (MTP) materialization and tuning for the data-driven path.

Some architectures carry Multi-Token-Prediction layers in the checkpoint that
``transformers`` never instantiates.  Upstream handles them post-save
(``copy_missing_tensors_from_source`` copies or RTN-quantizes the raw
tensors), which cannot *tune*: SignRound needs live modules and calibration
inputs.  This module materializes the tree as real modules - a deep-copied
sibling decoder layer plus prologue norms and the concat mixer built from
checkpoint tensors - so the standard block machinery can tune it as one
extra block:

    norm_e(e) concat norm_h(h) -> concat mixer -> decoder layer -> final norm

The embedding-side input ``e`` is synthesized from the chain's token ids
(position ``t`` consumes token ``t+1``'s embedding).  Trees are attached
under **checkpoint-spelled** paths so layer-config pins, packing, saving and
the missing-tensors pass all see the exact checkpoint names.
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch

from auto_round.logger import logger

__all__ = [
    "ensure_module_path",
    "synthesize_predictor_e",
    "predictor_forward",
    "bind_predictor_forward",
    "list_checkpoint_tensors",
    "load_checkpoint_tensor",
    "checkpoint_only_roots",
    "analyze_predictor_group",
    "pick_sibling_layer",
    "build_predictor_tree",
]


def ensure_module_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Walk/create intermediate shells for *path* and return the parent a leaf attaches to."""
    segments = path.split(".")
    parent = model
    for seg in segments[:-1]:
        child = parent._modules.get(seg)
        if child is None:
            if seg not in parent._modules and hasattr(parent, seg):
                delattr(parent, seg)  # stale plain attribute (e.g. mtp=None placeholder)
            child = torch.nn.Module()
            parent.add_module(seg, child)
        parent = child
    return parent


def synthesize_predictor_e(input_rows: torch.Tensor, embed=None) -> torch.Tensor:
    """Build the embedding-side predictor input (e-first concat).

    ``input_rows`` is either already-embedded chain rows or raw token ids
    together with ``embed``.  Position ``t`` consumes token ``t+1``'s
    embedding; the final position repeats the last row - the uniform
    convention of the verified predictor families.
    """
    if input_rows.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
        if embed is None:
            raise ValueError("token-id rows need the embedding module to build the predictor input")
        shifted = torch.cat([input_rows[:, 1:], input_rows[:, -1:]], dim=1)
        # calibration caches mark ignored positions with -100 (the standard
        # ignore index, incl. every sample's last position); those ids are
        # never valid lookups, so clamp before embedding. Their rows are
        # excluded from the loss by the valid-token mask.
        shifted = shifted.clamp_min(0)
        return embed(shifted)
    return torch.cat([input_rows[:, 1:], input_rows[:, -1:]], dim=1)


def _resolve_predictor_ref(shell: torch.nn.Module, spec):
    """Resolve a role ref: module handles pass through, string paths resolve
    LIVE against the model so wrappers the tuning machinery installs
    (WrapperLinear replacing the bare Linear) participate in the forward."""
    if isinstance(spec, str):
        model = getattr(shell, "_predictor_model", None)
        if model is None:
            raise RuntimeError("predictor ref given as a path but no model bound on the shell")
        return model.get_submodule(spec)
    return spec


def predictor_forward(shell: torch.nn.Module, hidden_states: torch.Tensor, **input_others):
    """Run a materialized predictor tree like a decoder block.

    The embedding-side input arrives as the ``_predictor_e`` auxiliary input
    (per-row tensors are concatenated here) or falls back to the value bound
    on the shell; every other keyword input passes through to the layer
    unchanged. Role refs resolve live (see ``_resolve_predictor_ref``) so the
    concat mixer's WrapperLinear is picked up once installed.
    """
    refs = getattr(shell, "_predictor_refs", None)
    if refs is None:
        raise RuntimeError("predictor forward called on a shell without bound role refs")
    e = input_others.pop("_predictor_e", None)
    if e is None:
        e = shell._predictor_e
    if e is None:
        raise RuntimeError(
            "predictor embedding-side input is not bound yet; synthesize it from the chain state before tuning"
        )
    if isinstance(e, (list, tuple)):
        e = torch.cat(list(e), dim=0)
    if e.device != hidden_states.device:
        e = e.to(hidden_states.device)
    x = torch.cat(
        [
            _resolve_predictor_ref(shell, refs["norm_e"])(e),
            _resolve_predictor_ref(shell, refs["norm_h"])(hidden_states),
        ],
        dim=-1,
    )
    x = _resolve_predictor_ref(shell, refs["fc"])(x)
    x = _resolve_predictor_ref(shell, refs["layer"])(x, **input_others)
    x = x[0] if isinstance(x, (tuple, list)) else x
    final_norm = refs.get("final_norm")
    return _resolve_predictor_ref(shell, final_norm)(x) if final_norm is not None else x


def bind_predictor_forward(
    shell: torch.nn.Module, refs: dict, e: torch.Tensor = None, model: torch.nn.Module = None
) -> None:
    """Attach the predictor forward and role refs to the group shell.

    ``refs`` maps role names to module handles OR module paths; paths resolve
    live at forward time so wrappers the tuning machinery installs later are
    picked up. The ref dict keeps handles out of ``named_modules`` (the real
    tree registers them once under their checkpoint paths), and the bound
    forward lets block-level machinery call the group like any decoder block.
    ``e`` may be bound later, right before the first forward.
    """
    shell._predictor_refs = refs
    shell._predictor_e = e
    if model is not None:
        # bypass nn.Module.__setattr__: registering the model as a submodule
        # would create a reference cycle (model -> shell -> model) that breaks
        # named_modules/apply with RecursionError
        object.__setattr__(shell, "_predictor_model", model)
    shell.forward = lambda hidden_states, **input_others: predictor_forward(shell, hidden_states, **input_others)


# ── checkpoint reading ──────────────────────────────────────────────────────


def list_checkpoint_tensors(source_dir: str) -> dict[str, tuple]:
    """Map every tensor name in the source checkpoint to ``(shape, dtype, file)``.

    Reads the safetensors index when present, else scans ``*.safetensors``.
    Tensors are NOT loaded - metadata only.
    """
    from safetensors import safe_open

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    files = []
    if os.path.isfile(index_path):
        import json

        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(fn for fn in os.listdir(source_dir) if fn.endswith(".safetensors"))
    out: dict[str, tuple] = {}
    for fn in files:
        with safe_open(os.path.join(source_dir, fn), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                slice_meta = f.get_slice(key)
                out[key] = (tuple(slice_meta.get_shape()), slice_meta.get_dtype(), fn)
    return out


def load_checkpoint_tensor(source_dir: str, ckpt_tensors: dict[str, tuple], name: str) -> torch.Tensor:
    """Load one tensor from the source checkpoint by name."""
    from safetensors import safe_open

    meta = ckpt_tensors[name]
    with safe_open(os.path.join(source_dir, meta[2]), framework="pt") as f:
        return f.get_tensor(name)


def checkpoint_only_roots(ckpt_tensors: dict[str, tuple], model: torch.nn.Module) -> list[str]:
    """Root prefixes of checkpoint tensors with no counterpart under the model.

    A layer is checkpoint-only when neither it nor any ancestor exists among
    the model's module names; the returned roots are the maximal such
    prefixes (e.g. ``mtp`` or ``model.layers.80``).
    """
    module_names = {n for n, _ in model.named_modules()}
    missing_layers = set()
    for name in ckpt_tensors:
        parent = name.rsplit(".", 1)[0]
        if parent in module_names:
            continue
        # find the highest missing ancestor
        segs = parent.split(".")
        root = segs[0]
        for i in range(1, len(segs)):
            if ".".join(segs[:i]) in module_names:
                root = ".".join(segs[: i + 1])
        missing_layers.add(root)
    roots: list[str] = []
    for layer in sorted(missing_layers):
        if not any(r == layer or layer.startswith(r + ".") for r in roots):
            roots.append(layer)
    return roots


# ── role analysis ───────────────────────────────────────────────────────────


def _canonical_leaf(rel: str) -> str:
    """Canonicalize a parameter path for cross-family matching (digits stripped)."""
    return ".".join("N" if seg.isdigit() else seg for seg in rel.split("."))


def analyze_predictor_group(ckpt_tensors: dict[str, tuple], group: str, hidden: Optional[int]) -> Optional[dict]:
    """Classify a checkpoint-only group into predictor roles by shape and name.

    The group-level 2D ``[hidden, 2*hidden]`` weight is the concat mixer; 1D
    ``[hidden]`` vectors with embedding/hidden keywords are the prologue
    norms; any remaining 1D vector is the final norm; the decoder-layer
    subtree root is found by descending while every deep tensor shares one
    head component.  Returns None when the group does not match the uniform
    predictor pattern.
    """
    if hidden is None:
        return None
    direct: dict[str, str] = {}
    deep: list[str] = []
    for n in ckpt_tensors:
        if not (n == group or n.startswith(group + ".")):
            continue
        rel = n[len(group) + 1 :]
        if rel.count(".") == 1 and rel.endswith(".weight"):
            direct[rel[: -len(".weight")]] = n
        elif rel.count(".") >= 2:
            deep.append(n)
    if not deep:
        return None
    fc = norm_e = norm_h = final_norm = None
    for leaf, full in direct.items():
        shape = ckpt_tensors[full][0]
        low = leaf.lower()
        if len(shape) == 2 and fc is None and shape[0] == hidden and shape[1] == 2 * hidden:
            fc = full
        elif len(shape) == 1 and shape[0] == hidden:
            if norm_e is None and ("embed" in low or low.endswith("enorm") or low.startswith("e_")):
                norm_e = full
            elif norm_h is None and ("hidden" in low or low.endswith("hnorm") or low.startswith("h_")):
                norm_h = full
            elif final_norm is None:
                final_norm = full
    if fc is None or norm_e is None or norm_h is None:
        return None
    rels = [n[len(group) + 1 :] for n in deep]
    root = ""
    while True:
        heads = {r[len(root) + 1 :].split(".")[0] if root else r.split(".")[0] for r in rels}
        if len(heads) != 1:
            break
        nxt = f"{root}.{next(iter(heads))}" if root else next(iter(heads))
        if any(r == nxt for r in rels) or not all(r == nxt or r.startswith(nxt + ".") for r in rels):
            break
        root = nxt
    return {
        "prefix": group,
        "fc": fc,
        "norm_e": norm_e,
        "norm_h": norm_h,
        "final_norm": final_norm,
        "layer_root": f"{group}.{root}" if root else group,
    }


# ── sibling selection and tree build ────────────────────────────────────────


def _param_checkpoint_source(ckpt_tensors: dict[str, tuple], layer_root: str, param_rel: str) -> Optional[str]:
    """Resolve a sibling parameter path to a checkpoint tensor under the tree.

    Matches the canonical leaf (digits stripped) first, then the exact path.
    """
    want_canon = _canonical_leaf(param_rel)
    canon_map = {_canonical_leaf(n[len(layer_root) + 1 :]): n for n in ckpt_tensors if n.startswith(layer_root + ".")}
    if want_canon in canon_map:
        return canon_map[want_canon]
    return canon_map.get(param_rel)


def pick_sibling_layer(
    model: torch.nn.Module, ckpt_tensors: dict[str, tuple], info: dict, all_blocks: list
) -> Optional[tuple]:
    """Pick the decoder block whose parameter set best matches the group's layer.

    Every sibling parameter must resolve to a checkpoint source (complete
    coverage); blocks of the same family auto-select by canonical-leaf
    overlap.  Returns ``(block_name, module)`` or None.
    """
    layer_root = info["layer_root"]
    ckpt_set = {_canonical_leaf(n[len(layer_root) + 1 :]) for n in ckpt_tensors if n.startswith(layer_root + ".")}
    best = None
    for block in all_blocks:
        for bname in block:
            if bname == layer_root or bname.startswith(layer_root + ".") or layer_root.startswith(bname + "."):
                continue
            mod = model.get_submodule(bname) if bname else None
            if mod is None or not any(True for _ in mod.children()) or not any(True for _ in mod.parameters()):
                continue
            params = [rel for rel, _ in mod.named_parameters()]
            if not params:
                continue
            sources = [_param_checkpoint_source(ckpt_tensors, layer_root, rel) for rel in params]
            if any(s is None for s in sources):
                continue
            score = len({_canonical_leaf(rel) for rel in params} & ckpt_set)
            if best is None or score > best[0]:
                best = (score, bname, mod)
    if best is None:
        return None
    return best[1], best[2]


def _norm_prototype(sibling: torch.nn.Module, hidden: int) -> Optional[torch.nn.Module]:
    """First sub-module of the sibling with exactly one 1-D [hidden] parameter."""
    for m in sibling.modules():
        if m is sibling:
            continue
        params = list(m.parameters())
        if len(params) == 1 and params[0].dim() == 1 and params[0].numel() == hidden:
            return m
    return None


def build_predictor_tree(
    model: torch.nn.Module,
    source_dir: str,
    ckpt_tensors: dict[str, tuple],
    info: dict,
    sibling: torch.nn.Module,
) -> set:
    """Materialize the predictor tree under its checkpoint-spelled paths.

    The sibling decoder layer is deep-copied and reloaded from checkpoint
    tensors; prologue/final norms are cloned from the sibling's norm
    prototype; the concat mixer is a fresh Linear.  Returns the set of
    consumed checkpoint tensor names.
    """
    hidden = ckpt_tensors[info["fc"]][0][0]
    layer_root = info["layer_root"]
    layer_mod = copy.deepcopy(sibling)
    # the snapshot copies instance-level forwards too (replacement wrappers)
    # whose closures bind the ORIGINAL module - calling them re-enters the
    # source block instead of the copy.  Restore each module's class forward.
    for m in layer_mod.modules():
        if "forward" in m.__dict__:
            cls_fwd = getattr(type(m), "forward", None)
            if cls_fwd is not None:
                m.forward = cls_fwd.__get__(m, type(m))
    parent = ensure_module_path(model, layer_root)
    parent.add_module(layer_root.rsplit(".", 1)[-1], layer_mod)
    claimed = set()

    canon_map = {_canonical_leaf(n[len(layer_root) + 1 :]): n for n in ckpt_tensors if n.startswith(layer_root + ".")}
    for pname, p in layer_mod.named_parameters():
        src = canon_map.get(_canonical_leaf(pname)) or canon_map.get(pname)
        if src is None:
            raise RuntimeError(f"predictor layer parameter {pname} has no checkpoint source under {layer_root}")
        with torch.no_grad():
            p.data.copy_(load_checkpoint_tensor(source_dir, ckpt_tensors, src))
        claimed.add(src)

    proto = _norm_prototype(sibling, hidden)
    for role in ("norm_e", "norm_h", "final_norm"):
        tensor_name = info.get(role)
        if tensor_name is None:
            continue
        if proto is None:
            raise RuntimeError(f"no norm prototype inside the sibling for {role}")
        mod = copy.deepcopy(proto)
        with torch.no_grad():
            mod.weight.data.copy_(load_checkpoint_tensor(source_dir, ckpt_tensors, tensor_name))
        mod_path = tensor_name[: -len(".weight")]
        mod_parent = ensure_module_path(model, mod_path)
        mod_parent.add_module(mod_path.rsplit(".", 1)[-1], mod)
        claimed.add(tensor_name)

    fc_meta = ckpt_tensors[info["fc"]]
    fc = torch.nn.Linear(int(fc_meta[0][1]), int(fc_meta[0][0]), bias=False)
    with torch.no_grad():
        fc.weight.copy_(load_checkpoint_tensor(source_dir, ckpt_tensors, info["fc"]))
    fc_path = info["fc"][: -len(".weight")]
    fc_parent = ensure_module_path(model, fc_path)
    fc_parent.add_module(fc_path.rsplit(".", 1)[-1], fc)
    claimed.add(info["fc"])

    logger.info("materialized predictor tree %s (layer from sibling, %d tensors)", info["prefix"], len(claimed))
    return claimed
