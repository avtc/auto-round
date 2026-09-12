#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device placement for (possibly device-mapped) tuning blocks.

The data-driven path can host blocks whose modules sit on several
accelerator devices (a block that does not fit one device is sharded by
the upstream placement machinery). Two placement problems arise there:

* ``nn.Module.to()`` only moves *registered* parameters and buffers --
  exotic containers (e.g. gated-delta-net style modules holding plain
  python lists of parameters) keep their leaves on the old device and
  crash the first cross-device op at call time.
* Mirroring such a block for data-parallel tuning must reproduce the
  stage layout on a *different* device set, not squash it onto one
  device.

:func:`tensor_leaves` walks every tensor leaf of a module tree including
the unregistered ones; :func:`place_module_tree` moves them all onto
devices chosen by a per-module-name resolver; :func:`leaf_device_map`
reports where they ended up (placement bookkeeping -- on CUDA the tensor
devices are authoritative, CPU normalizes ``cpu:i`` to ``cpu``).
"""

from typing import Callable, Dict, Iterator, List, Tuple

import torch


def _iter_container_leaves(obj, prefix: str, holder=None, key=None):
    """Yield ``(qualname, tensor, holder, key)`` for tensor leaves in a plain container.

    ``holder`` is the IMMEDIATE container object (list/dict/tuple) holding
    the tensor and ``key`` its index/key inside it; both are None for a
    bare tensor. The holder/key pair is what lets :func:`_assign_leaf`
    replace non-Parameter tensors inside containers.
    """
    if torch.is_tensor(obj):
        yield prefix, obj, holder, key
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _iter_container_leaves(item, f"{prefix}[{i}]", obj, i)
    elif isinstance(obj, dict):
        for k, item in obj.items():
            yield from _iter_container_leaves(item, f"{prefix}[{k!r}]", obj, k)


def tensor_leaves(module: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor, object, object]]:
    """Yield every tensor leaf of ``module``, registered or not.

    Registered parameters/buffers come from the module registry; anything
    else reachable through instance ``__dict__`` attributes (plain tensors,
    parameters and buffers inside lists/tuples/dicts) is walked explicitly
    -- those are the leaves ``.to()`` silently leaves behind.
    """
    registered = set()
    for name, p in module.named_parameters(recurse=True, remove_duplicate=False):
        registered.add(id(p))
        yield name, p, None, None
    for name, b in module.named_buffers(recurse=True, remove_duplicate=False):
        registered.add(id(b))
        yield name, b, None, None
    for mod_name, mod in module.named_modules():
        for attr, val in vars(mod).items():
            if mod is module and attr in ("_parameters", "_buffers"):
                continue
            for path, tensor, holder, key in _iter_container_leaves(val, attr):
                if id(tensor) in registered:
                    continue
                full = f"{mod_name}.{path}" if mod_name else path
                yield full, tensor, (mod, holder, attr) if holder is not None else None, key


def _assign_leaf(tensor: torch.Tensor, target: torch.device, container, key) -> bool:
    """Move one leaf onto ``target``; returns True when a move happened.

    Parameters and registered buffers keep their object identity (``.data``
    swap -- optimizers, hooks and the module registries keep working); plain
    tensors held in containers are replaced inside their container (a bare
    attribute's tensor cannot be swapped through the object, so it also
    takes the ``.data`` swap).
    """
    if tensor.device == target:
        return False
    if isinstance(tensor, torch.nn.Parameter):
        # keep leaf identity: swap the storage, not the object (optimizers,
        # hooks and container references keep working)
        tensor.data = tensor.data.to(target)
        return True
    if container is None:
        tensor.data = tensor.data.to(target)
        return True
    moved = tensor.to(target)
    mod, holder, attr = container
    if holder is None:
        # bare module attribute: replace through the module itself
        setattr(mod, attr, moved)
    elif isinstance(holder, list):
        holder[key] = moved
    elif isinstance(holder, tuple):
        # rebuild the tuple; the holder captured at walk time goes stale
        # after the first replacement, so use the attr name directly
        new = list(vars(mod).get(attr, holder))
        new[key] = moved
        setattr(mod, attr, tuple(new))
    elif isinstance(holder, dict):
        holder[key] = moved
    return True


def place_module_tree(module: torch.nn.Module, device_of: Callable[[str], torch.device]) -> int:
    """Move every tensor leaf onto the device ``device_of(module_name)`` picks.

    ``device_of`` receives the *owning module's* qualified name (the name of
    the module whose ``__dict__``/registry holds the leaf, not the leaf
    path). Parameters keep their object identity (``.data`` swap) so
    optimizers and hooks keep working; plain tensors inside containers are
    replaced in place. Returns the number of leaves actually moved.
    """
    moved = 0
    for mod_name, mod in module.named_modules():
        target = device_of(mod_name)
        for _path, tensor, container, key in _module_leaves(mod):
            if _assign_leaf(tensor, target, container, key):
                moved += 1
    return moved


def _module_leaves(mod: torch.nn.Module):
    """Leaves owned by THIS module only (registry + its own __dict__ attrs)."""
    registered = set()
    for name, p in mod.named_parameters(recurse=False, remove_duplicate=False):
        registered.add(id(p))
        yield name, p, None, None
    for name, b in mod.named_buffers(recurse=False, remove_duplicate=False):
        registered.add(id(b))
        yield name, b, None, None
    for attr, val in vars(mod).items():
        for path, tensor, holder, key in _iter_container_leaves(val, attr):
            if id(tensor) in registered:
                continue
            # (mod, holder, attr): holder None for a bare attribute tensor
            yield path, tensor, (mod, holder, attr), key


def leaf_device_map(module: torch.nn.Module) -> Dict[str, torch.device]:
    """Report the device of every tensor leaf (registered or not)."""
    return {name: tensor.device for name, tensor, _c, _k in tensor_leaves(module)}


def accelerator_stage_devices(block: torch.nn.Module) -> List[torch.device]:
    """Ordered unique accelerator devices hosting the block's tensor leaves.

    The order is first-appearance over ``named_modules`` iteration, which is
    the module-tree definition order -- a stable stage ordering for
    device-mapped placement.
    """
    seen: List[torch.device] = []
    for name, tensor, _c, _k in tensor_leaves(block):
        dev = tensor.device
        if dev.type in ("cuda", "xpu", "hpu") and dev not in seen:
            seen.append(dev)
    return seen
