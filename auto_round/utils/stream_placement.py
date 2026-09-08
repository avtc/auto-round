# Copyright (c) 2024 Intel Corporation
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

"""Per-module device placement for streamed blocks (mapped staging).

Streaming pins each block whole on ONE device; mapped placement instead
distributes the block's modules across several GPUs via a placement
template, so per-block tuning state (fp32 values, grads, weights) lives
with its module and the block no longer has to fit a single card.

Two template forms:

* dict/regex string ``"model.layers.self_attn.q_proj:0,gate_proj:1"`` --
  exact leaf names or regexes, mirroring ``--device_map`` semantics.
* device list ``"0,1,2,3"`` -- contiguous parameter-balanced partition of
  the block's atomic module groups (repeated containers such as MoE
  experts stay whole on one device, keeping router gathers per-device).
"""

import re

import torch

from auto_round.logger import logger

_DEVICE_TOKEN = re.compile(r"^(?:cuda|gpu)(?::(\d+))?$|^cpu$|^xpu(?::(\d+))?$|^hpu(?::(\d+))?$|^mps$|^npu(?::(\d+))?$")


def is_placement_template(value) -> bool:
    """True when ``value`` names modules, not just devices.

    ``"0,1"`` / ``"cuda:0"`` are plain device lists (today's prefetch
    rotation semantics); ``"q_proj:0,mlp.*:1"`` is a placement template.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    value = value.strip()
    if _DEVICE_TOKEN.match(value):
        return False  # a single device token ("cuda:0", "cpu", "0")
    parts = [part for part in value.split(",") if part.strip()]
    return bool(parts) and all(":" in part for part in parts)


def parse_device_template(template: str) -> dict:
    """Parse ``"name:device,regex:device"`` into an ordered spec dict."""
    spec: dict = {}
    for part in template.replace(" ", "").split(","):
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"invalid device_map entry {part!r}: mapped placement entries must be 'module:device' "
                "(e.g. 'self_attn.*:0,mlp.*:1')"
            )
        name, _, dev = part.partition(":")
        if not name or not dev:
            raise ValueError(f"invalid device_map entry {part!r}: expected 'module:device'")
        spec[name] = dev
    if not spec:
        raise ValueError("device_map placement template is empty")
    return spec


def _normalize_device(dev: str) -> torch.device:
    if dev.isdigit():
        return torch.device("cuda", int(dev))
    return torch.device(dev)


def _atomic_groups(leaf_names: list) -> list:
    """Group leaf names by their repeated container (e.g. ``experts.3``).

    Leaves that live under the same indexed container (MoE experts, layer
    lists) form one atomic unit so a container is never split across
    devices. Standalone leaves are their own unit.
    """
    indexed = re.compile(r"^(.*)\.(\d+)\.[^.]+$")
    groups: dict = {}
    order: list = []
    for name in leaf_names:
        m = indexed.match(name)
        key = f"{m.group(1)}.{m.group(2)}" if m else name
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(name)
    return [(key, groups[key]) for key in order]


def _leaf_param_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=False):
        total += p.numel() * p.element_size()
    return total


def resolve_block_placement(block: torch.nn.Module, template, fallback: torch.device) -> dict:
    """Resolve ``template`` into ``{leaf_name: device}`` for one block.

    Leaves are modules with no children and at least one parameter, named
    relative to ``block`` (the same names regex templates match against).
    Unmatched leaves land on ``fallback``.
    """
    leaf_names = [
        n for n, m in block.named_modules() if not list(m.children()) and any(True for _ in m.parameters(recurse=False))
    ]
    if not leaf_names:
        return {}
    get_mod = dict(block.named_modules())

    placement: dict = {}
    if isinstance(template, str) and is_placement_template(template):
        spec = parse_device_template(template)
        for leaf in leaf_names:
            dev = None
            if leaf in spec:
                dev = spec[leaf]
            else:
                # longest regex match wins, mirroring set_non_auto_device_map
                for pattern, candidate in spec.items():
                    try:
                        if re.match(pattern, leaf):
                            dev = candidate
                            break
                    except re.error as e:
                        raise ValueError(f"invalid device_map pattern {pattern!r}: {e}") from e
            placement[leaf] = _normalize_device(dev) if dev is not None else torch.device(fallback)
        matched = {p for p in spec if any(re.match(p, leaf) or leaf == p for leaf in leaf_names)}
        unmatched = [p for p in spec if p not in matched]
        if unmatched:
            logger.warning("[stream-mapped] device_map entries matched no module in this block: %s", unmatched)
        return placement

    # device list: contiguous parameter-balanced partition over atomic groups
    devices = [d.strip() for d in str(template).split(",") if d.strip()]
    if not devices:
        raise ValueError("device_map placement template is empty")
    devices = [_normalize_device(d) for d in devices]
    if len(devices) == 1:
        return {leaf: devices[0] for leaf in leaf_names}
    groups = _atomic_groups(leaf_names)
    weights = [sum(_leaf_param_bytes(get_mod[leaf]) for leaf in leaves) for _, leaves in groups]
    total = sum(weights) or 1
    target = total / len(devices)
    placement = {}
    dev_idx = 0
    acc = 0
    for (_key, leaves), w in zip(groups, weights):
        placement.update({leaf: devices[dev_idx] for leaf in leaves})
        acc += w
        if acc >= target and dev_idx < len(devices) - 1:
            dev_idx += 1
            acc = 0
    return placement


def placement_device_of(placement: dict, tensor_name: str, fallback) -> str:
    """Checkpoint tensor name -> staging device.

    Strips the parameter/buffer attribute (``.weight``/``.bias``) to find
    the owning leaf; anything unmatched (setup modules, block-level
    buffers) goes to ``fallback``.
    """
    if not placement:
        return str(fallback)
    leaf = tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name
    dev = placement.get(leaf)
    if dev is None:
        return str(fallback)
    return str(dev)
