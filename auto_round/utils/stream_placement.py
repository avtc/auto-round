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
from typing import Optional

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


def _shared_atoms(leaf_names: list, shared_leaf_groups) -> list:
    """Match shared-quantization groups (comma-key layer_config entries) to
    block-relative leaf names.

    A group like ``["q_proj", "k_proj", "v_proj"]`` binds every parent whose
    children include exactly those members (``self_attn.q_proj`` +
    ``self_attn.k_proj`` + ``self_attn.v_proj``). Returns matched groups as
    lists of leaf names - each is an atomic placement unit, because shared
    scales/zeros are searched on the merged group.
    """
    atoms = []
    for group in shared_leaf_groups or []:
        members = [str(m).strip() for m in group if str(m).strip()]
        if len(members) < 2:
            continue
        by_parent: dict = {}
        for leaf in leaf_names:
            parent, _, last = leaf.rpartition(".")
            if last in members:
                by_parent.setdefault(parent, []).append(leaf)
        for parent, leaves in by_parent.items():
            if parent == "":
                continue  # top-level bare names: no sibling context to bind
            if {leaf.rpartition(".")[2] for leaf in leaves} == set(members):
                atoms.append(leaves)
    return atoms


def _leaf_param_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=False):
        total += p.numel() * p.element_size()
    return total


def resolve_block_placement(block: torch.nn.Module, template, fallback: torch.device, shared_leaf_groups=None) -> dict:
    """Resolve ``template`` into ``{leaf_name: device}`` for one block.

    Leaves are modules with no children and at least one parameter, named
    relative to ``block`` (the same names regex templates match against).
    Unmatched leaves land on ``fallback``. ``shared_leaf_groups`` (parsed
    comma-key layer_config entries) are kept whole on one device under the
    device-list partition and warn on straddle under a template.
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
        for atom in _shared_atoms(leaf_names, shared_leaf_groups):
            devs = {placement.get(leaf) for leaf in atom}
            if len(devs) > 1:
                logger.warning(
                    "[stream-mapped] shared-quantization group %s straddles devices %s under the given "
                    "template; shared scales/zeros are searched on the merged group - keep it on one device",
                    sorted(atom),
                    sorted(str(d) for d in devs),
                )
        return placement

    # device list: contiguous parameter-balanced partition over atomic groups
    devices = [d.strip() for d in str(template).split(",") if d.strip()]
    if not devices:
        raise ValueError("device_map placement template is empty")
    devices = [_normalize_device(d) for d in devices]
    if len(devices) == 1:
        return {leaf: devices[0] for leaf in leaf_names}
    groups = _atomic_groups(leaf_names)
    # shared-quantization groups are atomic too: merge any container groups
    # they intersect so the merged search never spans devices
    for atom in _shared_atoms(leaf_names, shared_leaf_groups):
        merged: set = set(atom)
        rebuilt = []
        for key, leaves in groups:
            if merged & set(leaves):
                merged |= set(leaves)
            else:
                rebuilt.append((key, leaves))
        rebuilt.append(("shared:" + atom[0].rpartition(".")[2], sorted(merged)))
        groups = rebuilt
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


def block_signature(block: torch.nn.Module) -> str:
    """Stable structure signature: same layout -> same key, any index.

    Strips digit runs from leaf names (``experts.17.gate_proj`` ->
    ``experts.*.gate_proj``) and pairs them with the leaf type and parameter
    shapes. Models that interleave block layouts (linear- vs full-attention,
    MTP tails) yield one signature per distinct layout.
    """
    parts = []
    for name, module in block.named_modules():
        if list(module.children()):
            continue
        if not any(True for _ in module.parameters(recurse=False)):
            continue
        pattern = re.sub(r"\d+", "*", name)
        shapes = ",".join(str(tuple(int(s) for s in p.shape)) for p in module.parameters(recurse=False))
        parts.append(f"{pattern}:{type(module).__name__}:{shapes}")
    return "|".join(parts)


def partition_flow_order(flow_units: list, devices: list, budgets: Optional[dict] = None) -> dict:
    """Assign flow-ordered units to devices, balancing VRAM.

    ``flow_units`` is a list of ``(names, param_bytes, input_bytes)`` in
    observed execution order; units flagged in ``atom_units`` (same length
    boolean list) are placement-atomic (expert projections, shared-quant
    groups) and are distributed greedily for balance - their routing order
    is data-dependent, not index-dependent, so order does not matter. The
    remaining units stay CONTIGUOUS along the flow order and cut at the
    cheapest boundaries (boundary cost = the next unit's input bytes), so
    per-forward activation traffic is minimal.

    Returns ``{name: device}`` for all names in the units.
    """
    if len(devices) == 1:
        return {n: devices[0] for names, _b, _i, _a in flow_units for n in names}

    # normalize to tuples (names, bytes, in_bytes, is_atom)
    units = [(tuple(names), int(p), int(i), bool(a)) for names, p, i, a in flow_units]
    total = sum(b for _n, b, _i, _a in units) or 1
    target = total / len(devices)
    placement: dict = {}
    loads = {d: 0 for d in devices}

    # phase 1: atoms -> least-loaded device (VRAM balance; experts dominate)
    atom_units = [u for u in units if u[3]]
    atom_units.sort(key=lambda u: -u[1])
    for names, b, _i, _a in atom_units:
        dev = min(devices, key=lambda d: loads[d])
        placement.update({n: dev for n in names})
        loads[dev] += b

    # phase 2: non-atom units -> contiguous along flow order, cheapest cuts
    seq = [u for u in units if not u[3]]
    if seq:
        budgets = budgets or {}
        hi = max([target * 1.15] + [max(0.0, budgets.get(str(d), target) * 1.15) for d in devices])
        lo = target * 0.5
        n = len(seq)
        m = len(devices)
        # dp[i][k]: min cut cost to split seq[i:] into k segments (each in
        # [lo, hi]); cuts sit BETWEEN units, cost = next unit's input bytes
        INF = float("inf")
        cut = [seq[j + 1][2] if j + 1 < n else 0 for j in range(n)]
        dp = [[INF] * (m + 1) for _ in range(n + 1)]
        choice = [[-1] * (m + 1) for _ in range(n + 1)]
        dp[n][0] = 0
        for i in range(n, -1, -1):
            for k in range(1, m + 1):
                acc = 0
                for j in range(i, n):
                    acc += seq[j][1]
                    if acc > hi:
                        break
                    seg_ok = k == 1 and j == n - 1 or k > 1
                    if not seg_ok:
                        continue
                    if k == 1:
                        if j == n - 1 and dp[j + 1][0] < INF and acc >= lo * 0.999 or (j == n - 1 and acc < lo):
                            cand = dp[j + 1][0]
                            if cand < dp[i][k]:
                                dp[i][k] = cand
                                choice[i][k] = j
                    else:
                        rest = dp[j + 1][k - 1]
                        if rest < INF:
                            cand = cut[j] + rest
                            if cand < dp[i][k]:
                                dp[i][k] = cand
                                choice[i][k] = j
        # extract segments for m devices; if m infeasible, try fewer
        k_used = m
        while k_used > 1 and dp[0][k_used] == INF:
            k_used -= 1
        segments = []
        i, k = 0, k_used
        while i < n and k > 0:
            j = choice[i][k]
            if j < 0:
                break
            segments.append(seq[i : j + 1])
            i, k = j + 1, k - 1
        # assign segments largest-first to least-loaded devices
        segments.sort(key=lambda s: -sum(u[1] for u in s))
        free = sorted(devices, key=lambda d: loads[d])
        for seg in segments:
            dev = free.pop(0) if free else devices[-1]
            placement.update({nm: dev for u in seg for nm in u[0]})
            loads[dev] += sum(u[1] for u in seg)
    return placement


class FlowProbe:
    """Record the module execution order of a block's first forward.

    One-shot pre-hooks on every leaf note the order (and first-input size);
    a post-hook on the block itself finalizes after one complete forward.
    Used to derive device placements along the observed data flow, which can
    differ from registration order, and to price device-boundary cuts by the
    actual activation bytes crossing them.
    """

    def __init__(self, block: torch.nn.Module, on_complete):
        self.records: list = []
        self._done = False
        self._on_complete = on_complete
        self._hooks = []
        rel_leaves = [
            (n, m)
            for n, m in block.named_modules()
            if not list(m.children()) and any(True for _ in m.parameters(recurse=False))
        ]

        def _first_bytes(args) -> int:
            for a in args:
                if torch.is_tensor(a):
                    return a.numel() * a.element_size()
            return 0

        for rel, leaf in rel_leaves:

            def _pre(module, args, _rel=rel):
                if self._done:
                    return
                self.records.append((_rel, _first_bytes(args)))

            self._hooks.append(leaf.register_forward_pre_hook(_pre))

        def _post(module, args, output):
            self.finish()

        self._hooks.append(block.register_forward_hook(_post))

    def finish(self):
        if self._done:
            return
        self._done = True
        for h in self._hooks:
            try:
                h.remove()
            except Exception:  # pragma: no cover - hook already gone
                pass
        if self._on_complete is not None:
            self._on_complete(self.records)
