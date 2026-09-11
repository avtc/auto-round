# Copyright (c) 2026 Intel Corporation
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

"""ZeRO-2-lite tune-state sharding for single-process data-parallel tuning.

Full-mirror DDP (``ReplicaGroup``) keeps a complete fp32 copy of every
tuning value, its gradient and (transiently) a best-MSE snapshot on every
replica GPU -- for large blocks (e.g. ~4B-param MoE blocks) that mirror
alone exceeds a 24 GB card. This module shards only the *tune state*:

- bf16 block weights stay mirrored on every replica (the forward needs
  them anyway);
- the fp32 rounding values ``v`` (plus any other round keys such as
  ``bias_v``), their gradients and best-MSE snapshots are split into
  contiguous 1/N shards, one owner per shard;
- a shape-keyed *stage* buffer per replica backs ``params[key].data`` so
  the wrapper's forward still sees a full-size ``v``: a checkpointed
  pre-forward hook re-gathers the stage from the shards (fp32-exact), and
  ``torch.utils.checkpoint`` (non-reentrant) re-runs that gather during
  the backward recompute -- so only ONE module's full ``v`` is ever live
  at a time;
- a ``register_post_accumulate_grad_hook`` on each round leaf deposits
  bf16 slice copies of the freshly accumulated full gradient into the
  owner's per-depositor inbox slots and immediately frees the full
  gradient -- steady-state gradient memory is shards + bf16 inboxes only;
- after the threaded backward join, each owner folds its inbox into an
  fp32 grad shard (deposit-then-reduce: no cross-replica waits inside
  hooks, deadlock-free by construction);
- the SignRound update (elementwise for momentum 0) runs shard-locally,
  and teardown scatters the stepped (or best-captured) shards back into
  the home wrapper's full-size values.

Standing design principle (never violate): wrapper Parameter objects stay
the autograd leaves and gradients accumulate on them exactly as in serial
training; the deposit hook then consumes and frees each leaf's gradient
right after its node's backward (steady-state gradient memory = shards);
state moves ONLY through pre-allocated, reused exchange buffers via
``copy_``.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from auto_round.compressors.utils import _best_param_key_is_round as _is_round_key
from auto_round.logger import logger

__all__ = ["ZeroReplicaGroup", "split_bounds", "is_zero_candidate"]


def split_bounds(numel: int, world: int) -> List[Tuple[int, int]]:
    """Contiguous fair shards: ``world`` half-open [start, end) bounds."""
    base, rem = divmod(numel, world)
    bounds = []
    start = 0
    for r in range(world):
        size = base + (1 if r < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def is_zero_candidate(module) -> bool:
    """A wrapper module whose params carry at least one round key."""
    params = getattr(module, "params", None)
    return isinstance(params, dict) and any(_is_round_key(k) for k in params)


class _Entry:
    """One round leaf Parameter shared across replicas (same object on the
    home, deep-copied per mirror)."""

    __slots__ = ("module_name", "key", "numel", "bounds", "param_by_replica", "home_param_id")

    def __init__(self, module_name, key, numel):
        self.module_name = module_name
        self.key = key
        self.numel = numel
        self.bounds = None  # set once world is known
        self.param_by_replica: List[nn.Parameter] = []
        self.home_param_id: Optional[int] = None

    @property
    def uid(self):
        return f"{self.module_name}.{self.key}"


class _ZeROShell(nn.Module):
    """Checkpoints the wrapper forward and re-gathers its staged values.

    ``forward`` runs under non-reentrant checkpoint; the gather lives
    INSIDE the checkpointed callable so the backward recompute re-runs it
    against the (frozen-for-the-iteration) shards and reproduces the exact
    stage content the forward saw. The wrapped module's Parameters stay
    the autograd leaves; this shell only adds an outer module around it.
    The wrapper contract (``params``, ``orig_layer``) is delegated so
    downstream walks (tuning-param collection, best-params capture) keep
    seeing the wrapper.
    """

    def __init__(self, wrapped: nn.Module, entries: List[_Entry], group: "ZeroReplicaGroup", replica_index: int):
        super().__init__()
        # NOT registered as a submodule -- registering it made
        # named_modules() yield the wrapper twice (shell + wrapped), which
        # duplicated every tuning param in optimizer collections. Stored as
        # a plain attribute, mirroring how the wrapper itself keeps
        # orig_layer out of the module tree. __getattr__ delegates the rest
        # of the wrapper contract (weight, bits, ...).
        self.__dict__["wrapped"] = wrapped
        self._entries = entries
        self._group = group
        self._replica_index = replica_index

    @property
    def params(self):
        return self.wrapped.params

    @property
    def orig_layer(self):
        return self.wrapped.orig_layer

    def _gathered_forward(self, *args, **kwargs):
        self._group.gather_stage(self._replica_index, self._entries)
        return self.wrapped(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self._gathered_forward, *args, use_reentrant=False, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__.get("wrapped")
            if wrapped is not None:
                return getattr(wrapped, name)
            raise


class ZeroReplicaGroup:
    """Replica group with ZeRO-2-lite sharded tune state.

    Mirrors the public surface of ``ReplicaGroup`` that the tune loop uses
    (``replicas``, ``world``, ``plan``, ``run_threaded``-style execution is
    supplied by the caller) and adds the sharded gather / exchange / step /
    capture / teardown phases.

    INVARIANT: same-shape entries share one stage buffer per replica, so a
    wrapper's ``params[key].data`` only holds that module's own values
    between its gather and its use (forward or recompute). Mid-loop full-
    value reads (best-params capture!) must therefore go through the SHARDS
    (``capture_best``), never through wrapper param reads -- the caller's
    full-tree ``collect_best_params`` must be replaced for round params
    while this group is active.
    """

    def __init__(self, block: nn.Module, plan, momentum: Optional[float] = None):
        if plan.world < 2:
            raise ValueError("ZeRO-2-lite requires world >= 2")
        if momentum is not None and float(momentum) != 0.0:
            # shard-local sign updates are elementwise and carry no momentum
            # buffers; silently dropping momentum would change the algorithm
            raise ValueError(f"ZeRO-2-lite requires momentum=0, got {momentum}")
        self.plan = plan
        self.world = plan.world
        self.home = block
        self.devices = [d for d in plan.devices]

        # ---- collect round entries from the home tree -------------------
        self.entries: List[_Entry] = []
        for name, mod in block.named_modules():
            params = getattr(mod, "params", None)
            if not isinstance(params, dict):
                continue
            mod_entries = []
            for key, p in params.items():
                if not _is_round_key(key) or not isinstance(p, nn.Parameter):
                    continue
                e = _Entry(name, key, p.numel())
                e.home_param_id = id(p)
                mod_entries.append(e)
                self.entries.append(e)
            if mod_entries:
                for e in mod_entries:
                    e.bounds = split_bounds(e.numel, self.world)
        if not self.entries:
            raise ValueError("no round tuning parameters found to shard")

        # ---- mirrors (proven deepcopy path + its hardening) --------------
        from auto_round.algorithms.quantization.sign_round.data_parallel import (
            _enforce_mirror_device_,
            _relocate_params,
        )

        self.mirrors: List[nn.Module] = []
        for d in self.devices[1:]:
            mirror = copy.deepcopy(block).to(d)
            _relocate_params(mirror, d)
            strays = _enforce_mirror_device_(mirror, d)
            if strays:
                logger.warning(
                    "[tune-zero] mirror device sweep moved %d straggler tensor(s) onto %s",
                    len(strays),
                    d,
                )
            self.mirrors.append(mirror)

        # map home entries -> mirror params by (mirror index, module_name, key)
        mirror_params: Dict[Tuple[int, str, str], nn.Parameter] = {}
        for m_idx, m in enumerate(self.mirrors):
            for name, mod in m.named_modules():
                params = getattr(mod, "params", None)
                if isinstance(params, dict):
                    for key, p in params.items():
                        mirror_params[(m_idx, name, key)] = p
        for e in self.entries:
            home_p = _param_by_entry(block, e)
            e.param_by_replica = [home_p] + [
                mirror_params[(m_idx, e.module_name, e.key)] for m_idx in range(len(self.mirrors))
            ]

        # ---- persistent buffers ------------------------------------------
        # stage: per (replica, shape) full-size fp32, backs param.data
        self._stages: List[Dict[Tuple[int, ...], torch.Tensor]] = [dict() for _ in range(self.world)]
        # shards: per (replica, entry) fp32 shard + grad shard + snapshot
        self._v_shard: List[Dict[str, torch.Tensor]] = [dict() for _ in range(self.world)]
        self._g_shard: List[Dict[str, torch.Tensor]] = [dict() for _ in range(self.world)]
        self._snap_shard: List[Dict[str, torch.Tensor]] = [dict() for _ in range(self.world)]
        # inboxes: per (owner, entry, depositor) bf16 slot
        self._inbox: List[Dict[str, List[Optional[torch.Tensor]]]] = [dict() for _ in range(self.world)]
        self._captured = False
        self._hooks = []

        replicas = [block] + self.mirrors
        self.replicas = replicas
        for r, rep in enumerate(replicas):
            dev = self.devices[r]
            for e in self.entries:
                p = e.param_by_replica[r]
                flat = p.data.reshape(-1)
                lo, hi = e.bounds[r]
                shard = flat[lo:hi].detach().clone().to(dev)  # fp32-exact initial split
                stage = self._stages[r].get(tuple(p.shape))
                if stage is None:
                    stage = torch.empty(p.shape, dtype=torch.float32, device=dev)
                    self._stages[r][tuple(p.shape)] = stage
                with torch.no_grad():
                    stage.reshape(-1)[lo:hi].copy_(shard)
                self._v_shard[r][e.uid] = shard
                self._g_shard[r][e.uid] = torch.zeros_like(shard)
                self._snap_shard[r][e.uid] = torch.zeros_like(shard)
                self._inbox[r][e.uid] = [None] * self.world
                # the Parameter object stays the leaf; only its storage moves
                p.data = stage
        # initial gather: every replica stage gets every owner's slice
        for r in range(self.world):
            self._gather_replica(r)

        # ---- shells + grad hooks on every replica ------------------------
        for r, rep in enumerate(replicas):
            for name, mod in rep.named_modules():
                if not is_zero_candidate(mod):
                    continue
                mod_entries = [e for e in self.entries if e.module_name == name]
                shell = _ZeROShell(mod, mod_entries, self, r)
                _replace_module(rep, name, shell)
                for e in mod_entries:
                    p = e.param_by_replica[r]
                    self._hooks.append(_register_deposit_hook(p, self, e, r))

        # weight copies of the shells: mirrors already deepcopied BEFORE
        # sharding, so wrapper weights are identical to the full-mirror path
        self._pool = None
        self._teardown_done = False
        logger.info(
            "[tune-zero] engaged: world=%d entries=%d (est. %.2f GiB fp32 tune state sharded per replica)",
            self.world,
            len(self.entries),
            sum(e.numel for e in self.entries) * 4 * (self.world - 1) / self.world / 2**30,
        )

    # ------------------------------------------------------------------ #
    # gather / exchange
    # ------------------------------------------------------------------ #
    def _gather_replica(self, r: int) -> None:
        """Refresh replica r's stages from the CURRENT shards (fp32-exact)."""
        for e in self.entries:
            stage = self._stages[r][tuple(e.param_by_replica[r].shape)]
            flat = stage.reshape(-1)
            for owner in range(self.world):
                lo, hi = e.bounds[owner]
                flat[lo:hi].copy_(self._v_shard[owner][e.uid])

    def gather_stage(self, r: int, entries: List[_Entry]) -> None:
        """Pre-forward (and recompute) hook: refresh replica r's stages.

        Each shell knows its replica index, so the refresh is correct from
        any thread (worker pools run replicas in arbitrary threads); shards
        are frozen while an iteration is in flight, so concurrent reads are
        safe and every replica stages identical values.
        """
        for e in entries:
            stage = self._stages[r][tuple(e.param_by_replica[r].shape)]
            flat = stage.reshape(-1)
            for owner in range(self.world):
                lo, hi = e.bounds[owner]
                flat[lo:hi].copy_(self._v_shard[owner][e.uid])

    # ------------------------------------------------------------------ #
    # gradient deposit (post-accumulate hook) + reduce
    # ------------------------------------------------------------------ #
    def deposit_grad(self, r: int, e: _Entry, grad: torch.Tensor) -> None:
        """Replica r deposits bf16 slices of a freshly finalized full grad
        into every owner's inbox slot and the hook frees the full grad."""
        flat = grad.reshape(-1).detach()
        for owner in range(self.world):
            lo, hi = e.bounds[owner]
            slot = self._inbox[owner][e.uid][r]
            if slot is None or slot.shape != flat[lo:hi].shape or slot.device != self.devices[owner]:
                slot = torch.empty(hi - lo, dtype=torch.bfloat16, device=self.devices[owner])
                self._inbox[owner][e.uid][r] = slot
            slot.copy_(flat[lo:hi].to(torch.bfloat16).to(self.devices[owner], non_blocking=False))

    def reduce_inboxes(self) -> None:
        """After the backward join: each owner folds its bf16 inbox slots
        into its fp32 grad shard. Deterministic sum order (depositor index)."""
        for owner in range(self.world):
            for e in self.entries:
                acc = self._g_shard[owner][e.uid]
                acc.zero_()
                for depositor in range(self.world):
                    slot = self._inbox[owner][e.uid][depositor]
                    if slot is not None:
                        acc.add_(slot.to(torch.float32))
                # clear after folding: a later iteration where a leaf gets no
                # grad (e.g. a skipped MoE expert) must contribute ZERO, not
                # this iteration's deposits
                self._inbox[owner][e.uid] = [None] * self.world

    def run_threaded(self, fns) -> None:
        """Same contract as ReplicaGroup.run_threaded: one callable per
        replica (home first), persistent pool, spawn fallback on width
        mismatch, first failure re-raised."""
        from auto_round.algorithms.quantization.sign_round.data_parallel import run_threaded_with_pool

        run_threaded_with_pool(self, fns)

    def sync_grads(self, params_per_replica=None, prof=None, sign_exchange: bool = False) -> None:
        """ZeRO exchange phase -- called where the full-mirror lane allreduces.

        Round leaves were already deposited to owner inboxes by their
        post-accumulate hooks during backward; this folds the inboxes into
        the fp32 grad shards. Any OTHER parameters handed in (the minmax
        scale/zp set, which stays full-mirrored) go through the plain
        value allreduce, mirroring the full-mirror lane's semantics.
        """
        from auto_round.algorithms.quantization.sign_round.data_parallel import (
            _param_grad_buffers,
            _write_back_grads,
            halving_doubling_allreduce,
        )
        from auto_round.utils.tune_profile import stage as _stage

        with _stage(prof, "exchange"):
            self.reduce_inboxes()
        round_ids = {id(p) for e in self.entries for p in e.param_by_replica}
        # one buffer PER REPLICA (params_per_replica is already per-replica);
        # concatenating them into one would corrupt the allreduce geometry
        extra_per_replica = [[p for p in ps if id(p) not in round_ids] for ps in (params_per_replica or [])]
        extra_per_replica = [ps for ps in extra_per_replica if ps]
        if len(extra_per_replica) == self.world:
            bufs = _param_grad_buffers(extra_per_replica)
            if all(b is not None for b in bufs):
                with _stage(prof, "exchange"):
                    halving_doubling_allreduce(bufs, scale=1.0 / self.world, transport="bf16")
                    for buf, ps in zip(bufs, extra_per_replica):
                        _write_back_grads(buf, ps)
        elif extra_per_replica:
            logger.warning(
                "[tune-zero] skipping minmax exchange: got %d replica param lists, world=%d",
                len(extra_per_replica),
                self.world,
            )

    def reset_exchange(self) -> None:
        """Drop any deposits from warm-up / aborted iterations so the first
        real iteration starts from clean grad shards and inboxes."""
        for owner in range(self.world):
            for e in self.entries:
                self._g_shard[owner][e.uid].zero_()
                self._inbox[owner][e.uid] = [None] * self.world

    # ------------------------------------------------------------------ #
    # step / capture / teardown
    # ------------------------------------------------------------------ #
    def step(self, lr_by_param: Dict[int, float]) -> None:
        """Shard-local SignRound update: v -= lr * sign(g).

        Elementwise for momentum 0 (the only supported regime);
        ``lr_by_param`` maps the HOME parameter object id to the group's
        CURRENT lr (the scheduler decays it per iteration -- a build-time lr
        would freeze the schedule).
        """
        with torch.no_grad():
            for owner in range(self.world):
                for e in self.entries:
                    lr = float(lr_by_param.get(e.home_param_id, 0.0))
                    shard = self._v_shard[owner][e.uid]
                    grad = self._g_shard[owner][e.uid]
                    shard.add_(torch.sign(grad), alpha=-lr)

    def capture_best(self) -> None:
        """Snapshot the current shards (cheap: sharded, GPU-local)."""
        for owner in range(self.world):
            for e in self.entries:
                self._snap_shard[owner][e.uid].copy_(self._v_shard[owner][e.uid])
        self._captured = True

    def teardown(self) -> None:
        """Scatter the final (best-captured, else last) shards into the home
        wrapper's full-size values and restore plain module trees."""
        if self._teardown_done:
            return
        self._teardown_done = True
        source = self._snap_shard if self._captured else self._v_shard
        for r, rep in enumerate(self.replicas):
            for name, mod in list(rep.named_modules()):
                if isinstance(mod, _ZeROShell):
                    _replace_module(rep, name, mod.wrapped)
        for e in self.entries:
            home_p = e.param_by_replica[0]
            full = torch.empty_like(home_p.data, dtype=torch.float32, device=home_p.device)
            flat = full.reshape(-1)
            for owner in range(self.world):
                lo, hi = e.bounds[owner]
                flat[lo:hi].copy_(source[owner][e.uid].to(home_p.device))
            home_p.data = full
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown()
            self._pool = None
        del self.mirrors[:]
        self.replicas = [self.home]  # replicas[0] retained; mirrors released
        # exchange state is dead after the scatter; release the copies
        self._g_shard = [dict() for _ in range(self.world)]
        self._snap_shard = [dict() for _ in range(self.world)]
        self._stages = [dict() for _ in range(self.world)]
        self._inbox = [dict() for _ in range(self.world)]


def _param_by_entry(block: nn.Module, e: _Entry) -> nn.Parameter:
    mod = _get_module(block, e.module_name)
    return mod.params[e.key]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    for part in name.split("."):
        model = getattr(model, part)
    return model


def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    last = parts[-1]
    if last in getattr(parent, "_modules", {}):
        parent._modules[last] = new_module  # ModuleList / Sequential children
    else:
        setattr(parent, last, new_module)


def _register_deposit_hook(param: nn.Parameter, group: "ZeroReplicaGroup", e: _Entry, r: int):
    def _hook(p):
        if p.grad is None:
            return
        group.deposit_grad(r, e, p.grad)
        p.grad = None  # free the full-size gradient immediately

    if hasattr(param, "register_post_accumulate_grad_hook"):
        return param.register_post_accumulate_grad_hook(_hook)
    raise RuntimeError("post-accumulate grad hooks are required for ZeRO-2-lite (torch >= 2.1)")
