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
"""The batched AutoRound-format packer must not engage for other format families.

``auto_round:llm_compressor`` begins with the ``auto_round`` prefix but packs
through the compressed-tensors ``pack_layer`` -- routing its same-shape module
groups through the AutoRound qweight/qzeros/scales batched packer silently
produces a mixed-format checkpoint (CT-packed modules alongside GPTQ-packed
ones) that no single consumer can load.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from auto_round.compressors.config_resolution import resolve_scheme_value
from auto_round.compressors.utils import _format_supports_batched_pack
from auto_round.export.export_to_autoround.export import pack_layers_batched
from auto_round.export.formats import resolve_formats


def _fmt(format: str, scheme_name: str = "W4A16", model=None, **overrides):
    """Resolve to the REAL format instance the quantizer would use."""
    scheme = resolve_scheme_value(scheme_name, overrides)
    result = resolve_formats(scheme, format=format, layer_config=None, model=model)
    return result.formats[0]


def test_gate_rejects_llm_compressor_wrapper():
    # "auto_round:llm_compressor" resolves to an AutoRoundFormat WRAPPER whose
    # output_format attr is plain "auto_round" (per-module packing delegates
    # to the compressed-tensors packer) -- the gate must look at the routing,
    # not the wrapper's identity attributes.
    fmt = _fmt("auto_round:llm_compressor")
    assert type(fmt).__name__ == "AutoRoundFormat"
    assert fmt.output_format == "auto_round"  # the trap the gate must survive
    assert not _format_supports_batched_pack(fmt)


def test_gate_accepts_routed_sym_and_asym_defaults():
    # plain "auto_round" silently routes sym -> gptq packer and 4-bit asym ->
    # awq packer via the backend wrapper; both routed packers share the
    # column-sliceable qlinear layout, so both are batchable (each through
    # the ROUTED backend's own format string).
    sym = _fmt("auto_round", sym=True)
    assert sym.backend is not None and "gptq" in type(sym.backend).__name__.lower()
    assert _format_supports_batched_pack(sym)

    asym = _fmt("auto_round", sym=False, model=nn.ModuleDict({"q": nn.Linear(64, 64)}))
    assert asym.backend is not None and "awq" in type(asym.backend).__name__.lower()
    assert _format_supports_batched_pack(asym)


def test_gate_accepts_unrouted_autoround_format():
    # a scheme that matches no routing branch (3-bit int asym) keeps
    # backend=None -- the only case where per-module and batched packing use
    # the same qlinear_torch packer.
    fmt = _fmt("auto_round", sym=False, bits=3)
    assert fmt.backend is None
    assert _format_supports_batched_pack(fmt)


def test_gate_still_rejects_fp_and_fake_variants():
    for fmt in ("auto_round:nv_fp4", "auto_round:mxfp8", "auto_round:fake"):
        assert not _format_supports_batched_pack(SimpleNamespace(output_format=fmt, format_name="auto_round"))


def test_batched_packer_refuses_llm_compressor_backend():
    m = nn.ModuleDict({"a": nn.Linear(64, 8, bias=False)})
    m["a"].bits, m["a"].group_size, m["a"].sym = 4, 32, False
    m["a"].act_bits, m["a"].data_type = 16, "int"
    m["a"].scale = torch.rand(8, 2) * 0.01 + 0.001
    m["a"].zp = torch.zeros(8, 2)
    with pytest.raises(ValueError, match="llm_compressor"):
        pack_layers_batched(["a"], m, backend="auto_round:llm_compressor")


class _TwoExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = nn.Linear(32, 32)
        self.e1 = nn.Linear(32, 32)
        self.global_name = "blk"
        for m in (self.e0, self.e1):
            m.bits, m.group_size, m.sym, m.act_bits, m.data_type = 4, 32, False, 16, "int"
            m.scale = torch.rand(32, 1) * 0.01 + 0.001
            m.zp = torch.zeros(32, 1)


def _run_immediate_pack_block(monkeypatch, tmp_path, pack_device="cuda:0"):
    """Drive immediate_pack_block with faked contexts; returns (batched_calls, per_module_calls)."""
    import sys

    from auto_round.compressors import utils as cu
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext

    fmt = _fmt("auto_round", sym=False, bits=3)  # unrouted: batched-eligible
    fake_cc = SimpleNamespace(is_immediate_packing=True, formats=[fmt])
    fake_mc = SimpleNamespace(model=nn.ModuleDict({"blk": _TwoExpertBlock()}))
    # plain functions replace classmethods; class-attribute access passes no args
    monkeypatch.setattr(CompressContext, "get_context", lambda: fake_cc)
    monkeypatch.setattr(ModelContext, "get_context", lambda: fake_mc)
    # the package attribute "auto_round.utils.device_manager" is the singleton
    # INSTANCE (it shadows the submodule), so patch the real module via sys.modules
    _dmm = sys.modules["auto_round.utils.device_manager"]
    monkeypatch.setattr(_dmm, "get_packing_device", lambda d: torch.device(pack_device))
    batched, per_module = [], []
    monkeypatch.setattr(
        "auto_round.export.export_to_autoround.export.pack_layers_batched",
        lambda names, model, backend=None, device=None, **kw: batched.append(list(names)) or (list(names), []),
    )
    monkeypatch.setattr(cu, "immediate_pack", lambda name, cfg: per_module.append(name))
    cu.immediate_pack_block(fake_mc.model["blk"], "blk", layer_config={}, nblocks=1)
    return batched, per_module


def test_env_var_forces_per_module_path(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_BATCHED_PACKING", "0")
    batched, per_module = _run_immediate_pack_block(monkeypatch, tmp_path)
    assert batched == []  # disabled: the batched packer must not run
    assert sorted(per_module) == ["blk.e0", "blk.e1"]


def test_batched_engages_without_env_var(monkeypatch, tmp_path):
    # Control for the test above: same harness, env unset ("auto" on a cuda
    # pack device) -> the batched packer IS consulted (proves the disable
    # test is not vacuously green).
    monkeypatch.delenv("AR_BATCHED_PACKING", raising=False)
    batched, per_module = _run_immediate_pack_block(monkeypatch, tmp_path)
    assert batched == [["blk.e0", "blk.e1"]]
    assert per_module == []


def test_env_var_forces_batched_on_cpu(monkeypatch, tmp_path):
    # AR_BATCHED_PACKING=1 overrides the "auto" CPU default (benching).
    monkeypatch.setenv("AR_BATCHED_PACKING", "1")
    batched, per_module = _run_immediate_pack_block(monkeypatch, tmp_path, pack_device="cpu")
    assert batched == [["blk.e0", "blk.e1"]]
    assert per_module == []


def test_auto_disables_batched_on_cpu(monkeypatch, tmp_path):
    # "auto" (env unset) keeps the per-module path on CPU pack devices.
    monkeypatch.delenv("AR_BATCHED_PACKING", raising=False)
    batched, per_module = _run_immediate_pack_block(monkeypatch, tmp_path, pack_device="cpu")
    assert batched == []
    assert sorted(per_module) == ["blk.e0", "blk.e1"]
