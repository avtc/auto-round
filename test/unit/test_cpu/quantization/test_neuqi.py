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
import logging

import pytest
import torch

from auto_round.data_type.int import quant_tensor_asym
from auto_round.data_type.neuqi import _log_search_engaged, neuqi_search_scale_zero, quant_tensor_opt_rtn_asym
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import get_quant_func


def _heavy_tailed_data(n_groups=64, group_size=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = torch.randn(n_groups, group_size, generator=gen)
    # inject a few large outliers per group to make min/max suboptimal
    data[:, :5] *= 120.0
    return data


def _mse(x, y):
    return ((x.float() - y.float()) ** 2).mean().item()


def _oracle_loss_fp64(data, qw, scale, zp, maxq):
    """fp64 loss of one group under fixed (scale, zp) -- the tie-break oracle."""
    r = torch.round(data.double() / scale.double()).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r + zp.double()).clamp_(0, maxq)
    deq = scale.double() * (q - zp.double())
    loss = (deq - data.double()) ** 2
    if qw is not None:
        loss = loss * qw.double()
    return loss.sum(-1)


class TestNeuqiSearch:
    def test_search_returns_valid_scale_and_zp(self):
        """The brute-force grid sweep is the only sweep: it must return a valid
        (scale, zp) pair for weighted data."""
        import auto_round.data_type.neuqi as N

        torch.manual_seed(5)
        data = torch.randn(512, 64) * 0.05
        qw = torch.rand(512, 64) + 0.1
        scale, zp = N.neuqi_search_scale_zero(data, bits=4, qw=qw, coarse_n=4, fine_n=3)
        assert scale.shape == (512, 1) and zp.shape == (512, 1)
        assert (zp >= 0).all() and (zp <= 15).all()

    def test_beats_minmax_on_heavy_tailed_data(self):
        data = _heavy_tailed_data()
        bits, group_size = 4, 128

        minmax_qdq, _, _ = quant_tensor_asym(data.clone(), bits=bits, group_size=group_size)
        neuqi_qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=bits, group_size=group_size)

        assert _mse(neuqi_qdq, data) < _mse(minmax_qdq, data)

    def test_zero_point_integral_and_in_range(self):
        data = _heavy_tailed_data(n_groups=32)
        scale, zp = neuqi_search_scale_zero(data, bits=4)
        assert torch.all(zp == zp.round())
        assert torch.all(zp >= 0) and torch.all(zp <= 15)
        assert torch.all(scale > 0)

    def test_close_to_dense_exhaustive_reference(self):
        """On a small case the search must approach a dense exhaustive grid."""
        gen = torch.Generator().manual_seed(3)
        n, g, bits = 8, 16, 2
        maxq = 2**bits - 1
        data = torch.randn(n, g, generator=gen)
        data[:, 0] *= 50.0

        scale, zp = neuqi_search_scale_zero(data, bits=bits)
        r = torch.round(data / scale)
        q = (r + zp).clamp(0, maxq)
        our_loss = (((scale * (q - zp)) - data) ** 2).sum().item()

        # brute force: dense additive scale grid per group x all integer zero points
        best = torch.full((n,), float("inf"))
        wmin = torch.clamp(data.min(-1).values, max=0)
        wmax = torch.clamp(data.max(-1).values, min=0)
        s0 = ((wmax - wmin) / maxq).clamp(min=1e-5)
        for i in range(401):
            frac = 0.05 + (1.0 - 0.05) * i / 400
            s = (s0 * frac).unsqueeze(-1)
            r = torch.round(data / s)
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                loss = ((s * (q - z) - data) ** 2).sum(-1)
                best = torch.minimum(best, loss)
        exhaustive_loss = best.sum().item()
        # within 1% of the (much denser) exhaustive grid
        assert our_loss <= exhaustive_loss * 1.01 + 1e-9

    def test_weighting_changes_solution(self):
        """A per-element weight that de-emphasizes outliers must pull the scale down."""
        gen = torch.Generator().manual_seed(1)
        data = torch.randn(16, 64, generator=gen)
        data[:, :4] *= 100.0

        qw = torch.ones_like(data)
        qw[:, :4] = 1e-4  # ignore outliers

        scale_plain, _ = neuqi_search_scale_zero(data, bits=4)
        scale_weighted, _ = neuqi_search_scale_zero(data, bits=4, qw=qw)
        # ignoring outliers must never increase the scale and must shrink it somewhere
        assert torch.all(scale_weighted <= scale_plain + 1e-6)
        assert torch.any(scale_weighted < scale_plain)

        def weighted_loss(scale):
            maxq = 15
            r = torch.round(data / scale).clamp(-maxq - 1, 2 * maxq + 1)
            losses = []
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                losses.append((((scale * (q - z) - data) ** 2) * qw).sum(-1))
            return torch.stack(losses, -1).min(-1).values.sum()

        assert weighted_loss(scale_weighted) < weighted_loss(scale_plain)

    def test_constant_group_is_exact(self):
        data = torch.full((4, 128), 0.25)
        qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        # the fp16 scale cast limits exactness to one quantization step (0.25 / 15)
        assert (qdq - data).abs().max() <= 0.25 / 15
        assert torch.all(scale > 0)

    def test_shapes_preserved(self):
        gen = torch.Generator().manual_seed(2)
        for shape in [(4, 33, 128), (17, 65), (3, 5, 7, 11)]:
            tensor = torch.randn(*shape, generator=gen)
            qdq, scale, zp = quant_tensor_opt_rtn_asym(tensor.clone(), bits=4, group_size=32)
            assert qdq.shape == tensor.shape


class _LogCapture(logging.Handler):
    records: list

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


class TestNeuqiLogging:
    def test_coarse_grid_is_recreated_per_call(self):
        """A cached grid is a cross-stream hazard in the expert fan-out: the
        tensor is filled on the first caller's stream and read from other
        threads' streams with no event ordering (49aa6405, device-side assert
        on the streaming parent). Recreating it per call keeps creation on the
        calling thread's own stream -- this locks that invariant in."""
        import auto_round.data_type.neuqi as N

        args = (0.001, 8, torch.device("cpu"), torch.float32)
        a = N._coarse_grid(*args)
        b = N._coarse_grid(*args)
        assert a is not b, "grid must not be shared state across calls"
        assert not hasattr(N._coarse_grid, "cache_info"), "tensor-returning helper must not be lru_cache'd"
        torch.testing.assert_close(a, b)  # values stay deterministic across calls

    def test_engaged_log_line(self):
        _log_search_engaged.cache_clear()
        cap = _LogCapture()
        ar_logger = logging.getLogger("autoround")
        ar_logger.addHandler(cap)
        try:
            quant_tensor_opt_rtn_asym(_heavy_tailed_data(n_groups=4).clone(), bits=4, group_size=128)
        finally:
            ar_logger.removeHandler(cap)
        assert any("[NeUQI]" in m and "search active" in m for m in cap.records)


class TestNeuqiSymLogging:
    def test_sym_engaged_logs_once_per_unique_grid(self):
        """Grid sweeps cycle configs in one process; the engaged line must fire
        once per unique (coarse, fine), not once per call. The project logger
        ("autoround") does not propagate, so capture with a direct handler
        (same pattern as TestNeuqiLogging._LogCapture)."""
        import logging

        import auto_round.data_type.neuqi as N

        records = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        lg = logging.getLogger("autoround")
        h = _Cap()
        lg.addHandler(h)
        try:
            N._sym_engaged_logged.clear()
            N._log_sym_search_engaged(64, 32)
            N._log_sym_search_engaged(64, 32)
            N._log_sym_search_engaged(128, 64)
            N._log_sym_search_engaged(64, 32)
        finally:
            lg.removeHandler(h)
        msgs = [m for m in records if "two-stage symmetric" in m]
        assert len(msgs) == 2  # (64,32) once + (128,64) once


class TestNeuqiIntegration:
    def test_registration_and_dispatch(self):
        assert "opt_rtn_int_asym" in QUANT_FUNC_WITH_DTYPE
        func, name = get_quant_func(
            "int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="neuqi"
        )
        assert name == "opt_rtn_int_asym"
        assert func is QUANT_FUNC_WITH_DTYPE["opt_rtn_int_asym"]
        # symmetric path must remain untouched
        _, sym_name = get_quant_func("int", 4, True, disable_opt_rtn=False, group_size=None, iters=0)
        assert sym_name == "opt_rtn_int_sym"

    def test_search_beats_plain_minmax(self):
        """With NeUQI opted in (the only way it runs now), the joint search must
        beat the plain min/max initializer on heavy-tailed groups."""
        data = _heavy_tailed_data(n_groups=16)
        qdq_enabled, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        qdq_plain, _, _ = quant_tensor_asym(data.clone(), bits=4, group_size=128)
        assert _mse(qdq_enabled, data) < _mse(qdq_plain, data)

    def test_imatrix_weighting_improves_weighted_mse(self):
        data = _heavy_tailed_data(n_groups=32)
        imatrix = torch.rand(128) + 0.1

        qdq_plain, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        qdq_im, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128, imatrix=imatrix)

        qw = imatrix.expand(32, 128)
        weighted = lambda q: (((q - data) ** 2) * qw).sum().item()  # noqa: E731
        assert weighted(qdq_im) < weighted(qdq_plain)

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_bits_smoke(self, bits):
        data = _heavy_tailed_data(n_groups=8, group_size=64, seed=bits)
        qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=bits, group_size=64)
        maxq = 2**bits - 1
        assert torch.all(zp >= 0) and torch.all(zp <= maxq)
        assert torch.all(scale > 0)
        assert qdq.shape == data.shape


class TestNeuqiFusedSweep:
    """The torch.compile-fused zero-point sweep must reproduce the eager reference.

    ``AR_NEUQI_BACKEND=compile`` routes every chunk through the fused expression.
    Selections must match the eager sweep; when compilation is unavailable in the
    environment (e.g. no host C++ compiler for Inductor) the dispatch must fall
    back to the eager expression and still return correct results."""

    @staticmethod
    def _oracle_loss_fp64(data, qw, scale, zp, maxq):
        return _oracle_loss_fp64(data, qw, scale, zp, maxq)

    def _search_with_backend(self, monkeypatch, backend, data, bits, qw, **kwargs):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", backend)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        scale, zp = N.neuqi_search_scale_zero(data.clone(), bits, qw=qw.clone() if qw is not None else None, **kwargs)
        return scale, zp, N

    @pytest.mark.enable_torch_compile
    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    @pytest.mark.parametrize("weighted", [False, True])
    def test_fused_matches_eager(self, monkeypatch, bits, weighted):
        gen = torch.Generator().manual_seed(bits * 10 + int(weighted))
        n, g = 512, 128
        data = torch.randn(n, g, generator=gen)
        data[:, :5] *= 120.0  # heavy tails: near-tie zero points are common here
        qw = torch.rand(n, g, generator=gen) + 0.1 if weighted else None

        scale_e, zp_e, _ = self._search_with_backend(monkeypatch, "eager", data, bits, qw, coarse_n=8, fine_n=4)
        scale_f, zp_f, N = self._search_with_backend(monkeypatch, "compile", data, bits, qw, coarse_n=8, fine_n=4)

        if N._fused_zp_broken or not N._fused_zp_fns:
            pytest.skip("torch.compile/inductor unavailable in this environment")

        mismatch = (zp_e != zp_f).logical_or(scale_e != scale_f).nonzero().flatten()
        for i in mismatch.tolist():
            # a flip is acceptable only between zero points whose fp64 losses tie
            # at the last fp32 ulp (summation-order artifacts), never a real loss change
            loss_e = self._oracle_loss_fp64(
                data[i], qw[i] if qw is not None else None, scale_e[i], zp_e[i], 2**bits - 1
            )
            loss_f = self._oracle_loss_fp64(
                data[i], qw[i] if qw is not None else None, scale_f[i], zp_f[i], 2**bits - 1
            )
            assert (loss_f - loss_e).abs().item() <= 1e-6 * loss_e.abs().item()

    @pytest.mark.enable_torch_compile
    def test_last_layout_matches_mid_reference(self, monkeypatch):
        """The CUDA layout (group axis last) must match the reference layout's
        selections up to fp64 ties -- it runs on CUDA, its numerics are validated
        here on CPU."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        gen = torch.Generator().manual_seed(31)
        data = torch.randn(512, 64, generator=gen)
        data[:, :4] *= 90.0
        qw = torch.rand(512, 64, generator=gen) + 0.1

        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        monkeypatch.setenv("AR_NEUQI_LAYOUT", "mid")
        scale_m, zp_m = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=16, fine_n=4)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setenv("AR_NEUQI_LAYOUT", "last")
        scale_l, zp_l = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=16, fine_n=4)

        if N._fused_zp_broken or not N._fused_zp_fns:
            pytest.skip("torch.compile/inductor unavailable in this environment")

        mismatch = (zp_m != zp_l).logical_or(scale_m != scale_l).nonzero().flatten()
        maxq = 15
        for i in mismatch.tolist():
            loss_m = self._oracle_loss_fp64(data[i], qw[i], scale_m[i], zp_m[i], maxq)
            loss_l = self._oracle_loss_fp64(data[i], qw[i], scale_l[i], zp_l[i], maxq)
            assert (loss_l - loss_m).abs().item() <= 1e-6 * loss_m.abs().item()

    def test_compile_failure_falls_back_to_eager(self, monkeypatch):
        """A broken host toolchain must degrade to the eager sweep, not crash."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        data = _heavy_tailed_data(n_groups=16, seed=11)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated inductor failure")

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setattr(N, "_get_fused_zp_fn", _boom)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)

        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=4, fine_n=2)
        assert N._fused_zp_broken is True  # permanently latched after the first failure
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        scale_e, zp_e = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=4, fine_n=2)
        torch.testing.assert_close(scale_f, scale_e)
        torch.testing.assert_close(zp_f, zp_e)

    def test_backend_policy(self, monkeypatch):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        assert N._zp_wants_compile("cuda") is True
        assert N._zp_wants_compile("cpu") is False
        assert N._zp_wants_compile("hpu") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        assert N._zp_wants_compile("cpu") is True
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        assert N._zp_wants_compile("cuda") is False
        # unrecognized values behave as the reference sweep
        monkeypatch.setenv("AR_NEUQI_BACKEND", "nonsense")
        assert N._zp_wants_compile("cuda") is False

    def test_layout_policy(self, monkeypatch):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_LAYOUT", "auto")
        assert N._zp_expr_for("cuda") is N._zp_expr_last
        assert N._zp_expr_for("cpu") is N._zp_expr_mid
        assert N._zp_expr_for("hpu") is N._zp_expr_mid
        monkeypatch.setenv("AR_NEUQI_LAYOUT", "last")
        assert N._zp_expr_for("cpu") is N._zp_expr_last
        monkeypatch.setenv("AR_NEUQI_LAYOUT", "mid")
        assert N._zp_expr_for("cuda") is N._zp_expr_mid

    def test_cpu_default_is_eager(self, monkeypatch):
        """AR_NEUQI_BACKEND defaults to auto: CPU tensors never trigger a compile."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        N.neuqi_search_scale_zero(_heavy_tailed_data(n_groups=8, seed=13).clone(), bits=4, coarse_n=4, fine_n=2)
        assert not N._fused_zp_fns  # no compile attempt on CPU under auto


class TestNeuqiBatchedSweep:
    """The all-candidates batched sweep must reproduce the per-candidate search.

    ``AR_NEUQI_BATCH=on`` evaluates every coarse/fine candidate in one fused
    call per group chunk ([C, K, Z, g], min over z fused in-kernel, then first-min
    over candidates). Selections must match the per-candidate sweep; ties may
    resolve differently only within fp64-oracle tolerance."""

    def _search(self, monkeypatch, backend, batch, data, bits, qw, **kwargs):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", backend)
        monkeypatch.setenv("AR_NEUQI_BATCH", batch)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale, zp = N.neuqi_search_scale_zero(data.clone(), bits, qw=qw.clone() if qw is not None else None, **kwargs)
        return scale, zp, N

    @pytest.mark.enable_torch_compile
    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("bits", [2, 4])
    def test_batched_matches_per_candidate(self, monkeypatch, weighted, bits):
        gen = torch.Generator().manual_seed(bits * 7 + int(weighted))
        n, g = 384, 64
        data = torch.randn(n, g, generator=gen)
        data[:, :4] *= 100.0
        qw = torch.rand(n, g, generator=gen) + 0.1 if weighted else None

        scale_ref, zp_ref, _ = self._search(monkeypatch, "eager", "off", data, bits, qw, coarse_n=8, fine_n=3)
        scale_b, zp_b, N = self._search(monkeypatch, "compile", "on", data, bits, qw, coarse_n=8, fine_n=3)

        if N._batched_broken or N._zp_expr_batched not in N._fused_zp_fns:
            pytest.skip("torch.compile/inductor unavailable in this environment")

        mismatch = (zp_ref != zp_b).logical_or(scale_ref != scale_b).nonzero().flatten()
        maxq = 2**bits - 1
        for i in mismatch.tolist():
            loss_ref = self._oracle_loss_fp64(data[i], qw[i] if qw is not None else None, scale_ref[i], zp_ref[i], maxq)
            loss_b = self._oracle_loss_fp64(data[i], qw[i] if qw is not None else None, scale_b[i], zp_b[i], maxq)
            assert (loss_b - loss_ref).abs().item() <= 1e-6 * loss_ref.abs().item()

    def test_batch_failure_falls_back_to_per_candidate(self, monkeypatch):
        """A batched-call failure must degrade to the per-candidate sweep, not crash."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        data = _heavy_tailed_data(n_groups=24, group_size=64, seed=17)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setenv("AR_NEUQI_BATCH", "on")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        monkeypatch.setattr(N, "_eval_batched_chunk", lambda *a, **k: None)  # simulate failure

        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)

        # reference: per-candidate sweep (batch=off never touches the patched fn)
        monkeypatch.setenv("AR_NEUQI_BATCH", "off")
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        torch.testing.assert_close(scale_f, scale_ref)
        torch.testing.assert_close(zp_f, zp_ref)

    def test_batch_policy(self, monkeypatch):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        monkeypatch.setenv("AR_NEUQI_BATCH", "auto")
        assert N._zp_batch_wanted("cuda") is True
        assert N._zp_batch_wanted("cpu") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        assert N._zp_batch_wanted("cuda") is False  # batched needs the fused backend
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setenv("AR_NEUQI_BATCH", "off")
        assert N._zp_batch_wanted("cuda") is False
        monkeypatch.setenv("AR_NEUQI_BATCH", "on")
        assert N._zp_batch_wanted("cpu") is True  # forced (e.g. for tests / A/B)


class TestNeuqiTritonSweep:
    """The extension Triton sweep integrates behind the batched dispatch.

    Triton kernels cannot run on CPU hosts, so the plumbing (scale construction,
    first-min bookkeeping, latch fallbacks, policy) is tested with a mock that
    honors the kernel's contract; the kernel itself is exercised on CUDA by
    ``test/unit/test_cuda/quantization/test_neuqi_triton.py`` and by the
    self-contained GPU benchmark."""

    def _mock_sweep_from(self, monkeypatch, impl):
        import auto_round.data_type.neuqi as N

        monkeypatch.setattr(N, "_triton_checked", True)
        monkeypatch.setattr(N, "_triton_sweep", impl)
        monkeypatch.setattr(N, "_triton_broken", False)

    def test_triton_mock_matches_reference(self, monkeypatch):
        """Driving the batched passes through a contract-honoring mock must
        reproduce the per-candidate reference selections exactly."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        def mock(data, qw, scales, maxq):  # same contract as neuqi_sweep_triton
            zp_grid = torch.arange(0, maxq + 1, device=data.device, dtype=torch.float32)
            return N._zp_expr_batched(data, qw, scales, zp_grid, maxq)

        self._mock_sweep_from(monkeypatch, mock)
        data = _heavy_tailed_data(n_groups=64, group_size=64, seed=23)
        qw = torch.rand(64, 64) + 0.1

        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        monkeypatch.setenv("AR_NEUQI_BATCH", "on")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale_t, zp_t = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=8, fine_n=3)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        monkeypatch.setenv("AR_NEUQI_BATCH", "off")
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=8, fine_n=3)
        torch.testing.assert_close(scale_t, scale_ref)
        torch.testing.assert_close(zp_t, zp_ref)

    def test_triton_failure_latches_to_compile_path(self, monkeypatch):
        """A raising Triton sweep must latch off and finish via the torch paths."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        def boom(data, qw, scales, maxq):
            raise RuntimeError("simulated triton failure")

        self._mock_sweep_from(monkeypatch, boom)
        data = _heavy_tailed_data(n_groups=32, group_size=64, seed=29)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        monkeypatch.setenv("AR_NEUQI_BATCH", "on")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        assert N._triton_broken is True

        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        torch.testing.assert_close(scale_f, scale_ref)
        torch.testing.assert_close(zp_f, zp_ref)

    def test_resolver_logs_engagement(self, monkeypatch):
        """A successful resolution announces itself: the Triton arm is silent
        otherwise, and live runs must be able to tell which backend served."""
        import sys
        import types

        import auto_round.data_type.neuqi as N

        stub = types.ModuleType("auto_round_extension.triton.neuqi_sweep")
        stub.neuqi_sweep_triton = object()
        monkeypatch.setitem(sys.modules, "auto_round_extension.triton.neuqi_sweep", stub)
        recorded = []
        monkeypatch.setattr(N, "logger", types.SimpleNamespace(info=recorded.append))
        monkeypatch.setattr(N, "_triton_checked", False)
        monkeypatch.setattr(N, "_triton_sweep", None)
        assert N._triton_sweep_fn() is stub.neuqi_sweep_triton
        assert any("Triton zero-point sweep engaged" in str(r) for r in recorded)

    def test_triton_policy(self, monkeypatch):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        monkeypatch.setenv("AR_NEUQI_BATCH", "auto")
        assert N._zp_wants_triton("cuda") is True
        assert N._zp_wants_triton("cpu") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        assert N._zp_wants_triton("cuda") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        assert N._zp_wants_triton("cpu") is True  # forced (availability still checked)
        # the Triton sweep IS the batched evaluator: batching off disables it
        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        monkeypatch.setenv("AR_NEUQI_BATCH", "off")
        assert N._zp_batch_wanted("cuda") is False


class TestAsymSearchAPI:
    """RTNConfig(asym_search) enum: auto | neuqi | minmax (asym path only).

    The joint NeUQI search is opt-in: only 'neuqi' resolves the optimized
    asym initializer; 'auto' (default) and 'minmax' use plain min/max."""

    def test_default_auto_resolves_plain_minmax(self):
        from auto_round.data_type.int import quant_tensor_asym as plain_asym

        func, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0)
        assert "opt_rtn" not in name
        assert name.endswith("_asym")
        assert func is plain_asym

    def test_neuqi_opt_in_resolves_joint_search(self):
        func, name = get_quant_func(
            "int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="neuqi"
        )
        assert name == "opt_rtn_int_asym"
        assert func is QUANT_FUNC_WITH_DTYPE["opt_rtn_int_asym"]

    def test_minmax_resolves_plain_asym(self):
        from auto_round.data_type.int import quant_tensor_asym as plain_asym

        func, name = get_quant_func(
            "int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax"
        )
        assert "opt_rtn" not in name
        assert name.endswith("_asym")
        assert func is plain_asym

    def test_minmax_does_not_touch_sym_path(self):
        _, sym_name = get_quant_func(
            "int", 4, True, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax"
        )
        assert sym_name == "opt_rtn_int_sym"

    def test_minmax_output_matches_plain_quant(self):
        torch.manual_seed(7)
        data = _heavy_tailed_data(n_groups=8, group_size=64, seed=7)
        func, _ = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, asym_search="minmax")
        from auto_round.data_type.int import quant_tensor_asym

        expected = quant_tensor_asym(data.clone(), bits=4, group_size=64)
        actual = func(data.clone(), bits=4, group_size=64)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_force_neuqi_accepted_when_engageable(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=False, asym_search="neuqi")
        assert cfg.asym_search == "neuqi"

    def test_non_auto_with_sym_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        # neuqi + sym engages the two-stage symmetric scale search (opt_rtn_int_sym_neuqi)
        RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, asym_search="neuqi")
        with pytest.raises(ValueError, match="asymmetric path only"):
            RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, asym_search="minmax")

    def test_neuqi_with_opt_rtn_disabled_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match='asym_search="neuqi" requires the optimized-RTN path'):
            RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=True, asym_search="neuqi")

    def test_coarse_grid_matches_logspace(self):
        """The eager coarse-grid helper must be numerically identical to the
        previous inline torch.logspace."""
        import math

        from auto_round.data_type.neuqi import _coarse_grid

        for lo in (0.03125, 0.25):
            got = _coarse_grid(lo, 64, torch.device("cpu"), torch.float32)
            want = torch.logspace(math.log10(lo), 0.0, 64, dtype=torch.float32)
            assert got.shape == want.shape
            assert torch.equal(got, want)

    def test_coarse_grid_is_dynamo_opaque(self):
        """torch.logspace lowers to an fp64 libdevice.pow inside compiled
        regions; inductor fused it into the batched sweep's index math where
        Triton rejects fp64 ("unexpected type fp64") -- see the BPT worker
        crash. The grid must be built eagerly and enter traced regions as a
        plain tensor input (no logspace op in the traced graph)."""
        import torch._dynamo as dynamo

        from auto_round.data_type.neuqi import _coarse_grid

        assert getattr(_coarse_grid, "_torchdynamo_disable", False) is True

        def traced(x):
            return _coarse_grid(0.25, 8, x.device, torch.float32) * x

        ex = dynamo.explain(traced)(torch.ones(8))
        op_names = [getattr(op, "__name__", str(op)) for ops in ex.ops_per_graph for op in ops]
        assert not any("logspace" in n for n in op_names), f"logspace leaked into the traced graph: {op_names}"
        assert any("mul" in n for n in op_names), "the sweep math itself should stay traced"

    def test_unknown_value_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match="not one of"):
            RTNConfig(bits=4, group_size=32, sym=False, asym_search="grid")

    def test_default_is_auto(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig(bits=4, group_size=32, sym=False)
        assert cfg.asym_search == "auto"


class TestNeuqiItersGuard:
    """--enable_neuqi with iters>0 is a silent no-op today; it must fail fast."""

    def test_neuqi_with_iters_raises(self):
        import pytest

        from auto_round.data_type.utils import get_quant_func

        with pytest.raises(ValueError, match="zero-shot path"):
            get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=128, iters=20, asym_search="neuqi")

    def test_neuqi_zero_shot_still_resolves(self):
        from auto_round.data_type.utils import get_quant_func

        fn, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=128, iters=0, asym_search="neuqi")
        assert name.startswith("opt_rtn_int_asym")


class TestNeuqiSymSearch:
    """Two-stage symmetric scale search (zero point fixed at 0)."""

    def _skewed_data(self, n_groups=512, group_size=128, seed=0):
        gen = torch.Generator().manual_seed(seed)
        data = torch.randn(n_groups, group_size, generator=gen)
        data[:, :6] *= 60.0  # outliers make the clip search matter
        # half the groups skew negative so the mirrored clamp family is in play
        data[::2] *= -1.0
        return data

    def _sym_loss(self, data, qw, scale):
        """Literal loss of the shipped symmetric formula (signed scales allowed)."""
        iq = (data / scale).round().clamp(-8, 7)
        err = (scale * iq - data) ** 2
        if qw is not None:
            err = err * qw
        return err.sum(-1)

    def test_returns_signed_scales_with_magnitude_floor(self):
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data()
        qw = torch.rand(*data.shape) + 0.05
        scale = neuqi_search_scale_sym(data, bits=4, qw=qw, coarse_n=32, fine_n=16)
        assert scale.shape == (data.shape[0], 1)
        assert (scale.abs() >= 1e-5).all()
        # both clamp conventions must actually win somewhere on skewed data
        assert (scale < 0).any() and (scale > 0).any()

    def test_mirrored_family_matches_literal_negative_scale(self):
        """The mirrored-clamp loss used inside the search must equal the literal
        negative-scale evaluation of the shipped formula (guards the sign of the
        mirrored error term)."""
        from auto_round.data_type.neuqi import _sym_loss_chunk

        torch.manual_seed(3)
        data = torch.randn(64, 96)
        qw = torch.rand(64, 96) + 0.1
        scales = torch.rand(64, 8) * 0.05 + 0.005  # [C, K] positive candidates
        loss, idx, mirror = _sym_loss_chunk(data, qw, scales, nmax=8)
        for c in range(data.shape[0]):
            k = idx[c].item()
            s = scales[c, k]
            signed = -s if mirror[c] else s
            ref = self._sym_loss(data[c : c + 1], qw[c : c + 1], torch.tensor([[signed]]))[0]
            assert torch.allclose(loss[c], ref, rtol=1e-4, atol=1e-3)

    def test_signed_search_beats_positive_only_bruteforce_on_skewed_groups(self):
        """Skewed groups: allowing the mirrored family must reach (near-)the signed
        brute-force optimum, strictly below the positive-only optimum."""
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(128, 96, seed=7)
        qw = torch.rand(*data.shape) + 0.05
        scale = neuqi_search_scale_sym(data, bits=4, qw=qw, coarse_n=64, fine_n=32)
        got = self._sym_loss(data, qw, scale)
        # dense positive-only reference over the same anchor
        s0 = data.abs().amax(dim=-1, keepdim=True) / 8
        fracs = torch.logspace(-0.9, 0.3, 2000)
        ref = torch.full_like(got, float("inf"))
        for f in fracs:
            l = self._sym_loss(data, qw, s0 * f)
            ref = torch.minimum(ref, l)
        # aggregate must win: the mirrored family is reachable, and the few
        # two-stage basin misses (coarse grid skips a narrow basin) stay small
        assert got.sum() <= ref.sum() * 1.001
        # and on genuinely skewed groups it must beat the positive-only optimum
        assert (got < ref - 1e-3).float().mean() > 0.25

    def test_covers_incumbent_uniform_grid(self):
        """Total weighted loss must not exceed search_scales (both searches see the
        same data, weights and signed-scale families)."""
        from auto_round.data_type.int import search_scales
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(512, 128, seed=11)
        qw = torch.rand(*data.shape) + 0.05
        s_neuqi = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=64, fine_n=32)
        s_old = search_scales(data.clone(), bits=4, qw=qw.clone())
        l_neuqi = self._sym_loss(data, qw, s_neuqi).sum().item()
        l_old = self._sym_loss(data, qw, s_old).sum().item()
        assert l_neuqi <= l_old * 1.001

    def test_union_variant_dominates_both_parents_per_group(self, monkeypatch):
        """AR_NEUQI_SYM_UNION=1: per-group loss must be <= BOTH the incumbent
        search_scales result and the plain two-stage result (candidate superset)."""
        from auto_round.data_type.int import search_scales
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(256, 96, seed=23)
        qw = torch.rand(*data.shape) + 0.05
        monkeypatch.setenv("AR_NEUQI_SYM_UNION", "0")
        s_b = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=64, fine_n=32)
        s_a = search_scales(data.clone(), bits=4, qw=qw.clone())
        monkeypatch.setenv("AR_NEUQI_SYM_UNION", "1")
        s_c = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=64, fine_n=32)
        monkeypatch.delenv("AR_NEUQI_SYM_UNION")
        l_c = self._sym_loss(data, qw, s_c)
        l_b = self._sym_loss(data, qw, s_b)
        l_a = self._sym_loss(data, qw, s_a)
        # dominance up to fp32 summation slack
        assert (l_c <= l_a + 1e-2 * l_a.abs() + 1e-3).all()
        assert (l_c <= l_b + 1e-2 * l_b.abs() + 1e-3).all()
        assert (l_c < l_b - 1e-3).any() or (l_c < l_a - 1e-3).any()  # union actually adds wins

    def test_union_env_off_matches_plain_two_stage_exactly(self, monkeypatch):
        monkeypatch.setenv("AR_NEUQI_SYM_UNION", "0")
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(128, 96, seed=29)
        qw = torch.rand(*data.shape) + 0.05
        a = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=32, fine_n=16)
        monkeypatch.delenv("AR_NEUQI_SYM_UNION")
        b = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=32, fine_n=16)
        assert torch.equal(a, b)


class TestNeuqiSymIntegration:
    def test_sym_neuqi_dispatch(self):
        fn, name = get_quant_func(
            "int", 4, True, disable_opt_rtn=False, group_size=32, iters=0, asym_search="neuqi", weight_path=False
        )
        assert name == "opt_rtn_int_sym_neuqi"
        assert "opt_rtn_int_sym_neuqi" in QUANT_FUNC_WITH_DTYPE

    def test_sym_auto_still_uses_incumbent(self):
        fn, name = get_quant_func(
            "int", 4, True, disable_opt_rtn=False, group_size=32, iters=0, asym_search="auto", weight_path=False
        )
        assert name == "opt_rtn_int_sym"

    def test_wrapper_qdq_matches_returned_signed_scale(self):
        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_sym_neuqi

        torch.manual_seed(13)
        w = torch.randn(32, 256)
        w[:, :20] *= -40.0  # negative-heavy columns exercise the mirrored family
        qdq, scale, zp = quant_tensor_opt_rtn_sym_neuqi(w, bits=4, group_size=32)
        assert zp == 8
        assert scale.shape == (32 * 256 // 32, 1)
        s2d = torch.repeat_interleave(scale.reshape(32, 8), 32, dim=1)
        manual = (w / s2d).round().clamp(-8, 7) * s2d
        assert torch.allclose(qdq, manual, rtol=1e-3, atol=1e-4)
        assert (scale < 0).any()  # mirrored family won somewhere

    def test_sym_wrapper_supports_imatrix_off_1d_2d(self):
        """qw=None (imatrix off), per-column 1-D and per-row 2-D weights all flow;
        weighting must steer the search (lower weighted loss than the unweighted
        solution under the same weights)."""
        from auto_round.data_type.neuqi import neuqi_search_scale_sym, quant_tensor_opt_rtn_sym_neuqi

        torch.manual_seed(17)
        w = torch.randn(32, 256)
        w[:, :24] *= -40.0
        for im in (None, torch.rand(256) + 0.05, torch.rand(32, 256) + 0.05):
            qdq, scale, _ = quant_tensor_opt_rtn_sym_neuqi(w.clone(), bits=4, group_size=32, imatrix=im)
            assert torch.isfinite(qdq).all()

        d = w.reshape(-1, 32)
        qw = (torch.rand(32, 256) + 0.05).reshape(-1, 32)
        s_q = neuqi_search_scale_sym(d.clone(), bits=4, qw=qw.clone())
        s_p = neuqi_search_scale_sym(d.clone(), bits=4, qw=None)

        def wloss(s):
            iq = (d / s).round().clamp(-8, 7)
            return ((s * iq - d) ** 2 * qw).sum().item()

        assert wloss(s_q) < wloss(s_p)

    def test_config_allows_neuqi_sym_rejects_minmax_sym(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        RTNConfig(bits=4, group_size=32, sym=True, asym_search="neuqi")  # must not raise
        with pytest.raises(ValueError, match="minmax"):
            RTNConfig(bits=4, group_size=32, sym=True, asym_search="minmax")
