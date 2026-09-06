# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""lm_head tuning inputs from the calibration chain tail (streaming, iters>0).

At iters>0 the streaming zero-shot pass tunes lm_head with the chain's final
hidden states, exactly like the data-driven path tunes outside-block layers.
``_lm_head_tune_inputs_`` derives the per-sample row lists (unwrapping
dict-shaped chain tails) or returns ``None`` to keep the closed-form search.
"""

import inspect
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from auto_round.compressors.orchestrator import CompressionOrchestrator


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class _NoNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 8)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 8)


class _TrailingTree(nn.Module):
    """lm_head followed by an attached checkpoint-only placeholder subtree -
    the module-order "last leaf" lands inside the placeholder, not on lm_head."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)
        self.mtp = nn.Module()
        self.mtp.pre_fc_norm_hidden = nn.LayerNorm(4)


class _BackboneWrapper(nn.Module):
    """Text backbone + vision tower wrapper: the vision norms sit between the
    last text block and lm_head and share the 1D-weight (and even width)
    signature, so the final-norm scan must be restricted to the backbone."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Block()])
        self.model.language_model.norm = nn.LayerNorm(4)
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


def _orch(iters, remain, model=None):
    orch = SimpleNamespace(model=model or _Tiny())
    orch._max_tune_iters = lambda: iters
    for name in ("_lm_head_tune_inputs_", "_final_norm_module_", "_resolve_lm_head_name_"):
        setattr(orch, name, MethodType(getattr(CompressionOrchestrator, name), orch))
    # staticmethods: attach the plain function (instance attrs never bind self)
    orch._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows
    orch._block_backbone_prefix_ = CompressionOrchestrator._block_backbone_prefix_
    return orch


def _rows(n=2):
    return [torch.randn(1, 5, 4) for _ in range(n)]


class TestOutsideTuneChunking:
    """Position-chunked forward/backward for huge-output outside-block layers.

    A 248k-vocab lm_head tuned at seqlen 2048 OOMs a 24GB GPU if the MSE is
    computed over the full logits; chunking rows keeps transients bounded and
    must not change the tune (gradients accumulate to the same values)."""

    def test_rows_per_chunk_clamps_to_output_budget(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        # 64M-element budget: big vocab -> few rows per chunk, small dims -> everything
        assert qmod._outside_tune_rows_per_chunk(248320, 2048, 2**26) == 270
        assert qmod._outside_tune_rows_per_chunk(64, 32, 2**26) == 32
        assert qmod._outside_tune_rows_per_chunk(7, 100, 2**5) == 4  # at least one row

    def test_chunked_tune_matches_single_shot(self, monkeypatch):
        """Same layer, same seed: many tiny chunks must reproduce the single-shot
        tuned parameter bit-for-bit (gradient accumulation is exact)."""
        from types import MethodType

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        torch.manual_seed(3)
        layer = torch.nn.Linear(16, 5)
        fp = [torch.randn(1, 13, 16) for _ in range(2)]

        results = []
        for cap in (2**26, 2**4):  # single shot vs 3 rows per chunk
            torch.manual_seed(11)
            fresh = torch.nn.Linear(16, 5)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            fresh.global_name = "lm_head"
            fresh.bits = 8
            fresh.group_size = 32
            fresh.sym = True
            fresh.data_type = "int"
            fresh.scale_dtype = None
            fresh.iters = 3
            fresh.act_bits = 16
            fresh.act_sym = True
            fresh.act_data_type = None
            fresh.act_group_size = None
            cfg = type(
                "C",
                (),
                {
                    "iters": 3,
                    "lr": 5e-3,
                    "compute_lr": lambda self, bits: None,
                    "compute_minmax_lr": lambda self, bits: None,
                },
            )()
            quant = SimpleNamespace(
                config=cfg,
                _config=cfg,
                iters=3,
                lr=5e-3,
                minmax_lr=1e-3,
                lr_scheduler=None,
                enable_minmax_tuning=False,
                gradient_accumulate_steps=1,
                not_use_best_mse=False,
                dynamic_max_gap=0,
                optimizer=qmod.SignSGD,
                lr_is_auto=False,
                model=torch.nn.Module(),
                calibration_context=SimpleNamespace(batch_size=1),
                model_context=SimpleNamespace(amp=False, amp_dtype=torch.bfloat16),
                compress_context=SimpleNamespace(enable_torch_compile=False, cache_device="cpu"),
            )
            for name in (
                "_get_scaler",
                "_scale_loss_and_backward",
                "_step",
                "_maybe_log_low_bit_lr",
            ):
                setattr(quant, name, MethodType(getattr(qmod.SignRoundQuantizer, name), quant))
            quant._preallocate_tuning_grads_ = qmod.SignRoundQuantizer._preallocate_tuning_grads_
            setattr(quant, "_best_param_device", MethodType(qmod.SignRoundQuantizer._best_param_device, quant))
            quant._logged_low_bit_lr = set()
            monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", cap, raising=False)
            qmod.SignRoundQuantizer.quantize_layer_outside_block(
                quant, fresh, fp_inputs=[t.clone() for t in fp], input_ids=None
            )
            results.append(fresh.weight.detach().clone())
        assert torch.equal(results[0], results[1]), "chunked tune diverged from single-shot"

    def test_huge_value_parameter_skips_torch_compile(self, monkeypatch):
        """Inductor's backward adds a full-size buffer for billion-element
        rounding parameters; the wrapper must stay eager for those."""
        from types import MethodType

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        captured = {}
        real_wrapper = qmod.WrapperLinear

        def recording_wrapper(layer, **kwargs):
            captured.update(kwargs)
            return real_wrapper(layer, **kwargs)

        monkeypatch.setattr(qmod, "WrapperLinear", recording_wrapper)
        torch.manual_seed(5)
        layer = torch.nn.Linear(16, 5)
        layer.global_name = "lm_head"
        layer.bits = 8
        layer.group_size = 32
        layer.sym = True
        layer.data_type = "int"
        layer.scale_dtype = None
        layer.iters = 2
        layer.act_bits = 16
        layer.act_sym = True
        layer.act_data_type = None
        layer.act_group_size = None
        cfg = type(
            "C",
            (),
            {
                "iters": 2,
                "lr": 5e-3,
                "compute_lr": lambda self, bits: None,
                "compute_minmax_lr": lambda self, bits: None,
            },
        )()
        quant = SimpleNamespace(
            config=cfg,
            _config=cfg,
            iters=2,
            lr=5e-3,
            minmax_lr=1e-3,
            lr_scheduler=None,
            enable_minmax_tuning=False,
            gradient_accumulate_steps=1,
            not_use_best_mse=False,
            dynamic_max_gap=0,
            optimizer=qmod.SignSGD,
            lr_is_auto=False,
            model=torch.nn.Module(),
            calibration_context=SimpleNamespace(batch_size=1),
            model_context=SimpleNamespace(amp=False, amp_dtype=torch.bfloat16),
            compress_context=SimpleNamespace(enable_torch_compile=True, cache_device="cpu"),
        )
        for name in ("_get_scaler", "_scale_loss_and_backward", "_step", "_maybe_log_low_bit_lr"):
            setattr(quant, name, MethodType(getattr(qmod.SignRoundQuantizer, name), quant))
        quant._preallocate_tuning_grads_ = qmod.SignRoundQuantizer._preallocate_tuning_grads_
        setattr(quant, "_best_param_device", MethodType(qmod.SignRoundQuantizer._best_param_device, quant))
        quant._logged_low_bit_lr = set()
        fp = [torch.randn(1, 7, 16) for _ in range(2)]

        # 16x5=80 elements fit the default budget: compile honored
        qmod.SignRoundQuantizer.quantize_layer_outside_block(quant, layer, fp_inputs=[t.clone() for t in fp])
        assert captured["enable_torch_compile"] is True

        # shrink the budget below the layer size: compile must be skipped
        monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", 16, raising=False)
        fresh = torch.nn.Linear(16, 5)
        for attr, val in (
            ("global_name", "lm_head"),
            ("bits", 8),
            ("group_size", 32),
            ("sym", True),
            ("data_type", "int"),
            ("scale_dtype", None),
            ("iters", 2),
            ("act_bits", 16),
            ("act_sym", True),
            ("act_data_type", None),
            ("act_group_size", None),
        ):
            setattr(fresh, attr, val)
        qmod.SignRoundQuantizer.quantize_layer_outside_block(quant, fresh, fp_inputs=[t.clone() for t in fp])
        assert captured["enable_torch_compile"] is False


def _mk_quant_linear(out_f, in_f, bits=4, group_size=4):
    layer = torch.nn.Linear(in_f, out_f)
    layer.global_name = "lm_head"
    layer.bits = bits
    layer.group_size = group_size
    layer.sym = True
    layer.data_type = "int"
    layer.scale_dtype = None
    layer.iters = 2
    layer.act_bits = 16
    layer.act_sym = True
    layer.act_data_type = None
    layer.act_group_size = None
    return layer


class TestRowBlockedWrapperForward:
    """Huge-weight wrappers quantize one row block at a time; outputs and
    gradients must match the full-tensor path exactly (groups never straddle
    output rows)."""

    def _wrapper(self, layer, minmax=False):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(layer, enable_minmax_tuning=minmax, enable_torch_compile=False, device="cpu")

    def test_blocked_output_and_gradients_match(self, monkeypatch):
        import auto_round.wrapper as wmod

        torch.manual_seed(9)
        layer = _mk_quant_linear(24, 10, group_size=4)  # 240 elements
        x = torch.randn(3, 2, 10)
        target = torch.randn(3, 2, 24)

        results = []
        for cap in (2**26, 60):  # full path vs 4-row blocks (4 rows x 10 in)
            torch.manual_seed(21)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            out = wrapper(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            results.append(
                (out.detach().clone(), wrapper.value.grad.detach().clone(), wrapper.min_scale.grad.detach().clone())
            )
        assert torch.allclose(results[0][0], results[1][0], atol=1e-6), "blocked forward output diverged"
        assert torch.allclose(results[0][1], results[1][1], atol=1e-7), "blocked value grad diverged"
        assert torch.allclose(results[0][2], results[1][2], atol=1e-7), "blocked min_scale grad diverged"

    def test_blocked_with_per_group_init_scale_matches(self, monkeypatch):
        """Per-group init_scale (OptRTN/AWQ anchors) must follow the same row
        window as the other tuning parameters."""
        import auto_round.wrapper as wmod

        torch.manual_seed(4)
        layer = _mk_quant_linear(24, 10, group_size=4)  # 6 groups per row -> 144 groups
        x = torch.randn(2, 3, 10)
        target = torch.randn(2, 3, 24)
        n_groups = 24 * 3  # ceil handled by layout: in=10, gs=4 -> 3 groups/row

        results = []
        for cap, scale_shape in ((2**26, (n_groups, 1)), (60, (n_groups, 1))):
            torch.manual_seed(23)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            torch.manual_seed(31)
            wrapper.init_scale = torch.rand(scale_shape) * 0.01 + 0.005
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            out = wrapper(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            results.append(out.detach().clone())
        assert torch.allclose(results[0], results[1], atol=1e-6), "blocked init_scale forward diverged"

    def test_per_tensor_group_size_never_blocks(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=0)
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is False

    def test_attached_none_scheme_fields_still_block(self, monkeypatch):
        """The plan machinery attaches every scheme field (None when unset) to
        quantized layers; a None super_bits must not disable blocking."""
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        layer.super_bits = None
        layer.super_group_size = None
        layer.rotation_config = None
        layer.act_dynamic = True
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is True

    def test_real_super_bits_never_blocks(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        layer.super_bits = 6
        layer.super_group_size = 8
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is False

    def test_unwrapper_matches_full_path_when_blocked(self, monkeypatch):
        """The final quantize/dequantize in unwrapper must equal the full-tensor
        computation when it runs row-blocked (multi-block budget)."""
        import auto_round.wrapper as wmod

        torch.manual_seed(11)
        layer = _mk_quant_linear(24, 10, group_size=4)

        outputs = []
        for cap in (2**26, 40):  # full path vs 4-row blocks
            torch.manual_seed(5)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            with torch.no_grad():
                torch.manual_seed(6)
                for p in wrapper.parameters():
                    p.add_(torch.randn_like(p) * 1e-3)
                # per-group OptRTN/AWQ anchors take part in the final quantize
                wrapper.init_scale = torch.rand(24 * 3, 1) * 0.01 + 0.005
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            best = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}
            restored = wrapper.unwrapper(best)
            outputs.append(restored.weight.detach().clone())
        assert torch.equal(outputs[0], outputs[1]), "blocked unwrapper diverged from full quantize"

    def test_row_block_bounds_partition_rows(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 40, raising=False)
        bounds = wrapper.row_block_bounds()
        flat = [r for b in bounds for r in b]
        assert flat[0] == 0 and flat[-1] == 24
        assert all(e > s for s, e in bounds)
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)), "gaps/overlaps"
        assert len(bounds) > 1, "expected multiple blocks under the tiny budget"


class TestLmHeadNameResolution:
    """lm_head resolves from the quantization plan, never from module order."""

    def test_trailing_placeholder_tree_still_resolves_lm_head(self):
        orch = _orch(200, ["lm_head"], model=_TrailingTree())
        assert orch._resolve_lm_head_name_(["lm_head"]) == "lm_head"

    def test_trailing_placeholder_tree_tunes_end_to_end(self):
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        orch = _orch(200, ["lm_head"], model=_TrailingTree())
        out = orch._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is not None  # no silent closed-form fallback

    def test_unpinned_lm_head_resolves_to_none(self, capfd):
        assert _orch(200, ["norm"])._resolve_lm_head_name_(["norm"]) is None
        assert "warning" not in capfd.readouterr().err.lower()

    def test_multiple_candidates_warn(self, capfd):
        names = ["lm_head", "decoder.lm_head"]
        orch = _orch(200, names)
        assert orch._resolve_lm_head_name_(names) == "lm_head"
        assert "multiple lm_head candidates" in capfd.readouterr().err

    def test_substring_fallback_resolves_prefixed_heads(self):
        orch = _orch(200, ["model.lm_head_proj"])
        assert orch._resolve_lm_head_name_(["model.lm_head_proj"]) == "model.lm_head_proj"


class _FlatMoe(nn.Module):
    """Flat MoE layout: blocks and the final norm at the model root, with a
    top-level lm_head (the checkpoint-only predictor is a plain extra block
    in the list, not an attached placeholder)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([_Block(), _Block()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class _VisionWrapper(nn.Module):
    """Vision-wrapper layout: text backbone nested one level deeper, with
    same-width 1D norms in the projector and the vision tower sitting between
    the backbone norm and lm_head - the width cross-check alone would NOT
    reject these decoys."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(8, 4)
        self.model.language_model.layers = nn.ModuleList([_Block()])
        self.model.language_model.norm = nn.LayerNorm(4)
        self.model.multi_modal_projector = nn.Module()
        self.model.multi_modal_projector.norm = nn.LayerNorm(4)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.ln_pre = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class _FlatLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class TestFinalNormDiscovery:
    def test_wrapper_vision_norms_are_not_picked(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.language_model.layers.0"]])
        orch = _orch(200, ["lm_head"], model=_BackboneWrapper())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.language_model.norm"
        assert mod is orch.model.model.language_model.norm

    def test_flat_model_norm_still_found(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.layers.0"]])
        orch = _orch(200, ["lm_head"], model=_FlatLlama())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.norm"
        assert mod is orch.model.model.norm

    def test_flat_moe_layout(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.layers.0"], ["model.layers.1"]])
        orch = _orch(200, ["lm_head"], model=_FlatMoe())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.norm"
        assert mod is orch.model.model.norm

    def test_vision_wrapper_same_width_decoys_rejected(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "get_block_names", lambda model: [["model.language_model.layers.0"]])
        orch = _orch(200, ["lm_head"], model=_VisionWrapper())
        name, mod = orch._final_norm_module_("lm_head")
        assert name == "model.language_model.norm"
        assert mod is orch.model.model.language_model.norm
        # the decoys share the hidden width; only the backbone filter rejects them
        assert orch.model.model.vision_tower.ln_pre.weight.numel() == 4
        assert orch.model.model.multi_modal_projector.norm.weight.numel() == 4


class TestLmHeadTuneInputs:
    def test_zero_shot_run_keeps_closed_form(self, capfd):
        out = _orch(0, ["lm_head"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["lm_head"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_unpinned_lm_head_untouched(self, capfd):
        out = _orch(200, ["norm"])._lm_head_tune_inputs_({"fp_inputs": _rows(), "token_ids": []}, ["norm"])
        assert out is None
        assert "falls back" not in capfd.readouterr().err

    def test_pinned_tuning_run_returns_post_norm_rows(self):
        rows, qrows, ids = _rows(3), _rows(3), [torch.randint(0, 8, (1, 5)) for _ in range(3)]
        state = {"fp_inputs": rows, "q_inputs": qrows, "token_ids": ids}
        orch = _orch(200, ["lm_head"])
        fp, q, tok = orch._lm_head_tune_inputs_(state, ["lm_head"])
        norm = orch.model.norm
        for got, src_rows in ((fp, rows), (q, qrows)):
            assert tok is ids
            for r, s in zip(got, src_rows):
                torch.testing.assert_close(r, norm(s.to(norm.weight.dtype)), msg="rows must be post-final-norm")
        assert fp is not rows  # fresh list: the tune loop reassigns entries in place

    def test_missing_final_norm_falls_back(self, capfd):
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        out = _orch(200, ["lm_head"], model=_NoNorm())._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is None
        assert "cannot locate the final norm" in capfd.readouterr().err

    def test_meta_final_norm_without_streamer_falls_back(self, capfd):
        orch = _orch(200, ["lm_head"])
        with torch.device("meta"):
            orch.model.norm = nn.LayerNorm(4)
        state = {"fp_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 2}
        out = orch._lm_head_tune_inputs_(state, ["lm_head"], streamer=None)
        assert out is None
        assert "final norm is still meta" in capfd.readouterr().err

    def test_dict_chain_tail_unwraps_hidden_states(self):
        rows, qrows, ids = _rows(2), _rows(2), [torch.randint(0, 8, (1, 5)) for _ in range(2)]
        state = {
            "fp_inputs": {"hidden_states": rows, "prev_topk_indices": torch.zeros(2)},
            "q_inputs": {"hidden_states": qrows, "prev_topk_indices": torch.zeros(2)},
            "token_ids": ids,
        }
        orch = _orch(200, ["lm_head"])
        fp, q, _ = orch._lm_head_tune_inputs_(state, ["lm_head"])
        norm = orch.model.norm
        for got, src_rows in ((fp, rows), (q, qrows)):
            for r, s in zip(got, src_rows):
                torch.testing.assert_close(r, norm(s.to(norm.weight.dtype)))

    def test_missing_chain_state_warns_and_falls_back(self, capfd):
        out = _orch(200, ["lm_head"])._lm_head_tune_inputs_({}, ["lm_head"])
        assert out is None
        assert "falls back" in capfd.readouterr().err

    def test_malformed_rows_warn_and_fall_back(self, capfd):
        state = {"fp_inputs": torch.zeros(2), "token_ids": [torch.zeros(1, 5)]}
        out = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert out is None
        assert "falls back" in capfd.readouterr().err

    def test_mis_shaped_q_rows_tune_on_fp_inputs(self, capfd):
        state = {"fp_inputs": _rows(3), "q_inputs": _rows(2), "token_ids": [torch.zeros(1, 5)] * 3}
        fp, q, _ = _orch(200, ["lm_head"])._lm_head_tune_inputs_(state, ["lm_head"])
        assert fp is not None and q is None
        assert "enable_quanted_input" in capfd.readouterr().err


class TestLmHeadTuneWiring:
    """Contract pins for the outside-block loop call site."""

    def test_zero_shot_loop_wires_tune_kwargs(self):
        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "self._lm_head_tune_inputs_(" in src
        i = src.index("compress_layer_outside_block(")
        window = src[i : i + 400]
        assert "**tune_kwargs" in window
        assert '"fp_inputs": _fp_rows' in src
        assert '"input_ids": _token_ids' in src

    def test_helper_only_runs_under_streaming(self):
        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "if streamer is not None else None" in src


class TestTuningGradBuffers:
    """Outside-block tuning must keep gradient buffers stable in memory."""

    def _params(self):
        w = torch.nn.Parameter(torch.randn(6, 5))
        s = torch.nn.Parameter(torch.randn(6, 1))
        return [w, s]

    def test_preallocate_creates_zero_grads(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        params = self._params()
        opt = torch.optim.SGD(params, lr=1e-3)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        for p in params:
            assert p.grad is not None
            assert p.grad.shape == p.shape
            assert torch.count_nonzero(p.grad) == 0

    def test_preallocate_idempotent_and_keeps_values(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        params = self._params()
        opt = torch.optim.SGD(params, lr=1e-3)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        params[0].grad.add_(1.0)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        assert torch.all(params[0].grad == 1.0), "pre-allocation must not clobber live gradients"

    def test_optimizer_step_zeroes_in_place(self):
        """The shared step helper must zero grads without dropping the buffers."""
        import inspect

        from auto_round.algorithms.quantization.sign_round import quantizer as qmod

        src = inspect.getsource(qmod.SignRoundQuantizer._step)
        assert "zero_grad(set_to_none=False)" in src

    def test_outside_block_loop_releases_cached_blocks(self):
        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "torch.cuda.empty_cache()" in src


class TestGradScatterSlice:
    """The row-window view must produce the exact gradients of a plain slice."""

    def test_blocked_gradients_match_plain_slice(self):
        from auto_round.wrapper import _GradScatterSlice

        torch.manual_seed(7)
        param = torch.nn.Parameter(torch.randn(12, 8))
        x = torch.randn(3, 8)
        ref = torch.randn(3, 6)

        plain = torch.nn.Parameter(param.detach().clone())
        plain_out = torch.nn.functional.linear(x, plain[6:12])
        torch.nn.functional.mse_loss(plain_out, ref, reduction="sum").backward()

        param.grad = torch.zeros_like(param)
        scatter_out = torch.nn.functional.linear(x, _GradScatterSlice.apply(param, 6, 12))
        torch.nn.functional.mse_loss(scatter_out, ref, reduction="sum").backward()

        assert param.grad[0:6].abs().sum() == 0, "outside rows must stay zero"
        assert torch.allclose(param.grad[6:12], plain.grad[6:12], atol=1e-6), "scatter must equal plain-slice grad"

    def test_accumulates_across_multiple_views(self):
        from auto_round.wrapper import _GradScatterSlice

        param = torch.nn.Parameter(torch.randn(8, 4))
        param.grad = torch.zeros_like(param)
        y = torch.randn(2, 4)
        for _ in range(3):
            out = torch.nn.functional.linear(y, _GradScatterSlice.apply(param, 2, 5))
            out.sum().backward()
        assert torch.allclose(param.grad[2:5], torch.ones(3, 4) * y.sum(0).unsqueeze(0) * 3, atol=1e-6)
        assert param.grad[0:2].abs().sum() == 0 and param.grad[5:].abs().sum() == 0


class TestSignSGDMemory:
    """The sign update must not allocate a gradient-sized temporary."""

    def _step_ref(self, param, grad_seq, **kw):
        import copy

        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        ref = torch.nn.Parameter(param.detach().clone())
        opt = SignSGD([ref], lr=kw["lr"])
        for g in grad_seq:
            ref.grad = g.clone()
            opt.step()
            opt.zero_grad(set_to_none=False)
        return ref.detach()

    def test_plain_update_matches_reference_and_leaves_grad_zeroed(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        torch.manual_seed(3)
        base = torch.randn(8, 5)
        param = torch.nn.Parameter(base.clone())
        opt = SignSGD([param], lr=0.1)
        grad = torch.randn(8, 5)
        ref = base - 0.1 * torch.sign(grad)
        param.grad = grad
        opt.step()
        assert torch.allclose(param.detach(), ref, atol=1e-6)
        opt.zero_grad(set_to_none=False)
        assert (
            param.grad is not None and torch.count_nonzero(param.grad) == 0
        ), "in-place sign must leave the grad buffer zeroable in place"

    def test_momentum_buffer_survives_in_place_sign(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        torch.manual_seed(4)
        base = torch.randn(6, 3)
        param = torch.nn.Parameter(base.clone())
        opt = SignSGD([param], lr=0.05, momentum=0.9)
        g1, g2 = torch.randn(6, 3), torch.randn(6, 3)
        param.grad = g1.clone()
        opt.step()
        opt.zero_grad(set_to_none=False)
        param.grad = g2.clone()
        opt.step()
        # reference: v2 = 0.9*sign-free momentum chain is irrelevant; SignSGD
        # applies sign(v) each step, so track v explicitly
        v1 = g1.clone()
        p1 = base - 0.05 * torch.sign(v1)
        v2 = 0.9 * v1 + g2
        p2 = p1 - 0.05 * torch.sign(v2)
        assert torch.allclose(param.detach(), p2, atol=1e-6), "momentum buffer must stay unsigned"


class TestRowBlockThreshold:
    """Blocking must engage only for genuinely huge weights."""

    def test_default_threshold_spares_ffn_sized_layers(self):
        """A 67M-element FFN projection (the old threshold) stays whole-layer:
        blocking it would split the compiled graph and add per-block overhead
        for no memory benefit (~2GiB of intermediates fits easily)."""
        from auto_round.wrapper import WrapperLinear

        layer = _mk_quant_linear(2048, 32768, group_size=128)  # 2**26 elements
        assert layer.weight.numel() == 2**26
        w = WrapperLinear(layer, enable_minmax_tuning=False, enable_torch_compile=False, device="cpu")
        assert w.row_block_active() is False

    def test_threshold_still_catches_vocabulary_heads(self, monkeypatch):
        import auto_round.wrapper as wmod
        from auto_round.wrapper import WrapperLinear

        layer = _mk_quant_linear(64, 512, group_size=128)
        w = WrapperLinear(layer, enable_minmax_tuning=False, enable_torch_compile=False, device="cpu")
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 64, raising=False)
        assert w.row_block_active() is True


class TestBestParamSnapshotDevice:
    """Huge-layer snapshots must park on the host, not beside the live params."""

    def test_device_selection(self):
        import torch

        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        q = SimpleNamespace(compress_context=SimpleNamespace(cache_device=torch.device("cuda:0")))
        fn = SignRoundQuantizer._best_param_device
        assert fn(q, 2**28) == torch.device("cpu"), "vocabulary-head snapshots must park on the host"
        assert fn(q, 1024) == torch.device("cuda:0"), "small layers keep the context cache device"


class TestUnwrapStreaming:
    """Huge-layer unwrap must tolerate host-resident best parameters."""

    def test_blocked_unwrap_accepts_cpu_best_params(self, monkeypatch):
        import auto_round.wrapper as wmod

        torch.manual_seed(13)
        layer = _mk_quant_linear(24, 10, group_size=4)
        wrapper = self._make(layer)
        with torch.no_grad():
            torch.manual_seed(14)
            for p in wrapper.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 40, raising=False)
        cpu_best = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}
        restored = wrapper.unwrapper(cpu_best)
        assert restored is layer or restored.weight.shape == (24, 10)

    @staticmethod
    def _make(layer):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(layer, enable_minmax_tuning=True, enable_torch_compile=False, device="cpu")
