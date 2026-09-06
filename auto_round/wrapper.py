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

import math
from math import ceil

import torch
import transformers
from torch.functional import F

import auto_round.envs as envs
from auto_round.compressors.utils import is_nv_fp
from auto_round.data_type import get_quant_func, reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    compile_func,
    deepspeed_exists,
    set_module,
)

if deepspeed_exists:
    from deepspeed import comm as dist
    from deepspeed.module_inject import LinearAllreduce, LinearLayer


# Weight-element budget for row-blocked fake-quant forwards: layers whose
# weight exceeds this many elements compute their quantized-weight math one
# output-row block at a time, because the eager quant functions materialize
# several full-size fp32 intermediates (observed ~4.7GiB each on a 248k-vocab
# lm_head) that cannot coexist with the rounding parameter and its gradient.
# Blocking is exact: quantization groups never straddle output rows, so the
# per-block results and gradients equal the full-tensor computation.
# Row blocking exists for weights whose full-width fp32 fake-quant intermediates
# (~6x the weight bytes) cannot share a 24GB GPU with the tuning residents:
# full-vocabulary lm_heads. Typical FFN projections stay on the fast whole-
# layer path (compiled graph, single GEMM); the threshold is far below their
# size on purpose.
_ROW_BLOCKED_WEIGHT_ELEMS = 2**27


class _GradScatterSlice(torch.autograd.Function):
    """Row-window view of a tuning parameter with block-sized gradient memory.

    Slicing a leaf parameter directly makes autograd materialize a dense
    gradient the size of the whole parameter (SliceBackward starts from a full
    zeros tensor), which is precisely the allocation that overflows a 24GB GPU
    for a 248k-vocabulary lm_head. This view accumulates its gradient straight
    into the parameter's ``.grad`` buffer instead, so every intermediate stays
    the size of one row window."""

    @staticmethod
    def forward(ctx, param, g_start, g_end):
        ctx.param = param
        ctx.g_start = g_start
        ctx.g_end = g_end
        return param.detach()[g_start:g_end].clone()

    @staticmethod
    def backward(ctx, grad_out):
        param = ctx.param
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        with torch.no_grad():
            param.grad[ctx.g_start : ctx.g_end] += grad_out.to(param.grad.dtype)
        return None, None, None


def get_scale_shape(weight, group_size):
    """Computes the shape of the scale tensor for quantization based on the weight tensor and group size.

    Args:
      weight (torch.Tensor): The weight tensor of the layer.
      group_size (int or tuple): The size of the groups for quantization.

    Returns:
      The shape of the scale tensor to be used for quantization.
    """
    if isinstance(group_size, tuple):
        assert len(weight.shape) == len(group_size), f"Expected group_size is {len(weight.shape)}D but get {group_size}"
        return (weight.shape[0] // group_size[0], weight.shape[1] // group_size[1])
    if group_size == 0:
        return 1
    elif group_size == -1 or weight.shape[1] < group_size:
        shape = weight.shape[0]
    else:
        shape = weight.shape[0] * ceil(weight.shape[1] / group_size)

    return shape


class WrapperLinear(torch.nn.Module):
    """A wrapper for linear/conv1d layers to enable quantization and tuning.

    This module wraps an existing linear or conv1d layer and provides additional functionality
    for quantization, parameter tuning, and activation/bias normalization.

    Args:
        orig_layer (torch.nn.Module): The original layer to be wrapped (linear or conv1d).
        enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
        enable_norm_bias_tuning (bool): Whether to enable normalization and tuning of the bias term.
        enable_torch_compile (bool): Whether to enable torch compilation.
        device (str): Device on which to run computations (e.g., 'cpu' or 'cuda').
    """

    minmax_scale_bound = (0.0, 1.0)

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        enable_torch_compile=True,
        disable_opt_rtn=True,
        **kwargs,
    ):
        """Initializes the WrapperLinear module.

        Args:
            orig_layer (torch.nn.Module): The original layer to wrap.
            enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
            enable_norm_bias_tuning (bool): Whether to enable normalization and tuning for the bias term.
            device (str): The computation device, such as 'cpu' or 'cuda'.
        """
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.orig_layer.iters = kwargs.pop("iters", 200)
        self.disable_opt_rtn = disable_opt_rtn
        self.output_device = device
        self.device = self.orig_layer.tuning_device if hasattr(self.orig_layer, "tuning_device") else device
        self.enable_minmax_tuning = enable_minmax_tuning
        self.enable_round_tuning = enable_round_tuning
        self.enable_torch_compile = enable_torch_compile
        self.enable_norm_bias_tuning = enable_norm_bias_tuning and (orig_layer.bias is not None)
        self.enable_act_quant = self.orig_layer.act_bits <= 8
        self.weight_global_scale = getattr(self.orig_layer, "weight_global_scale", None)
        if is_nv_fp(self.orig_layer.data_type) and self.weight_global_scale is None:
            from auto_round.data_type.nvfp import calculate_gparam

            weight_global_scale = calculate_gparam(self.orig_layer.weight, self.orig_layer.group_size)
            setattr(self, "weight_global_scale", weight_global_scale)
            self.weight_global_scale = self.weight_global_scale.to(self.orig_layer.weight.device)
        if hasattr(self.orig_layer, "scale_dtype") and self.orig_layer.scale_dtype == torch.float32:
            self.q_scale_thresh = 1e-8
        else:
            self.q_scale_thresh = 1e-5
        self._init_tuning_params_and_quant_func()
        self._row_block_decision = None
        if deepspeed_exists:
            if type(self.orig_layer) in (torch.nn.Linear, LinearLayer):
                self.orig_forward = self.linear_forward
            elif type(self.orig_layer) == LinearAllreduce:
                self.orig_forward = self.all_reduce_linear_forward
                self.mp_group = self.orig_layer.mp_group
            else:
                self.orig_forward = self.conv1d_forward
        else:
            self.orig_forward = self.linear_forward if type(self.orig_layer) == torch.nn.Linear else self.conv1d_forward

    @property
    def weight(self):
        return self.orig_layer.weight

    @property
    def bias(self):
        return self.orig_layer.bias

    def _init_tuning_params_and_quant_func(self):
        """Initializes tuning parameters and quantization functions.

        This method sets up required parameters and functions for weight quantization,
        activation quantization, and bias/normalization.
        """
        self.params = {}
        p_dtype = torch.float32  ##parameter dtype

        orig_layer = self.orig_layer
        orig_weight = getattr(orig_layer, "get_weight", lambda: orig_layer.weight)()
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            orig_weight = orig_weight.t()
        weight_reshape, _, _ = reshape_pad_tensor_by_group_size(orig_weight.data, orig_layer.group_size)

        if self.enable_round_tuning:
            self.weight_min = (
                torch.clamp(weight_reshape.amin(dim=(-2, -1)), max=0)
                if isinstance(orig_layer.group_size, tuple)
                else torch.clamp(weight_reshape.min(1)[0], max=0)
            )
            self.weight_max = (
                torch.clamp(weight_reshape.amax(dim=(-2, -1)), min=0)
                if isinstance(orig_layer.group_size, tuple)
                else torch.clamp(weight_reshape.max(1)[0], min=0)
            )
        else:
            self.weight_min = None
            self.weight_max = None
        # AWQ clip-as-init: cap the tunable weight range to the per-group clip
        # range searched by AWQ (``apply_clip`` with ``clip_as_init=True``).
        # This initializes the range used by quant_tensor_sym/asym, leaving
        # min_scale/max_scale to tune a coefficient on top. Only the standard
        # (non-tuple) group layout maps onto weight_min/weight_max here.
        awq_clip_min = getattr(orig_layer, "awq_clip_min", None)
        awq_clip_max = getattr(orig_layer, "awq_clip_max", None)
        if awq_clip_max is not None and self.weight_min is not None and not isinstance(orig_layer.group_size, tuple):
            clip_max_flat = awq_clip_max.reshape(-1).to(self.weight_max.device, self.weight_max.dtype)
            if awq_clip_min is None:
                clip_min_flat = -clip_max_flat
            else:
                clip_min_flat = awq_clip_min.reshape(-1).to(self.weight_min.device, self.weight_min.dtype)
            if clip_max_flat.numel() == self.weight_max.numel() and clip_min_flat.numel() == self.weight_min.numel():
                self.weight_max = torch.minimum(self.weight_max, clip_max_flat)
                self.weight_min = torch.maximum(self.weight_min, clip_min_flat)
        self._init_params(
            "value", p_dtype, weight_reshape.shape, 0, self.enable_round_tuning and self.orig_layer.bits < 16
        )
        # Min-max scale initialization
        shape = get_scale_shape(orig_weight, orig_layer.group_size)
        self._init_params("min_scale", p_dtype, shape, 1.0, (self.enable_minmax_tuning and self.orig_layer.bits < 16))
        self._init_params("max_scale", p_dtype, shape, 1.0, (self.enable_minmax_tuning and self.orig_layer.bits < 16))

        self.weight_quant_func, self.data_type = get_quant_func(
            orig_layer.data_type,
            orig_layer.bits,
            orig_layer.sym,
            self.disable_opt_rtn,
            orig_layer.group_size,
            iters=orig_layer.iters,
        )
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

        if self.enable_act_quant:
            self.act_quant_func, self.act_data_type = get_quant_func(
                orig_layer.act_data_type,
                orig_layer.act_bits,
                orig_layer.act_sym,
                disable_opt_rtn=True,
                iters=orig_layer.iters,
            )
            if self.enable_torch_compile:
                self.act_quant_func = compile_func(self.act_quant_func, self.device)
            self._init_params(
                "act_max_scale", p_dtype, (1), 1.0, envs.AR_ENABLE_ACT_MINMAX_TUNING or (not orig_layer.act_dynamic)
            )
            self._init_params("act_min_scale", p_dtype, (1), 1.0, envs.AR_ENABLE_ACT_MINMAX_TUNING)

        # Bias tuning
        if self.enable_norm_bias_tuning:
            self._init_params("bias_v", p_dtype, self.orig_layer.bias.shape, 0, True)
            from auto_round.data_type.int import quant_tensor_asym_wo_round

            self.bias_quant_func = quant_tensor_asym_wo_round
            self.params["bias_v"] = self.bias_v

    def _init_params(self, name, dtype, shape, value, tunable):
        """Initializes a parameter for tuning or uses a constant if tuning is disabled.

        Args:
            name (str): Name of the parameter.
            dtype (torch.dtype): Data type of the parameter.
            shape (tuple): Shape of the parameter.
            value (float): Initial value for the parameter.
            tunable (bool): Whether the parameter should be tunable.
        """
        if tunable:
            p = torch.nn.Parameter(torch.ones(shape, device=self.device, dtype=dtype) * value, requires_grad=True)
            self.params.update({name: p})
        else:
            p = torch.tensor(1.0 * value, device=self.device, dtype=dtype)

        setattr(self, name, p)

    def _qdq_weight_block(self, weight, value, min_scale, max_scale, tensor_min, tensor_max, init_scale=None):
        """Fake-quantize an explicit block of rows (see ``_qdq_weight``).

        Split out so the row-blocked forward can bound the fp32 intermediates
        of huge layers; the kwargs are identical to the full-tensor call.
        ``init_scale`` may be passed pre-sliced by the blocked caller."""
        quant_kwargs = {}
        if hasattr(self.orig_layer, "super_bits"):
            quant_kwargs["super_bits"] = self.orig_layer.super_bits
            quant_kwargs["super_group_size"] = self.orig_layer.super_group_size
        if hasattr(self, "_extra_quant_kwargs"):
            quant_kwargs.update(self._extra_quant_kwargs())
        weight_q, scale, zp = self.weight_quant_func(
            weight,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=tensor_min,
            tensor_max=tensor_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            imatrix=self.orig_layer.imatrix.to(weight.device) if hasattr(self.orig_layer, "imatrix") else None,
            global_scale=getattr(self, "weight_global_scale", None),
            init_scale=init_scale if init_scale is not None else getattr(self, "init_scale", None),
            **quant_kwargs,
        )
        weight_q = weight_q.to(weight.dtype)
        return weight_q, scale, zp

    def _slice_tunable(self, t, g_start, g_end):
        """Slice a tuning parameter by group window, keeping scalars intact."""
        if t is None or t.numel() == 1:
            return t
        return t[g_start:g_end]

    def _qdq_weight(self, value, min_scale, max_scale):
        """Quantizes and dequantizes weights with tuning parameters.

        Args:
            value (torch.Tensor): Value added for rounding for tuning.
            min_scale (torch.Tensor): Minimum scale for the min value of quantization.
            max_scale (torch.Tensor): Maximum scale for the max value of quantization.

        Returns:
            tuple: Quantized weight, scale, and zero point.
        """
        if self.orig_layer.bits >= 16:
            return self.orig_layer.weight, None, None
        min_bound, max_bound = self.minmax_scale_bound
        min_scale.data.clamp_(min_bound, max_bound)
        max_scale.data.clamp_(min_bound, max_bound)
        weight = self.orig_layer.weight
        if weight.device.type == "meta":
            weight = self.orig_layer.get_weight().to(self.device)
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            weight = weight.t()

        weight = weight.to(self.device)
        if weight.dim() == 2 and self.row_block_active():
            # the final quantize/dequantize of a huge layer runs the same
            # fp32 intermediates as the forward; keep them block-sized too
            out_features, in_features = weight.shape
            group_size = self.orig_layer.group_size
            groups_per_row = (in_features + group_size - 1) // group_size if 0 < group_size < in_features else 1
            weight_q_parts, scale_parts, zp_parts = [], [], []
            for b_start, b_end in self.row_block_bounds():
                g_start, g_end = b_start * groups_per_row, b_end * groups_per_row
                wq, sc, zp = self._qdq_weight_block(
                    weight[b_start:b_end],
                    self._slice_tunable(value, g_start, g_end),
                    self._slice_tunable(min_scale, g_start, g_end),
                    self._slice_tunable(max_scale, g_start, g_end),
                    self._slice_tunable(self.weight_min, g_start, g_end),
                    self._slice_tunable(self.weight_max, g_start, g_end),
                    init_scale=self._sliced_init_scale(g_start, g_end, out_features * groups_per_row),
                )
                weight_q_parts.append(wq)
                scale_parts.append(sc)
                zp_parts.append(zp)
                del wq, sc, zp
            weight_q = torch.cat(weight_q_parts, dim=0)
            scale = torch.cat(scale_parts, dim=0) if isinstance(scale_parts[0], torch.Tensor) else scale_parts[0]
            zp = torch.cat(zp_parts, dim=0) if isinstance(zp_parts[0], torch.Tensor) else zp_parts[0]
        else:
            weight_q, scale, zp = self._qdq_weight_block(
                weight,
                value,
                min_scale,
                max_scale,
                self.weight_min,
                self.weight_max,
            )
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            weight_q = weight_q.t()
        return weight_q, scale, zp

    def row_block_active(self):
        """Compute (once) and return the row-blocked-forward decision.

        Tune loops must ask this BEFORE the first forward so their loss loop
        picks the per-block backward path instead of a full forward whose
        graphs for every block stay alive at once."""
        return self._use_row_blocked_output()

    def _use_row_blocked_output(self):
        """Whether the forward should compute the quantized weight one output-row
        block at a time; decided once per wrapper."""
        if self._row_block_decision is None:
            self._row_block_decision = self._compute_row_blocked_eligibility()
            if self._row_block_decision:
                logger.debug(
                    "[tune] row-blocked weight forward for %s (%d elements)",
                    getattr(self.orig_layer, "global_name", "layer"),
                    self.orig_layer.weight.numel(),
                )
        return self._row_block_decision

    def _compute_row_blocked_eligibility(self):
        """Eligibility for row-blocked fake-quant forwards. Only plain Linear
        layers with a weight beyond the element budget qualify: meta weights
        are materialized whole by ``get_weight`` anyway, non-linear forwards
        take un-sliced weights, k-quant super groups and per-tensor group
        layouts (group_size 0) share scale across rows so blocking would
        change the math."""
        layer = self.orig_layer
        if type(layer) is not torch.nn.Linear or self.orig_forward != self.linear_forward:
            return False
        weight = layer.weight
        if weight.dim() != 2 or weight.device.type == "meta":
            return False
        # scheme fields are attached to every quantized layer (None when unset),
        # so test the value, not the attribute's presence
        if getattr(layer, "super_bits", None) is not None:
            return False
        group_size = getattr(layer, "group_size", -1)
        if not isinstance(group_size, int) or group_size == 0:
            return False
        return weight.numel() > _ROW_BLOCKED_WEIGHT_ELEMS

    def row_block_bounds(self):
        """Output-row windows used by the row-blocked paths.

        The window shrinks below the element budget when the GPU is nearly
        full, so a block's fp32 intermediates always fit the free pool (about
        six block-sized fp32 arrays are live in the quantize/backward math).
        Boundaries always land on whole rows, so any window size reproduces
        the full-tensor computation exactly."""
        weight = self.orig_layer.weight
        out_features, in_features = weight.shape
        rows = max(1, _ROW_BLOCKED_WEIGHT_ELEMS // max(1, in_features))
        if weight.is_cuda:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(weight.device)
            except (RuntimeError, ValueError):  # pragma: no cover - exotic devices
                free_bytes = None
            if free_bytes is not None:
                # ~6 fp32 arrays of rows x in_features live per block; use half
                # the free pool to leave room for transients and fragmentation
                budget_rows = int(free_bytes * 0.5 // (in_features * 4 * 6))
                rows = max(1024, min(rows, budget_rows))
        return [(start, min(start + rows, out_features)) for start in range(0, out_features, rows)]

    def _sliced_init_scale(self, g_start, g_end, total_groups=None):
        """init_scale sliced to a group window, with full-width fallback."""
        init_scale = getattr(self, "init_scale", None)
        if not isinstance(init_scale, torch.Tensor) or init_scale.dim() < 1:
            return init_scale
        expected = total_groups
        if expected is None:
            weight = self.orig_layer.weight
            _, in_features = weight.shape
            group_size = self.orig_layer.group_size
            groups_per_row = (in_features + group_size - 1) // group_size if 0 < group_size < in_features else 1
            expected = weight.shape[0] * groups_per_row
        if init_scale.shape[0] == expected:
            # per-group initial scales follow the same row-major window
            return init_scale[g_start:g_end]
        logger.warning_once(
            "[tune] row-blocked forward keeps an init_scale of shape %s (expected leading dim %d); "
            "passing it through unsliced",
            tuple(init_scale.shape),
            expected,
        )
        return init_scale

    def _row_block_params(self, start, end):
        """Sliced tuning parameters and weight for one output-row window."""
        weight = self.orig_layer.weight.to(self.device)
        _, in_features = weight.shape
        group_size = self.orig_layer.group_size
        groups_per_row = (in_features + group_size - 1) // group_size if 0 < group_size < in_features else 1
        g_start, g_end = start * groups_per_row, end * groups_per_row
        block_init_scale = self._sliced_init_scale(g_start, g_end, weight.shape[0] * groups_per_row)
        min_bound, max_bound = self.minmax_scale_bound
        self.min_scale.data.clamp_(min_bound, max_bound)
        self.max_scale.data.clamp_(min_bound, max_bound)
        weight_q, *_ = self._qdq_weight_block(
            weight[start:end],
            _GradScatterSlice.apply(self.value, g_start, g_end),
            _GradScatterSlice.apply(self.min_scale, g_start, g_end),
            _GradScatterSlice.apply(self.max_scale, g_start, g_end),
            self.weight_min[g_start:g_end] if self.weight_min is not None else None,
            self.weight_max[g_start:g_end] if self.weight_max is not None else None,
            init_scale=block_init_scale,
        )
        return weight_q

    def forward_rows(self, x, start, end, bias=None):
        """Output columns ``[start:end)`` computed with the row-blocked fake-quant
        math, keeping only this block's autograd graph alive. Tuning loops use
        this to backward per block; the summed loss equals the full-tensor
        loss because the MSE decomposes over output columns."""
        block_bias = bias[start:end] if bias is not None else None
        return self.linear_forward(x, self._row_block_params(start, end), block_bias)

    def _row_blocked_output(self, x, bias):
        """Forward with per-row-block fake-quant math (see ``_qdq_weight_block``).

        Groups are laid out row-major in the flattened tuning parameters, so a
        window of output rows maps to a consecutive window of groups; blocking
        therefore reproduces the full-tensor computation exactly while the
        fp32 intermediates stay bounded."""
        outputs = [self.forward_rows(x, start, end, bias) for start, end in self.row_block_bounds()]
        return torch.cat(outputs, dim=-1)

    def _qdq_act(self, x, act_min_scale=torch.tensor(1.0), act_max_scale=torch.tensor(1.0), act_max=None):
        """Quantizes and dequantizes activations.

        Args:
            x (torch.Tensor): Input activations.
            act_max_scale (torch.Tensor): Maximum scale for the act_max
            act_max (torch.Tensor, optional): Maximum value for activation quantization. Defaults to None.

        Returns:
            tuple: Quantized activation, scale, and zero point.
        """
        act_max_scale.data.clamp_(0, 1.0)
        act_min_scale.data.clamp_(0, 1.0)
        env_act_scale = envs.AR_ACT_SCALE  # fixed activation ratio,prioritize to use this one if set
        x, scale, zp = self.act_quant_func(
            x,
            bits=self.orig_layer.act_bits,
            group_size=self.orig_layer.act_group_size,
            scale_dtype=self.orig_layer.scale_dtype,
            q_scale_thresh=self.q_scale_thresh,
            data_type=self.act_data_type,
            tensor_max=act_max,  # for static
            max_scale=act_max_scale if math.isclose(env_act_scale, 1.0, rel_tol=1e-6) else env_act_scale,
            min_scale=act_min_scale if math.isclose(env_act_scale, 1.0, rel_tol=1e-6) else env_act_scale,
            global_scale=getattr(self, "input_global_scale", None),
        )
        return x, scale, zp

    def _qdq_bias(self, bias, bias_v):
        """Quantizes and dequantizes bias.

        Args:
            bias (torch.Tensor): Bias tensor to be quantized.
            bias_v (torch.Tensor): Value added for rounding for tuning.

        Returns:
            tuple: Quantized bias, scale, and zero point.
        """
        bias_bits = 4  ## hard code
        bias_group_size = -1
        bias, scale, zp = self.bias_quant_func(
            bias,
            bits=bias_bits,
            group_size=bias_group_size,
            v=bias_v,
            q_scale_thresh=self.q_scale_thresh,
            global_scale=getattr(self, "weight_global_scale", None),
        )
        return bias, scale, zp

    def unwrapper(self, best_params):
        """Restores the original layer by applying the best tuning parameters.

        Args:
            best_params (dict): Dictionary containing the best tuning parameters.

        Returns:
            torch.nn.Module: The unwrapped and restored original layer.
        """

        def _preserve_global_name(layer):
            if hasattr(self.orig_layer, "global_name") and not hasattr(layer, "global_name"):
                layer.global_name = self.orig_layer.global_name
            return layer

        best_params = best_params or {}
        v = best_params.get("value", torch.tensor(0.0)).to(self.device)
        min_scale = best_params.get("min_scale", torch.tensor(1.0)).to(self.device)
        max_scale = best_params.get("max_scale", torch.tensor(1.0)).to(self.device)

        if self.orig_layer.weight.device.type == "meta":
            self.orig_layer.to(self.device)
        # Unwrapper weight
        qdq_weight, scale, zp = self._qdq_weight(v, min_scale, max_scale)
        # if hasattr(self.orig_layer, "imatrix"):
        #     self.orig_layer.imatrix = None
        self.orig_layer.weight.data.copy_(qdq_weight)
        self.orig_layer.weight.grad = None

        shape = qdq_weight.shape
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            shape = qdq_weight.t().shape

        def _set_dict_attr(attr_dict, attr_name):
            for key in attr_dict.keys():
                if key == attr_name:
                    setattr(self.orig_layer, attr_name, attr_dict[key].reshape(shape[0], -1).to("cpu"))
                else:
                    name = "w_" + key
                    setattr(self.orig_layer, name, attr_dict[key].to("cpu"))

        if not isinstance(self.orig_layer.group_size, tuple):
            if isinstance(scale, dict):
                _set_dict_attr(scale, "scale")
            elif scale is None:
                self.orig_layer.scale = None
            elif scale.numel() > 1:
                self.orig_layer.scale = scale.reshape(shape[0], -1).to("cpu")
            else:
                self.orig_layer.scale = scale.view(-1).to("cpu")
        else:
            self.orig_layer.scale = scale.to("cpu")

        if zp is not None:
            if isinstance(zp, dict):
                _set_dict_attr(zp, "zp")
            elif isinstance(zp, torch.Tensor):
                if zp.numel() > 1:
                    zp = zp.reshape(shape[0], -1)
                    self.orig_layer.zp = zp.to("cpu")
                else:
                    self.orig_layer.zp = zp.view(-1).to("cpu")
            else:
                self.orig_layer.zp = zp
        else:
            self.orig_layer.zp = None

        if self.weight_global_scale is not None:
            global_scale = self.weight_global_scale
            assert global_scale.numel() == 1
            self.orig_layer.weight_global_scale = global_scale.to("cpu")

        # Unwrapper bias
        if self.enable_norm_bias_tuning and "bias_v" in best_params.keys():  ##fake quant
            bias_v = best_params["bias_v"].to(self.device)
            bias = self.orig_layer.bias
            if bias is not None and bias.device.type == "meta":
                bias = self.orig_layer.get_bias().to(self.device)
            bias, _, _ = self._qdq_bias(bias, bias_v)
            self.orig_layer.bias.grad = None
            self.orig_layer.bias.data.copy_(bias)

        if hasattr(self.orig_layer, "update"):
            self.orig_layer.update()
            self.orig_layer.to("meta")

        # Unwrapper act
        if self.enable_act_quant:
            if not self.orig_layer.act_dynamic:
                act_max_scale = best_params.get("act_max_scale", torch.tensor(1.0)).to(self.device)
                act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
                if act_max is not None:
                    tmp_shape = 1
                    if self.orig_layer.act_group_size > 1:
                        tmp_shape = (act_max.shape[0], self.orig_layer.act_group_size)
                    elif self.orig_layer.act_group_size == -1:
                        tmp_shape = (act_max.shape[0], 1)
                    _, act_scale, _ = self._qdq_act(
                        torch.zeros(tmp_shape).to(self.device),
                        act_min_scale=self.act_min_scale,
                        act_max_scale=self.act_max_scale,
                        act_max=act_max,
                    )
                    self.orig_layer.act_max = self.orig_layer.act_max * act_max_scale.item()
                    self.orig_layer.act_max = self.orig_layer.act_max.to("cpu")
                else:
                    act_scale = torch.ones(1, dtype=self.orig_layer.scale_dtype)
                self.orig_layer.act_scale = act_scale.to("cpu")

            self.orig_layer.q_scale_thresh = self.q_scale_thresh
            self.orig_layer.data_type = self.data_type
            self.orig_layer.act_min_scale = self.act_min_scale
            self.orig_layer.act_max_scale = self.act_max_scale

            self.orig_layer.act_data_type = self.act_data_type
            self.orig_layer.act_quant_func = self.act_quant_func
            wrapper_layer = WrapperWALayer(
                self.orig_layer,
                enable_torch_compile=self.enable_torch_compile,
                device=self.device,
            )
            return _preserve_global_name(wrapper_layer)

        return _preserve_global_name(self.orig_layer)

    def linear_forward(self, x, weight, bias):
        """Performs the forward pass for a linear layer.

        Args:
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor for the linear layer.
            bias (torch.Tensor): Bias tensor for the linear layer.

        Returns:
            torch.Tensor: Output tensor after applying the linear layer.
        """
        return F.linear(x, weight, bias)  # pylint: disable=E1102

    def all_reduce_linear_forward(self, x, weight, bias):
        """Performs the forward pass for a linear layer.

        Args:
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor for the linear layer.
            bias (torch.Tensor): Bias tensor for the linear layer.

        Returns:
            torch.Tensor: Output tensor after applying the linear layer.
        """
        output = torch.matmul(x, weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if bias is not None:
            output += bias
        return output

    def conv1d_forward(self, x, weight, bias):
        """Performs the forward pass for a Conv1D layer.

        Args:
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor for the Conv1D layer.
            bias (torch.Tensor): Bias tensor for the Conv1D layer.

        Returns:
            torch.Tensor: Output tensor after applying the Conv1D layer.
        """
        size_out = x.size()[:-1] + (self.orig_layer.nf,)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
        x = x.view(*size_out)
        return x

    def forward(self, x):
        """Executes the forward pass with quantized weights and optional bias/activation quantization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the wrapped layer.
        """
        # logger.info(self.orig_layer.global_name)
        x = x.to(self.device)
        row_blocked = self._use_row_blocked_output()
        weight_q = None
        if not row_blocked:
            weight_q, *_ = self._qdq_weight(self.value, self.min_scale, self.max_scale)

        if self.enable_act_quant:
            # Run orig_layer's forward_pre_hooks (e.g., online Hadamard transform)
            # BEFORE activation quantization, to match inference behavior in WrapperWALayer.
            for hook in self.orig_layer._forward_pre_hooks.values():
                result = hook(self.orig_layer, (x,))
                if result is not None:
                    x = result[0] if isinstance(result, tuple) else result
            act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None
            x, _, _ = self._qdq_act(
                x, act_max_scale=self.act_max_scale, act_min_scale=self.act_min_scale, act_max=act_max
            )
        elif len(self.orig_layer._forward_pre_hooks) > 0:
            # Even without act_quant, run pre-hooks for online Hadamard correctness
            # (when fuse_online_to_weight=False, pre-hooks transform activations).
            for hook in self.orig_layer._forward_pre_hooks.values():
                result = hook(self.orig_layer, (x,))
                if result is not None:
                    x = result[0] if isinstance(result, tuple) else result

        # pylint: disable=not-callable
        bias = self.orig_layer.bias
        if bias is not None and bias.device.type == "meta":
            bias = self.orig_layer.get_bias().to(self.device)
        if self.enable_norm_bias_tuning:
            bias, _, _ = self._qdq_bias(bias, self.bias_v)

        if row_blocked:
            output = self._row_blocked_output(x, bias)
        else:
            output = self.orig_forward(x, weight_q, bias)
        output = output.to(self.output_device)

        # Execute post-hooks from orig_layer (e.g., v_proj per-head Hadamard
        # when online rotation is not fused into weights).
        for hook in self.orig_layer._forward_hooks.values():
            hook_result = hook(self.orig_layer, (x,), output)
            if hook_result is not None:
                output = hook_result

        return output


class WrapperWALayer(torch.nn.Module):

    def __init__(self, orig_layer, enable_torch_compile=True, device="cpu"):
        super(WrapperWALayer, self).__init__()
        self.orig_layer = orig_layer
        self.enable_torch_compile = enable_torch_compile
        self.device = device
        self.data_type = orig_layer.data_type if hasattr(orig_layer, "data_type") else None
        self.act_data_type = orig_layer.act_data_type if hasattr(orig_layer, "act_data_type") else None
        self.act_quant_func = self.orig_layer.act_quant_func
        if self.enable_torch_compile:
            self.act_quant_func = compile_func(self.act_quant_func, self.device)
        self.extra_repr_org = orig_layer.extra_repr

        # Steal forward_pre_hooks from orig_layer (e.g., Hadamard transform hooks)
        # and remove them from orig_layer so they won't fire again inside orig_layer.forward().
        # We will run them explicitly in our forward() BEFORE activation quantization.
        self._stolen_pre_hooks = list(orig_layer._forward_pre_hooks.values())
        orig_layer._forward_pre_hooks.clear()

    @property
    def weight(self):
        """Exposes the weight of the wrapped layer for external access."""
        return self.orig_layer.weight

    @property
    def bias(self):
        """Exposes the bias of the wrapped layer for external access."""
        return self.orig_layer.bias

    def forward(self, x):
        # 1) Run stolen pre_hooks first (e.g., online Hadamard) → smooths activation
        for hook in self._stolen_pre_hooks:
            result = hook(self.orig_layer, (x,))
            if result is not None:
                x = result[0] if isinstance(result, tuple) else result

        # 2) Activation quantization on the smoothed activation
        import auto_round.envs as envs

        act_scale = envs.AR_ACT_SCALE
        act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None

        max_scale = self.orig_layer.act_max_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale
        min_scale = self.orig_layer.act_min_scale if math.isclose(act_scale, 1.0, rel_tol=1e-6) else act_scale
        if act_max is None:
            x, _, _ = self.orig_layer.act_quant_func(
                x,
                bits=self.orig_layer.act_bits,
                group_size=self.orig_layer.act_group_size,
                scale_dtype=self.orig_layer.scale_dtype,
                q_scale_thresh=self.orig_layer.q_scale_thresh,
                data_type=self.orig_layer.act_data_type,
                min_scale=min_scale,
                max_scale=max_scale,
            )
        else:
            x, _, _ = self.orig_layer.act_quant_func(
                x,
                bits=self.orig_layer.act_bits,
                group_size=self.orig_layer.act_group_size,
                scale_dtype=self.orig_layer.scale_dtype,
                q_scale_thresh=self.orig_layer.q_scale_thresh,
                data_type=self.orig_layer.act_data_type,
                act_max=act_max,
            )
        # 3) Linear computation via orig_layer (pre_hooks already removed, no double execution)
        return self.orig_layer.forward(x)

    def extra_repr(self):
        return f"{self.extra_repr_org()}, weight_type={self.data_type}, act_data_type={self.act_data_type}"


class WrapperLayerNorm(torch.nn.Module):
    """A wrapper for layer normalization with quantized weights.

    This class wraps a given layer normalization module and applies quantization without round
    to its weights. The quantization is parameterized by the number of bits and
    an optional group size.
    """

    def __init__(self, orig_layer, bit=4, group_size=-1, device="cpu"):
        super(WrapperLayerNorm, self).__init__()
        self.orig_layer = orig_layer
        self.bits = bit
        self.group_size = group_size
        self.device = self.orig_layer.tuning_device if hasattr(self.orig_layer, "tuning_device") else device
        self.output_device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_pad_tensor_by_group_size(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype), self.group_size
            )[0],
            requires_grad=True,
        )
        self.params = {"v": self.v}
        from auto_round.data_type.int import quant_tensor_asym_wo_round

        self.quant_func = quant_tensor_asym_wo_round

    def unwrapper(self, best_params):
        if best_params is None:
            return self.orig_layer
        v = best_params["v"].to(self.device)
        weight_q, _, _ = self.quant_func(
            self.orig_layer.weight, self.bits, self.group_size, v, q_scale_thresh=self.q_scale_thresh
        )
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, input):
        input = input.to(self.device)
        weight_q, _, _ = self.quant_func(
            self.orig_layer.weight, self.bits, self.group_size, self.v, q_scale_thresh=self.q_scale_thresh
        )
        import torch.nn.functional as F

        return F.layer_norm(
            input, self.orig_layer.normalized_shape, weight_q, self.orig_layer.bias, self.orig_layer.eps
        ).to(self.output_device)


class WrapperLlamaNorm(torch.nn.Module):
    """A wrapper for Llama normalization in HF with fake quantized weights without rounding.

    This class wraps a given layer normalization module and applies quantization without rounding
    to its weights. The quantization is parameterized by the number of bits and
    an optional group size.
    """

    def __init__(self, orig_layer, bit=4, group_size=-1, device="cpu"):
        super(WrapperLlamaNorm, self).__init__()
        self.orig_layer = orig_layer
        self.bits = bit
        self.group_size = group_size
        self.device = self.orig_layer.tuning_device if hasattr(self.orig_layer, "tuning_device") else device
        self.output_device = device
        weight_dtype = torch.float32
        self.q_scale_thresh = 1e-5
        self.v = torch.nn.Parameter(
            reshape_pad_tensor_by_group_size(
                torch.zeros(self.orig_layer.weight.shape, device=self.device, dtype=weight_dtype), self.group_size
            )[0],
            requires_grad=True,
        )
        self.params = {"v": self.v}
        from auto_round.data_type.int import quant_tensor_asym_wo_round

        self.quant_func = quant_tensor_asym_wo_round

    def unwrapper(self, best_params):
        if best_params is None:
            return self.orig_layer
        v = best_params["v"].to(self.device)
        weight_q, _, _ = self.quant_func(
            self.orig_layer.weight, self.bits, self.group_size, v, q_scale_thresh=self.q_scale_thresh
        )
        self.orig_layer.q_scale_thresh = self.q_scale_thresh
        self.orig_layer.weight.data.copy_(weight_q)
        return self.orig_layer

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.device)
        weight_q, _, _ = self.quant_func(
            self.orig_layer.weight, self.bits, self.group_size, self.v, q_scale_thresh=self.q_scale_thresh
        )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.orig_layer.variance_epsilon)
        return (weight_q * hidden_states.to(input_dtype)).to(self.output_device)


NORM_MAPPING = {}
NORM_MAPPING["LayerNorm"] = WrapperLayerNorm
NORM_MAPPING["LlamaRMSNorm"] = WrapperLlamaNorm
NORM_MAPPING["Qwen2RMSNorm"] = WrapperLlamaNorm
NORM_MAPPING["Phi3RMSNorm"] = WrapperLlamaNorm
NORM_MAPPING["MistralRMSNorm"] = WrapperLlamaNorm
NORM_MAPPING["Qwen3RMSNorm"] = WrapperLlamaNorm
NORM_MAPPING["Qwen3_5MoeRMSNorm"] = WrapperLlamaNorm


class WrapperMultiblock(torch.nn.Module):
    """A wrapper for a list of modules to be act as a single block.

    Args:
    module_list: The list of modules to wrap.
    """

    def __init__(self, module_list):
        super(WrapperMultiblock, self).__init__()
        self.layers = torch.nn.ModuleList(module_list)

    def forward(self, x, *args, **kwargs):
        hidden_states = x
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(hidden_states, *args, **kwargs)
            hidden_states = layer_outputs
            if isinstance(hidden_states, tuple) or isinstance(hidden_states, list):
                hidden_states = layer_outputs[0]
        return hidden_states


def wrapper_block(
    block,
    enable_minmax_tuning,
    enable_norm_bias_tuning,
    enable_torch_compile=True,
    device="cpu",
    wrapper_cls=WrapperLinear,
    **kwargs,
):
    """Wraps the layers in the given block with a custom Wrapper module.

    Args:
        block: The input block containing linear and conv1d layers to be wrapped.
        enable_minmax_tuning: A boolean indicating whether min-max tuning is enabled.
        enable_norm_bias_tuning: A boolean indicating whether normalization and bias tuning is enabled.
        enable_torch_compile: A boolean indicating whether to enable torch compilation.
        device: The device to which the wrapped layers should be moved.

    Returns:
        list: A list of names of the wrapped layers and unwrapped layers.
    """
    quantized_layers = []
    unquantized_layers = []
    for n, m in block.named_modules():
        if type(m) in SUPPORTED_LAYER_TYPES:
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = wrapper_cls(
                m,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=enable_norm_bias_tuning,
                enable_torch_compile=enable_torch_compile,
                device=device,
                **kwargs,
            )
            set_module(block, n, new_m)
            quantized_layers.append(n)

        elif enable_norm_bias_tuning:
            if "norm" in m.__class__.__name__.lower():
                if m.__class__.__name__ in NORM_MAPPING.keys():
                    wrapper_layer_class = NORM_MAPPING[m.__class__.__name__]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                elif "RMSNorm" in m.__class__.__name__:
                    logger.warning_once(
                        f"use LlamaRMSNorm to wrap {m.__class__.__name__}, please check the correctness yourself"
                    )
                    wrapper_layer_class = NORM_MAPPING["LlamaRMSNorm"]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                else:
                    logger.warning_once(f"{m.__class__.__name__} is not supported")
    return quantized_layers, unquantized_layers


@torch.no_grad()
def unwrapper_layer(model, layer, layer_name, best_params):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """

    if hasattr(layer, "orig_layer"):
        orig_layer = layer.unwrapper(best_params)
        act_max = getattr(orig_layer, "act_max", None)
        act_scale = getattr(orig_layer, "act_scale", None)
        if (
            "lm_head" in layer_name
            and getattr(layer, "enable_act_quant", False)
            and not getattr(orig_layer, "act_dynamic", True)
            and act_scale is not None
            and act_max is None
        ):
            logger.warning_once(
                "Static activation quantization for lm_head is not fully supported yet. "
                "lm_head activation statistics are missing, so activation scale falls back to unit scale."
            )
        orig_layer = orig_layer.to("cpu")
        set_module(model, layer_name, orig_layer)


@torch.no_grad()
def unwrapper_block(block, best_params):
    """Unwraps the WrapperLinear and WrapperTransformerConv1d modules in the given block.

    Args:
    block: The input block containing wrapped modules to be unwrapped.
    vs: A dictionary of scaling parameters for the wrapped modules.
    min_scales: A dictionary of minimum scaling values for the wrapped modules.
    max_scales: A dictionary of maximum scaling values for the wrapped modules.
    """
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            if n in best_params.keys():
                best_param = best_params[n]
            else:
                best_param = None
            orig_layer = m.unwrapper(best_param)
            set_module(block, n, orig_layer)
