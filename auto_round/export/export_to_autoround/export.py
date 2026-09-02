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

import copy
import functools
import inspect
import json
import os
from dataclasses import fields
from enum import Enum
from types import SimpleNamespace
from typing import Callable, Union

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

from auto_round.compressors.utils import is_mx_fp, is_nv_fp, is_standard_fp
from auto_round.export.export_to_autoround.utils import check_neq_config
from auto_round.export.formats import BackendDataType
from auto_round.export.utils import (
    filter_quantization_config,
    get_autogptq_packing_qlinear,
    is_immediate_saving_mode,
    release_layer_safely,
    resolve_pipeline_export_layout,
    save_model,
    save_pretrained_artifact,
)
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
    check_start_with_block_name,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    set_module,
    to_standard_regex,
    unsupported_meta_device,
)


def dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits=16):
    """
    Dynamically imports and returns the appropriate QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "auto_round" "awq" and "gptq".
        bits (int): The number of bits for quantization.
        group_size (int): The group size for quantization.
        sym (bool): Flag indicating whether to use symmetric quantization.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.

    Raises:
        ValueError: If the backend is not supported.
    """
    if "auto_round" in backend and "awq" not in backend and "gptq" not in backend:
        if act_bits <= 8:  ##easily have bug for other configuration, need to refine code later
            import auto_round.export.export_to_autoround.qlinear_triton_act

            return auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return QuantLinear
    elif "gptqmodel" in backend:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return functools.partial(QuantLinear, g_idx=True)
    elif "auto_round" in backend and "gptq" in backend and "gptqmodel" not in backend:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        return QuantLinear
    elif "awq" in backend:
        from ..export_to_awq.utils import WQLinear_GEMM

        return WQLinear_GEMM
    elif "gptq" in backend and "gptqmodel" not in backend:  ## have g_idx
        return get_autogptq_packing_qlinear(backend, bits, group_size, sym)
    else:
        raise ValueError(f"unsupported backend: '{backend}'. Supported backends are: {', '.join(SUPPORTED_FORMATS)}")


def pack_qact_layer(name, model):
    layer = get_module(model, name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if layer.bits > 8:
        return

    device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    act_bits = layer.act_bits

    act_scale = layer.act_scale if hasattr(layer, "act_scale") else None
    w_bf16_to_fp8_scale = layer.w_bf16_to_fp8_scale if hasattr(layer, "w_bf16_to_fp8_scale") else None
    scale = layer.scale
    zp = layer.zp
    import auto_round.export.export_to_autoround.qlinear_triton_act

    QuantLinear = auto_round.export.export_to_autoround.qlinear_triton_act.QuantLinear

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias is not None
    use_pc = False
    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits, group_size, in_features, out_features, bias, weight_dtype=layer.weight.dtype, use_pc=use_pc
    )
    new_layer.device = device
    set_module(model, name, new_layer)
    qlayer = new_layer

    qlayer.to("cpu")

    qlayer.pack(layer, scale, zp, act_scale, w_bf16_to_fp8_scale, device)
    qlayer.to(device)


def pack_layer(layer_name, model, backend, device=None):
    """
    Packs a model layer for quantization based on its type and configuration.

    This function retrieves the specified layer from the model, checks its
    compatibility for quantization, and replaces it with a quantized version
    if applicable. The quantization process depends on the layer's bit-width,
    group size, symmetry, and activation bits.

    Args:
        layer_name (str): The name of the layer to be packed.
        model (torch.nn.Module): The model containing the layer.
        backend (str): The backend framework to be used for quantization.

    Returns:
        None: The function modifies the model in place.
    """
    layer = get_module(model, layer_name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:  ##already packed
        return

    # A resumed disk-streamed run only
    # materializes/quantizes the blocks it didn't already finish in a prior
    # (crashed) process. Blocks it skipped are never touched in *this*
    # process and stay on the meta device, while their packed weights
    # already live in shard files the previous process flushed to disk (see
    # ShardWriter._discover_existing_shards). There is nothing to pack here
    # -- attempting to would fail (no real weight data to read `.scale`
    # from) and would be redundant even if it didn't, since the on-disk
    # export for this layer is already complete.
    if layer.weight.device.type == "meta":
        return

    if int(layer.act_bits) <= 8:
        return pack_qact_layer(layer_name, model)

    if not check_to_quantized(layer):
        return

    orig_device = layer.weight.device
    bits = layer.bits
    group_size = layer.group_size
    sym = layer.sym
    act_bits = layer.act_bits

    scale = layer.scale
    zp = layer.zp
    QuantLinear = dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, act_bits)

    if type(layer) == nn.Linear:
        in_features = layer.in_features
        out_features = layer.out_features
    elif type(layer) == nn.Conv2d:
        in_features = layer.in_channels
        out_features = layer.out_channels
    elif type(layer) == transformers.pytorch_utils.Conv1D:
        in_features = layer.weight.shape[0]
        out_features = layer.weight.shape[1]
    bias = layer.bias is not None

    new_layer = QuantLinear(  ##pylint: disable=E1123
        bits, group_size, in_features, out_features, bias=bias, weight_dtype=layer.weight.dtype
    )
    new_layer.device = orig_device
    set_module(model, layer_name, new_layer)
    qlayer = new_layer
    import auto_round_extension.torch.qlinear_torch

    if (
        sym
        and isinstance(zp, torch.Tensor)
        and isinstance(QuantLinear, (auto_round_extension.torch.qlinear_torch.QuantLinear))
    ):
        zp = int(zp.flatten()[0])

    qlayer.to("cpu")
    # Force to float32 to be compatible with torch 2.0
    sig = inspect.signature(qlayer.pack)
    param_count = len(sig.parameters)
    if param_count == 2:
        qlayer.pack(layer, scale, device=device)
    else:
        qlayer.pack(layer, scale, zp, None, device=device)
    qlayer.to(orig_device)

    # Inject rotation buffers right after packing so that
    # ShardWriter.save_module() captures them before offloading to meta.
    if hasattr(model, "_rotation_config"):
        from auto_round.algorithms.transforms import inject_rotation_buffers_on_layer

        inject_rotation_buffers_on_layer(layer_name, qlayer, model)

    # Note: release weight and bias explicitly, in case they are referenced elsewhere
    release_layer_safely(layer)


def pack_layers_batched(names, model, backend, device=None, max_batch_bytes=256 * 1024 * 1024):
    """Pack same-shape Linear layers (MoE experts) with one pack pass per group.

    Per-module ``pack_layer`` spends its wall time on Python overhead, tensor
    allocations, and host<->device round trips (~50 ms/module); a large MoE
    block pays that hundreds of times (once per expert projection). This
    batches every group of modules
    sharing (QuantLinear class, bits, group_size, sym, act_bits, shape, bias)
    into one ``QuantLinear`` whose out_features is the stacked total, packs
    the row-stacked weight once, then column-slices the packed artifacts back
    into per-module ``QuantLinear`` shells -- bitwise-identical results
    (qweight/qzeros/scales are all column-aligned by out_features).

    Modules that cannot be batched (non-Linear, meta/packed, act_bits <= 8,
    3-bit, non qlinear_torch backends) are returned unpacked for the caller
    to route through the legacy per-module path.

    Returns (packed_names, leftover_names).
    """
    import torch as _torch

    from auto_round.utils.model import get_module as _get_module

    if "llm_compressor" in backend:
        raise ValueError(
            f"pack_layers_batched packs the AutoRound qweight/qzeros/scales format and must not be "
            f"used for backend '{backend}': compressed-tensors exports pack through their own "
            f"pack_layer, and mixing the two produces a checkpoint no single consumer can load."
        )

    leftover = []
    groups = {}
    for name in names:
        layer = _get_module(model, name)
        if hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        if type(layer) not in SUPPORTED_LAYER_TYPES:
            continue  # already packed or not quantizable
        if layer.weight.device.type == "meta":
            continue
        if int(getattr(layer, "act_bits", 16)) <= 8:
            leftover.append(name)  # qact path: per-module only
            continue
        if int(layer.bits) not in (2, 4, 8):
            leftover.append(name)  # 3-bit pack path: per-module only
            continue
        if type(layer) is not nn.Linear:
            leftover.append(name)
            continue
        if layer.in_features % 32 != 0 or layer.out_features % 32 != 0:
            # The qweight/qzeros lane math (in // 32 * bits rows, out // 32 * bits
            # columns) is exact only for multiples of 32; other shapes keep the
            # per-module path, which preserves its existing behavior for them.
            leftover.append(name)
            continue
        QuantLinear = dynamic_import_quant_linear_for_packing(
            backend, layer.bits, layer.group_size, layer.sym, int(getattr(layer, "act_bits", 16))
        )
        if QuantLinear not in _batched_packer_classes():
            leftover.append(name)  # unfamiliar pack implementation: per-module
            continue
        if QuantLinear is _wqlinear_gemm_cls() and int(layer.bits) != 4:
            leftover.append(name)  # the awq packer is 4-bit only: per-module
            continue
        key = (
            layer.bits,
            layer.group_size,
            bool(layer.sym),
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
            layer.weight.dtype,
        )
        groups.setdefault(key, []).append(name)

    packed = []
    for (bits, group_size, sym, in_f, out_f, bias, w_dtype), group_names in groups.items():
        if len(group_names) < 2:
            leftover.extend(group_names)
            continue
        members = []
        for n in group_names:
            layer = _get_module(model, n)
            if hasattr(layer, "orig_layer"):
                layer = layer.orig_layer
            members.append((n, layer))
        zp_list = [layer.zp for _, layer in members]
        zp_all_int = all(not isinstance(z, _torch.Tensor) for z in zp_list)
        zp_all_tensor = all(isinstance(z, _torch.Tensor) for z in zp_list)
        if not sym and zp_all_int and len({int(z) for z in zp_list}) != 1:
            logger.warning(
                "batched pack: per-module integer zero points differ within a same-shape "
                "group; falling back to per-module pack for %d module(s)",
                len(group_names),
            )
            leftover.extend(group_names)
            continue
        if sym and zp_all_tensor and any(z.numel() > 1 for z in zp_list):
            # per-module pack_layer only collapses sym tensor zero points to a
            # scalar for the plain packer; the gptq packer's tensor path applies
            # its zp - 1 convention per element -- keep such groups per-module.
            leftover.extend(group_names)
            continue
        if not sym and not zp_all_int and not zp_all_tensor:
            logger.warning(
                "batched pack: mixed tensor and integer zero points within a same-shape "
                "group; falling back to per-module pack for %d module(s)",
                len(group_names),
            )
            leftover.extend(group_names)
            continue
        QuantLinear = dynamic_import_quant_linear_for_packing(backend, bits, group_size, sym, 16)

        # chunk the stack so the int32 intermediate stays bounded
        per_module_elems = out_f * in_f
        per_chunk = max(1, max_batch_bytes // (per_module_elems * 4))
        for start in range(0, len(members), per_chunk):
            layers = members[start : start + per_chunk]
            n_out = out_f * len(layers)

            stacked_w = _torch.cat([layer.weight.data for _, layer in layers], dim=0)
            stacked_scale = _torch.cat(
                [layer.scale.data if isinstance(layer.scale, nn.Parameter) else layer.scale for _, layer in layers],
                dim=0,
            )
            stacked_bias = None
            if bias:
                stacked_bias = _torch.cat([layer.bias.data for _, layer in layers], dim=0)

            is_awq = QuantLinear is _wqlinear_gemm_cls()
            shim = SimpleNamespace(weight=stacked_w, bias=stacked_bias)
            if sym:
                zp0 = zp_list[0]
                zp_arg = int(zp0.flatten()[0]) if isinstance(zp0, _torch.Tensor) else zp0
            elif zp_all_int:
                zp_arg = int(zp_list[0])
            else:
                zp_arg = _torch.cat(zp_list, dim=0)
            if is_awq:
                # the awq packer has no .pack(): from_linear packs directly and
                # packs qweight along out ([in, out // pack_num] layout). It
                # expects scales/zeros in [groups, out] orientation (the
                # per-module awq path transposes them the same way).
                shim.in_features, shim.out_features = in_f, n_out
                awq_scale = stacked_scale.t().contiguous()
                if isinstance(zp_arg, _torch.Tensor):
                    awq_zp = zp_arg.t().contiguous().to(_torch.float32)
                else:
                    awq_zp = zp_arg
                big = QuantLinear.from_linear(shim, bits, group_size, scales=awq_scale, zeros=awq_zp, device=device)
                del awq_scale
            else:
                big = QuantLinear(bits, group_size, in_f, n_out, bias=bias, weight_dtype=w_dtype)
                big.device = stacked_w.device
                big.to("cpu")
                big.pack(shim, stacked_scale, zp_arg, None, device=device)
            del stacked_w, stacked_scale

            oq = out_f // 32 * bits  # qzeros columns per module (both families)
            pack_num = 32 // bits  # awq packs qweight along out with this factor
            for i, (n, layer) in enumerate(layers):
                lo, hi = i * out_f, (i + 1) * out_f
                if is_awq:
                    q = QuantLinear(bits, group_size, in_f, out_f, bias, str(layer.weight.device))
                    qw = big.qweight[:, lo // pack_num : hi // pack_num]
                else:
                    q = QuantLinear(bits, group_size, in_f, out_f, bias=bias, weight_dtype=w_dtype)
                    q.device = layer.weight.device
                    qw = big.qweight[:, lo:hi]
                q.qweight = qw.contiguous()
                q.qzeros = big.qzeros[:, i * oq : (i + 1) * oq].contiguous()
                q.scales = big.scales[:, lo:hi].contiguous()
                if bias:
                    q.bias = big.bias[lo:hi].contiguous()
                if hasattr(model, "_rotation_config"):
                    from auto_round.algorithms.transforms import inject_rotation_buffers_on_layer

                    inject_rotation_buffers_on_layer(n, q, model)
                set_module(model, n, q)
                release_layer_safely(layer)
                packed.append(n)
            del big
    return packed, leftover


def _qlinear_torch_cls():
    import auto_round_extension.torch.qlinear_torch

    return auto_round_extension.torch.qlinear_torch.QuantLinear


def _batched_packer_classes():
    """Packer classes whose packed layout can be column-sliced by out_features.

    - qlinear_torch (plain auto_round): qweight [in // 32 * bits, out],
      zero points stored directly.
    - qlinear_torch_zp (auto_round:auto_gptq): identical layout, zero points
      stored as zp - 1 and unpacked with +1 (handled inside the packer).
    - WQLinear_GEMM (auto_round:auto_awq): qweight packs along out instead
      ([in, out // pack_num]); handled by the awq-specific slice below.
    """
    import auto_round_extension.torch.qlinear_torch
    import auto_round_extension.torch.qlinear_torch_zp
    from auto_round.export.export_to_awq.utils import WQLinear_GEMM

    return (
        auto_round_extension.torch.qlinear_torch.QuantLinear,
        auto_round_extension.torch.qlinear_torch_zp.QuantLinear,
        WQLinear_GEMM,
    )


def _wqlinear_gemm_cls():
    from auto_round.export.export_to_awq.utils import WQLinear_GEMM

    return WQLinear_GEMM


def save_quantized_as_autoround(
    output_dir: str,
    model: torch.nn.Module,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace=True,
    backend="auto_round:exllamav2",
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    **kwargs,
):
    """
    Saves a quantized model in the auto-round format.

    Args:
        output_dir (str): The directory where the quantized model will be saved.
        inplace (bool, optional): If True, modifies the model in place. Otherwise, creates a deepcopy of the model.
                                Default is True.
        backend (str, optional): The backend to be used for quantization.
                                  Default is "autoround:exllamav2".
        **kwargs: Additional keyword arguments including:
            - model (nn.Module): The model to be quantized.
            - layer_config (dict): The layer configuration for each layer.
            - serialization_dict (dict): The serialization configuration.
            - tokenizer (Tokenizer, optional): The tokenizer to be saved.

    Returns:
        None

    Raises:
        ValueError: If the backend is not supported.
    """
    # IF using sym, we change to gptq sym kernel to avoid compiling from auto_round source
    if (
        (serialization_dict.get("sym") is None or serialization_dict.get("sym"))
        and ("gptq" not in backend and "awq" not in backend)
        and (BackendDataType.FP8_STATIC.value not in backend)
    ):
        backend = backend.replace("auto_round", "auto_round:auto_gptq")

    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    quantization_config = serialization_dict
    quantization_config["block_name_to_quantize"] = quantization_config.pop("to_quant_block_names", None)
    quantization_config["quant_method"] = "auto-round"
    quantization_config["packing_format"] = backend

    processor = kwargs.get("processor", None)
    image_processor = kwargs.get("image_processor", None)

    extra_config = {}
    block_name_to_quantize = quantization_config["block_name_to_quantize"]
    if isinstance(block_name_to_quantize, str):
        block_name_to_quantize = [name.strip() for name in block_name_to_quantize.split(",")]
    elif isinstance(block_name_to_quantize, list):
        block_name_to_quantize = [
            os.path.commonprefix(item).rstrip(".") if isinstance(item, list) else item
            for item in block_name_to_quantize
        ]

    scheme_keys = [f.name for f in fields(QuantizationScheme)]
    for layer_name, cfg in layer_config.items():
        if not cfg["in_blocks"] and cfg["bits"] <= 8:  # lm head
            extra_config[layer_name] = {key: cfg.get(key) for key in scheme_keys}
        elif cfg["in_blocks"] or (
            block_name_to_quantize is not None and check_start_with_block_name(layer_name, block_name_to_quantize)
        ):
            neq_keys = check_neq_config(cfg, **{k: quantization_config.get(k) for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[layer_name] = {}
                for key in neq_keys:
                    if cfg.get(key) is not None:
                        extra_config[layer_name][key] = cfg[key]

    regex_config = quantization_config.pop("regex_config")
    if regex_config is not None:
        for name, cfg in regex_config.items():
            regex_name = to_standard_regex(name)
            neq_keys = check_neq_config(cfg, **{k: quantization_config.get(k) for k in scheme_keys})
            if len(neq_keys) > 0:
                extra_config[regex_name] = {}
                for key in neq_keys:
                    if cfg.get(key) is not None:
                        extra_config[regex_name][key] = cfg[key]

    if len(extra_config) > 0:
        quantization_config["extra_config"] = extra_config

    names = list(layer_config.keys())
    if not unsupported_meta_device(model):
        for name in tqdm(names, desc="packing", leave=True):
            pack_layer(name, model, backend, device)
    filter_quantization_config(quantization_config)

    # Inject rotation buffers into QuantLinear modules (if applicable).
    # For shard-based saving the buffers are already injected per-layer in
    # pack_layer(); this call handles the non-shard path where modules are
    # still alive on a real device.
    if hasattr(model, "_rotation_config"):
        from auto_round.algorithms.transforms import inject_rotation_buffers_bulk

        inject_rotation_buffers_bulk(model, quantization_config)

    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    if output_dir is None:
        return model

    if output_dir is None:
        model.tokenizer = tokenizer
        return model
    immediate_saving = is_immediate_saving_mode(model, serialization_dict)
    if os.path.exists(output_dir) and not immediate_saving:
        logger.warning(f"{output_dir} already exists, this may cause model conflict")
    model_output_dir = output_dir
    processor_output_dir = output_dir
    if output_dir:
        model_output_dir, processor_output_dir, _ = resolve_pipeline_export_layout(model, output_dir)

    save_pretrained_artifact(tokenizer, processor_output_dir, artifact_name="tokenizer")

    if processor is not None:
        processor.save_pretrained(processor_output_dir)
    if image_processor is not None:
        image_processor.save_pretrained(processor_output_dir)
    if quantization_config.get("act_bits", 16) <= 8:
        dtype = torch.bfloat16
    elif "awq" in quantization_config.get("packing_format", "auto_round:auto_gptq"):
        dtype = torch.float16  ## awq vllm kernel only supports float16 on cuda
    else:
        dtype = None
    save_model(
        model, model_output_dir, safe_serialization=safe_serialization, dtype=dtype, immediate_saving=immediate_saving
    )

    # Save rotation config to config.json for load-time reconstruction
    if hasattr(model, "_rotation_config"):
        from auto_round.algorithms.transforms import save_rotation_config

        save_rotation_config(model, model_output_dir)

    return model
