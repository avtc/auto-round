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

"""Offline GGUF conversion from an exported checkpoint directory.

stream_quantization writes compressed-tensors packed shards immediately per
block and cannot host the single-file GGUF container; this module converts
such an export (or any HF checkpoint) to GGUF as an independent, idempotent
pass afterwards.

The bundled llama.cpp conversion already knows how to read a checkpoint
directory lazily (mmap-backed tensors, spooled writer) and to unpack
compressed-tensors pack-quantized weights itself (``dequant_model``), so the
offline pass simply runs it on the export with the checkpoint's
quantization_config intact - unlike the interactive flow, which strips it
because quantization happens in-process. GGUF is always a re-quantization
onto GGML grids, with the official per-tensor type mixing driven by the
target format (e.g. q4_k_m); AutoRound's ggml_quant implements the python
side of the quantizers.
"""

import os
from pathlib import Path

import torch

from auto_round.export.export_to_gguf.llama_cpp_conversion import get_conversion
from auto_round.logger import logger

# Tensor categories the bundled conversion's own loop forces to float32
# regardless of the target format (router gates, state-space kernels,
# positional embeddings, ...). The offline selector override replaces
# tensor_force_quant entirely, so it must enforce the same protections
# itself; keep in sync with the inline list in the conversion base's
# prepare_tensors.
_ALWAYS_F32_TENSOR_KEYS = (
    "FFN_GATE_INP",
    "FFN_GATE_INP_SHEXP",
    "POS_EMBD",
    "TOKEN_TYPES",
    "SSM_CONV1D",
    "SHORTCONV_CONV",
    "TIME_MIX_FIRST",
    "TIME_MIX_W1",
    "TIME_MIX_W2",
    "TIME_MIX_DECAY_W1",
    "TIME_MIX_DECAY_W2",
    "TIME_MIX_LERP_FUSED",
    "POSNET_NORM1",
    "POSNET_NORM2",
    "V_ENC_EMBD_POS",
    "A_ENC_EMBD_POS",
    "ALTUP_CORRECT_COEF",
    "ALTUP_PREDICT_COEF",
    "SSM_CONV1D_Q",
    "SSM_CONV1D_K",
    "SSM_CONV1D_V",
    "INDEXER_PROJ",
)


def _install_ggml_quant_bridge():
    """Route the conversion's python quantizer to AutoRound's ggml_quant.

    The pip gguf package implements only some GGML types in python; the rest
    raise NotImplementedError. AutoRound's own packing implements the full
    quantized support surface and is what the interactive GGUF export uses,
    so delegating keeps offline numerics identical to it. Float storage
    types (F32/F16/BF16) stay with the original implementation, which casts
    (or byte-views for BF16) correctly. Returns the patching context manager.
    """
    from contextlib import contextmanager

    import gguf as _gguf
    import numpy as np

    from auto_round.export.export_to_gguf.packing import ggml_quant

    original_quantize = _gguf.quants.quantize
    float_types = {
        _gguf.GGMLQuantizationType.F32,
        _gguf.GGMLQuantizationType.F16,
        _gguf.GGMLQuantizationType.BF16,
    }

    def quantize(data, qtype):
        if qtype in float_types:
            # the original handles float storage (astype, bf16 byte view) and
            # materializes lazy mmap tensors itself
            return original_quantize(data, qtype)
        # np.array materializes gguf's lazy mmap tensors (ndarray subclasses)
        # into a base-class array torch.from_numpy can consume
        out = ggml_quant(torch.from_numpy(np.array(data)), qtype.name.lower(), device="cpu")
        return out if isinstance(out, np.ndarray) else np.asarray(out)

    @contextmanager
    def _bridge():
        with _patch_object(_gguf.quants, "quantize", quantize):
            yield

    return _bridge()


def _patch_object(obj, attr, value):
    """Scoped attribute replacement without pulling unittest.mock into the
    production path."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        saved = getattr(obj, attr)
        setattr(obj, attr, value)
        try:
            yield
        finally:
            setattr(obj, attr, saved)

    return _ctx()


def _make_official_mixed_selector(model_instance, hparams, ftype):
    """Per-tensor qtype selection for mixed GGUF formats (q4_k_m and friends).

    The base conversion resolves only the simple file types on its own; the
    official llama.cpp per-tensor mixing rules live in GGUFDTypeSelector. The
    instance override chains: the model class's own rules first (some
    architectures force specific types), then the always-float32 protections
    the base loop would have applied, then the selector. Simple formats
    select a single qtype and are untouched by this.
    """
    from types import MethodType

    import gguf as _gguf

    from auto_round.export.export_to_gguf.gguf_dtype import GGUFDTypeSelector

    selector = GGUFDTypeSelector(
        hparams,
        ftype,
        model_arch=model_instance.model_arch,
        has_tied_embeddings=bool(hparams.get("tie_word_embeddings", False)),
    )
    selector.n_attention_wv = _count_attention_wv_tensors(model_instance)
    original_tensor_force_quant = model_instance.tensor_force_quant
    always_f32_keys = tuple(getattr(_gguf.MODEL_TENSOR, key) for key in _ALWAYS_F32_TENSOR_KEYS)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        result = original_tensor_force_quant(name, new_name, bid, n_dims)
        if not isinstance(result, bool):
            return result
        # 1D tensors and norms only handle F32; non-weight suffixes stay float
        if n_dims <= 1 or new_name.endswith("_norm.weight") or not new_name.endswith((".weight", ".lora_a", ".lora_b")):
            return _gguf.GGMLQuantizationType.F32
        if any(self.match_model_tensor_name(new_name, key, bid) for key in always_f32_keys):
            return _gguf.GGMLQuantizationType.F32
        return selector.select_qtype(new_name, n_dims, fallback_index=bid or 0)

    return MethodType(tensor_force_quant, model_instance)


def _count_attention_wv_tensors(model_instance):
    """Count attention-value tensors the way the interactive loop does, so the
    selector's wv-indexed rules (q6_k boosts in mixed formats) see the same
    numbering; ``None`` lets the selector fall back to its internal counter."""
    count = 0
    for name in model_instance.model_tensors:
        try:
            new_name = model_instance.map_tensor_name(name)
        except Exception:  # pylint: disable=broad-except
            continue
        if any(key in new_name for key in ("attn_v.weight", "attn_qkv.weight", "attn_kv_b.weight")):
            count += 1
    return count or None


def convert_checkpoint_to_gguf(
    checkpoint_dir: str,
    output_dir: str,
    gguf_format: str = "gguf:q4_k_m",
) -> str:
    """Convert an exported (or plain) HF checkpoint directory to a GGUF file.

    Returns the path of the written ``.gguf``. The pass is independent and
    idempotent: re-running on the same input rewrites the same file.
    """
    from auto_round.export.export_to_gguf.config import ModelType
    from auto_round.export.export_to_gguf.export import FTYPE_MAP
    from auto_round.export.export_to_gguf.special_handle import handle_special_model

    if not os.path.isfile(os.path.join(checkpoint_dir, "config.json")):
        raise ValueError(f"no config.json under {checkpoint_dir}; expected an exported checkpoint directory")
    output_type = gguf_format.split(":")[-1].lower()
    if output_type not in FTYPE_MAP or output_type == "auto":
        supported = ", ".join(sorted(k for k in FTYPE_MAP if k != "auto"))
        raise ValueError(f"unsupported gguf format {gguf_format!r}; expected one of {supported}")
    # the conversion derives the file name from model metadata only when the
    # output path is an existing directory
    os.makedirs(output_dir, exist_ok=True)

    conversion = get_conversion(checkpoint_dir, model_type=ModelType.TEXT)
    hparams = conversion.ModelBase.load_hparams(Path(checkpoint_dir), False)
    if "mistral" in str(hparams.get("model_type", "")) and "params.json" in os.listdir(checkpoint_dir):
        hparams = conversion.ModelBase.load_hparams(Path(checkpoint_dir), True)
    # keep quantization_config in hparams: the conversion dequantizes
    # compressed-tensors packed weights itself when it is present
    model_architecture = conversion.get_model_architecture(hparams, conversion.model_type(ModelType.TEXT))
    model_class = conversion.get_model_class(model_architecture, model_type=ModelType.TEXT)
    model_instance = model_class(
        hparams=hparams,
        dir_model=Path(checkpoint_dir),
        ftype=FTYPE_MAP[output_type],
        fname_out=Path(output_dir),
        is_big_endian=False,
        model_name=Path(checkpoint_dir).name,
        split_max_tensors=False,
        split_max_size=0,
        dry_run=False,
        small_first_shard=False,
        use_temp_file=True,
    )
    model_instance.tensor_force_quant = _make_official_mixed_selector(model_instance, hparams, FTYPE_MAP[output_type])
    model_instance = handle_special_model(model_instance, model_architecture)
    logger.info("offline gguf conversion: %s -> %s (%s)", checkpoint_dir, output_dir, gguf_format)
    with _install_ggml_quant_bridge():
        model_instance.write()

    written = model_instance.fname_out
    if not os.path.isfile(str(written)):
        raise RuntimeError(f"gguf conversion finished but no .gguf file was written under {output_dir}")
    return str(written)
