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
directory lazily (mmap-backed tensors, bounded host memory) and to unpack
compressed-tensors pack-quantized weights itself (``dequant_model``), so the
offline pass simply runs it on the export with the checkpoint's
quantization_config intact - unlike the interactive flow, which strips it
because quantization happens in-process. GGUF is always a re-quantization
onto GGML grids: numerics equal converting the model directly, with the
official per-tensor type mixing driven by the target format (e.g. q4_k_m).
"""

import glob
import os
from pathlib import Path

import torch

from auto_round.export.export_to_gguf.llama_cpp_conversion import get_conversion
from auto_round.logger import logger


def _install_ggml_quant_bridge():
    """Route the conversion's python quantizer to AutoRound's ggml_quant.

    The pip gguf package implements only some GGML types in python; the rest
    raise NotImplementedError. AutoRound's own packing implements the full
    support surface (classics and k-quants) and is what the interactive GGUF
    export uses, so delegating keeps offline numerics identical to it.
    Returns the patching context manager.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    import gguf as _gguf
    import numpy as np

    from auto_round.export.export_to_gguf.packing import ggml_quant

    _FLOAT_TYPES = {"f32", "f16", "bf16"}

    def quantize(data, qtype):
        # np.array materializes gguf's lazy mmap tensors (ndarray subclasses)
        # into a base-class array; float types pass through unquantized
        materialized = np.array(data)
        if qtype.name.lower() in _FLOAT_TYPES:
            return materialized
        out = ggml_quant(torch.from_numpy(materialized), qtype.name.lower(), device="cpu")
        return out if isinstance(out, np.ndarray) else np.asarray(out)

    @contextmanager
    def _bridge():
        with patch.object(_gguf.quants, "quantize", side_effect=quantize):
            yield

    return _bridge()


def _make_official_mixed_selector(model_instance, hparams, ftype, model_architecture):
    """Per-tensor qtype selection for mixed GGUF formats (q4_k_m and friends).

    The base conversion resolves only the simple file types on its own; the
    official llama.cpp per-tensor mixing rules live in GGUFDTypeSelector, so
    bind them onto the instance's tensor_force_quant. Simple formats select a
    single qtype and are untouched by this.
    """
    from types import MethodType

    from auto_round.export.export_to_gguf.gguf_dtype import GGUFDTypeSelector

    selector = GGUFDTypeSelector(
        hparams,
        ftype,
        model_arch=model_architecture,
        has_tied_embeddings=bool(hparams.get("tie_word_embeddings", False)),
    )

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # select_qtype returns the GGMLQuantizationType directly; 1D tensors
        # and always-float names are handled by the conversion loop afterwards
        return selector.select_qtype(new_name, n_dims, fallback_index=bid or 0)

    return MethodType(tensor_force_quant, model_instance)


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
    if output_type not in FTYPE_MAP:
        raise ValueError(f"unsupported gguf format {gguf_format!r}; expected one of {', '.join(sorted(FTYPE_MAP))}")
    # the conversion derives the file name from model metadata only when the
    # output path is an existing directory
    os.makedirs(output_dir, exist_ok=True)

    conversion = get_conversion(checkpoint_dir, model_type=ModelType.TEXT)
    is_mistral_format = False
    hparams = conversion.ModelBase.load_hparams(Path(checkpoint_dir), is_mistral_format)
    # keep quantization_config in hparams: the conversion dequantizes
    # compressed-tensors packed weights itself when it is present
    model_architecture = conversion.get_model_architecture(hparams, conversion.model_type(ModelType.TEXT))
    model_class = conversion.get_model_class(model_architecture, model_type=ModelType.TEXT)
    ftype = FTYPE_MAP[output_type]
    model_instance = model_class(
        hparams=hparams,
        dir_model=Path(checkpoint_dir),
        ftype=ftype,
        fname_out=Path(output_dir),
        is_big_endian=False,
        model_name=Path(checkpoint_dir).name,
        split_max_tensors=False,
        split_max_size=0,
        dry_run=False,
        small_first_shard=False,
    )
    model_instance.tensor_force_quant = _make_official_mixed_selector(
        model_instance, hparams, ftype, model_architecture
    )
    model_instance = handle_special_model(model_instance, model_architecture)
    logger.info("offline gguf conversion: %s -> %s (%s)", checkpoint_dir, output_dir, gguf_format)
    with _install_ggml_quant_bridge():
        model_instance.write()

    written = sorted(glob.glob(os.path.join(output_dir, "*.gguf")))
    if not written:
        raise RuntimeError(f"gguf conversion finished but no .gguf file was written under {output_dir}")
    return written[0]
