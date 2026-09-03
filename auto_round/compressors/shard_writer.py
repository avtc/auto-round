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

import json
import os
import struct
from collections import OrderedDict
from typing import Optional, Union

import torch

from auto_round.compressors.utils import _get_save_folder_name
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext
from auto_round.experimental.attention import is_attention_calibration_tensor_name
from auto_round.logger import logger
from auto_round.utils import (
    get_lm_head_name,
    get_module,
    get_reverse_checkpoint_conversion_mapping,
    revert_checkpoint_conversion_mapping,
)

DEFAULT_MAX_SHARD_SIZE = "5GB"


class ShardWriter:
    """
    Handles shard-saving of model parameters to disk with memory management.
    """

    _instance = None
    _initialized = False

    model = None
    lm_head_name = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}
        return cls._instance

    def __init__(
        self,
        model: torch.nn.Module,
        bits: int,
        max_shard_size: Optional[Union[int, str]] = None,
        safe_serialization: bool = True,
    ) -> None:
        if ShardWriter._initialized:
            return
        self.model = model
        self.lm_head_name = get_lm_head_name(self.model)
        self.max_shard_size = self._parse_size(max_shard_size or DEFAULT_MAX_SHARD_SIZE)
        self.safe_serialization = safe_serialization

        # Internal State
        self.use_safetensors = self._check_safetensors()
        self.shard_suffix = "safetensors" if self.use_safetensors else "bin"
        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0
        self.shard_meta = []  # List of {tmp_file: str, params: list}
        self.global_weight_map = {}
        self.shard_counter = 0
        self.reverse_checkpoint_conversion_mapping = get_reverse_checkpoint_conversion_mapping(self.model)

        # Persistent set of all parameter names already flushed to a shard file.
        # Maintained incrementally in _flush_shard to avoid O(N^2) rebuilds in _add_tensor.
        self._all_saved = set()

        # Stats
        self.total_param_elems = 0
        self.total_param_size_bytes = 0
        self.skipped_meta_tensors = []

        ShardWriter._initialized = True

    @property
    def output_dir(self) -> str:
        """Derive the output directory from the current CompressContext at access time.

        Reading from context rather than caching the path at construction time ensures
        the ShardWriter always uses the final export directory even if
        ``CompressContext.output_dir`` is updated after the ShardWriter was created
        (e.g. by ``_get_export_dir()`` in ``quantize_and_save()``).
        """
        compress_context = CompressContext.get_context()
        formats = compress_context.formats
        base_dir = _get_save_folder_name(formats[0])
        subfolder = getattr(self.model, "_autoround_pipeline_subfolder", None)
        if subfolder:
            base_dir = os.path.join(base_dir, subfolder)
        return os.path.join(base_dir, "")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton state so the next instantiation creates a fresh ShardWriter."""
        cls._initialized = False
        cls._instance = None

    @staticmethod
    def _read_safetensors_header(path: str) -> Optional[dict]:
        """Parse a safetensors header (tensor_name -> dtype/shape) without loading data."""
        try:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            return header
        except (OSError, ValueError, struct.error, OverflowError):
            return None

    def adopt_existing_shards(self) -> int:
        """Adopt shard files a previous (crashed) run already wrote to the output dir.

        Resume support: the manifest marks a block done only after its tensors are
        durably flushed, so a resumed process must not re-write those tensors -- but
        a fresh ShardWriter starts at ``shard_counter = 0`` and would silently
        OVERWRITE ``model-shard-00001`` on its first flush, destroying the crashed
        run's data and dropping its tensors from the final index. Adoption
        reconstructs the writer's bookkeeping (shard_meta / _all_saved / counters /
        size stats) from the safetensors headers on disk.

        Rules:
        - only the highest-numbered shard may be unparseable (a crash mid-flush):
          it is deleted, its tensors belong to the un-marked in-flight block and
          will be re-written by the resumed run;
        - an unparseable non-tail shard is corruption: hard error (never silent
          data loss);
        - ``.bin`` shards cannot be adopted (no cheap header): hard error telling
          the user resume requires safe_serialization.

        Returns the number of adopted shards.
        """
        import re

        output_dir = self.output_dir
        if not os.path.isdir(output_dir):
            return 0

        # shard sources, keyed by their ordinal position in the sequence:
        # - temp files (``model-shard-NNN``): a crash during the block loop
        # - finalized files (``model-000NN-of-000MM`` / single ``model.safetensors``):
        #   a crash AFTER the writer finalized (e.g. in the export stage that
        #   follows) - without adopting these, a fresh writer restarts at
        #   counter 0, rewrites tensors and its finalize clobbers the index,
        #   orphaning every previously written shard
        temp_pattern = re.compile(r"^model-shard-(\d+)\.safetensors$")
        final_pattern = re.compile(r"^model-(\d{5})-of-\d{5}\.safetensors$")
        seq = {}  # ordinal -> (fname, is_final)
        single_file = None
        for fname in os.listdir(output_dir):
            mobj = temp_pattern.match(fname)
            if mobj:
                seq[int(mobj.group(1))] = (fname, False)
                continue
            mobj = final_pattern.match(fname)
            if mobj:
                seq[int(mobj.group(1))] = (fname, True)
                continue
            if fname == "model.safetensors":
                single_file = fname
        found = {num: fname for num, (fname, _is_final) in seq.items() if not _is_final}
        if not seq and single_file is not None:
            seq[1] = (single_file, True)
        if not seq:
            bin_shards = [f for f in os.listdir(output_dir) if re.match(r"^model-shard-\d+\.bin$", f)]
            if bin_shards:
                raise RuntimeError(
                    f"ShardWriter resume: found torch .bin shards in {output_dir} but adoption requires "
                    "safetensors headers (safe_serialization). Use a fresh --output_dir or enable safetensors."
                )
            return 0

        from auto_round.utils.checkpoint_streamer import _DTYPE_BYTES

        dtype_sizes = _DTYPE_BYTES
        numbers = sorted(seq)
        for num in numbers:
            path = os.path.join(output_dir, seq[num][0])
            header = self._read_safetensors_header(path)
            if header is None and seq[num][1]:
                continue  # finalized files are complete by construction
            if header is None:
                if num == numbers[-1]:
                    logger.warning(
                        "ShardWriter resume: tail shard %s is incomplete (crash mid-flush); deleting it -- "
                        "its block was not marked done and will be re-done",
                        found[num],
                    )
                    os.remove(path)
                    continue
                raise RuntimeError(
                    f"ShardWriter resume: shard {found[num]} in {output_dir} is corrupt (only the tail "
                    "shard of a crashed run may be incomplete). Use a fresh --output_dir."
                )
            # a crash DURING the data write (after the header flushed) leaves
            # a header-valid but truncated file: verify the declared data end
            # fits inside the actual file size before adopting
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(0, 2)
                file_size = f.tell()
            max_end = 0
            for spec in header.values():
                try:
                    max_end = max(max_end, int(spec["data_offsets"][1]))
                except (KeyError, TypeError, ValueError, IndexError):
                    max_end = file_size  # unexpected spec: fall through to the size check
                    break
            if 8 + header_len + max_end > file_size:
                if num == numbers[-1]:
                    logger.warning(
                        "ShardWriter resume: tail shard %s is truncated (crash mid-data-write); deleting it -- "
                        "its block was not marked done and will be re-done",
                        found[num],
                    )
                    os.remove(path)
                    continue
                raise RuntimeError(
                    f"ShardWriter resume: shard {found[num]} in {output_dir} is truncated (only the tail "
                    "shard of a crashed run may be incomplete). Use a fresh --output_dir."
                )
            params = list(header.keys())
            fname, _is_final = seq[num]
            self.shard_meta.append({"tmp_file": fname, "params": params, "dir": output_dir})
            self._all_saved.update(params)
            for name, meta in header.items():
                numel = 1
                for dim in meta.get("shape", []):
                    numel *= dim
                self.total_param_elems += numel
                self.total_param_size_bytes += numel * dtype_sizes.get(meta.get("dtype", "F32"), 4)
            self.shard_counter = max(self.shard_counter, num)

        if self.shard_meta:
            logger.info(
                "ShardWriter resume: adopted %d existing shard(s) from %s (%d tensors); new shards continue at %05d",
                len(self.shard_meta),
                output_dir,
                len(self._all_saved),
                self.shard_counter + 1,
            )
        return len(self.shard_meta)

    @classmethod
    def get_shard_writer(cls, *args, **kwargs) -> Optional["ShardWriter"]:
        """Return the current singleton instance, or None if not yet initialized.

        Callers that require a valid writer should guard the result with
        ``if self.compress_context.is_immediate_saving`` before use.
        """
        return cls._instance

    def _parse_size(self, size_str: str) -> int:
        if isinstance(size_str, int):
            return size_str
        s = size_str.strip().upper()
        units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
        for unit, mult in units.items():
            if s.endswith(unit):
                return int(float(s[: -len(unit)]) * mult)
        return int(s)

    def _check_safetensors(self) -> bool:
        if self.safe_serialization:
            try:
                import safetensors.torch

                return True
            except ImportError:
                logger.warning("safetensors not installed; falling back to torch.save.")
        return False

    def save_module(self, m: torch.nn.Module, name: str = None) -> None:
        """Extracts and accumulates tensors from a module."""
        prefix = name if name is not None else getattr(m, "global_name", "model")
        sd = m.state_dict()

        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if is_attention_calibration_tensor_name(k):
                continue
            param_name = f"{prefix}.{k}"
            self._add_tensor(param_name, v)

    def save_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Accumulate a single raw (checkpoint-spelled) tensor for saving."""
        self._add_tensor(name, tensor)

    def _expand_fused_experts(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Expand a fused 3D expert parameter into per-expert 2D weight tensors.

        Non-quantized MoE modules (e.g. thinker in Qwen3-Omni) are kept fused
        during quantization to prevent memory eviction under disk offloading.
        This method converts them to the per-expert checkpoint format at save time.

        Talker MoE modules stay fused during quantization, but save_pretrained
        still needs concrete per-expert 2D checkpoint keys. Saving the original
        fused tensor under a reverse-mapped wildcard key (for example
        ``experts.*.gate_proj.weight``) breaks reload.

        Returns:
            List of (key, 2D tensor) pairs, or None if not a fused expert param.
        """
        from auto_round.modeling.fused_moe.replace_modules import MOE_SKIP_PREFIXES
        from auto_round.utils.missing_tensors import split_fused_expert_tensors

        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            return None
        prefix, attr_name = parts

        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        skip_prefixes = MOE_SKIP_PREFIXES.get(model_type, []) if model_type is not None else []
        if not any(
            prefix == skip_prefix.rstrip(".") or prefix.startswith(skip_prefix) for skip_prefix in skip_prefixes
        ):
            return None

        expanded = split_fused_expert_tensors({name: tensor})
        if set(expanded) == {name}:
            return None
        return list(expanded.items())

    def _add_tensor(self, name: str, tensor: torch.Tensor):
        if is_attention_calibration_tensor_name(name):
            return
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "meta":
            self.skipped_meta_tensors.append(name)
            return

        # Guard against duplicate saving of the same parameter
        if name in self._all_saved or name in self.current_shard_tensors:
            return

        # Expand fused 3D expert parameters into per-expert 2D tensors if necessary
        if tensor.dim() == 3:
            expanded = self._expand_fused_experts(name, tensor)
            if expanded is not None:
                self._all_saved.add(name)
                for sub_name, sub_tensor in expanded:
                    self._add_tensor(sub_name, sub_tensor)
                return

        # transformers will handle _checkpoint_conversion_mapping automatically if is_immediate_saving=False
        name = revert_checkpoint_conversion_mapping(name, self.reverse_checkpoint_conversion_mapping)

        t_size = tensor.nbytes
        self.total_param_elems += tensor.numel()
        self.total_param_size_bytes += t_size
        tensor = tensor.detach().cpu()
        # Keep an oversized tensor with any buffered tensors so it does not
        # leave a tiny shard immediately before its own shard.
        if t_size > self.max_shard_size:
            self.current_shard_tensors[name] = tensor
            self.current_shard_size += t_size
            self._flush_shard()
        # If adding exceeds limit, flush first
        elif self.current_shard_size + t_size > self.max_shard_size and self.current_shard_size > 0:
            self._flush_shard()
            self.current_shard_tensors[name] = tensor
            self.current_shard_size = t_size
        else:
            self.current_shard_tensors[name] = tensor
            self.current_shard_size += t_size

    def _handle_tied_weights(self):
        """
        Detects tied weights in the current shard and ensures they are only saved once.
        This is done by tracking storage pointers of tensors and skipping duplicates.
        """
        storage_map = set()
        filtered_tensors = {}

        for name, tensor in self.current_shard_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                filtered_tensors[name] = tensor
                continue

            ptr = tensor.untyped_storage().data_ptr() + tensor.storage_offset() * tensor.element_size()
            if ptr not in storage_map:
                storage_map.add(ptr)
                filtered_tensors[name] = tensor
        self.current_shard_tensors = filtered_tensors

    def _flush_shard(self):
        if not self.current_shard_tensors:
            return

        self.shard_counter += 1
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        tmp_name = f"model-shard-{self.shard_counter:05d}.{self.shard_suffix}"
        tmp_path = os.path.join(output_dir, tmp_name)
        self._handle_tied_weights()

        if self.use_safetensors:
            from safetensors.torch import save_file

            # Ensure tensors are contiguous in-place to avoid duplicating them in a separate dict,
            # which can increase peak RAM usage during saving.
            for k, v in list(self.current_shard_tensors.items()):
                if isinstance(v, torch.Tensor) and not v.is_contiguous():
                    self.current_shard_tensors[k] = v.contiguous()
            save_file(self.current_shard_tensors, tmp_path)
        else:
            torch.save(self.current_shard_tensors, tmp_path)

        saved_params = list(self.current_shard_tensors.keys())
        self.shard_meta.append({"tmp_file": tmp_name, "params": saved_params, "dir": output_dir})
        self._all_saved.update(saved_params)

        # Offload logic: move modules to meta device once all params are saved
        self._offload_to_meta(saved_params)

        self.current_shard_tensors = OrderedDict()
        self.current_shard_size = 0

    def _offload_to_meta(self, saved_params):
        """Attempts to move fully saved modules to the 'meta' device to free RAM."""
        for param_full_name in saved_params:
            module_path = param_full_name.rsplit(".", 1)[0]

            module = get_module(self.model, module_path)
            # Check if all parameters of this module are now in '_all_saved'
            if (
                module is not None
                and isinstance(module, torch.nn.Module)
                and all(f"{module_path}.{k}" in self._all_saved for k in module.state_dict().keys())
            ):
                module.to("meta")

    def finalize(self) -> None:
        """Saves remaining weights, renames files, and writes the index JSON."""
        # 1. Capture remaining weights not yet saved
        full_sd = self.model.state_dict()
        tie_word_embeddings = False
        if hasattr(self.model, "config") and hasattr(self.model.config, "tie_word_embeddings"):
            tie_word_embeddings = self.model.config.tie_word_embeddings

        finalize_skipped_meta_tensors = []
        for pname, tensor in full_sd.items():
            if pname in self._all_saved:
                continue
            if tensor.device.type == "meta":
                continue
            layer_name = ".".join(pname.split(".")[:-1])
            if self.lm_head_name is not None and layer_name == self.lm_head_name and tie_word_embeddings:
                lm_head_module = get_module(self.model, self.lm_head_name)
                lm_head_module.to("meta")  # Must to meta, otherwise model's saver will dump it again
                continue
            self._add_tensor(pname, tensor.detach().to("cpu"))

        self._flush_shard()

        total_skipped = len(self.skipped_meta_tensors) + len(finalize_skipped_meta_tensors)
        if total_skipped > 0:
            examples = (self.skipped_meta_tensors + finalize_skipped_meta_tensors)[:5]

        # 2. Rename temp files to HF standard and map weights
        if self.shard_counter == 0:
            logger.warning("No tensors saved.")
            return

        output_dir = self.output_dir
        for idx, meta in enumerate(self.shard_meta, start=1):
            shard_dir = meta.get("dir", output_dir)
            old_path = os.path.join(shard_dir, meta["tmp_file"])
            new_name = (
                f"model.{self.shard_suffix}"
                if self.shard_counter == 1
                else f"model-{idx:05d}-of-{self.shard_counter:05d}.{self.shard_suffix}"
            )
            new_path = os.path.join(shard_dir, new_name)
            os.replace(old_path, new_path)
            for p in meta["params"]:
                self.global_weight_map[p] = new_name

        # 3. Write Index JSON
        index_ext = "safetensors.index.json" if self.use_safetensors else "bin.index.json"
        index_path = os.path.join(output_dir, f"model.{index_ext}")

        index_data = {
            "metadata": {
                "format": "safetensors" if self.use_safetensors else "pytorch",
                "total_shards": self.shard_counter,
                "total_parameters": int(self.total_param_elems),
                "total_size": int(self.total_param_size_bytes),
            },
            "weight_map": self.global_weight_map,
        }

        if self.shard_counter > 1:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)

        logger.info(f"model has been saved to {self.output_dir}")

    @torch.no_grad()
    def write(self, m: torch.nn.Module = None, name: str = None, is_finalize: bool = False) -> None:
        if m is None and name is None and not is_finalize and not is_finalize:
            raise ValueError("Must specify either name or m")
        if m is None and name is not None:
            m = get_module(self.model, name)
            # Perform the save
        if m is not None:
            self.save_module(m, name)

        if is_finalize:
            self.finalize()
            ShardWriter._initialized = False
            ShardWriter._instance = None
