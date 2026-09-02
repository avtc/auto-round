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

"""Generic CLI adapter for algorithm-owned parameter declarations."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import replace
from typing import Any, get_args, get_origin, get_type_hints

from auto_round.algorithms.config import AlgorithmConfig, AlgorithmParameterRegistry
from auto_round.algorithms.registry import (
    get_algorithm_entry,
    iter_algorithm_entries,
    resolve_algorithm_alias,
    resolve_algorithm_names,
)


def _parameter_registry(config_cls: type) -> AlgorithmParameterRegistry:
    if _has_custom_register_args(config_cls):
        return config_cls.get_registered_args()
    return _fallback_parameter_registry(config_cls)


def _has_custom_register_args(config_cls: type) -> bool:
    if not issubclass(config_cls, AlgorithmConfig):
        return False
    for cls in config_cls.__mro__:
        if cls is AlgorithmConfig:
            break
        if "register_args" in cls.__dict__:
            return True
    return False


def _fallback_parameter_registry(config_cls: type) -> AlgorithmParameterRegistry:
    """Derive simple CLI arguments from a config that has no custom registration."""
    registry = AlgorithmParameterRegistry()
    init = config_cls.__init__
    try:
        type_hints = get_type_hints(init)
    except (NameError, TypeError):
        type_hints = {}

    for name, parameter in inspect.signature(init).parameters.items():
        if name in {"self", "algorithm"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = type_hints.get(name, parameter.annotation)
        default = parameter.default
        if annotation is inspect.Parameter.empty:
            annotation = type(default) if default is not inspect.Parameter.empty and default is not None else str
        if annotation is bool or isinstance(default, bool):
            action = argparse.BooleanOptionalAction
            kwargs = {"action": action, "default": argparse.SUPPRESS}
        else:
            origin = get_origin(annotation)
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            value_type = args[0] if origin is not None and args else annotation
            kwargs = {"type": value_type, "default": argparse.SUPPRESS}
        registry.add_argument(f"--{name}", field=name, **kwargs)
    return registry


def _argument_compatibility_key(parameter) -> tuple:
    kwargs = parameter.argparse_kwargs
    return (
        kwargs.get("action"),
        kwargs.get("type"),
        tuple(kwargs.get("choices", ())) if kwargs.get("choices") is not None else None,
        kwargs.get("nargs"),
        kwargs.get("const"),
        kwargs.get("dest", parameter.dest),
    )


def _existing_action_compatibility_key(action) -> tuple:
    return (
        type(action),
        action.type,
        tuple(action.choices) if action.choices is not None else None,
        action.nargs,
        action.const,
        action.dest,
    )


def _check_existing_parser_argument(parser, parameter, *, fallback: bool = False) -> bool:
    """Reuse an existing common argument when its parsing semantics match."""
    actions = [
        parser._option_string_actions[option]
        for option in parameter.option_strings
        if option in parser._option_string_actions
    ]
    if not actions:
        return False
    action = actions[0]
    parameter_key = _argument_compatibility_key(parameter)
    action_key = _existing_action_compatibility_key(action)
    if parameter_key[1:] != action_key[1:]:
        if fallback:
            return True
        option = next(option for option in parameter.option_strings if option in parser._option_string_actions)
        raise ValueError(f"incompatible shared CLI argument {option!r}")
    return True


def _merge_parameter(merged, parameter):
    """Merge one shared CLI parameter or raise for incompatible definitions."""
    overlapping_options = set(merged.option_strings) & set(parameter.option_strings)
    if not overlapping_options and merged.dest != parameter.dest:
        return None
    if merged.dest != parameter.dest or _argument_compatibility_key(merged) != _argument_compatibility_key(parameter):
        option = sorted(overlapping_options)[0] if overlapping_options else merged.dest
        raise ValueError(f"incompatible shared CLI argument {option!r}")
    kwargs = dict(merged.argparse_kwargs)
    kwargs["default"] = argparse.SUPPRESS
    return replace(
        merged,
        option_strings=tuple(dict.fromkeys(merged.option_strings + parameter.option_strings)),
        argparse_kwargs=kwargs,
    )


def _add_parameter(group, parameter) -> None:
    kwargs = dict(parameter.argparse_kwargs)
    kwargs.pop("fallback", None)
    if kwargs.get("action") == "boolean_optional":
        kwargs["action"] = argparse.BooleanOptionalAction
    group.add_argument(*parameter.option_strings, **kwargs)


def _add_registered_arguments(group, registry: AlgorithmParameterRegistry) -> None:
    mutex_groups = {}
    for parameter in registry.parameters:
        target = group
        if parameter.mutex_group is not None:
            if parameter.mutex_group not in mutex_groups:
                mutex_groups[parameter.mutex_group] = group.add_mutually_exclusive_group()
            target = mutex_groups[parameter.mutex_group]
        _add_parameter(target, parameter)


class AlgorithmHandler:
    """Generic discovery, argparse adaptation, and config construction."""

    @classmethod
    def get(cls, name: str) -> type:
        entry = get_algorithm_entry(name)
        if entry.config_factory is None:
            raise KeyError(f"No config class registered for algorithm '{name}'.")
        return entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())

    @classmethod
    def resolve_alias(cls, user_name: str) -> str | None:
        return resolve_algorithm_alias(user_name)

    @classmethod
    def add_group(cls, name: str, group) -> None:
        _add_registered_arguments(group, _parameter_registry(cls.get(name)))

    @classmethod
    def add_groups(cls, parser) -> None:
        merged_parameters = []
        parameters_by_group = {}
        locations = []
        for entry in iter_algorithm_entries():
            if entry.config_factory is None:
                continue
            config_cls = (
                entry.config_factory if isinstance(entry.config_factory, type) else type(entry.config_factory())
            )
            registry = _parameter_registry(config_cls)
            fallback = not _has_custom_register_args(config_cls)
            if registry.parameters:
                group_name = f"Algorithm: {entry.name}"
                parameters_by_group.setdefault(group_name, [])
                for parameter in registry.parameters:
                    if _check_existing_parser_argument(parser, parameter, fallback=fallback):
                        continue
                    existing_index = next(
                        (
                            index
                            for index, item in enumerate(merged_parameters)
                            if locations[index][0] != group_name
                            and (
                                item.dest == parameter.dest or set(item.option_strings) & set(parameter.option_strings)
                            )
                        ),
                        None,
                    )
                    if existing_index is not None:
                        merged = _merge_parameter(merged_parameters[existing_index], parameter)
                        merged_parameters[existing_index] = merged
                        existing_group, existing_position = locations[existing_index]
                        parameters_by_group[existing_group][existing_position] = merged
                        continue
                    merged_parameters.append(parameter)
                    parameters_by_group[group_name].append(parameter)
                    locations.append((group_name, len(parameters_by_group[group_name]) - 1))

        for group_name, parameters in parameters_by_group.items():
            group = parser.add_argument_group(group_name)
            registry = AlgorithmParameterRegistry()
            registry.parameters = parameters
            _add_registered_arguments(group, registry)

    @classmethod
    def build_configs(cls, args, common_kwargs: dict[str, Any]) -> list:
        raw = getattr(args, "algorithm", None) or ""
        names = [name.strip().lower() for name in raw.split(",") if name.strip()]

        if getattr(args, "rotation_hadamard_type", None) and "hadamard" not in names:
            names.append("hadamard")

        canonical = resolve_algorithm_names(names, ignore_unknown=True)
        seen = set(canonical)
        if not ({"rtn", "auto_round"} & seen):
            canonical.append("rtn" if "awq" in seen or getattr(args, "iters", 0) == 0 else "auto_round")
        if getattr(args, "iters", None) == 0:
            canonical = ["rtn" if name == "auto_round" else name for name in canonical]

        configs = []
        built_configs = {}
        if "awq" in canonical:
            awq_entry = get_algorithm_entry("awq")
            awq_cls = cls.get("awq")
            awq_kwargs = _parameter_registry(awq_cls).config_kwargs(args)
            if getattr(awq_cls, "cli_include_common_args", True):
                awq_kwargs.update(common_kwargs)
            if isinstance(awq_entry.config_factory, type):
                awq_config = awq_entry.config_factory(**awq_kwargs)
            elif awq_kwargs:
                awq_config = awq_cls(**awq_kwargs)
            else:
                awq_config = awq_entry.config_factory()
            explicit_opt_rtn = getattr(args, "disable_opt_rtn", None)
            if explicit_opt_rtn is not None:
                awq_config.disable_opt_rtn = explicit_opt_rtn
            built_configs["awq"] = awq_config
        awq_disable_opt_rtn = getattr(built_configs.get("awq"), "disable_opt_rtn", None)
        for name in canonical:
            if name in built_configs:
                configs.append(built_configs[name])
                continue
            entry = get_algorithm_entry(name)
            config_cls = cls.get(name)
            kwargs = _parameter_registry(config_cls).config_kwargs(args)
            if getattr(config_cls, "cli_include_common_args", True):
                kwargs.update(common_kwargs)
            if name == "rtn" and getattr(args, "disable_opt_rtn", None) is None and awq_disable_opt_rtn is not None:
                kwargs["disable_opt_rtn"] = awq_disable_opt_rtn
            if isinstance(entry.config_factory, type):
                config = entry.config_factory(**kwargs)
            elif kwargs:
                config = config_cls(**kwargs)
            else:
                config = entry.config_factory()
            configs.append(config)
        return configs

    @classmethod
    def format_listing(cls) -> str:
        lines = []
        for entry in iter_algorithm_entries():
            if entry.config_factory is None:
                continue
            other = [alias for alias in entry.aliases if alias != entry.name]
            alias_str = f" (aliases: {', '.join(other)})" if other else ""
            lines.append(f"- {entry.name}{alias_str}: {entry.summary}")
        return "\n".join(lines)

    @classmethod
    def format_detail(cls, name: str) -> str:
        canonical = cls.resolve_alias(name)
        if canonical is None:
            supported = [entry.name for entry in iter_algorithm_entries() if entry.config_factory is not None]
            raise ValueError(f"Unknown algorithm '{name}'. Supported: {', '.join(supported)}.")
        entry = get_algorithm_entry(canonical)
        lines = [f"{entry.name}: {entry.summary}"]
        other = [alias for alias in entry.aliases if alias != entry.name]
        if other:
            lines.append(f"Aliases: {', '.join(other)}")
        temp = argparse.ArgumentParser(add_help=False)
        group = temp.add_argument_group(f"Flags for {entry.name}")
        cls.add_group(canonical, group)
        for action in group._group_actions:
            flags = ", ".join(action.option_strings)
            default = f" (default: {action.default})" if action.default is not None else ""
            lines.append(f"  {flags}: {action.help or ''}{default}")
        return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================


def _parse_bool_or_mode(value: str) -> bool | str:
    """Parse AWQ duo_scaling's tri-state: true / false / both."""
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "both":
        return "both"
    raise argparse.ArgumentTypeError("Expected one of: true, false, both")


# ============================================================================
# Algorithm implementations  (auto-registered via __init_subclass__)
# ============================================================================


class AWQ(AlgorithmHandler):
    name = "awq"
    aliases = ("awq",)
    summary = "Activation-Aware Weight Quantization (pre-processing)."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--awq_duo_scaling",
            dest="duo_scaling",
            default=True,
            type=_parse_bool_or_mode,
            metavar="{true,false,both}",
            help="Use activation+weight duo scaling (true/false/both).",
        )
        group.add_argument(
            "--awq_n_grid",
            dest="n_grid",
            default=20,
            type=int,
            help="Number of grid-search points for AWQ scaling ratio.",
        )
        group.add_argument(
            "--awq_seqlen",
            dest="awq_seqlen",
            default=None,
            type=int,
            help=(
                "Maximum sequence length used by AWQ calibration. "
                "This is distinct from the global calibration --seqlen."
            ),
        )
        group.add_argument(
            "--awq_smooth_batch_size",
            dest="awq_smooth_batch_size",
            default=None,
            type=int,
            help="Microbatch size for AWQ parent replay during scale search; <=0 disables microbatching.",
        )
        group.add_argument(
            "--awq_apply_clip",
            dest="awq_apply_clip",
            action="store_true",
            help="Search and hard-clamp per-group AWQ weight clipping after smoothing.",
        )
        group.add_argument(
            "--awq_clip_as_init",
            dest="awq_clip_as_init",
            action="store_true",
            help=(
                "Use the searched AWQ clip to initialize the block quantizer's "
                "weight range instead of hard-clamping (requires --awq_apply_clip)."
            ),
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.awq.config import AWQConfig

        awq_seqlen = getattr(args, "awq_seqlen", None)
        return AWQConfig(
            duo_scaling=getattr(args, "duo_scaling", True),
            n_grid=getattr(args, "n_grid", 20),
            apply_clip=getattr(args, "awq_apply_clip", False),
            clip_as_init=getattr(args, "awq_clip_as_init", False),
            awq_seqlen=512 if awq_seqlen is None else awq_seqlen,
            smooth_batch_size=getattr(args, "awq_smooth_batch_size", None),
            **common_kwargs,
        )


class SVDQuant(AlgorithmHandler):
    name = "svdquant"
    aliases = ("svdquant",)
    summary = "SVD low-rank decomposition before residual quantization."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument("--svdquant-rank", default=32, type=int, help="SVDQuant low-rank size.")
        group.add_argument(
            "--enable-svdquant-smooth",
            dest="svdquant_smooth_enabled",
            default=False,
            action="store_true",
            help="Enable SVDQuant activation-aware smoothing.",
        )
        group.add_argument(
            "--svdquant-smooth-num-grids",
            default=20,
            type=int,
            help="Number of candidates per SVDQuant smooth search grid family.",
        )
        group.add_argument(
            "--svdquant-smooth-max-calibration-calls",
            default=128,
            type=int,
            help="Maximum calibration calls retained per SVDQuant smooth group.",
        )
        group.add_argument(
            "--svdquant-residual-iters",
            default=1,
            type=int,
            help="Number of alternating low-rank and residual quantization iterations.",
        )
        group.add_argument(
            "--enable-svdquant-residual-early-stop",
            dest="svdquant_residual_early_stop",
            default=False,
            action="store_true",
            help="Stop residual iteration when reconstruction error no longer improves.",
        )
        group.add_argument(
            "--svdquant-low-rank-dtype",
            default="bf16",
            choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
            help="Data type for the SVDQuant low-rank branch.",
        )
        group.add_argument(
            "--svdquant-target-modules",
            default=None,
            type=str,
            help="Comma-separated module-name substrings to transform.",
        )
        group.add_argument(
            "--svdquant-exclude-modules",
            default=None,
            type=str,
            help="Comma-separated module-name substrings to exclude.",
        )
        group.add_argument(
            "--svdquant-model-adapter",
            default="auto",
            choices=["auto", "identity", "flux"],
            help="Architecture adapter used by SVDQuant export.",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

        return SVDQuantConfig(
            rank=getattr(args, "svdquant_rank", 32),
            smooth_enabled=getattr(args, "svdquant_smooth_enabled", False),
            smooth_num_grids=getattr(args, "svdquant_smooth_num_grids", 20),
            smooth_max_calibration_calls=getattr(args, "svdquant_smooth_max_calibration_calls", 128),
            residual_iters=getattr(args, "svdquant_residual_iters", 1),
            residual_early_stop=getattr(args, "svdquant_residual_early_stop", False),
            low_rank_dtype=getattr(args, "svdquant_low_rank_dtype", "bf16"),
            target_modules=getattr(args, "svdquant_target_modules", None),
            exclude_modules=getattr(args, "svdquant_exclude_modules", None),
            model_adapter=getattr(args, "svdquant_model_adapter", "auto"),
            **common_kwargs,
        )


class RTN(AlgorithmHandler):
    name = "rtn"
    aliases = ("rtn",)
    summary = "Round-To-Nearest quantization."
    config_factory = None

    def register(self, group) -> None:
        mutex = group.add_mutually_exclusive_group()
        mutex.add_argument(
            "--disable_opt_rtn",
            dest="disable_opt_rtn",
            default=None,
            action="store_const",
            const=True,
            help="Force plain RTN (disable optimized path).",
        )
        mutex.add_argument(
            "--enable_opt_rtn",
            dest="disable_opt_rtn",
            action="store_const",
            const=False,
            help="Force optimized RTN path.",
        )
        group.add_argument(
            "--enable_neuqi",
            dest="enable_neuqi",
            default=False,
            action="store_true",
            help=(
                "Opt into the NeUQI joint (scale, zero-point) search for asymmetric "
                "optimized-RTN (asym_search='neuqi'). Without it, asymmetric layers "
                "use the plain min/max initializer. Ignored for symmetric quantization."
            ),
        )
        group.add_argument(
            "--imatrix_enabled",
            dest="imatrix_enabled",
            default="auto",
            choices=["auto", "true", "false"],
            help=(
                "Force the activation-imatrix weighting for the optimized-RTN search "
                "on/off, overriding the scheme rules (sym int<8 -> on, asym -> off). "
                "'auto' keeps the rules. Forcing it on under --stream_checkpoint also "
                "requires --stream_calibration."
            ),
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        asym_search = "neuqi" if getattr(args, "enable_neuqi", False) else "auto"
        cfg = RTNConfig(
            disable_opt_rtn=getattr(args, "disable_opt_rtn", None),
            asym_search=asym_search,
            **common_kwargs,
        )
        imatrix = getattr(args, "imatrix_enabled", "auto")
        if imatrix == "true":
            cfg.forced_imatrix = True
        elif imatrix == "false":
            cfg.forced_imatrix = False
        return cfg


class AutoRound(AlgorithmHandler):
    name = "auto_round"
    aliases = ("auto_round", "autoround", "sign_round", "signround")
    summary = "SignRound-style iterative block quantization."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--iters", "--iter", default=None, type=int, help="Number of optimization iterations per block."
        )
        group.add_argument("--lr", default=None, type=float, help="Learning rate for rounding optimization.")
        group.add_argument("--minmax_lr", default=None, type=float, help="Learning rate for min-max tuning.")
        group.add_argument("--momentum", default=0.0, type=float, help="Momentum factor for the optimizer.")
        group.add_argument("--nblocks", default=1, type=int, help="Number of blocks to optimize together.")
        minmax_mutex = group.add_mutually_exclusive_group()
        minmax_mutex.add_argument(
            "--enable_minmax_tuning",
            default=True,
            dest="enable_minmax_tuning",
            action="store_true",
            help="Tune weight min/max ranges.",
        )
        minmax_mutex.add_argument(
            "--no-enable_minmax_tuning",
            "--disable_minmax_tuning",
            dest="enable_minmax_tuning",
            action="store_false",
            help="Disable weight min/max tuning.",
        )
        group.add_argument(
            "--enable_norm_bias_tuning",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Tune normalization and bias terms.",
        )
        group.add_argument(
            "--gradient_accumulate_steps", default=1, type=int, help="Gradient accumulation steps per update."
        )
        group.add_argument(
            "--enable_alg_ext",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Enable experimental SignRound extension.",
        )
        group.add_argument(
            "--not_use_best_mse",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Skip restoring best-MSE checkpoint.",
        )
        quanted_input_mutex = group.add_mutually_exclusive_group()
        quanted_input_mutex.add_argument(
            "--enable_quanted_input",
            default=True,
            dest="enable_quanted_input",
            action="store_true",
            help="Consume quantized output of previous blocks.",
        )
        quanted_input_mutex.add_argument(
            "--no-enable_quanted_input",
            "--disable_quanted_input",
            dest="enable_quanted_input",
            action="store_false",
            help="Disable quantized-input propagation across blocks.",
        )
        group.add_argument(
            "--enable_adam",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Use the Adam-based SignRound variant.",
        )
        group.add_argument(
            "--enable_lfq",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Enable last-block LM cross-entropy (LFQ) loss for the final transformer block (experimental).",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        return SignRoundConfig(
            iters=getattr(args, "iters", 200),
            lr=getattr(args, "lr", None),
            minmax_lr=getattr(args, "minmax_lr", None),
            momentum=getattr(args, "momentum", 0.0),
            nblocks=getattr(args, "nblocks", 1),
            enable_minmax_tuning=getattr(args, "enable_minmax_tuning", True),
            enable_norm_bias_tuning=getattr(args, "enable_norm_bias_tuning", False),
            gradient_accumulate_steps=getattr(args, "gradient_accumulate_steps", 1),
            enable_alg_ext=getattr(args, "enable_alg_ext", False),
            not_use_best_mse=getattr(args, "not_use_best_mse", False),
            enable_quanted_input=getattr(args, "enable_quanted_input", True),
            enable_adam=getattr(args, "enable_adam", False),
            enable_lfq=getattr(args, "enable_lfq", False),
            **common_kwargs,
        )


class Hadamard(AlgorithmHandler):
    name = "hadamard"
    aliases = ("hadamard", "random_hadamard", "quarot_hadamard")
    summary = "Hadamard rotation/transform applied before quantization."
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--rotation_type",
            "--rotation-hadamard-type",
            dest="rotation_hadamard_type",
            default=None,
            choices=["hadamard", "random_hadamard", "quarot_hadamard"],
            help="Hadamard transform variant.",
        )
        group.add_argument(
            "--rotation_backend",
            dest="rotation_backend",
            default="auto",
            choices=["auto", "inplace", "transform"],
            help="Rotation backend to use.",
        )
        group.add_argument(
            "--rotation_block_size",
            dest="rotation_block_size",
            default=None,
            type=int,
            help="Grouped Hadamard block size.",
        )
        group.add_argument(
            "--fuse_online_to_weight",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="Fuse online Hadamard rotation into weights.",
        )
        group.add_argument(
            "--allow_online_rotation",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Allow online activation rotation.",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.hadamard.config import RotationConfig

        hadamard_type = getattr(args, "rotation_hadamard_type", None) or "hadamard"
        return RotationConfig(
            hadamard_type=hadamard_type,
            backend=getattr(args, "rotation_backend", "auto"),
            block_size=getattr(args, "rotation_block_size", None),
            fuse_online_to_weight=getattr(args, "fuse_online_to_weight", None),
            allow_online_rotation=getattr(args, "allow_online_rotation", True),
        )


class PreSINQ(AlgorithmHandler):
    name = "presinq"
    aliases = ("presinq", "pre_sinq")
    summary = (
        "Calibration-free function-preserving Sinkhorn column-scale fold applied "
        "before quantization (composable: --algorithm 'presinq,auto_round')."
    )
    config_factory = None

    def register(self, group) -> None:
        group.add_argument(
            "--presinq_group_size",
            dest="presinq_group_size",
            default=None,
            type=int,
            help="PreSINQ tile width in input channels (locked default 64 when unset).",
        )
        group.add_argument(
            "--presinq_n_iter",
            dest="presinq_n_iter",
            default=None,
            type=int,
            help="Sinkhorn iterations per tile (locked default 4 when unset).",
        )
        group.add_argument(
            "--presinq_n_repeat",
            dest="presinq_n_repeat",
            default=None,
            type=int,
            help="Whole-model folding passes (locked default 3 when unset).",
        )
        group.add_argument(
            "--presinq_parallel_folds",
            dest="presinq_parallel_folds",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="Fan per-expert folds out over all visible GPUs (MoE).",
        )

    def build(self, args, common_kwargs: dict[str, Any]):
        from auto_round.algorithms.transforms.presinq.config import PreSINQConfig

        kwargs = {
            key: value
            for key, value in {
                "group_size": getattr(args, "presinq_group_size", None),
                "n_iter": getattr(args, "presinq_n_iter", None),
                "n_repeat": getattr(args, "presinq_n_repeat", None),
                "parallel_folds": getattr(args, "presinq_parallel_folds", None),
            }.items()
            if value is not None
        }
        return PreSINQConfig(**kwargs)


def _register_builtin_algorithm_factories() -> None:
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
    from auto_round.algorithms.transforms.awq.config import AWQConfig
    from auto_round.algorithms.transforms.hadamard.config import RotationConfig
    from auto_round.algorithms.transforms.presinq.config import PreSINQConfig
    from auto_round.algorithms.transforms.svdquant.config import SVDQuantConfig

    register_algorithm("rtn", aliases=("rtn",), config_factory=RTNConfig, cli_handler=RTN, summary=RTN.summary)
    register_algorithm(
        "auto_round",
        aliases=("auto_round", "autoround", "sign_round", "signround"),
        config_factory=SignRoundConfig,
        cli_handler=AutoRound,
        summary=AutoRound.summary,
    )
    register_algorithm("awq", aliases=("awq",), config_factory=AWQConfig, cli_handler=AWQ, summary=AWQ.summary)
    register_algorithm(
        "svdquant",
        aliases=("svdquant",),
        config_factory=SVDQuantConfig,
        cli_handler=SVDQuant,
        summary=SVDQuant.summary,
    )
    register_algorithm(
        "hadamard",
        aliases=("hadamard", "random_hadamard", "quarot_hadamard"),
        config_factory=RotationConfig,
        cli_handler=Hadamard,
        summary=Hadamard.summary,
        alias_factories={
            "random_hadamard": lambda: RotationConfig(hadamard_type="random_hadamard"),
            "quarot_hadamard": lambda: RotationConfig(hadamard_type="quarot_hadamard"),
        },
    )
    register_algorithm(
        "presinq",
        aliases=("presinq", "pre_sinq"),
        config_factory=PreSINQConfig,
        cli_handler=PreSINQ,
        summary=PreSINQ.summary,
    )


_register_builtin_algorithm_factories()
