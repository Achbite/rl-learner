"""Resolve the task-neutral Learner configuration at process start."""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Callable, Mapping

import yaml

from src.contracts.identity import (
    bind_runtime_lineage,
    training_config_digest,
    validate_config,
)


_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)")
_DEVICE = re.compile(r"[A-Za-z0-9_.:-]+")


def _nonempty(name: str, value: str) -> str:
    if value == "":
        raise ValueError(f"{name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must not contain a newline")
    return value


def _integer(name: str, value: str) -> int:
    value = _nonempty(name, value)
    if _INTEGER.fullmatch(value) is None:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _floating(name: str, value: str) -> float:
    value = _nonempty(name, value)
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a floating-point number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _boolean(name: str, value: str) -> bool:
    value = _nonempty(name, value)
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{name} must be exactly true or false")


def _device(name: str, value: str) -> str:
    value = _nonempty(name, value)
    if _DEVICE.fullmatch(value) is None:
        raise ValueError(f"{name} contains invalid device characters")
    return value


_Override = tuple[tuple[str, ...], Callable[[str, str], object]]


ENVIRONMENT_OVERRIDES: dict[str, _Override] = {
    "RL_TRAINING_DEVICE": (("training", "device"), _device),
    "RL_TRAINING_SEED": (("training", "seed"), _integer),
    "RL_PPO_LEARNING_RATE": (("training", "learning_rate"), _floating),
    "RL_PPO_GAMMA": (("training", "gamma"), _floating),
    "RL_PPO_GAE_LAMBDA": (("training", "gae_lambda"), _floating),
    "RL_PPO_CLIP_EPSILON": (("training", "clip_epsilon"), _floating),
    "RL_PPO_VALUE_CLIP_EPSILON": (
        ("training", "value_clip_epsilon"),
        _floating,
    ),
    "RL_PPO_ENTROPY_COEF": (("training", "entropy_coef"), _floating),
    "RL_PPO_VALUE_COEF": (("training", "value_coef"), _floating),
    "RL_PPO_MAX_GRAD_NORM": (("training", "max_grad_norm"), _floating),
    "RL_PPO_N_EPOCHS": (("training", "n_epochs"), _integer),
    "RL_PPO_MINI_BATCH_SIZE": (
        ("training", "mini_batch_size"),
        _integer,
    ),
    "RL_PPO_NORMALIZE_ADVANTAGE": (
        ("training", "normalize_advantage"),
        _boolean,
    ),
    "RL_PPO_MAX_POLICY_LAG": (("training", "max_policy_lag"), _integer),
    "RL_PPO_TRAIN_BATCH_SIZE": (
        ("sample_pool", "train_batch_size"),
        _integer,
    ),
}


_INTERNAL_ENVIRONMENT_OVERRIDES: dict[str, _Override] = {
    "RL_TRAINING_FINALIZE_REQUEST_PATH": (
        ("sample_pool", "finalize_request_path"),
        _nonempty,
    ),
    "RL_TRAINING_FINALIZE_COMPLETE_PATH": (
        ("sample_pool", "finalize_complete_path"),
        _nonempty,
    ),
}

_ALLOWED_NONCONFIG_ENVIRONMENT = {
    "RL_TRAINING_WORKSPACE",
    "RL_MODEL_LINEAGE_ID",
    "RL_METRICS_SOURCE_ID",
    "RL_LEARNER_INSTANCE",
}

_RETIRED_RUNTIME_ENVIRONMENT = {
    "RL_ARCHIVE_INTERVAL_UPDATES",
    "RL_LOCAL_TRAIN_ROOT",
    "RL_METRICS_PORT",
    "RL_MODEL_ARTIFACT_ROOT",
}

CLI_OVERRIDE_FIELDS = {
    ("model", "initial_model_path"),
    ("model_distributor", "host"),
    ("model_distributor", "port"),
    ("aiserver_status", "host"),
    ("aiserver_status", "port"),
    ("dashboard", "server_port"),
}

_PLATFORM_IDENTITY_KEYS = {"task_id", "run_id", "pod_attempt_id"}


def _reject_unknown_environment(environment: Mapping[str, str]) -> None:
    for name in sorted(environment):
        if (
            name in ENVIRONMENT_OVERRIDES
            or name in _INTERNAL_ENVIRONMENT_OVERRIDES
            or name in _ALLOWED_NONCONFIG_ENVIRONMENT
        ):
            continue
        if name in {"RL_TASK_ID", "RL_RUN_ID", "RL_POD_ATTEMPT_ID"}:
            raise ValueError(
                f"platform control identity is not a Learner input: {name}"
            )
        if (
            name in _RETIRED_RUNTIME_ENVIRONMENT
            or name.startswith(
                (
                    "RL_PPO_",
                    "RL_TASK_",
                    "RL_AISERVER_",
                    "RL_MODEL_DISTRIBUTOR_",
                    "RL_SAMPLE_POOL_",
                )
            )
        ):
            raise ValueError(f"unknown component configuration environment: {name}")
        if name.startswith("RL_TRAINING_"):
            raise ValueError(f"unknown Learner environment: {name}")


def _assign(document: dict, path: tuple[str, ...], value: object) -> None:
    cursor = document
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"configuration section is missing: {'.'.join(path[:-1])}")
        cursor = child
    if path[-1] not in cursor:
        raise ValueError(
            f"override has no config field: {'.'.join(path)}"
        )
    cursor[path[-1]] = value


def _normalize_cli_override(path: tuple[str, ...], value: object) -> object:
    field = ".".join(path)
    if path not in CLI_OVERRIDE_FIELDS:
        raise ValueError(f"unsupported Learner CLI override: {field}")
    if path in {
        ("model_distributor", "port"),
        ("aiserver_status", "port"),
        ("dashboard", "server_port"),
    }:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} CLI value must be an integer")
        if not 1 <= value <= 65535:
            raise ValueError(f"{field} CLI value must be in [1, 65535]")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} CLI value must be a string")
    return _nonempty(field, value)


def _resolve_path(
    config_path: Path,
    field: str,
    value: object,
    *,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    selected = Path(_nonempty(field, value)).expanduser()
    if not selected.is_absolute():
        selected = config_path.parent / selected
    return os.path.abspath(selected)


def _reject_platform_config_keys(value: object, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            if name in _PLATFORM_IDENTITY_KEYS:
                location = ".".join((*path, name))
                raise ValueError(
                    f"platform control identity is not a Learner config field: {location}"
                )
            _reject_platform_config_keys(child, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_platform_config_keys(child, (*path, str(index)))


def load_effective_config(
    path: str,
    environment: Mapping[str, str] | None = None,
    cli_overrides: Mapping[tuple[str, ...], object] | None = None,
) -> dict:
    """Apply allowlisted startup overrides and return a validated config."""
    selected_environment = os.environ if environment is None else environment
    _reject_unknown_environment(selected_environment)

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Learner config root must be a mapping")
    _reject_platform_config_keys(loaded)
    config = copy.deepcopy(loaded)

    applied: list[dict[str, object]] = []
    for name, (target, parser) in ENVIRONMENT_OVERRIDES.items():
        if name not in selected_environment:
            continue
        parsed = parser(name, selected_environment[name])
        _assign(config, target, parsed)
        applied.append(
            {
                "environment": name,
                "field": ".".join(target),
                "value": parsed,
            }
        )

    internal_applied: list[dict[str, object]] = []
    for name, (target, parser) in _INTERNAL_ENVIRONMENT_OVERRIDES.items():
        if name not in selected_environment:
            continue
        parsed = parser(name, selected_environment[name])
        _assign(config, target, parsed)
        internal_applied.append(
            {
                "environment": name,
                "field": ".".join(target),
                "value": parsed,
            }
        )

    if "RL_TRAINING_SEED" in selected_environment:
        seed = int(config["training"]["seed"])
        config["model"]["bootstrap_seed"] = seed

    cli_applied: list[dict[str, object]] = []
    for target, raw_value in (cli_overrides or {}).items():
        normalized_target = tuple(target)
        value = _normalize_cli_override(normalized_target, raw_value)
        _assign(config, normalized_target, value)
        cli_applied.append(
            {
                "field": ".".join(normalized_target),
                "value": value,
            }
        )

    config["model"]["local_train_dir"] = _resolve_path(
        config_path,
        "model.local_train_dir",
        config["model"].get("local_train_dir"),
    )
    config["model"]["initial_model_path"] = _resolve_path(
        config_path,
        "model.initial_model_path",
        config["model"].get("initial_model_path"),
        optional=True,
    )
    config["log"]["log_dir"] = _resolve_path(
        config_path,
        "log.log_dir",
        config["log"].get("log_dir"),
    )

    initial_model = config["model"]["initial_model_path"]
    if initial_model is not None:
        initial_path = Path(initial_model)
        if (
            initial_path.name != "SaveModel.onnx"
            or initial_path.is_symlink()
            or not initial_path.is_file()
        ):
            raise ValueError(
                "model.initial_model_path must be an explicit regular, "
                "non-symlink SaveModel.onnx"
            )
        train_root = Path(config["model"]["local_train_dir"]).resolve()
        if initial_path.resolve().is_relative_to(train_root):
            raise ValueError(
                "model.initial_model_path must be outside the fresh training "
                "workspace"
            )

    sample = config.get("sample_pool", {})
    target_samples = int(sample.get("train_batch_size", 0))
    max_fragment_samples = int(sample.get("max_fragment_samples", 0))
    sample["max_train_batch_size"] = (
        target_samples + max_fragment_samples - 1
    )

    config = bind_runtime_lineage(config, selected_environment)
    validate_config(config)
    digest = training_config_digest(config).hex
    config["_effective_config"] = {
        "config_path": str(config_path),
        "environment_overrides": applied,
        "internal_environment_overrides": internal_applied,
        "cli_overrides": cli_applied,
        "training_config_digest": digest,
    }
    return config


def effective_config_log(config: dict) -> dict:
    """Return the bounded, non-secret startup configuration fact."""
    training = config["training"]
    sample = config["sample_pool"]
    model = config["model"]
    return {
        "source": copy.deepcopy(config.get("_effective_config", {})),
        "training": {
            key: training[key]
            for key in (
                "device",
                "seed",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "clip_epsilon",
                "value_clip_epsilon",
                "entropy_coef",
                "value_coef",
                "max_grad_norm",
                "n_epochs",
                "mini_batch_size",
                "normalize_advantage",
                "max_policy_lag",
            )
        },
        "sample_pool": {
            "host": sample["host"],
            "port": sample["port"],
            "train_batch_size": sample["train_batch_size"],
            "max_fragment_samples": sample["max_fragment_samples"],
            "max_train_batch_size": sample["max_train_batch_size"],
        },
        "model": {
            "local_train_dir": model["local_train_dir"],
            "initial_model_path": model["initial_model_path"],
            "archive_interval_updates": model["archive_interval_updates"],
            "publication_retention_steps": model[
                "publication_retention_steps"
            ],
        },
        "model_distributor": copy.deepcopy(config["model_distributor"]),
        "aiserver_status": copy.deepcopy(config["aiserver_status"]),
        "dashboard": copy.deepcopy(config["dashboard"]),
    }
