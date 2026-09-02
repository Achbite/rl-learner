"""Validate Learner-owned runtime, service and model identities."""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Mapping

from proto import common_pb2, training_pb2


RUNTIME_LINEAGE_PLACEHOLDER = "__FRESH_INTERNAL_LINEAGE_REQUIRED__"
RUNTIME_LINEAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _require_exact_keys(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def service_identity(
    component: str, instance_id: str, lifecycle_epoch: int = 1
) -> common_pb2.ServiceInstanceIdentity:
    if not component or not instance_id or lifecycle_epoch <= 0:
        raise ValueError("service identity is incomplete")
    return common_pb2.ServiceInstanceIdentity(
        component=component,
        instance_id=instance_id,
        lifecycle_epoch=lifecycle_epoch,
    )


def bind_runtime_lineage(
    config: dict,
    environment: Mapping[str, str] | None = None,
) -> dict:
    """Bind the fresh internal lineage without mutating the template."""
    result = copy.deepcopy(config)
    configured = str(result.get("identity", {}).get("model_lineage_id", ""))
    selected_environment = os.environ if environment is None else environment
    lineage = str(selected_environment.get("RL_MODEL_LINEAGE_ID", configured))
    if (
        not lineage
        or lineage == RUNTIME_LINEAGE_PLACEHOLDER
        or RUNTIME_LINEAGE.fullmatch(lineage) is None
    ):
        raise ValueError(
            "the launcher must bind a fresh internal model lineage"
        )
    result["identity"]["model_lineage_id"] = lineage
    return result


def model_identity_document(message: training_pb2.ModelIdentity) -> dict:
    has_step = message.HasField("model_step")
    has_any_identity = bool(message.model_lineage_id or has_step)
    if not has_any_identity:
        return {}
    if not (
        message.model_lineage_id and has_step
    ):
        raise ValueError("model identity is partially populated")
    return {
        "model_lineage_id": message.model_lineage_id,
        "model_step": int(message.model_step),
    }


def validate_model_manifest(
    message: training_pb2.ModelArtifactManifest,
) -> None:
    if (
        not message.identity.model_lineage_id
        or not message.identity.HasField("model_step")
        or int(message.size_bytes) <= 0
        or int(message.published_at_unix_ms) <= 0
    ):
        raise ValueError("model manifest is incomplete")


def write_manifest_file(
    path: Path, message: training_pb2.ModelArtifactManifest
) -> None:
    if path.name != "manifest.pb":
        raise ValueError("model manifest must use the canonical protobuf name")
    validate_model_manifest(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("model manifest must not be a symbolic link")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = message.SerializeToString(deterministic=True)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_manifest_file(path: Path) -> training_pb2.ModelArtifactManifest:
    if path.name != "manifest.pb":
        raise ValueError("model manifest must use the canonical protobuf name")
    if path.is_symlink() or not path.is_file():
        raise ValueError("model manifest must be a regular file")
    payload = path.read_bytes()
    message = training_pb2.ModelArtifactManifest()
    try:
        message.ParseFromString(payload)
    except Exception as error:
        raise ValueError("model manifest is not valid protobuf") from error
    if message.SerializeToString(deterministic=True) != payload:
        raise ValueError("model manifest bytes are not canonical")
    validate_model_manifest(message)
    return message


def validate_config(config: dict) -> None:
    required_sections = {
        "training",
        "model",
        "policy",
        "identity",
        "sample_pool",
        "model_distributor",
        "aiserver_status",
        "metric_events",
        "dashboard",
        "log",
    }
    root_keys = set(config)
    if root_keys not in (
        required_sections,
        required_sections | {"_effective_config"},
    ):
        raise ValueError(
            "Learner config keys differ: "
            f"missing={sorted(required_sections - root_keys)} "
            f"unknown={sorted(root_keys - required_sections - {'_effective_config'})}"
        )
    expected_keys = {
        "identity": {"model_lineage_id"},
        "training": {
            "device", "seed", "learning_rate", "gamma", "gae_lambda",
            "tmax", "clip_epsilon", "value_clip_epsilon",
            "entropy_coef", "value_coef", "max_grad_norm", "n_epochs",
            "train_batch_size", "mini_batch_size", "normalize_advantage",
        },
        "model": {
            "observation_dimension", "action_count", "hidden_dimension",
            "initial_model_path", "local_train_dir",
            "archive_interval_updates", "publication_retention_steps",
            "bootstrap_seed",
        },
        "policy": {"action_mask_mode"},
        "sample_pool": {
            "host", "port", "get_timeout_ms", "lease_timeout_ms",
            "finalize_request_path", "finalize_complete_path",
            "shutdown_drain_timeout_ms",
        },
        "model_distributor": {"host", "port"},
        "aiserver_status": {"host", "port", "initial_model_ack_timeout_sec"},
        "metric_events": {
            "server_enabled", "server_port", "aiserver_relay_enabled",
        },
        "dashboard": {"enabled", "server_port", "backend"},
        "log": {"console_level", "file_level", "log_dir"},
    }
    for section_name, keys in expected_keys.items():
        section = config[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        _require_exact_keys(section, keys, section_name)
    if "_effective_config" in config:
        effective = config["_effective_config"]
        if not isinstance(effective, dict):
            raise ValueError("_effective_config must be a mapping")
        required_effective = {
            "config_path", "environment_overrides",
            "internal_environment_overrides", "cli_overrides",
        }
        optional_effective = {"platform_environment"}
        actual_effective = set(effective)
        if (
            not required_effective.issubset(actual_effective)
            or actual_effective - required_effective - optional_effective
        ):
            raise ValueError("_effective_config keys differ")
    lineage = str(config["identity"].get("model_lineage_id", ""))
    if lineage != RUNTIME_LINEAGE_PLACEHOLDER and (
        not lineage or RUNTIME_LINEAGE.fullmatch(lineage) is None
    ):
        raise ValueError("identity.model_lineage_id is invalid")
    model = config["model"]
    training = config["training"]
    if "initial_model_path" not in model:
        raise ValueError("model.initial_model_path default is required")
    initial_model_path = model["initial_model_path"]
    if initial_model_path is not None and (
        not isinstance(initial_model_path, str) or not initial_model_path
    ):
        raise ValueError("model.initial_model_path must be null or a path")
    if not isinstance(model.get("local_train_dir"), str) or not model[
        "local_train_dir"
    ]:
        raise ValueError("model.local_train_dir must be configured")
    if (
        int(model["observation_dimension"]) <= 0
        or int(model["action_count"]) <= 0
        or int(model["hidden_dimension"]) <= 0
        or int(model["archive_interval_updates"]) <= 0
        or int(model["publication_retention_steps"]) <= 0
    ):
        raise ValueError("model publication parameters are invalid")
    if config["policy"]["action_mask_mode"] not in {"disabled", "required"}:
        raise ValueError("policy.action_mask_mode is invalid")
    finite_training_values = (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "value_clip_epsilon",
        "entropy_coef",
        "value_coef",
        "max_grad_norm",
    )
    for name in finite_training_values:
        value = float(training[name])
        if not math.isfinite(value):
            raise ValueError(f"training.{name} must be finite")
    if (
        float(training["learning_rate"]) <= 0.0
        or not 0.0 <= float(training["gamma"]) <= 1.0
        or not 0.0 <= float(training["gae_lambda"]) <= 1.0
        or float(training["clip_epsilon"]) <= 0.0
        or float(training["value_clip_epsilon"]) <= 0.0
        or float(training["entropy_coef"]) < 0.0
        or float(training["value_coef"]) < 0.0
        or float(training["max_grad_norm"]) <= 0.0
    ):
        raise ValueError("training parameters contradict PPO execution")
    if (
        isinstance(training["normalize_advantage"], bool) is False
        or int(training["seed"]) < 0
        or int(model["bootstrap_seed"]) < 0
        or int(training["n_epochs"]) <= 0
        or int(training["train_batch_size"]) <= 0
        or int(training["mini_batch_size"]) <= 0
        or int(training["tmax"]) <= 0
        or int(config["sample_pool"]["get_timeout_ms"]) <= 0
        or int(config["sample_pool"]["lease_timeout_ms"]) <= 0
        or int(config["sample_pool"]["shutdown_drain_timeout_ms"])
        <= 0
        or not str(
            config["sample_pool"]["finalize_request_path"]
        ).startswith("/")
        or not str(
            config["sample_pool"]["finalize_complete_path"]
        ).startswith("/")
        or config["sample_pool"]["finalize_request_path"]
        == config["sample_pool"]["finalize_complete_path"]
    ):
        raise ValueError("integer training parameters are invalid")
    for section_name in (
        "sample_pool",
        "model_distributor",
        "aiserver_status",
    ):
        section = config[section_name]
        if not isinstance(section.get("host"), str) or not section["host"]:
            raise ValueError(f"{section_name}.host must be configured")
        port = section.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{section_name}.port must be in [1, 65535]")
    initial_ack_timeout = config["aiserver_status"][
        "initial_model_ack_timeout_sec"
    ]
    if initial_ack_timeout is not None and (
        isinstance(initial_ack_timeout, bool)
        or not isinstance(initial_ack_timeout, (int, float))
        or not math.isfinite(float(initial_ack_timeout))
        or float(initial_ack_timeout) <= 0.0
    ):
        raise ValueError(
            "aiserver_status.initial_model_ack_timeout_sec must be null or "
            "a positive finite number"
        )
    metric_events = config["metric_events"]
    metric_server_port = metric_events.get("server_port")
    if (
        not isinstance(metric_events.get("server_enabled"), bool)
        or not isinstance(
            metric_events.get("aiserver_relay_enabled"), bool
        )
        or isinstance(metric_server_port, bool)
        or not isinstance(metric_server_port, int)
        or not 1 <= metric_server_port <= 65535
    ):
        raise ValueError("metric_events configuration is invalid")
    dashboard = config["dashboard"]
    dashboard_port = dashboard.get("server_port")
    if (
        not isinstance(dashboard.get("enabled"), bool)
        or isinstance(dashboard_port, bool)
        or not isinstance(dashboard_port, int)
        or not 1 <= dashboard_port <= 65535
        or not isinstance(dashboard.get("backend"), str)
        or not dashboard["backend"]
    ):
        raise ValueError("dashboard configuration is invalid")
