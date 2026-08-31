"""Build identities from the canonical rl-contracts training descriptor."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from proto import common_pb2, training_pb2


SHA256 = re.compile(r"[a-f0-9]{64}")
CONTRACT_VERSION = "0.15.0"
RUNTIME_LINEAGE_PLACEHOLDER = "__FRESH_INTERNAL_LINEAGE_REQUIRED__"
RUNTIME_LINEAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _require_exact_keys(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


@lru_cache(maxsize=8)
def _load_training_contract(path_text: str) -> dict:
    path = Path(path_text)
    digest_path = path.with_suffix(".sha256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"training contract must be a regular file: {path}")
    if digest_path.is_symlink() or not digest_path.is_file():
        raise ValueError(
            f"training contract digest must be a regular file: {digest_path}"
        )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest_path.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("training contract digest does not match its bytes")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("training contract is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("training contract root must be an object")
    _require_exact_keys(
        document,
        {
            "action_count",
            "action_schema",
            "hidden_dimension",
            "model_architecture_id",
            "observation_dimension",
            "observation_schema",
            "policy",
            "reward_schema",
            "rollout",
            "tensor_dtype",
            "training_contract_id",
        },
        "training contract",
    )
    for name in ("observation_schema", "action_schema", "reward_schema"):
        schema = document[name]
        if not isinstance(schema, dict):
            raise ValueError(f"training contract {name} must be an object")
        _require_exact_keys(
            schema,
            {"canonical_digest", "schema_id", "schema_version"},
            f"training contract {name}",
        )
        if (
            not isinstance(schema["schema_id"], str)
            or not schema["schema_id"]
            or re.search(r"\.v[0-9]+$", schema["schema_id"])
            or isinstance(schema["schema_version"], bool)
            or not isinstance(schema["schema_version"], int)
            or schema["schema_version"] <= 0
            or SHA256.fullmatch(str(schema["canonical_digest"])) is None
        ):
            raise ValueError(f"training contract {name} is invalid")
    policy = document["policy"]
    if not isinstance(policy, dict):
        raise ValueError("training contract policy must be an object")
    _require_exact_keys(
        policy,
        {"distribution_schema_id", "sampling", "temperature"},
        "training contract policy",
    )
    rollout = document["rollout"]
    if not isinstance(rollout, dict):
        raise ValueError("training contract rollout must be an object")
    _require_exact_keys(
        rollout,
        {
            "finite_rule_id",
            "gae_formula_id",
            "model_pin_semantics_id",
            "numeric_dtype",
            "terminal_bootstrap_semantics_id",
            "value_head_abi_id",
            "value_target_formula_id",
        },
        "training contract rollout",
    )
    identifiers = [
        document["training_contract_id"],
        document["model_architecture_id"],
        policy["distribution_schema_id"],
        *rollout.values(),
    ]
    if any(
        not isinstance(value, str)
        or not value
        or re.search(r"\.v[0-9]+$", value)
        for value in identifiers
    ):
        raise ValueError("training contract contains a versioned or empty ID")
    if (
        document["training_contract_id"] != "maze.training"
        or document["model_architecture_id"] != "maze.mlp-17x64x64"
        or document["tensor_dtype"] != "float32"
        or document["observation_dimension"] != 17
        or document["action_count"] != 9
        or document["hidden_dimension"] != 64
        or policy != {
            "distribution_schema_id": "categorical.logits",
            "sampling": "stochastic",
            "temperature": 1.0,
        }
        or rollout["numeric_dtype"] != "float32"
    ):
        raise ValueError("training contract is unsupported by this Learner")
    result = copy.deepcopy(document)
    result["canonical_digest"] = digest
    return result


def training_contract(config: dict) -> dict:
    path = str(config["contract"]["training_contract_path"])
    return copy.deepcopy(_load_training_contract(path))


def _digest(value: str) -> common_pb2.ContentDigest:
    if SHA256.fullmatch(str(value)) is None:
        raise ValueError("digest must be lower-case SHA-256")
    return common_pb2.ContentDigest(
        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
        hex=str(value),
    )


def content_digest(value: str) -> common_pb2.ContentDigest:
    return _digest(value)


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


def contract_identity(config: dict) -> common_pb2.ContractIdentity:
    contract = config["contract"]
    if (
        contract.get("package_name") != "rl-contracts"
        or contract.get("package_version") != CONTRACT_VERSION
        or not contract.get("platform")
        or SHA256.fullmatch(str(contract.get("generator_identity", "")))
        is None
    ):
        raise ValueError("contract identity is not the selected 0.15.0 artifact")
    return common_pb2.ContractIdentity(
        package_name=contract["package_name"],
        package_version=contract["package_version"],
        source_digest=_digest(contract["source_digest"]),
        artifact_digest=_digest(contract["artifact_digest"]),
        platform=contract["platform"],
        generator_identity=contract["generator_identity"],
    )


def schema_identity(document: dict) -> common_pb2.SchemaIdentity:
    if (
        not document.get("schema_id")
        or int(document.get("schema_version", 0)) <= 0
    ):
        raise ValueError("schema identity is incomplete")
    return common_pb2.SchemaIdentity(
        schema_id=str(document["schema_id"]),
        schema_version=int(document["schema_version"]),
        canonical_digest=_digest(document["canonical_digest"]),
    )


def training_contract_digest(config: dict) -> common_pb2.ContentDigest:
    """Return the sole wire identity for the canonical training descriptor."""
    return _digest(training_contract(config)["canonical_digest"])


def rollout_estimator_profile(
    config: dict,
) -> training_pb2.RolloutEstimatorProfile:
    training = config["training"]
    profile = training_pb2.RolloutEstimatorProfile(
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        tmax=int(training["tmax"]),
    )
    digest = hashlib.sha256(
        profile.SerializeToString(deterministic=True)
    ).hexdigest()
    profile.profile_digest.CopyFrom(_digest(digest))
    return profile


def canonical_config_digest(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def training_config_document(config: dict) -> dict:
    """Return the exact task-neutral configuration bound to model identity."""
    contract = training_contract(config)
    training = config["training"]
    model = config["model"]
    return {
        "training_contract_digest": contract["canonical_digest"],
        "training": {
            key: training[key]
            for key in (
                "device",
                "seed",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "tmax",
                "clip_epsilon",
                "value_clip_epsilon",
                "entropy_coef",
                "value_coef",
                "max_grad_norm",
                "n_epochs",
                "train_batch_size",
                "mini_batch_size",
                "normalize_advantage",
            )
        },
        "model": {
            "bootstrap_seed": model["bootstrap_seed"],
        },
    }


def training_config_digest(config: dict) -> common_pb2.ContentDigest:
    actual = canonical_config_digest(training_config_document(config))
    return _digest(actual)


def model_identity_document(message: training_pb2.ModelIdentity) -> dict:
    has_step = message.HasField("model_step")
    has_any_identity = bool(
        message.model_lineage_id
        or has_step
        or message.artifact_digest.hex
        or message.manifest_digest.hex
    )
    if not has_any_identity:
        return {}
    if not (
        message.model_lineage_id
        and has_step
        and message.artifact_digest.hex
        and message.manifest_digest.hex
    ):
        raise ValueError("model identity is partially populated")
    return {
        "model_lineage_id": message.model_lineage_id,
        "model_step": int(message.model_step),
        "artifact_digest": message.artifact_digest.hex,
        "manifest_digest": message.manifest_digest.hex,
    }


def schema_document(message: common_pb2.SchemaIdentity) -> dict:
    return {
        "schema_id": message.schema_id,
        "schema_version": int(message.schema_version),
        "canonical_digest": message.canonical_digest.hex,
    }


def contract_document(message: common_pb2.ContractIdentity) -> dict:
    return {
        "package_name": message.package_name,
        "package_version": message.package_version,
        "source_digest": message.source_digest.hex,
        "artifact_digest": message.artifact_digest.hex,
        "platform": message.platform,
        "generator_identity": message.generator_identity,
    }


def finalize_manifest(
    message: training_pb2.ModelArtifactManifest,
) -> training_pb2.ModelArtifactManifest:
    """Return a manifest whose identity digest covers canonical protobuf bytes."""
    result = training_pb2.ModelArtifactManifest()
    result.CopyFrom(message)
    result.identity.ClearField("manifest_digest")
    digest = hashlib.sha256(
        result.SerializeToString(deterministic=True)
    ).hexdigest()
    result.identity.manifest_digest.CopyFrom(_digest(digest))
    return result


def validate_manifest_digest(
    message: training_pb2.ModelArtifactManifest,
) -> None:
    expected = finalize_manifest(message).identity.manifest_digest
    if (
        message.identity.manifest_digest.algorithm
        != common_pb2.DIGEST_ALGORITHM_SHA256
        or message.identity.manifest_digest.hex != expected.hex
    ):
        raise ValueError("model manifest digest is invalid")


def write_manifest_file(
    path: Path, message: training_pb2.ModelArtifactManifest
) -> None:
    if path.name != "manifest.pb":
        raise ValueError("model manifest must use the canonical protobuf name")
    validate_manifest_digest(message)
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
    validate_manifest_digest(message)
    return message


def validate_config(config: dict) -> None:
    required_sections = {
        "training",
        "model",
        "identity",
        "contract",
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
        "contract": {
            "package_name", "package_version", "source_digest",
            "artifact_digest", "platform", "generator_identity",
            "training_contract_path",
        },
        "identity": {"model_lineage_id"},
        "training": {
            "device", "seed", "learning_rate", "gamma", "gae_lambda",
            "tmax", "clip_epsilon", "value_clip_epsilon",
            "entropy_coef", "value_coef", "max_grad_norm", "n_epochs",
            "train_batch_size", "mini_batch_size", "normalize_advantage",
        },
        "model": {
            "initial_model_path", "local_train_dir",
            "archive_interval_updates", "publication_retention_steps",
            "bootstrap_seed",
        },
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
        _require_exact_keys(
            effective,
            {
                "config_path", "environment_overrides",
                "internal_environment_overrides", "cli_overrides",
                "training_config_digest",
            },
            "_effective_config",
        )
    contract_identity(config)
    lineage = str(config["identity"].get("model_lineage_id", ""))
    if lineage != RUNTIME_LINEAGE_PLACEHOLDER and (
        not lineage or RUNTIME_LINEAGE.fullmatch(lineage) is None
    ):
        raise ValueError("identity.model_lineage_id is invalid")
    contract = training_contract(config)
    training_contract_digest(config)
    training_config_digest(config)
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
        int(model["archive_interval_updates"]) <= 0
        or int(model["publication_retention_steps"]) <= 0
    ):
        raise ValueError("model publication parameters are invalid")
    if contract["policy"]["sampling"] != "stochastic":
        raise ValueError("training contract policy sampling is unsupported")
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
    profile = rollout_estimator_profile(config)
    if (
        isinstance(training["normalize_advantage"], bool) is False
        or int(training["seed"]) < 0
        or int(model["bootstrap_seed"]) < 0
        or int(training["n_epochs"]) <= 0
        or int(training["train_batch_size"]) <= 0
        or int(training["mini_batch_size"]) <= 0
        or int(training["tmax"]) <= 0
        or not profile.profile_digest.hex
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
