"""Build and validate the exact rl-contracts 0.13.0 training identities."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from typing import Any, Mapping

from proto import common_pb2, training_pb2


SHA256 = re.compile(r"[a-f0-9]{64}")
CONTRACT_VERSION = "0.13.0"
RUNTIME_LINEAGE_PLACEHOLDER = "__FRESH_INTERNAL_LINEAGE_REQUIRED__"
RUNTIME_LINEAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REWARD_SCHEMA_ID = "maze.reward.v4"
REWARD_SCHEMA_DIGEST = (
    "ed284084b79413473d5053b6d3f69320d2a4639c81451ba598ca45ac8ce15929"
)
TRAINING_SEMANTICS_DIGEST = (
    "6cd834542f8263135b4bfd069f372ddfdb99334060d305f58b00ce56eea10b4c"
)


def _digest(value: str) -> common_pb2.ContentDigest:
    if SHA256.fullmatch(str(value)) is None:
        raise ValueError("digest must be lower-case SHA-256")
    return common_pb2.ContentDigest(
        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
        hex=str(value),
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


def contract_identity(config: dict) -> common_pb2.ContractIdentity:
    contract = config["contract"]
    if (
        contract.get("package_name") != "rl-contracts"
        or contract.get("package_version") != CONTRACT_VERSION
        or not contract.get("platform")
        or SHA256.fullmatch(str(contract.get("generator_identity", "")))
        is None
    ):
        raise ValueError("contract identity is not the selected 0.13.0 artifact")
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


def training_semantics(config: dict) -> training_pb2.TrainingSemanticsIdentity:
    semantics = config["training_semantics"]
    result = training_pb2.TrainingSemanticsIdentity(
        training_contract_id=str(semantics["training_contract_id"]),
        observation_schema=schema_identity(semantics["observation_schema"]),
        action_schema=schema_identity(semantics["action_schema"]),
        reward_schema=schema_identity(semantics["reward_schema"]),
        policy_distribution_schema_id=str(
            semantics["policy_distribution_schema_id"]
        ),
        model_architecture_id=str(semantics["model_architecture_id"]),
        semantics_digest=_digest(semantics["semantics_digest"]),
    )
    if not all(
        (
            result.training_contract_id,
            result.policy_distribution_schema_id,
            result.model_architecture_id,
        )
    ):
        raise ValueError("training semantics is incomplete")
    return result


def policy_spec_digest(config: dict) -> common_pb2.ContentDigest:
    return _digest(config["policy"]["policy_spec_digest"])


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
    semantics = config["training_semantics"]
    training = config["training"]
    model = config["model"]
    return {
        "training_semantics_digest": semantics["semantics_digest"],
        "policy_spec_digest": config["policy"]["policy_spec_digest"],
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
        "model": {
            key: model[key]
            for key in (
                "obs_dim",
                "action_dim",
                "hidden_dim",
                "bootstrap_seed",
                "tensor_dtype",
            )
        },
        "sample": {
            "train_batch_size": config["sample_pool"][
                "train_batch_size"
            ]
        },
    }


def training_config_digest(config: dict) -> common_pb2.ContentDigest:
    actual = canonical_config_digest(training_config_document(config))
    return _digest(actual)


def model_identity(document: dict) -> training_pb2.ModelIdentity:
    identity = document["identity"]
    if not identity.get("model_lineage_id"):
        raise ValueError("model lineage is required")
    if "model_version" in identity:
        raise ValueError("legacy model_version is not accepted")
    model_step = identity.get("model_step")
    if (
        isinstance(model_step, bool)
        or not isinstance(model_step, int)
        or model_step < 0
        or model_step > (1 << 64) - 1
    ):
        raise ValueError("model_step must be a uint64 integer")
    return training_pb2.ModelIdentity(
        model_lineage_id=str(identity["model_lineage_id"]),
        model_step=model_step,
        artifact_digest=_digest(identity["artifact_digest"]),
        manifest_digest=_digest(identity["manifest_digest"]),
    )


def model_identity_document(message: training_pb2.ModelIdentity) -> dict:
    return {
        "model_lineage_id": message.model_lineage_id,
        "model_step": int(message.model_step),
        "artifact_digest": message.artifact_digest.hex,
        "manifest_digest": message.manifest_digest.hex,
    }


def manifest_message(document: dict) -> training_pb2.ModelArtifactManifest:
    if int(document["manifest_schema_version"]) != 2:
        raise ValueError("training manifest_schema_version must be 2")
    if "model_version" in document or "model_version" in document.get(
        "identity", {}
    ):
        raise ValueError("legacy model_version is not accepted")
    message = training_pb2.ModelArtifactManifest(
        manifest_schema_version=int(document["manifest_schema_version"]),
        contract=contract_identity({"contract": document["contract"]}),
        identity=model_identity(document),
        observation_schema=schema_identity(document["observation_schema"]),
        action_schema=schema_identity(document["action_schema"]),
        model_architecture_id=str(document["model_architecture_id"]),
        tensor_dtype=str(document["tensor_dtype"]),
        artifact_uri=str(document["artifact_uri"]),
        model_file=str(document["model_file"]),
        size_bytes=int(document["size_bytes"]),
        seed=int(document["seed"]),
        train_updates=int(document["train_updates"]),
        trained_samples=int(document["trained_samples"]),
        training_config_digest=_digest(document["training_config_digest"]),
        training_semantics=_semantics_from_document(
            document["training_semantics"]
        ),
        published_at_unix_ms=int(document["published_at_unix_ms"]),
        ready=bool(document["ready"]),
    )
    message.input_shape.extend(int(value) for value in document["input_shape"])
    message.action_shape.extend(
        int(value) for value in document["action_shape"]
    )
    message.value_shape.extend(int(value) for value in document["value_shape"])
    if (
        not message.identity.HasField("model_step")
        or int(message.identity.model_step) != int(message.train_updates)
    ):
        raise ValueError("model_step must equal train_updates")
    return message


def _semantics_from_document(
    document: dict,
) -> training_pb2.TrainingSemanticsIdentity:
    return training_pb2.TrainingSemanticsIdentity(
        training_contract_id=str(document["training_contract_id"]),
        observation_schema=schema_identity(document["observation_schema"]),
        action_schema=schema_identity(document["action_schema"]),
        reward_schema=schema_identity(document["reward_schema"]),
        policy_distribution_schema_id=str(
            document["policy_distribution_schema_id"]
        ),
        model_architecture_id=str(document["model_architecture_id"]),
        semantics_digest=_digest(document["semantics_digest"]),
    )


def schema_document(message: common_pb2.SchemaIdentity) -> dict:
    return {
        "schema_id": message.schema_id,
        "schema_version": int(message.schema_version),
        "canonical_digest": message.canonical_digest.hex,
    }


def semantics_document(
    message: training_pb2.TrainingSemanticsIdentity,
) -> dict:
    return {
        "training_contract_id": message.training_contract_id,
        "observation_schema": schema_document(message.observation_schema),
        "action_schema": schema_document(message.action_schema),
        "reward_schema": schema_document(message.reward_schema),
        "policy_distribution_schema_id": (
            message.policy_distribution_schema_id
        ),
        "model_architecture_id": message.model_architecture_id,
        "semantics_digest": message.semantics_digest.hex,
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


def finalize_manifest_digest(document: dict) -> dict:
    result = copy.deepcopy(document)
    result["identity"]["manifest_digest"] = "0" * 64
    message = manifest_message(result)
    message.identity.ClearField("manifest_digest")
    digest = hashlib.sha256(
        message.SerializeToString(deterministic=True)
    ).hexdigest()
    result["identity"]["manifest_digest"] = digest
    return result


def validate_config(config: dict) -> None:
    required_sections = {
        "training",
        "model",
        "policy",
        "identity",
        "training_semantics",
        "contract",
        "sample_pool",
        "model_distributor",
        "aiserver_status",
        "dashboard",
        "log",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    contract_identity(config)
    lineage = str(config["identity"].get("model_lineage_id", ""))
    if lineage != RUNTIME_LINEAGE_PLACEHOLDER and (
        not lineage or RUNTIME_LINEAGE.fullmatch(lineage) is None
    ):
        raise ValueError("identity.model_lineage_id is invalid")
    semantics = training_semantics(config)
    policy_spec_digest(config)
    actual_training_digest = training_config_digest(config).hex
    expected_training_digest = config["identity"].get(
        "expected_training_config_digest"
    )
    if expected_training_digest is not None and (
        _digest(expected_training_digest).hex != actual_training_digest
    ):
        raise ValueError(
            "expected_training_config_digest does not match the effective "
            "training configuration"
        )
    model = config["model"]
    training = config["training"]
    policy = config["policy"]
    retired_model_fields = {
        "archive_on_graceful_shutdown",
        "initial_checkpoint",
        "initial_model_dir",
        "serving_retention_versions",
        "publication_retention_versions",
    }
    present_retired_fields = sorted(retired_model_fields.intersection(model))
    if present_retired_fields:
        raise ValueError(
            "retired model publication fields are not allowed: "
            + ", ".join(present_retired_fields)
        )
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
        or int(model["publication_retention_steps"]) != 101
    ):
        raise ValueError("model publication parameters are invalid")
    if (
        int(model["obs_dim"]) != 17
        or int(model["action_dim"]) != 9
        or int(model["hidden_dim"]) != 64
        or semantics.observation_schema.schema_id != "maze.observation.v3"
        or semantics.action_schema.schema_id != "maze.action.v1"
        or semantics.reward_schema.schema_id != REWARD_SCHEMA_ID
        or semantics.reward_schema.schema_version != 1
        or semantics.reward_schema.canonical_digest.hex
        != REWARD_SCHEMA_DIGEST
        or semantics.semantics_digest.hex != TRAINING_SEMANTICS_DIGEST
        or semantics.model_architecture_id != "maze.mlp-17x64x64.v1"
    ):
        raise ValueError(
            "model/schema does not match the locked Reward V4 17x64x64 contract"
        )
    if (
        policy.get("distribution_schema_id")
        != semantics.policy_distribution_schema_id
        or policy.get("training_sampling") != "stochastic"
        or float(policy.get("training_temperature")) != 1.0
    ):
        raise ValueError("policy sampling contract is invalid")
    numeric_ranges: list[tuple[str, float, float, bool]] = [
        ("learning_rate", 0.0, 1.0, False),
        ("gamma", 0.0, 1.0, True),
        ("gae_lambda", 0.0, 1.0, True),
        ("clip_epsilon", 0.0, 1.0, False),
        ("value_clip_epsilon", 0.0, 1.0, False),
        ("entropy_coef", 0.0, 10.0, True),
        ("value_coef", 0.0, 10.0, True),
        ("max_grad_norm", 0.0, 100.0, False),
    ]
    for name, minimum, maximum, include_minimum in numeric_ranges:
        value = float(training[name])
        if not math.isfinite(value) or value > maximum or (
            value < minimum if include_minimum else value <= minimum
        ):
            raise ValueError(f"training.{name} is outside the locked range")
    sample = config["sample_pool"]
    train_batch_size = int(sample["train_batch_size"])
    max_fragment_samples = int(sample["max_fragment_samples"])
    max_train_batch_size = int(sample["max_train_batch_size"])
    if (
        isinstance(training["normalize_advantage"], bool) is False
        or int(training["seed"]) < 0
        or int(model["bootstrap_seed"]) < 0
        or int(training["seed"]) != int(model["bootstrap_seed"])
        or int(training["n_epochs"]) <= 0
        or int(training["mini_batch_size"]) <= 0
        or int(training["max_policy_lag"]) < 0
        or int(model["publication_retention_steps"])
        < int(training["max_policy_lag"]) + 1
        or train_batch_size <= 0
        or max_fragment_samples <= 0
        or max_train_batch_size
        != train_batch_size + max_fragment_samples - 1
        or int(config["sample_pool"]["max_sample_age_ms"]) <= 0
        or int(config["sample_pool"]["finalize_drain_timeout_ms"])
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
    if "startup_timeout_sec" in config["model_distributor"]:
        raise ValueError(
            "model_distributor.startup_timeout_sec is retired; initial model "
            "ACK waiting belongs to aiserver_status"
        )
    if "initial_model_ack_timeout_sec" not in config["aiserver_status"]:
        raise ValueError(
            "aiserver_status.initial_model_ack_timeout_sec default is required"
        )
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
