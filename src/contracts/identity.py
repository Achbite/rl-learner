"""Build and validate the exact rl-contracts 0.9.1 training identities."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

from proto import common_pb2, training_pb2


SHA256 = re.compile(r"[a-f0-9]{64}")
CONTRACT_VERSION = "0.9.1"
REWARD_SCHEMA_ID = "maze.reward.v4"
REWARD_SCHEMA_DIGEST = (
    "ed284084b79413473d5053b6d3f69320d2a4639c81451ba598ca45ac8ce15929"
)
TRAINING_SEMANTICS_DIGEST = (
    "6cd834542f8263135b4bfd069f372ddfdb99334060d305f58b00ce56eea10b4c"
)
TRAINING_CONFIG_DIGEST = (
    "b8a98bd14abc5f09e57c65516ff1eae8222b9515b058d76c34af4a88dee7551f"
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
        raise ValueError("contract identity is not the selected 0.9.1 artifact")
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


def training_config_digest(config: dict) -> common_pb2.ContentDigest:
    return _digest(config["identity"]["training_config_digest"])


def model_identity(document: dict) -> training_pb2.ModelIdentity:
    identity = document["identity"]
    if not identity.get("model_lineage_id"):
        raise ValueError("model lineage is required")
    return training_pb2.ModelIdentity(
        model_lineage_id=str(identity["model_lineage_id"]),
        model_version=int(identity["model_version"]),
        artifact_digest=_digest(identity["artifact_digest"]),
        manifest_digest=_digest(identity["manifest_digest"]),
    )


def model_identity_document(message: training_pb2.ModelIdentity) -> dict:
    return {
        "model_lineage_id": message.model_lineage_id,
        "model_version": int(message.model_version),
        "artifact_digest": message.artifact_digest.hex,
        "manifest_digest": message.manifest_digest.hex,
    }


def manifest_message(document: dict) -> training_pb2.ModelArtifactManifest:
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
        "sample_distributor",
        "model_distributor",
        "aiserver_status",
        "dashboard",
        "log",
    }
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    contract_identity(config)
    semantics = training_semantics(config)
    policy_spec_digest(config)
    training_config_digest(config)
    model = config["model"]
    training = config["training"]
    policy = config["policy"]
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
        or training_config_digest(config).hex != TRAINING_CONFIG_DIGEST
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
        or policy.get("evaluation_sampling") != "argmax"
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
        if value > maximum or (
            value < minimum if include_minimum else value <= minimum
        ):
            raise ValueError(f"training.{name} is outside the locked range")
    if (
        int(training["n_epochs"]) <= 0
        or int(training["mini_batch_size"]) <= 0
        or int(training["max_policy_lag"]) < 0
        or int(config["sample_distributor"]["train_batch_size"]) != 512
        or int(config["sample_distributor"]["max_train_batch_size"])
        < int(config["sample_distributor"]["train_batch_size"])
        or int(config["sample_distributor"]["max_sample_age_ms"]) <= 0
    ):
        raise ValueError("integer training parameters are invalid")


def canonical_config_digest(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
