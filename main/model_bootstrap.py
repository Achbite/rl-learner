"""Export a deterministic untrained model with the current wire manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.training_runtime import atomic_write_json, sha256_file
from src.contracts.identity import (
    contract_document,
    contract_identity,
    finalize_manifest_digest,
    schema_document,
    semantics_document,
    training_config_digest,
    training_semantics,
    validate_config,
)
from src.log.logger import setup_logger
from src.training.ppo_trainer import PPOTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    validate_config(config)
    return config


def export_initial_model(
    config: dict,
    logger,
    output_root: str,
    artifact_uri_prefix: str = "",
) -> dict:
    validate_config(config)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trainer = PPOTrainer(config)
    model_file = "SaveModel.onnx"
    model_path = root / model_file
    temporary = root / f".{model_file}.{os.getpid()}.tmp"
    trainer.export_onnx(str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, model_path)
    semantics = training_semantics(config)
    artifact_uri = (
        f"{artifact_uri_prefix.rstrip('/')}/{model_file}"
        if artifact_uri_prefix
        else model_path.as_uri()
    )
    document = {
        "manifest_schema_version": 1,
        "contract": contract_document(contract_identity(config)),
        "identity": {
            "model_lineage_id": config["identity"]["model_lineage_id"],
            "model_version": 0,
            "artifact_digest": sha256_file(model_path),
            "manifest_digest": "0" * 64,
        },
        "observation_schema": schema_document(
            semantics.observation_schema
        ),
        "action_schema": schema_document(semantics.action_schema),
        "model_architecture_id": semantics.model_architecture_id,
        "tensor_dtype": config["model"]["tensor_dtype"],
        "input_shape": [1, int(config["model"]["obs_dim"])],
        "action_shape": [1, int(config["model"]["action_dim"])],
        "value_shape": [1, 1],
        "artifact_uri": artifact_uri,
        "model_file": model_file,
        "size_bytes": model_path.stat().st_size,
        "seed": int(config["model"]["bootstrap_seed"]),
        "train_updates": 0,
        "trained_samples": 0,
        "training_config_digest": training_config_digest(config).hex,
        "training_semantics": semantics_document(semantics),
        "published_at_unix_ms": int(time.time() * 1000),
        "ready": True,
    }
    document = finalize_manifest_digest(document)
    atomic_write_json(root / "manifest.json", document)
    logger.info(
        "initial model exported: lineage=%s version=0 artifact=%s manifest=%s",
        document["identity"]["model_lineage_id"],
        document["identity"]["artifact_digest"],
        document["identity"]["manifest_digest"],
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learner_config.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--artifact-uri-prefix", default="")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    log = config["log"]
    logger = setup_logger(
        "ModelBootstrap",
        console_level=log["console_level"],
        file_level=log["file_level"],
        log_dir=log["log_dir"],
    )
    export_initial_model(
        config,
        logger,
        arguments.output_root,
        arguments.artifact_uri_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
