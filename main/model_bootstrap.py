"""Export a deterministic untrained model with the current wire manifest."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.training_runtime import sha256_file
from proto import training_pb2
from src.config.effective_config import load_effective_config
from src.contracts.identity import (
    content_digest,
    finalize_manifest,
    rollout_estimator_profile,
    training_config_digest,
    training_contract_digest,
    validate_config,
    write_manifest_file,
)
from src.log.logger import setup_logger
from src.training.ppo_trainer import PPOTrainer


def load_config(path: str) -> dict:
    return load_effective_config(path)


def export_initial_model(
    config: dict,
    logger,
    output_root: str,
) -> training_pb2.ModelArtifactManifest:
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
    manifest = training_pb2.ModelArtifactManifest(
        identity=training_pb2.ModelIdentity(
            model_lineage_id=config["identity"]["model_lineage_id"],
            model_step=0,
            artifact_digest=content_digest(sha256_file(model_path)),
        ),
        size_bytes=model_path.stat().st_size,
        trained_samples=0,
        training_config_digest=training_config_digest(config),
        training_contract_digest=training_contract_digest(config),
        published_at_unix_ms=int(time.time() * 1000),
        rollout_estimator_profile=rollout_estimator_profile(config),
    )
    manifest = finalize_manifest(manifest)
    write_manifest_file(root / "manifest.pb", manifest)
    logger.info(
        "initial model exported: lineage=%s step=0 artifact=%s manifest=%s",
        manifest.identity.model_lineage_id,
        manifest.identity.artifact_digest.hex,
        manifest.identity.manifest_digest.hex,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learner_config.yaml")
    parser.add_argument("--output-root", required=True)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
