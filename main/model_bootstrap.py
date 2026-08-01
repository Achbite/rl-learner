"""Atomically publish the initial ONNX policy and its identity manifest."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.log.logger import setup_logger
from src.training.ppo_trainer import PPOTrainer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str, document: dict):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def export_initial_model(
    config: dict, logger, output_root: str | None = None
) -> dict:
    model_config = config.get("model", {})
    seed = int(
        os.environ.get(
            "MAZE_MODEL_BOOTSTRAP_SEED",
            model_config.get("bootstrap_seed", 0),
        )
    )
    model_version = 0
    obs_dim = int(model_config.get("obs_dim", 13))
    action_dim = int(model_config.get("action_dim", 9))
    p2p_root = (
        output_root
        or os.environ.get("MAZE_MODEL_ARTIFACT_ROOT")
        or model_config.get("local_train_dir", "models/local-train")
    )
    model_file = f"model_v{model_version:06d}.onnx"
    model_path = os.path.join(p2p_root, model_file)
    manifest_path = os.path.join(p2p_root, "manifest.json")
    temporary_model = f"{model_path}.tmp.{os.getpid()}"

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(p2p_root, exist_ok=True)

    trainer = PPOTrainer(config)
    trainer.export_onnx(temporary_model)
    with open(temporary_model, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary_model, model_path)

    manifest = {
        "schema_version": 1,
        "contract_version": "0.6.0",
        "model_version": model_version,
        "artifact_uri": Path(model_path).resolve().as_uri(),
        "model_file": model_file,
        "size_bytes": os.path.getsize(model_path),
        "sha256": sha256_file(model_path),
        "input_shape": [1, obs_dim],
        "action_shape": [1, action_dim],
        "value_shape": [1, 1],
        "seed": seed,
        "ready": True,
        "published_ts_ms": int(time.time() * 1000),
    }
    atomic_write_json(manifest_path, manifest)
    logger.info(
        "初始模型已发布: version=%d sha256=%s",
        model_version,
        manifest["sha256"],
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Publish the initial AIServer ONNX model"
    )
    parser.add_argument("--config", default="configs/learner_config.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    config = load_config(args.config)
    log_config = config.get("log", {})
    logger = setup_logger(
        "ModelBootstrap",
        console_level=log_config.get("console_level", "INFO"),
        file_level=log_config.get("file_level", "DEBUG"),
        log_dir=log_config.get("log_dir", "logs"),
    )
    export_initial_model(config, logger, args.output_root)


if __name__ == "__main__":
    main()
