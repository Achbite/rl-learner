"""Task-neutral leased-sample PPO runtime for rl-contracts 0.10.0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import grpc
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import common_pb2, training_pb2, training_pb2_grpc
from src.contracts.identity import (
    contract_document,
    contract_identity,
    finalize_manifest_digest,
    manifest_message,
    model_identity_document,
    policy_spec_digest,
    schema_document,
    semantics_document,
    service_identity,
    training_config_digest,
    training_semantics,
    validate_config,
)
from src.log.logger import setup_logger
from src.metrics.metrics_backend import create_backend
from src.training.ppo_trainer import PPOTrainer


_stop_requested = threading.Event()


def _handle_signal(_signal, _frame) -> None:
    _stop_requested.set()


def _same_message(left, right) -> bool:
    return left.SerializeToString(deterministic=True) == right.SerializeToString(
        deterministic=True
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    validate_config(config)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_dict(document: dict | None) -> dict:
    if not document:
        return {}
    identity = document.get("identity", document)
    return {
        "model_lineage_id": str(identity.get("model_lineage_id", "")),
        "model_version": int(identity.get("model_version", -1)),
        "artifact_digest": str(identity.get("artifact_digest", "")),
        "manifest_digest": str(identity.get("manifest_digest", "")),
    }


def _identity_equal(left: dict | None, right: dict | None) -> bool:
    return bool(left and right) and _identity_dict(left) == _identity_dict(right)


def training_chain_status(
    actor: dict,
    distributor: dict,
    learner: dict,
    model: dict,
    error: str = "",
) -> dict:
    """Return task-neutral readiness from exact service/model identities."""
    reasons: list[str] = []
    if error:
        reasons.append("learner_update_error")
    for name, document in (
        ("actor", actor),
        ("sample_pool", distributor),
        ("model_distributor", model),
    ):
        if document.get("error"):
            reasons.append(f"{name}_status_error")
        if not document.get("ready"):
            reasons.append(f"{name}_not_ready")
        if not document.get("instance_id"):
            reasons.append(f"{name}_instance_missing")
    for field in ("ingress_ready", "pool_ready"):
        if not distributor.get(field):
            reasons.append(f"sample_pool_{field}_false")

    learner_model = learner.get("model_identity", {})
    actor_model = actor.get("model_identity", {})
    published_model = model.get("latest_model_identity", {})
    acknowledged_model = model.get("latest_ack_model_identity", {})
    if not _identity_equal(learner_model, published_model):
        reasons.append("published_model_identity_mismatch")
    if not _identity_equal(actor_model, acknowledged_model):
        reasons.append("actor_model_ack_mismatch")
    if model.get("latest_ack_status") != "MODEL_LOAD_STATUS_LOADED":
        reasons.append("actor_model_ack_not_loaded")

    learner_version = int(_identity_dict(learner_model).get("model_version", -1))
    actor_version = int(_identity_dict(actor_model).get("model_version", -1))
    model_lag = learner_version - actor_version
    if learner_version < 0 or actor_version < 0 or not 0 <= model_lag <= 1:
        reasons.append("actor_model_lag_invalid")
    actual_batch_size = int(learner.get("actual_batch_size", 0) or 0)
    if actual_batch_size:
        policy_lag = int(learner.get("policy_lag", -1))
        maximum = int(learner.get("max_policy_lag", -1))
        if policy_lag < 0 or maximum < 0 or policy_lag > maximum:
            reasons.append("training_policy_lag_invalid")
    return {
        "ready": not reasons,
        "state": "ready" if not reasons else "degraded",
        "reasons": reasons,
        "model_lag": model_lag,
        "error": error,
    }


class ModelPublisher:
    ARCHIVE_MODEL_FILE = "SaveModel.onnx"
    ARCHIVE_CHECKPOINT_FILE = "checkpoint.pt"
    ARCHIVE_MANIFEST_FILE = "manifest.json"

    def __init__(self, config: dict):
        validate_config(config)
        model = config["model"]
        self.config = config
        self.seed = int(model["bootstrap_seed"])
        self.obs_dim = int(model["obs_dim"])
        self.action_dim = int(model["action_dim"])
        self.tensor_dtype = str(model["tensor_dtype"])
        self.contract = contract_identity(config)
        self.semantics = training_semantics(config)
        self.training_digest = training_config_digest(config)
        self.lineage_id = str(config["identity"]["model_lineage_id"])
        configured_root = str(model["local_train_dir"])
        self.local_train_root = Path(
            os.environ.get("RL_LOCAL_TRAIN_ROOT", configured_root)
        ).resolve()
        self.runtime_dir = self.local_train_root / "runtime"
        self.published_dir = self.runtime_dir / "serving"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.update_dir = self.runtime_dir / "receipts"
        self.state_path = self.runtime_dir / "state.json"
        self.archive_dir = self.local_train_root / "archive"
        self.metrics_dir = self.local_train_root / "metrics"
        self.archive_interval_updates = int(
            os.environ.get(
                "RL_ARCHIVE_INTERVAL_UPDATES",
                str(model["archive_interval_updates"]),
            )
        )
        self.archive_on_graceful_shutdown = bool(
            model["archive_on_graceful_shutdown"]
        )
        self.serving_retention_versions = int(
            model["serving_retention_versions"]
        )
        if self.archive_interval_updates <= 0:
            raise ValueError("archive_interval_updates must be positive")
        if self.serving_retention_versions < 2:
            raise ValueError("serving_retention_versions must be at least 2")
        self.initial_checkpoint_identity: dict = {}
        self._prepared = False

    def prepare(self) -> None:
        for directory in (
            self.published_dir,
            self.checkpoint_dir,
            self.archive_dir,
            self.update_dir,
            self.metrics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        occupied = [
            path
            for path in (
                self.state_path,
                *self.published_dir.glob("*"),
                *self.checkpoint_dir.glob("*"),
                *self.update_dir.glob("*"),
                *self.archive_dir.glob("*"),
                *self.metrics_dir.glob("*"),
            )
            if path.exists()
        ]
        if occupied:
            raise RuntimeError(
                "local-train was not cleaned before Learner startup: "
                + ", ".join(str(path) for path in occupied[:5])
            )
        self._prepared = True

    def model_path(self, version: int) -> Path:
        return self.published_dir / f"model_v{version:06d}.onnx"

    def manifest_path(self, version: int) -> Path:
        return self.published_dir / f"manifest_v{version:06d}.json"

    def checkpoint_path(self, version: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_v{version:06d}.pt"

    def archive_path(self, version: int) -> Path:
        return self.archive_dir / f"{version:06d}"

    def archive_manifest_path(self, version: int) -> Path:
        return self.archive_path(version) / self.ARCHIVE_MANIFEST_FILE

    def receipt_path(self, train_update_id: str) -> Path:
        return self.update_dir / f"{train_update_id}.json"

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_initial_checkpoint(
        self, trainer: PPOTrainer, checkpoint_value: str
    ) -> dict:
        checkpoint_path = Path(checkpoint_value).resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(
                f"initial checkpoint does not exist: {checkpoint_path}"
            )
        if checkpoint_path == self.local_train_root or (
            self.local_train_root in checkpoint_path.parents
        ):
            raise RuntimeError("initial checkpoint must be outside local-train")
        checkpoint = self._load_checkpoint(checkpoint_path)
        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "model_version",
            "torch_rng_state",
            "numpy_rng_state",
            "metadata",
        }
        if not required.issubset(checkpoint):
            raise RuntimeError("initial checkpoint is incomplete")
        metadata = checkpoint["metadata"]
        if (
            metadata.get("model_lineage_id") != self.lineage_id
            or metadata.get("observation_schema")
            != schema_document(self.semantics.observation_schema)
            or metadata.get("training_config_digest")
            != self.training_digest.hex
        ):
            raise RuntimeError("initial checkpoint identity is incompatible")
        if not trainer.load_checkpoint(str(checkpoint_path)):
            raise RuntimeError("initial checkpoint could not be loaded")
        self.initial_checkpoint_identity = {
            "initial_checkpoint": str(checkpoint_path),
            "initial_checkpoint_digest": sha256_file(checkpoint_path),
            "initial_model_version": int(checkpoint["model_version"]),
        }
        return {
            "train_updates": int(metadata["train_updates"]),
            "trained_samples": int(metadata["trained_samples"]),
            **self.initial_checkpoint_identity,
        }

    def _checkpoint_metadata(
        self,
        *,
        train_update_id: str,
        behavior_model: dict | None,
        batch_ids: list[str],
        stats: dict,
        sample_count: int,
        train_updates: int,
        trained_samples: int,
    ) -> dict:
        return {
            "train_update_id": train_update_id,
            "behavior_model": behavior_model or {},
            "batch_ids": list(batch_ids),
            "stats": dict(stats),
            "sample_count": int(sample_count),
            "train_updates": int(train_updates),
            "trained_samples": int(trained_samples),
            "model_lineage_id": self.lineage_id,
            "observation_schema": schema_document(
                self.semantics.observation_schema
            ),
            "action_schema": schema_document(self.semantics.action_schema),
            "training_config_digest": self.training_digest.hex,
            **self.initial_checkpoint_identity,
        }

    def commit_optimizer_checkpoint(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model: dict,
        batch_ids: list[str],
        stats: dict,
        sample_count: int,
        train_updates: int,
        trained_samples: int,
    ) -> Path:
        if not self._prepared:
            raise RuntimeError("model publisher is not prepared")
        path = self.checkpoint_path(trainer.model_version)
        metadata = self._checkpoint_metadata(
            train_update_id=train_update_id,
            behavior_model=behavior_model,
            batch_ids=batch_ids,
            stats=stats,
            sample_count=sample_count,
            train_updates=train_updates,
            trained_samples=trained_samples,
        )
        if path.exists():
            checkpoint = self._load_checkpoint(path)
            if (
                checkpoint.get("model_version") != trainer.model_version
                or checkpoint.get("metadata", {}).get("train_update_id")
                != train_update_id
            ):
                raise RuntimeError(f"checkpoint identity conflicts: {path}")
            return path
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        trainer.save_checkpoint(str(temporary), metadata=metadata)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return path

    def publish_runtime(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model: dict | None,
        batch_ids: list[str],
        stats: dict | None = None,
        sample_count: int = 0,
        train_updates: int = 0,
        trained_samples: int = 0,
        checkpoint_precommitted: bool = False,
    ) -> dict:
        if not self._prepared:
            raise RuntimeError("model publisher is not prepared")
        version = trainer.model_version
        model_path = self.model_path(version)
        checkpoint_path = self.checkpoint_path(version)
        manifest_path = self.manifest_path(version)
        if manifest_path.exists() or model_path.exists():
            raise RuntimeError(f"runtime model version already exists: {version}")
        temporary_model = model_path.with_name(
            f".{model_path.name}.{os.getpid()}.tmp"
        )
        temporary_checkpoint = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        trainer.export_onnx(str(temporary_model))
        metadata = self._checkpoint_metadata(
            train_update_id=train_update_id,
            behavior_model=behavior_model,
            batch_ids=batch_ids,
            stats=stats or {},
            sample_count=sample_count,
            train_updates=train_updates,
            trained_samples=trained_samples,
        )
        if checkpoint_precommitted:
            checkpoint = self._load_checkpoint(checkpoint_path)
            if (
                checkpoint.get("model_version") != version
                or checkpoint.get("metadata", {}).get("train_update_id")
                != train_update_id
            ):
                raise RuntimeError("precommitted checkpoint identity mismatch")
        else:
            trainer.save_checkpoint(str(temporary_checkpoint), metadata=metadata)
        for path in (
            [temporary_model]
            if checkpoint_precommitted
            else [temporary_model, temporary_checkpoint]
        ):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if not checkpoint_precommitted:
            os.replace(temporary_checkpoint, checkpoint_path)
        os.replace(temporary_model, model_path)
        artifact_digest = sha256_file(model_path)
        document = {
            "manifest_schema_version": 1,
            "contract": contract_document(self.contract),
            "identity": {
                "model_lineage_id": self.lineage_id,
                "model_version": version,
                "artifact_digest": artifact_digest,
                "manifest_digest": "0" * 64,
            },
            "observation_schema": schema_document(
                self.semantics.observation_schema
            ),
            "action_schema": schema_document(self.semantics.action_schema),
            "model_architecture_id": self.semantics.model_architecture_id,
            "tensor_dtype": self.tensor_dtype,
            "input_shape": [1, self.obs_dim],
            "action_shape": [1, self.action_dim],
            "value_shape": [1, 1],
            "artifact_uri": model_path.as_uri(),
            "model_file": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "seed": self.seed,
            "train_updates": int(train_updates),
            "trained_samples": int(trained_samples),
            "training_config_digest": self.training_digest.hex,
            "training_semantics": semantics_document(self.semantics),
            "published_at_unix_ms": int(time.time() * 1000),
            "ready": True,
        }
        document = finalize_manifest_digest(document)
        runtime_document = {
            **document,
            "checkpoint_file": checkpoint_path.name,
            "train_update_id": train_update_id,
            "behavior_model": behavior_model or {},
            "batch_ids": list(batch_ids),
            **self.initial_checkpoint_identity,
        }
        atomic_write_json(manifest_path, runtime_document)
        atomic_write_json(
            self.state_path,
            {
                "schema_version": 1,
                "latest_model": runtime_document["identity"],
                "latest_manifest": str(manifest_path),
                "latest_checkpoint": str(checkpoint_path),
                "train_updates": int(train_updates),
                "trained_samples": int(trained_samples),
                "updated_at_unix_ms": int(time.time() * 1000),
                **self.initial_checkpoint_identity,
            },
        )
        return runtime_document

    def complete_manifest(
        self, version: int, train_update_id: str | None = None
    ) -> dict | None:
        path = self.manifest_path(version)
        model_path = self.model_path(version)
        checkpoint_path = self.checkpoint_path(version)
        if not (path.is_file() and model_path.is_file() and checkpoint_path.is_file()):
            return None
        try:
            document = read_json(path)
            checkpoint = self._load_checkpoint(checkpoint_path)
            canonical = {
                key: document[key]
                for key in (
                    "manifest_schema_version",
                    "contract",
                    "identity",
                    "observation_schema",
                    "action_schema",
                    "model_architecture_id",
                    "tensor_dtype",
                    "input_shape",
                    "action_shape",
                    "value_shape",
                    "artifact_uri",
                    "model_file",
                    "size_bytes",
                    "seed",
                    "train_updates",
                    "trained_samples",
                    "training_config_digest",
                    "training_semantics",
                    "published_at_unix_ms",
                    "ready",
                )
            }
            expected = finalize_manifest_digest(canonical)
        except (OSError, ValueError, KeyError, RuntimeError):
            return None
        metadata = checkpoint.get("metadata", {})
        if (
            document.get("contract") != contract_document(self.contract)
            or document.get("identity", {}).get("model_version") != version
            or document.get("identity", {}).get("artifact_digest")
            != sha256_file(model_path)
            or document.get("identity", {}).get("manifest_digest")
            != expected["identity"]["manifest_digest"]
            or document.get("model_file") != model_path.name
            or document.get("size_bytes") != model_path.stat().st_size
            or document.get("training_semantics")
            != semantics_document(self.semantics)
            or not document.get("ready")
            or checkpoint.get("model_version") != version
            or metadata.get("train_update_id")
            != document.get("train_update_id")
            or (
                train_update_id is not None
                and document.get("train_update_id") != train_update_id
            )
        ):
            return None
        return document

    def complete_manifests(self) -> list[dict]:
        result: list[dict] = []
        for path in sorted(self.published_dir.glob("manifest_v*.json")):
            try:
                version = int(path.stem.removeprefix("manifest_v"))
            except ValueError:
                continue
            document = self.complete_manifest(version)
            if document:
                result.append(document)
        return result

    def latest_complete_checkpoint(self) -> Path | None:
        manifests = self.complete_manifests()
        if not manifests:
            return None
        version = int(manifests[-1]["identity"]["model_version"])
        return self.checkpoint_path(version)

    def should_archive(self, version: int) -> bool:
        return version == 0 or version % self.archive_interval_updates == 0

    def archive_version(self, version: int, reason: str) -> dict:
        manifest = self.complete_manifest(version)
        if manifest is None:
            raise RuntimeError(f"cannot archive incomplete model v{version}")
        target = self.archive_path(version)
        artifact_digest = manifest["identity"]["artifact_digest"]
        if target.exists():
            existing = read_json(self.archive_manifest_path(version))
            if (
                existing.get("runtime_manifest_identity")
                != manifest["identity"]
                or sha256_file(target / self.ARCHIVE_MODEL_FILE)
                != artifact_digest
            ):
                raise RuntimeError(f"archive identity conflicts: {target}")
            return existing
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            shutil.copyfile(
                self.model_path(version), temporary / self.ARCHIVE_MODEL_FILE
            )
            shutil.copyfile(
                self.checkpoint_path(version),
                temporary / self.ARCHIVE_CHECKPOINT_FILE,
            )
            archive_manifest = {
                "schema_version": 1,
                "runtime_manifest_identity": manifest["identity"],
                "runtime_manifest_digest": manifest["identity"][
                    "manifest_digest"
                ],
                "model_file": self.ARCHIVE_MODEL_FILE,
                "checkpoint_file": self.ARCHIVE_CHECKPOINT_FILE,
                "artifact_digest": artifact_digest,
                "archive_reason": reason,
                "archived_at_unix_ms": int(time.time() * 1000),
            }
            atomic_write_json(
                temporary / self.ARCHIVE_MANIFEST_FILE, archive_manifest
            )
            os.replace(temporary, target)
            return archive_manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def prune_runtime(self, current_version: int) -> None:
        minimum = current_version - self.serving_retention_versions + 1
        for manifest in self.complete_manifests():
            version = int(manifest["identity"]["model_version"])
            if version >= minimum:
                continue
            for path in (
                self.manifest_path(version),
                self.model_path(version),
                self.checkpoint_path(version),
            ):
                path.unlink(missing_ok=True)


class LeaseRenewer:
    def __init__(
        self,
        stub,
        consumer: common_pb2.ServiceInstanceIdentity,
        delivery_id: str,
        lease_timeout_ms: int,
    ):
        self.stub = stub
        self.request = training_pb2.RenewLeaseReq(
            consumer=consumer,
            delivery_id=delivery_id,
            lease_timeout_ms=lease_timeout_ms,
        )
        self.interval = max(0.2, lease_timeout_ms / 3000.0)
        self.failure_deadline = max(1.0, lease_timeout_ms / 1000.0 * 0.8)
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self) -> None:
        last_success = time.monotonic()
        while not self._stop.wait(self.interval):
            try:
                response = self.stub.RenewLease(self.request, timeout=2.0)
                if response.result != training_pb2.DELIVERY_RESULT_APPLIED:
                    self.error = response.message or "lease renewal rejected"
                    return
                last_success = time.monotonic()
            except grpc.RpcError as error:
                if time.monotonic() - last_success >= self.failure_deadline:
                    self.error = error.details() or str(error)
                    return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


class TrainingRuntime:
    def __init__(self, config: dict, initial_checkpoint: str = ""):
        validate_config(config)
        self.config = config
        self.logger = setup_logger("TrainingRuntime")
        self.contract = contract_identity(config)
        self.semantics = training_semantics(config)
        self.policy_digest = policy_spec_digest(config)
        self.trainer = PPOTrainer(config)
        self.publisher = ModelPublisher(config)
        if (
            self.trainer.max_policy_lag + 1
            > self.publisher.serving_retention_versions
        ):
            raise ValueError(
                "max_policy_lag requires more retained serving versions"
            )

        learner_name = os.environ.get("RL_LEARNER_INSTANCE", "learner-0")
        instance = f"{learner_name}-{os.getpid()}-{int(time.time() * 1000)}"
        self.learner_service = service_identity("learner", instance, 1)
        self.sequence = 0
        self.train_updates = 0
        self.trained_samples = 0
        self._run_start_train_updates = 0
        self._run_start_trained_samples = 0
        self.initial_model_version = 0
        self.last_stats: dict = {}
        self.model_manifests: dict[int, dict] = {}
        self._last_archive_version: int | None = None
        self._metrics_context = {
            "behavior_model": {},
            "actual_batch_size": 0,
            "disposition": "STARTING",
            "train_update_id": "",
            "error": "",
        }
        self._metrics_lock = threading.RLock()
        self._committed_learner_metrics = {
            "model_identity": {},
            "model_version": self.trainer.model_version,
            "model_step": self.train_updates,
            "run_train_updates": self.train_updates,
            "run_trained_samples": self.trained_samples,
            "policy_lag": 0,
            "max_policy_lag": self.trainer.max_policy_lag,
        }
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._rate_snapshot: dict[str, float] = {}
        self._last_actor_snapshot: dict = {}
        self._last_distributor_snapshot: dict = {}
        self._last_model_snapshot: dict = {}
        self._last_resource_time = time.monotonic()
        self._last_process_cpu = time.process_time()

        self.publisher.prepare()
        configured_checkpoint = str(config["model"]["initial_checkpoint"] or "")
        checkpoint = (
            initial_checkpoint
            or os.environ.get("RL_INITIAL_CHECKPOINT", "")
            or configured_checkpoint
        )
        self._startup_mode = "fresh"
        if checkpoint:
            self._startup_mode = "initial-checkpoint"
            restored = self.publisher.load_initial_checkpoint(
                self.trainer, checkpoint
            )
            self.train_updates = int(restored["train_updates"])
            self.trained_samples = int(restored["trained_samples"])
        self._run_start_train_updates = self.train_updates
        self._run_start_trained_samples = self.trained_samples

        sample = config["sample_distributor"]
        sample_host = os.environ.get(
            "RL_SAMPLE_POOL_HOST", str(sample["host"])
        )
        sample_port = int(
            os.environ.get("RL_SAMPLE_POOL_PORT", str(sample["port"]))
        )
        self.train_batch_size = int(sample["train_batch_size"])
        self.max_train_batch_size = int(sample["max_train_batch_size"])
        self.max_sample_age_ms = int(sample["max_sample_age_ms"])
        self.get_timeout_ms = int(sample["get_timeout_ms"])
        self.lease_timeout_ms = int(sample["lease_timeout_ms"])
        self.finalize_drain_timeout_ms = int(
            sample["finalize_drain_timeout_ms"]
        )
        self.finalize_request_path = Path(
            os.environ.get(
                "RL_TRAINING_FINALIZE_REQUEST_PATH",
                str(sample["finalize_request_path"]),
            )
        )
        self.finalize_complete_path = Path(
            os.environ.get(
                "RL_TRAINING_FINALIZE_COMPLETE_PATH",
                str(sample["finalize_complete_path"]),
            )
        )
        self.finalize_request_path.unlink(missing_ok=True)
        self.finalize_complete_path.unlink(missing_ok=True)
        self._finalized = False
        self.shutdown_drain_timeout_ms = int(
            sample["shutdown_drain_timeout_ms"]
        )
        self.demand_ttl_ms = int(sample["demand_ttl_ms"])
        self.demand_refresh_interval_ms = int(
            sample["demand_refresh_interval_ms"]
        )
        self.demand_max_fragments = int(sample["demand_max_fragments"])
        self.demand_max_estimated_bytes = int(
            sample["demand_max_estimated_bytes"]
        )
        if self.demand_max_fragments < self.max_train_batch_size:
            raise ValueError(
                "demand_max_fragments must cover one-sample partial fragments"
            )
        self.demand_id = f"{self.learner_service.instance_id}-training-demand"
        self._demand_epoch = 0
        self._demand_active = False
        self._last_demand_refresh = 0.0
        self.sample_address = f"{sample_host}:{sample_port}"
        self.sample_channel = grpc.insecure_channel(self.sample_address)
        self.sample_stub = training_pb2_grpc.SampleDistributorServiceStub(
            self.sample_channel
        )

        model = config["model_distributor"]
        model_host = os.environ.get(
            "RL_MODEL_DISTRIBUTOR_HOST", str(model["host"])
        )
        model_port = int(
            os.environ.get("RL_MODEL_DISTRIBUTOR_PORT", str(model["port"]))
        )
        self.model_startup_timeout = float(model["startup_timeout_sec"])
        self.model_address = f"{model_host}:{model_port}"
        self.model_channel = grpc.insecure_channel(self.model_address)
        self.model_stub = training_pb2_grpc.ModelDistributorServiceStub(
            self.model_channel
        )

        actor = config["aiserver_status"]
        actor_host = os.environ.get("RL_AISERVER_HOST", str(actor["host"]))
        actor_port = int(
            os.environ.get("RL_AISERVER_PORT", str(actor["port"]))
        )
        self.actor_address = f"{actor_host}:{actor_port}"
        self.actor_channel = grpc.insecure_channel(self.actor_address)
        self.actor_stub = training_pb2_grpc.AIServerTrainingStatusServiceStub(
            self.actor_channel
        )

        dashboard = config["dashboard"]
        self.metrics_backend = create_backend(
            str(dashboard["backend"]), str(self.publisher.metrics_dir)
        )

    @staticmethod
    def _manifest_for_wire(document: dict) -> dict:
        return {
            key: document[key]
            for key in (
                "manifest_schema_version",
                "contract",
                "identity",
                "observation_schema",
                "action_schema",
                "model_architecture_id",
                "tensor_dtype",
                "input_shape",
                "action_shape",
                "value_shape",
                "artifact_uri",
                "model_file",
                "size_bytes",
                "seed",
                "train_updates",
                "trained_samples",
                "training_config_digest",
                "training_semantics",
                "published_at_unix_ms",
                "ready",
            )
        }

    def _register(self, document: dict) -> None:
        response = self.model_stub.RegisterModel(
            training_pb2.RegisterModelReq(
                manifest=manifest_message(self._manifest_for_wire(document))
            ),
            timeout=5.0,
        )
        if response.result not in (
            training_pb2.MODEL_REGISTER_RESULT_REGISTERED,
            training_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
        ):
            raise RuntimeError(
                f"model registration rejected: {response.message}"
            )
        expected = manifest_message(self._manifest_for_wire(document))
        if not _same_message(response.manifest, expected):
            raise RuntimeError("model distributor returned a different manifest")

    def _wait_initial_model_loaded(self, document: dict) -> None:
        expected = manifest_message(self._manifest_for_wire(document)).identity
        deadline = time.monotonic() + self.model_startup_timeout
        last = ""
        while time.monotonic() < deadline and not _stop_requested.is_set():
            try:
                status = self.model_stub.GetModelDistributorStatus(
                    training_pb2.ModelDistributorStatusReq(), timeout=2.0
                )
                if (
                    status.ready
                    and status.latest_ack_status
                    == training_pb2.MODEL_LOAD_STATUS_LOADED
                    and _same_message(status.latest_ack_model, expected)
                ):
                    return
                last = (
                    f"status={training_pb2.ModelLoadStatus.Name(status.latest_ack_status)} "
                    f"ack={model_identity_document(status.latest_ack_model)}"
                )
            except grpc.RpcError as error:
                last = error.details() or str(error)
            time.sleep(0.2)
        raise RuntimeError(f"AIServer did not ACK exact model identity: {last}")

    def _initialize_models(self) -> None:
        version = self.trainer.model_version
        update_id = (
            "initial-checkpoint"
            if self._startup_mode == "initial-checkpoint"
            else "bootstrap-v0"
        )
        document = self.publisher.publish_runtime(
            self.trainer,
            train_update_id=update_id,
            behavior_model=None,
            batch_ids=[],
            train_updates=self.train_updates,
            trained_samples=self.trained_samples,
        )
        self.model_manifests[version] = document
        self.initial_model_version = version
        self._register(document)
        self._wait_initial_model_loaded(document)
        self.publisher.archive_version(version, self._startup_mode)
        self._last_archive_version = version
        self._commit_learner_metrics(
            document,
            behavior_model={},
            actual_batch_size=0,
            disposition="READY",
            train_update_id=update_id,
            train_updates=self.train_updates,
            trained_samples=self.trained_samples,
            stats={},
        )
        self.logger.info(
            "Learner training ready: model=v%d artifact=%s startup=%s",
            version,
            document["identity"]["artifact_digest"],
            self._startup_mode,
        )

    def _sample_pool_status(self):
        return self.sample_stub.GetStatus(
            training_pb2.DistributorStatusReq(), timeout=2.0
        )

    def _demand_message(self) -> training_pb2.SampleDemand:
        return training_pb2.SampleDemand(
            demand_id=self.demand_id,
            demand_epoch=self.trainer.model_version + 1,
            consumer=self.learner_service,
            contract=self.contract,
            training_semantics=self.semantics,
            freshness=training_pb2.SampleFreshnessPolicy(
                model_lineage_id=self.publisher.lineage_id,
                reference_model_version=self.trainer.model_version,
                max_version_lag=self.trainer.max_policy_lag,
                max_sample_age_ms=self.max_sample_age_ms,
                distribution_schema_id=(
                    self.semantics.policy_distribution_schema_id
                ),
                policy_spec_digest=self.policy_digest,
            ),
            assembly=training_pb2.BatchAssemblySpec(
                target_samples=self.train_batch_size,
                max_samples=self.max_train_batch_size,
                mode=training_pb2.BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
            ),
            max_buffered_samples=self.max_train_batch_size,
            max_buffered_fragments=self.demand_max_fragments,
            max_buffered_estimated_bytes=self.demand_max_estimated_bytes,
            expires_at_unix_ms=int(time.time() * 1000) + self.demand_ttl_ms,
        )

    def _upsert_demand(self, force: bool = False) -> None:
        epoch = self.trainer.model_version + 1
        now = time.monotonic()
        if (
            not force
            and self._demand_active
            and self._demand_epoch == epoch
            and (now - self._last_demand_refresh) * 1000
            < self.demand_refresh_interval_ms
        ):
            return
        demand = self._demand_message()
        response = self.sample_stub.UpsertSampleDemand(
            training_pb2.UpsertSampleDemandReq(demand=demand), timeout=3.0
        )
        if response.result not in (
            training_pb2.SAMPLE_DEMAND_RESULT_APPLIED,
            training_pb2.SAMPLE_DEMAND_RESULT_ALREADY_APPLIED,
        ):
            raise RuntimeError(f"sample demand rejected: {response.message}")
        if (
            response.demand.demand_id != demand.demand_id
            or int(response.demand.demand_epoch) != int(demand.demand_epoch)
            or not _same_message(response.demand.consumer, self.learner_service)
        ):
            raise RuntimeError("sample distributor returned another demand")
        self._demand_epoch = epoch
        self._demand_active = True
        self._last_demand_refresh = now

    def _release_demand(self, required: bool) -> None:
        if not self._demand_active:
            return
        released = False
        try:
            response = self.sample_stub.ReleaseSampleDemand(
                training_pb2.ReleaseSampleDemandReq(
                    consumer=self.learner_service,
                    contract=self.contract,
                    demand_id=self.demand_id,
                    demand_epoch=self._demand_epoch,
                ),
                timeout=3.0,
            )
            if response.result not in (
                training_pb2.SAMPLE_DEMAND_RESULT_RELEASED,
                training_pb2.SAMPLE_DEMAND_RESULT_NOT_FOUND,
            ):
                raise RuntimeError(
                    f"sample demand release rejected: {response.message}"
                )
            released = True
        except (grpc.RpcError, RuntimeError) as error:
            if required:
                raise RuntimeError("sample demand release failed") from error
            self.logger.error("sample demand release failed: %s", error)
        if released:
            self._demand_active = False

    def _assert_sample_pool_ready(self) -> None:
        status = self._sample_pool_status()
        if not (
            status.ready
            and status.ingress_ready
            and status.pool_ready
            and _same_message(status.contract, self.contract)
            and status.distributor.component == "sample-distributor"
            and status.active_demand_count == 1
            and int(status.active_demand_epoch) == self._demand_epoch
        ):
            raise RuntimeError(
                "sample distributor is not ready for the exact demand"
            )

    def _get_batch(
        self,
        mode: int = training_pb2.BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
    ):
        return self.sample_stub.GetBatch(
            training_pb2.GetBatchReq(
                assembly=training_pb2.BatchAssemblySpec(
                    target_samples=self.train_batch_size,
                    max_samples=self.max_train_batch_size,
                    mode=mode,
                ),
                timeout_ms=self.get_timeout_ms,
                consumer=self.learner_service,
                lease_timeout_ms=self.lease_timeout_ms,
                freshness=training_pb2.SampleFreshnessPolicy(
                    model_lineage_id=self.publisher.lineage_id,
                    reference_model_version=self.trainer.model_version,
                    max_version_lag=self.trainer.max_policy_lag,
                    max_sample_age_ms=self.max_sample_age_ms,
                    distribution_schema_id=(
                        self.semantics.policy_distribution_schema_id
                    ),
                    policy_spec_digest=self.policy_digest,
                ),
                required_semantics=self.semantics,
            ),
            timeout=max(2.0, self.get_timeout_ms / 1000.0 + 1.0),
        )

    @staticmethod
    def _validate_sample(sample: training_pb2.Sample) -> None:
        values = [
            *sample.observation,
            *sample.next_observation,
            sample.reward,
            sample.old_log_probability,
            sample.old_value_prediction,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sample contains non-finite values")
        if sample.action < 0 or sample.action >= 9:
            raise ValueError("sample action is outside maze.action.v1")
        if len(sample.observation) != 17 or len(sample.next_observation) != 17:
            raise ValueError("sample does not match maze.observation.v3")
        if sample.terminated and sample.truncated:
            raise ValueError("sample cannot be terminated and truncated")
        expected = (
            training_pb2.TRANSITION_END_KIND_ENVIRONMENT_TERMINATED
            if sample.terminated
            else training_pb2.TRANSITION_END_KIND_EXTERNAL_TRUNCATION
            if sample.truncated
            else training_pb2.TRANSITION_END_KIND_CONTINUING
        )
        if sample.end_kind != expected:
            raise ValueError("sample end_kind conflicts with terminal flags")

    def _validate_delivery(
        self, response, *, allow_partial: bool = False
    ) -> dict:
        minimum_samples = 1 if allow_partial else self.train_batch_size
        if (
            response.result != training_pb2.GET_BATCH_RESULT_LEASED
            or response.actual_batch_size < minimum_samples
            or response.actual_batch_size > self.max_train_batch_size
            or response.returned_samples != response.actual_batch_size
            or response.returned_fragments != len(response.batches)
            or not response.delivery_id
        ):
            raise ValueError("sample delivery violates bounded batch assembly")
        sample_count = 0
        behavior_versions: set[int] = set()
        oldest_created_at = 0
        newest_created_at = 0
        now_ms = int(time.time() * 1000)
        for batch in response.batches:
            behavior = batch.behavior_policy
            version = int(behavior.model_version)
            lag = self.trainer.model_version - version
            if (
                not batch.batch_id
                or not batch.trajectory_id
                or not _same_message(batch.contract, self.contract)
                or not _same_message(batch.training_semantics, self.semantics)
                or behavior.model_lineage_id != self.publisher.lineage_id
                or behavior.distribution_schema_id
                != self.semantics.policy_distribution_schema_id
                or not _same_message(
                    behavior.policy_spec_digest, self.policy_digest
                )
                or lag < 0
                or lag > self.trainer.max_policy_lag
                or version not in self.model_manifests
                or batch.created_at_unix_ms <= 0
                or now_ms - int(batch.created_at_unix_ms)
                > self.max_sample_age_ms
                or not batch.producer.instance_id
                or batch.producer.component != "aiserver"
                or batch.first_action_step > batch.last_action_step
            ):
                raise ValueError("sample batch identity is invalid")
            full_identity = manifest_message(
                self._manifest_for_wire(self.model_manifests[version])
            ).identity
            if (
                full_identity.model_lineage_id != behavior.model_lineage_id
                or int(full_identity.model_version) != version
            ):
                raise ValueError("behavior policy cannot resolve to a model artifact")
            behavior_versions.add(version)
            created_at = int(batch.created_at_unix_ms)
            oldest_created_at = (
                created_at
                if oldest_created_at == 0
                else min(oldest_created_at, created_at)
            )
            newest_created_at = max(newest_created_at, created_at)
            digest_copy = training_pb2.SampleBatch()
            digest_copy.CopyFrom(batch)
            supplied = digest_copy.payload_digest.hex
            digest_copy.ClearField("payload_digest")
            actual = hashlib.sha256(
                digest_copy.SerializeToString(deterministic=True)
            ).hexdigest()
            if supplied != actual:
                raise ValueError("sample batch payload digest mismatch")
            for index, sample in enumerate(batch.samples):
                self._validate_sample(sample)
                if index != len(batch.samples) - 1 and (
                    sample.terminated or sample.truncated
                ):
                    raise ValueError("fragment continues after end transition")
                sample_count += 1
            terminal = bool(batch.samples and batch.samples[-1].terminated)
            truncated = bool(batch.samples and batch.samples[-1].truncated)
            if terminal:
                if batch.bootstrap_valid or batch.bootstrap_value != 0.0:
                    raise ValueError("terminated fragment must not bootstrap")
            elif not batch.bootstrap_valid or not math.isfinite(
                batch.bootstrap_value
            ):
                raise ValueError(
                    "continuing or truncated fragment requires bootstrap"
                )
            if batch.trajectory_end != (terminal or truncated):
                raise ValueError("trajectory_end conflicts with final sample")
        if sample_count != response.actual_batch_size:
            raise ValueError("delivery sample count does not match response")
        minimum_version = min(behavior_versions)
        maximum_version = max(behavior_versions)
        if (
            response.minimum_behavior_model_version != minimum_version
            or response.maximum_behavior_model_version != maximum_version
            or response.oldest_sample_created_at_unix_ms
            != oldest_created_at
            or response.newest_sample_created_at_unix_ms
            != newest_created_at
        ):
            raise ValueError("delivery summary does not match fragment identities")
        models = [
            dict(self.model_manifests[version]["identity"])
            for version in sorted(behavior_versions)
        ]
        return {
            "model_lineage_id": self.publisher.lineage_id,
            "minimum_model_version": minimum_version,
            "maximum_model_version": maximum_version,
            "models": models,
        }

    def _training_samples(self, batches) -> list[dict]:
        result: list[dict] = []
        for batch in batches:
            trajectory = [
                {
                    "observation": list(sample.observation),
                    "next_observation": list(sample.next_observation),
                    "action": int(sample.action),
                    "reward": float(sample.reward),
                    "old_log_probability": float(
                        sample.old_log_probability
                    ),
                    "old_value_prediction": float(
                        sample.old_value_prediction
                    ),
                    "terminated": bool(sample.terminated),
                    "truncated": bool(sample.truncated),
                    "action_step": int(sample.action_step),
                }
                for sample in batch.samples
            ]
            processed = self.trainer.compute_gae(
                trajectory,
                bootstrap_value=float(batch.bootstrap_value),
                bootstrap_valid=bool(batch.bootstrap_valid),
            )
            for sample in processed:
                sample["behavior_model_version"] = int(
                    batch.behavior_policy.model_version
                )
            result.extend(processed)
        return result

    def _ack(
        self,
        delivery_id: str,
        disposition: int,
        train_update_id: str = "",
    ) -> None:
        response = self.sample_stub.AckBatch(
            training_pb2.AckBatchReq(
                consumer=self.learner_service,
                delivery_id=delivery_id,
                disposition=disposition,
                train_update_id=train_update_id,
            ),
            timeout=3.0,
        )
        if response.result not in (
            training_pb2.DELIVERY_RESULT_APPLIED,
            training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
        ):
            raise RuntimeError(f"sample ACK failed: {response.message}")

    def _nack(self, delivery_id: str, reason: str) -> None:
        try:
            response = self.sample_stub.NackBatch(
                training_pb2.NackBatchReq(
                    consumer=self.learner_service,
                    delivery_id=delivery_id,
                    reason=reason[:256],
                ),
                timeout=3.0,
            )
            if response.result not in (
                training_pb2.DELIVERY_RESULT_APPLIED,
                training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
            ):
                self.logger.error("sample NACK rejected: %s", response.message)
        except grpc.RpcError as error:
            self.logger.error("sample NACK failed: %s", error)

    def _write_receipt(self, update_id: str, document: dict) -> None:
        atomic_write_json(self.publisher.receipt_path(update_id), document)

    def _train_delivery(self, response, *, allow_partial: bool = False) -> None:
        behavior_identity = self._validate_delivery(
            response, allow_partial=allow_partial
        )
        update_number = self.train_updates + 1
        update_id = f"train-update-{update_number:08d}"
        batch_ids = [batch.batch_id for batch in response.batches]
        receipt = {
            "schema_version": 1,
            "train_update_id": update_id,
            "delivery_id": response.delivery_id,
            "behavior_model": behavior_identity,
            "batch_ids": batch_ids,
            "state": "LEASED",
            "sample_count": int(response.actual_batch_size),
        }
        self._write_receipt(update_id, receipt)
        renewer = LeaseRenewer(
            self.sample_stub,
            self.learner_service,
            response.delivery_id,
            self.lease_timeout_ms,
        ).start()
        try:
            training_samples = self._training_samples(response.batches)
            stats = self.trainer.train_on_batch(training_samples)
            if renewer.error:
                raise RuntimeError(f"sample lease lost: {renewer.error}")
            target_updates = self.train_updates + 1
            target_samples = self.trained_samples + len(training_samples)
            self.publisher.commit_optimizer_checkpoint(
                self.trainer,
                train_update_id=update_id,
                behavior_model=behavior_identity,
                batch_ids=batch_ids,
                stats=stats,
                sample_count=len(training_samples),
                train_updates=target_updates,
                trained_samples=target_samples,
            )
            receipt.update(
                {
                    "state": "OPTIMIZER_COMMITTED",
                    "target_model_version": self.trainer.model_version,
                    "stats": stats,
                    "train_updates": target_updates,
                    "trained_samples": target_samples,
                }
            )
            self._write_receipt(update_id, receipt)
            manifest = self.publisher.publish_runtime(
                self.trainer,
                train_update_id=update_id,
                behavior_model=behavior_identity,
                batch_ids=batch_ids,
                stats=stats,
                sample_count=len(training_samples),
                train_updates=target_updates,
                trained_samples=target_samples,
                checkpoint_precommitted=True,
            )
            self.model_manifests[self.trainer.model_version] = manifest
            receipt.update(
                {"state": "MODEL_COMMITTED", "model": manifest["identity"]}
            )
            self._write_receipt(update_id, receipt)
            with self._metrics_lock:
                self._register(manifest)
                receipt["state"] = "REGISTERED"
                self._write_receipt(update_id, receipt)
                self._ack(
                    response.delivery_id,
                    training_pb2.ACK_DISPOSITION_TRAINED,
                    update_id,
                )
                receipt["state"] = "ACKED"
                self._write_receipt(update_id, receipt)
                self._commit_learner_metrics(
                    manifest,
                    behavior_model=behavior_identity,
                    actual_batch_size=len(training_samples),
                    disposition="TRAINED",
                    train_update_id=update_id,
                    train_updates=target_updates,
                    trained_samples=target_samples,
                    stats=stats,
                )
            if self.publisher.should_archive(self.trainer.model_version):
                self.publisher.archive_version(
                    self.trainer.model_version, "interval"
                )
                self._last_archive_version = self.trainer.model_version
            self.publisher.prune_runtime(self.trainer.model_version)
            self._record_metrics()
        except Exception:
            if receipt.get("state") == "LEASED":
                self._nack(response.delivery_id, "learner update failed")
            raise
        finally:
            renewer.close()

    @staticmethod
    def _metric_snapshot(snapshot: training_pb2.MetricSnapshot) -> tuple[dict, dict]:
        descriptors = {item.field_id: item for item in snapshot.descriptors}
        values: dict[str, float] = {}
        labels: dict[str, str] = {}
        for item in snapshot.values:
            descriptor = descriptors.get(item.field_id)
            if descriptor is None:
                continue
            value = float(item.value)
            if not math.isfinite(value):
                continue
            values[item.field_id] = value
            labels[item.field_id] = descriptor.label
        return values, labels

    @staticmethod
    def _component_error_snapshot(component: str, error: str) -> dict:
        return {
            "component": component,
            "ready": False,
            "error": error,
            "timestamp": time.time(),
        }

    def _actor_snapshot(self) -> dict:
        try:
            status = self.actor_stub.GetAIServerStatus(
                training_pb2.AIServerStatusReq(), timeout=1.5
            )
            if not _same_message(status.contract, self.contract):
                raise RuntimeError("AIServer contract identity mismatch")
            values, labels = self._metric_snapshot(status.metrics)
            reward_components: dict[str, float] = {}
            prefix = "server.reward.component."
            suffix = ".transition_mean.v1"
            for field_id, value in values.items():
                if field_id.startswith(prefix) and field_id.endswith(suffix):
                    reward_components[field_id[len(prefix) : -len(suffix)]] = value
            episode_return = values.get(
                "server.episode.learning_return.mean.v1", 0.0
            )
            success_value = values.get(
                "server.episode.success.agent_rate.v1", 0.0
            )
            if success_value > 1.0:
                success_value /= 100.0
            return {
                "ready": bool(status.ready),
                "state": training_pb2.AIServerState.Name(status.state),
                "instance_id": status.aiserver.instance_id,
                "lifecycle_epoch": int(status.aiserver.lifecycle_epoch),
                "model_identity": model_identity_document(status.loaded_model),
                "staged_model_identity": model_identity_document(
                    status.staged_model
                ),
                "produced": int(status.produced_unique_samples),
                "produced_batches": int(status.produced_unique_batches),
                "accepted": int(status.accepted_unique_samples),
                "push_attempts": int(status.push_attempt_count),
                "duplicate_push_attempts": int(
                    status.duplicate_push_attempt_count
                ),
                "rejected_push_attempts": int(
                    status.rejected_push_attempt_count
                ),
                "retry_attempts": int(status.retry_attempt_count),
                "final_drop": int(status.final_drop_unique_samples),
                "outbound_pending": int(status.outbound_queue_samples),
                "inference_count": int(status.inference_count),
                "inference_mean_ms": (
                    float(status.inference_latency_sum_ms)
                    / max(1, int(status.inference_count))
                ),
                "inference_max_ms": float(status.inference_latency_max_ms),
                "push_rpc_count": int(status.push_rpc_count),
                "push_rpc_mean_ms": (
                    float(status.push_rpc_latency_sum_ms)
                    / max(1, int(status.push_rpc_count))
                ),
                "push_rpc_max_ms": float(status.push_rpc_latency_max_ms),
                "metric_source": {
                    "instance_id": status.metrics.source.instance_id,
                    "lifecycle_epoch": int(
                        status.metrics.source.lifecycle_epoch
                    ),
                    "sequence": int(status.metrics.sequence),
                    "timestamp_unix_ms": int(
                        status.metrics.timestamp_unix_ms
                    ),
                },
                "metric_values": values,
                "metric_labels": labels,
                "episodes": {
                    "mean_agent_return": episode_return,
                    "min_agent_return": values.get(
                        "server.episode.learning_return.min.v1",
                        episode_return,
                    ),
                    "max_agent_return": values.get(
                        "server.episode.learning_return.max.v1",
                        episode_return,
                    ),
                    "agent_success_rate": success_value,
                    "reward_components": reward_components,
                },
                "error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError, ValueError) as error:
            return self._component_error_snapshot("aiserver", str(error))

    def _distributor_snapshot(self) -> dict:
        try:
            status = self._sample_pool_status()
            if not _same_message(status.contract, self.contract):
                raise RuntimeError(
                    "sample distributor contract identity mismatch"
                )
            if status.distributor.component != "sample-distributor":
                raise RuntimeError("sample distributor component mismatch")
            return {
                "ready": bool(status.ready),
                "ingress_ready": bool(status.ingress_ready),
                "pool_ready": bool(status.pool_ready),
                "instance_id": status.distributor.instance_id,
                "lifecycle_epoch": int(status.distributor.lifecycle_epoch),
                "accepted": int(status.accepted_unique_samples),
                "accepted_batches": int(status.accepted_unique_batches),
                "acked": int(status.acked_unique_samples),
                "acked_batches": int(status.acked_unique_batches),
                "trained": int(status.trained_sample_count),
                "stale": int(status.stale_sample_count),
                "invalid": int(status.invalid_sample_count),
                "shutdown_untrained": int(
                    status.shutdown_untrained_sample_count
                ),
                "ready_samples": int(status.ready_queue_samples),
                "ready_fragments": int(status.ready_queue_fragments),
                "leased_samples": int(status.leased_samples),
                "leased_fragments": int(status.leased_fragments),
                "resident_samples": int(status.resident_samples),
                "resident_fragments": int(status.resident_fragments),
                "reserved_samples": int(status.reserved_samples),
                "reserved_fragments": int(status.reserved_fragments),
                "active_demand_count": int(status.active_demand_count),
                "active_demand_epoch": int(status.active_demand_epoch),
                "credit_requests": int(status.credit_request_count),
                "credit_grants": int(status.credit_grant_count),
                "credit_commits": int(status.credit_commit_count),
                "credit_releases": int(status.credit_release_count),
                "credit_expirations": int(status.credit_expire_count),
                "credit_revocations": int(status.credit_revoke_count),
                "credit_wait_no_demand": int(
                    status.credit_wait_no_demand_count
                ),
                "credit_wait_inflight_limit": int(
                    status.credit_wait_inflight_limit_count
                ),
                "credit_wait_capacity": int(
                    status.credit_wait_capacity_count
                ),
                "credit_wait_draining": int(
                    status.credit_wait_draining_count
                ),
                "redelivery_count": int(status.redelivery_count),
                "nack_count": int(status.nack_count),
                "expired_lease_count": int(status.expired_lease_count),
                "last_error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError) as error:
            return self._component_error_snapshot(
                "sample-distributor", str(error)
            )

    def _model_snapshot(self) -> dict:
        try:
            status = self.model_stub.GetModelDistributorStatus(
                training_pb2.ModelDistributorStatusReq(), timeout=1.5
            )
            if not _same_message(status.contract, self.contract):
                raise RuntimeError("model distributor contract identity mismatch")
            return {
                "ready": bool(status.ready),
                "instance_id": status.distributor.instance_id,
                "lifecycle_epoch": int(status.distributor.lifecycle_epoch),
                "registered_model_count": int(status.registered_model_count),
                "latest_model_identity": model_identity_document(
                    status.latest_model
                ),
                "latest_ack_model_identity": model_identity_document(
                    status.latest_ack_model
                ),
                "latest_ack_status": training_pb2.ModelLoadStatus.Name(
                    status.latest_ack_status
                ),
                "latest_ack_aiserver": status.latest_ack_aiserver.instance_id,
                "last_error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError) as error:
            return self._component_error_snapshot(
                "model-distributor", str(error)
            )

    def _resource_snapshot(self) -> dict:
        now = time.monotonic()
        process_cpu = time.process_time()
        elapsed = max(now - self._last_resource_time, 1e-6)
        cpu = max(0.0, (process_cpu - self._last_process_cpu) / elapsed * 100.0)
        self._last_resource_time = now
        self._last_process_cpu = process_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = float(usage.ru_maxrss) / 1024.0
        if sys.platform == "darwin":
            rss_mb /= 1024.0
        return {"cpu_percent": cpu, "memory_mb": rss_mb}

    def _rates(self, actor: dict, distributor: dict, timestamp: float) -> dict:
        counters = {
            "produced": float(actor.get("produced", 0)),
            "accepted": float(distributor.get("accepted", 0)),
            "acked": float(distributor.get("acked", 0)),
            "trained": float(distributor.get("trained", 0)),
            "timestamp": timestamp,
        }
        previous = self._rate_snapshot
        self._rate_snapshot = counters
        elapsed = timestamp - float(previous.get("timestamp", timestamp))
        if elapsed <= 0:
            return {
                "produced_sps": 0.0,
                "accepted_sps": 0.0,
                "acked_sps": 0.0,
                "trained_sps": 0.0,
            }
        return {
            f"{name}_sps": max(
                0.0, (counters[name] - float(previous.get(name, counters[name]))) / elapsed
            )
            for name in ("produced", "accepted", "acked", "trained")
        }

    def _commit_learner_metrics(
        self,
        manifest: dict,
        *,
        behavior_model: dict,
        actual_batch_size: int,
        disposition: str,
        train_update_id: str,
        train_updates: int,
        trained_samples: int,
        stats: dict,
    ) -> None:
        identity = dict(manifest["identity"])
        committed = {
            "model_identity": identity,
            "model_version": int(identity["model_version"]),
            "model_step": int(train_updates),
            "train_updates": int(train_updates),
            "trained_samples": int(trained_samples),
            "run_train_updates": int(
                train_updates - self._run_start_train_updates
            ),
            "run_trained_samples": int(
                trained_samples - self._run_start_trained_samples
            ),
            "policy_lag": int(stats.get("policy_lag", 0)),
            "max_policy_lag": self.trainer.max_policy_lag,
            **dict(stats),
        }
        with self._metrics_lock:
            self.train_updates = int(train_updates)
            self.trained_samples = int(trained_samples)
            self.last_stats = dict(stats)
            self._committed_learner_metrics = committed
            self._metrics_context = {
                "behavior_model": dict(behavior_model),
                "actual_batch_size": int(actual_batch_size),
                "disposition": disposition,
                "train_update_id": train_update_id,
                "error": "",
            }

    def _learner_metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            context = copy.deepcopy(self._metrics_context)
            committed = copy.deepcopy(self._committed_learner_metrics)
        return {
            "instance_id": self.learner_service.instance_id,
            "lifecycle_epoch": int(self.learner_service.lifecycle_epoch),
            **committed,
            "initial_model_version": int(self.initial_model_version),
            "train_batch_size": int(self.train_batch_size),
            "max_train_batch_size": int(self.max_train_batch_size),
            "actual_batch_size": int(context.get("actual_batch_size", 0)),
            "behavior_model": context.get("behavior_model", {}),
            "disposition": context.get("disposition", ""),
            "train_update_id": context.get("train_update_id", ""),
        }

    def _record_metrics(self) -> None:
        with self._metrics_lock:
            actor = self._actor_snapshot()
            distributor = self._distributor_snapshot()
            model = self._model_snapshot()
            now = time.time()
            context = copy.deepcopy(self._metrics_context)
            learner = self._learner_metrics_snapshot()
            chain = training_chain_status(
                actor,
                distributor,
                learner,
                model,
                str(context.get("error", "")),
            )
            self.sequence += 1
            record = {
                "schema_version": 3,
                "mode": "training",
                "sequence": self.sequence,
                "timestamp": now,
                "interval_ms": 1000,
                "contract": contract_document(self.contract),
                "training_semantics": semantics_document(self.semantics),
                "learner": learner,
                "actor": actor,
                "distributor": distributor,
                "model": model,
                "chain": chain,
                "rates": self._rates(actor, distributor, now),
                "resources": {"learner": self._resource_snapshot()},
            }
            self.metrics_backend.write(record)
            self._last_actor_snapshot = actor
            self._last_distributor_snapshot = distributor
            self._last_model_snapshot = model

    def _metrics_loop(self) -> None:
        while not self._metrics_stop.wait(1.0):
            try:
                self._record_metrics()
            except Exception as error:
                self.logger.error("metrics snapshot failed: %s", error)

    def _start_metrics(self) -> None:
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop,
            name="learner-metrics",
            daemon=True,
        )
        self._metrics_thread.start()

    def _stop_metrics(self) -> None:
        self._metrics_stop.set()
        if self._metrics_thread:
            self._metrics_thread.join(timeout=3.0)

    def _shutdown_drain(self) -> None:
        deadline = time.monotonic() + self.shutdown_drain_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                status = self._sample_pool_status()
            except grpc.RpcError:
                return
            if status.ready_queue_samples == 0 and status.leased_samples == 0:
                return
            response = self._get_batch(
                training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE
            )
            if response.result == training_pb2.GET_BATCH_RESULT_LEASED:
                self._ack(
                    response.delivery_id,
                    training_pb2.ACK_DISPOSITION_SHUTDOWN_UNTRAINED,
                )
            else:
                time.sleep(0.1)

    def _finalize_training(self) -> None:
        deadline = (
            time.monotonic() + self.finalize_drain_timeout_ms / 1000.0
        )
        while time.monotonic() < deadline:
            status = self._sample_pool_status()
            if (
                status.ready_queue_samples == 0
                and status.leased_samples == 0
                and status.reserved_samples == 0
            ):
                self._release_demand(required=True)
                with self._metrics_lock:
                    self._metrics_context["disposition"] = "FINALIZED"
                self._record_metrics()
                self.finalize_complete_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                self.finalize_complete_path.write_text(
                    f"{self.train_updates} {self.trained_samples}\n",
                    encoding="utf-8",
                )
                self._finalized = True
                return
            response = self._get_batch(
                training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE
            )
            if response.result == training_pb2.GET_BATCH_RESULT_LEASED:
                self._train_delivery(response, allow_partial=True)
                continue
            if response.result != training_pb2.GET_BATCH_RESULT_TIMEOUT:
                raise RuntimeError(
                    f"final sample drain failed: {response.message}"
                )
            time.sleep(0.1)
        raise RuntimeError("final sample drain timed out")

    def run(self) -> int:
        self._initialize_models()
        self._upsert_demand(force=True)
        self._start_metrics()
        self._record_metrics()
        try:
            while not _stop_requested.is_set():
                if self.finalize_request_path.is_file():
                    self._finalize_training()
                    while not _stop_requested.wait(0.2):
                        pass
                    break
                self._upsert_demand()
                self._assert_sample_pool_ready()
                response = self._get_batch()
                if response.result == training_pb2.GET_BATCH_RESULT_TIMEOUT:
                    continue
                if response.result != training_pb2.GET_BATCH_RESULT_LEASED:
                    raise RuntimeError(
                        f"sample GetBatch failed: {response.message}"
                    )
                try:
                    self._train_delivery(response)
                except ValueError as error:
                    self._ack(
                        response.delivery_id,
                        training_pb2.ACK_DISPOSITION_INVALID,
                    )
                    raise RuntimeError(str(error)) from error
            with self._metrics_lock:
                self._metrics_context["disposition"] = "STOPPING"
            self._release_demand(required=True)
            self._shutdown_drain()
            if (
                self.publisher.archive_on_graceful_shutdown
                and self._last_archive_version != self.trainer.model_version
            ):
                self.publisher.archive_version(
                    self.trainer.model_version, "graceful-shutdown"
                )
            self._record_metrics()
            return 0
        except Exception as error:
            with self._metrics_lock:
                self._metrics_context["error"] = str(error)
                self._metrics_context["disposition"] = "FAILED"
            self.logger.exception("Learner training failed")
            try:
                self._record_metrics()
            except Exception:
                pass
            return 1
        finally:
            self._release_demand(required=False)
            self._stop_metrics()
            self.metrics_backend.close()
            self.actor_channel.close()
            self.model_channel.close()
            self.sample_channel.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/learner_config.yaml"
    )
    parser.add_argument("--initial-checkpoint", default="")
    arguments = parser.parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    config = load_config(arguments.config)
    runtime = TrainingRuntime(config, arguments.initial_checkpoint)
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
