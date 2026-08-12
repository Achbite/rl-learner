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
from typing import Callable, Iterable

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
from src.metrics.metrics_backend import DisabledMetricsBackend, create_backend
from src.training.ppo_trainer import PPOTrainer


_stop_requested = threading.Event()


class _UpdateCommitError(RuntimeError):
    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = int(attempts)


class _UpdateCommitOutcomeUncertain(_UpdateCommitError):
    """The remote side effect may have succeeded despite the RPC failure."""


class _UpdateCommitNotApplied(_UpdateCommitError):
    """The exact remote state proves that the side effect was not applied."""


class _SamplePoolUnavailable(RuntimeError):
    """The current Sample Pool authority is temporarily unable to serve."""


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
    if actor.get("client_session_recent") is not True:
        reasons.append("client_session_not_recent")

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

    def should_archive(self, run_train_updates: int) -> bool:
        return (
            run_train_updates > 0
            and run_train_updates % self.archive_interval_updates == 0
        )

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
        authority_check: Callable[[], None],
    ):
        self.stub = stub
        self.request = training_pb2.RenewLeaseReq(
            consumer=consumer,
            delivery_id=delivery_id,
            lease_timeout_ms=lease_timeout_ms,
        )
        self.authority_check = authority_check
        self._renew_lock = threading.Lock()
        self.interval = max(0.2, lease_timeout_ms / 3000.0)
        self.failure_deadline = max(1.0, lease_timeout_ms / 1000.0 * 0.8)
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.renew_now()
        self._thread.start()
        return self

    def renew_now(self) -> None:
        with self._renew_lock:
            response = self.stub.RenewLease(self.request, timeout=2.0)
            positive_result = response.result in (
                training_pb2.DELIVERY_RESULT_APPLIED,
                training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
            )
            if (response.ret_code == 0) != positive_result:
                raise RuntimeError(
                    "lease renewal response is contradictory: "
                    f"ret_code={response.ret_code}, result={response.result}"
                )
            if not positive_result:
                raise RuntimeError(response.message or "lease renewal rejected")
            if response.delivery_id != self.request.delivery_id:
                raise RuntimeError("lease renewal returned another delivery")
            if int(response.lease_deadline_unix_ms) <= int(time.time() * 1000):
                raise RuntimeError("lease renewal returned a non-future deadline")
            self.authority_check()

    def _run(self) -> None:
        last_success = time.monotonic()
        while not self._stop.wait(self.interval):
            try:
                self.renew_now()
                last_success = time.monotonic()
            except grpc.RpcError as error:
                if time.monotonic() - last_success >= self.failure_deadline:
                    self.error = error.details() or str(error)
                    return
            except Exception as error:
                self.error = str(error)
                return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


class TrainingRuntime:
    UPDATE_COMMIT_MAX_ATTEMPTS = 3
    UPDATE_COMMIT_RETRY_DELAY_SEC = 0.05
    SAMPLE_RETRY_INITIAL_SEC = 0.05
    SAMPLE_RETRY_MAX_SEC = 1.0
    GET_BATCH_RECONCILE_POLL_SEC = 0.5
    GET_BATCH_RECONCILE_STABLE_WINDOW_SEC = 0.5
    GET_BATCH_RECONCILE_CONFIRMATIONS = 2
    DEMAND_RELEASE_RETRY_TIMEOUT_SEC = 5.0
    SHUTDOWN_RECONCILE_MARGIN_SEC = 5.0

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
        self._metrics_poll_lock = threading.Lock()
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
            source_version = int(restored["initial_model_version"])
            self.trainer.reserve_initial_checkpoint_publication_version(
                source_version
            )
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
        self._demand_authority: (
            common_pb2.ServiceInstanceIdentity | None
        ) = None
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
        self.metrics_backend = self._create_metrics_backend(
            str(dashboard["backend"]), str(self.publisher.metrics_dir)
        )

    def _create_metrics_backend(self, backend_type: str, metrics_dir: str):
        try:
            return create_backend(backend_type, metrics_dir)
        except OSError as error:
            self.logger.error(
                "metrics backend unavailable; training will continue without "
                "local metrics persistence: %s",
                error,
            )
            return DisabledMetricsBackend(str(error))

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

    @staticmethod
    def _model_distributor_authority(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            valid = (
                authority.component == "model-distributor"
                and bool(authority.instance_id)
                and int(authority.lifecycle_epoch) > 0
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError("model distributor authority is invalid")
        result = common_pb2.ServiceInstanceIdentity()
        try:
            result.CopyFrom(authority)
        except TypeError as error:
            raise RuntimeError(
                "model distributor authority has an invalid wire type"
            ) from error
        return result

    @staticmethod
    def _sample_distributor_authority(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            valid = (
                authority.component == "sample-distributor"
                and bool(authority.instance_id)
                and int(authority.lifecycle_epoch) > 0
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError("sample distributor authority is invalid")
        result = common_pb2.ServiceInstanceIdentity()
        try:
            result.CopyFrom(authority)
        except TypeError as error:
            raise RuntimeError(
                "sample distributor authority has an invalid wire type"
            ) from error
        return result

    @staticmethod
    def _authority_document(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> dict:
        return {
            "component": authority.component,
            "instance_id": authority.instance_id,
            "lifecycle_epoch": int(authority.lifecycle_epoch),
        }

    @staticmethod
    def _same_authority(
        left: common_pb2.ServiceInstanceIdentity,
        right: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        return (
            left.component == right.component
            and left.instance_id == right.instance_id
            and int(left.lifecycle_epoch) == int(right.lifecycle_epoch)
        )

    def _pin_model_distributor_authority(
        self,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            status = self.model_stub.GetModelDistributorStatus(
                training_pb2.ModelDistributorStatusReq(), timeout=3.0
            )
        except grpc.RpcError as error:
            raise RuntimeError(
                f"model distributor authority could not be pinned: {error}"
            ) from error
        if not status.ready or not _same_message(status.contract, self.contract):
            raise RuntimeError(
                "model distributor authority could not be pinned: "
                "distributor is not ready for the exact contract"
            )
        try:
            return self._model_distributor_authority(status.distributor)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"model distributor authority could not be pinned: {error}"
            ) from error

    def _lookup_initial_model_manifest(
        self,
        *,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
        version: int | None = None,
        latest: bool = False,
    ) -> tuple[
        training_pb2.ModelArtifactManifest | None,
        common_pb2.ServiceInstanceIdentity,
    ]:
        if latest == (version is not None):
            raise ValueError(
                "initial model lookup requires exactly one selector"
            )
        selector = training_pb2.ModelIdentity(
            model_lineage_id=self.publisher.lineage_id,
            model_version=0 if version is None else int(version),
        )
        try:
            response = self.model_stub.GetModelManifest(
                training_pb2.GetModelManifestReq(
                    requested_model=selector,
                    requester=self.learner_service,
                    latest_in_lineage=latest,
                ),
                timeout=3.0,
            )
        except grpc.RpcError as error:
            raise RuntimeError(
                "initial model distributor preflight lookup failed"
            ) from error
        try:
            response_authority = self._model_distributor_authority(
                response.distributor
            )
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "initial model preflight returned invalid authority"
            ) from error
        try:
            manifest_present = response.HasField("manifest")
        except (AttributeError, ValueError):
            manifest_present = getattr(response, "manifest", None) is not None
        if response.ret_code != 0:
            if manifest_present:
                raise RuntimeError(
                    "initial model preflight returned a manifest with a "
                    "negative ret_code"
                )
            if not self._same_authority(
                response_authority, pinned_authority
            ):
                raise RuntimeError(
                    "initial model absence is uncertain after distributor "
                    "authority changed"
                )
            return None, response_authority
        if not manifest_present:
            raise RuntimeError(
                "initial model preflight returned success without a manifest"
            )
        manifest = response.manifest
        identity = manifest.identity
        if (
            not manifest.ready
            or not _same_message(manifest.contract, self.contract)
            or not _same_message(
                manifest.training_semantics, self.semantics
            )
            or not _same_message(
                manifest.training_config_digest,
                self.publisher.training_digest,
            )
            or identity.model_lineage_id != self.publisher.lineage_id
            or (version is not None and int(identity.model_version) != version)
            or not identity.artifact_digest.hex
            or not identity.manifest_digest.hex
            or int(manifest.train_updates) < 0
            or int(manifest.trained_samples) < 0
        ):
            raise RuntimeError(
                "initial model preflight returned an incompatible manifest"
            )
        result = training_pb2.ModelArtifactManifest()
        result.CopyFrom(manifest)
        return result, response_authority

    @staticmethod
    def _initial_model_version(
        manifest: training_pb2.ModelArtifactManifest | None,
    ) -> int | None:
        return (
            None
            if manifest is None
            else int(manifest.identity.model_version)
        )

    def _assert_initial_latest_not_newer(
        self,
        target_version: int,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        latest, _ = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            latest=True,
        )
        latest_version = self._initial_model_version(latest)
        if latest_version is not None and latest_version > target_version:
            raise RuntimeError(
                "model distributor already contains a newer publication: "
                f"latest={latest_version}, target={target_version}"
            )
        return pinned_authority

    def _initial_model_requires_registration(
        self,
        document: dict,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[bool, common_pb2.ServiceInstanceIdentity]:
        expected = manifest_message(self._manifest_for_wire(document))
        target_version = int(expected.identity.model_version)
        latest, latest_authority = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            latest=True,
        )
        latest_version = self._initial_model_version(latest)
        if latest_version is not None and latest_version > target_version:
            raise RuntimeError(
                "model distributor already contains a newer publication: "
                f"latest={latest_version}, target={target_version}"
            )
        if latest_version == target_version:
            if not _same_message(latest, expected):
                raise RuntimeError(
                    "initial publication target slot conflicts with the "
                    "registered manifest"
                )
            return False, latest_authority

        if self._startup_mode == "initial-checkpoint" and latest is not None:
            source_version = int(
                self.publisher.initial_checkpoint_identity[
                    "initial_model_version"
                ]
            )
            if latest_version == source_version and (
                latest.identity.artifact_digest.hex
                != expected.identity.artifact_digest.hex
                or int(latest.train_updates) != int(expected.train_updates)
                or int(latest.trained_samples)
                != int(expected.trained_samples)
            ):
                raise RuntimeError(
                    "checkpoint source conflicts with the surviving "
                    "registered publication"
                )

        target, target_authority = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            version=target_version,
        )
        if target is None:
            return True, target_authority
        if not _same_message(target, expected):
            raise RuntimeError(
                "initial publication target slot conflicts with the "
                "registered manifest"
            )
        return False, target_authority

    def _register(
        self, document: dict
    ) -> common_pb2.ServiceInstanceIdentity:
        response = self.model_stub.RegisterModel(
            training_pb2.RegisterModelReq(
                manifest=manifest_message(self._manifest_for_wire(document))
            ),
            timeout=5.0,
        )
        positive_result = response.result in (
            training_pb2.MODEL_REGISTER_RESULT_REGISTERED,
            training_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
        )
        if (response.ret_code == 0) != positive_result:
            raise _UpdateCommitOutcomeUncertain(
                "model registration response is contradictory: "
                f"ret_code={response.ret_code}, result={response.result}"
            )
        if not positive_result:
            raise RuntimeError(
                f"model registration rejected: {response.message}"
            )
        expected = manifest_message(self._manifest_for_wire(document))
        if not _same_message(response.manifest, expected):
            raise RuntimeError("model distributor returned a different manifest")
        try:
            return self._model_distributor_authority(response.distributor)
        except (AttributeError, RuntimeError) as error:
            raise _UpdateCommitOutcomeUncertain(
                "registered model response has invalid distributor authority"
            ) from error

    def _exact_model_registered(
        self,
        document: dict,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[
        bool | None, common_pb2.ServiceInstanceIdentity | None
    ]:
        expected = manifest_message(self._manifest_for_wire(document))
        try:
            response = self.model_stub.GetModelManifest(
                training_pb2.GetModelManifestReq(
                    requested_model=expected.identity,
                    requester=self.learner_service,
                ),
                timeout=3.0,
            )
        except grpc.RpcError:
            return None, None
        try:
            response_authority = self._model_distributor_authority(
                response.distributor
            )
        except (AttributeError, RuntimeError) as error:
            raise _UpdateCommitOutcomeUncertain(
                "exact model lookup has invalid distributor authority"
            ) from error
        if response.ret_code != 0:
            try:
                manifest_present = response.HasField("manifest")
            except (AttributeError, ValueError):
                manifest_present = getattr(response, "manifest", None) is not None
            if manifest_present:
                raise _UpdateCommitOutcomeUncertain(
                    "exact model lookup returned a manifest with a negative "
                    "ret_code"
                )
            if not self._same_authority(
                response_authority, pinned_authority
            ):
                raise _UpdateCommitOutcomeUncertain(
                    "model distributor authority changed before exact absence"
                )
            return False, response_authority
        if not _same_message(response.manifest, expected):
            raise _UpdateCommitOutcomeUncertain(
                "model distributor returned a conflicting exact identity"
            )
        return True, response_authority

    def _register_idempotently(
        self,
        document: dict,
        lease_error: Callable[[], str],
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[int, common_pb2.ServiceInstanceIdentity]:
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(1, self.UPDATE_COMMIT_MAX_ATTEMPTS + 1):
            current_lease_error = str(lease_error())
            if current_lease_error:
                last_error = RuntimeError(
                    "sample lease lost before model registration attempt "
                    f"{attempt}: {current_lease_error}"
                )
                break
            attempts = attempt
            try:
                registered_authority = self._register(document)
                return attempt, registered_authority
            except _UpdateCommitOutcomeUncertain as error:
                last_error = error
                break
            except Exception as error:
                last_error = error
                if attempt == self.UPDATE_COMMIT_MAX_ATTEMPTS:
                    break
                self.logger.warning(
                    "model register attempt %d failed; retrying exact manifest: %s",
                    attempt,
                    error,
                )
                time.sleep(self.UPDATE_COMMIT_RETRY_DELAY_SEC)

        try:
            registered, resolved_authority = self._exact_model_registered(
                document, pinned_authority
            )
        except _UpdateCommitOutcomeUncertain as error:
            error.attempts = attempts
            raise
        if registered:
            if resolved_authority is None:
                raise RuntimeError(
                    "exact registration evidence has no distributor authority"
                )
            return attempts, resolved_authority
        if registered is False:
            raise _UpdateCommitNotApplied(
                "model registration was not applied after "
                f"{attempts} attempts: {last_error}",
                attempts,
            ) from last_error
        raise _UpdateCommitOutcomeUncertain(
            "model registration outcome is uncertain after "
            f"{attempts} attempts: {last_error}",
            attempts,
        ) from last_error

    def _wait_initial_model_loaded(self, document: dict) -> None:
        expected = manifest_message(self._manifest_for_wire(document)).identity
        deadline = time.monotonic() + self.model_startup_timeout
        last = ""
        while time.monotonic() < deadline and not _stop_requested.is_set():
            try:
                status = self.model_stub.GetModelDistributorStatus(
                    training_pb2.ModelDistributorStatusReq(), timeout=2.0
                )
                try:
                    self._model_distributor_authority(status.distributor)
                    status_authority_valid = True
                except (AttributeError, RuntimeError):
                    status_authority_valid = False
                if (
                    status.ready
                    and _same_message(status.contract, self.contract)
                    and status_authority_valid
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
            f"initial-checkpoint-republish-v{version}"
            if self._startup_mode == "initial-checkpoint"
            else "bootstrap-v0"
        )
        pinned_authority = self._pin_model_distributor_authority()
        pinned_authority = self._assert_initial_latest_not_newer(
            version, pinned_authority
        )
        document = self.publisher.publish_runtime(
            self.trainer,
            train_update_id=update_id,
            behavior_model=None,
            batch_ids=[],
            train_updates=self.train_updates,
            trained_samples=self.trained_samples,
        )
        register_required, pinned_authority = (
            self._initial_model_requires_registration(
                document, pinned_authority
            )
        )
        if register_required:
            _, pinned_authority = self._register_idempotently(
                document,
                lambda: "",
                pinned_authority,
            )
        self.model_manifests[version] = document
        self.initial_model_version = version
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

    @staticmethod
    def _is_retryable_sample_rpc(error: grpc.RpcError) -> bool:
        try:
            code = error.code()
        except Exception:
            return False
        return code in (
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.CANCELLED,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.UNKNOWN,
            grpc.StatusCode.UNAVAILABLE,
        )

    @staticmethod
    def _rpc_error_text(error: grpc.RpcError) -> str:
        try:
            details = error.details()
        except Exception:
            details = ""
        return details or str(error)

    def _mark_sample_wait(
        self,
        disposition: str,
        operation: str,
        error: object,
        attempt: int,
    ) -> None:
        message = f"{operation}: {error}"
        with self._metrics_lock:
            current = str(self._metrics_context.get("disposition", ""))
            if current not in (
                "WAITING_FOR_SAMPLE_POOL",
                "GET_BATCH_OUTCOME_UNKNOWN",
            ):
                self._sample_wait_resume_disposition = current or "READY"
            self._metrics_context["disposition"] = disposition
            self._metrics_context["error"] = message
        should_log = attempt == 1 or (attempt & (attempt - 1)) == 0
        warning = getattr(self.logger, "warning", None)
        if should_log and callable(warning):
            warning(
                "%s; waiting for the same Sample Pool authority "
                "(attempt=%d)",
                message,
                attempt,
            )

    def _clear_sample_wait(self) -> None:
        with self._metrics_lock:
            if self._metrics_context.get("disposition") in (
                "WAITING_FOR_SAMPLE_POOL",
                "GET_BATCH_OUTCOME_UNKNOWN",
            ):
                self._metrics_context["disposition"] = getattr(
                    self, "_sample_wait_resume_disposition", "READY"
                )
                self._metrics_context["error"] = ""
        self._sample_wait_resume_disposition = "READY"

    def _wait_sample_retry(
        self,
        attempt: int,
        deadline: float | None = None,
        *,
        ignore_stop: bool = False,
    ) -> bool:
        delay = min(
            self.SAMPLE_RETRY_MAX_SEC,
            self.SAMPLE_RETRY_INITIAL_SEC * (2 ** min(attempt - 1, 8)),
        )
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
            if delay <= 0:
                return True
        if ignore_stop:
            time.sleep(delay)
            return False
        return _stop_requested.wait(delay)

    def _ready_sample_distributor_authority(
        self, status
    ) -> common_pb2.ServiceInstanceIdentity:
        if not _same_message(status.contract, self.contract):
            raise RuntimeError(
                "sample distributor status has another contract identity"
            )
        authority = self._sample_distributor_authority(status.distributor)
        if not status.ready:
            raise _SamplePoolUnavailable(
                "sample distributor is not ready for the exact contract"
            )
        return authority

    def _sample_status_for_authority(
        self,
        expected: common_pb2.ServiceInstanceIdentity,
        operation: str,
    ):
        status = self._sample_pool_status()
        actual = self._ready_sample_distributor_authority(status)
        if not self._same_authority(actual, expected):
            raise RuntimeError(
                f"sample distributor authority changed during {operation}"
            )
        return status

    def _assert_lease_authority(
        self, expected: common_pb2.ServiceInstanceIdentity
    ) -> None:
        self._sample_status_for_authority(expected, "the lease")

    def _resolvable_model_identity(
        self, version: int
    ) -> training_pb2.ModelIdentity | None:
        document = self.model_manifests.get(int(version))
        if not isinstance(document, dict):
            return None
        try:
            manifest = manifest_message(self._manifest_for_wire(document))
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if (
            not manifest.ready
            or not _same_message(manifest.contract, self.contract)
            or not _same_message(
                manifest.training_semantics, self.semantics
            )
            or not _same_message(
                manifest.training_config_digest,
                self.publisher.training_digest,
            )
            or manifest.identity.model_lineage_id
            != self.publisher.lineage_id
            or int(manifest.identity.model_version) != int(version)
        ):
            return None
        return manifest.identity

    def _effective_max_policy_lag(self) -> int:
        current_version = int(self.trainer.model_version)
        if self._resolvable_model_identity(current_version) is None:
            raise RuntimeError(
                "current model manifest is not locally resolvable"
            )
        configured_max = int(self.trainer.max_policy_lag)
        effective_max = 0
        for lag in range(1, configured_max + 1):
            version = current_version - lag
            if version < 0 or self._resolvable_model_identity(version) is None:
                break
            effective_max = lag
        return effective_max

    def _demand_message(self) -> training_pb2.SampleDemand:
        effective_max_policy_lag = self._effective_max_policy_lag()
        return training_pb2.SampleDemand(
            demand_id=self.demand_id,
            demand_epoch=self.trainer.model_version + 1,
            consumer=self.learner_service,
            contract=self.contract,
            training_semantics=self.semantics,
            freshness=training_pb2.SampleFreshnessPolicy(
                model_lineage_id=self.publisher.lineage_id,
                reference_model_version=self.trainer.model_version,
                max_version_lag=effective_max_policy_lag,
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
        preflight_authority = self._ready_sample_distributor_authority(
            self._sample_pool_status()
        )
        if self._demand_active:
            if self._demand_authority is None:
                raise RuntimeError(
                    "active sample demand has no pinned distributor authority"
                )
            pinned_authority = self._sample_distributor_authority(
                self._demand_authority
            )
            if not self._same_authority(
                preflight_authority, pinned_authority
            ):
                raise RuntimeError(
                    "sample distributor authority changed while demand was active"
                )
        negative_authority = (
            self._sample_distributor_authority(self._demand_authority)
            if self._demand_active and self._demand_authority is not None
            else preflight_authority
        )
        response = self.sample_stub.UpsertSampleDemand(
            training_pb2.UpsertSampleDemandReq(demand=demand), timeout=3.0
        )
        response_authority = self._sample_distributor_authority(
            response.distributor
        )
        positive_result = response.result in (
            training_pb2.SAMPLE_DEMAND_RESULT_APPLIED,
            training_pb2.SAMPLE_DEMAND_RESULT_ALREADY_APPLIED,
        )
        if (response.ret_code == 0) != positive_result:
            raise RuntimeError(
                "sample demand upsert response is contradictory: "
                f"ret_code={response.ret_code}, result={response.result}"
            )
        if not positive_result:
            if not self._same_authority(
                response_authority, negative_authority
            ):
                raise RuntimeError(
                    "sample demand upsert outcome is uncertain after "
                    "distributor authority changed"
                )
            raise RuntimeError(f"sample demand rejected: {response.message}")
        if not self._same_authority(
            response_authority, preflight_authority
        ):
            raise RuntimeError(
                "sample demand upsert was applied by a different distributor "
                "authority"
            )
        if not _same_message(response.demand, demand):
            raise RuntimeError(
                "sample distributor did not echo the exact applied demand"
            )
        status = self._sample_status_for_authority(
            response_authority, "demand upsert"
        )
        if (
            int(status.active_demand_count) != 1
            or int(status.active_demand_epoch) != int(demand.demand_epoch)
        ):
            raise RuntimeError(
                "sample distributor did not expose the applied demand epoch"
            )
        self._demand_epoch = epoch
        self._demand_active = True
        self._demand_authority = response_authority
        self._last_demand_refresh = now

    def _release_demand(self, required: bool) -> None:
        if not self._demand_active:
            return
        if self._demand_authority is None:
            raise RuntimeError(
                "sample demand release has no pinned distributor authority"
            )
        pinned_authority = self._sample_distributor_authority(
            self._demand_authority
        )
        released = False
        try:
            request = training_pb2.ReleaseSampleDemandReq(
                consumer=self.learner_service,
                contract=self.contract,
                demand_id=self.demand_id,
                demand_epoch=self._demand_epoch,
            )
            release_deadline = (
                time.monotonic() + self.DEMAND_RELEASE_RETRY_TIMEOUT_SEC
            )
            attempt = 0
            release_waited = False
            while True:
                attempt += 1
                try:
                    response = self.sample_stub.ReleaseSampleDemand(
                        request, timeout=3.0
                    )
                    break
                except grpc.RpcError as error:
                    if (
                        not required
                        or not self._is_retryable_sample_rpc(error)
                        or time.monotonic() >= release_deadline
                    ):
                        raise
                    self._mark_sample_wait(
                        "WAITING_FOR_SAMPLE_POOL",
                        "sample demand release outcome unknown",
                        self._rpc_error_text(error),
                        attempt,
                    )
                    release_waited = True
                    if self._wait_sample_retry(
                        attempt,
                        release_deadline,
                        ignore_stop=True,
                    ):
                        raise
            response_authority = self._sample_distributor_authority(
                response.distributor
            )
            if not self._same_authority(
                response_authority, pinned_authority
            ):
                raise RuntimeError(
                    "sample demand release authority changed"
                )
            terminal_result = response.result in (
                training_pb2.SAMPLE_DEMAND_RESULT_RELEASED,
                training_pb2.SAMPLE_DEMAND_RESULT_NOT_FOUND,
            )
            if (response.ret_code == 0) != terminal_result:
                raise RuntimeError(
                    "sample demand release response is contradictory: "
                    f"ret_code={response.ret_code}, result={response.result}"
                )
            if not terminal_result:
                if not self._same_authority(
                    response_authority, pinned_authority
                ):
                    raise RuntimeError(
                        "sample demand release outcome is uncertain after "
                        "distributor authority changed"
                    )
                raise RuntimeError(
                    f"sample demand release rejected: {response.message}"
                )
            if (
                response.result
                == training_pb2.SAMPLE_DEMAND_RESULT_NOT_FOUND
                and not self._same_authority(
                    response_authority, pinned_authority
                )
            ):
                raise RuntimeError(
                    "sample demand absence came from another distributor authority"
                )
            try:
                demand_present = response.HasField("demand")
            except (AttributeError, ValueError):
                demand_present = getattr(response, "demand", None) is not None
            if (
                demand_present
                or int(response.reserved_samples) != 0
                or int(response.reserved_fragments) != 0
                or int(response.reserved_estimated_bytes) != 0
            ):
                raise RuntimeError(
                    "sample demand release did not return an empty terminal state"
                )
            status = self._sample_status_for_authority(
                response_authority, "demand release"
            )
            if (
                int(status.active_demand_count) != 0
                or int(status.active_demand_epoch) != 0
                or int(status.reserved_samples) != 0
                or int(status.reserved_fragments) != 0
                or int(status.reserved_estimated_bytes) != 0
            ):
                raise RuntimeError(
                    "sample distributor did not expose an empty released demand state"
                )
            if release_waited:
                self._clear_sample_wait()
            released = True
        except (grpc.RpcError, RuntimeError) as error:
            if required:
                raise RuntimeError("sample demand release failed") from error
            self.logger.error("sample demand release failed: %s", error)
        if released:
            self._demand_active = False
            self._demand_authority = None

    def _assert_sample_pool_ready(
        self,
    ) -> common_pb2.ServiceInstanceIdentity:
        status = self._sample_pool_status()
        try:
            authority = self._ready_sample_distributor_authority(status)
        except _SamplePoolUnavailable as error:
            raise _SamplePoolUnavailable(
                "sample distributor is not ready for the exact demand"
            ) from error
        except RuntimeError as error:
            raise RuntimeError(
                "sample distributor does not match the exact demand"
            ) from error
        if not self._demand_active or self._demand_authority is None:
            raise RuntimeError(
                "sample distributor readiness has no active pinned demand"
            )
        pinned_authority = self._sample_distributor_authority(
            self._demand_authority
        )
        if not self._same_authority(authority, pinned_authority):
            raise RuntimeError(
                "sample distributor authority changed while demand was active"
            )
        if not (
            status.ingress_ready
            and status.pool_ready
            and status.active_demand_count == 1
            and int(status.active_demand_epoch) == self._demand_epoch
        ):
            raise _SamplePoolUnavailable(
                "sample distributor is not ready for the exact demand"
            )
        return authority

    def _wait_for_sample_pool(
        self, *, force_demand: bool = False
    ) -> common_pb2.ServiceInstanceIdentity | None:
        attempt = 0
        while not _stop_requested.is_set():
            try:
                self._upsert_demand(force=force_demand)
                authority = self._assert_sample_pool_ready()
                self._clear_sample_wait()
                return authority
            except _SamplePoolUnavailable as error:
                attempt += 1
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "sample pool readiness",
                    error,
                    attempt,
                )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                attempt += 1
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "sample pool transport",
                    self._rpc_error_text(error),
                    attempt,
                )
            if self._wait_sample_retry(attempt):
                return None
        return None

    def _get_batch_recovery_status(
        self, expected: common_pb2.ServiceInstanceIdentity
    ):
        status = self._sample_pool_status()
        if not _same_message(status.contract, self.contract):
            raise RuntimeError(
                "sample distributor contract changed while GetBatch outcome "
                "was unknown"
            )
        actual = self._sample_distributor_authority(status.distributor)
        if not self._same_authority(actual, expected):
            raise RuntimeError(
                "sample distributor authority changed while GetBatch outcome "
                "was unknown"
            )
        if int(status.max_concurrent_consumers) != 1:
            raise RuntimeError(
                "GetBatch recovery requires the declared single-consumer "
                "delivery contract"
            )
        leased_samples = int(status.leased_samples)
        leased_fragments = int(status.leased_fragments)
        active_consumers = int(status.active_consumer_count)
        if (
            leased_samples < 0
            or leased_fragments < 0
            or active_consumers < 0
            or (leased_samples == 0 and leased_fragments != 0)
            or (leased_samples == 0 and active_consumers != 0)
            or (leased_samples > 0 and leased_fragments == 0)
            or leased_fragments > leased_samples
            or (leased_samples > 0 and active_consumers != 1)
        ):
            raise RuntimeError(
                "sample distributor returned contradictory lease status while "
                "GetBatch outcome was unknown"
            )
        return status

    def _reconcile_get_batch_outcome(
        self,
        expected: common_pb2.ServiceInstanceIdentity,
        reason: str,
        deadline: float | None = None,
        *,
        ignore_stop: bool = False,
    ) -> bool:
        expected = self._sample_distributor_authority(expected)
        attempt = 0
        stable_started: float | None = None
        stable_confirmations = 0
        while (ignore_stop or not _stop_requested.is_set()) and (
            deadline is None or time.monotonic() < deadline
        ):
            attempt += 1
            try:
                status = self._get_batch_recovery_status(expected)
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                stable_started = None
                stable_confirmations = 0
                self._mark_sample_wait(
                    "GET_BATCH_OUTCOME_UNKNOWN",
                    "GetBatch reconciliation transport",
                    self._rpc_error_text(error),
                    attempt,
                )
                if self._wait_sample_retry(
                    attempt, deadline, ignore_stop=ignore_stop
                ):
                    return False
                continue

            if (
                not status.ready
                or not status.pool_ready
                or int(status.leased_samples) > 0
            ):
                stable_started = None
                stable_confirmations = 0
                state = (
                    f"hidden lease samples={int(status.leased_samples)}"
                    if int(status.leased_samples) > 0
                    else "pool not ready"
                )
                self._mark_sample_wait(
                    "GET_BATCH_OUTCOME_UNKNOWN",
                    "GetBatch reconciliation",
                    state,
                    attempt,
                )
            else:
                now = time.monotonic()
                if stable_started is None:
                    stable_started = now
                stable_confirmations += 1
                self._mark_sample_wait(
                    "GET_BATCH_OUTCOME_UNKNOWN",
                    "GetBatch reconciliation",
                    "confirming stable zero-lease window",
                    attempt,
                )
                if (
                    stable_confirmations
                    >= self.GET_BATCH_RECONCILE_CONFIRMATIONS
                    and now - stable_started
                    >= self.GET_BATCH_RECONCILE_STABLE_WINDOW_SEC
                ):
                    # The server-side cancellation fence prevents the timed-out
                    # handler from creating a later lease. Requiring repeated
                    # zero-lease observations also keeps a retry out of a
                    # transient status/lease projection boundary.
                    self._clear_sample_wait()
                    info = getattr(self.logger, "info", None)
                    if callable(info):
                        info(
                            "GetBatch outcome reconciled without training: %s",
                            reason,
                        )
                    return True
            poll_delay = self.GET_BATCH_RECONCILE_POLL_SEC
            if deadline is not None:
                poll_delay = min(
                    poll_delay, max(0.0, deadline - time.monotonic())
                )
                if poll_delay <= 0:
                    return False
            if ignore_stop:
                time.sleep(poll_delay)
            elif _stop_requested.wait(poll_delay):
                return False
        return False

    def _get_batch(
        self,
        mode: int = training_pb2.BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
        ready_authority: common_pb2.ServiceInstanceIdentity | None = None,
    ):
        effective_max_policy_lag = self._effective_max_policy_lag()
        response = self.sample_stub.GetBatch(
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
                    max_version_lag=effective_max_policy_lag,
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
        if response.result == training_pb2.GET_BATCH_RESULT_BUSY:
            response_authority = self._sample_distributor_authority(
                response.distributor
            )
            if ready_authority is None:
                ready_authority = self._ready_sample_distributor_authority(
                    self._sample_pool_status()
                )
            else:
                ready_authority = self._sample_distributor_authority(
                    ready_authority
                )
            if (
                response.ret_code == 0
                or not self._same_authority(
                    response_authority, ready_authority
                )
                or int(response.leased_samples) <= 0
                or bool(response.delivery_id)
                or len(response.batches) != 0
            ):
                raise RuntimeError(
                    "sample distributor returned an incoherent busy delivery"
                )
            return response
        if response.result != training_pb2.GET_BATCH_RESULT_LEASED:
            return response
        if ready_authority is None:
            ready_authority = self._ready_sample_distributor_authority(
                self._sample_pool_status()
            )
        else:
            ready_authority = self._sample_distributor_authority(
                ready_authority
            )
        response_authority = self._sample_distributor_authority(
            response.distributor
        )
        minimum_samples = (
            1
            if mode == training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE
            else self.train_batch_size
        )
        if (
            response.ret_code != 0
            or not self._same_authority(
                response_authority, ready_authority
            )
            or not response.delivery_id
            or int(response.actual_batch_size) < minimum_samples
            or int(response.actual_batch_size) > self.max_train_batch_size
            or int(response.returned_samples)
            != int(response.actual_batch_size)
            or int(response.returned_fragments) != len(response.batches)
            or int(response.leased_samples)
            != int(response.actual_batch_size)
            or int(response.lease_deadline_unix_ms)
            <= int(time.time() * 1000)
        ):
            raise RuntimeError(
                "sample distributor returned an incoherent leased delivery"
            )
        return response

    def _get_batch_recovering(
        self,
        *,
        mode: int = training_pb2.BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
        ready_authority: common_pb2.ServiceInstanceIdentity,
        deadline: float | None = None,
        ignore_stop: bool = False,
    ):
        expected = self._sample_distributor_authority(ready_authority)
        try:
            response = self._get_batch(
                mode=mode, ready_authority=expected
            )
        except grpc.RpcError as error:
            if not self._is_retryable_sample_rpc(error):
                raise
            reason = self._rpc_error_text(error)
            self._mark_sample_wait(
                "GET_BATCH_OUTCOME_UNKNOWN",
                "GetBatch transport outcome unknown",
                reason,
                1,
            )
            self._reconcile_get_batch_outcome(
                expected,
                reason,
                deadline=deadline,
                ignore_stop=ignore_stop,
            )
            return None
        if response.result == training_pb2.GET_BATCH_RESULT_BUSY:
            reason = response.message or "Sample Pool reports an active lease"
            self._mark_sample_wait(
                "GET_BATCH_OUTCOME_UNKNOWN",
                "GetBatch busy recovery",
                reason,
                1,
            )
            self._reconcile_get_batch_outcome(
                expected,
                reason,
                deadline=deadline,
                ignore_stop=ignore_stop,
            )
            return None
        return response

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
        effective_max_policy_lag = self._effective_max_policy_lag()
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
            full_identity = self._resolvable_model_identity(version)
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
                or lag > effective_max_policy_lag
                or full_identity is None
                or batch.created_at_unix_ms <= 0
                or now_ms - int(batch.created_at_unix_ms)
                > self.max_sample_age_ms
                or not batch.producer.instance_id
                or batch.producer.component != "aiserver"
                or batch.first_action_step > batch.last_action_step
            ):
                raise ValueError("sample batch identity is invalid")
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
        expected_authority: common_pb2.ServiceInstanceIdentity | None = None,
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
        positive_result = response.result in (
            training_pb2.DELIVERY_RESULT_APPLIED,
            training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
        )
        if (response.ret_code == 0) != positive_result:
            raise _UpdateCommitOutcomeUncertain(
                "sample ACK response is contradictory: "
                f"ret_code={response.ret_code}, result={response.result}"
            )
        if not positive_result:
            if response.delivery_id != delivery_id:
                raise _UpdateCommitOutcomeUncertain(
                    "sample ACK negative response does not identify the "
                    "requested delivery"
                )
            if expected_authority is None:
                raise _UpdateCommitOutcomeUncertain(
                    "sample ACK negative response has no pinned lease authority"
                )
            try:
                self._assert_lease_authority(expected_authority)
            except (grpc.RpcError, RuntimeError) as error:
                raise _UpdateCommitOutcomeUncertain(
                    "sample ACK negative response cannot be bound to the "
                    "pinned lease authority"
                ) from error
            raise RuntimeError(f"sample ACK failed: {response.message}")
        if (
            response.delivery_id != delivery_id
            or int(response.disposition) != int(disposition)
            or response.train_update_id != train_update_id
        ):
            raise _UpdateCommitOutcomeUncertain(
                "sample ACK positive response does not echo the exact request"
            )

    def _ack_idempotently(
        self,
        delivery_id: str,
        disposition: int,
        train_update_id: str,
        expected_authority: common_pb2.ServiceInstanceIdentity,
    ) -> int:
        last_error: Exception | None = None
        saw_uncertain_outcome = False
        for attempt in range(1, self.UPDATE_COMMIT_MAX_ATTEMPTS + 1):
            try:
                self._ack(
                    delivery_id,
                    disposition,
                    train_update_id,
                    expected_authority,
                )
                return attempt
            except Exception as error:
                last_error = error
                saw_uncertain_outcome = saw_uncertain_outcome or isinstance(
                    error, (grpc.RpcError, _UpdateCommitOutcomeUncertain)
                )
                if attempt == self.UPDATE_COMMIT_MAX_ATTEMPTS:
                    break
                self.logger.warning(
                    "sample ACK attempt %d failed; retrying exact delivery/update: %s",
                    attempt,
                    error,
                )
                time.sleep(self.UPDATE_COMMIT_RETRY_DELAY_SEC)
        if not saw_uncertain_outcome:
            raise RuntimeError(
                "sample ACK was not applied after "
                f"{self.UPDATE_COMMIT_MAX_ATTEMPTS} attempts: {last_error}"
            ) from last_error
        raise _UpdateCommitOutcomeUncertain(
            "sample ACK outcome is uncertain after "
            f"{self.UPDATE_COMMIT_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

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

    def _capture_update_rollback(self, update_id: str) -> dict:
        base_version = int(self.trainer.model_version)
        target_version = base_version + 1
        rollback_path = (
            self.publisher.checkpoint_dir / f".{update_id}.rollback.pt"
        )
        if rollback_path.exists():
            raise RuntimeError(
                f"update rollback checkpoint already exists: {rollback_path}"
            )
        candidate_paths = (
            self.publisher.checkpoint_path(target_version),
            self.publisher.model_path(target_version),
            self.publisher.manifest_path(target_version),
        )
        temporary_paths = tuple(
            path.with_name(f".{path.name}.{os.getpid()}.tmp")
            for path in (
                *candidate_paths,
                self.publisher.state_path,
                self.publisher.receipt_path(update_id),
            )
        )
        state_exists = self.publisher.state_path.is_file()
        context = {
            "base_version": base_version,
            "target_version": target_version,
            "rollback_path": rollback_path,
            "model_training": bool(self.trainer.model.training),
            "cuda_rng_state": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            "state_exists": state_exists,
            "state_document": (
                copy.deepcopy(read_json(self.publisher.state_path))
                if state_exists
                else None
            ),
            "tracked_paths": {
                path: path.exists()
                for path in (*candidate_paths, *temporary_paths)
            },
            "manifest_present": target_version in self.model_manifests,
            "manifest_document": copy.deepcopy(
                self.model_manifests.get(target_version)
            ),
            "retained": False,
        }
        try:
            self.trainer.save_checkpoint(
                str(rollback_path),
                metadata={
                    "transaction_role": "pre-update-rollback",
                    "train_update_id": update_id,
                },
            )
            with rollback_path.open("rb") as stream:
                os.fsync(stream.fileno())
        except Exception:
            rollback_path.unlink(missing_ok=True)
            raise
        return context

    @staticmethod
    def _discard_update_rollback(context: dict | None) -> None:
        if context is not None:
            context["rollback_path"].unlink(missing_ok=True)

    def _restore_update_rollback(self, context: dict) -> None:
        if not self.trainer.load_checkpoint(str(context["rollback_path"])):
            raise RuntimeError("pre-update rollback checkpoint could not be loaded")
        self.trainer.model.train(bool(context["model_training"]))
        if context["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(context["cuda_rng_state"])
        if int(self.trainer.model_version) != int(context["base_version"]):
            raise RuntimeError("pre-update model version was not restored")

        for path, existed in context["tracked_paths"].items():
            if not existed:
                path.unlink(missing_ok=True)
        if context["state_exists"]:
            atomic_write_json(
                self.publisher.state_path,
                context["state_document"],
            )
        else:
            self.publisher.state_path.unlink(missing_ok=True)

        target_version = int(context["target_version"])
        if context["manifest_present"]:
            self.model_manifests[target_version] = context[
                "manifest_document"
            ]
        else:
            self.model_manifests.pop(target_version, None)
        self._discard_update_rollback(context)

    def _train_delivery(self, response, *, allow_partial: bool = False) -> None:
        behavior_identity = self._validate_delivery(
            response, allow_partial=allow_partial
        )
        delivery_authority = self._sample_distributor_authority(
            response.distributor
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
            "sample_distributor_authority": self._authority_document(
                delivery_authority
            ),
        }
        self._write_receipt(update_id, receipt)
        renewer = LeaseRenewer(
            self.sample_stub,
            self.learner_service,
            response.delivery_id,
            self.lease_timeout_ms,
            lambda: self._assert_lease_authority(delivery_authority),
        ).start()
        rollback: dict | None = None
        training_succeeded = False
        register_started = False
        model_registered = False
        ack_started = False
        sample_acked = False
        transaction_complete = False
        manifest: dict | None = None
        stats: dict = {}
        training_samples: list[dict] = []
        target_updates = self.train_updates
        target_samples = self.trained_samples
        try:
            try:
                rollback = self._capture_update_rollback(update_id)
            except Exception as error:
                raise RuntimeError(
                    f"pre-update rollback capture failed: {error}"
                ) from error
            training_samples = self._training_samples(response.batches)
            stats = self.trainer.train_on_batch(training_samples)
            training_succeeded = True
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
            if renewer.error:
                raise RuntimeError(f"sample lease lost: {renewer.error}")
            try:
                renewer.renew_now()
            except Exception as error:
                raise RuntimeError(
                    "sample lease health check before model registration failed"
                ) from error
            if renewer.error:
                raise RuntimeError(f"sample lease lost: {renewer.error}")
            with self._metrics_lock:
                pinned_authority = self._pin_model_distributor_authority()
                receipt["model_distributor_authority"] = (
                    self._authority_document(pinned_authority)
                )
                self._write_receipt(update_id, receipt)
                register_started = True
                (
                    receipt["register_attempts"],
                    resolved_authority,
                ) = self._register_idempotently(
                    manifest,
                    lambda: str(renewer.error or ""),
                    pinned_authority,
                )
                receipt["model_distributor_resolved_authority"] = (
                    self._authority_document(resolved_authority)
                )
                model_registered = True
                self._discard_update_rollback(rollback)
                rollback = None
                receipt["state"] = "REGISTERED"
                self._write_receipt(update_id, receipt)
                ack_started = True
                receipt["ack_attempts"] = self._ack_idempotently(
                    response.delivery_id,
                    training_pb2.ACK_DISPOSITION_TRAINED,
                    update_id,
                    delivery_authority,
                )
                sample_acked = True
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
                transaction_complete = True
            if self.publisher.should_archive(
                target_updates - self._run_start_train_updates
            ):
                self.publisher.archive_version(
                    self.trainer.model_version, "interval"
                )
                self._last_archive_version = self.trainer.model_version
            self.publisher.prune_runtime(self.trainer.model_version)
        except Exception as error:
            failed_state = str(receipt.get("state", "LEASED"))
            if training_succeeded and not model_registered:
                if register_started:
                    receipt.setdefault(
                        "register_attempts",
                        int(getattr(error, "attempts", 0)),
                    )
                if isinstance(error, _UpdateCommitOutcomeUncertain):
                    if rollback is not None:
                        rollback["retained"] = True
                    receipt.update(
                        {
                            "state": "REGISTER_PENDING",
                            "failed_state": failed_state,
                            "last_error": str(error),
                            "recovery_checkpoint": (
                                str(rollback["rollback_path"])
                                if rollback is not None
                                else ""
                            ),
                        }
                    )
                    self._write_receipt(update_id, receipt)
                    if manifest is not None:
                        self._commit_learner_metrics(
                            manifest,
                            behavior_model=behavior_identity,
                            actual_batch_size=len(training_samples),
                            disposition="REGISTER_PENDING",
                            train_update_id=update_id,
                            train_updates=target_updates,
                            trained_samples=target_samples,
                            stats=stats,
                        )
                else:
                    try:
                        if rollback is None:
                            raise RuntimeError(
                                "pre-update rollback checkpoint is unavailable"
                            )
                        self._restore_update_rollback(rollback)
                        rollback = None
                    except Exception as rollback_error:
                        if rollback is not None:
                            rollback["retained"] = True
                        receipt.update(
                            {
                                "state": "ROLLBACK_FAILED",
                                "failed_state": failed_state,
                                "last_error": str(error),
                                "rollback_error": str(rollback_error),
                            }
                        )
                        self._write_receipt(update_id, receipt)
                        raise RuntimeError(
                            "learner update and rollback both failed"
                        ) from rollback_error
                    receipt.update(
                        {
                            "state": "ROLLED_BACK",
                            "failed_state": failed_state,
                            "last_error": str(error),
                            "restored_model_version": self.trainer.model_version,
                        }
                    )
                    self._write_receipt(update_id, receipt)
                    self._nack(response.delivery_id, "learner update rolled back")
            elif model_registered and not transaction_complete:
                if ack_started:
                    receipt.setdefault(
                        "ack_attempts", self.UPDATE_COMMIT_MAX_ATTEMPTS
                    )
                if sample_acked:
                    pending_state = "ACKED_COMMIT_PENDING"
                elif not ack_started:
                    pending_state = "REGISTERED_COMMIT_PENDING"
                elif isinstance(error, _UpdateCommitOutcomeUncertain):
                    pending_state = "ACK_PENDING"
                else:
                    pending_state = "ACK_REJECTED"
                receipt.update(
                    {
                        "state": pending_state,
                        "failed_state": failed_state,
                        "last_error": str(error),
                    }
                )
                self._write_receipt(update_id, receipt)
                if manifest is not None:
                    self._commit_learner_metrics(
                        manifest,
                        behavior_model=behavior_identity,
                        actual_batch_size=len(training_samples),
                        disposition=pending_state,
                        train_update_id=update_id,
                        train_updates=target_updates,
                        trained_samples=target_samples,
                        stats=stats,
                    )
            elif (
                not training_succeeded
                and not isinstance(error, ValueError)
                and receipt.get("state") == "LEASED"
            ):
                self._nack(response.delivery_id, "learner update failed")
            if training_succeeded and isinstance(error, ValueError):
                raise RuntimeError(
                    f"learner update commit failed: {error}"
                ) from error
            raise
        finally:
            renewer.close()
            if rollback is not None and not rollback.get("retained"):
                self._discard_update_rollback(rollback)

    @staticmethod
    def _metric_snapshot(
        snapshot: training_pb2.MetricSnapshot,
    ) -> tuple[dict, dict, dict, dict]:
        descriptors = {item.field_id: item for item in snapshot.descriptors}
        values: dict[str, float] = {}
        labels: dict[str, str] = {}
        statistics: dict[str, dict] = {}
        descriptor_documents: dict[str, dict] = {}
        for item in snapshot.values:
            descriptor = descriptors.get(item.field_id)
            if descriptor is None:
                continue
            value = float(item.value)
            if not math.isfinite(value):
                continue
            values[item.field_id] = value
            labels[item.field_id] = descriptor.label
            descriptor_documents[item.field_id] = {
                "field_id": descriptor.field_id,
                "label": descriptor.label,
                "group": descriptor.group,
                "dimension": descriptor.dimension,
                "unit": descriptor.unit,
                "scope": descriptor.scope,
                "statistic": descriptor.statistic,
                "value_kind": training_pb2.MetricValueKind.Name(
                    descriptor.value_kind
                ),
                "owner_component": descriptor.owner_component,
                "aggregation_kind": training_pb2.MetricAggregationKind.Name(
                    descriptor.aggregation_kind
                ),
                "window_kind": training_pb2.MetricWindowKind.Name(
                    descriptor.window_kind
                ),
                "schema_identity": {
                    "schema_id": descriptor.schema_identity.schema_id,
                    "schema_version": int(
                        descriptor.schema_identity.schema_version
                    ),
                    "canonical_digest": {
                        "algorithm": common_pb2.DigestAlgorithm.Name(
                            descriptor.schema_identity.canonical_digest.algorithm
                        ),
                        "hex": descriptor.schema_identity.canonical_digest.hex,
                    },
                },
            }
            statistic = {
                "value": value,
                "scope": descriptor.scope,
                "statistic": descriptor.statistic,
                "aggregation_kind": training_pb2.MetricAggregationKind.Name(
                    descriptor.aggregation_kind
                ),
                "window_kind": training_pb2.MetricWindowKind.Name(
                    descriptor.window_kind
                ),
                "window_start_unix_ms": int(item.window_start_unix_ms),
                "window_end_unix_ms": int(item.window_end_unix_ms),
            }
            count = int(item.count)
            raw_sum = float(item.sum)
            if count > 0:
                if not math.isfinite(raw_sum):
                    raise ValueError(
                        f"metric {item.field_id} has a non-finite raw sum"
                    )
                statistic["sum"] = raw_sum
                statistic["count"] = count
            statistics[item.field_id] = statistic
        return values, labels, statistics, descriptor_documents

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
            values, labels, statistics, descriptors = self._metric_snapshot(
                status.metrics
            )
            reward_components: dict[str, float] = {}
            transition_reward_components: dict[str, float] = {}
            latest_reward_components: dict[str, float] = {}
            prefix = "server.training.reward.component."
            episode_suffix = ".episode_mean.v1"
            transition_suffix = ".transition_mean.v1"
            latest_suffix = ".latest_episode_mean.v1"
            for field_id, value in values.items():
                if not field_id.startswith(prefix):
                    continue
                if field_id.endswith(episode_suffix):
                    reward_components[
                        field_id[len(prefix) : -len(episode_suffix)]
                    ] = value
                elif field_id.endswith(transition_suffix):
                    transition_reward_components[
                        field_id[len(prefix) : -len(transition_suffix)]
                    ] = value
                elif field_id.endswith(latest_suffix):
                    latest_reward_components[
                        field_id[len(prefix) : -len(latest_suffix)]
                    ] = value
            if not transition_reward_components:
                legacy_prefix = "server.reward.component."
                for field_id, value in values.items():
                    if field_id.startswith(legacy_prefix) and field_id.endswith(
                        transition_suffix
                    ):
                        transition_reward_components[
                            field_id[
                                len(legacy_prefix) : -len(transition_suffix)
                            ]
                        ] = value
            episode_return = values.get(
                "server.training.episode.learning_return.mean.v1"
            )
            latest_episode_return = values.get(
                "server.training.episode.learning_return.latest_mean.v1"
            )
            success_value = values.get(
                "server.training.episode.success.agent_rate.v1"
            )
            if success_value is not None and success_value > 1.0:
                success_value /= 100.0
            return {
                "ready": bool(status.ready),
                "state": training_pb2.AIServerState.Name(status.state),
                "instance_id": status.aiserver.instance_id,
                "lifecycle_epoch": int(status.aiserver.lifecycle_epoch),
                "active_sessions": int(
                    status.active_actor_session_count
                ),
                "active_trajectories": int(
                    status.active_trajectory_count
                ),
                "client_session_recent": (
                    values.get("server.client.session_recent.v1") == 1.0
                ),
                "client_last_activity_unix_ms": int(
                    values.get(
                        "server.client.last_activity_unix_ms.v1", 0.0
                    )
                ),
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
                "producer_stale_before_ingress": int(
                    status.producer_stale_count
                ),
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
                "metric_statistics": statistics,
                "metric_descriptors": descriptors,
                "episodes": {
                    "mean_agent_return": episode_return,
                    "latest_agent_return": latest_episode_return,
                    "min_agent_return": values.get(
                        "server.episode.learning_return.min.v1",
                    ),
                    "max_agent_return": values.get(
                        "server.episode.learning_return.max.v1",
                    ),
                    "agent_success_rate": success_value,
                    "any_success_rate": values.get(
                        "server.training.episode.success.any_rate.v1"
                    ),
                    "all_success_rate": values.get(
                        "server.training.episode.success.all_rate.v1"
                    ),
                    "reward_components": reward_components,
                    "transition_reward_components": (
                        transition_reward_components
                    ),
                    "latest_reward_components": latest_reward_components,
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
        # Observability is intentionally outside the PPO transaction lock.
        # Component RPC latency and metrics-file I/O must never block model
        # update, registration, or sample settlement.
        with self._metrics_poll_lock:
            actor = self._actor_snapshot()
            distributor = self._distributor_snapshot()
            model = self._model_snapshot()
            now = time.time()
            learner = self._learner_metrics_snapshot()
            with self._metrics_lock:
                context = copy.deepcopy(self._metrics_context)
                self.sequence += 1
                sequence = self.sequence
                rates = self._rates(actor, distributor, now)
            chain = training_chain_status(
                actor,
                distributor,
                learner,
                model,
                str(context.get("error", "")),
            )
            record = {
                "schema_version": 3,
                "mode": "training",
                "sequence": sequence,
                "timestamp": now,
                "interval_ms": 1000,
                "contract": contract_document(self.contract),
                "training_semantics": semantics_document(self.semantics),
                "learner": learner,
                "actor": actor,
                "distributor": distributor,
                "model": model,
                "chain": chain,
                "rates": rates,
                "resources": {"learner": self._resource_snapshot()},
            }
            self.metrics_backend.write(record)
            with self._metrics_lock:
                self._last_actor_snapshot = actor
                self._last_distributor_snapshot = distributor
                self._last_model_snapshot = model

    def _record_metrics_best_effort(self, phase: str) -> None:
        try:
            self._record_metrics()
        except Exception as error:
            self.logger.error(
                "metrics snapshot failed during %s: %s", phase, error
            )

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

    def _shutdown_sample_authority(
        self,
        ready_authority: common_pb2.ServiceInstanceIdentity | None,
    ) -> common_pb2.ServiceInstanceIdentity | None:
        if self._demand_active:
            if self._demand_authority is None:
                raise RuntimeError(
                    "active sample demand has no shutdown authority"
                )
            return self._sample_distributor_authority(
                self._demand_authority
            )
        if ready_authority is None:
            return None
        return self._sample_distributor_authority(ready_authority)

    def _shutdown_drain(
        self,
        expected_authority: common_pb2.ServiceInstanceIdentity | None,
    ) -> None:
        minimum_reconcile_sec = (
            self.lease_timeout_ms / 1000.0
            + self.SHUTDOWN_RECONCILE_MARGIN_SEC
        )
        budget_sec = max(
            self.shutdown_drain_timeout_ms / 1000.0,
            minimum_reconcile_sec,
        )
        deadline = time.monotonic() + budget_sec
        pinned = (
            self._sample_distributor_authority(expected_authority)
            if expected_authority is not None
            else None
        )
        attempt = 0
        last_counts = (0, 0, 0)
        while time.monotonic() < deadline:
            attempt += 1
            try:
                status = self._sample_pool_status()
                if not _same_message(status.contract, self.contract):
                    raise RuntimeError(
                        "sample distributor contract changed during shutdown"
                    )
                authority = self._sample_distributor_authority(
                    status.distributor
                )
                if pinned is None:
                    pinned = authority
                elif not self._same_authority(authority, pinned):
                    raise RuntimeError(
                        "sample distributor authority changed during shutdown"
                    )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown sample status",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue

            ready_samples = int(status.ready_queue_samples)
            leased_samples = int(status.leased_samples)
            reserved_samples = int(status.reserved_samples)
            reserved_fragments = int(status.reserved_fragments)
            reserved_bytes = int(status.reserved_estimated_bytes)
            if (
                min(
                    ready_samples,
                    leased_samples,
                    reserved_samples,
                    reserved_fragments,
                    reserved_bytes,
                )
                < 0
                or (reserved_samples == 0 and reserved_fragments != 0)
                or (reserved_samples == 0 and reserved_bytes != 0)
                or (reserved_samples > 0 and reserved_fragments == 0)
            ):
                raise RuntimeError(
                    "sample distributor returned contradictory shutdown counts"
                )
            last_counts = (
                ready_samples,
                leased_samples,
                reserved_samples,
            )
            if not status.ready or not status.pool_ready:
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown sample readiness",
                    "sample distributor is not ready",
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue
            if last_counts == (0, 0, 0):
                self._clear_sample_wait()
                return
            if leased_samples > 0:
                reconciled = self._reconcile_get_batch_outcome(
                    pinned,
                    "shutdown observed an unresolved delivery",
                    deadline=deadline,
                    ignore_stop=True,
                )
                if not reconciled:
                    break
                continue
            if ready_samples == 0:
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown sample reservation",
                    f"waiting for {reserved_samples} reserved samples",
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue
            response = self._get_batch_recovering(
                mode=training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE,
                ready_authority=pinned,
                deadline=deadline,
                ignore_stop=True,
            )
            if response is None:
                continue
            if response.result == training_pb2.GET_BATCH_RESULT_LEASED:
                self._ack(
                    response.delivery_id,
                    training_pb2.ACK_DISPOSITION_SHUTDOWN_UNTRAINED,
                    expected_authority=self._sample_distributor_authority(
                        response.distributor
                    ),
                )
                continue
            if response.result != training_pb2.GET_BATCH_RESULT_TIMEOUT:
                raise RuntimeError(
                    f"shutdown sample drain failed: {response.message}"
                )
            self._wait_sample_retry(
                attempt, deadline, ignore_stop=True
            )
        raise RuntimeError(
            "shutdown sample drain did not settle: "
            f"ready={last_counts[0]}, leased={last_counts[1]}, "
            f"reserved={last_counts[2]}"
        )

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
                self.finalize_complete_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                self.finalize_complete_path.write_text(
                    f"{self.train_updates} {self.trained_samples}\n",
                    encoding="utf-8",
                )
                self._finalized = True
                return
            authority = self._ready_sample_distributor_authority(status)
            response = self._get_batch_recovering(
                mode=training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE,
                ready_authority=authority,
                deadline=deadline,
            )
            if response is None:
                continue
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
        try:
            self._start_metrics()
            self._initialize_models()
            ready_authority = self._wait_for_sample_pool(
                force_demand=True
            )
            while not _stop_requested.is_set():
                if ready_authority is None:
                    break
                if self.finalize_request_path.is_file():
                    self._finalize_training()
                    while not _stop_requested.wait(0.2):
                        pass
                    break
                response = self._get_batch_recovering(
                    ready_authority=ready_authority
                )
                if response is None:
                    ready_authority = self._wait_for_sample_pool()
                    continue
                if response.result == training_pb2.GET_BATCH_RESULT_TIMEOUT:
                    ready_authority = self._wait_for_sample_pool()
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
                        expected_authority=self._sample_distributor_authority(
                            response.distributor
                        ),
                    )
                    raise RuntimeError(str(error)) from error
                ready_authority = self._wait_for_sample_pool()
            with self._metrics_lock:
                self._metrics_context["disposition"] = "STOPPING"
            shutdown_authority = self._shutdown_sample_authority(
                ready_authority
            )
            self._release_demand(required=True)
            self._shutdown_drain(shutdown_authority)
            if (
                self.publisher.archive_on_graceful_shutdown
                and self._last_archive_version != self.trainer.model_version
            ):
                self.publisher.archive_version(
                    self.trainer.model_version, "graceful-shutdown"
                )
            self._record_metrics_best_effort("graceful shutdown")
            return 0
        except Exception as error:
            with self._metrics_lock:
                self._metrics_context["error"] = str(error)
                self._metrics_context["disposition"] = "FAILED"
            self.logger.exception("Learner training failed")
            self._record_metrics_best_effort("failure reporting")
            return 1
        finally:
            self._release_demand(required=False)
            self._stop_metrics()
            try:
                self.metrics_backend.close()
            except Exception as error:
                self.logger.error("metrics backend close failed: %s", error)
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
