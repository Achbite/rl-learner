"""Task-neutral leased-sample PPO runtime for rl-contracts 0.13.0."""

from __future__ import annotations

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import common_pb2, training_pb2, training_pb2_grpc
from src.contracts.identity import (
    bind_runtime_lineage,
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
from src.config.command_line import parse_startup_arguments
from src.config.effective_config import effective_config_log, load_effective_config
from src.log.logger import setup_logger
from src.metrics.metric_events import (
    AIServerMetricRelay,
    LocalMetricProjector,
    LocalTrainUpdateMetricWriter,
    MetricEventContractError,
    MetricSchemaCatalog,
    RawMetricBatchStore,
    default_metric_schema_directory,
)
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
    return load_effective_config(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_dict(document: dict | None) -> dict:
    if not document:
        return {}
    identity = document.get("identity", document)
    return {
        "model_lineage_id": str(identity.get("model_lineage_id", "")),
        "model_step": int(identity.get("model_step", -1)),
        "artifact_digest": str(identity.get("artifact_digest", "")),
        "manifest_digest": str(identity.get("manifest_digest", "")),
    }


def _identity_equal(left: dict | None, right: dict | None) -> bool:
    return bool(left and right) and _identity_dict(left) == _identity_dict(right)


def training_chain_status(
    actor: dict,
    sample_pool: dict,
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
        ("sample_pool", sample_pool),
        ("model_distributor", model),
    ):
        if document.get("error"):
            reasons.append(f"{name}_status_error")
        if not document.get("ready"):
            reasons.append(f"{name}_not_ready")
        if not document.get("instance_id"):
            reasons.append(f"{name}_instance_missing")
    for field in ("ingress_ready", "pool_ready"):
        if not sample_pool.get(field):
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

    learner_step = int(_identity_dict(learner_model).get("model_step", -1))
    actor_step = int(_identity_dict(actor_model).get("model_step", -1))
    model_lag = learner_step - actor_step
    if learner_step < 0 or actor_step < 0 or not 0 <= model_lag <= 1:
        reasons.append("actor_model_lag_invalid")
    actual_batch_size = int(learner.get("actual_batch_size", 0) or 0)
    if actual_batch_size:
        policy_lag = int(learner.get("policy_lag", -1))
        maximum = int(learner.get("max_policy_lag", -1))
        if policy_lag < 0 or maximum < 0 or policy_lag > maximum:
            reasons.append("training_policy_lag_invalid")

    actor_state = str(actor.get("state", ""))
    actor_lifecycle_ready = (
        actor_state == "AISERVER_STATE_READY"
        or (not actor_state and actor.get("ready") is True)
    )
    server_pod_reasons: list[str] = []
    if actor.get("error"):
        server_pod_reasons.append("actor_status_error")
    if not actor.get("instance_id"):
        server_pod_reasons.append("actor_instance_missing")
    if not actor_lifecycle_ready:
        server_pod_reasons.append("actor_lifecycle_not_ready")
    if actor_step < 0:
        server_pod_reasons.append("active_model_missing")
    if actor.get("client_session_recent") is not True:
        server_pod_reasons.append("client_session_not_recent")

    staged_step = int(
        _identity_dict(actor.get("staged_model_identity", {})).get(
            "model_step", -1
        )
    )
    latest_step = int(
        _identity_dict(published_model).get("model_step", -1)
    )
    if actor_step < 0:
        model_sync_state = "waiting_for_initial_model"
        model_sync_lag = None
    elif latest_step < 0:
        model_sync_state = "unknown"
        model_sync_lag = None
    elif actor_step < latest_step:
        model_sync_state = "catching_up"
        model_sync_lag = latest_step - actor_step
    elif actor_step == latest_step:
        model_sync_state = "synchronized"
        model_sync_lag = 0
    else:
        model_sync_state = "actor_ahead"
        model_sync_lag = actor_step - latest_step
    return {
        "ready": not reasons,
        "state": "ready" if not reasons else "degraded",
        "reasons": reasons,
        "model_lag": model_lag,
        "server_pod": {
            "ready": not server_pod_reasons,
            "state": (
                "running" if not server_pod_reasons else "not_ready"
            ),
            "reasons": server_pod_reasons,
        },
        "model_sync": {
            "state": model_sync_state,
            "active_model_step": (
                None if actor_step < 0 else actor_step
            ),
            "staged_model_step": (
                None if staged_step < 0 else staged_step
            ),
            "latest_model_step": (
                None if latest_step < 0 else latest_step
            ),
            "lag": model_sync_lag,
        },
        "error": error,
    }


class ModelPublisher:
    MODEL_FILE = "SaveModel.onnx"
    MANIFEST_FILE = "manifest.json"
    METADATA_FILE = "metadata.json"
    MIN_ROLLING_PUBLICATIONS = 101
    MAX_MODEL_STEP = (1 << 64) - 1
    PROVENANCE_KEYS = (
        "initial_model_path",
        "initial_model_artifact_digest",
    )
    LOCAL_METADATA_KEYS = {
        "schema_version",
        "model_identity",
        "training_config_digest",
        "train_update_id",
        "behavior_model",
        "batch_ids",
        "stats",
        "sample_count",
        "train_updates",
        "trained_samples",
    }
    WIRE_MANIFEST_KEYS = {
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
    }

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
        self.local_train_root = Path(str(model["local_train_dir"])).resolve()
        self.runtime_dir = self.local_train_root / "runtime"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.update_dir = self.runtime_dir / "receipts"
        self.state_path = self.runtime_dir / "state.json"
        self.publication_dir = self.local_train_root
        self.metrics_dir = self.local_train_root / "metrics"
        self.archive_interval_updates = int(model["archive_interval_updates"])
        self.publication_retention_steps = int(
            model["publication_retention_steps"]
        )
        if self.archive_interval_updates <= 0:
            raise ValueError("archive_interval_updates must be positive")
        if (
            self.publication_retention_steps
            != self.MIN_ROLLING_PUBLICATIONS
        ):
            raise ValueError(
                "publication_retention_steps must be exactly 101"
            )
        self.initial_model_provenance: dict = {}
        self._prepared = False

    def prepare(self) -> str:
        for directory in (
            self.checkpoint_dir,
            self.update_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._recover_private_publication_directories()
        self._validate_storage_layout()
        has_state = self.state_path.exists() or self.state_path.is_symlink()
        has_checkpoint = any(self.checkpoint_dir.iterdir())
        has_receipt = any(self.update_dir.iterdir())
        has_publication = bool(self._canonical_step_directories())
        if any((has_state, has_checkpoint, has_receipt, has_publication)):
            raise RuntimeError(
                "training workspace contains previous state; start through "
                "run.sh with a fresh isolated workspace"
            )
        self._prepared = True
        return "fresh"

    def _validate_storage_layout(self) -> None:
        allowed_root_entries = {"runtime", "metrics"}
        for path in self.local_train_root.iterdir():
            if path.name in allowed_root_entries:
                continue
            if path.name.isdigit():
                step = int(path.name)
                if (
                    path.name == self.step_name(step)
                    and not path.is_symlink()
                    and path.is_dir()
                ):
                    continue
            if path.name not in allowed_root_entries:
                raise RuntimeError(
                    "unknown training workspace entry requires review: "
                    f"{path}"
                )
        if self.runtime_dir.is_symlink() or not self.runtime_dir.is_dir():
            raise RuntimeError("runtime storage must be a regular directory")
        runtime_entries = {path.name: path for path in self.runtime_dir.iterdir()}
        if set(runtime_entries) - {"state.json", "checkpoints", "receipts"}:
            raise RuntimeError("runtime storage contains unknown entries")
        for name, expected in (
            ("checkpoints", self.checkpoint_dir),
            ("receipts", self.update_dir),
        ):
            path = runtime_entries.get(name, expected)
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"runtime {name} must be a regular directory")
        if self.state_path.is_symlink():
            raise RuntimeError("runtime state must not be a symbolic link")

    @classmethod
    def step_name(cls, step: int) -> str:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step > cls.MAX_MODEL_STEP
        ):
            raise ValueError("model step must be a uint64 integer")
        return f"{step:07d}"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _is_canonical_publication_path(
        self, path: Path, step: int
    ) -> bool:
        return (
            path.parent == self.publication_dir
            and path.name == self.step_name(step)
            and not path.is_symlink()
        )

    def _private_publication_directory_step(
        self, path: Path
    ) -> int | None:
        if path.parent != self.publication_dir or path.is_symlink():
            return None
        for prefix in (
            ".publication-",
            ".prune-",
            ".rollback-delete-",
        ):
            if not path.name.startswith(prefix):
                continue
            remainder = path.name[len(prefix) :]
            token, separator, _suffix = remainder.partition("-")
            if not separator or not token.isdigit():
                return None
            step = int(token)
            try:
                if token != self.step_name(step):
                    return None
            except ValueError:
                return None
            return step
        return None

    def _recover_private_publication_directories(self) -> None:
        for path in tuple(self.publication_dir.iterdir()):
            if not path.name.startswith("."):
                continue
            if self._private_publication_directory_step(path) is None:
                raise RuntimeError(
                    "unknown private publication entry requires review: "
                    f"{path}"
                )
            if not path.is_dir():
                raise RuntimeError(
                    f"private publication entry is not a directory: {path}"
                )
            shutil.rmtree(path)
        self._fsync_directory(self.publication_dir)

    def _temporary_publication_path(self, step: int) -> Path:
        return self.publication_dir / (
            f".publication-{self.step_name(step)}-"
            f"{os.getpid()}-{time.time_ns()}.tmp"
        )

    def checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / (
            f"publication-{self.step_name(step)}.checkpoint.pt"
        )

    def model_path(self, step: int) -> Path:
        return self.publication_path(step) / self.MODEL_FILE

    def manifest_path(self, step: int) -> Path:
        return self.publication_path(step) / self.MANIFEST_FILE

    def publication_path(self, step: int) -> Path:
        return self.publication_dir / self.step_name(step)

    def metadata_path(self, step: int) -> Path:
        return self.publication_path(step) / self.METADATA_FILE

    def receipt_path(self, train_update_id: str) -> Path:
        return self.update_dir / f"{train_update_id}.json"

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def load_initial_model(
        self, trainer: PPOTrainer, model_value: str
    ) -> dict:
        requested = Path(model_value).expanduser()
        if requested.is_symlink():
            raise RuntimeError("initial model must not be a symlink")
        try:
            model_path = requested.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(
                f"initial model does not exist: {requested}"
            ) from error
        if not model_path.is_file() or model_path.name != self.MODEL_FILE:
            raise RuntimeError(
                f"initial model must be an explicit {self.MODEL_FILE} file"
            )
        if model_path.parent == self.local_train_root or (
            self.local_train_root in model_path.parents
        ):
            raise RuntimeError(
                "initial model must be outside the fresh training workspace"
            )
        if not trainer.load_onnx_weights(str(model_path)):
            raise RuntimeError("initial model weights could not be loaded")
        self.initial_model_provenance = {
            "initial_model_path": str(model_path),
            "initial_model_artifact_digest": sha256_file(model_path),
        }
        return dict(self.initial_model_provenance)

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
            "model_architecture_id": self.semantics.model_architecture_id,
            "tensor_dtype": self.tensor_dtype,
            "training_config_digest": self.training_digest.hex,
            **self.initial_model_provenance,
        }

    def commit_optimizer_checkpoint(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model: dict | None,
        batch_ids: list[str],
        stats: dict,
        sample_count: int,
        train_updates: int,
        trained_samples: int,
    ) -> Path:
        if not self._prepared:
            raise RuntimeError("model publisher is not prepared")
        if int(trainer.model_step) != int(train_updates):
            raise RuntimeError("model_step must equal train_updates")
        if int(trained_samples) < 0 or int(sample_count) < 0:
            raise RuntimeError("training counters must be non-negative")
        path = self.checkpoint_path(trainer.model_step)
        metadata = self._checkpoint_metadata(
            train_update_id=train_update_id,
            behavior_model=behavior_model,
            batch_ids=batch_ids,
            stats=stats,
            sample_count=sample_count,
            train_updates=train_updates,
            trained_samples=trained_samples,
        )
        if path.is_symlink():
            raise RuntimeError(f"checkpoint must not be a symlink: {path}")
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"checkpoint must be a regular file: {path}")
            checkpoint = self._load_checkpoint(path)
            if (
                checkpoint.get("model_step") != trainer.model_step
                or checkpoint.get("metadata") != metadata
            ):
                raise RuntimeError(f"checkpoint identity conflicts: {path}")
            return path
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}-{time.time_ns()}.tmp"
        )
        try:
            trainer.save_checkpoint(str(temporary), metadata=metadata)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._fsync_directory(self.checkpoint_dir)
            return path
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

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
        step = trainer.model_step
        if int(step) != int(train_updates):
            raise RuntimeError("model_step must equal train_updates")
        if int(trained_samples) < 0 or int(sample_count) < 0:
            raise RuntimeError("training counters must be non-negative")
        if behavior_model:
            if "model_step" in behavior_model:
                minimum_behavior_step = behavior_model["model_step"]
                maximum_behavior_step = behavior_model["model_step"]
            else:
                minimum_behavior_step = behavior_model.get(
                    "minimum_model_step"
                )
                maximum_behavior_step = behavior_model.get(
                    "maximum_model_step"
                )
            if (
                minimum_behavior_step is None
                or maximum_behavior_step is None
                or int(minimum_behavior_step) != int(maximum_behavior_step)
            ):
                raise RuntimeError(
                    "training publication requires one behavior model step"
                )
        target = self.publication_path(step)
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"model publication already exists: {target}")
        temporary = self._temporary_publication_path(step)
        temporary.mkdir(parents=False, exist_ok=False)
        temporary_model = temporary / self.MODEL_FILE
        temporary_manifest = temporary / self.MANIFEST_FILE
        temporary_metadata = temporary / self.METADATA_FILE
        private_checkpoint = self.checkpoint_path(step)
        checkpoint_existed = (
            private_checkpoint.exists() or private_checkpoint.is_symlink()
        )
        checkpoint_created = False
        published = False
        try:
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
                if (
                    private_checkpoint.is_symlink()
                    or not private_checkpoint.is_file()
                ):
                    raise RuntimeError(
                        "precommitted private checkpoint is unavailable: "
                        f"{private_checkpoint}"
                    )
                checkpoint = self._load_checkpoint(private_checkpoint)
                if (
                    checkpoint.get("model_step") != step
                    or checkpoint.get("metadata") != metadata
                ):
                    raise RuntimeError(
                        "precommitted checkpoint identity mismatch"
                    )
            else:
                self.commit_optimizer_checkpoint(
                    trainer,
                    train_update_id=train_update_id,
                    behavior_model=behavior_model,
                    batch_ids=batch_ids,
                    stats=stats or {},
                    sample_count=sample_count,
                    train_updates=train_updates,
                    trained_samples=trained_samples,
                )
                checkpoint_created = not checkpoint_existed
            for path in (temporary_model, private_checkpoint):
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            artifact_digest = sha256_file(temporary_model)
            final_model_path = self.model_path(step)
            document = {
                "manifest_schema_version": 2,
                "contract": contract_document(self.contract),
                "identity": {
                    "model_lineage_id": self.lineage_id,
                    "model_step": step,
                    "artifact_digest": artifact_digest,
                    "manifest_digest": "0" * 64,
                },
                "observation_schema": schema_document(
                    self.semantics.observation_schema
                ),
                "action_schema": schema_document(
                    self.semantics.action_schema
                ),
                "model_architecture_id": self.semantics.model_architecture_id,
                "tensor_dtype": self.tensor_dtype,
                "input_shape": [1, self.obs_dim],
                "action_shape": [1, self.action_dim],
                "value_shape": [1, 1],
                "artifact_uri": final_model_path.as_uri(),
                "model_file": self.MODEL_FILE,
                "size_bytes": temporary_model.stat().st_size,
                "seed": self.seed,
                "train_updates": int(train_updates),
                "trained_samples": int(trained_samples),
                "training_config_digest": self.training_digest.hex,
                "training_semantics": semantics_document(self.semantics),
                "published_at_unix_ms": int(time.time() * 1000),
                "ready": True,
            }
            document = finalize_manifest_digest(document)
            local_metadata = {
                "schema_version": 2,
                "model_identity": document["identity"],
                "training_config_digest": self.training_digest.hex,
                "train_update_id": train_update_id,
                "behavior_model": behavior_model or {},
                "batch_ids": list(batch_ids),
                "stats": dict(stats or {}),
                "sample_count": int(sample_count),
                "train_updates": int(train_updates),
                "trained_samples": int(trained_samples),
                **self.initial_model_provenance,
            }
            retention = self.retention_for_updates(int(train_updates))
            runtime_document = {
                **document,
                "train_update_id": train_update_id,
                "behavior_model": behavior_model or {},
                "batch_ids": list(batch_ids),
                "retention": retention,
                **self.initial_model_provenance,
            }
            atomic_write_json(temporary_manifest, document)
            atomic_write_json(temporary_metadata, local_metadata)
            for path in (
                temporary_model,
                temporary_manifest,
                temporary_metadata,
            ):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(
                        f"publication file is not a regular file: {path}"
                    )
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            self._fsync_directory(temporary)
            os.replace(temporary, target)
            published = True
            self._fsync_directory(self.publication_dir)
            atomic_write_json(
                self.state_path,
                {
                    "schema_version": 1,
                    "latest_model": runtime_document["identity"],
                    "latest_manifest": str(self.manifest_path(step)),
                    "latest_checkpoint": str(self.checkpoint_path(step)),
                    "train_updates": int(train_updates),
                    "trained_samples": int(trained_samples),
                    "updated_at_unix_ms": int(time.time() * 1000),
                    **self.initial_model_provenance,
                },
            )
            return runtime_document
        except Exception:
            if published:
                self.remove_publication_for_rollback(step)
            elif temporary.exists():
                if (
                    self._private_publication_directory_step(temporary)
                    is None
                ):
                    raise RuntimeError(
                        f"refusing to remove unverified temporary path: {temporary}"
                    )
                shutil.rmtree(temporary)
                self._fsync_directory(self.publication_dir)
            if checkpoint_created:
                private_checkpoint.unlink(missing_ok=True)
                self._fsync_directory(self.checkpoint_dir)
            raise

    def complete_manifest(
        self, version: int, train_update_id: str | None = None
    ) -> dict | None:
        publication = self.publication_path(version)
        path = self.manifest_path(version)
        model_path = self.model_path(version)
        checkpoint_path = self.checkpoint_path(version)
        metadata_path = self.metadata_path(version)
        if (
            not self._is_canonical_publication_path(publication, version)
            or not publication.is_dir()
        ):
            return None
        try:
            entries = {entry.name: entry for entry in publication.iterdir()}
        except OSError:
            return None
        required = {
            self.MODEL_FILE,
            self.MANIFEST_FILE,
            self.METADATA_FILE,
        }
        if set(entries) != required or any(
            entry.is_symlink() or not entry.is_file()
            for entry in entries.values()
        ):
            return None
        try:
            document = read_json(path)
            local_metadata = read_json(metadata_path)
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                return None
            checkpoint = self._load_checkpoint(checkpoint_path)
            if set(document) != self.WIRE_MANIFEST_KEYS:
                return None
            expected = finalize_manifest_digest(document)
            artifact_digest = sha256_file(model_path)
            model_size = model_path.stat().st_size
        except (OSError, ValueError, KeyError, RuntimeError, TypeError):
            return None
        metadata = checkpoint.get("metadata", {})
        if not isinstance(local_metadata, dict):
            return None
        local_metadata_keys = set(local_metadata)
        if local_metadata_keys not in (
            self.LOCAL_METADATA_KEYS,
            self.LOCAL_METADATA_KEYS | set(self.PROVENANCE_KEYS),
        ):
            return None
        retention = self.retention_for_updates(
            int(local_metadata.get("train_updates", -1))
        )
        expected_checkpoint_metadata = {
            "train_update_id": local_metadata.get("train_update_id"),
            "behavior_model": local_metadata.get("behavior_model", {}),
            "batch_ids": local_metadata.get("batch_ids", []),
            "stats": local_metadata.get("stats", {}),
            "sample_count": local_metadata.get("sample_count"),
            "train_updates": local_metadata.get("train_updates"),
            "trained_samples": local_metadata.get("trained_samples"),
            "model_lineage_id": self.lineage_id,
            "observation_schema": schema_document(
                self.semantics.observation_schema
            ),
            "action_schema": schema_document(self.semantics.action_schema),
            "model_architecture_id": self.semantics.model_architecture_id,
            "tensor_dtype": self.tensor_dtype,
            "training_config_digest": self.training_digest.hex,
            **{
                key: local_metadata[key]
                for key in self.PROVENANCE_KEYS
                if key in local_metadata
            },
        }
        if (
            document.get("manifest_schema_version") != 2
            or document.get("contract") != contract_document(self.contract)
            or document.get("identity", {}).get("model_lineage_id")
            != self.lineage_id
            or document.get("identity", {}).get("model_step") != version
            or document.get("identity", {}).get("model_step")
            != document.get("train_updates")
            or document.get("identity", {}).get("artifact_digest")
            != artifact_digest
            or document.get("identity", {}).get("manifest_digest")
            != expected["identity"]["manifest_digest"]
            or document.get("artifact_uri") != model_path.as_uri()
            or document.get("model_file") != self.MODEL_FILE
            or document.get("size_bytes") != model_size
            or document.get("training_semantics")
            != semantics_document(self.semantics)
            or document.get("observation_schema")
            != schema_document(self.semantics.observation_schema)
            or document.get("action_schema")
            != schema_document(self.semantics.action_schema)
            or document.get("model_architecture_id")
            != self.semantics.model_architecture_id
            or document.get("tensor_dtype") != self.tensor_dtype
            or document.get("input_shape") != [1, self.obs_dim]
            or document.get("action_shape") != [1, self.action_dim]
            or document.get("value_shape") != [1, 1]
            or document.get("seed") != self.seed
            or document.get("training_config_digest")
            != self.training_digest.hex
            or not document.get("ready")
            or checkpoint.get("model_step") != version
            or metadata != expected_checkpoint_metadata
            or local_metadata.get("schema_version") != 2
            or local_metadata.get("model_identity") != document.get("identity")
            or local_metadata.get("training_config_digest")
            != document.get("training_config_digest")
            or local_metadata.get("training_config_digest")
            != metadata.get("training_config_digest")
            or local_metadata.get("train_updates")
            != document.get("train_updates")
            or local_metadata.get("train_updates") != version
            or local_metadata.get("trained_samples")
            != document.get("trained_samples")
            or local_metadata.get("behavior_model")
            != metadata.get("behavior_model")
            or local_metadata.get("batch_ids") != metadata.get("batch_ids")
            or local_metadata.get("stats") != metadata.get("stats")
            or local_metadata.get("sample_count")
            != metadata.get("sample_count")
            or local_metadata.get("train_updates")
            != metadata.get("train_updates")
            or local_metadata.get("trained_samples")
            != metadata.get("trained_samples")
            or metadata.get("model_lineage_id") != self.lineage_id
            or metadata.get("observation_schema")
            != schema_document(self.semantics.observation_schema)
            or metadata.get("action_schema")
            != schema_document(self.semantics.action_schema)
            or metadata.get("model_architecture_id")
            != self.semantics.model_architecture_id
            or metadata.get("tensor_dtype") != self.tensor_dtype
            or metadata.get("training_config_digest")
            != self.training_digest.hex
            or any(
                local_metadata.get(key) != metadata.get(key)
                for key in self.PROVENANCE_KEYS
            )
            or (
                train_update_id is not None
                and local_metadata.get("train_update_id") != train_update_id
            )
        ):
            return None
        return {
            **document,
            "train_update_id": local_metadata["train_update_id"],
            "behavior_model": local_metadata.get("behavior_model", {}),
            "batch_ids": local_metadata.get("batch_ids", []),
            "retention": retention,
            **{
                key: local_metadata[key]
                for key in self.PROVENANCE_KEYS
                if key in local_metadata
            },
        }

    def complete_manifests(self) -> list[dict]:
        result: list[dict] = []
        for step, _path in self._canonical_step_directories():
            document = self.complete_manifest(step)
            if document:
                result.append(document)
        return result

    def _canonical_step_directories(self) -> list[tuple[int, Path]]:
        candidates: list[tuple[int, Path]] = []
        for path in self.publication_dir.iterdir():
            if not path.is_dir() or path.is_symlink() or not path.name.isdigit():
                continue
            try:
                step = int(path.name)
                if path.name != self.step_name(step):
                    continue
            except ValueError:
                continue
            candidates.append((step, path))
        return sorted(candidates, key=lambda item: item[0])

    def latest_complete_checkpoint(self) -> Path | None:
        manifests = self.complete_manifests()
        if not manifests:
            return None
        version = int(manifests[-1]["identity"]["model_step"])
        return self.checkpoint_path(version)

    def should_mark_permanent(self, run_train_updates: int) -> bool:
        return (
            run_train_updates > 0
            and run_train_updates % self.archive_interval_updates == 0
        )

    def retention_for_updates(self, run_train_updates: int) -> dict:
        if self.should_mark_permanent(run_train_updates):
            return {"class": "permanent", "reason": "interval"}
        return {"class": "rolling", "reason": ""}

    def remove_publication_for_rollback(self, version: int) -> None:
        target = self.publication_path(version)
        if not target.exists():
            return
        manifest = self.complete_manifest(version)
        if (
            manifest is None
            or not self._is_canonical_publication_path(target, version)
        ):
            raise RuntimeError(
                f"refusing to rollback unverified publication: {target}"
            )
        quarantine = self.publication_dir / (
            f".rollback-delete-{self.step_name(version)}-"
            f"{os.getpid()}-{time.time_ns()}"
        )
        os.replace(target, quarantine)
        self._fsync_directory(self.publication_dir)
        shutil.rmtree(quarantine)
        self._fsync_directory(self.publication_dir)

    def prune_publications(
        self, current_step: int, protected_steps: Iterable[int] = ()
    ) -> list[int]:
        current = int(current_step)
        self.step_name(current)
        minimum = max(
            0,
            current - self.publication_retention_steps + 1,
        )
        protected = {int(version) for version in protected_steps}
        protected.add(current)
        removed: list[int] = []
        for version, target in self._canonical_step_directories():
            if version >= minimum or version in protected:
                continue
            verified = self.complete_manifest(version)
            if verified is None:
                raise RuntimeError(
                    f"refusing to prune unverified publication: {target}"
                )
            if verified.get("retention", {}).get("class") == "permanent":
                continue
            if (
                verified.get("retention", {}).get("class") != "rolling"
                or not self._is_canonical_publication_path(target, version)
            ):
                raise RuntimeError(
                    f"refusing to prune invalid publication: {target}"
                )
            quarantine = self.publication_dir / (
                f".prune-{self.step_name(version)}-"
                f"{os.getpid()}-{time.time_ns()}"
            )
            os.replace(target, quarantine)
            self._fsync_directory(self.publication_dir)
            if (
                self._private_publication_directory_step(quarantine)
                != version
            ):
                raise RuntimeError(
                    f"prune quarantine identity mismatch: {quarantine}"
                )
            shutil.rmtree(quarantine)
            self._fsync_directory(self.publication_dir)
            checkpoint_path = self.checkpoint_path(version)
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                raise RuntimeError(
                    f"private checkpoint disappeared during pruning: {checkpoint_path}"
                )
            checkpoint_path.unlink()
            self._fsync_directory(self.checkpoint_dir)
            removed.append(version)
        return removed


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
    SHUTDOWN_RECONCILE_MARGIN_SEC = 5.0

    def __init__(self, config: dict):
        validate_config(config)
        self.config = config
        log = config["log"]
        self.logger = setup_logger(
            "TrainingRuntime",
            console_level=str(log["console_level"]),
            file_level=str(log["file_level"]),
            log_dir=str(log["log_dir"]),
        )
        self.logger.info(
            "effective learner config: %s",
            json.dumps(effective_config_log(config), sort_keys=True),
        )
        self.contract = contract_identity(config)
        self.semantics = training_semantics(config)
        self.policy_digest = policy_spec_digest(config)
        self.trainer = PPOTrainer(config)
        self.publisher = ModelPublisher(config)
        if (
            self.trainer.max_policy_lag + 1
            > self.publisher.publication_retention_steps
        ):
            raise ValueError(
                "max_policy_lag requires more retained model publications"
            )

        learner_name = os.environ.get("RL_LEARNER_INSTANCE", "learner-0")
        instance = f"{learner_name}-{os.getpid()}-{int(time.time() * 1000)}"
        self.learner_service = service_identity("learner", instance, 1)
        self.metrics_source_id = str(
            os.environ.get("RL_METRICS_SOURCE_ID", "")
        )
        self.sequence = 0
        self.train_updates = 0
        self.trained_samples = 0
        self._run_start_train_updates = 0
        self._run_start_trained_samples = 0
        self.initial_model_step = 0
        self.last_stats: dict = {}
        self.model_manifests: dict[int, dict] = {}
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
            "model_step": self.trainer.model_step,
            "train_updates": self.train_updates,
            "run_train_updates": self.train_updates,
            "run_trained_samples": self.trained_samples,
            "policy_lag": 0,
            "max_policy_lag": self.trainer.max_policy_lag,
        }
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._rate_snapshot: dict[str, float] = {}
        self._last_actor_snapshot: dict = {}
        self._last_sample_pool_snapshot: dict = {}
        self._last_model_snapshot: dict = {}
        self._last_resource_time = time.monotonic()
        self._last_process_cpu = time.process_time()

        self.publisher.prepare()
        model_path = str(config["model"].get("initial_model_path") or "")
        if model_path:
            self._startup_mode = "inherited-weights"
            self.publisher.load_initial_model(self.trainer, model_path)
        else:
            self._startup_mode = "fresh"
        self._run_start_train_updates = 0
        self._run_start_trained_samples = 0

        sample = config["sample_pool"]
        sample_host = str(sample["host"])
        sample_port = int(sample["port"])
        self.train_batch_size = int(sample["train_batch_size"])
        self.max_train_batch_size = int(sample["max_train_batch_size"])
        self.max_sample_age_ms = int(sample["max_sample_age_ms"])
        self.get_timeout_ms = int(sample["get_timeout_ms"])
        self.lease_timeout_ms = int(sample["lease_timeout_ms"])
        self.finalize_drain_timeout_ms = int(
            sample["finalize_drain_timeout_ms"]
        )
        self.finalize_request_path = Path(sample["finalize_request_path"])
        self.finalize_complete_path = Path(sample["finalize_complete_path"])
        self.finalize_request_path.unlink(missing_ok=True)
        self.finalize_complete_path.unlink(missing_ok=True)
        self._finalized = False
        self.shutdown_drain_timeout_ms = int(
            sample["shutdown_drain_timeout_ms"]
        )
        self.sample_address = f"{sample_host}:{sample_port}"
        self.sample_channel = grpc.insecure_channel(self.sample_address)
        self.sample_stub = training_pb2_grpc.SamplePoolConsumerServiceStub(
            self.sample_channel
        )

        model = config["model_distributor"]
        model_host = str(model["host"])
        model_port = int(model["port"])
        self.model_address = f"{model_host}:{model_port}"
        self.model_channel = grpc.insecure_channel(self.model_address)
        self.model_stub = training_pb2_grpc.ModelDistributorServiceStub(
            self.model_channel
        )

        actor = config["aiserver_status"]
        actor_host = str(actor["host"])
        actor_port = int(actor["port"])
        configured_initial_ack_timeout = actor.get(
            "initial_model_ack_timeout_sec"
        )
        self.initial_model_ack_timeout = (
            None
            if configured_initial_ack_timeout is None
            else float(configured_initial_ack_timeout)
        )
        self.actor_address = f"{actor_host}:{actor_port}"
        self.actor_channel = grpc.insecure_channel(self.actor_address)
        self.actor_stub = training_pb2_grpc.AIServerTrainingStatusServiceStub(
            self.actor_channel
        )
        self.metric_event_stub = training_pb2_grpc.MetricEventServiceStub(
            self.actor_channel
        )

        dashboard = config["dashboard"]
        self.metrics_backend = self._create_metrics_backend(
            str(dashboard["backend"]), str(self.publisher.metrics_dir)
        )
        self.metric_event_store: RawMetricBatchStore | None = None
        self.metric_event_writer: LocalTrainUpdateMetricWriter | None = None
        self.metric_event_relay: AIServerMetricRelay | None = None
        self.metric_event_projector: LocalMetricProjector | None = None
        self._metric_event_disabled_reason = ""
        self._metric_event_write_failure_count = 0
        self._metric_event_write_failure_started_at = 0.0
        self._initialize_metric_events()

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

    def _initialize_metric_events(self) -> None:
        store: RawMetricBatchStore | None = None
        try:
            catalog = MetricSchemaCatalog.load(
                default_metric_schema_directory()
            )
            store = RawMetricBatchStore(
                self.publisher.metrics_dir / "metric-events.sqlite3",
                self.contract,
                catalog,
            )
            writer = LocalTrainUpdateMetricWriter(
                store,
                self.learner_service,
                initial_train_update_sequence=self.train_updates,
            )
            relay = AIServerMetricRelay(
                store=store,
                contract=self.contract,
                consumer=self.learner_service,
                status_stub=self.actor_stub,
                event_stub=self.metric_event_stub,
                logger=self.logger,
            )
            projector = LocalMetricProjector(store)
        except (OSError, MetricEventContractError, RuntimeError) as error:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass
            self._metric_event_disabled_reason = str(error)
            self.logger.error(
                "immutable metric-event persistence unavailable; PPO training "
                "continues with this source marked unavailable: %s",
                error,
            )
            return
        self.metric_event_store = store
        self.metric_event_writer = writer
        self.metric_event_relay = relay
        self.metric_event_projector = projector

    def _start_metric_events(self) -> None:
        relay = getattr(self, "metric_event_relay", None)
        if relay is None:
            return
        try:
            relay.start()
        except Exception as error:
            self._metric_event_disabled_reason = str(error)
            self.logger.error("metric-event relay start failed: %s", error)

    def _stop_metric_events(self) -> None:
        writer = getattr(self, "metric_event_writer", None)
        store = getattr(self, "metric_event_store", None)
        relay = getattr(self, "metric_event_relay", None)
        if writer is not None:
            try:
                writer.finalize()
            except Exception as error:
                self.logger.error(
                    "Learner metric-event finalization failed: %s", error
                )
                if store is not None:
                    try:
                        store.mark_incomplete(
                            self.learner_service,
                            "learner_source_finalization_failed",
                        )
                    except Exception:
                        pass
        if relay is not None:
            relay.close()
        if store is not None:
            try:
                store.close()
            except Exception as error:
                self.logger.error("metric-event store close failed: %s", error)

    def _metric_event_snapshot(self) -> dict:
        store = getattr(self, "metric_event_store", None)
        if store is None:
            return {
                "enabled": False,
                "incomplete": True,
                "reason": getattr(
                    self, "_metric_event_disabled_reason", "uninitialized"
                ),
            }
        try:
            snapshot = store.snapshot()
            relay = getattr(self, "metric_event_relay", None)
            if relay is not None:
                snapshot["aiserver_relay"] = relay.snapshot()
            return snapshot
        except Exception as error:
            return {
                "enabled": False,
                "incomplete": True,
                "reason": str(error),
            }

    def _metric_event_view_snapshot(self) -> dict:
        projector = getattr(self, "metric_event_projector", None)
        if projector is None:
            return {
                "status": "unavailable",
                "reason": getattr(
                    self, "_metric_event_disabled_reason", "uninitialized"
                ),
            }
        try:
            return projector.snapshot()
        except Exception as error:
            self.logger.error("metric-event projection failed: %s", error)
            return {"status": "incomplete", "reason": str(error)}

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
    def _sample_pool_authority(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            valid = (
                authority.component == "sample-pool"
                and bool(authority.instance_id)
                and int(authority.lifecycle_epoch) > 0
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError("sample pool authority is invalid")
        result = common_pb2.ServiceInstanceIdentity()
        try:
            result.CopyFrom(authority)
        except TypeError as error:
            raise RuntimeError(
                "sample pool authority has an invalid wire type"
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
        self._validate_model_status_available_range(status)
        try:
            return self._model_distributor_authority(status.distributor)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"model distributor authority could not be pinned: {error}"
            ) from error

    @staticmethod
    def _has_field(message, name: str) -> bool:
        try:
            return bool(message.HasField(name))
        except (AttributeError, ValueError):
            return False

    @classmethod
    def _validate_model_manifest_available_range(
        cls,
        response,
        manifest_version: int,
        *,
        latest_selector: bool,
    ) -> tuple[int, int]:
        if not cls._has_field(
            response, "available_floor_model_step"
        ) or not cls._has_field(response, "latest_available_model_step"):
            raise RuntimeError(
                "model manifest response is missing the 0.13 available range"
            )
        floor = int(response.available_floor_model_step)
        latest = int(response.latest_available_model_step)
        if (
            floor < 0
            or floor > latest
            or not floor <= int(manifest_version) <= latest
            or (latest_selector and int(manifest_version) != latest)
        ):
            raise RuntimeError(
                "model manifest response has an incoherent available range"
            )
        return floor, latest

    @classmethod
    def _validate_model_status_available_range(cls, status) -> tuple[int, int] | None:
        has_latest_model = cls._has_field(status, "latest_model")
        has_floor = cls._has_field(
            status, "available_floor_model_step"
        )
        has_latest = cls._has_field(
            status, "latest_available_model_step"
        )
        if not has_latest_model:
            if has_floor or has_latest:
                raise RuntimeError(
                    "model status exposes a range without a latest model"
                )
            return None
        if not has_floor or not has_latest:
            raise RuntimeError(
                "model status is missing the 0.13 available range"
            )
        floor = int(status.available_floor_model_step)
        latest = int(status.latest_available_model_step)
        if (
            floor < 0
            or floor > latest
            or latest != int(status.latest_model.model_step)
            or not cls._has_field(status.latest_model, "model_step")
        ):
            raise RuntimeError(
                "model status has an incoherent available range"
            )
        return floor, latest

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
            model_step=0 if version is None else int(version),
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
        self._validate_model_manifest_available_range(
            response,
            int(identity.model_step),
            latest_selector=latest,
        )
        if (
            not manifest.ready
            or int(manifest.manifest_schema_version) != 2
            or not _same_message(manifest.contract, self.contract)
            or not _same_message(
                manifest.training_semantics, self.semantics
            )
            or not _same_message(
                manifest.training_config_digest,
                self.publisher.training_digest,
            )
            or identity.model_lineage_id != self.publisher.lineage_id
            or not self._has_field(identity, "model_step")
            or (version is not None and int(identity.model_step) != version)
            or not identity.artifact_digest.hex
            or not identity.manifest_digest.hex
            or int(identity.model_step) != int(manifest.train_updates)
        ):
            raise RuntimeError(
                "initial model preflight returned an incompatible manifest"
            )
        result = training_pb2.ModelArtifactManifest()
        result.CopyFrom(manifest)
        return result, response_authority

    @staticmethod
    def _initial_model_step(
        manifest: training_pb2.ModelArtifactManifest | None,
    ) -> int | None:
        return (
            None
            if manifest is None
            else int(manifest.identity.model_step)
        )

    def _assert_initial_latest_not_newer(
        self,
        target_step: int,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        latest, _ = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            latest=True,
        )
        latest_step = self._initial_model_step(latest)
        if latest_step is not None and latest_step > target_step:
            raise RuntimeError(
                "model distributor already contains a newer publication: "
                f"latest={latest_step}, target={target_step}"
            )
        return pinned_authority

    def _initial_model_requires_registration(
        self,
        document: dict,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[bool, common_pb2.ServiceInstanceIdentity]:
        expected = manifest_message(self._manifest_for_wire(document))
        target_step = int(expected.identity.model_step)
        latest, latest_authority = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            latest=True,
        )
        latest_step = self._initial_model_step(latest)
        if latest_step is not None and latest_step > target_step:
            raise RuntimeError(
                "model distributor already contains a newer publication: "
                f"latest={latest_step}, target={target_step}"
            )
        if latest_step == target_step:
            if not _same_message(latest, expected):
                raise RuntimeError(
                    "initial publication target slot conflicts with the "
                    "registered manifest"
                )
            return False, latest_authority

        target, target_authority = self._lookup_initial_model_manifest(
            pinned_authority=pinned_authority,
            version=target_step,
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
        try:
            self._validate_model_manifest_available_range(
                response,
                int(response.manifest.identity.model_step),
                latest_selector=False,
            )
        except RuntimeError as error:
            raise _UpdateCommitOutcomeUncertain(
                "exact model lookup returned an incoherent available range"
            ) from error
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

    def _wait_initial_model_loaded(self, document: dict) -> bool:
        expected = manifest_message(self._manifest_for_wire(document)).identity
        deadline = (
            None
            if self.initial_model_ack_timeout is None
            else time.monotonic() + self.initial_model_ack_timeout
        )
        self.logger.info(
            "Waiting for AIServer exact bootstrap ACK: model_step=%d "
            "artifact=%s timeout=%s",
            int(expected.model_step),
            expected.artifact_digest.hex,
            (
                "unbounded"
                if self.initial_model_ack_timeout is None
                else f"{self.initial_model_ack_timeout:g}s"
            ),
        )
        last = ""
        while (
            not _stop_requested.is_set()
            and (deadline is None or time.monotonic() < deadline)
        ):
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
                    self.logger.info(
                        "AIServer exact bootstrap ACK received: "
                        "model_step=%d artifact=%s",
                        int(expected.model_step),
                        expected.artifact_digest.hex,
                    )
                    return True
                last = (
                    f"status={training_pb2.ModelLoadStatus.Name(status.latest_ack_status)} "
                    f"ack={model_identity_document(status.latest_ack_model)}"
                )
            except grpc.RpcError as error:
                last = error.details() or str(error)
            time.sleep(0.2)
        if _stop_requested.is_set():
            self.logger.info(
                "Stop requested after bootstrap registration; skipping "
                "AIServer ACK readiness"
            )
            return False
        raise RuntimeError(
            "AIServer did not ACK exact bootstrap model identity before the "
            f"configured timeout: {last}"
        )

    def _initialize_models(self) -> bool:
        version = self.trainer.model_step
        self.initial_model_step = 0
        update_id = (
            "inherited-bootstrap-v0"
            if self._startup_mode == "inherited-weights"
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
        if not self._wait_initial_model_loaded(document):
            self.logger.info(
                "Learner bootstrap published and registered; stopping before "
                "AIServer activation: model_step=%d artifact=%s",
                version,
                document["identity"]["artifact_digest"],
            )
            return False
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
            "Learner training ready: model_step=%d artifact=%s startup=%s",
            version,
            document["identity"]["artifact_digest"],
            self._startup_mode,
        )
        return True

    def _sample_pool_status(self):
        return self.sample_stub.GetStatus(
            training_pb2.SamplePoolStatusReq(), timeout=2.0
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

    def _ready_sample_pool_authority(
        self, status
    ) -> common_pb2.ServiceInstanceIdentity:
        if not _same_message(status.contract, self.contract):
            raise RuntimeError(
                "sample pool status has another contract identity"
            )
        authority = self._sample_pool_authority(status.sample_pool)
        if not status.ready or not status.pool_ready:
            raise _SamplePoolUnavailable(
                "sample pool is not ready for the exact contract"
            )
        return authority

    def _sample_status_for_authority(
        self,
        expected: common_pb2.ServiceInstanceIdentity,
        operation: str,
    ):
        status = self._sample_pool_status()
        actual = self._ready_sample_pool_authority(status)
        if not self._same_authority(actual, expected):
            raise RuntimeError(
                f"sample pool authority changed during {operation}"
            )
        return status

    def _assert_lease_authority(
        self, expected: common_pb2.ServiceInstanceIdentity
    ) -> None:
        self._sample_status_for_authority(expected, "the lease")

    def _resolvable_model_identity(
        self, step: int
    ) -> training_pb2.ModelIdentity | None:
        document = self.model_manifests.get(int(step))
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
            or not self._has_field(manifest.identity, "model_step")
            or int(manifest.identity.model_step) != int(step)
            or int(manifest.identity.model_step) != int(manifest.train_updates)
        ):
            return None
        return manifest.identity

    def _effective_max_policy_lag(self) -> int:
        current_step = int(self.trainer.model_step)
        if self._resolvable_model_identity(current_step) is None:
            raise RuntimeError(
                "current model manifest is not locally resolvable"
            )
        configured_max = int(self.trainer.max_policy_lag)
        effective_max = 0
        for lag in range(1, configured_max + 1):
            step = current_step - lag
            if step < 0 or self._resolvable_model_identity(step) is None:
                break
            effective_max = lag
        return effective_max

    def _assert_sample_pool_ready(
        self,
    ) -> common_pb2.ServiceInstanceIdentity:
        return self._ready_sample_pool_authority(self._sample_pool_status())

    def _wait_for_sample_pool(
        self,
    ) -> common_pb2.ServiceInstanceIdentity | None:
        attempt = 0
        while not _stop_requested.is_set():
            try:
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
                "sample pool contract changed while GetBatch outcome "
                "was unknown"
            )
        actual = self._sample_pool_authority(status.sample_pool)
        if not self._same_authority(actual, expected):
            raise RuntimeError(
                "sample pool authority changed while GetBatch outcome "
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
                "sample pool returned contradictory lease status while "
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
        expected = self._sample_pool_authority(expected)
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
                    reference_model_step=self.trainer.model_step,
                    max_model_step_lag=effective_max_policy_lag,
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
            response_authority = self._sample_pool_authority(
                response.sample_pool
            )
            if ready_authority is None:
                ready_authority = self._ready_sample_pool_authority(
                    self._sample_pool_status()
                )
            else:
                ready_authority = self._sample_pool_authority(
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
                    "sample pool returned an incoherent busy delivery"
                )
            return response
        if response.result != training_pb2.GET_BATCH_RESULT_LEASED:
            return response
        if ready_authority is None:
            ready_authority = self._ready_sample_pool_authority(
                self._sample_pool_status()
            )
        else:
            ready_authority = self._sample_pool_authority(
                ready_authority
            )
        response_authority = self._sample_pool_authority(
            response.sample_pool
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
                "sample pool returned an incoherent leased delivery"
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
        expected = self._sample_pool_authority(ready_authority)
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
        behavior_steps: set[int] = set()
        oldest_created_at = 0
        newest_created_at = 0
        now_ms = int(time.time() * 1000)
        for batch in response.batches:
            behavior = batch.behavior_policy
            step = int(behavior.model_step)
            lag = self.trainer.model_step - step
            full_identity = self._resolvable_model_identity(step)
            if (
                not batch.batch_id
                or not batch.trajectory_id
                or not _same_message(batch.contract, self.contract)
                or not _same_message(batch.training_semantics, self.semantics)
                or behavior.model_lineage_id != self.publisher.lineage_id
                or not self._has_field(behavior, "model_step")
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
                or int(full_identity.model_step) != step
            ):
                raise ValueError("behavior policy cannot resolve to a model artifact")
            behavior_steps.add(step)
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
        if len(behavior_steps) != 1:
            raise ValueError(
                "one training delivery must contain exactly one behavior model step"
            )
        minimum_step = min(behavior_steps)
        maximum_step = max(behavior_steps)
        if (
            not self._has_field(response, "minimum_behavior_model_step")
            or not self._has_field(response, "maximum_behavior_model_step")
            or response.minimum_behavior_model_step != minimum_step
            or response.maximum_behavior_model_step != maximum_step
            or response.oldest_sample_created_at_unix_ms
            != oldest_created_at
            or response.newest_sample_created_at_unix_ms
            != newest_created_at
        ):
            raise ValueError("delivery summary does not match fragment identities")
        models = [
            dict(self.model_manifests[step]["identity"])
            for step in sorted(behavior_steps)
        ]
        return {
            "model_lineage_id": self.publisher.lineage_id,
            "minimum_model_step": minimum_step,
            "maximum_model_step": maximum_step,
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
                sample["behavior_model_step"] = int(
                    batch.behavior_policy.model_step
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
        base_step = int(self.trainer.model_step)
        target_step = base_step + 1
        rollback_path = (
            self.publisher.checkpoint_dir / f".{update_id}.rollback.pt"
        )
        if rollback_path.exists():
            raise RuntimeError(
                f"update rollback checkpoint already exists: {rollback_path}"
            )
        publication_path = self.publisher.publication_path(target_step)
        private_checkpoint_path = self.publisher.checkpoint_path(
            target_step
        )
        candidate_paths = (publication_path, private_checkpoint_path)
        temporary_paths = tuple(
            path.with_name(f".{path.name}.{os.getpid()}.tmp")
            for path in (
                self.publisher.state_path,
                self.publisher.receipt_path(update_id),
            )
        )
        state_exists = self.publisher.state_path.is_file()
        context = {
            "base_step": base_step,
            "target_step": target_step,
            "publication_path": publication_path,
            "private_checkpoint_path": private_checkpoint_path,
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
            "manifest_present": target_step in self.model_manifests,
            "manifest_document": copy.deepcopy(
                self.model_manifests.get(target_step)
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
        if int(self.trainer.model_step) != int(context["base_step"]):
            raise RuntimeError("pre-update model step was not restored")

        for path, existed in context["tracked_paths"].items():
            if existed or not path.exists():
                continue
            if path == context["publication_path"]:
                self.publisher.remove_publication_for_rollback(
                    int(context["target_step"])
                )
            elif path.is_dir() or path.is_symlink():
                raise RuntimeError(
                    f"refusing to remove unexpected rollback path: {path}"
                )
            else:
                path.unlink()
        if context["state_exists"]:
            atomic_write_json(
                self.publisher.state_path,
                context["state_document"],
            )
        else:
            self.publisher.state_path.unlink(missing_ok=True)

        target_step = int(context["target_step"])
        if context["manifest_present"]:
            self.model_manifests[target_step] = context[
                "manifest_document"
            ]
        else:
            self.model_manifests.pop(target_step, None)
        self._discard_update_rollback(context)

    def _train_delivery(self, response, *, allow_partial: bool = False) -> None:
        behavior_identity = self._validate_delivery(
            response, allow_partial=allow_partial
        )
        delivery_authority = self._sample_pool_authority(
            response.sample_pool
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
            "sample_pool_authority": self._authority_document(
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
        raw_metric_sum_counts: dict[str, dict[str, float | int]] = {}
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
            raw_metric_reader = getattr(
                self.trainer, "raw_metric_sum_counts", None
            )
            if callable(raw_metric_reader):
                raw_metric_sum_counts = raw_metric_reader()
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
                    "target_model_step": self.trainer.model_step,
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
            self.model_manifests[self.trainer.model_step] = manifest
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
            self._record_train_update_fact_best_effort(
                update_id=update_id,
                delivery_id=response.delivery_id,
                manifest=manifest,
                behavior_identity=behavior_identity,
                train_updates=target_updates,
                trained_samples=target_samples,
                actual_batch_size=len(training_samples),
                raw_metric_sum_counts=raw_metric_sum_counts,
            )
            minimum_protected = max(
                0,
                int(self.trainer.model_step)
                - int(self.trainer.max_policy_lag),
            )
            self.publisher.prune_publications(
                self.trainer.model_step,
                protected_steps=range(
                    minimum_protected,
                    int(self.trainer.model_step) + 1,
                ),
            )
            for version in tuple(self.model_manifests):
                if version < minimum_protected:
                    self.model_manifests.pop(version, None)
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
                            "restored_model_step": self.trainer.model_step,
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

    def _sample_pool_snapshot(self) -> dict:
        try:
            status = self._sample_pool_status()
            if not _same_message(status.contract, self.contract):
                raise RuntimeError(
                    "sample pool contract identity mismatch"
                )
            if status.sample_pool.component != "sample-pool":
                raise RuntimeError("sample pool component mismatch")
            return {
                "ready": bool(status.ready),
                "ingress_ready": bool(status.ingress_ready),
                "pool_ready": bool(status.pool_ready),
                "instance_id": status.sample_pool.instance_id,
                "lifecycle_epoch": int(status.sample_pool.lifecycle_epoch),
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
                "finalized": bool(status.finalized),
                "finalization_id": status.finalization_id,
                "finalized_at_unix_ms": int(
                    status.finalized_at_unix_ms
                ),
                "finalized_samples": int(
                    status.finalized_sample_count
                ),
                "finalized_fragments": int(
                    status.finalized_fragment_count
                ),
                "ready_samples": int(status.ready_queue_samples),
                "ready_fragments": int(status.ready_queue_fragments),
                "leased_samples": int(status.leased_samples),
                "leased_fragments": int(status.leased_fragments),
                "resident_samples": int(status.resident_samples),
                "resident_fragments": int(status.resident_fragments),
                "resident_estimated_bytes": int(
                    status.resident_estimated_bytes
                ),
                "evicted_samples": int(status.evicted_sample_count),
                "evicted_fragments": int(status.evicted_fragment_count),
                "redelivery_count": int(status.redelivery_count),
                "nack_count": int(status.nack_count),
                "expired_lease_count": int(status.expired_lease_count),
                "last_error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError) as error:
            return self._component_error_snapshot(
                "sample-pool", str(error)
            )

    def _model_snapshot(self) -> dict:
        try:
            status = self.model_stub.GetModelDistributorStatus(
                training_pb2.ModelDistributorStatusReq(), timeout=1.5
            )
            if not _same_message(status.contract, self.contract):
                raise RuntimeError("model distributor contract identity mismatch")
            available_range = self._validate_model_status_available_range(
                status
            )
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
                "available_floor_model_step": (
                    None if available_range is None else available_range[0]
                ),
                "latest_available_model_step": (
                    None if available_range is None else available_range[1]
                ),
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

    def _rates(self, actor: dict, sample_pool: dict, timestamp: float) -> dict:
        counters = {
            "produced": float(actor.get("produced", 0)),
            "accepted": float(sample_pool.get("accepted", 0)),
            "acked": float(sample_pool.get("acked", 0)),
            "trained": float(sample_pool.get("trained", 0)),
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
        model_step = int(identity["model_step"])
        if (
            model_step != int(train_updates)
            or model_step != int(self.trainer.model_step)
            or (
                "model_step" in stats
                and int(stats["model_step"]) != model_step
            )
        ):
            raise RuntimeError(
                "metric model_step must equal committed train_updates"
            )
        committed = {
            "model_identity": identity,
            "model_step": model_step,
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

    def _record_train_update_fact_best_effort(
        self,
        *,
        update_id: str,
        delivery_id: str,
        manifest: dict | None,
        behavior_identity: dict,
        train_updates: int,
        trained_samples: int,
        actual_batch_size: int,
        raw_metric_sum_counts: dict[str, dict[str, float | int]],
    ) -> None:
        writer = getattr(self, "metric_event_writer", None)
        if writer is None or manifest is None:
            return
        try:
            statistics = []
            for field_id in sorted(raw_metric_sum_counts):
                statistic = raw_metric_sum_counts[field_id]
                raw_sum = float(statistic["sum"])
                count = int(statistic["count"])
                if count <= 0 or not math.isfinite(raw_sum):
                    raise MetricEventContractError(
                        f"PPO raw sum/count is invalid: {field_id}"
                    )
                statistics.append(
                    training_pb2.RawMetricSumCount(
                        field_id=field_id,
                        sum=raw_sum,
                        count=count,
                    )
                )
            if not statistics:
                raise MetricEventContractError(
                    "committed train update has no raw PPO statistics"
                )
            writer.append(
                training_pb2.TrainUpdateMetricFact(
                    train_update_id=update_id,
                    train_update_sequence=int(train_updates),
                    published_model=manifest_message(manifest).identity,
                    delivery_id=delivery_id,
                    training_semantics=self.semantics,
                    cumulative_trained_samples=int(trained_samples),
                    actual_batch_size=int(actual_batch_size),
                    minimum_behavior_model_step=int(
                        behavior_identity["minimum_model_step"]
                    ),
                    maximum_behavior_model_step=int(
                        behavior_identity["maximum_model_step"]
                    ),
                    ppo_statistics=statistics,
                    behavior_model_lineage_id=str(
                        behavior_identity["model_lineage_id"]
                    ),
                ),
                committed_at_unix_ms=int(time.time() * 1000),
            )
            failure_count = int(
                getattr(self, "_metric_event_write_failure_count", 0)
            )
            if failure_count:
                elapsed = max(
                    0.0,
                    time.monotonic()
                    - float(
                        getattr(
                            self,
                            "_metric_event_write_failure_started_at",
                            time.monotonic(),
                        )
                    ),
                )
                self.logger.info(
                    "TrainUpdate metric persistence recovered after %d "
                    "failed update(s) and %.1fs; any recorded sequence gap "
                    "remains visible in history",
                    failure_count,
                    elapsed,
                )
                self._metric_event_write_failure_count = 0
                self._metric_event_write_failure_started_at = 0.0
        except Exception as error:
            failure_count = int(
                getattr(self, "_metric_event_write_failure_count", 0)
            ) + 1
            self._metric_event_write_failure_count = failure_count
            if failure_count == 1:
                self._metric_event_write_failure_started_at = time.monotonic()
                self.logger.error(
                    "committed TrainUpdate metric fact was not persisted; PPO "
                    "transaction remains committed and repeated failures will "
                    "be suppressed until recovery: %s",
                    error,
                )
            store = getattr(self, "metric_event_store", None)
            if store is not None and failure_count == 1:
                try:
                    store.mark_incomplete(
                        self.learner_service,
                        "train_update_fact_persistence_failed",
                    )
                except Exception:
                    pass

    def _learner_metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            context = copy.deepcopy(self._metrics_context)
            committed = copy.deepcopy(self._committed_learner_metrics)
        return {
            "instance_id": self.learner_service.instance_id,
            "lifecycle_epoch": int(self.learner_service.lifecycle_epoch),
            **committed,
            "initial_model_step": int(self.initial_model_step),
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
            sample_pool = self._sample_pool_snapshot()
            model = self._model_snapshot()
            now = time.time()
            learner = self._learner_metrics_snapshot()
            with self._metrics_lock:
                context = copy.deepcopy(self._metrics_context)
                self.sequence += 1
                sequence = self.sequence
                rates = self._rates(actor, sample_pool, now)
            chain = training_chain_status(
                actor,
                sample_pool,
                learner,
                model,
                str(context.get("error", "")),
            )
            record = {
                "schema_version": 3,
                "mode": "training",
                "metrics_source_id": self.metrics_source_id,
                "sequence": sequence,
                "timestamp": now,
                "interval_ms": 1000,
                "contract": contract_document(self.contract),
                "training_semantics": semantics_document(self.semantics),
                "learner": learner,
                "actor": actor,
                "sample_pool": sample_pool,
                "model": model,
                "chain": chain,
                "rates": rates,
                "resources": {"learner": self._resource_snapshot()},
                "metric_events": self._metric_event_snapshot(),
                "metric_event_views": self._metric_event_view_snapshot(),
            }
            self.metrics_backend.write(record)
            with self._metrics_lock:
                self._last_actor_snapshot = actor
                self._last_sample_pool_snapshot = sample_pool
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
        if ready_authority is None:
            return None
        return self._sample_pool_authority(ready_authority)

    def _sample_pool_finalization_id(self) -> str:
        identity = self.learner_service.SerializeToString(
            deterministic=True
        )
        digest = hashlib.sha256(
            identity + b"\0sample-pool-shutdown"
        ).hexdigest()
        return f"learner-shutdown-{digest}"

    @staticmethod
    def _finalization_count_fields(message) -> tuple[int, ...]:
        return (
            int(message.settled_samples),
            int(message.settled_fragments),
            int(message.ready_samples),
            int(message.ready_fragments),
            int(message.leased_samples),
            int(message.leased_fragments),
            int(message.resident_samples),
            int(message.resident_fragments),
        )

    def _validate_finalized_sample_pool_status(
        self,
        status,
        expected_authority: common_pb2.ServiceInstanceIdentity,
        finalization_id: str,
        response,
    ) -> None:
        if not _same_message(status.contract, self.contract):
            raise RuntimeError(
                "sample pool contract changed after finalization"
            )
        actual = self._sample_pool_authority(status.sample_pool)
        if not self._same_authority(actual, expected_authority):
            raise RuntimeError(
                "sample pool authority changed after finalization"
            )
        if (
            not status.finalized
            or status.finalization_id != finalization_id
            or int(status.finalized_at_unix_ms) <= 0
            or int(status.finalized_at_unix_ms)
            != int(response.finalized_at_unix_ms)
            or int(status.finalized_sample_count)
            != int(response.settled_samples)
            or int(status.finalized_fragment_count)
            != int(response.settled_fragments)
        ):
            raise RuntimeError(
                "sample pool returned contradictory finalization identity"
            )
        counts = {
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
            "ready": int(status.ready_queue_samples),
            "ready_fragments": int(status.ready_queue_fragments),
            "leased": int(status.leased_samples),
            "leased_fragments": int(status.leased_fragments),
            "resident": int(status.resident_samples),
            "resident_fragments": int(status.resident_fragments),
            "resident_bytes": int(status.resident_estimated_bytes),
            "evicted": int(status.evicted_sample_count),
            "evicted_fragments": int(status.evicted_fragment_count),
        }
        if min(counts.values()) < 0:
            raise RuntimeError(
                "sample pool returned negative finalized accounting"
            )
        if (
            status.ready
            or status.ingress_ready
            or status.pool_ready
            or counts["ready"] != 0
            or counts["ready_fragments"] != 0
            or counts["leased"] != 0
            or counts["leased_fragments"] != 0
            or counts["resident"] != 0
            or counts["resident_fragments"] != 0
            or counts["resident_bytes"] != 0
        ):
            raise RuntimeError(
                "sample pool retained live data after finalization"
            )
        if counts["acked"] != (
            counts["trained"]
            + counts["stale"]
            + counts["invalid"]
            + counts["shutdown_untrained"]
        ):
            raise RuntimeError(
                "sample pool returned contradictory finalized disposition "
                "accounting"
            )
        if counts["accepted"] != counts["acked"] + counts["evicted"]:
            raise RuntimeError(
                "sample pool returned contradictory finalized sample "
                "accounting"
            )
        if counts["accepted_batches"] != (
            counts["acked_batches"] + counts["evicted_fragments"]
        ):
            raise RuntimeError(
                "sample pool returned contradictory finalized fragment "
                "accounting"
            )

    def _finalize_sample_pool(
        self,
        expected_authority: common_pb2.ServiceInstanceIdentity,
        deadline: float,
    ) -> None:
        expected = self._sample_pool_authority(expected_authority)
        finalization_id = self._sample_pool_finalization_id()
        request = training_pb2.FinalizeSamplePoolReq(
            consumer=self.learner_service,
            expected_sample_pool=expected,
            finalization_id=finalization_id,
        )
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            timeout = min(2.0, max(0.001, deadline - time.monotonic()))
            try:
                response = self.sample_stub.FinalizeSamplePool(
                    request, timeout=timeout
                )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown SamplePool finalization",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue

            actual = self._sample_pool_authority(response.sample_pool)
            if not self._same_authority(actual, expected):
                raise RuntimeError(
                    "sample pool authority changed during finalization"
                )
            if response.finalization_id != finalization_id:
                raise RuntimeError(
                    "sample pool returned another finalization identity"
                )
            response_counts = self._finalization_count_fields(response)
            if min(response_counts) < 0:
                raise RuntimeError(
                    "sample pool returned negative finalization counts"
                )

            if (
                response.result
                == training_pb2.SAMPLE_POOL_FINALIZE_RESULT_REJECTED_ACTIVE_LEASE
            ):
                if int(response.leased_samples) <= 0:
                    raise RuntimeError(
                        "sample pool rejected finalization without an active "
                        "lease"
                    )
                if not self._reconcile_get_batch_outcome(
                    expected,
                    "SamplePool finalization observed an active delivery",
                    deadline=deadline,
                    ignore_stop=True,
                ):
                    break
                continue

            if response.ret_code != 0 or response.result not in (
                training_pb2.SAMPLE_POOL_FINALIZE_RESULT_FINALIZED,
                training_pb2.SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED,
            ):
                raise RuntimeError(
                    "shutdown SamplePool finalization failed: "
                    f"{response.message}"
                )
            if (
                any(response_counts[2:])
                or int(response.finalized_at_unix_ms) <= 0
            ):
                raise RuntimeError(
                    "sample pool returned a non-empty finalization result"
                )
            try:
                status = self._sample_pool_status()
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "finalized SamplePool status",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue
            self._validate_finalized_sample_pool_status(
                status, expected, finalization_id, response
            )
            self._clear_sample_wait()
            self.logger.info(
                "SamplePool finalized: id=%s settled_samples=%d "
                "settled_fragments=%d",
                finalization_id,
                int(response.settled_samples),
                int(response.settled_fragments),
            )
            return
        raise RuntimeError(
            "shutdown SamplePool finalization did not settle before timeout"
        )

    def _shutdown_finalize_sample_pool(
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
            self._sample_pool_authority(expected_authority)
            if expected_authority is not None
            else None
        )
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                status = self._sample_pool_status()
                if not _same_message(status.contract, self.contract):
                    raise RuntimeError(
                        "sample pool contract changed during shutdown"
                    )
                authority = self._sample_pool_authority(
                    status.sample_pool
                )
                if pinned is None:
                    pinned = authority
                elif not self._same_authority(authority, pinned):
                    raise RuntimeError(
                        "sample pool authority changed during shutdown"
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
            if min(ready_samples, leased_samples) < 0:
                raise RuntimeError(
                    "sample pool returned contradictory shutdown counts"
                )
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
            self._finalize_sample_pool(pinned, deadline)
            return
        raise RuntimeError(
            "shutdown SamplePool lease reconciliation did not settle before "
            "timeout"
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
            ):
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
            authority = self._ready_sample_pool_authority(status)
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
            self._start_metric_events()
            self._start_metrics()
            ready_authority = None
            if self._initialize_models():
                ready_authority = self._wait_for_sample_pool()
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
                        expected_authority=self._sample_pool_authority(
                            response.sample_pool
                        ),
                    )
                    raise RuntimeError(str(error)) from error
                ready_authority = self._wait_for_sample_pool()
            with self._metrics_lock:
                self._metrics_context["disposition"] = "STOPPING"
            shutdown_authority = self._shutdown_sample_authority(
                ready_authority
            )
            self._shutdown_finalize_sample_pool(shutdown_authority)
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
            self._stop_metrics()
            self._stop_metric_events()
            try:
                self.metrics_backend.close()
            except Exception as error:
                self.logger.error("metrics backend close failed: %s", error)
            self.actor_channel.close()
            self.model_channel.close()
            self.sample_channel.close()


def main() -> int:
    arguments = parse_startup_arguments()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    config = load_effective_config(
        arguments.config_path,
        cli_overrides=arguments.cli_overrides,
    )
    runtime = TrainingRuntime(config)
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
