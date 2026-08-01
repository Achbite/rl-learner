"""Run the leased-fragment PPO training and model publication loop."""

from __future__ import annotations

import argparse
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

from proto import maze_pb2, maze_pb2_grpc
from src.log.logger import setup_logger
from src.metrics.metrics_backend import create_backend
from src.training.ppo_trainer import PPOTrainer


_stop_requested = threading.Event()


def _handle_signal(_signal, _frame):
    _stop_requested.set()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


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


def manifest_message(document: dict):
    message = maze_pb2.ModelArtifactManifest(
        schema_version=document["schema_version"],
        contract_version=document["contract_version"],
        model_version=document["model_version"],
        artifact_uri=document["artifact_uri"],
        model_file=document["model_file"],
        size_bytes=document["size_bytes"],
        sha256=document["sha256"],
        seed=document["seed"],
        ready=document["ready"],
        published_ts_ms=document["published_ts_ms"],
    )
    message.input_shape.extend(document["input_shape"])
    message.action_shape.extend(document["action_shape"])
    message.value_shape.extend(document["value_shape"])
    return message


class ModelPublisher:
    ARCHIVE_MODEL_FILE = "SaveModel.onnx"
    ARCHIVE_CHECKPOINT_FILE = "checkpoint.pt"
    ARCHIVE_MANIFEST_FILE = "manifest.json"

    def __init__(self, config: dict):
        model = config.get("model", {})
        self.seed = int(model.get("bootstrap_seed", 0))
        self.obs_dim = int(model.get("obs_dim", 13))
        self.action_dim = int(model.get("action_dim", 9))
        self.local_train_root = Path(
            os.environ.get(
                "MAZE_LOCAL_TRAIN_ROOT",
                model.get("local_train_dir", "models/local-train"),
            )
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
                "MAZE_ARCHIVE_INTERVAL_UPDATES",
                model.get("archive_interval_updates", 200),
            )
        )
        self.archive_on_graceful_shutdown = str(
            os.environ.get(
                "MAZE_ARCHIVE_ON_GRACEFUL_SHUTDOWN",
                model.get("archive_on_graceful_shutdown", True),
            )
        ).lower() not in ("0", "false", "no")
        self.serving_retention_versions = int(
            os.environ.get(
                "MAZE_SERVING_RETENTION_VERSIONS",
                model.get("serving_retention_versions", 2),
            )
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
            raise RuntimeError(
                "initial checkpoint must be outside local-train"
            )
        checkpoint = self._load_checkpoint(checkpoint_path)
        required = (
            "model_state_dict",
            "optimizer_state_dict",
            "model_version",
            "torch_rng_state",
            "numpy_rng_state",
        )
        if any(key not in checkpoint for key in required):
            raise RuntimeError("initial checkpoint is incomplete")
        if not trainer.load_checkpoint(str(checkpoint_path)):
            raise RuntimeError(
                f"cannot load initial checkpoint: {checkpoint_path}"
            )
        metadata = checkpoint.get("metadata", {})
        self.initial_checkpoint_identity = {
            "initial_checkpoint": str(checkpoint_path),
            "initial_checkpoint_sha256": sha256_file(checkpoint_path),
            "initial_model_version": int(checkpoint["model_version"]),
        }
        return {
            "train_updates": int(
                metadata.get("train_updates", checkpoint["model_version"])
            ),
            "trained_samples": int(metadata.get("trained_samples", 0)),
            **self.initial_checkpoint_identity,
        }

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

    def publish_runtime(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model_version: int | None,
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
        if manifest_path.exists():
            raise RuntimeError(
                f"runtime model version already exists: {manifest_path}"
            )
        if model_path.exists():
            model_path.unlink()
        if checkpoint_path.exists() and not checkpoint_precommitted:
            raise RuntimeError(
                f"runtime checkpoint already exists: {checkpoint_path}"
            )
        temporary_model = model_path.with_name(
            f".{model_path.name}.{os.getpid()}.tmp"
        )
        temporary_checkpoint = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        metadata = {
            "train_update_id": train_update_id,
            "behavior_model_version": behavior_model_version,
            "batch_ids": batch_ids,
            "stats": stats or {},
            "sample_count": sample_count,
            "train_updates": train_updates,
            "trained_samples": trained_samples,
            **self.initial_checkpoint_identity,
        }
        trainer.export_onnx(str(temporary_model))
        if checkpoint_precommitted:
            checkpoint = self._load_checkpoint(checkpoint_path)
            checkpoint_metadata = checkpoint.get("metadata", {})
            if (
                checkpoint.get("model_version") != version
                or checkpoint_metadata.get("train_update_id")
                != train_update_id
            ):
                raise RuntimeError(
                    "precommitted checkpoint identity does not match"
                )
        else:
            trainer.save_checkpoint(
                str(temporary_checkpoint), metadata=metadata
            )
        sync_paths = [temporary_model]
        if not checkpoint_precommitted:
            sync_paths.append(temporary_checkpoint)
        for path in sync_paths:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if not checkpoint_precommitted:
            os.replace(temporary_checkpoint, checkpoint_path)
        os.replace(temporary_model, model_path)

        manifest = {
            "schema_version": 1,
            "contract_version": "0.6.0",
            "model_version": version,
            "artifact_uri": model_path.as_uri(),
            "model_file": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "input_shape": [1, self.obs_dim],
            "action_shape": [1, self.action_dim],
            "value_shape": [1, 1],
            "seed": self.seed,
            "ready": True,
            "published_ts_ms": int(time.time() * 1000),
            "checkpoint_file": checkpoint_path.name,
            "train_update_id": train_update_id,
            "behavior_model_version": behavior_model_version,
            "batch_ids": batch_ids,
            "train_updates": train_updates,
            "trained_samples": trained_samples,
            **self.initial_checkpoint_identity,
        }
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            self.state_path,
            {
                "schema_version": 1,
                "latest_model_version": version,
                "latest_manifest": str(manifest_path),
                "latest_checkpoint": str(checkpoint_path),
                "train_updates": train_updates,
                "trained_samples": trained_samples,
                "updated_ts_ms": int(time.time() * 1000),
                **self.initial_checkpoint_identity,
            },
        )
        return manifest

    def commit_optimizer_checkpoint(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model_version: int,
        batch_ids: list[str],
        stats: dict,
        sample_count: int,
        train_updates: int,
        trained_samples: int,
    ) -> Path:
        if not self._prepared:
            raise RuntimeError("model publisher is not prepared")
        checkpoint_path = self.checkpoint_path(trainer.model_version)
        if checkpoint_path.exists():
            checkpoint = self._load_checkpoint(checkpoint_path)
            if (
                checkpoint.get("model_version") != trainer.model_version
                or checkpoint.get("metadata", {}).get("train_update_id")
                != train_update_id
            ):
                raise RuntimeError(
                    f"checkpoint identity conflicts: {checkpoint_path}"
                )
            return checkpoint_path
        temporary = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        trainer.save_checkpoint(
            str(temporary),
            metadata={
                "train_update_id": train_update_id,
                "behavior_model_version": behavior_model_version,
                "batch_ids": batch_ids,
                "stats": stats,
                "sample_count": sample_count,
                "train_updates": train_updates,
                "trained_samples": trained_samples,
                **self.initial_checkpoint_identity,
            },
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def complete_manifest(
        self,
        version: int,
        train_update_id: str | None = None,
    ) -> dict | None:
        manifest_path = self.manifest_path(version)
        model_path = self.model_path(version)
        checkpoint_path = self.checkpoint_path(version)
        if not (
            manifest_path.is_file()
            and model_path.is_file()
            and checkpoint_path.is_file()
        ):
            return None
        try:
            manifest = read_json(manifest_path)
            checkpoint = self._load_checkpoint(checkpoint_path)
            model_size = model_path.stat().st_size
            model_checksum = sha256_file(model_path)
        except (OSError, ValueError, KeyError, RuntimeError):
            return None
        metadata = checkpoint.get("metadata", {})
        if (
            manifest.get("schema_version") != 1
            or manifest.get("contract_version") != "0.6.0"
            or manifest.get("model_version") != version
            or manifest.get("model_file") != model_path.name
            or not manifest.get("ready")
            or manifest.get("size_bytes") != model_size
            or manifest.get("sha256") != model_checksum
            or checkpoint.get("model_version") != version
            or metadata.get("train_update_id")
            != manifest.get("train_update_id")
            or (
                train_update_id is not None
                and metadata.get("train_update_id") != train_update_id
            )
        ):
            return None
        return manifest

    def complete_manifests(self) -> list[dict]:
        result = []
        for path in sorted(self.published_dir.glob("manifest_v*.json")):
            try:
                version = int(path.stem.removeprefix("manifest_v"))
            except ValueError:
                continue
            manifest = self.complete_manifest(version)
            if manifest is not None:
                result.append(manifest)
        return result

    def latest_complete_checkpoint(self) -> Path | None:
        manifests = self.complete_manifests()
        if not manifests:
            return None
        return self.checkpoint_path(manifests[-1]["model_version"])

    def checkpoint_metadata(self, version: int) -> dict:
        checkpoint = self._load_checkpoint(self.checkpoint_path(version))
        return checkpoint.get("metadata", {})

    def should_archive(self, version: int) -> bool:
        return version == 0 or version % self.archive_interval_updates == 0

    def archive_version(self, version: int, reason: str) -> dict:
        manifest = self.complete_manifest(version)
        if manifest is None:
            raise RuntimeError(f"cannot archive incomplete model v{version}")
        target = self.archive_path(version)
        if target.exists():
            archive_manifest = self.archive_manifest_path(version)
            if not archive_manifest.is_file():
                raise RuntimeError(f"archive is incomplete: {target}")
            existing = read_json(archive_manifest)
            archived_model = target / str(existing.get("model_file", ""))
            archived_checkpoint = target / str(
                existing.get("checkpoint_file", "")
            )
            if (
                existing.get("model_version") != version
                or existing.get("sha256") != manifest["sha256"]
                or existing.get("model_file") != self.ARCHIVE_MODEL_FILE
                or existing.get("checkpoint_file")
                != self.ARCHIVE_CHECKPOINT_FILE
                or not archived_model.is_file()
                or not archived_checkpoint.is_file()
                or sha256_file(archived_model) != manifest["sha256"]
            ):
                raise RuntimeError(f"archive identity conflicts: {target}")
            return existing

        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            archived_model = temporary / self.ARCHIVE_MODEL_FILE
            archived_checkpoint = temporary / self.ARCHIVE_CHECKPOINT_FILE
            shutil.copyfile(self.model_path(version), archived_model)
            shutil.copyfile(
                self.checkpoint_path(version), archived_checkpoint
            )
            archive_manifest = {
                **manifest,
                "artifact_uri": (
                    target / self.ARCHIVE_MODEL_FILE
                ).as_uri(),
                "model_file": archived_model.name,
                "checkpoint_file": archived_checkpoint.name,
                "archive_reason": reason,
                "archived_ts_ms": int(time.time() * 1000),
            }
            atomic_write_json(
                temporary / self.ARCHIVE_MANIFEST_FILE,
                archive_manifest,
            )
            os.replace(temporary, target)
            return archive_manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def prune_runtime(self, current_version: int) -> None:
        minimum = current_version - self.serving_retention_versions + 1
        for manifest in self.complete_manifests():
            version = int(manifest["model_version"])
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
        consumer_id: str,
        delivery_id: str,
        lease_timeout_ms: int,
    ):
        self.stub = stub
        self.request = maze_pb2.RenewLeaseReq(
            consumer_instance_id=consumer_id,
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

    def _run(self):
        last_success = time.monotonic()
        last_error = ""
        while not self._stop.wait(self.interval):
            try:
                response = self.stub.RenewLease(self.request, timeout=2.0)
                if response.result != maze_pb2.DELIVERY_RESULT_APPLIED:
                    self.error = response.message
                    return
                last_success = time.monotonic()
                last_error = ""
            except grpc.RpcError as exc:
                last_error = exc.details() or str(exc)
                if time.monotonic() - last_success >= self.failure_deadline:
                    self.error = last_error
                    return

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)


class TrainingRuntime:
    def __init__(
        self,
        config: dict,
        initial_checkpoint: str = "",
    ):
        self.config = config
        self.learner_id = os.environ.get("MAZE_LEARNER_ID", "learner-0")
        self.consumer_id = (
            f"{self.learner_id}-{os.getpid()}-{int(time.time() * 1000)}"
        )
        self.logger = setup_logger("TrainingRuntime")
        self.trainer = PPOTrainer(config)
        self.publisher = ModelPublisher(config)
        self.sequence = 0
        self.train_updates = 0
        self.trained_samples = 0
        self.last_stats: dict = {}
        self._acked_update_ids: set[str] = set()
        self._accounted_update_ids: set[str] = set()
        self._recorded_update_ids: set[str] = set()
        self._behavior_checksums: dict[int, str] = {}
        self._last_archive_version: int | None = None
        self._metrics_lock = threading.Lock()
        self._metrics_stop = threading.Event()
        self._metrics_thread: threading.Thread | None = None
        self._metrics_context = {
            "behavior_model_version": -1,
            "actual_batch_size": 0,
            "disposition": "STARTING",
            "train_update_id": "",
            "error": "",
        }
        self._rate_snapshot: dict = {}
        self._last_actor_snapshot: dict = {}
        self._last_distributor_snapshot: dict = {}
        self._last_model_snapshot: dict = {}
        self._last_resource_time = time.monotonic()
        self._last_process_cpu = time.process_time()

        dashboard = config.get("dashboard", {})
        model = config.get("model", {})
        configured_checkpoint = str(
            model.get("initial_checkpoint", "") or ""
        )
        initial_checkpoint = str(
            initial_checkpoint
            or os.environ.get("MAZE_INITIAL_CHECKPOINT", "")
            or configured_checkpoint
        )
        self.publisher.prepare()
        self._startup_mode = "fresh"
        if initial_checkpoint:
            self._startup_mode = "initial-checkpoint"
            restored = self.publisher.load_initial_checkpoint(
                self.trainer, initial_checkpoint
            )
            self.train_updates = int(restored["train_updates"])
            self.trained_samples = int(restored["trained_samples"])

        sample = config.get("sample_distributor", {})
        sample_host = os.environ.get(
            "MAZE_SAMPLE_DISTRIBUTOR_HOST",
            sample.get("host", "maze-aiserver"),
        )
        sample_port = int(
            os.environ.get(
                "MAZE_SAMPLE_DISTRIBUTOR_PORT", sample.get("port", 9100)
            )
        )
        self.train_batch_size = int(sample.get("train_batch_size", 512))
        self.preference_timeout_ms = int(
            sample.get("current_version_wait_ms", 10000)
        )
        self.get_timeout_ms = int(sample.get("get_timeout_ms", 1000))
        self.lease_timeout_ms = int(
            sample.get("lease_timeout_ms", 30000)
        )
        self.shutdown_drain_timeout_ms = int(
            sample.get("shutdown_drain_timeout_ms", 20000)
        )
        self.sample_channel = grpc.insecure_channel(
            f"{sample_host}:{sample_port}"
        )
        self.sample_stub = maze_pb2_grpc.SampleDistributorServiceStub(
            self.sample_channel
        )

        model = config.get("model_distributor", {})
        model_host = os.environ.get(
            "MAZE_MODEL_DISTRIBUTOR_HOST", model.get("host", "127.0.0.1")
        )
        model_port = int(
            os.environ.get(
                "MAZE_MODEL_DISTRIBUTOR_PORT", model.get("port", 9200)
            )
        )
        self.model_channel = grpc.insecure_channel(
            f"{model_host}:{model_port}"
        )
        self.model_stub = maze_pb2_grpc.ModelDistributorServiceStub(
            self.model_channel
        )

        aiserver_host = os.environ.get("MAZE_AISERVER_HOST", "maze-aiserver")
        aiserver_port = int(os.environ.get("MAZE_AISERVER_PORT", 9002))
        self.aiserver_channel = grpc.insecure_channel(
            f"{aiserver_host}:{aiserver_port}"
        )
        self.aiserver_stub = maze_pb2_grpc.MazeServiceStub(
            self.aiserver_channel
        )

        self.metrics = create_backend(
            dashboard.get("backend", "jsonl"),
            str(self.publisher.metrics_dir),
        )
        self._restore_metric_counters()

    def _restore_metric_counters(self) -> None:
        metrics_dir_getter = getattr(self.metrics, "get_metrics_dir", None)
        if callable(metrics_dir_getter):
            for path in Path(metrics_dir_getter()).glob("*.jsonl"):
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    self.sequence = max(
                        self.sequence, int(record.get("sequence", 0))
                    )
                    update_id = str(
                        record.get("learner", {}).get(
                            "train_update_id",
                            record.get("train_update_id", ""),
                        )
                    )
                    if update_id:
                        self._recorded_update_ids.add(update_id)

        receipts = []
        for path in self.publisher.update_dir.glob("*.json"):
            try:
                receipt = read_json(path)
            except (OSError, ValueError):
                continue
            if (
                receipt.get("state") != "ACKED"
                or not receipt.get("train_update_id")
            ):
                continue
            receipts.append(receipt)

        receipts.sort(
            key=lambda item: (
                int(item.get("target_model_version", -1)),
                str(item.get("train_update_id", "")),
            )
        )
        for receipt in receipts:
            update_id = str(receipt["train_update_id"])
            if update_id in self._acked_update_ids:
                continue
            self._acked_update_ids.add(update_id)
            if update_id not in self._accounted_update_ids:
                self._accounted_update_ids.add(update_id)
                receipt_updates = receipt.get("train_updates")
                receipt_samples = receipt.get("trained_samples")
                if receipt_updates is None or receipt_samples is None:
                    self.train_updates += 1
                    self.trained_samples += int(
                        receipt.get("sample_count", 0)
                    )
                else:
                    self.train_updates = max(
                        self.train_updates, int(receipt_updates)
                    )
                    self.trained_samples = max(
                        self.trained_samples, int(receipt_samples)
                    )
            self.last_stats = receipt.get("stats", self.last_stats)

    def _register(
        self,
        manifest: dict,
        timeout: float = 30.0,
        interruptible: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""
        request = maze_pb2.RegisterModelReq(
            manifest=manifest_message(manifest)
        )
        while (
            time.monotonic() < deadline
            and (not interruptible or not _stop_requested.is_set())
        ):
            try:
                response = self.model_stub.RegisterModel(
                    request, timeout=2.0
                )
                if response.result in (
                    maze_pb2.MODEL_REGISTER_RESULT_REGISTERED,
                    maze_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
                ):
                    return
                last_error = response.message
                if (
                    response.result
                    == maze_pb2.MODEL_REGISTER_RESULT_REJECTED_CONFLICT
                ):
                    break
            except grpc.RpcError as exc:
                last_error = exc.details() or str(exc)
            time.sleep(0.2)
        raise RuntimeError(f"model registration failed: {last_error}")

    def _initialize_models(self) -> None:
        train_update_id = (
            "bootstrap-v0"
            if self._startup_mode == "fresh"
            else "initial-checkpoint"
        )
        manifest = self.publisher.publish_runtime(
            self.trainer,
            train_update_id=train_update_id,
            behavior_model_version=None,
            batch_ids=[],
            train_updates=self.train_updates,
            trained_samples=self.trained_samples,
        )
        self.publisher.archive_version(
            self.trainer.model_version,
            "bootstrap"
            if self._startup_mode == "fresh"
            else "initial-checkpoint",
        )
        self._last_archive_version = self.trainer.model_version
        self.logger.info(
            "启动模型已提交: version=%d checksum=%s mode=%s",
            self.trainer.model_version,
            manifest["sha256"],
            self._startup_mode,
        )
        manifests = self.publisher.complete_manifests()
        if not manifests:
            raise RuntimeError("runtime has no complete model manifest")
        for manifest in manifests:
            self._register(manifest)
            self._behavior_checksums[int(manifest["model_version"])] = str(
                manifest["sha256"]
            )
        if self.trainer.model_version not in self._behavior_checksums:
            raise RuntimeError(
                "current trainer version has no complete runtime model"
            )

    def _model_status(self):
        return self.model_stub.GetModelDistributorStatus(
            maze_pb2.ModelDistributorStatusReq(),
            timeout=2.0,
        )

    def _wait_loaded(self, version: int, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            try:
                status = self._model_status()
                if (
                    status.latest_ack_model_version >= version
                    and status.latest_ack_status
                    == maze_pb2.MODEL_LOAD_STATUS_LOADED
                ):
                    return
                last = (
                    f"ack_version={status.latest_ack_model_version}, "
                    f"ack_status={status.latest_ack_status}"
                )
            except grpc.RpcError as exc:
                last = exc.details() or str(exc)
            time.sleep(0.2)
        raise RuntimeError(
            f"AIServer did not ACK model v{version}: {last}"
        )

    def _get_batch(
        self,
        version: int,
        timeout_ms: int,
        policy: int,
    ):
        request = maze_pb2.GetBatchReq(
            batch_size=self.train_batch_size,
            timeout_ms=timeout_ms,
            consumer_instance_id=self.consumer_id,
            lease_timeout_ms=self.lease_timeout_ms,
            behavior_model_version=version,
            selection_policy=policy,
        )
        return self.sample_stub.GetBatch(
            request, timeout=max(2.0, timeout_ms / 1000.0 + 2.0)
        )

    def _ack(
        self,
        delivery_id: str,
        disposition: int,
        train_update_id: str = "",
    ):
        response = self.sample_stub.AckBatch(
            maze_pb2.AckBatchReq(
                consumer_instance_id=self.consumer_id,
                delivery_id=delivery_id,
                disposition=disposition,
                train_update_id=train_update_id,
            ),
            timeout=3.0,
        )
        if response.result not in (
            maze_pb2.DELIVERY_RESULT_APPLIED,
            maze_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
        ):
            raise RuntimeError(
                f"sample Ack failed: {response.message}"
            )
        return response

    def _sample_status(self):
        return self.sample_stub.GetStatus(
            maze_pb2.DistributorStatusReq(), timeout=2.0
        )

    def _drain_stale(self) -> None:
        status = self._sample_status()
        minimum = self.trainer.model_version - 1
        for version in status.behavior_versions:
            if (
                version.behavior_model_version >= minimum
                or version.ready_samples <= 0
            ):
                continue
            remaining = version.ready_samples
            while remaining > 0:
                response = self._get_batch(
                    version.behavior_model_version,
                    self.get_timeout_ms,
                    maze_pb2.BATCH_SELECTION_POLICY_DRAIN_AVAILABLE,
                )
                if response.result != maze_pb2.GET_BATCH_RESULT_LEASED:
                    break
                self._ack(
                    response.delivery_id,
                    maze_pb2.ACK_DISPOSITION_STALE,
                )
                remaining -= response.actual_batch_size

    def _select_batch(self):
        current = self.trainer.model_version
        response = self._get_batch(
            current,
            self.preference_timeout_ms,
            maze_pb2.BATCH_SELECTION_POLICY_TARGET_ONLY,
        )
        if response.result == maze_pb2.GET_BATCH_RESULT_LEASED:
            return response
        if (
            response.result != maze_pb2.GET_BATCH_RESULT_TIMEOUT
            or current == 0
        ):
            return None
        response = self._get_batch(
            current - 1,
            self.get_timeout_ms,
            maze_pb2.BATCH_SELECTION_POLICY_TARGET_ONLY,
        )
        return (
            response
            if response.result == maze_pb2.GET_BATCH_RESULT_LEASED
            else None
        )

    def _validate_fragments(
        self, batches: Iterable
    ) -> tuple[int, list[str]]:
        batches = list(batches)
        if not batches:
            raise ValueError("delivery has no fragments")
        version = batches[0].behavior_model_version
        checksum = batches[0].behavior_model_checksum
        expected_checksum = self._behavior_checksums.get(version)
        if expected_checksum is None:
            manifest = self.publisher.complete_manifest(version)
            if manifest is not None:
                expected_checksum = str(manifest["sha256"])
                self._behavior_checksums[version] = expected_checksum
        if checksum != expected_checksum:
            raise ValueError(
                "fragment behavior checksum does not match the published model"
            )
        batch_ids = []
        seen_batch_ids = set()
        for batch in batches:
            if batch.protocol_version != 3:
                raise ValueError("fragment protocol version is invalid")
            if batch.behavior_model_version != version:
                raise ValueError("delivery mixes behavior model versions")
            if (
                batch.behavior_model_checksum != checksum
                or len(batch.behavior_model_checksum) != 64
            ):
                raise ValueError("fragment behavior checksum is inconsistent")
            if not batch.bootstrap_valid or not math.isfinite(
                batch.bootstrap_value
            ):
                raise ValueError("fragment bootstrap is invalid")
            if len(batch.samples) == 0:
                raise ValueError("fragment is empty")
            if not batch.batch_id or batch.batch_id in seen_batch_ids:
                raise ValueError("fragment batch identity is invalid")
            seen_batch_ids.add(batch.batch_id)
            expected = batch.first_action_frame_id
            for index, sample in enumerate(batch.samples):
                if sample.action_frame_id != expected:
                    raise ValueError("fragment action frames are not contiguous")
                if sample.terminated and sample.truncated:
                    raise ValueError(
                        "sample cannot be terminated and truncated"
                    )
                if (
                    len(sample.obs) != self.trainer.obs_dim
                    or not all(math.isfinite(value) for value in sample.obs)
                    or sample.action < 0
                    or sample.action >= self.trainer.action_dim
                    or not all(
                        math.isfinite(value)
                        for value in (
                            sample.reward,
                            sample.old_log_prob,
                            sample.old_vpred,
                        )
                    )
                ):
                    raise ValueError("sample tensor or scalar is invalid")
                if (
                    (sample.terminated or sample.truncated)
                    and index != len(batch.samples) - 1
                ):
                    raise ValueError(
                        "terminal transition must end the fragment"
                    )
                expected += 1
            if expected - 1 != batch.last_action_frame_id:
                raise ValueError("fragment frame range is inconsistent")
            terminal = (
                batch.samples[-1].terminated
                or batch.samples[-1].truncated
            )
            if terminal != batch.is_episode_end:
                raise ValueError("fragment episode-end metadata is inconsistent")
            if batch.samples[-1].terminated and batch.bootstrap_value != 0.0:
                raise ValueError("terminated fragment bootstrap must be zero")
            batch_ids.append(batch.batch_id)
        return version, batch_ids

    def _training_samples(self, batches: Iterable) -> list[dict]:
        result = []
        for batch in batches:
            trajectory = [
                {
                    "obs": list(sample.obs),
                    "action": sample.action,
                    "reward": sample.reward,
                    "old_log_prob": sample.old_log_prob,
                    "old_vpred": sample.old_vpred,
                    "terminated": sample.terminated,
                    "truncated": sample.truncated,
                }
                for sample in batch.samples
            ]
            result.extend(
                self.trainer.compute_gae(
                    trajectory,
                    bootstrap_value=batch.bootstrap_value,
                    bootstrap_valid=batch.bootstrap_valid,
                )
            )
        return result

    def _update_id(self, behavior_version: int, batch_ids: list[str]) -> str:
        identity = json.dumps(
            {
                "behavior_model_version": behavior_version,
                "batch_ids": batch_ids,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _write_receipt(self, path: Path, receipt: dict, state: str, **extra):
        receipt = {
            **receipt,
            **extra,
            "state": state,
            "updated_ts_ms": int(time.time() * 1000),
        }
        atomic_write_json(path, receipt)
        return receipt

    def _checkpoint_committed(self, receipt: dict) -> bool:
        return (
            self.publisher.complete_manifest(
                receipt["target_model_version"],
                receipt["train_update_id"],
            )
            is not None
        )

    def _recover_committed_receipt(
        self, path: Path, receipt: dict
    ) -> dict:
        version = receipt["target_model_version"]
        manifest = self.publisher.complete_manifest(
            version, receipt["train_update_id"]
        )
        if manifest is None:
            return receipt
        metadata = self.publisher.checkpoint_metadata(version)
        return self._write_receipt(
            path,
            receipt,
            "RUNTIME_COMMITTED",
            manifest=manifest,
            stats=metadata.get("stats", {}),
            sample_count=int(metadata.get("sample_count", 0)),
            train_updates=int(
                metadata.get("train_updates", self.train_updates)
            ),
            trained_samples=int(
                metadata.get("trained_samples", self.trained_samples)
            ),
        )

    def _ensure_trainer_version(self, version: int) -> None:
        if self.trainer.model_version == version:
            return
        checkpoint = self.publisher.checkpoint_path(version)
        if not self.trainer.load_checkpoint(str(checkpoint)):
            raise RuntimeError(
                f"cannot restore committed model v{version}"
            )

    def _ensure_archive_for_receipt(
        self, path: Path, receipt: dict
    ) -> dict:
        version = int(receipt["target_model_version"])
        if not self.publisher.should_archive(version):
            return receipt
        archive = self.publisher.archive_version(version, "interval")
        self._last_archive_version = version
        return self._write_receipt(
            path,
            receipt,
            receipt["state"],
            archive_committed=True,
            archive_manifest=str(
                self.publisher.archive_manifest_path(version)
            ),
            archive_checksum=archive["sha256"],
        )

    def _remember_acked_receipt(self, receipt: dict) -> None:
        update_id = str(receipt["train_update_id"])
        self._acked_update_ids.add(update_id)
        if update_id not in self._accounted_update_ids:
            self._accounted_update_ids.add(update_id)
            self.train_updates = int(
                receipt.get("train_updates", self.train_updates + 1)
            )
            self.trained_samples = int(
                receipt.get(
                    "trained_samples",
                    self.trained_samples
                    + int(receipt.get("sample_count", 0)),
                )
            )
        self.last_stats = receipt.get("stats", self.last_stats)
        if update_id in self._recorded_update_ids:
            return
        self._record_metrics(
            behavior_version=int(receipt["behavior_model_version"]),
            actual_batch_size=int(receipt.get("sample_count", 0)),
            stats=self.last_stats,
            disposition="TRAINED",
            train_update_id=update_id,
            error="",
        )
        self._recorded_update_ids.add(update_id)

    def _reconcile_receipts(self) -> None:
        receipts = []
        for path in self.publisher.update_dir.glob("*.json"):
            try:
                receipt = read_json(path)
            except (OSError, ValueError):
                continue
            if not receipt.get("train_update_id"):
                continue
            receipts.append((path, receipt))
        receipts.sort(
            key=lambda item: (
                int(item[1].get("target_model_version", -1)),
                str(item[1].get("train_update_id", "")),
            )
        )

        for path, receipt in receipts:
            state = receipt.get("state")
            if state == "ACKED":
                self._remember_acked_receipt(receipt)
                continue
            if (
                state in ("LEASED", "OPTIMIZER_COMMITTED")
                and self._checkpoint_committed(receipt)
            ):
                receipt = self._recover_committed_receipt(path, receipt)
                state = receipt["state"]
            if state in ("RUNTIME_COMMITTED", "MODEL_COMMITTED"):
                receipt = self._ensure_archive_for_receipt(path, receipt)
                manifest = receipt.get("manifest") or (
                    self.publisher.complete_manifest(
                        int(receipt["target_model_version"]),
                        str(receipt["train_update_id"]),
                    )
                )
                if manifest is None:
                    continue
                try:
                    self._register(manifest, timeout=5.0)
                except (grpc.RpcError, RuntimeError):
                    continue
                receipt = self._write_receipt(
                    path, receipt, "REGISTERED", manifest=manifest
                )
                state = receipt["state"]
            if state != "REGISTERED" or not receipt.get("delivery_id"):
                continue
            try:
                self._ack(
                    str(receipt["delivery_id"]),
                    maze_pb2.ACK_DISPOSITION_TRAINED,
                    str(receipt["train_update_id"]),
                )
            except (grpc.RpcError, RuntimeError):
                continue
            receipt = self._write_receipt(path, receipt, "ACKED")
            self._remember_acked_receipt(receipt)

    def _process_delivery(self, delivery) -> None:
        try:
            behavior_version, batch_ids = self._validate_fragments(
                delivery.batches
            )
            delivered_samples = sum(
                len(batch.samples) for batch in delivery.batches
            )
            if (
                delivered_samples != delivery.actual_batch_size
                or delivered_samples < self.train_batch_size
                or delivery.behavior_model_version != behavior_version
            ):
                raise ValueError("delivery batch accounting is inconsistent")
        except ValueError as exc:
            self.logger.error("样本 fragment 无效: %s", exc)
            self._ack(
                delivery.delivery_id,
                maze_pb2.ACK_DISPOSITION_INVALID,
            )
            self._record_metrics(
                behavior_version=-1,
                actual_batch_size=delivery.actual_batch_size,
                stats={},
                disposition="INVALID",
                train_update_id="",
                error=str(exc),
            )
            return

        current_version = self.trainer.model_version
        if behavior_version < current_version - 1:
            self._ack(
                delivery.delivery_id, maze_pb2.ACK_DISPOSITION_STALE
            )
            return
        if behavior_version > current_version:
            self._ack(
                delivery.delivery_id, maze_pb2.ACK_DISPOSITION_INVALID
            )
            raise RuntimeError(
                "Sample Pool returned a future behavior model version"
            )

        train_update_id = self._update_id(behavior_version, batch_ids)
        receipt_path = self.publisher.receipt_path(train_update_id)
        target_version = current_version + 1
        receipt = (
            read_json(receipt_path)
            if receipt_path.is_file()
            else {
                "schema_version": 1,
                "train_update_id": train_update_id,
                "behavior_model_version": behavior_version,
                "batch_ids": batch_ids,
                "target_model_version": target_version,
                "created_ts_ms": int(time.time() * 1000),
            }
        )
        receipt["delivery_id"] = delivery.delivery_id
        if not receipt_path.is_file():
            receipt = self._write_receipt(
                receipt_path, receipt, "LEASED"
            )
        if (
            receipt["state"] in ("LEASED", "OPTIMIZER_COMMITTED")
            and self._checkpoint_committed(receipt)
        ):
            receipt = self._recover_committed_receipt(
                receipt_path, receipt
            )

        renewer = LeaseRenewer(
            self.sample_stub,
            self.consumer_id,
            delivery.delivery_id,
            self.lease_timeout_ms,
        ).start()
        try:
            if receipt["state"] == "LEASED":
                self._wait_loaded(current_version)
                samples = self._training_samples(delivery.batches)
                if len(samples) < self.train_batch_size:
                    raise RuntimeError(
                        "TARGET_ONLY delivery is smaller than train batch"
                    )
                stats = self.trainer.train_on_batch(samples)
                if self.trainer.model_version != target_version:
                    raise RuntimeError("PPO model version did not advance once")
                if not all(
                    math.isfinite(float(stats[field]))
                    for field in (
                        "policy_loss",
                        "value_loss",
                        "total_loss",
                        "entropy",
                        "approx_kl",
                        "clip_fraction",
                        "gradient_norm",
                    )
                ):
                    raise RuntimeError("PPO update produced non-finite metrics")
                committed_train_updates = self.train_updates + 1
                committed_trained_samples = (
                    self.trained_samples + len(samples)
                )
                self.publisher.commit_optimizer_checkpoint(
                    self.trainer,
                    train_update_id=train_update_id,
                    behavior_model_version=behavior_version,
                    batch_ids=batch_ids,
                    stats=stats,
                    sample_count=len(samples),
                    train_updates=committed_train_updates,
                    trained_samples=committed_trained_samples,
                )
                receipt = self._write_receipt(
                    receipt_path,
                    receipt,
                    "OPTIMIZER_COMMITTED",
                    stats=stats,
                    sample_count=len(samples),
                    train_updates=committed_train_updates,
                    trained_samples=committed_trained_samples,
                    optimizer_committed=True,
                )
                manifest = self.publisher.publish_runtime(
                    self.trainer,
                    train_update_id=train_update_id,
                    behavior_model_version=behavior_version,
                    batch_ids=batch_ids,
                    stats=stats,
                    sample_count=len(samples),
                    train_updates=committed_train_updates,
                    trained_samples=committed_trained_samples,
                    checkpoint_precommitted=True,
                )
                self._behavior_checksums[
                    int(manifest["model_version"])
                ] = str(manifest["sha256"])
                receipt = self._write_receipt(
                    receipt_path,
                    receipt,
                    "RUNTIME_COMMITTED",
                    manifest=manifest,
                    stats=stats,
                    sample_count=len(samples),
                    train_updates=committed_train_updates,
                    trained_samples=committed_trained_samples,
                    runtime_committed=True,
                )
                receipt = self._ensure_archive_for_receipt(
                    receipt_path, receipt
                )
            else:
                self._ensure_trainer_version(
                    receipt["target_model_version"]
                )
                stats = receipt.get("stats", {})

            if receipt["state"] in (
                "RUNTIME_COMMITTED",
                "MODEL_COMMITTED",
            ):
                receipt = self._ensure_archive_for_receipt(
                    receipt_path, receipt
                )
                manifest = receipt.get("manifest") or read_json(
                    self.publisher.manifest_path(
                        receipt["target_model_version"]
                    )
                )
                self._register(manifest, interruptible=False)
                receipt = self._write_receipt(
                    receipt_path, receipt, "REGISTERED"
                )

            if renewer.error:
                raise RuntimeError(
                    f"sample lease renewal failed: {renewer.error}"
                )
            if receipt["state"] in ("REGISTERED", "ACKED"):
                self._ack(
                    delivery.delivery_id,
                    maze_pb2.ACK_DISPOSITION_TRAINED,
                    train_update_id,
                )
                receipt = self._write_receipt(
                    receipt_path, receipt, "ACKED"
                )

            if receipt["state"] != "ACKED":
                raise RuntimeError(
                    "training receipt did not reach ACKED"
                )
            self._remember_acked_receipt(receipt)
            self.publisher.prune_runtime(self.trainer.model_version)
            sample_count = int(receipt.get("sample_count", 0))
            self.logger.info(
                "Train Update 完成: behavior=v%d model=v%d samples=%d update=%s",
                behavior_version,
                self.trainer.model_version,
                sample_count,
                train_update_id[:12],
            )
        finally:
            renewer.close()

    def _record_metrics(
        self,
        *,
        behavior_version: int,
        actual_batch_size: int,
        stats: dict,
        disposition: str,
        train_update_id: str,
        error: str,
    ) -> None:
        with self._metrics_lock:
            if stats:
                self.last_stats = stats
            self._metrics_context = {
                "behavior_model_version": behavior_version,
                "actual_batch_size": actual_batch_size,
                "disposition": disposition,
                "train_update_id": train_update_id,
                "error": error,
            }
            self.sequence += 1
            timestamp = time.time()
            monotonic_now = time.monotonic()
            try:
                distributor_status = self._sample_status()
                distributor = {
                    "service_name": "LocalSampleService",
                    "ready": distributor_status.ready,
                    "ingress_ready": distributor_status.ingress_ready,
                    "pool_ready": distributor_status.pool_ready,
                    "backend_type": maze_pb2.SampleBackendType.Name(
                        distributor_status.backend_type
                    ),
                    "max_concurrent_consumers": (
                        distributor_status.max_concurrent_consumers
                    ),
                    "active_consumer_count": (
                        distributor_status.active_consumer_count
                    ),
                    "consumer_busy_count": (
                        distributor_status.consumer_busy_count
                    ),
                    "instance_id": (
                        distributor_status.distributor_instance_id
                    ),
                    "push_attempts": (
                        distributor_status.push_attempt_count
                    ),
                    "accepted": (
                        distributor_status.accepted_unique_samples
                    ),
                    "duplicates": (
                        distributor_status.duplicate_sample_attempts
                    ),
                    "rejected": (
                        distributor_status.rejected_sample_attempts
                    ),
                    "acked": distributor_status.acked_unique_samples,
                    "ready_samples": (
                        distributor_status.ready_queue_samples
                    ),
                    "leased_samples": distributor_status.leased_samples,
                    "resident_samples": distributor_status.resident_samples,
                    "trained": distributor_status.trained_sample_count,
                    "stale": distributor_status.stale_sample_count,
                    "invalid": distributor_status.invalid_sample_count,
                    "shutdown_untrained": (
                        distributor_status.shutdown_untrained_sample_count
                    ),
                    "redelivered": distributor_status.redelivery_count,
                    "lease_renewals": distributor_status.lease_renew_count,
                    "pressure": maze_pb2.PressureState.Name(
                        distributor_status.pressure_state
                    ),
                }
                self._last_distributor_snapshot = distributor
            except grpc.RpcError as exc:
                distributor = self._component_error_snapshot(
                    self._last_distributor_snapshot,
                    exc.details() or str(exc),
                )
            try:
                actor_status = self.aiserver_stub.GetAIServerStatus(
                    maze_pb2.AIServerStatusReq(),
                    timeout=0.75,
                )
                episodes = actor_status.episode_metrics
                actor = {
                    "ready": actor_status.ready,
                    "instance_id": actor_status.producer_instance_id,
                    "state": maze_pb2.AIServerState.Name(
                        actor_status.state
                    ),
                    "workload_mode": maze_pb2.WorkloadMode.Name(
                        actor_status.workload_mode
                    ),
                    "produced": actor_status.produced_unique_samples,
                    "accepted": actor_status.accepted_unique_samples,
                    "outbound_pending": (
                        actor_status.outbound_queue_samples
                    ),
                    "final_drop": (
                        actor_status.final_drop_unique_samples
                    ),
                    "model_version": actor_status.loaded_model_version,
                    "model_checksum": actor_status.loaded_model_checksum,
                    "staged_model_version": (
                        actor_status.staged_model_version
                    ),
                    "model_switches": actor_status.model_switch_count,
                    "quarantined_samples": (
                        actor_status.quarantined_sample_count
                    ),
                    "update_rpc_mean_ms": (
                        actor_status.update_rpc_latency_sum_ms
                        / actor_status.update_rpc_count
                        if actor_status.update_rpc_count
                        else 0.0
                    ),
                    "update_rpc_max_ms": (
                        actor_status.update_rpc_latency_max_ms
                    ),
                    "inference_mean_ms": (
                        actor_status.inference_latency_sum_ms
                        / actor_status.inference_count
                        if actor_status.inference_count
                        else 0.0
                    ),
                    "inference_max_ms": (
                        actor_status.inference_latency_max_ms
                    ),
                    "push_rpc_mean_ms": (
                        actor_status.push_rpc_latency_sum_ms
                        / actor_status.push_rpc_count
                        if actor_status.push_rpc_count
                        else 0.0
                    ),
                    "episodes": {
                        "window_size": episodes.configured_window_size,
                        "completed": episodes.completed_episode_count,
                        "agents": episodes.completed_agent_count,
                        "mean_agent_return": episodes.mean_agent_return,
                        "min_agent_return": episodes.min_agent_return,
                        "max_agent_return": episodes.max_agent_return,
                        "agent_success_count": (
                            episodes.agent_success_count
                        ),
                        "agent_success_rate": episodes.agent_success_rate,
                        "any_success_count": (
                            episodes.environment_any_success_count
                        ),
                        "any_success_rate": (
                            episodes.environment_any_success_rate
                        ),
                        "all_success_count": (
                            episodes.environment_all_success_count
                        ),
                        "all_success_rate": (
                            episodes.environment_all_success_rate
                        ),
                        "excluded": episodes.excluded_episode_count,
                        "termination_reasons": {
                            maze_pb2.TerminationReason.Name(item.reason): (
                                item.count
                            )
                            for item in episodes.termination_counts
                        },
                        "reward_components": dict(
                            episodes.reward_component_mean
                        ),
                    },
                }
                self._last_actor_snapshot = actor
            except grpc.RpcError as exc:
                actor = self._component_error_snapshot(
                    self._last_actor_snapshot,
                    exc.details() or str(exc),
                    {"episodes": {}},
                )
            try:
                model_status = self._model_status()
                model = {
                    "ready": model_status.ready,
                    "latest_version": model_status.latest_model_version,
                    "latest_checksum": model_status.latest_model_checksum,
                    "latest_ack_version": (
                        model_status.latest_ack_model_version
                    ),
                    "latest_ack_status": maze_pb2.ModelLoadStatus.Name(
                        model_status.latest_ack_status
                    ),
                    "serving_retention_versions": (
                        self.publisher.serving_retention_versions
                    ),
                    "last_archive_version": self._last_archive_version,
                    "next_archive_version": (
                        (
                            self.trainer.model_version
                            // self.publisher.archive_interval_updates
                        )
                        + 1
                    )
                    * self.publisher.archive_interval_updates,
                }
                self._last_model_snapshot = model
            except grpc.RpcError as exc:
                model = self._component_error_snapshot(
                    self._last_model_snapshot,
                    exc.details() or str(exc),
                    {
                        "last_archive_version": (
                            self._last_archive_version
                        )
                    },
                )

            interval = self._rate_interval(
                monotonic_now, actor, distributor
            )
            resources = self._process_resources(monotonic_now)
            learner = {
                "train_updates": self.train_updates,
                "trained_samples": self.trained_samples,
                "actual_batch_size": actual_batch_size,
                "behavior_model_version": behavior_version,
                "model_version": self.trainer.model_version,
                "model_lag": max(
                    0,
                    self.trainer.model_version
                    - int(actor.get("model_version", 0)),
                ),
                "train_update_id": train_update_id,
                "ack_disposition": disposition,
                "startup_mode": self._startup_mode,
                "resources": resources,
                **self.last_stats,
                "error": error,
            }
            record = {
                "schema_version": 3,
                "mode": "training",
                "sequence": self.sequence,
                "train_step": self.train_updates,
                "timestamp": timestamp,
                "timestamp_ms": int(timestamp * 1000),
                "interval_ms": int(interval["seconds"] * 1000),
                "rates": interval["rates"],
                "actor": actor,
                "distributor": distributor,
                "learner": learner,
                "model": model,
                "chain": {
                    "ready": bool(
                        actor.get("ready")
                        and distributor.get("ready")
                        and model.get("ready")
                        and not error
                    ),
                    "error": error,
                },
                "model_version": self.trainer.model_version,
                "behavior_model_version": behavior_version,
                "trained_samples": self.trained_samples,
                "mean_episode_reward": actor.get("episodes", {}).get(
                    "mean_agent_return", 0.0
                ),
                "pass_rate": actor.get("episodes", {}).get(
                    "agent_success_rate", 0.0
                ),
                **self.last_stats,
            }
            self.metrics.write(record)

    @staticmethod
    def _component_error_snapshot(
        previous: dict, error: str, fallback: dict | None = None
    ) -> dict:
        snapshot = dict(fallback or {})
        snapshot.update(previous)
        snapshot["ready"] = False
        snapshot["error"] = error
        return snapshot

    def _rate_interval(
        self, monotonic_now: float, actor: dict, distributor: dict
    ) -> dict:
        current = {
            "timestamp": monotonic_now,
            "actor_instance": actor.get("instance_id", ""),
            "distributor_instance": distributor.get("instance_id", ""),
            "produced": int(actor.get("produced", 0)),
            "accepted": int(distributor.get("accepted", 0)),
            "acked": int(distributor.get("acked", 0)),
            "trained": int(self.trained_samples),
        }
        previous = self._rate_snapshot
        self._rate_snapshot = current
        if not previous:
            return {
                "seconds": 1.0,
                "rates": {
                    "produced_sps": 0.0,
                    "accepted_sps": 0.0,
                    "acked_sps": 0.0,
                    "trained_sps": 0.0,
                },
            }
        seconds = max(
            0.001, monotonic_now - float(previous["timestamp"])
        )
        actor_reset = (
            current["actor_instance"] != previous["actor_instance"]
        )
        distributor_reset = (
            current["distributor_instance"]
            != previous["distributor_instance"]
        )

        def rate(name: str, reset: bool) -> float:
            if reset or current[name] < previous[name]:
                return 0.0
            return (current[name] - previous[name]) / seconds

        return {
            "seconds": seconds,
            "rates": {
                "produced_sps": rate("produced", actor_reset),
                "accepted_sps": rate("accepted", distributor_reset),
                "acked_sps": rate("acked", distributor_reset),
                "trained_sps": rate("trained", False),
            },
        }

    def _process_resources(self, monotonic_now: float) -> dict:
        process_cpu = time.process_time()
        elapsed = max(0.001, monotonic_now - self._last_resource_time)
        cpu_percent = max(
            0.0,
            (process_cpu - self._last_process_cpu) / elapsed * 100.0,
        )
        self._last_resource_time = monotonic_now
        self._last_process_cpu = process_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_multiplier = 1 if sys.platform == "darwin" else 1024
        return {
            "cpu_percent": cpu_percent,
            "rss_bytes": int(usage.ru_maxrss * rss_multiplier),
            "gpu_memory_bytes": None,
        }

    def _metrics_loop(self) -> None:
        while not self._metrics_stop.wait(1.0):
            context = dict(self._metrics_context)
            try:
                self._record_metrics(
                    behavior_version=int(
                        context["behavior_model_version"]
                    ),
                    actual_batch_size=int(context["actual_batch_size"]),
                    stats=self.last_stats,
                    disposition=str(context["disposition"]),
                    train_update_id=str(context["train_update_id"]),
                    error=str(context["error"]),
                )
            except Exception as exc:
                self.logger.debug("周期指标采集失败: %s", exc)

    def _drain_shutdown(self) -> None:
        deadline = (
            time.monotonic() + self.shutdown_drain_timeout_ms / 1000.0
        )
        while time.monotonic() < deadline:
            try:
                status = self._sample_status()
            except grpc.RpcError:
                return
            versions = [
                item
                for item in status.behavior_versions
                if item.ready_samples > 0
            ]
            if not versions:
                return
            for version in versions:
                response = self._get_batch(
                    version.behavior_model_version,
                    self.get_timeout_ms,
                    maze_pb2.BATCH_SELECTION_POLICY_DRAIN_AVAILABLE,
                )
                if response.result == maze_pb2.GET_BATCH_RESULT_LEASED:
                    self._ack(
                        response.delivery_id,
                        maze_pb2.ACK_DISPOSITION_SHUTDOWN_UNTRAINED,
                    )
                elif response.result != maze_pb2.GET_BATCH_RESULT_TIMEOUT:
                    return

    def run(self) -> None:
        sample_distributor_connected = False
        sample_distributor_waiting = False
        initialized = False
        try:
            self._initialize_models()
            initialized = True
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop,
                name="learner-metrics",
                daemon=True,
            )
            self._metrics_thread.start()
            self._record_metrics(
                behavior_version=-1,
                actual_batch_size=0,
                stats={},
                disposition="READY",
                train_update_id="",
                error="",
            )
            self.logger.info(
                "Learner training 就绪: model=v%d batch=%d startup=%s",
                self.trainer.model_version,
                self.train_batch_size,
                self._startup_mode,
            )
            while not _stop_requested.is_set():
                try:
                    self._drain_stale()
                    self._reconcile_receipts()
                    delivery = self._select_batch()
                except grpc.RpcError as exc:
                    if not sample_distributor_waiting:
                        log_wait = (
                            self.logger.warning
                            if sample_distributor_connected
                            else self.logger.info
                        )
                        log_wait(
                            "等待 LocalSampleService: %s",
                            exc.details() or str(exc),
                        )
                        sample_distributor_waiting = True
                    _stop_requested.wait(0.5)
                    continue
                if sample_distributor_waiting:
                    self.logger.info(
                        "LocalSampleService 连接%s",
                        "已恢复"
                        if sample_distributor_connected
                        else "已建立",
                    )
                sample_distributor_connected = True
                sample_distributor_waiting = False
                if delivery is None:
                    continue
                self._process_delivery(delivery)
        finally:
            self._metrics_stop.set()
            if self._metrics_thread is not None:
                self._metrics_thread.join(timeout=5.0)
            if _stop_requested.is_set():
                self._drain_shutdown()
                if (
                    initialized
                    and self.publisher.archive_on_graceful_shutdown
                ):
                    self.publisher.archive_version(
                        self.trainer.model_version, "graceful-shutdown"
                    )
                    self._last_archive_version = (
                        self.trainer.model_version
                    )
                    self._record_metrics(
                        behavior_version=self.trainer.model_version,
                        actual_batch_size=0,
                        stats=self.last_stats,
                        disposition="STOPPED",
                        train_update_id="",
                        error="",
                    )
            self.metrics.close()
            self.sample_channel.close()
            self.model_channel.close()
            self.aiserver_channel.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPO training")
    parser.add_argument("--config", default="configs/learner_config.yaml")
    parser.add_argument(
        "--initial-checkpoint",
        default=os.environ.get("MAZE_INITIAL_CHECKPOINT", ""),
    )
    args = parser.parse_args()
    _stop_requested.clear()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    runtime = TrainingRuntime(
        load_config(args.config),
        initial_checkpoint=args.initial_checkpoint,
    )
    runtime.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
