"""R-PIN processed-transition PPO runtime for the current rl-contracts."""

from __future__ import annotations

import copy
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
from typing import Callable

import grpc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto.common import identity_pb2 as common_pb2
from proto.metrics import training_pb2 as training_metrics_pb2
from proto.training import training_pb2
from proto.training import training_pb2_grpc
from proto.metrics import registry_pb2 as metric_registry_pb2
from proto.training import model_identity_pb2
from src.contracts.identity import (
    has_field,
    same_service_instance,
    model_identity_document,
    read_manifest_file,
    service_identity,
    validate_config,
    write_manifest_file,
)
from src.config.command_line import parse_startup_arguments
from src.config.effective_config import effective_config_log, load_effective_config
from src.log.logger import setup_logger
from src.metrics.metric_events import (
    LocalTrainUpdateMetricWriter,
    MetricEventContractError,
    RawMetricBatchStore,
    create_learner_metric_event_server,
)
from src.preview.collector import PreviewCollector
from src.training.ppo_trainer import PPOTrainer
from src.training.sample_pool_consumer import SamplePoolConsumer, training_samples
from src.training.update_commit import (
    MAX_ATTEMPTS, RETRY_DELAY_SEC, UpdateCommitNotApplied, UpdateCommitOutcomeUncertain,
)


_stop_requested = threading.Event()


def _handle_signal(_signal, _frame) -> None:
    _stop_requested.set()


def _same_message(left, right) -> bool:
    return left.SerializeToString(deterministic=True) == right.SerializeToString(
        deterministic=True
    )


def load_config(path: str) -> dict:
    return load_effective_config(path)


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


class ModelPublisher:
    MODEL_FILE = "SaveModel.onnx"
    MANIFEST_FILE = "manifest.pb"
    MAX_MODEL_STEP = (1 << 64) - 1
    PROVENANCE_KEYS = (
        "initial_model_path",
    )
    def __init__(self, config: dict):
        validate_config(config)
        model = config["model"]
        self.config = config
        self.obs_dim = int(model["observation_dimension"])
        self.action_dim = int(model["action_count"])
        self.lineage_id = str(config["identity"]["model_lineage_id"])
        self.local_train_root = Path(str(model["local_train_dir"])).resolve()
        self.runtime_dir = self.local_train_root / "runtime"
        self.checkpoint_dir = self.runtime_dir / "checkpoints"
        self.update_dir = self.runtime_dir / "receipts"
        self.state_path = self.runtime_dir / "state.json"
        self.publication_dir = self.local_train_root
        self.metrics_dir = self.local_train_root / "metrics"
        self.archive_interval_updates = int(model["archive_interval_updates"])
        if self.archive_interval_updates <= 0:
            raise ValueError("archive_interval_updates must be positive")
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

    def receipt_path(self, train_update_id: str) -> Path:
        return self.update_dir / f"{train_update_id}.json"

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        return PPOTrainer.read_checkpoint(str(path), "cpu")

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
        if not model_path.is_file():
            raise RuntimeError("initial model must be an explicit regular file")
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
        }
        return dict(self.initial_model_provenance)

    def _checkpoint_metadata(
        self,
        *,
        train_update_id: str,
        behavior_model: dict | None,
        item_ids: list[str],
        stats: dict,
        sample_count: int,
        train_updates: int,
        trained_samples: int,
    ) -> dict:
        return {
            "train_update_id": train_update_id,
            "behavior_model": behavior_model or {},
            "item_ids": list(item_ids),
            "stats": dict(stats),
            "sample_count": int(sample_count),
            "train_updates": int(train_updates),
            "trained_samples": int(trained_samples),
            "model_lineage_id": self.lineage_id,
            **self.initial_model_provenance,
        }

    def commit_optimizer_checkpoint(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model: dict | None,
        item_ids: list[str],
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
            item_ids=item_ids,
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
        item_ids: list[str],
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
            minimum_behavior_step = behavior_model.get(
                "minimum_model_step", behavior_model.get("model_step")
            )
            maximum_behavior_step = behavior_model.get(
                "maximum_model_step", behavior_model.get("model_step")
            )
            if (
                minimum_behavior_step is None
                or maximum_behavior_step is None
                or int(minimum_behavior_step) > int(maximum_behavior_step)
            ):
                raise RuntimeError("behavior model step range is invalid")
        target = self.publication_path(step)
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"model publication already exists: {target}")
        temporary = self._temporary_publication_path(step)
        temporary.mkdir(parents=False, exist_ok=False)
        temporary_model = temporary / self.MODEL_FILE
        temporary_manifest = temporary / self.MANIFEST_FILE
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
                item_ids=item_ids,
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
                    item_ids=item_ids,
                    stats=stats or {},
                    sample_count=sample_count,
                    train_updates=train_updates,
                    trained_samples=trained_samples,
                )
                checkpoint_created = not checkpoint_existed
            for path in (temporary_model, private_checkpoint):
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            manifest = training_pb2.ModelArtifactManifest(
                identity=model_identity_pb2.ModelIdentity(
                    model_lineage_id=self.lineage_id,
                    model_step=step,
                ),
                size_bytes=temporary_model.stat().st_size,
                trained_samples=int(trained_samples),
                published_at_unix_ms=int(time.time() * 1000),
            )
            identity = model_identity_document(manifest.identity)
            retention = self.retention_for_updates(int(train_updates))
            runtime_document = {
                "manifest": manifest,
                "identity": identity,
                "train_update_id": train_update_id,
                "behavior_model": behavior_model or {},
                "item_ids": list(item_ids),
                "retention": retention,
                **self.initial_model_provenance,
            }
            write_manifest_file(temporary_manifest, manifest)
            for path in (temporary_model, temporary_manifest):
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
        self, model_step: int, train_update_id: str | None = None
    ) -> dict | None:
        publication = self.publication_path(model_step)
        path = self.manifest_path(model_step)
        model_path = self.model_path(model_step)
        checkpoint_path = self.checkpoint_path(model_step)
        if (
            not self._is_canonical_publication_path(publication, model_step)
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
        }
        if set(entries) != required or any(
            entry.is_symlink() or not entry.is_file()
            for entry in entries.values()
        ):
            return None
        try:
            manifest = read_manifest_file(path)
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                return None
            checkpoint = self._load_checkpoint(checkpoint_path)
            model_size = model_path.stat().st_size
        except (OSError, ValueError, KeyError, RuntimeError, TypeError):
            return None
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            return None
        metadata_keys = {
            "train_update_id",
            "behavior_model",
            "item_ids",
            "stats",
            "sample_count",
            "train_updates",
            "trained_samples",
            "model_lineage_id",
        }
        if set(metadata) not in (
            metadata_keys,
            metadata_keys | set(self.PROVENANCE_KEYS),
        ):
            return None
        if (
            manifest.identity.model_lineage_id != self.lineage_id
            or not manifest.identity.HasField("model_step")
            or int(manifest.identity.model_step) != model_step
            or int(manifest.size_bytes) != model_size
            or int(manifest.published_at_unix_ms) <= 0
            or checkpoint.get("model_step") != model_step
            or not metadata.get("train_update_id")
            or not isinstance(metadata.get("behavior_model"), dict)
            or not isinstance(metadata.get("item_ids"), list)
            or not all(
                isinstance(item_id, str) and item_id
                for item_id in metadata.get("item_ids", [])
            )
            or not isinstance(metadata.get("stats"), dict)
            or int(metadata.get("sample_count", -1)) < 0
            or int(metadata.get("train_updates", -1)) != model_step
            or int(metadata.get("trained_samples", -1))
            != int(manifest.trained_samples)
            or metadata.get("model_lineage_id") != self.lineage_id
            or (
                train_update_id is not None
                and metadata.get("train_update_id") != train_update_id
            )
        ):
            return None
        return {
            "manifest": manifest,
            "identity": model_identity_document(manifest.identity),
            "train_update_id": metadata["train_update_id"],
            "behavior_model": metadata.get("behavior_model", {}),
            "item_ids": metadata.get("item_ids", []),
            "retention": self.retention_for_updates(model_step),
            **{
                key: metadata[key]
                for key in self.PROVENANCE_KEYS
                if key in metadata
            },
        }

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

    def should_mark_permanent(self, run_train_updates: int) -> bool:
        return (
            run_train_updates > 0
            and run_train_updates % self.archive_interval_updates == 0
        )

    def retention_for_updates(self, run_train_updates: int) -> dict:
        if self.should_mark_permanent(run_train_updates):
            return {"class": "permanent", "reason": "interval"}
        return {"class": "rolling", "reason": ""}

    def remove_publication_for_rollback(self, model_step: int) -> None:
        target = self.publication_path(model_step)
        if not target.exists():
            return
        manifest = self.complete_manifest(model_step)
        if (
            manifest is None
            or not self._is_canonical_publication_path(target, model_step)
        ):
            raise RuntimeError(
                f"refusing to rollback unverified publication: {target}"
            )
        quarantine = self.publication_dir / (
            f".rollback-delete-{self.step_name(model_step)}-"
            f"{os.getpid()}-{time.time_ns()}"
        )
        os.replace(target, quarantine)
        self._fsync_directory(self.publication_dir)
        shutil.rmtree(quarantine)
        self._fsync_directory(self.publication_dir)


class TrainingRuntime:
    MANAGED_METRIC_READY_PATH = Path(
        "/run/rl/learner-metric-ready.json"
    )
    METRIC_FINAL_ACK_TIMEOUT_SEC = 10.0

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
        self.trainer = PPOTrainer(config)
        self.publisher = ModelPublisher(config)

        learner_name = os.environ.get("RL_LEARNER_INSTANCE", "learner-0")
        instance = f"{learner_name}-{os.getpid()}-{int(time.time() * 1000)}"
        self.learner_service = service_identity("learner", instance, 1)
        self.metrics_source_id = str(
            os.environ.get("RL_METRICS_SOURCE_ID", "")
        )
        self.train_updates = 0
        self.trained_samples = 0
        self._run_start_train_updates = 0
        self._run_start_trained_samples = 0
        self.initial_model_step = 0
        self.last_stats: dict = {}
        self.model_manifests: dict[int, dict] = {}
        self._metrics_context = {
            "behavior_model": {},
            "actual_batch_size": None,
            "requested_train_batch_size": int(
                config["training"]["train_batch_size"]
            ),
            "pool_draw_slot_count": None,
            "unique_item_count": None,
            "duplicate_item_slot_count": None,
            "sample_evaluation_count": None,
            "optimizer_step_count": None,
            "disposition": "STARTING",
            "train_update_id": "",
            "error": "",
        }
        self._metrics_lock = threading.RLock()
        self._committed_learner_metrics = {
            "model_identity": {},
            "model_step": self.trainer.model_step,
            "train_updates": self.train_updates,
            "run_train_updates": self.train_updates,
            "run_trained_samples": self.trained_samples,
        }
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
        self.train_batch_size = int(config["training"]["train_batch_size"])
        self.finalize_request_path = Path(sample["finalize_request_path"])
        self.finalize_complete_path = Path(sample["finalize_complete_path"])
        self.finalize_request_path.unlink(missing_ok=True)
        self.finalize_complete_path.unlink(missing_ok=True)
        self._finalized = False
        self.sample_address = f"{sample_host}:{sample_port}"
        self.pool = SamplePoolConsumer(
            self.sample_address, sample, self.learner_service, self.train_batch_size,
            _stop_requested, self.logger, self._report_sample_wait,
        )

        model = config["model_distributor"]
        model_host = str(model["host"])
        model_port = int(model["port"])
        self.model_address = f"{model_host}:{model_port}"
        self.model_channel = grpc.insecure_channel(self.model_address)
        self.model_stub = training_pb2_grpc.ModelDistributorServiceStub(
            self.model_channel
        )

        dashboard = config["dashboard"]
        self.metric_event_store: RawMetricBatchStore | None = None
        self.metric_event_writer: LocalTrainUpdateMetricWriter | None = None
        self.metric_event_server = None
        self.metric_event_server_started = False
        self.metric_event_server_port = int(
            config["metric_events"]["server_port"]
        )
        self.metric_event_server_enabled = bool(
            config["metric_events"]["server_enabled"]
        )
        self.metric_event_relay_enabled = bool(
            config["metric_events"]["aiserver_relay_enabled"]
        )
        self._initialize_metric_events()
        self.preview = None
        if dashboard["enabled"]:
            actor = config["aiserver_status"]
            self.preview = PreviewCollector(
                endpoints={"learner": f"127.0.0.1:{self.metric_event_server_port}",
                           "aiserver": f"{actor['host']}:{actor['port']}", "sample_pool": self.sample_address,
                           "distributor": self.model_address},
                learner_source=self.learner_service, source_id=self.metrics_source_id,
                directory=self.publisher.metrics_dir, backend=dashboard["backend"], logger=self.logger,
                collect_aiserver=self.metric_event_relay_enabled,
                binding_timeout=actor["initial_model_ack_timeout_sec"],
                mean_window_ms=dashboard.get("mean_window_ms", 60000),
                bucket_ms=dashboard.get("time_bucket_ms", 5000),
            )

    def _initialize_metric_events(self) -> None:
        store: RawMetricBatchStore | None = None
        try:
            store = RawMetricBatchStore(
                self.publisher.metrics_dir / "metric-events.sqlite3",
            )
            writer = LocalTrainUpdateMetricWriter(
                store,
                self.learner_service,
                initial_train_update_sequence=self.train_updates,
            )
            event_server = None
            if self.metric_event_server_enabled:
                event_server = create_learner_metric_event_server(
                    store=store,
                    source=self.learner_service,
                    port=self.metric_event_server_port,
                    writer=writer,
                    status_snapshot=self._learner_metrics_snapshot,
                )
        except (OSError, MetricEventContractError, RuntimeError) as error:
            if store is not None:
                try:
                    store.close()
                except Exception as close_error:
                    raise RuntimeError(
                        "metric-event initialization and store cleanup failed: "
                        f"{close_error}"
                    ) from error
            raise
        self.metric_event_store = store
        self.metric_event_writer = writer
        self.metric_event_server = event_server

    def _publish_metric_ready(self) -> None:
        if os.environ.get("RL_INFRA_MANAGED") != "true":
            return
        store = self.metric_event_store
        if store is None:
            raise RuntimeError("Learner metric store is unavailable")
        document = {
            "component": self.learner_service.component,
            "instance_id": self.learner_service.instance_id,
            "lifecycle_epoch": int(
                self.learner_service.lifecycle_epoch
            ),
            "container_port": self.metric_event_server_port,
        }
        destination = self.MANAGED_METRIC_READY_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(document, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)

    def _start_metric_events(self) -> None:
        server = getattr(self, "metric_event_server", None)
        if self.metric_event_server_enabled:
            if server is None:
                raise RuntimeError(
                    "Learner raw MetricEvent service is required but unavailable"
                )
            server.start()
            self.metric_event_server_started = True
            self._publish_metric_ready()

    def _stop_metric_events(self) -> None:
        writer = getattr(self, "metric_event_writer", None)
        store = getattr(self, "metric_event_store", None)
        server = getattr(self, "metric_event_server", None)
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
        if (
            store is not None
            and server is not None
            and self.metric_event_server_started
        ):
            try:
                final_acknowledged = store.wait_for_final_export_ack(
                    self.learner_service,
                    self.METRIC_FINAL_ACK_TIMEOUT_SEC,
                )
                if not final_acknowledged:
                    store.mark_incomplete(
                        self.learner_service,
                        "infra_final_ack_timeout",
                    )
                    self.logger.error(
                        "Learner raw metric final batch was not acknowledged "
                        "before %.1fs shutdown deadline",
                        self.METRIC_FINAL_ACK_TIMEOUT_SEC,
                    )
            except Exception as error:
                self.logger.error(
                    "Learner metric final ACK wait failed: %s", error
                )
        if self.preview is not None:
            self.preview.close()
        if server is not None:
            try:
                server.stop(2.0).wait(timeout=3.0)
            except Exception as error:
                self.logger.error(
                    "Learner raw metric server stop failed: %s", error
                )
        self.MANAGED_METRIC_READY_PATH.unlink(missing_ok=True)
        if store is not None:
            try:
                store.close()
            except Exception as error:
                self.logger.error("metric-event store close failed: %s", error)

    @staticmethod
    def _model_distributor_authority(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            valid = (
                bool(authority.component)
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
    def _authority_document(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> dict:
        return {
            "component": authority.component,
            "instance_id": authority.instance_id,
            "lifecycle_epoch": int(authority.lifecycle_epoch),
        }

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
        if not status.ready:
            raise RuntimeError(
                "model distributor authority could not be pinned: "
                "distributor is not ready"
            )
        self._validate_model_status_available_range(status)
        try:
            return self._model_distributor_authority(status.distributor)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                f"model distributor authority could not be pinned: {error}"
            ) from error

    @staticmethod
    def _validate_model_manifest_available_range(
        response,
        model_step: int,
        *,
        latest_selector: bool,
    ) -> tuple[int, int]:
        if not has_field(
            response, "available_floor_model_step"
        ) or not has_field(response, "latest_available_model_step"):
            raise RuntimeError(
                "model manifest response is missing the available range"
            )
        floor = int(response.available_floor_model_step)
        latest = int(response.latest_available_model_step)
        if (
            floor < 0
            or floor > latest
            or not floor <= int(model_step) <= latest
            or (latest_selector and int(model_step) != latest)
        ):
            raise RuntimeError(
                "model manifest response has an incoherent available range"
            )
        return floor, latest

    @staticmethod
    def _validate_model_status_available_range(status) -> tuple[int, int] | None:
        has_latest_model = has_field(status, "latest_model")
        has_floor = has_field(
            status, "available_floor_model_step"
        )
        has_latest = has_field(
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
                "model status is missing the available range"
            )
        floor = int(status.available_floor_model_step)
        latest = int(status.latest_available_model_step)
        if (
            floor < 0
            or floor > latest
            or latest != int(status.latest_model.model_step)
            or not has_field(status.latest_model, "model_step")
        ):
            raise RuntimeError(
                "model status has an incoherent available range"
            )
        return floor, latest

    def _lookup_initial_model_manifest(
        self,
        *,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
        model_step: int | None = None,
        latest: bool = False,
    ) -> tuple[
        training_pb2.ModelArtifactManifest | None,
        common_pb2.ServiceInstanceIdentity,
    ]:
        if latest == (model_step is not None):
            raise ValueError(
                "initial model lookup requires exactly one selector"
            )
        selector = model_identity_pb2.ModelIdentity(
            model_lineage_id=self.publisher.lineage_id,
            model_step=0 if model_step is None else int(model_step),
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
        if response.result == training_pb2.MODEL_LOOKUP_RESULT_NOT_FOUND:
            if manifest_present:
                raise RuntimeError(
                    "initial model preflight returned NOT_FOUND with a manifest"
                )
            if not same_service_instance(
                response_authority, pinned_authority
            ):
                raise RuntimeError(
                    "initial model absence is uncertain after distributor "
                    "authority changed"
                )
            return None, response_authority
        if response.result != training_pb2.MODEL_LOOKUP_RESULT_FOUND:
            raise RuntimeError(
                response.message or "initial model preflight was rejected"
            )
        if not manifest_present:
            raise RuntimeError(
                "initial model preflight returned FOUND without a manifest"
            )
        manifest = response.manifest
        identity = manifest.identity
        self._validate_model_manifest_available_range(
            response,
            int(identity.model_step),
            latest_selector=latest,
        )
        if (
            identity.model_lineage_id != self.publisher.lineage_id
            or not has_field(identity, "model_step")
            or (
                model_step is not None
                and int(identity.model_step) != model_step
            )
            or int(manifest.size_bytes) <= 0
            or int(manifest.published_at_unix_ms) <= 0
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
        expected = document["manifest"]
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
            model_step=target_step,
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
        expected = document["manifest"]
        response = self.model_stub.RegisterModel(
            training_pb2.RegisterModelReq(
                manifest=expected,
                local_artifact_path=str(
                    self.publisher.model_path(int(expected.identity.model_step))
                ),
            ),
            timeout=5.0,
        )
        positive_result = response.result in (
            training_pb2.MODEL_REGISTER_RESULT_REGISTERED,
            training_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED,
        )
        if not positive_result:
            raise RuntimeError(
                f"model registration rejected: {response.message}"
            )
        if not _same_message(response.manifest, expected):
            raise RuntimeError("model distributor returned a different manifest")
        try:
            return self._model_distributor_authority(response.distributor)
        except (AttributeError, RuntimeError) as error:
            raise UpdateCommitOutcomeUncertain(
                "registered model response has invalid distributor authority"
            ) from error

    def _exact_model_registered(
        self,
        document: dict,
        pinned_authority: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[
        bool | None, common_pb2.ServiceInstanceIdentity | None
    ]:
        expected = document["manifest"]
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
            raise UpdateCommitOutcomeUncertain(
                "exact model lookup has invalid distributor authority"
            ) from error
        if response.result == training_pb2.MODEL_LOOKUP_RESULT_NOT_FOUND:
            try:
                manifest_present = response.HasField("manifest")
            except (AttributeError, ValueError):
                manifest_present = getattr(response, "manifest", None) is not None
            if manifest_present:
                raise UpdateCommitOutcomeUncertain(
                    "exact model lookup returned NOT_FOUND with a manifest"
                )
            if not same_service_instance(
                response_authority, pinned_authority
            ):
                raise UpdateCommitOutcomeUncertain(
                    "model distributor authority changed before exact absence"
                )
            return False, response_authority
        if response.result != training_pb2.MODEL_LOOKUP_RESULT_FOUND:
            raise UpdateCommitOutcomeUncertain(
                response.message or "exact model lookup was rejected"
            )
        if not _same_message(response.manifest, expected):
            raise UpdateCommitOutcomeUncertain(
                "model distributor returned a conflicting exact identity"
            )
        try:
            self._validate_model_manifest_available_range(
                response,
                int(response.manifest.identity.model_step),
                latest_selector=False,
            )
        except RuntimeError as error:
            raise UpdateCommitOutcomeUncertain(
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
        for attempt in range(1, MAX_ATTEMPTS + 1):
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
            except UpdateCommitOutcomeUncertain as error:
                last_error = error
                break
            except Exception as error:
                last_error = error
                if attempt == MAX_ATTEMPTS:
                    break
                self.logger.warning(
                    "model register attempt %d failed; retrying exact manifest: %s",
                    attempt,
                    error,
                )
                time.sleep(RETRY_DELAY_SEC)

        try:
            registered, resolved_authority = self._exact_model_registered(
                document, pinned_authority
            )
        except UpdateCommitOutcomeUncertain as error:
            error.attempts = attempts
            raise
        if registered:
            if resolved_authority is None:
                raise RuntimeError(
                    "exact registration evidence has no distributor authority"
                )
            return attempts, resolved_authority
        if registered is False:
            raise UpdateCommitNotApplied(
                "model registration was not applied after "
                f"{attempts} attempts: {last_error}",
                attempts,
            ) from last_error
        raise UpdateCommitOutcomeUncertain(
            "model registration outcome is uncertain after "
            f"{attempts} attempts: {last_error}",
            attempts,
        ) from last_error

    def _initialize_models(self) -> None:
        model_step = self.trainer.model_step
        self.initial_model_step = 0
        update_id = (
            "inherited-bootstrap"
            if self._startup_mode == "inherited-weights"
            else "bootstrap"
        )
        pinned_authority = self._pin_model_distributor_authority()
        pinned_authority = self._assert_initial_latest_not_newer(
            model_step, pinned_authority
        )
        document = self.publisher.publish_runtime(
            self.trainer,
            train_update_id=update_id,
            behavior_model=None,
            item_ids=[],
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
        self.model_manifests[model_step] = document
        if self.preview is not None:
            self.preview.expect_bootstrap(document["manifest"].identity, pinned_authority)
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
            "Learner training ready: lineage=%s model_step=%d startup=%s",
            document["identity"]["model_lineage_id"],
            model_step,
            self._startup_mode,
        )

    def _report_sample_wait(self, disposition: str, message: str) -> None:
        wait_states = ("WAITING_FOR_SAMPLE_POOL", "GET_BATCH_OUTCOME_UNKNOWN")
        with self._metrics_lock:
            current = self._metrics_context["disposition"]
            if disposition:
                if current not in wait_states:
                    self._sample_wait_resume_disposition = current
                self._metrics_context.update(disposition=disposition, error=message)
            elif current in wait_states:
                self._metrics_context.update(disposition=self._sample_wait_resume_disposition, error="")

    def _validate_transition(
        self, transition: training_pb2.ProcessedTransition
    ) -> None:
        values = [
            *transition.observation,
            transition.behavior_log_probability,
            transition.behavior_value,
            transition.advantage,
            transition.value_target,
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("processed transition contains non-finite values")
        if (
            not transition.item_id
            or not has_field(transition, "behavior_model_step")
            or int(transition.behavior_model_step) > int(self.trainer.model_step)
            or int(transition.action) < 0
            or int(transition.action) >= self.publisher.action_dim
            or len(transition.observation) != self.publisher.obs_dim
            or int(transition.created_at_unix_ms) <= 0
        ):
            raise ValueError("processed transition identity or shape is invalid")
        action_mask = list(transition.action_mask)
        if self.trainer.action_mask_mode == "required":
            if (
                len(action_mask) != self.publisher.action_dim
                or not any(action_mask)
                or not action_mask[int(transition.action)]
            ):
                raise ValueError(
                    "processed transition action mask differs from the contract"
                )
        elif action_mask:
            raise ValueError(
                "processed transition action mask is disabled by the contract"
            )

    def _validate_training_item(self, item):
        """Check this sample against the consuming policy and its published models."""
        transition = item.transition
        self._validate_transition(transition)
        step = int(transition.behavior_model_step)
        identity = model_identity_document(item.behavior_model)
        if (
            not identity
            or identity["model_step"] != step
            or identity["model_lineage_id"] != self.publisher.lineage_id
        ):
            raise ValueError(
                "processed transition source model does not match the "
                f"training lineage/step: item={transition.item_id} source={identity}"
            )
        document = self.model_manifests.get(step)
        if document is None or not _same_message(
            item.behavior_model, document["manifest"].identity
        ):
            raise ValueError(
                "processed transition source model is not a published "
                f"training model: item={transition.item_id} source={identity}"
            )
        return identity

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

    def _train_delivery(self, response) -> None:
        behavior_identity = self.pool.delivery_provenance(response, self._validate_training_item)
        delivery_authority = response.sample_pool
        update_number = self.train_updates + 1
        update_id = f"train-update-{update_number:08d}"
        item_ids = [item.transition.item_id for item in response.items]
        pool_draw_slot_count = len(response.items)
        unique_item_count = len(set(item_ids))
        duplicate_item_slot_count = len(item_ids) - unique_item_count
        receipt = {
            "train_update_id": update_id,
            "delivery_id": response.delivery_id,
            "behavior_model": behavior_identity,
            "item_ids": item_ids,
            "state": "LEASED",
            "transition_count": len(response.items),
            "pool_draw_slot_count": pool_draw_slot_count,
            "unique_item_count": unique_item_count,
            "duplicate_item_slot_count": duplicate_item_slot_count,
            "sample_pool_authority": self._authority_document(
                delivery_authority
            ),
        }
        self._write_receipt(update_id, receipt)
        renewer = self.pool.renew_during_update(response)
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
        samples: list[dict] = []
        sample_evaluation_count: int | None = None
        optimizer_step_count: int | None = None
        target_updates = self.train_updates
        target_samples = self.trained_samples
        try:
            try:
                rollback = self._capture_update_rollback(update_id)
            except Exception as error:
                raise RuntimeError(
                    f"pre-update rollback capture failed: {error}"
                ) from error
            samples = training_samples(response.items)
            stats = self.trainer.train_on_batch(samples)
            sample_evaluation_count = int(stats["sample_evaluation_count"])
            optimizer_step_count = int(stats["optimizer_step_count"])
            if sample_evaluation_count <= 0 or optimizer_step_count <= 0:
                raise RuntimeError(
                    "PPO update returned non-positive execution facts"
                )
            raw_metric_reader = getattr(
                self.trainer, "raw_metric_sum_counts", None
            )
            if callable(raw_metric_reader):
                raw_metric_sum_counts = raw_metric_reader()
            training_succeeded = True
            if renewer.error:
                raise RuntimeError(f"sample lease lost: {renewer.error}")
            target_updates = self.train_updates + 1
            target_samples = self.trained_samples + len(samples)
            self.publisher.commit_optimizer_checkpoint(
                self.trainer,
                train_update_id=update_id,
                behavior_model=behavior_identity,
                item_ids=item_ids,
                stats=stats,
                sample_count=len(samples),
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
                item_ids=item_ids,
                stats=stats,
                sample_count=len(samples),
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
                receipt["ack_attempts"] = self.pool.acknowledge_trained(response, update_id)
                sample_acked = True
                receipt["state"] = "ACKED"
                self._write_receipt(update_id, receipt)
                self._commit_learner_metrics(
                    manifest,
                    behavior_model=behavior_identity,
                    actual_batch_size=len(samples),
                    disposition="TRAINED",
                    train_update_id=update_id,
                    train_updates=target_updates,
                    trained_samples=target_samples,
                    stats=stats,
                    requested_train_batch_size=self.train_batch_size,
                    pool_draw_slot_count=pool_draw_slot_count,
                    unique_item_count=unique_item_count,
                    duplicate_item_slot_count=duplicate_item_slot_count,
                    sample_evaluation_count=sample_evaluation_count,
                    optimizer_step_count=optimizer_step_count,
                )
                transaction_complete = True
            self._record_train_update_fact(
                update_id=update_id,
                delivery_id=response.delivery_id,
                manifest=manifest,
                behavior_identity=behavior_identity,
                train_updates=target_updates,
                trained_samples=target_samples,
                actual_batch_size=len(samples),
                raw_metric_sum_counts=raw_metric_sum_counts,
                requested_train_batch_size=self.train_batch_size,
                pool_draw_slot_count=pool_draw_slot_count,
                unique_item_count=unique_item_count,
                duplicate_item_slot_count=duplicate_item_slot_count,
                sample_evaluation_count=sample_evaluation_count,
                optimizer_step_count=optimizer_step_count,
            )
        except Exception as error:
            failed_state = str(receipt.get("state", "LEASED"))
            if training_succeeded and not model_registered:
                if register_started:
                    receipt.setdefault(
                        "register_attempts",
                        int(getattr(error, "attempts", 0)),
                    )
                if isinstance(error, UpdateCommitOutcomeUncertain):
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
                            actual_batch_size=len(samples),
                            disposition="REGISTER_PENDING",
                            train_update_id=update_id,
                            train_updates=target_updates,
                            trained_samples=target_samples,
                            stats=stats,
                            requested_train_batch_size=self.train_batch_size,
                            pool_draw_slot_count=pool_draw_slot_count,
                            unique_item_count=unique_item_count,
                            duplicate_item_slot_count=(
                                duplicate_item_slot_count
                            ),
                            sample_evaluation_count=sample_evaluation_count,
                            optimizer_step_count=optimizer_step_count,
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
                    self.pool.release(response.delivery_id, "learner update rolled back")
            elif model_registered and not transaction_complete:
                if ack_started:
                    receipt.setdefault(
                        "ack_attempts", MAX_ATTEMPTS
                    )
                if sample_acked:
                    pending_state = "ACKED_COMMIT_PENDING"
                elif not ack_started:
                    pending_state = "REGISTERED_COMMIT_PENDING"
                elif isinstance(error, UpdateCommitOutcomeUncertain):
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
                        actual_batch_size=len(samples),
                        disposition=pending_state,
                        train_update_id=update_id,
                        train_updates=target_updates,
                        trained_samples=target_samples,
                        stats=stats,
                        requested_train_batch_size=self.train_batch_size,
                        pool_draw_slot_count=pool_draw_slot_count,
                        unique_item_count=unique_item_count,
                        duplicate_item_slot_count=duplicate_item_slot_count,
                        sample_evaluation_count=sample_evaluation_count,
                        optimizer_step_count=optimizer_step_count,
                    )
            elif (
                not training_succeeded
                and not isinstance(error, ValueError)
                and receipt.get("state") == "LEASED"
            ):
                self.pool.release(response.delivery_id, "learner update failed")
            if training_succeeded and isinstance(error, ValueError):
                raise RuntimeError(
                    f"learner update commit failed: {error}"
                ) from error
            raise
        finally:
            renewer.close()
            if rollback is not None and not rollback.get("retained"):
                self._discard_update_rollback(rollback)

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
        requested_train_batch_size: int | None = None,
        pool_draw_slot_count: int | None = None,
        unique_item_count: int | None = None,
        duplicate_item_slot_count: int | None = None,
        sample_evaluation_count: int | None = None,
        optimizer_step_count: int | None = None,
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
                "requested_train_batch_size": requested_train_batch_size,
                "pool_draw_slot_count": pool_draw_slot_count,
                "unique_item_count": unique_item_count,
                "duplicate_item_slot_count": duplicate_item_slot_count,
                "sample_evaluation_count": sample_evaluation_count,
                "optimizer_step_count": optimizer_step_count,
                "disposition": disposition,
                "train_update_id": train_update_id,
                "error": "",
            }

    def _record_train_update_fact(
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
        requested_train_batch_size: int,
        pool_draw_slot_count: int,
        unique_item_count: int,
        duplicate_item_slot_count: int,
        sample_evaluation_count: int,
        optimizer_step_count: int,
    ) -> None:
        writer = getattr(self, "metric_event_writer", None)
        if writer is None:
            raise RuntimeError("Learner TrainUpdate metric writer is unavailable")
        if manifest is None:
            raise RuntimeError("committed TrainUpdate has no model manifest")
        if (
            actual_batch_size <= 0
            or requested_train_batch_size != actual_batch_size
            or pool_draw_slot_count != actual_batch_size
            or unique_item_count < 0
            or duplicate_item_slot_count < 0
            or unique_item_count + duplicate_item_slot_count
            != pool_draw_slot_count
            or sample_evaluation_count <= 0
            or optimizer_step_count <= 0
        ):
            raise MetricEventContractError(
                "TrainUpdate execution facts are contradictory"
            )
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
                metric_registry_pb2.RawMetricSumCount(
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
            training_metrics_pb2.TrainUpdateMetricFact(
                train_update_id=update_id,
                train_update_sequence=int(train_updates),
                published_model=manifest["manifest"].identity,
                delivery_id=delivery_id,
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
                requested_train_batch_size=int(requested_train_batch_size),
                pool_draw_slot_count=int(pool_draw_slot_count),
                unique_item_count=int(unique_item_count),
                duplicate_item_slot_count=int(duplicate_item_slot_count),
                sample_evaluation_count=int(sample_evaluation_count),
                optimizer_step_count=int(optimizer_step_count),
            ),
            observed_at_unix_ms=int(time.time() * 1000),
        )

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
            "actual_batch_size": context.get("actual_batch_size"),
            "requested_train_batch_size": context.get(
                "requested_train_batch_size"
            ),
            "pool_draw_slot_count": context.get("pool_draw_slot_count"),
            "unique_item_count": context.get("unique_item_count"),
            "duplicate_item_slot_count": context.get(
                "duplicate_item_slot_count"
            ),
            "sample_evaluation_count": context.get(
                "sample_evaluation_count"
            ),
            "optimizer_step_count": context.get("optimizer_step_count"),
            "behavior_model": context.get("behavior_model", {}),
            "disposition": context.get("disposition", ""),
            "error": context.get("error", ""),
            "train_update_id": context.get("train_update_id", ""),
        }

    def _finalize_training(self) -> None:
        with self._metrics_lock:
            self._metrics_context["disposition"] = "FINALIZING_REPLAY_POOL"
        self.pool.finalize()
        self.finalize_complete_path.parent.mkdir(parents=True, exist_ok=True)
        self.finalize_complete_path.write_text(
            f"{self.train_updates} {self.trained_samples}\n",
            encoding="utf-8",
        )
        with self._metrics_lock:
            self._metrics_context["disposition"] = "FINALIZED"
        self._finalized = True

    def run(self) -> int:
        try:
            self._start_metric_events()
            if self.preview is not None:
                self.preview.start()
            self._initialize_models()
            while not _stop_requested.is_set():
                if self.finalize_request_path.is_file():
                    self._finalize_training()
                    while not _stop_requested.wait(0.2):
                        pass
                    break
                response = self.pool.next_batch()
                if response is None:
                    continue
                try:
                    self._train_delivery(response)
                except ValueError as error:
                    self.pool.acknowledge_invalid(response)
                    raise RuntimeError(str(error)) from error
            with self._metrics_lock:
                self._metrics_context["disposition"] = "STOPPING"
            self.pool.finalize()
            if self.preview is not None:
                self.preview.record_best_effort("graceful shutdown")
            return 0
        except Exception as error:
            with self._metrics_lock:
                self._metrics_context["error"] = str(error)
                self._metrics_context["disposition"] = "FAILED"
            self.logger.exception("Learner training failed")
            if self.preview is not None:
                self.preview.record_best_effort("failure reporting")
            return 1
        finally:
            self._stop_metric_events()
            self.model_channel.close()
            self.pool.close()


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
