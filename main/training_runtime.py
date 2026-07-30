"""Run the leased-fragment PPO training and model publication loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
        run_id=document["run_id"],
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
    def __init__(self, config: dict, run_id: str):
        model = config.get("model", {})
        self.run_id = run_id
        self.seed = int(model.get("bootstrap_seed", 0))
        self.obs_dim = int(model.get("obs_dim", 13))
        self.action_dim = int(model.get("action_dim", 9))
        self.published_dir = Path(
            os.environ.get(
                "MAZE_MODEL_ARTIFACT_ROOT",
                model.get("distribution_dir", "models/published"),
            )
        ).resolve() / run_id
        checkpoint_root = Path(
            os.environ.get(
                "MAZE_CHECKPOINT_ROOT",
                model.get("checkpoint_dir", "models/checkpoints"),
            )
        ).resolve()
        update_root = Path(
            os.environ.get(
                "MAZE_UPDATE_RECEIPT_ROOT",
                model.get("update_dir", "models/updates"),
            )
        ).resolve()
        self.checkpoint_dir = checkpoint_root / run_id
        self.update_dir = update_root / run_id
        for directory in (
            self.published_dir,
            self.checkpoint_dir,
            self.update_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def model_path(self, version: int) -> Path:
        return self.published_dir / f"model_v{version:06d}.onnx"

    def manifest_path(self, version: int) -> Path:
        return self.published_dir / f"manifest_v{version:06d}.json"

    def checkpoint_path(self, version: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_v{version:06d}.pt"

    def receipt_path(self, train_update_id: str) -> Path:
        return self.update_dir / f"{train_update_id}.json"

    def publish(
        self,
        trainer: PPOTrainer,
        *,
        train_update_id: str,
        behavior_model_version: int | None,
        batch_ids: list[str],
        stats: dict | None = None,
        sample_count: int = 0,
    ) -> dict:
        version = trainer.model_version
        model_path = self.model_path(version)
        checkpoint_path = self.checkpoint_path(version)
        manifest_path = self.manifest_path(version)
        temporary_model = model_path.with_name(
            f".{model_path.name}.{os.getpid()}.tmp"
        )
        temporary_checkpoint = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        metadata = {
            "run_id": self.run_id,
            "train_update_id": train_update_id,
            "behavior_model_version": behavior_model_version,
            "batch_ids": batch_ids,
            "stats": stats or {},
            "sample_count": sample_count,
        }
        trainer.export_onnx(str(temporary_model))
        trainer.save_checkpoint(str(temporary_checkpoint), metadata=metadata)
        for path in (temporary_checkpoint, temporary_model):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary_checkpoint, checkpoint_path)
        os.replace(temporary_model, model_path)

        manifest = {
            "schema_version": 1,
            "contract_version": "0.3.0",
            "run_id": self.run_id,
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
        }
        atomic_write_json(manifest_path, manifest)
        return manifest

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
            or manifest.get("contract_version") != "0.3.0"
            or manifest.get("run_id") != self.run_id
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


class LeaseRenewer:
    def __init__(
        self,
        stub,
        run_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_timeout_ms: int,
    ):
        self.stub = stub
        self.request = maze_pb2.RenewLeaseReq(
            run_id=run_id,
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
    def __init__(self, config: dict, run_id: str):
        self.config = config
        self.run_id = run_id
        self.learner_id = os.environ.get("MAZE_LEARNER_ID", "learner-0")
        self.consumer_id = (
            f"{self.learner_id}-{os.getpid()}-{int(time.time() * 1000)}"
        )
        self.logger = setup_logger("TrainingRuntime")
        self.trainer = PPOTrainer(config)
        self.publisher = ModelPublisher(config, run_id)
        self.sequence = 0
        self.train_updates = 0
        self.trained_samples = 0
        self.last_stats: dict = {}
        self._acked_update_ids: set[str] = set()
        self._recorded_update_ids: set[str] = set()
        self._behavior_checksums: dict[int, str] = {}

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

        dashboard = config.get("dashboard", {})
        self.metrics = create_backend(
            dashboard.get("backend", "jsonl"),
            dashboard.get("metrics_dir", "logs/metrics"),
            run_id=run_id,
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
                    if record.get("run_id") != self.run_id:
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
                receipt.get("run_id") != self.run_id
                or receipt.get("state") != "ACKED"
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
            self.train_updates += 1
            self.trained_samples += int(receipt.get("sample_count", 0))
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

    def _restore_or_publish_v0(self) -> None:
        checkpoint = self.publisher.latest_complete_checkpoint()
        if checkpoint is not None:
            if not self.trainer.load_checkpoint(str(checkpoint)):
                raise RuntimeError(f"cannot restore checkpoint: {checkpoint}")
        else:
            manifest = self.publisher.publish(
                self.trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            self.logger.info(
                "初始模型已提交: version=0 checksum=%s",
                manifest["sha256"],
            )
        manifests = self.publisher.complete_manifests()
        if not manifests:
            raise RuntimeError("checkpoint exists without a model manifest")
        for manifest in manifests:
            self._register(manifest)
            self._behavior_checksums[int(manifest["model_version"])] = str(
                manifest["sha256"]
            )

    def _model_status(self):
        return self.model_stub.GetModelDistributorStatus(
            maze_pb2.ModelDistributorStatusReq(run_id=self.run_id),
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
            run_id=self.run_id,
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
                run_id=self.run_id,
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
            maze_pb2.DistributorStatusReq(run_id=self.run_id), timeout=2.0
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
            if batch.protocol_version != 2 or batch.run_id != self.run_id:
                raise ValueError("fragment protocol or run identity is invalid")
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
                "run_id": self.run_id,
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
            "MODEL_COMMITTED",
            manifest=manifest,
            stats=metadata.get("stats", {}),
            sample_count=int(metadata.get("sample_count", 0)),
        )

    def _ensure_trainer_version(self, version: int) -> None:
        if self.trainer.model_version == version:
            return
        checkpoint = self.publisher.checkpoint_path(version)
        if not self.trainer.load_checkpoint(str(checkpoint)):
            raise RuntimeError(
                f"cannot restore committed model v{version}"
            )

    def _remember_acked_receipt(self, receipt: dict) -> None:
        update_id = str(receipt["train_update_id"])
        if update_id not in self._acked_update_ids:
            self._acked_update_ids.add(update_id)
            self.train_updates += 1
            self.trained_samples += int(receipt.get("sample_count", 0))
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
            if (
                receipt.get("run_id") != self.run_id
                or not receipt.get("train_update_id")
            ):
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
            if state == "LEASED" and self._checkpoint_committed(receipt):
                receipt = self._recover_committed_receipt(path, receipt)
                state = receipt["state"]
            if state == "MODEL_COMMITTED":
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
                "run_id": self.run_id,
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
            receipt["state"] == "LEASED"
            and self._checkpoint_committed(receipt)
        ):
            receipt = self._recover_committed_receipt(
                receipt_path, receipt
            )

        renewer = LeaseRenewer(
            self.sample_stub,
            self.run_id,
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
                manifest = self.publisher.publish(
                    self.trainer,
                    train_update_id=train_update_id,
                    behavior_model_version=behavior_version,
                    batch_ids=batch_ids,
                    stats=stats,
                    sample_count=len(samples),
                )
                self._behavior_checksums[
                    int(manifest["model_version"])
                ] = str(manifest["sha256"])
                receipt = self._write_receipt(
                    receipt_path,
                    receipt,
                    "MODEL_COMMITTED",
                    manifest=manifest,
                    stats=stats,
                    sample_count=len(samples),
                )
            else:
                self._ensure_trainer_version(
                    receipt["target_model_version"]
                )
                stats = receipt.get("stats", {})

            if receipt["state"] == "MODEL_COMMITTED":
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

            self._remember_acked_receipt(receipt)
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
        self.sequence += 1
        try:
            distributor_status = self._sample_status()
            distributor = {
                "ready": distributor_status.ready,
                "instance_id": distributor_status.distributor_instance_id,
                "accepted": distributor_status.accepted_unique_samples,
                "acked": distributor_status.acked_unique_samples,
                "ready_samples": distributor_status.ready_queue_samples,
                "leased_samples": distributor_status.leased_samples,
                "trained": distributor_status.trained_sample_count,
                "stale": distributor_status.stale_sample_count,
                "invalid": distributor_status.invalid_sample_count,
                "shutdown_untrained": (
                    distributor_status.shutdown_untrained_sample_count
                ),
                "redelivered": distributor_status.redelivery_count,
                "lease_renewals": distributor_status.lease_renew_count,
            }
        except grpc.RpcError as exc:
            distributor = {"ready": False, "error": exc.details() or str(exc)}
        try:
            actor_status = self.aiserver_stub.GetAIServerStatus(
                maze_pb2.AIServerStatusReq(run_id=self.run_id), timeout=2.0
            )
            actor = {
                "ready": actor_status.ready,
                "state": maze_pb2.AIServerState.Name(actor_status.state),
                "produced": actor_status.produced_unique_samples,
                "accepted": actor_status.accepted_unique_samples,
                "outbound_pending": actor_status.outbound_queue_samples,
                "final_drop": actor_status.final_drop_unique_samples,
                "model_version": actor_status.loaded_model_version,
                "model_checksum": actor_status.loaded_model_checksum,
                "staged_model_version": actor_status.staged_model_version,
                "model_switches": actor_status.model_switch_count,
                "quarantined_samples": actor_status.quarantined_sample_count,
            }
        except grpc.RpcError as exc:
            actor = {"ready": False, "error": exc.details() or str(exc)}
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
            }
        except grpc.RpcError as exc:
            model = {"ready": False, "error": exc.details() or str(exc)}

        learner = {
            "train_updates": self.train_updates,
            "trained_samples": self.trained_samples,
            "actual_batch_size": actual_batch_size,
            "behavior_model_version": behavior_version,
            "model_version": self.trainer.model_version,
            "train_update_id": train_update_id,
            "ack_disposition": disposition,
            **stats,
            "error": error,
        }
        record = {
            "schema_version": 2,
            "run_id": self.run_id,
            "mode": "training",
            "sequence": self.sequence,
            "train_step": self.train_updates,
            "timestamp": time.time(),
            "timestamp_ms": int(time.time() * 1000),
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
            **stats,
        }
        self.metrics.write(record)

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
        try:
            self._restore_or_publish_v0()
            self.logger.info(
                "Learner training 就绪: run=%s model=v%d batch=%d",
                self.run_id,
                self.trainer.model_version,
                self.train_batch_size,
            )
            while not _stop_requested.is_set():
                try:
                    self._drain_stale()
                    self._reconcile_receipts()
                    delivery = self._select_batch()
                except grpc.RpcError as exc:
                    self.logger.warning(
                        "等待 SampleDistributor: %s",
                        exc.details() or str(exc),
                    )
                    _stop_requested.wait(0.5)
                    continue
                if delivery is None:
                    continue
                self._process_delivery(delivery)
        finally:
            if _stop_requested.is_set():
                self._drain_shutdown()
            self.metrics.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPO training")
    parser.add_argument("--config", default="configs/learner_config.yaml")
    parser.add_argument(
        "--run-id", default=os.environ.get("MAZE_RUN_ID", "local-run")
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    runtime = TrainingRuntime(load_config(args.config), args.run_id)
    runtime.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
