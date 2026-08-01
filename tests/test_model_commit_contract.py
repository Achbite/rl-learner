import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from main.training_runtime import (
    ModelPublisher,
    TrainingRuntime,
    atomic_write_json,
    read_json,
)
from proto import maze_pb2
from src.training.ppo_trainer import PPOTrainer


def config(root: Path, archive_interval: int = 2) -> dict:
    return {
        "model": {
            "obs_dim": 3,
            "action_dim": 2,
            "hidden_dim": 8,
            "bootstrap_seed": 7,
            "local_train_dir": str(root / "models" / "local-train"),
            "archive_interval_updates": archive_interval,
            "archive_on_graceful_shutdown": True,
            "serving_retention_versions": 2,
        },
        "training": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 2,
            "mini_batch_size": 2,
            "normalize_advantage": True,
        },
    }


def samples() -> list[dict]:
    return [
        {
            "obs": [0.0, 0.1, 0.2],
            "action": 0,
            "old_log_prob": -0.7,
            "advantage": 1.0,
            "td_return": 1.5,
        },
        {
            "obs": [0.2, 0.3, 0.4],
            "action": 1,
            "old_log_prob": -0.6,
            "advantage": -0.5,
            "td_return": 0.25,
        },
    ]


def publish_update(
    trainer: PPOTrainer,
    publisher: ModelPublisher,
    update_number: int,
) -> tuple[dict, dict]:
    stats = trainer.train_on_batch(samples())
    version = trainer.model_version
    update_id = f"update-v{version}"
    publisher.commit_optimizer_checkpoint(
        trainer,
        train_update_id=update_id,
        behavior_model_version=version - 1,
        batch_ids=[f"batch-{update_number}"],
        stats=stats,
        sample_count=2,
        train_updates=version,
        trained_samples=version * 2,
    )
    manifest = publisher.publish_runtime(
        trainer,
        train_update_id=update_id,
        behavior_model_version=version - 1,
        batch_ids=[f"batch-{update_number}"],
        stats=stats,
        sample_count=2,
        train_updates=version,
        trained_samples=version * 2,
        checkpoint_precommitted=True,
    )
    return manifest, stats


class ModelCommitContractTest(unittest.TestCase):
    def test_prepare_rejects_unclean_local_train_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ModelPublisher(config(root))
            first.prepare()
            (first.metrics_dir / "existing.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            second = ModelPublisher(config(root))
            with self.assertRaisesRegex(RuntimeError, "was not cleaned"):
                second.prepare()

    def test_manifest_is_the_complete_runtime_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = PPOTrainer(config(root))
            publisher = ModelPublisher(config(root))
            publisher.prepare()
            publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            self.assertIsNotNone(publisher.complete_manifest(0))

            manifest, _ = publish_update(trainer, publisher, 1)
            self.assertEqual(manifest["contract_version"], "0.6.0")
            self.assertNotIn("run_id", manifest)
            self.assertIsNotNone(
                publisher.complete_manifest(1, "update-v1")
            )

            with publisher.model_path(1).open("ab") as stream:
                stream.write(b"corrupt")
            self.assertIsNone(publisher.complete_manifest(1))

    def test_archive_retention_and_explicit_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            trainer = PPOTrainer(config(source_root))
            publisher = ModelPublisher(config(source_root))
            publisher.prepare()
            publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            publisher.archive_version(0, "bootstrap")
            publish_update(trainer, publisher, 1)
            self.assertFalse(
                (publisher.archive_dir / "000001").exists()
            )
            publish_update(trainer, publisher, 2)
            publisher.archive_version(2, "interval")
            publisher.prune_runtime(2)

            self.assertTrue(
                (
                    publisher.archive_dir
                    / "000000"
                    / "manifest.json"
                ).is_file()
            )
            savepoint = publisher.archive_dir / "000002"
            self.assertTrue(
                (savepoint / "manifest.json").is_file()
            )
            self.assertTrue((savepoint / "SaveModel.onnx").is_file())
            self.assertTrue((savepoint / "checkpoint.pt").is_file())
            self.assertEqual(
                {path.name for path in savepoint.iterdir()},
                {"SaveModel.onnx", "checkpoint.pt", "manifest.json"},
            )
            archive_manifest = read_json(savepoint / "manifest.json")
            self.assertEqual(
                archive_manifest["artifact_uri"],
                (savepoint / "SaveModel.onnx").as_uri(),
            )
            self.assertFalse(publisher.model_path(0).exists())
            self.assertTrue(publisher.model_path(1).exists())
            self.assertTrue(publisher.model_path(2).exists())

            external_checkpoint = root / "savepoints" / "checkpoint.pt"
            external_checkpoint.parent.mkdir()
            shutil.copyfile(
                savepoint / "checkpoint.pt",
                external_checkpoint,
            )

            child_root = root / "child"
            resumed = PPOTrainer(config(child_root))
            child = ModelPublisher(config(child_root))
            child.prepare()
            restored = child.load_initial_checkpoint(
                resumed, str(external_checkpoint)
            )
            self.assertEqual(resumed.model_version, 2)
            self.assertEqual(restored["train_updates"], 2)
            self.assertEqual(restored["trained_samples"], 4)
            self.assertEqual(restored["initial_model_version"], 2)
            self.assertTrue(restored["initial_checkpoint_sha256"])
            runtime = TrainingRuntime.__new__(TrainingRuntime)
            runtime.trainer = resumed
            runtime.publisher = child
            runtime._startup_mode = "initial-checkpoint"
            runtime.train_updates = 2
            runtime.trained_samples = 4
            runtime._last_archive_version = None
            runtime._behavior_checksums = {}
            runtime.logger = mock.Mock()
            runtime._register = mock.Mock()
            runtime._initialize_models()

            child_manifest = child.complete_manifest(
                2, "initial-checkpoint"
            )
            self.assertIsNotNone(child_manifest)
            self.assertEqual(
                child_manifest["initial_model_version"], 2
            )
            self.assertEqual(
                child_manifest["initial_checkpoint_sha256"],
                restored["initial_checkpoint_sha256"],
            )
            self.assertNotIn("run_id", child_manifest)
            initial_archive = read_json(
                child.archive_dir
                / "000002"
                / "manifest.json"
            )
            self.assertEqual(
                initial_archive["archive_reason"], "initial-checkpoint"
            )
            self.assertEqual(runtime._last_archive_version, 2)

    def test_production_archive_interval_and_graceful_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = PPOTrainer(config(root, archive_interval=200))
            publisher = ModelPublisher(config(root, archive_interval=200))
            publisher.prepare()
            publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            publisher.archive_version(0, "bootstrap")
            self.assertFalse(publisher.should_archive(199))
            self.assertTrue(publisher.should_archive(200))
            self.assertFalse(publisher.should_archive(201))

            trainer._model_version = 200
            publisher.publish_runtime(
                trainer,
                train_update_id="update-v200",
                behavior_model_version=199,
                batch_ids=["batch-200"],
                train_updates=200,
                trained_samples=102400,
            )
            publisher.archive_version(200, "interval")
            trainer._model_version = 201
            publisher.publish_runtime(
                trainer,
                train_update_id="update-v201",
                behavior_model_version=200,
                batch_ids=["batch-201"],
                train_updates=201,
                trained_samples=102912,
            )
            publisher.archive_version(201, "graceful-shutdown")

            self.assertTrue(
                (
                    publisher.archive_dir
                    / "000000"
                    / "manifest.json"
                ).is_file()
            )
            self.assertFalse(
                (publisher.archive_dir / "000199").exists()
            )
            self.assertTrue(
                (
                    publisher.archive_dir
                    / "000200"
                    / "manifest.json"
                ).is_file()
            )
            final_manifest = read_json(
                publisher.archive_dir
                / "000201"
                / "manifest.json"
            )
            self.assertEqual(
                final_manifest["archive_reason"], "graceful-shutdown"
            )

    def test_checkpoint_restores_rng_for_deterministic_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = PPOTrainer(config(root))
            publisher = ModelPublisher(config(root))
            publisher.prepare()
            publisher.publish_runtime(
                first,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            first_stats = first.train_on_batch(samples())
            first_state = {
                key: value.detach().clone()
                for key, value in first.model.state_dict().items()
            }

            retry = PPOTrainer(config(root))
            self.assertTrue(
                retry.load_checkpoint(str(publisher.checkpoint_path(0)))
            )
            retry_stats = retry.train_on_batch(samples())
            self.assertEqual(first_stats, retry_stats)
            for key, expected in first_state.items():
                self.assertTrue(
                    torch.equal(expected, retry.model.state_dict()[key]),
                    key,
                )

    def test_receipt_reconciliation_closes_commit_fault_windows(self):
        class AlreadyAppliedSamplePool:
            def __init__(self):
                self.requests = []

            def AckBatch(self, request, timeout):
                self.requests.append((request, timeout))
                return maze_pb2.DeliveryRsp(
                    ret_code=0,
                    result=maze_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
                    delivery_id=request.delivery_id,
                    disposition=request.disposition,
                    train_update_id=request.train_update_id,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = PPOTrainer(config(root))
            publisher = ModelPublisher(config(root))
            publisher.prepare()
            publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            manifest, stats = publish_update(trainer, publisher, 1)

            for initial_state in (
                "LEASED",
                "RUNTIME_COMMITTED",
                "MODEL_COMMITTED",
                "REGISTERED",
            ):
                with self.subTest(initial_state=initial_state):
                    receipt_path = publisher.receipt_path("update-v1")
                    atomic_write_json(
                        receipt_path,
                        {
                            "schema_version": 1,
                            "train_update_id": "update-v1",
                            "behavior_model_version": 0,
                            "batch_ids": ["batch-1"],
                            "target_model_version": 1,
                            "delivery_id": "delivery-0",
                            "state": initial_state,
                            "manifest": manifest,
                            "stats": stats,
                            "sample_count": 2,
                            "train_updates": 1,
                            "trained_samples": 2,
                        },
                    )
                    sample_pool = AlreadyAppliedSamplePool()
                    runtime = TrainingRuntime.__new__(TrainingRuntime)
                    runtime.consumer_id = "learner-restarted"
                    runtime.publisher = publisher
                    runtime.sample_stub = sample_pool
                    runtime._acked_update_ids = set()
                    runtime._accounted_update_ids = set()
                    runtime._recorded_update_ids = set()
                    runtime._last_archive_version = None
                    runtime.train_updates = 0
                    runtime.trained_samples = 0
                    runtime.last_stats = {}
                    runtime._register = mock.Mock()
                    runtime._record_metrics = mock.Mock()

                    runtime._reconcile_receipts()

                    receipt = read_json(receipt_path)
                    self.assertEqual(receipt["state"], "ACKED")
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertEqual(runtime.trained_samples, 2)
                    self.assertEqual(len(sample_pool.requests), 1)
                    request, timeout = sample_pool.requests[0]
                    self.assertEqual(request.delivery_id, "delivery-0")
                    self.assertEqual(
                        request.disposition,
                        maze_pb2.ACK_DISPOSITION_TRAINED,
                    )
                    self.assertEqual(request.train_update_id, "update-v1")
                    self.assertEqual(timeout, 3.0)
                    runtime._record_metrics.assert_called_once()
                    if initial_state == "REGISTERED":
                        runtime._register.assert_not_called()
                    else:
                        runtime._register.assert_called_once()


if __name__ == "__main__":
    unittest.main()
