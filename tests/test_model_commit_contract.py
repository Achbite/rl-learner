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


def config(root: Path) -> dict:
    return {
        "model": {
            "obs_dim": 3,
            "action_dim": 2,
            "hidden_dim": 8,
            "bootstrap_seed": 7,
            "distribution_dir": str(root / "published"),
            "checkpoint_dir": str(root / "checkpoints"),
            "update_dir": str(root / "updates"),
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


class ModelCommitContractTest(unittest.TestCase):
    def test_manifest_is_the_complete_version_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = PPOTrainer(config(root))
            publisher = ModelPublisher(config(root), "run-commit")
            publisher.publish(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            self.assertIsNotNone(publisher.complete_manifest(0))

            stats = trainer.train_on_batch(samples())
            trainer.export_onnx(str(publisher.model_path(1)))
            trainer.save_checkpoint(
                str(publisher.checkpoint_path(1)),
                metadata={"train_update_id": "update-v1"},
            )
            self.assertIsNone(publisher.complete_manifest(1))
            self.assertEqual(
                publisher.latest_complete_checkpoint(),
                publisher.checkpoint_path(0),
            )

            publisher.publish(
                trainer,
                train_update_id="update-v1",
                behavior_model_version=0,
                batch_ids=["batch-0"],
                stats=stats,
                sample_count=2,
            )
            self.assertIsNotNone(
                publisher.complete_manifest(1, "update-v1")
            )

            with publisher.model_path(1).open("ab") as stream:
                stream.write(b"corrupt")
            self.assertIsNone(publisher.complete_manifest(1))

    def test_checkpoint_restores_rng_for_deterministic_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = PPOTrainer(config(root))
            publisher = ModelPublisher(config(root), "run-retry")
            publisher.publish(
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

    def test_receipt_reconciliation_closes_each_commit_fault_window(self):
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
            publisher = ModelPublisher(config(root), "run-reconcile")
            publisher.publish(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model_version=None,
                batch_ids=[],
            )
            stats = trainer.train_on_batch(samples())
            manifest = publisher.publish(
                trainer,
                train_update_id="update-v1",
                behavior_model_version=0,
                batch_ids=["batch-0"],
                stats=stats,
                sample_count=2,
            )

            for initial_state in (
                "LEASED",
                "MODEL_COMMITTED",
                "REGISTERED",
            ):
                with self.subTest(initial_state=initial_state):
                    receipt_path = publisher.receipt_path("update-v1")
                    atomic_write_json(
                        receipt_path,
                        {
                            "schema_version": 1,
                            "run_id": "run-reconcile",
                            "train_update_id": "update-v1",
                            "behavior_model_version": 0,
                            "batch_ids": ["batch-0"],
                            "target_model_version": 1,
                            "delivery_id": "delivery-0",
                            "state": initial_state,
                            "manifest": manifest,
                            "stats": stats,
                            "sample_count": 2,
                        },
                    )
                    sample_pool = AlreadyAppliedSamplePool()
                    runtime = TrainingRuntime.__new__(TrainingRuntime)
                    runtime.run_id = "run-reconcile"
                    runtime.consumer_id = "learner-restarted"
                    runtime.publisher = publisher
                    runtime.sample_stub = sample_pool
                    runtime._acked_update_ids = set()
                    runtime._recorded_update_ids = set()
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
