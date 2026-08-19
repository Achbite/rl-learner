import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml

from main.training_runtime import ModelPublisher, TrainingRuntime, read_json
from proto import training_pb2
from src.contracts.identity import (
    canonical_config_digest,
    contract_identity,
    manifest_message,
    service_identity,
    training_config_document,
    training_semantics,
)
from src.training.ppo_trainer import PPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def config(root: Path) -> dict:
    document = yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    document["model"]["local_train_dir"] = str(root / "local-train")
    document["model"]["archive_interval_updates"] = 50
    document["training"]["n_epochs"] = 1
    document["training"]["mini_batch_size"] = 2
    document["identity"]["training_config_digest"] = (
        canonical_config_digest(training_config_document(document))
    )
    return document


def neutral_samples() -> list[dict]:
    return [
        {
            "observation": [0.0] * 17,
            "action": 0,
            "old_log_probability": -2.0,
            "old_value_prediction": 0.1,
            "advantage": 0.5,
            "td_return": 0.6,
            "behavior_model_step": 0,
        },
        {
            "observation": [0.1] * 17,
            "action": 1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "advantage": -0.25,
            "td_return": -0.05,
            "behavior_model_step": 0,
        },
    ]


def model_authority():
    return service_identity(
        "model-distributor", "model-distributor-transaction-test", 1
    )


def model_status(contract):
    return training_pb2.ModelDistributorStatusRsp(
        distributor=model_authority(),
        ready=True,
        contract=contract,
    )


def sample_pool_authority():
    return service_identity(
        "sample-pool", "sample-pool-transaction-test", 1
    )


class FakeLeaseRenewer:
    def __init__(self, error: str = ""):
        self.error = error
        self.closed = False

    def start(self):
        return self

    def renew_now(self):
        return None

    def close(self) -> None:
        self.closed = True


class UpdateTransactionRecoveryTest(unittest.TestCase):
    def runtime(self, root: Path) -> tuple[TrainingRuntime, dict]:
        cfg = config(root)
        trainer = PPOTrainer(cfg)
        publisher = ModelPublisher(cfg)
        publisher.prepare()
        bootstrap = publisher.publish_runtime(
            trainer,
            train_update_id="bootstrap-v0",
            behavior_model=None,
            batch_ids=[],
            train_updates=0,
            trained_samples=0,
        )

        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.trainer = trainer
        runtime.publisher = publisher
        runtime.learner_service = service_identity(
            "learner", "learner-transaction-test", 1
        )
        runtime.sample_stub = SimpleNamespace()
        runtime.model_stub = SimpleNamespace(
            GetModelDistributorStatus=lambda request, timeout: model_status(
                runtime.contract
            )
        )
        runtime.lease_timeout_ms = 30000
        runtime.train_updates = 0
        runtime.trained_samples = 0
        runtime._run_start_train_updates = 0
        runtime._run_start_trained_samples = 0
        runtime.last_stats = {}
        runtime.model_manifests = {0: bootstrap}
        runtime._metrics_lock = threading.RLock()
        runtime._committed_learner_metrics = {}
        runtime._metrics_context = {}
        runtime.logger = SimpleNamespace(
            info=lambda *args: None,
            warning=lambda *args: None,
            error=lambda *args: None,
        )
        runtime._validate_delivery = (
            lambda response, allow_partial=False: bootstrap["identity"]
        )
        runtime._training_samples = lambda batches: neutral_samples()
        runtime._record_metrics = lambda: None
        runtime._nack = mock.Mock()
        return runtime, bootstrap

    @staticmethod
    def response():
        return SimpleNamespace(
            delivery_id="delivery-transaction-1",
            actual_batch_size=2,
            batches=[SimpleNamespace(batch_id="batch-transaction-1")],
            sample_pool=sample_pool_authority(),
        )

    @staticmethod
    def trainer_snapshot(trainer: PPOTrainer) -> dict:
        return {
            "model": copy.deepcopy(trainer.model.state_dict()),
            "optimizer": copy.deepcopy(trainer._optimizer.state_dict()),
            "model_step": trainer.model_step,
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "training": trainer.model.training,
        }

    def assert_nested_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
            return
        if isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
            return
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
            return
        if isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_equal(left_item, right_item)
            return
        self.assertEqual(left, right)

    def test_update_publishes_registers_acks_and_emits_one_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bootstrap = self.runtime(Path(directory))
            before = copy.deepcopy(runtime.trainer.model.state_dict())
            runtime._validate_delivery = lambda response, allow_partial=False: {
                "model_lineage_id": bootstrap["identity"]["model_lineage_id"],
                "minimum_model_step": 0,
                "maximum_model_step": 0,
                "models": [dict(bootstrap["identity"])],
            }
            order = []
            emitted = []

            def register(_manifest):
                order.append("register")
                return model_authority()

            def ack(*_args, **_kwargs):
                order.append("ack")

            def append(fact, committed_at_unix_ms):
                order.append("fact")
                emitted.append((copy.deepcopy(fact), committed_at_unix_ms))

            runtime._register = register
            runtime._ack = ack
            runtime.metric_event_writer = SimpleNamespace(append=append)
            runtime.metric_event_store = None
            with mock.patch(
                "main.training_runtime.LeaseRenewer",
                return_value=FakeLeaseRenewer(),
            ):
                runtime._train_delivery(self.response())

            self.assertEqual(order, ["register", "ack", "fact"])
            self.assertEqual(runtime.trainer.model_step, 1)
            self.assertEqual(runtime.train_updates, 1)
            self.assertEqual(runtime.trained_samples, 2)
            self.assertTrue(
                any(
                    not torch.equal(
                        value, runtime.trainer.model.state_dict()[key]
                    )
                    for key, value in before.items()
                )
            )
            self.assertIsNotNone(runtime.publisher.complete_manifest(1))
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ACKED")
            self.assertEqual(len(emitted), 1)
            fact, committed_at_unix_ms = emitted[0]
            self.assertEqual(fact.delivery_id, "delivery-transaction-1")
            self.assertEqual(fact.actual_batch_size, 2)
            self.assertEqual(fact.cumulative_trained_samples, 2)
            self.assertEqual(fact.published_model.model_step, 1)
            self.assertGreater(committed_at_unix_ms, 0)
            runtime._nack.assert_not_called()

    def test_lease_loss_after_optimizer_rolls_back_the_update(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            before = self.trainer_snapshot(runtime.trainer)
            publisher_state = read_json(runtime.publisher.state_path)
            renewer = FakeLeaseRenewer("injected lease loss after optimizer")

            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                with self.assertRaisesRegex(RuntimeError, "sample lease lost"):
                    runtime._train_delivery(self.response())

            self.assert_nested_equal(
                before, self.trainer_snapshot(runtime.trainer)
            )
            self.assertEqual(
                read_json(runtime.publisher.state_path), publisher_state
            )
            self.assertFalse(runtime.publisher.publication_path(1).exists())
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ROLLED_BACK")
            runtime._nack.assert_called_once_with(
                "delivery-transaction-1", "learner update rolled back"
            )
            self.assertTrue(renewer.closed)

    def test_checkpoint_or_export_failure_does_not_publish_a_model(self):
        for failed_step in ("checkpoint", "export"):
            with self.subTest(failed_step=failed_step):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    before = self.trainer_snapshot(runtime.trainer)
                    if failed_step == "checkpoint":
                        original = runtime.publisher.commit_optimizer_checkpoint

                        def fail_checkpoint(*args, **kwargs):
                            original(*args, **kwargs)
                            raise RuntimeError("injected checkpoint failure")

                        runtime.publisher.commit_optimizer_checkpoint = (
                            fail_checkpoint
                        )
                    else:
                        original = runtime.publisher.publish_runtime

                        def fail_export(*args, **kwargs):
                            original(*args, **kwargs)
                            raise RuntimeError("injected export failure")

                        runtime.publisher.publish_runtime = fail_export
                    runtime._register = mock.Mock()
                    runtime._ack = mock.Mock()

                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=FakeLeaseRenewer(),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, f"injected {failed_step} failure"
                        ):
                            runtime._train_delivery(self.response())

                    self.assert_nested_equal(
                        before, self.trainer_snapshot(runtime.trainer)
                    )
                    self.assertFalse(
                        runtime.publisher.publication_path(1).exists()
                    )
                    runtime._register.assert_not_called()
                    runtime._ack.assert_not_called()

    def test_bootstrap_wait_is_unbounded_until_the_exact_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bootstrap = self.runtime(Path(directory))
            expected = manifest_message(
                TrainingRuntime._manifest_for_wire(bootstrap)
            ).identity
            waiting = model_status(runtime.contract)
            loaded = model_status(runtime.contract)
            loaded.latest_ack_status = training_pb2.MODEL_LOAD_STATUS_LOADED
            loaded.latest_ack_model.CopyFrom(expected)
            runtime.model_stub.GetModelDistributorStatus = mock.Mock(
                side_effect=(waiting, loaded)
            )
            runtime.initial_model_ack_timeout = None

            with mock.patch("main.training_runtime.time.sleep"):
                self.assertTrue(
                    runtime._wait_initial_model_loaded(bootstrap)
                )

            self.assertEqual(
                runtime.model_stub.GetModelDistributorStatus.call_count, 2
            )

    def test_explicit_stop_ends_the_unbounded_bootstrap_wait_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bootstrap = self.runtime(Path(directory))
            runtime.initial_model_ack_timeout = None
            stop = threading.Event()
            stop.set()
            runtime.model_stub.GetModelDistributorStatus = mock.Mock()

            with mock.patch("main.training_runtime._stop_requested", stop):
                self.assertFalse(
                    runtime._wait_initial_model_loaded(bootstrap)
                )

            runtime.model_stub.GetModelDistributorStatus.assert_not_called()
