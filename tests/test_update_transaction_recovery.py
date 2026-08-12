import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import grpc
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


def samples() -> list[dict]:
    return [
        {
            "observation": [0.0] * 17,
            "action": 0,
            "old_log_probability": -2.0,
            "old_value_prediction": 0.1,
            "advantage": 0.5,
            "td_return": 0.6,
            "behavior_model_version": 0,
        },
        {
            "observation": [0.1] * 17,
            "action": 1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "advantage": -0.25,
            "td_return": -0.05,
            "behavior_model_version": 0,
        },
    ]


def model_authority(
    instance_id: str = "model-distributor-transaction-test",
    lifecycle_epoch: int = 1,
):
    return service_identity(
        "model-distributor", instance_id, lifecycle_epoch
    )


def model_status(contract=None, authority=None):
    return training_pb2.ModelDistributorStatusRsp(
        distributor=authority or model_authority(),
        ready=True,
        contract=contract,
    )


def sample_authority(
    instance_id: str = "sample-distributor-transaction-test",
    lifecycle_epoch: int = 1,
):
    return service_identity(
        "sample-distributor", instance_id, lifecycle_epoch
    )


class FakeLeaseRenewer:
    def __init__(self, error: str = "", renew_error: str = ""):
        self.error = error
        self.renew_error = renew_error
        self.closed = False
        self.renew_count = 0

    def start(self):
        return self

    def renew_now(self):
        self.renew_count += 1
        if self.renew_error:
            raise RuntimeError(self.renew_error)

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
        runtime._last_archive_version = 0
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
        runtime._training_samples = lambda batches: samples()
        runtime._record_metrics = lambda: None
        runtime._nack = mock.Mock()
        return runtime, bootstrap

    @staticmethod
    def response():
        return SimpleNamespace(
            delivery_id="delivery-transaction-1",
            actual_batch_size=2,
            batches=[SimpleNamespace(batch_id="batch-transaction-1")],
            distributor=sample_authority(),
        )

    @staticmethod
    def trainer_snapshot(trainer: PPOTrainer) -> dict:
        return {
            "model": copy.deepcopy(trainer.model.state_dict()),
            "optimizer": copy.deepcopy(trainer._optimizer.state_dict()),
            "model_version": trainer.model_version,
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

    def assert_trainer_matches_checkpoint(
        self, runtime: TrainingRuntime, version: int
    ) -> None:
        actual = self.trainer_snapshot(runtime.trainer)
        self.assert_checkpoint_matches_snapshot(
            runtime,
            runtime.publisher.checkpoint_path(version),
            actual,
        )

    def assert_checkpoint_matches_snapshot(
        self,
        runtime: TrainingRuntime,
        checkpoint_path: Path,
        expected: dict,
    ) -> None:
        restored = PPOTrainer(runtime.publisher.config)
        self.assertTrue(restored.load_checkpoint(str(checkpoint_path)))
        restored.model.train(bool(expected["training"]))
        self.assert_nested_equal(expected, self.trainer_snapshot(restored))

    def test_optimizer_success_then_lease_loss_restores_pre_update_state(self):
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
            self.assertEqual(read_json(runtime.publisher.state_path), publisher_state)
            self.assertFalse(runtime.publisher.checkpoint_path(1).exists())
            self.assertFalse(runtime.publisher.model_path(1).exists())
            self.assertFalse(runtime.publisher.manifest_path(1).exists())
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ROLLED_BACK")
            self.assertEqual(receipt["restored_model_version"], 0)
            runtime._nack.assert_called_once_with(
                "delivery-transaction-1", "learner update rolled back"
            )
            self.assertTrue(renewer.closed)
            self.assertEqual(
                list(runtime.publisher.checkpoint_dir.glob(".*.rollback.pt")),
                [],
            )

    def test_short_update_rechecks_lease_synchronously_before_register(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            before = self.trainer_snapshot(runtime.trainer)
            publisher_state = read_json(runtime.publisher.state_path)
            runtime._register = mock.Mock()
            renewer = FakeLeaseRenewer(
                renew_error="injected pre-register authority loss"
            )

            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "health check before model registration failed",
                ):
                    runtime._train_delivery(self.response())

            self.assertEqual(renewer.renew_count, 1)
            runtime._register.assert_not_called()
            self.assert_nested_equal(
                before, self.trainer_snapshot(runtime.trainer)
            )
            self.assertEqual(read_json(runtime.publisher.state_path), publisher_state)
            self.assertFalse(runtime.publisher.checkpoint_path(1).exists())
            self.assertFalse(runtime.publisher.model_path(1).exists())
            self.assertFalse(runtime.publisher.manifest_path(1).exists())
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ROLLED_BACK")
            self.assertEqual(receipt["failed_state"], "MODEL_COMMITTED")
            runtime._nack.assert_called_once()

    def test_pre_training_value_error_is_reserved_for_invalid_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))

            def invalid_samples(_batches):
                raise ValueError("injected invalid sample")

            runtime._training_samples = invalid_samples
            renewer = FakeLeaseRenewer()
            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                with self.assertRaisesRegex(ValueError, "invalid sample"):
                    runtime._train_delivery(self.response())

            runtime._nack.assert_not_called()
            self.assertEqual(runtime.trainer.model_version, 0)
            self.assertEqual(
                list(runtime.publisher.checkpoint_dir.glob(".*.rollback.pt")),
                [],
            )

    def test_initial_register_response_loss_reconciles_exact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bootstrap = self.runtime(Path(directory))
            runtime._startup_mode = "fresh"
            runtime.UPDATE_COMMIT_RETRY_DELAY_SEC = 0.0
            runtime.publisher.publish_runtime = mock.Mock(
                return_value=bootstrap
            )
            runtime.publisher.archive_version = mock.Mock()
            runtime._wait_initial_model_loaded = mock.Mock()
            runtime._commit_learner_metrics = mock.Mock()
            runtime.model_stub.RegisterModel = mock.Mock(
                side_effect=grpc.RpcError(
                    "injected initial register response loss"
                )
            )

            lookup_count = 0

            def exact_lookup(request, timeout):
                nonlocal lookup_count
                lookup_count += 1
                if lookup_count <= 3:
                    return training_pb2.GetModelManifestRsp(
                        ret_code=-1,
                        distributor=model_authority(),
                    )
                response = training_pb2.GetModelManifestRsp(
                    ret_code=0,
                    distributor=model_authority(),
                )
                response.manifest.CopyFrom(
                    manifest_message(
                        TrainingRuntime._manifest_for_wire(bootstrap)
                    )
                )
                return response

            runtime.model_stub.GetModelManifest = mock.Mock(
                side_effect=exact_lookup
            )

            runtime._initialize_models()

            self.assertEqual(runtime.model_stub.RegisterModel.call_count, 3)
            self.assertEqual(
                runtime.model_stub.GetModelManifest.call_count, 4
            )
            runtime._wait_initial_model_loaded.assert_called_once_with(
                bootstrap
            )
            runtime.publisher.archive_version.assert_called_once_with(
                0, "fresh"
            )
            self.assertEqual(runtime.initial_model_version, 0)
            self.assertEqual(runtime.model_manifests[0], bootstrap)

    def test_model_authority_pin_requires_ready_exact_contract_status(self):
        for case in ("not_ready", "wrong_contract"):
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    status = model_status(runtime.contract)
                    if case == "not_ready":
                        status.ready = False
                    else:
                        status.contract.package_version = "wrong-contract"
                    runtime.model_stub.GetModelDistributorStatus = mock.Mock(
                        return_value=status
                    )

                    with self.assertRaisesRegex(RuntimeError, "exact contract"):
                        runtime._pin_model_distributor_authority()

    def test_initial_loaded_status_requires_exact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, bootstrap = self.runtime(Path(directory))
            status = model_status(runtime.contract)
            status.contract.package_version = "wrong-contract"
            status.latest_ack_status = training_pb2.MODEL_LOAD_STATUS_LOADED
            status.latest_ack_model.CopyFrom(
                manifest_message(
                    TrainingRuntime._manifest_for_wire(bootstrap)
                ).identity
            )
            runtime.model_stub.GetModelDistributorStatus = mock.Mock(
                return_value=status
            )
            runtime.model_startup_timeout = 0.5

            with mock.patch(
                "main.training_runtime.time.monotonic",
                side_effect=(0.0, 0.0, 1.0),
            ), mock.patch("main.training_runtime.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "did not ACK"):
                    runtime._wait_initial_model_loaded(bootstrap)

            runtime.model_stub.GetModelDistributorStatus.assert_called_once()

    def test_register_response_loss_and_exact_absence_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            before = self.trainer_snapshot(runtime.trainer)
            publisher_state = read_json(runtime.publisher.state_path)
            register_calls = []

            def reject_register(manifest):
                register_calls.append(copy.deepcopy(manifest["identity"]))
                raise grpc.RpcError("injected register response loss")

            runtime._register = reject_register
            runtime.model_stub.GetModelManifest = (
                lambda request, timeout: SimpleNamespace(
                    ret_code=-1,
                    distributor=model_authority(),
                )
            )
            runtime._ack = mock.Mock()
            renewer = FakeLeaseRenewer()
            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "registration was not applied"
                ):
                    runtime._train_delivery(self.response())

            self.assertEqual(len(register_calls), 3)
            self.assertEqual(register_calls[0], register_calls[1])
            self.assertEqual(register_calls[1], register_calls[2])
            self.assert_nested_equal(
                before, self.trainer_snapshot(runtime.trainer)
            )
            self.assertEqual(read_json(runtime.publisher.state_path), publisher_state)
            self.assertFalse(runtime.publisher.checkpoint_path(1).exists())
            self.assertFalse(runtime.publisher.model_path(1).exists())
            self.assertFalse(runtime.publisher.manifest_path(1).exists())
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ROLLED_BACK")
            self.assertEqual(receipt["failed_state"], "MODEL_COMMITTED")
            self.assertEqual(receipt["register_attempts"], 3)
            self.assertEqual(
                receipt["model_distributor_authority"],
                TrainingRuntime._authority_document(model_authority()),
            )
            runtime._ack.assert_not_called()
            runtime._nack.assert_called_once()

    def test_changed_authority_positive_evidence_applies_but_absence_is_uncertain(
        self,
    ):
        cases = (
            ("response_loss_not_found", "REGISTER_PENDING", 3),
            ("response_loss_manifest_present", "ACKED", 3),
            ("direct_success", "ACKED", 1),
        )
        for case, expected_state, expected_attempts in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    pinned = model_authority()
                    changed = model_authority(
                        "model-distributor-restarted", 2
                    )
                    runtime.model_stub.GetModelDistributorStatus = mock.Mock(
                        return_value=model_status(runtime.contract, pinned)
                    )
                    register_documents = []

                    def register(document):
                        register_documents.append(copy.deepcopy(document))
                        raise grpc.RpcError(
                            "injected register response loss"
                        )

                    def register_rpc(request, timeout):
                        response = training_pb2.RegisterModelRsp(
                            result=(
                                training_pb2.MODEL_REGISTER_RESULT_REGISTERED
                            ),
                            distributor=changed,
                        )
                        response.manifest.CopyFrom(request.manifest)
                        return response

                    def exact_lookup(request, timeout):
                        if case == "response_loss_not_found":
                            return SimpleNamespace(
                                ret_code=-1,
                                distributor=changed,
                            )
                        expected = manifest_message(
                            TrainingRuntime._manifest_for_wire(
                                register_documents[0]
                            )
                        )
                        return SimpleNamespace(
                            ret_code=0,
                            manifest=expected,
                            distributor=changed,
                        )

                    if case == "direct_success":
                        runtime.model_stub.RegisterModel = mock.Mock(
                            side_effect=register_rpc
                        )
                        runtime._register = mock.Mock(
                            wraps=runtime._register
                        )
                    else:
                        runtime._register = mock.Mock(side_effect=register)
                    runtime.model_stub.GetModelManifest = mock.Mock(
                        side_effect=exact_lookup
                    )
                    runtime._ack = mock.Mock()
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        if expected_state == "REGISTER_PENDING":
                            with self.assertRaisesRegex(
                                RuntimeError, "authority changed"
                            ):
                                runtime._train_delivery(self.response())
                        else:
                            runtime._train_delivery(self.response())

                    self.assertEqual(
                        runtime._register.call_count, expected_attempts
                    )
                    runtime.model_stub.GetModelDistributorStatus.assert_called_once()
                    if case == "direct_success":
                        runtime.model_stub.RegisterModel.assert_called_once()
                        runtime.model_stub.GetModelManifest.assert_not_called()
                    else:
                        runtime.model_stub.GetModelManifest.assert_called_once()
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], expected_state)
                    self.assertEqual(
                        receipt["register_attempts"], expected_attempts
                    )
                    self.assertEqual(
                        receipt["model_distributor_authority"],
                        TrainingRuntime._authority_document(pinned),
                    )
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertIsNotNone(
                        runtime.publisher.complete_manifest(1)
                    )
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    if expected_state == "REGISTER_PENDING":
                        self.assertTrue(
                            Path(receipt["recovery_checkpoint"]).is_file()
                        )
                        self.assertNotIn(
                            "model_distributor_resolved_authority", receipt
                        )
                        runtime._ack.assert_not_called()
                    else:
                        self.assertEqual(
                            receipt[
                                "model_distributor_resolved_authority"
                            ],
                            TrainingRuntime._authority_document(changed),
                        )
                        runtime._ack.assert_called_once()
                        self.assertEqual(
                            list(
                                runtime.publisher.checkpoint_dir.glob(
                                    ".*.rollback.pt"
                                )
                            ),
                            [],
                        )
                    runtime._nack.assert_not_called()

    def test_contradictory_register_response_keeps_outcome_pending(self):
        cases = (
            (
                "error_code_with_positive_result",
                -1,
                training_pb2.MODEL_REGISTER_RESULT_REGISTERED,
            ),
            (
                "success_code_with_rejected_result",
                0,
                training_pb2.MODEL_REGISTER_RESULT_REJECTED_CONFLICT,
            ),
        )
        for case, ret_code, result in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    before = self.trainer_snapshot(runtime.trainer)

                    def contradictory_register(request, timeout):
                        response = training_pb2.RegisterModelRsp(
                            ret_code=ret_code,
                            result=result,
                            message="injected contradictory register response",
                            distributor=model_authority(),
                        )
                        response.manifest.CopyFrom(request.manifest)
                        return response

                    runtime.model_stub.RegisterModel = mock.Mock(
                        side_effect=contradictory_register
                    )
                    runtime.model_stub.GetModelManifest = mock.Mock(
                        side_effect=grpc.RpcError(
                            "injected exact lookup failure"
                        )
                    )
                    runtime._ack = mock.Mock()
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "registration outcome is uncertain",
                        ):
                            runtime._train_delivery(self.response())

                    runtime.model_stub.RegisterModel.assert_called_once()
                    runtime.model_stub.GetModelManifest.assert_called_once()
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertEqual(runtime.trained_samples, 2)
                    self.assertIsNotNone(runtime.publisher.complete_manifest(1))
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    self.assertEqual(
                        read_json(runtime.publisher.state_path)[
                            "latest_model"
                        ]["model_version"],
                        1,
                    )

                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], "REGISTER_PENDING")
                    self.assertEqual(receipt["failed_state"], "MODEL_COMMITTED")
                    self.assertEqual(receipt["register_attempts"], 1)
                    self.assertIn(
                        "model registration response is contradictory",
                        receipt["last_error"],
                    )
                    self.assertNotIn(
                        "model_distributor_resolved_authority", receipt
                    )
                    rollback_path = Path(receipt["recovery_checkpoint"])
                    self.assertTrue(rollback_path.is_file())
                    self.assert_checkpoint_matches_snapshot(
                        runtime, rollback_path, before
                    )
                    runtime._ack.assert_not_called()
                    runtime._nack.assert_not_called()

    def test_register_retry_stops_on_lease_loss_and_reconciles_exact_state(self):
        cases = (
            (RuntimeError("injected register rejection"), "absent", "ROLLED_BACK"),
            (grpc.RpcError("injected register response loss"), "absent", "ROLLED_BACK"),
            (grpc.RpcError("injected register response loss"), "present", "ACKED"),
            (
                grpc.RpcError("injected register response loss"),
                "unknown",
                "REGISTER_PENDING",
            ),
        )
        for first_error, exact_state, expected_receipt_state in cases:
            with self.subTest(
                exact_state=exact_state,
                first_error=type(first_error).__name__,
            ):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    before = self.trainer_snapshot(runtime.trainer)
                    publisher_state = read_json(runtime.publisher.state_path)
                    register_documents = []
                    renewer = FakeLeaseRenewer()

                    def fail_first_register(document):
                        register_documents.append(copy.deepcopy(document))
                        renewer.error = "injected lease loss during register retry"
                        raise first_error

                    def get_exact_manifest(request, timeout):
                        if exact_state == "unknown":
                            raise grpc.RpcError("injected exact lookup failure")
                        if exact_state == "absent":
                            return SimpleNamespace(
                                ret_code=-1,
                                distributor=model_authority(),
                            )
                        expected = manifest_message(
                            TrainingRuntime._manifest_for_wire(
                                register_documents[0]
                            )
                        )
                        return SimpleNamespace(
                            ret_code=0,
                            manifest=expected,
                            distributor=model_authority(),
                        )

                    runtime._register = mock.Mock(
                        side_effect=fail_first_register
                    )
                    runtime.model_stub.GetModelManifest = mock.Mock(
                        side_effect=get_exact_manifest
                    )
                    runtime._ack = mock.Mock()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        if exact_state == "present":
                            runtime._train_delivery(self.response())
                        else:
                            pattern = (
                                "registration was not applied"
                                if exact_state == "absent"
                                else "registration outcome is uncertain"
                            )
                            with self.assertRaisesRegex(RuntimeError, pattern):
                                runtime._train_delivery(self.response())

                    runtime._register.assert_called_once()
                    runtime.model_stub.GetModelManifest.assert_called_once()
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(
                        receipt["state"], expected_receipt_state
                    )
                    self.assertEqual(receipt["register_attempts"], 1)
                    if exact_state == "absent":
                        self.assert_nested_equal(
                            before, self.trainer_snapshot(runtime.trainer)
                        )
                        self.assertEqual(
                            read_json(runtime.publisher.state_path),
                            publisher_state,
                        )
                        self.assertFalse(
                            runtime.publisher.model_path(1).exists()
                        )
                        runtime._ack.assert_not_called()
                        runtime._nack.assert_called_once()
                    else:
                        self.assertEqual(runtime.trainer.model_version, 1)
                        self.assertIsNotNone(
                            runtime.publisher.complete_manifest(1)
                        )
                        runtime._nack.assert_not_called()
                        if exact_state == "present":
                            runtime._ack.assert_called_once()
                        else:
                            runtime._ack.assert_not_called()
                            self.assertTrue(
                                Path(receipt["recovery_checkpoint"]).is_file()
                            )

    def test_register_response_loss_reconciles_exact_registered_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            register_documents = []
            exact_requests = []

            def lose_register_response(document):
                register_documents.append(copy.deepcopy(document))
                raise grpc.RpcError("injected register response loss")

            def get_exact_manifest(request, timeout):
                exact_requests.append(copy.deepcopy(request))
                expected = manifest_message(
                    TrainingRuntime._manifest_for_wire(
                        register_documents[-1]
                    )
                )
                return SimpleNamespace(
                    ret_code=0,
                    manifest=expected,
                    distributor=model_authority(),
                )

            runtime._register = lose_register_response
            runtime.model_stub.GetModelManifest = get_exact_manifest
            runtime._ack = mock.Mock()
            renewer = FakeLeaseRenewer()
            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                runtime._train_delivery(self.response())

            self.assertEqual(len(register_documents), 3)
            self.assertEqual(register_documents, [register_documents[0]] * 3)
            self.assertEqual(len(exact_requests), 1)
            expected_identity = manifest_message(
                TrainingRuntime._manifest_for_wire(register_documents[0])
            ).identity
            self.assertEqual(
                exact_requests[0].requested_model.SerializeToString(
                    deterministic=True
                ),
                expected_identity.SerializeToString(deterministic=True),
            )
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ACKED")
            self.assertEqual(receipt["register_attempts"], 3)
            self.assertEqual(receipt["ack_attempts"], 1)
            self.assertEqual(
                receipt["model_distributor_authority"],
                TrainingRuntime._authority_document(model_authority()),
            )
            self.assertEqual(runtime.trainer.model_version, 1)
            self.assertEqual(runtime.train_updates, 1)
            self.assertIsNotNone(runtime.publisher.complete_manifest(1))
            self.assert_trainer_matches_checkpoint(runtime, 1)
            runtime._ack.assert_called_once_with(
                "delivery-transaction-1",
                training_pb2.ACK_DISPOSITION_TRAINED,
                "train-update-00000001",
                sample_authority(),
            )
            runtime._nack.assert_not_called()

    def test_negative_exact_lookup_with_manifest_keeps_outcome_pending(self):
        for manifest_kind in ("exact", "conflicting"):
            with self.subTest(manifest_kind=manifest_kind):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    before = self.trainer_snapshot(runtime.trainer)
                    register_documents = []

                    def lose_register_response(document):
                        register_documents.append(copy.deepcopy(document))
                        raise grpc.RpcError(
                            "injected register response loss"
                        )

                    def negative_lookup_with_manifest(request, timeout):
                        manifest = manifest_message(
                            TrainingRuntime._manifest_for_wire(
                                register_documents[-1]
                            )
                        )
                        if manifest_kind == "conflicting":
                            manifest.model_file = "conflicting-model.onnx"
                        response = training_pb2.GetModelManifestRsp(
                            ret_code=-1,
                            message="injected contradictory exact lookup",
                            distributor=model_authority(),
                        )
                        response.manifest.CopyFrom(manifest)
                        return response

                    runtime._register = mock.Mock(
                        side_effect=lose_register_response
                    )
                    runtime.model_stub.GetModelManifest = mock.Mock(
                        side_effect=negative_lookup_with_manifest
                    )
                    runtime._ack = mock.Mock()
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "manifest with a negative ret_code",
                        ):
                            runtime._train_delivery(self.response())

                    self.assertEqual(runtime._register.call_count, 3)
                    runtime.model_stub.GetModelManifest.assert_called_once()
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertEqual(runtime.trained_samples, 2)
                    self.assertIsNotNone(runtime.publisher.complete_manifest(1))
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    self.assertEqual(
                        read_json(runtime.publisher.state_path)[
                            "latest_model"
                        ]["model_version"],
                        1,
                    )

                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], "REGISTER_PENDING")
                    self.assertEqual(receipt["failed_state"], "MODEL_COMMITTED")
                    self.assertEqual(receipt["register_attempts"], 3)
                    self.assertIn(
                        "manifest with a negative ret_code",
                        receipt["last_error"],
                    )
                    self.assertNotIn(
                        "model_distributor_resolved_authority", receipt
                    )
                    rollback_path = Path(receipt["recovery_checkpoint"])
                    self.assertTrue(rollback_path.is_file())
                    self.assert_checkpoint_matches_snapshot(
                        runtime, rollback_path, before
                    )
                    runtime._ack.assert_not_called()
                    runtime._nack.assert_not_called()

    def test_register_outcome_uncertain_retains_forward_and_rollback_states(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            before = self.trainer_snapshot(runtime.trainer)
            runtime._register = mock.Mock(
                side_effect=grpc.RpcError(
                    "injected persistent register response loss"
                )
            )
            exact_lookup = mock.Mock(
                side_effect=grpc.RpcError("injected exact lookup failure")
            )
            runtime.model_stub.GetModelManifest = exact_lookup
            runtime._ack = mock.Mock()
            renewer = FakeLeaseRenewer()
            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "registration outcome is uncertain"
                ):
                    runtime._train_delivery(self.response())

            self.assertEqual(runtime._register.call_count, 3)
            exact_lookup.assert_called_once()
            self.assertEqual(runtime.trainer.model_version, 1)
            self.assertEqual(runtime.train_updates, 1)
            self.assertIsNotNone(runtime.publisher.complete_manifest(1))
            self.assert_trainer_matches_checkpoint(runtime, 1)
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "REGISTER_PENDING")
            self.assertEqual(receipt["failed_state"], "MODEL_COMMITTED")
            self.assertEqual(receipt["register_attempts"], 3)
            rollback_path = Path(receipt["recovery_checkpoint"])
            self.assertTrue(rollback_path.is_file())
            self.assert_checkpoint_matches_snapshot(
                runtime, rollback_path, before
            )
            runtime._ack.assert_not_called()
            runtime._nack.assert_not_called()

    def test_checkpoint_and_export_failures_restore_pre_update_state(self):
        for failed_step in ("checkpoint", "export"):
            with self.subTest(failed_step=failed_step):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    before = self.trainer_snapshot(runtime.trainer)
                    publisher_state = read_json(runtime.publisher.state_path)
                    if failed_step == "checkpoint":
                        original = runtime.publisher.commit_optimizer_checkpoint

                        def fail_checkpoint(*args, **kwargs):
                            original(*args, **kwargs)
                            raise ValueError("injected checkpoint failure")

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
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, f"injected {failed_step} failure"
                        ):
                            runtime._train_delivery(self.response())

                    self.assert_nested_equal(
                        before, self.trainer_snapshot(runtime.trainer)
                    )
                    self.assertEqual(
                        read_json(runtime.publisher.state_path), publisher_state
                    )
                    self.assertFalse(runtime.publisher.checkpoint_path(1).exists())
                    self.assertFalse(runtime.publisher.model_path(1).exists())
                    self.assertFalse(runtime.publisher.manifest_path(1).exists())
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], "ROLLED_BACK")
                    runtime._register.assert_not_called()
                    runtime._ack.assert_not_called()
                    runtime._nack.assert_called_once()

    def test_register_and_ack_response_loss_retry_the_same_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, _ = self.runtime(Path(directory))
            register_calls = []
            ack_calls = []

            def register(manifest):
                register_calls.append(copy.deepcopy(manifest["identity"]))
                if len(register_calls) == 1:
                    raise RuntimeError("injected register response loss")
                return model_authority()

            def ack(
                delivery_id,
                disposition,
                train_update_id,
                expected_authority,
            ):
                ack_calls.append((delivery_id, disposition, train_update_id))
                if len(ack_calls) == 1:
                    raise RuntimeError("injected ACK response loss")

            runtime._register = register
            runtime._ack = ack
            renewer = FakeLeaseRenewer()
            with mock.patch(
                "main.training_runtime.LeaseRenewer", return_value=renewer
            ):
                runtime._train_delivery(self.response())

            self.assertEqual(register_calls, [register_calls[0]] * 2)
            self.assertEqual(
                ack_calls,
                [
                    (
                        "delivery-transaction-1",
                        training_pb2.ACK_DISPOSITION_TRAINED,
                        "train-update-00000001",
                    )
                ]
                * 2,
            )
            receipt = read_json(
                runtime.publisher.receipt_path("train-update-00000001")
            )
            self.assertEqual(receipt["state"], "ACKED")
            self.assertEqual(receipt["register_attempts"], 2)
            self.assertEqual(receipt["ack_attempts"], 2)
            self.assertEqual(
                receipt["model_distributor_authority"],
                TrainingRuntime._authority_document(model_authority()),
            )
            self.assertEqual(runtime.trainer.model_version, 1)
            self.assertEqual(runtime.train_updates, 1)
            self.assertIsNotNone(runtime.publisher.complete_manifest(1))
            self.assert_trainer_matches_checkpoint(runtime, 1)
            runtime._nack.assert_not_called()
            self.assertEqual(
                list(runtime.publisher.checkpoint_dir.glob(".*.rollback.pt")),
                [],
            )

    def test_contradictory_ack_response_keeps_registered_model_forward(self):
        cases = (
            (
                "error_code_with_applied_result",
                -1,
                training_pb2.DELIVERY_RESULT_APPLIED,
            ),
            (
                "success_code_with_rejected_result",
                0,
                training_pb2.DELIVERY_RESULT_REJECTED,
            ),
        )
        for case, ret_code, result in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    runtime._register = mock.Mock(
                        return_value=model_authority()
                    )
                    runtime.sample_stub.AckBatch = mock.Mock(
                        return_value=training_pb2.DeliveryRsp(
                            ret_code=ret_code,
                            result=result,
                            message="injected contradictory ACK response",
                        )
                    )
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "sample ACK outcome is uncertain",
                        ):
                            runtime._train_delivery(self.response())

                    runtime._register.assert_called_once()
                    self.assertEqual(
                        runtime.sample_stub.AckBatch.call_count, 3
                    )
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertEqual(runtime.trained_samples, 2)
                    self.assertIsNotNone(runtime.publisher.complete_manifest(1))
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    self.assertEqual(
                        read_json(runtime.publisher.state_path)[
                            "latest_model"
                        ]["model_version"],
                        1,
                    )

                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], "ACK_PENDING")
                    self.assertEqual(receipt["failed_state"], "REGISTERED")
                    self.assertEqual(receipt["ack_attempts"], 3)
                    self.assertIn(
                        "sample ACK response is contradictory",
                        receipt["last_error"],
                    )
                    self.assertEqual(
                        list(
                            runtime.publisher.checkpoint_dir.glob(
                                ".*.rollback.pt"
                            )
                        ),
                        [],
                    )
                    runtime._nack.assert_not_called()

    def test_positive_ack_response_requires_exact_request_echo(self):
        for mismatched_field in (
            "delivery_id",
            "disposition",
            "train_update_id",
        ):
            with self.subTest(mismatched_field=mismatched_field):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    runtime._register = mock.Mock(
                        return_value=model_authority()
                    )
                    response = training_pb2.DeliveryRsp(
                        ret_code=0,
                        result=training_pb2.DELIVERY_RESULT_APPLIED,
                        delivery_id="delivery-transaction-1",
                        disposition=training_pb2.ACK_DISPOSITION_TRAINED,
                        train_update_id="train-update-00000001",
                    )
                    if mismatched_field == "delivery_id":
                        response.delivery_id = "another-delivery"
                    elif mismatched_field == "disposition":
                        response.disposition = (
                            training_pb2.ACK_DISPOSITION_STALE
                        )
                    else:
                        response.train_update_id = "another-update"
                    runtime.sample_stub.AckBatch = mock.Mock(
                        return_value=response
                    )
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "sample ACK outcome is uncertain",
                        ):
                            runtime._train_delivery(self.response())

                    self.assertEqual(
                        runtime.sample_stub.AckBatch.call_count, 3
                    )
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertEqual(runtime.trained_samples, 2)
                    self.assertIsNotNone(runtime.publisher.complete_manifest(1))
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], "ACK_PENDING")
                    self.assertEqual(receipt["failed_state"], "REGISTERED")
                    self.assertEqual(receipt["ack_attempts"], 3)
                    self.assertIn(
                        "does not echo the exact request",
                        receipt["last_error"],
                    )
                    self.assertEqual(
                        list(
                            runtime.publisher.checkpoint_dir.glob(
                                ".*.rollback.pt"
                            )
                        ),
                        [],
                    )
                    runtime._nack.assert_not_called()

    def test_ack_negative_requires_the_original_ready_lease_authority(self):
        cases = (
            ("same_authority_not_found", "ACK_REJECTED"),
            ("changed_authority_not_found", "ACK_PENDING"),
            ("changed_authority_positive", "ACKED"),
        )
        for case, expected_state in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    runtime._register = mock.Mock(
                        return_value=model_authority()
                    )
                    lease_authority = sample_authority()
                    changed_authority = sample_authority(
                        "sample-distributor-restarted", 2
                    )
                    status_authority = (
                        changed_authority
                        if case != "same_authority_not_found"
                        else lease_authority
                    )
                    runtime.sample_stub.GetStatus = mock.Mock(
                        return_value=training_pb2.DistributorStatusRsp(
                            ready=True,
                            contract=runtime.contract,
                            distributor=status_authority,
                        )
                    )
                    if case == "changed_authority_positive":
                        ack_response = training_pb2.DeliveryRsp(
                            ret_code=0,
                            result=training_pb2.DELIVERY_RESULT_APPLIED,
                            delivery_id="delivery-transaction-1",
                            disposition=(
                                training_pb2.ACK_DISPOSITION_TRAINED
                            ),
                            train_update_id="train-update-00000001",
                        )
                    else:
                        ack_response = training_pb2.DeliveryRsp(
                            ret_code=1,
                            result=training_pb2.DELIVERY_RESULT_NOT_FOUND,
                            delivery_id="delivery-transaction-1",
                            message="delivery not found",
                        )
                    runtime.sample_stub.AckBatch = mock.Mock(
                        return_value=ack_response
                    )
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        if expected_state == "ACKED":
                            runtime._train_delivery(self.response())
                        else:
                            pattern = (
                                "sample ACK outcome is uncertain"
                                if expected_state == "ACK_PENDING"
                                else "sample ACK was not applied"
                            )
                            with self.assertRaisesRegex(RuntimeError, pattern):
                                runtime._train_delivery(self.response())

                    expected_attempts = 1 if expected_state == "ACKED" else 3
                    self.assertEqual(
                        runtime.sample_stub.AckBatch.call_count,
                        expected_attempts,
                    )
                    if expected_state == "ACKED":
                        runtime.sample_stub.GetStatus.assert_not_called()
                    else:
                        self.assertEqual(
                            runtime.sample_stub.GetStatus.call_count, 3
                        )
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], expected_state)
                    self.assertEqual(
                        receipt["sample_distributor_authority"],
                        TrainingRuntime._authority_document(lease_authority),
                    )
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    runtime._nack.assert_not_called()

    def test_ack_exhaustion_keeps_registered_model_forward(self):
        cases = (
            (
                RuntimeError("injected persistent ACK rejection"),
                "sample ACK was not applied",
                "ACK_REJECTED",
            ),
            (
                grpc.RpcError("injected persistent ACK response loss"),
                "sample ACK outcome is uncertain",
                "ACK_PENDING",
            ),
        )
        for injected_error, error_pattern, expected_state in cases:
            with self.subTest(expected_state=expected_state):
                with tempfile.TemporaryDirectory() as directory:
                    runtime, _ = self.runtime(Path(directory))
                    runtime._register = mock.Mock(
                        return_value=model_authority()
                    )
                    runtime._ack = mock.Mock(side_effect=injected_error)
                    renewer = FakeLeaseRenewer()
                    with mock.patch(
                        "main.training_runtime.LeaseRenewer",
                        return_value=renewer,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, error_pattern
                        ):
                            runtime._train_delivery(self.response())

                    self.assertEqual(runtime._ack.call_count, 3)
                    self.assertEqual(runtime.trainer.model_version, 1)
                    self.assertEqual(runtime.train_updates, 1)
                    self.assertIsNotNone(runtime.publisher.complete_manifest(1))
                    self.assert_trainer_matches_checkpoint(runtime, 1)
                    self.assertEqual(
                        read_json(runtime.publisher.state_path)[
                            "latest_model"
                        ]["model_version"],
                        1,
                    )
                    receipt = read_json(
                        runtime.publisher.receipt_path(
                            "train-update-00000001"
                        )
                    )
                    self.assertEqual(receipt["state"], expected_state)
                    self.assertEqual(receipt["failed_state"], "REGISTERED")
                    self.assertEqual(receipt["ack_attempts"], 3)
                    runtime._nack.assert_not_called()
                    self.assertEqual(
                        list(
                            runtime.publisher.checkpoint_dir.glob(
                                ".*.rollback.pt"
                            )
                        ),
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
