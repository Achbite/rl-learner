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

from main.training_runtime import (
    ModelPublisher,
    TrainingRuntime,
    read_json,
)
from proto import training_pb2
from src.contracts.identity import (
    canonical_config_digest,
    contract_identity,
    finalize_manifest_digest,
    manifest_message,
    service_identity,
    training_config_document,
)
from src.training.ppo_trainer import PPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def config(root: Path, archive_interval: int = 2) -> dict:
    document = yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    document["model"]["local_train_dir"] = str(root / "local-train")
    document["model"]["archive_interval_updates"] = archive_interval
    document["training"]["n_epochs"] = 1
    document["training"]["mini_batch_size"] = 2
    document["identity"]["training_config_digest"] = (
        canonical_config_digest(training_config_document(document))
    )
    return document


def samples(behavior_version: int) -> list[dict]:
    return [
        {
            "observation": [0.0] * 17,
            "action": 0,
            "old_log_probability": -2.0,
            "old_value_prediction": 0.1,
            "advantage": 0.5,
            "td_return": 0.6,
            "behavior_model_version": behavior_version,
        },
        {
            "observation": [0.1] * 17,
            "action": 1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "advantage": -0.25,
            "td_return": -0.05,
            "behavior_model_version": behavior_version,
        },
    ]


def model_authority():
    return service_identity(
        "model-distributor", "model-distributor-resume-test", 1
    )


def previous_model_authority():
    return service_identity(
        "model-distributor", "model-distributor-resume-test-previous", 1
    )


def sample_authority():
    return service_identity(
        "sample-distributor", "sample-distributor-resume-test", 1
    )


class FakeLeaseRenewer:
    def __init__(self):
        self.error = ""
        self.closed = False
        self.renew_count = 0

    def start(self):
        return self

    def renew_now(self):
        self.renew_count += 1

    def close(self):
        self.closed = True


class FakeModelDistributor:
    def __init__(self, runtime, manifests=()):
        self.runtime = runtime
        self.authority = model_authority()
        self.status_authority = self.authority
        self.registry = {
            int(manifest.identity.model_version): copy.deepcopy(manifest)
            for manifest in manifests
        }
        self.register_requests = []
        self.lookup_requests = []
        self.ack_version = max(self.registry, default=None)

    def GetModelDistributorStatus(self, request, timeout):
        response = training_pb2.ModelDistributorStatusRsp(
            contract=self.runtime.contract,
            distributor=self.status_authority,
            ready=True,
            registered_model_count=len(self.registry),
        )
        if self.registry:
            latest = self.registry[max(self.registry)]
            response.latest_model.CopyFrom(latest.identity)
        if self.ack_version is not None:
            response.latest_ack_status = (
                training_pb2.MODEL_LOAD_STATUS_LOADED
            )
            response.latest_ack_model.CopyFrom(
                self.registry[self.ack_version].identity
            )
        return response

    def GetModelManifest(self, request, timeout):
        self.lookup_requests.append(copy.deepcopy(request))
        if request.latest_in_lineage:
            version = max(self.registry, default=None)
        else:
            version = int(request.requested_model.model_version)
            if version not in self.registry:
                version = None
        if version is None:
            return training_pb2.GetModelManifestRsp(
                ret_code=-1,
                message="model not found",
                distributor=self.authority,
            )
        response = training_pb2.GetModelManifestRsp(
            ret_code=0,
            message="model found",
            distributor=self.authority,
        )
        response.manifest.CopyFrom(self.registry[version])
        return response

    def RegisterModel(self, request, timeout):
        manifest = copy.deepcopy(request.manifest)
        self.register_requests.append(manifest)
        version = int(manifest.identity.model_version)
        existing = self.registry.get(version)
        if existing is not None and existing != manifest:
            return training_pb2.RegisterModelRsp(
                ret_code=-1,
                result=(
                    training_pb2.MODEL_REGISTER_RESULT_REJECTED_CONFLICT
                ),
                message="model slot conflict",
                distributor=self.authority,
            )
        result = (
            training_pb2.MODEL_REGISTER_RESULT_ALREADY_REGISTERED
            if existing is not None
            else training_pb2.MODEL_REGISTER_RESULT_REGISTERED
        )
        self.registry[version] = manifest
        self.ack_version = version
        response = training_pb2.RegisterModelRsp(
            ret_code=0,
            result=result,
            message="registered",
            distributor=self.authority,
        )
        response.manifest.CopyFrom(manifest)
        return response


class CheckpointResumePublicationTest(unittest.TestCase):
    def assert_nested_equal(self, left, right):
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

    @staticmethod
    def trainer_snapshot(trainer):
        return {
            "model": copy.deepcopy(trainer.model.state_dict()),
            "optimizer": copy.deepcopy(trainer._optimizer.state_dict()),
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": copy.deepcopy(np.random.get_state()),
            "cuda_rng": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            "training": trainer.model.training,
        }

    @staticmethod
    def close_runtime(runtime):
        runtime.metrics_backend.close()
        runtime.actor_channel.close()
        runtime.model_channel.close()
        runtime.sample_channel.close()

    def make_source(self, root: Path, updates: int = 1):
        cfg = config(root / "source")
        trainer = PPOTrainer(cfg)
        publisher = ModelPublisher(cfg)
        publisher.prepare()
        manifest = publisher.publish_runtime(
            trainer,
            train_update_id="bootstrap-v0",
            behavior_model=None,
            batch_ids=[],
            train_updates=0,
            trained_samples=0,
        )
        for update in range(1, updates + 1):
            stats = trainer.train_on_batch(
                samples(trainer.model_version),
                behavior_model_version=trainer.model_version,
            )
            update_id = f"source-update-{update}"
            publisher.commit_optimizer_checkpoint(
                trainer,
                train_update_id=update_id,
                behavior_model=manifest["identity"],
                batch_ids=[f"source-batch-{update}"],
                stats=stats,
                sample_count=2,
                train_updates=update,
                trained_samples=update * 2,
            )
            manifest = publisher.publish_runtime(
                trainer,
                train_update_id=update_id,
                behavior_model=manifest["identity"],
                batch_ids=[f"source-batch-{update}"],
                stats=stats,
                sample_count=2,
                train_updates=update,
                trained_samples=update * 2,
                checkpoint_precommitted=True,
            )
        return cfg, trainer, publisher, manifest

    @staticmethod
    def reversioned_manifest(document, version):
        candidate = copy.deepcopy(
            TrainingRuntime._manifest_for_wire(document)
        )
        candidate["identity"]["model_version"] = version
        candidate["identity"]["manifest_digest"] = "0" * 64
        return manifest_message(finalize_manifest_digest(candidate))

    @staticmethod
    def artifact_conflicting_manifest(document):
        candidate = copy.deepcopy(
            TrainingRuntime._manifest_for_wire(document)
        )
        candidate["identity"]["artifact_digest"] = "f" * 64
        candidate["identity"]["manifest_digest"] = "0" * 64
        return manifest_message(finalize_manifest_digest(candidate))

    def make_runtime(self, root: Path, checkpoint: Path):
        runtime = TrainingRuntime(
            config(root / "resumed"), str(checkpoint)
        )
        runtime.UPDATE_COMMIT_RETRY_DELAY_SEC = 0.0
        runtime.model_startup_timeout = 0.5
        return runtime

    def test_publication_reserve_changes_only_version_and_is_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, publisher, _ = self.make_source(Path(directory))
            checkpoint = publisher.checkpoint_path(1)
            trainer = PPOTrainer(cfg)
            self.assertTrue(trainer.load_checkpoint(str(checkpoint)))
            before = self.trainer_snapshot(trainer)

            self.assertEqual(
                trainer.reserve_initial_checkpoint_publication_version(1),
                2,
            )

            self.assertEqual(trainer.model_version, 2)
            self.assert_nested_equal(before, self.trainer_snapshot(trainer))
            with self.assertRaisesRegex(RuntimeError, "already reserved"):
                trainer.reserve_initial_checkpoint_publication_version(1)

            reloaded = PPOTrainer(cfg)
            self.assertTrue(reloaded.load_checkpoint(str(checkpoint)))
            reloaded.reserve_initial_checkpoint_publication_version(1)
            self.assertTrue(reloaded.load_checkpoint(str(checkpoint)))
            with self.assertRaisesRegex(RuntimeError, "already reserved"):
                reloaded.reserve_initial_checkpoint_publication_version(1)

            mismatched = PPOTrainer(cfg)
            self.assertTrue(mismatched.load_checkpoint(str(checkpoint)))
            mismatch_before = self.trainer_snapshot(mismatched)
            with self.assertRaisesRegex(RuntimeError, "source version mismatch"):
                mismatched.reserve_initial_checkpoint_publication_version(0)
            self.assertEqual(mismatched.model_version, 1)
            self.assert_nested_equal(
                mismatch_before, self.trainer_snapshot(mismatched)
            )

            unloaded = PPOTrainer(cfg)
            with self.assertRaisesRegex(RuntimeError, "loaded checkpoint"):
                unloaded.reserve_initial_checkpoint_publication_version(0)

            overflow = PPOTrainer(cfg)
            self.assertTrue(overflow.load_checkpoint(str(checkpoint)))
            overflow._model_version = PPOTrainer.MAX_MODEL_VERSION
            overflow._checkpoint_source_model_version = (
                PPOTrainer.MAX_MODEL_VERSION
            )
            with self.assertRaisesRegex(OverflowError, "uint64"):
                overflow.reserve_initial_checkpoint_publication_version(
                    PPOTrainer.MAX_MODEL_VERSION
                )
            self.assertEqual(
                overflow.model_version, PPOTrainer.MAX_MODEL_VERSION
            )
            self.assertFalse(overflow._publication_version_reserved)

    def test_surviving_source_registers_new_baseline_and_first_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, source_manifest = self.make_source(root)
            runtime = self.make_runtime(
                root, source_publisher.checkpoint_path(1)
            )
            try:
                source_wire = manifest_message(
                    TrainingRuntime._manifest_for_wire(source_manifest)
                )
                distributor = FakeModelDistributor(runtime, [source_wire])
                runtime.model_stub = distributor

                runtime._initialize_models()

                self.assertEqual(runtime.trainer.model_version, 2)
                self.assertEqual(runtime.initial_model_version, 2)
                self.assertEqual(runtime.train_updates, 1)
                self.assertEqual(runtime.trained_samples, 2)
                self.assertEqual(len(distributor.register_requests), 1)
                self.assertEqual(
                    distributor.register_requests[0].identity.model_version, 2
                )
                self.assertEqual(
                    distributor.register_requests[0].identity.artifact_digest,
                    source_wire.identity.artifact_digest,
                )
                baseline = runtime.model_manifests[2]
                self.assertEqual(baseline["train_updates"], 1)
                self.assertEqual(baseline["trained_samples"], 2)
                self.assertEqual(baseline["initial_model_version"], 1)
                self.assertEqual(
                    baseline["train_update_id"],
                    "initial-checkpoint-republish-v2",
                )
                checkpoint = ModelPublisher._load_checkpoint(
                    runtime.publisher.checkpoint_path(2)
                )
                self.assertEqual(checkpoint["model_version"], 2)
                self.assertEqual(
                    checkpoint["metadata"]["initial_model_version"], 1
                )
                metrics = runtime._learner_metrics_snapshot()
                self.assertEqual(metrics["model_version"], 2)
                self.assertEqual(metrics["initial_model_version"], 2)
                self.assertEqual(metrics["train_updates"], 1)
                self.assertEqual(metrics["run_train_updates"], 0)
                self.assertEqual(metrics["run_trained_samples"], 0)
                self.assertEqual(
                    runtime.trainer.model_version
                    - runtime.initial_model_version,
                    metrics["run_train_updates"],
                )
                self.assertTrue(runtime.publisher.archive_path(2).is_dir())

                runtime._validate_delivery = (
                    lambda response, allow_partial=False: baseline["identity"]
                )
                runtime._training_samples = lambda batches: samples(2)
                runtime._record_metrics = lambda: None
                runtime._ack_idempotently = mock.Mock(return_value=1)
                runtime._nack = mock.Mock()
                response = SimpleNamespace(
                    delivery_id="resume-delivery-1",
                    actual_batch_size=2,
                    batches=[SimpleNamespace(batch_id="resume-batch-1")],
                    distributor=sample_authority(),
                )
                renewer = FakeLeaseRenewer()
                with mock.patch(
                    "main.training_runtime.LeaseRenewer",
                    return_value=renewer,
                ):
                    runtime._train_delivery(response)

                self.assertTrue(renewer.closed)
                self.assertEqual(runtime.trainer.model_version, 3)
                self.assertEqual(runtime.train_updates, 2)
                self.assertEqual(runtime.trained_samples, 4)
                self.assertEqual(runtime.initial_model_version, 2)
                metrics = runtime._learner_metrics_snapshot()
                self.assertEqual(metrics["run_train_updates"], 1)
                self.assertEqual(metrics["run_trained_samples"], 2)
                self.assertEqual(
                    runtime.trainer.model_version
                    - runtime.initial_model_version,
                    metrics["run_train_updates"],
                )
                self.assertEqual(
                    runtime.model_manifests[3]["initial_model_version"], 1
                )
                receipt = read_json(
                    runtime.publisher.receipt_path("train-update-00000002")
                )
                self.assertEqual(receipt["state"], "ACKED")
                self.assertEqual(receipt["target_model_version"], 3)
                self.assertFalse(runtime.publisher.archive_path(3).exists())

                response.delivery_id = "resume-delivery-2"
                response.batches = [
                    SimpleNamespace(batch_id="resume-batch-2")
                ]
                second_renewer = FakeLeaseRenewer()
                with mock.patch(
                    "main.training_runtime.LeaseRenewer",
                    return_value=second_renewer,
                ):
                    runtime._train_delivery(response)

                self.assertTrue(second_renewer.closed)
                self.assertEqual(runtime.trainer.model_version, 4)
                self.assertEqual(runtime.train_updates, 3)
                self.assertEqual(runtime.trained_samples, 6)
                self.assertEqual(runtime.initial_model_version, 2)
                metrics = runtime._learner_metrics_snapshot()
                self.assertEqual(metrics["run_train_updates"], 2)
                self.assertEqual(metrics["run_trained_samples"], 4)
                self.assertEqual(
                    runtime.trainer.model_version
                    - runtime.initial_model_version,
                    metrics["run_train_updates"],
                )
                self.assertTrue(runtime.publisher.archive_path(4).is_dir())
            finally:
                self.close_runtime(runtime)

    def test_existing_exact_target_is_adopted_without_register(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, _ = self.make_source(root)
            runtime = self.make_runtime(
                root, source_publisher.checkpoint_path(1)
            )
            try:
                candidate = runtime.publisher.publish_runtime(
                    runtime.trainer,
                    train_update_id="initial-checkpoint-republish-v2",
                    behavior_model=None,
                    batch_ids=[],
                    train_updates=runtime.train_updates,
                    trained_samples=runtime.trained_samples,
                )
                candidate_wire = manifest_message(
                    TrainingRuntime._manifest_for_wire(candidate)
                )
                distributor = FakeModelDistributor(
                    runtime, [candidate_wire]
                )
                runtime.model_stub = distributor
                with mock.patch.object(
                    runtime.publisher,
                    "publish_runtime",
                    return_value=candidate,
                ):
                    runtime._initialize_models()

                self.assertEqual(distributor.register_requests, [])
                self.assertEqual(runtime.model_manifests[2], candidate)
                self.assertEqual(runtime.initial_model_version, 2)
            finally:
                self.close_runtime(runtime)

    def test_changed_authority_positive_target_is_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, _ = self.make_source(root)
            runtime = self.make_runtime(
                root, source_publisher.checkpoint_path(1)
            )
            try:
                candidate = runtime.publisher.publish_runtime(
                    runtime.trainer,
                    train_update_id="initial-checkpoint-republish-v2",
                    behavior_model=None,
                    batch_ids=[],
                    train_updates=runtime.train_updates,
                    trained_samples=runtime.trained_samples,
                )
                distributor = FakeModelDistributor(
                    runtime,
                    [
                        manifest_message(
                            TrainingRuntime._manifest_for_wire(candidate)
                        )
                    ],
                )
                distributor.status_authority = previous_model_authority()
                runtime.model_stub = distributor
                with mock.patch.object(
                    runtime.publisher,
                    "publish_runtime",
                    return_value=candidate,
                ):
                    runtime._initialize_models()

                self.assertEqual(distributor.register_requests, [])
                self.assertEqual(runtime.model_manifests[2], candidate)
            finally:
                self.close_runtime(runtime)

    def test_changed_authority_absence_is_not_a_registration_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, source_manifest = self.make_source(root)
            runtime = self.make_runtime(
                root, source_publisher.checkpoint_path(1)
            )
            try:
                distributor = FakeModelDistributor(
                    runtime,
                    [
                        manifest_message(
                            TrainingRuntime._manifest_for_wire(
                                source_manifest
                            )
                        )
                    ],
                )
                distributor.status_authority = previous_model_authority()
                runtime.model_stub = distributor
                with self.assertRaisesRegex(
                    RuntimeError, "authority changed"
                ):
                    runtime._initialize_models()

                self.assertEqual(distributor.register_requests, [])
                self.assertEqual(runtime.model_manifests, {})
            finally:
                self.close_runtime(runtime)

    def test_replaying_source_conflicts_with_existing_target_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, source_manifest = self.make_source(root)
            source_checkpoint = source_publisher.checkpoint_path(1)
            first = self.make_runtime(root / "first", source_checkpoint)
            try:
                source_wire = manifest_message(
                    TrainingRuntime._manifest_for_wire(source_manifest)
                )
                distributor = FakeModelDistributor(first, [source_wire])
                first.model_stub = distributor
                first._initialize_models()
                self.assertEqual(len(distributor.register_requests), 1)
            finally:
                self.close_runtime(first)

            second = self.make_runtime(root / "second", source_checkpoint)
            try:
                distributor.runtime = second
                second.model_stub = distributor
                with self.assertRaisesRegex(
                    RuntimeError, "target slot conflicts"
                ):
                    second._initialize_models()

                self.assertEqual(second.trainer.model_version, 2)
                self.assertEqual(second.model_manifests, {})
                self.assertEqual(len(distributor.register_requests), 1)
                self.assertTrue(second.publisher.model_path(2).is_file())
            finally:
                self.close_runtime(second)

    def test_conflicting_surviving_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, source_publisher, source_manifest = self.make_source(root)
            runtime = self.make_runtime(
                root, source_publisher.checkpoint_path(1)
            )
            try:
                distributor = FakeModelDistributor(
                    runtime,
                    [self.artifact_conflicting_manifest(source_manifest)],
                )
                runtime.model_stub = distributor
                with self.assertRaisesRegex(
                    RuntimeError, "checkpoint source conflicts"
                ):
                    runtime._initialize_models()

                self.assertEqual(runtime.trainer.model_version, 2)
                self.assertEqual(runtime.model_manifests, {})
                self.assertEqual(distributor.register_requests, [])
                self.assertTrue(runtime.publisher.model_path(2).is_file())
            finally:
                self.close_runtime(runtime)

    def test_newer_or_conflicting_remote_facts_fail_closed(self):
        cases = ("newer", "target_conflict")
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    _, _, source_publisher, source_manifest = (
                        self.make_source(root)
                    )
                    runtime = self.make_runtime(
                        root, source_publisher.checkpoint_path(1)
                    )
                    try:
                        remote_version = 3 if case == "newer" else 2
                        remote = self.reversioned_manifest(
                            source_manifest, remote_version
                        )
                        distributor = FakeModelDistributor(runtime, [remote])
                        runtime.model_stub = distributor
                        publisher_spy = mock.Mock(
                            wraps=runtime.publisher.publish_runtime
                        )
                        with mock.patch.object(
                            runtime.publisher,
                            "publish_runtime",
                            publisher_spy,
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError,
                                (
                                    "newer publication"
                                    if case == "newer"
                                    else "target slot conflicts"
                                ),
                            ):
                                runtime._initialize_models()

                        self.assertEqual(runtime.trainer.model_version, 2)
                        self.assertEqual(distributor.register_requests, [])
                        if case == "newer":
                            publisher_spy.assert_not_called()
                            self.assertFalse(
                                runtime.publisher.model_path(2).exists()
                            )
                        else:
                            publisher_spy.assert_called_once()
                            self.assertTrue(
                                runtime.publisher.model_path(2).is_file()
                            )
                        self.assertEqual(runtime.model_manifests, {})
                    finally:
                        self.close_runtime(runtime)

    def test_archive_cadence_uses_run_updates_not_publication_version(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = ModelPublisher(config(Path(directory)))
            self.assertFalse(publisher.should_archive(0))
            self.assertFalse(publisher.should_archive(1))
            self.assertTrue(publisher.should_archive(2))
            self.assertFalse(publisher.should_archive(3))


if __name__ == "__main__":
    unittest.main()
