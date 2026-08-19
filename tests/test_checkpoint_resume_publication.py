import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from main.training_runtime import (
    ModelPublisher,
    TrainingRuntime,
    atomic_write_json,
)
from proto import training_pb2
from src.contracts.identity import (
    canonical_config_digest,
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
            "behavior_model_step": behavior_version,
        },
        {
            "observation": [0.1] * 17,
            "action": 1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "advantage": -0.25,
            "td_return": -0.05,
            "behavior_model_step": behavior_version,
        },
    ]


def model_authority():
    return service_identity(
        "model-distributor", "model-distributor-resume-test", 1
    )


class FakeModelDistributor:
    def __init__(self, runtime, manifests=()):
        self.runtime = runtime
        self.authority = model_authority()
        self.registry = {
            int(manifest.identity.model_step): copy.deepcopy(manifest)
            for manifest in manifests
        }
        self.register_requests = []
        self.ack_version = max(self.registry, default=None)

    def GetModelDistributorStatus(self, request, timeout):
        response = training_pb2.ModelDistributorStatusRsp(
            contract=self.runtime.contract,
            distributor=self.authority,
            ready=True,
            registered_model_count=len(self.registry),
        )
        if self.registry:
            latest = self.registry[max(self.registry)]
            response.latest_model.CopyFrom(latest.identity)
            response.available_floor_model_step = min(self.registry)
            response.latest_available_model_step = max(self.registry)
        if self.ack_version is not None:
            response.latest_ack_status = training_pb2.MODEL_LOAD_STATUS_LOADED
            response.latest_ack_model.CopyFrom(
                self.registry[self.ack_version].identity
            )
        return response

    def GetModelManifest(self, request, timeout):
        if request.latest_in_lineage:
            version = max(self.registry, default=None)
        else:
            requested = int(request.requested_model.model_step)
            version = requested if requested in self.registry else None
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
        response.available_floor_model_step = min(self.registry)
        response.latest_available_model_step = max(self.registry)
        return response

    def RegisterModel(self, request, timeout):
        manifest = copy.deepcopy(request.manifest)
        self.register_requests.append(manifest)
        version = int(manifest.identity.model_step)
        existing = self.registry.get(version)
        if existing is not None and existing != manifest:
            return training_pb2.RegisterModelRsp(
                ret_code=-1,
                result=training_pb2.MODEL_REGISTER_RESULT_REJECTED_CONFLICT,
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
        self.assertEqual(publisher.prepare(), "fresh")
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
                samples(trainer.model_step),
                behavior_model_step=trainer.model_step,
            )
            update_id = f"train-update-{update:08d}"
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
            atomic_write_json(
                publisher.receipt_path(update_id),
                {
                    "schema_version": 1,
                    "train_update_id": update_id,
                    "state": "ACKED",
                    "model": manifest["identity"],
                    "train_updates": update,
                    "trained_samples": update * 2,
                },
            )
        return cfg, trainer, publisher, manifest

    def test_inherited_publication_loads_only_weights_and_starts_at_v0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source_trainer, source_publisher, _ = (
                self.make_source(root)
            )
            child_cfg = config(root / "child")
            child_cfg["identity"]["model_lineage_id"] = "child-lineage"
            child_cfg["training"]["seed"] = 7
            child_cfg["model"]["bootstrap_seed"] = 7
            child_cfg["identity"]["training_config_digest"] = (
                canonical_config_digest(training_config_document(child_cfg))
            )
            expected_fresh = PPOTrainer(child_cfg)
            expected_snapshot = self.trainer_snapshot(expected_fresh)
            save_directory = root / "models" / "save" / "0000001"
            save_directory.mkdir(parents=True)
            shutil.copy2(
                source_publisher.model_path(1),
                save_directory / "SaveModel.onnx",
            )

            initial_model = save_directory / "SaveModel.onnx"
            child_cfg["model"]["initial_model_path"] = str(initial_model)
            runtime = TrainingRuntime(child_cfg)
            try:
                self.assertEqual(runtime._startup_mode, "inherited-weights")
                self.assertEqual(runtime.trainer.model_step, 0)
                self.assertEqual(runtime.train_updates, 0)
                self.assertEqual(runtime.trained_samples, 0)
                self.assert_nested_equal(
                    source_trainer.model.state_dict(),
                    runtime.trainer.model.state_dict(),
                )
                actual = self.trainer_snapshot(runtime.trainer)
                self.assert_nested_equal(
                    expected_snapshot["optimizer"], actual["optimizer"]
                )
                self.assert_nested_equal(
                    expected_snapshot["torch_rng"], actual["torch_rng"]
                )
                self.assert_nested_equal(
                    expected_snapshot["numpy_rng"], actual["numpy_rng"]
                )
                self.assertEqual(
                    runtime.publisher.initial_model_provenance[
                        "initial_model_path"
                    ],
                    str(initial_model),
                )
                runtime.model_stub = FakeModelDistributor(runtime)
                runtime._wait_initial_model_loaded = mock.Mock()
                runtime._initialize_models()
                self.assertEqual(len(runtime.model_stub.register_requests), 1)
                inherited = runtime.publisher.complete_manifest(0)
                self.assertIsNotNone(inherited)
                self.assertEqual(inherited["identity"]["model_step"], 0)
                self.assertEqual(
                    inherited["identity"]["model_lineage_id"], "child-lineage"
                )
                self.assertEqual(
                    inherited["train_update_id"], "inherited-bootstrap-v0"
                )
                self.assertEqual(
                    inherited["initial_model_path"], str(initial_model)
                )
            finally:
                self.close_runtime(runtime)

    def test_weight_inheritance_is_one_shot_and_requires_a_fresh_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _, publisher, _ = self.make_source(root)
            model = str(publisher.model_path(1))
            trainer = PPOTrainer(cfg)
            self.assertTrue(trainer.load_onnx_weights(model))
            with self.assertRaisesRegex(RuntimeError, "fresh trainer"):
                trainer.load_onnx_weights(model)

            trained = PPOTrainer(cfg)
            trained.train_on_batch(samples(0), behavior_model_step=0)
            with self.assertRaisesRegex(RuntimeError, "fresh trainer"):
                trained.load_onnx_weights(model)

    def test_previous_workspace_is_never_auto_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _, _, _ = self.make_source(root)
            with self.assertRaisesRegex(RuntimeError, "previous state"):
                ModelPublisher(cfg).prepare()

    def test_initial_model_requires_an_explicit_onnx_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = ModelPublisher(config(root / "child"))
            self.assertEqual(child.prepare(), "fresh")
            model_directory = root / "models" / "save" / "2355"
            model_directory.mkdir(parents=True)
            (model_directory / "SaveModel.onnx").write_bytes(b"not-used")
            with self.assertRaisesRegex(RuntimeError, "explicit SaveModel.onnx"):
                child.load_initial_model(
                    PPOTrainer(child.config), str(model_directory)
                )
            wrong_name = model_directory / "model.onnx"
            wrong_name.write_bytes(b"not-used")
            with self.assertRaisesRegex(RuntimeError, "explicit SaveModel.onnx"):
                child.load_initial_model(
                    PPOTrainer(child.config), str(wrong_name)
                )

    def test_archive_cadence_uses_global_train_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = ModelPublisher(config(Path(directory)))
            self.assertFalse(publisher.should_mark_permanent(0))
            self.assertFalse(publisher.should_mark_permanent(1))
            self.assertTrue(publisher.should_mark_permanent(2))
            self.assertFalse(publisher.should_mark_permanent(3))
