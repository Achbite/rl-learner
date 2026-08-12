import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from main.training_runtime import ModelPublisher, TrainingRuntime, load_config
from src.contracts.identity import (
    canonical_config_digest,
    manifest_message,
    training_config_digest,
    training_config_document,
    validate_config,
)
from src.metrics.metrics_backend import DisabledMetricsBackend
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


def samples() -> list[dict]:
    return [
        {
            "observation": [0.0] * 17,
            "action": 0,
            "old_log_probability": -2.0,
            "old_value_prediction": 0.1,
            "advantage": 0.5,
            "td_return": 0.6,
        },
        {
            "observation": [0.1] * 17,
            "action": 1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "advantage": -0.25,
            "td_return": -0.05,
        },
    ]


def publish_update(trainer, publisher, behavior) -> dict:
    stats = trainer.train_on_batch(
        samples(), behavior_model_version=behavior["model_version"]
    )
    update_id = f"update-v{trainer.model_version}"
    publisher.commit_optimizer_checkpoint(
        trainer,
        train_update_id=update_id,
        behavior_model=behavior,
        batch_ids=[f"batch-{trainer.model_version}"],
        stats=stats,
        sample_count=2,
        train_updates=trainer.model_version,
        trained_samples=trainer.model_version * 2,
    )
    return publisher.publish_runtime(
        trainer,
        train_update_id=update_id,
        behavior_model=behavior,
        batch_ids=[f"batch-{trainer.model_version}"],
        stats=stats,
        sample_count=2,
        train_updates=trainer.model_version,
        trained_samples=trainer.model_version * 2,
        checkpoint_precommitted=True,
    )


class ModelCommitContractTest(unittest.TestCase):
    def test_config_is_complete_and_old_environment_alias_is_ignored(self):
        document = load_config(str(ROOT / "configs" / "learner_config.yaml"))
        self.assertEqual(document["contract"]["package_version"], "0.10.0")
        self.assertEqual(document["model"]["obs_dim"], 17)
        self.assertEqual(
            document["training_semantics"]["reward_schema"]["schema_id"],
            "maze.reward.v4",
        )
        self.assertNotIn("map_id", document)
        self.assertNotIn("reward", document)
        sample = document["sample_distributor"]
        self.assertGreaterEqual(
            sample["demand_max_fragments"],
            sample["max_train_batch_size"],
        )

    def test_archive_interval_uses_only_canonical_runtime_override(self):
        document = config(Path("/tmp/learner-archive-override"), 200)
        with mock.patch.dict(
            "os.environ",
            {
                "RL_ARCHIVE_INTERVAL_UPDATES": "2",
                "MAZE_ARCHIVE_INTERVAL_UPDATES": "1",
            },
            clear=False,
        ):
            publisher = ModelPublisher(document)
        self.assertEqual(publisher.archive_interval_updates, 2)

        with mock.patch.dict(
            "os.environ",
            {"MAZE_ARCHIVE_INTERVAL_UPDATES": "1"},
            clear=True,
        ):
            publisher = ModelPublisher(document)
        self.assertEqual(publisher.archive_interval_updates, 200)

    def test_training_identity_digest_is_canonical_and_seed_specific(self):
        document = yaml.safe_load(
            (ROOT / "configs" / "learner_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        semantics = copy.deepcopy(document["training_semantics"])
        configured_semantics_digest = semantics.pop("semantics_digest")
        self.assertEqual(
            canonical_config_digest(semantics),
            configured_semantics_digest,
        )

        base_digest = (
            "b8a98bd14abc5f09e57c65516ff1eae8"
            "222b9515b058d76c34af4a88dee7551f"
        )
        self.assertEqual(
            document["identity"]["training_config_digest"],
            base_digest,
        )
        self.assertEqual(
            canonical_config_digest(training_config_document(document)),
            base_digest,
        )
        self.assertEqual(training_config_digest(document).hex, base_digest)

        seed_one = copy.deepcopy(document)
        seed_one["identity"]["model_lineage_id"] = "maze-fixed-map-seed-1"
        seed_one["training"]["seed"] = 1
        seed_one["model"]["bootstrap_seed"] = 1
        seed_one_digest = (
            "f61cdd19203538269fc18aa5ba349d4b"
            "877bdbc5103b763acab71300289ab2e0"
        )
        seed_one["identity"]["training_config_digest"] = seed_one_digest
        self.assertEqual(
            canonical_config_digest(training_config_document(seed_one)),
            seed_one_digest,
        )
        self.assertEqual(
            training_config_digest(seed_one).hex,
            seed_one_digest,
        )
        validate_config(seed_one)

        stale_seed_one = copy.deepcopy(seed_one)
        stale_seed_one["identity"]["training_config_digest"] = base_digest
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_config(stale_seed_one)

        missing_declaration = copy.deepcopy(seed_one)
        del missing_declaration["identity"]["training_config_digest"]
        with self.assertRaisesRegex(ValueError, "is required"):
            validate_config(missing_declaration)

    def test_prepare_rejects_unclean_local_train_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ModelPublisher(config(Path(directory)))
            first.prepare()
            first.state_path.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "was not cleaned"):
                ModelPublisher(config(Path(directory))).prepare()

    def test_prepare_ignores_optional_metrics_history(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ModelPublisher(config(Path(directory)))
            first.prepare()
            first.metrics_dir.mkdir(parents=True)
            (first.metrics_dir / "old.jsonl").write_text("{}\n")

            ModelPublisher(config(Path(directory))).prepare()

    def test_runtime_starts_with_metrics_path_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "local-train" / "metrics"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text("metrics path is unavailable\n")

            runtime = TrainingRuntime(config(root))
            try:
                self.assertIsInstance(
                    runtime.metrics_backend, DisabledMetricsBackend
                )
                self.assertTrue(metrics_path.is_file())
            finally:
                runtime.metrics_backend.close()
                runtime.actor_channel.close()
                runtime.model_channel.close()
                runtime.sample_channel.close()

    def test_manifest_binds_full_identity_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            trainer = PPOTrainer(cfg)
            publisher = ModelPublisher(cfg)
            publisher.prepare()
            manifest = publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model=None,
                batch_ids=[],
            )
            self.assertEqual(manifest["contract"]["package_version"], "0.10.0")
            self.assertEqual(manifest["observation_schema"]["schema_id"], "maze.observation.v3")
            self.assertEqual(manifest["input_shape"], [1, 17])
            wire = manifest_message(TrainingRuntime._manifest_for_wire(manifest))
            self.assertEqual(wire.identity.model_version, 0)
            self.assertIsNotNone(publisher.complete_manifest(0))
            with publisher.model_path(0).open("ab") as stream:
                stream.write(b"corrupt")
            self.assertIsNone(publisher.complete_manifest(0))

    def test_archive_retention_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            trainer = PPOTrainer(cfg)
            publisher = ModelPublisher(cfg)
            publisher.prepare()
            first = publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model=None,
                batch_ids=[],
            )
            publisher.archive_version(0, "bootstrap")
            second = publish_update(trainer, publisher, first["identity"])
            third = publish_update(trainer, publisher, second["identity"])
            fourth = publish_update(trainer, publisher, third["identity"])
            publisher.archive_version(3, "interval")
            publisher.prune_runtime(3)
            self.assertFalse(publisher.model_path(0).exists())
            self.assertTrue(publisher.model_path(1).exists())
            self.assertTrue(publisher.model_path(2).exists())
            self.assertTrue(publisher.model_path(3).exists())
            archive = publisher.archive_path(3)
            self.assertEqual(
                {path.name for path in archive.iterdir()},
                {"SaveModel.onnx", "checkpoint.pt", "manifest.json"},
            )

            external = root / "external-checkpoint.pt"
            shutil.copyfile(archive / "checkpoint.pt", external)
            child_cfg = config(root / "child")
            child_cfg["identity"]["model_lineage_id"] = "different-lineage"
            child_cfg["identity"]["training_config_digest"] = (
                canonical_config_digest(training_config_document(child_cfg))
            )
            child = ModelPublisher(child_cfg)
            child.prepare()
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                child.load_initial_checkpoint(PPOTrainer(child_cfg), str(external))

    def test_checkpoint_restores_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            trainer = PPOTrainer(cfg)
            publisher = ModelPublisher(cfg)
            publisher.prepare()
            publisher.publish_runtime(
                trainer,
                train_update_id="bootstrap-v0",
                behavior_model=None,
                batch_ids=[],
            )
            expected = trainer.train_on_batch(samples(), behavior_model_version=0)
            expected_state = copy.deepcopy(trainer.model.state_dict())
            retry = PPOTrainer(cfg)
            self.assertTrue(retry.load_checkpoint(str(publisher.checkpoint_path(0))))
            actual = retry.train_on_batch(samples(), behavior_model_version=0)
            self.assertEqual(actual, expected)
            for key, value in expected_state.items():
                self.assertTrue(torch.equal(value, retry.model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
