import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from main.training_runtime import ModelPublisher, TrainingRuntime, load_config
from src.contracts.identity import manifest_message
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
        self.assertEqual(document["contract"]["package_version"], "0.8.0")
        self.assertEqual(document["model"]["obs_dim"], 17)
        self.assertNotIn("map_id", document)
        self.assertNotIn("reward", document)

    def test_prepare_rejects_unclean_local_train_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ModelPublisher(config(Path(directory)))
            first.prepare()
            (first.metrics_dir / "old.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "was not cleaned"):
                ModelPublisher(config(Path(directory))).prepare()

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
            self.assertEqual(manifest["contract"]["package_version"], "0.8.0")
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
            publisher.archive_version(2, "interval")
            publisher.prune_runtime(2)
            self.assertFalse(publisher.model_path(0).exists())
            self.assertTrue(publisher.model_path(1).exists())
            self.assertTrue(publisher.model_path(2).exists())
            archive = publisher.archive_path(2)
            self.assertEqual(
                {path.name for path in archive.iterdir()},
                {"SaveModel.onnx", "checkpoint.pt", "manifest.json"},
            )

            external = root / "external-checkpoint.pt"
            shutil.copyfile(archive / "checkpoint.pt", external)
            child_cfg = config(root / "child")
            child_cfg["identity"]["model_lineage_id"] = "different-lineage"
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
