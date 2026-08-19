import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from main.training_runtime import (
    ModelPublisher,
    TrainingRuntime,
    load_config,
    read_json,
)
from src.contracts.identity import (
    canonical_config_digest,
    finalize_manifest_digest,
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
        samples(), behavior_model_step=behavior["model_step"]
    )
    update_id = f"update-v{trainer.model_step}"
    publisher.commit_optimizer_checkpoint(
        trainer,
        train_update_id=update_id,
        behavior_model=behavior,
        batch_ids=[f"batch-{trainer.model_step}"],
        stats=stats,
        sample_count=2,
        train_updates=trainer.model_step,
        trained_samples=trainer.model_step * 2,
    )
    return publisher.publish_runtime(
        trainer,
        train_update_id=update_id,
        behavior_model=behavior,
        batch_ids=[f"batch-{trainer.model_step}"],
        stats=stats,
        sample_count=2,
        train_updates=trainer.model_step,
        trained_samples=trainer.model_step * 2,
        checkpoint_precommitted=True,
    )


class LightweightTrainer:
    def __init__(self, version: int = 0):
        self.model_step = version

    def export_onnx(self, path: str) -> None:
        Path(path).write_bytes(f"onnx-v{self.model_step}".encode())

    def save_checkpoint(self, path: str, metadata: dict) -> None:
        torch.save(
            {
                "model_step": self.model_step,
                "metadata": copy.deepcopy(metadata),
            },
            path,
        )


class ModelCommitContractTest(unittest.TestCase):
    def test_config_is_complete_and_old_environment_alias_is_ignored(self):
        with mock.patch.dict(
            "os.environ", {"RL_MODEL_LINEAGE_ID": "maze-test-model"}
        ):
            document = load_config(
                str(ROOT / "configs" / "learner_config.yaml")
            )
        self.assertEqual(document["contract"]["package_version"], "0.13.0")
        self.assertEqual(document["model"]["obs_dim"], 17)
        self.assertEqual(
            document["training_semantics"]["reward_schema"]["schema_id"],
            "maze.reward.v4",
        )
        self.assertNotIn("map_id", document)
        self.assertNotIn("reward", document)
        sample = document["sample_pool"]

    def test_runtime_config_requires_a_bound_internal_lineage(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "fresh internal"):
                load_config(str(ROOT / "configs" / "learner_config.yaml"))

    def test_archive_interval_uses_only_effective_config(self):
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
        self.assertEqual(publisher.archive_interval_updates, 200)

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
            "ba719ca401ca9bb496b2a209078c019be"
            "f6014cdc2fc53d4d39433e8e22e9931"
        )
        self.assertNotIn("training_config_digest", document["identity"])
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
            "8a7fd4b63fc116e30bc55f26ed50110"
            "020428e878913270debc268471a7be335"
        )
        self.assertEqual(
            canonical_config_digest(training_config_document(seed_one)),
            seed_one_digest,
        )
        self.assertEqual(
            training_config_digest(seed_one).hex,
            seed_one_digest,
        )
        validate_config(seed_one)

        expected_seed_one = copy.deepcopy(seed_one)
        expected_seed_one["identity"][
            "expected_training_config_digest"
        ] = seed_one_digest
        validate_config(expected_seed_one)

        stale_seed_one = copy.deepcopy(expected_seed_one)
        stale_seed_one["identity"][
            "expected_training_config_digest"
        ] = base_digest
        with self.assertRaisesRegex(
            ValueError, "expected_training_config_digest"
        ):
            validate_config(stale_seed_one)

        implicit_initial_model = copy.deepcopy(document)
        implicit_initial_model["model"]["initial_model_dir"] = (
            "models/save/002355"
        )
        with self.assertRaisesRegex(ValueError, "retired model"):
            validate_config(implicit_initial_model)

    def test_prepare_rejects_unclean_local_train_data(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ModelPublisher(config(Path(directory)))
            first.prepare()
            first.state_path.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "previous state"):
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
            self.assertEqual(manifest["contract"]["package_version"], "0.13.0")
            self.assertEqual(manifest["observation_schema"]["schema_id"], "maze.observation.v3")
            self.assertEqual(manifest["input_shape"], [1, 17])
            wire = manifest_message(TrainingRuntime._manifest_for_wire(manifest))
            self.assertEqual(wire.identity.model_step, 0)
            self.assertIsNotNone(publisher.complete_manifest(0))
            metadata_path = publisher.metadata_path(0)
            metadata = read_json(metadata_path)
            self.assertEqual(
                metadata["training_config_digest"],
                manifest["training_config_digest"],
            )
            metadata["training_config_digest"] = "f" * 64
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )
            self.assertIsNone(publisher.complete_manifest(0))
            metadata["training_config_digest"] = manifest[
                "training_config_digest"
            ]
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )
            self.assertIsNotNone(publisher.complete_manifest(0))
            with publisher.model_path(0).open("ab") as stream:
                stream.write(b"corrupt")
            self.assertIsNone(publisher.complete_manifest(0))

    def test_canonical_publication_retains_recent_101_and_permanent_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            publisher = ModelPublisher(cfg)
            publisher.prepare()
            source_trainer = PPOTrainer(cfg)
            manifest = publisher.publish_runtime(
                source_trainer,
                train_update_id="update-v0",
                behavior_model=None,
                batch_ids=[],
                train_updates=0,
                trained_samples=0,
            )
            wire_document = yaml.safe_load(
                publisher.manifest_path(0).read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(wire_document), ModelPublisher.WIRE_MANIFEST_KEYS
            )
            self.assertNotIn("train_update_id", wire_document)
            self.assertEqual(manifest["retention"]["class"], "rolling")
            trainer = LightweightTrainer()
            for version in range(1, 104):
                trainer.model_step = version
                publisher.publish_runtime(
                    trainer,
                    train_update_id=f"update-v{version}",
                    behavior_model=None,
                    batch_ids=[],
                    train_updates=version,
                    trained_samples=version,
                )
            permanent_metadata = publisher.metadata_path(2).read_bytes()
            self.assertEqual(
                publisher.complete_manifest(2)["retention"],
                {"class": "permanent", "reason": "interval"},
            )
            with mock.patch.object(
                publisher,
                "complete_manifest",
                wraps=publisher.complete_manifest,
            ) as validator:
                self.assertEqual(publisher.prune_publications(103), [0, 1])
            self.assertEqual(
                [call.args[0] for call in validator.call_args_list], [0, 1, 2]
            )
            self.assertFalse(publisher.publication_path(0).exists())
            self.assertFalse(publisher.publication_path(1).exists())
            self.assertTrue(publisher.publication_path(2).is_dir())
            self.assertEqual(
                publisher.metadata_path(2).read_bytes(), permanent_metadata
            )
            self.assertEqual(
                [
                    item["identity"]["model_step"]
                    for item in publisher.complete_manifests()
                    if item["retention"]["class"] == "rolling"
                ],
                list(range(3, 104, 2)),
            )
            publication = publisher.publication_path(103)
            self.assertEqual(
                {path.name for path in publication.iterdir()},
                {
                    "SaveModel.onnx",
                    "manifest.json",
                    "metadata.json",
                },
            )
            self.assertTrue(publisher.checkpoint_path(103).is_file())

    def test_step_name_is_minimum_width_seven_without_wraparound(self):
        self.assertEqual(ModelPublisher.step_name(0), "0000000")
        self.assertEqual(ModelPublisher.step_name(9999999), "9999999")
        self.assertEqual(ModelPublisher.step_name(10000000), "10000000")
        self.assertEqual(
            ModelPublisher.step_name((1 << 64) - 1),
            "18446744073709551615",
        )
        for value in (True, -1, 1 << 64, 1.0, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ModelPublisher.step_name(value)

    def test_failed_export_leaves_no_visible_or_private_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = ModelPublisher(config(Path(directory)))
            publisher.prepare()
            trainer = LightweightTrainer()
            trainer.export_onnx = mock.Mock(
                side_effect=RuntimeError("injected export failure")
            )
            with self.assertRaisesRegex(RuntimeError, "injected"):
                publisher.publish_runtime(
                    trainer,
                    train_update_id="bootstrap-v0",
                    behavior_model=None,
                    batch_ids=[],
                )
            self.assertFalse(publisher.publication_path(0).exists())
            self.assertEqual(publisher._canonical_step_directories(), [])

    def test_prepare_recovers_only_known_private_publication_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = ModelPublisher(config(root))
            publisher.publication_dir.mkdir(parents=True)
            for name in (
                ".publication-0000001-crash.tmp",
                ".prune-0000002-crash",
                ".rollback-delete-0000003-crash",
            ):
                path = publisher.publication_dir / name
                path.mkdir()
                (path / "partial").write_bytes(b"partial")
            publisher.prepare()
            self.assertFalse(
                any(
                    path.name.startswith(".")
                    for path in publisher.publication_dir.iterdir()
                )
            )

            unknown = publisher.publication_dir / ".unowned-entry"
            unknown.mkdir()
            second = ModelPublisher(config(root))
            with self.assertRaisesRegex(RuntimeError, "requires review"):
                second.prepare()
            self.assertTrue(unknown.is_dir())

    def test_state_write_failure_hides_publication_and_keeps_private_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            publisher = ModelPublisher(config(Path(directory)))
            publisher.prepare()
            trainer = LightweightTrainer(1)
            checkpoint = publisher.commit_optimizer_checkpoint(
                trainer,
                train_update_id="update-v1",
                behavior_model={
                    "minimum_model_step": 0,
                    "maximum_model_step": 0,
                },
                batch_ids=["batch-v1"],
                stats={},
                sample_count=2,
                train_updates=1,
                trained_samples=2,
            )
            from main import training_runtime

            original_write = training_runtime.atomic_write_json

            def fail_state(path, document):
                if Path(path) == publisher.state_path:
                    raise OSError("injected state write failure")
                return original_write(path, document)

            with mock.patch(
                "main.training_runtime.atomic_write_json",
                side_effect=fail_state,
            ):
                with self.assertRaisesRegex(OSError, "state write failure"):
                    publisher.publish_runtime(
                        trainer,
                        train_update_id="update-v1",
                        behavior_model={
                            "minimum_model_step": 0,
                            "maximum_model_step": 0,
                        },
                        batch_ids=["batch-v1"],
                        stats={},
                        sample_count=2,
                        train_updates=1,
                        trained_samples=2,
                        checkpoint_precommitted=True,
                    )
            self.assertFalse(publisher.publication_path(1).exists())
            self.assertFalse(publisher.state_path.exists())
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(publisher._canonical_step_directories(), [])

    def test_complete_manifest_rejects_mutated_wire_identity_and_shape(self):
        mutations = (
            lambda document: document["identity"].__setitem__(
                "model_lineage_id", "conflicting-lineage"
            ),
            lambda document: document.__setitem__(
                "training_config_digest", "f" * 64
            ),
            lambda document: document.__setitem__("input_shape", [1, 99]),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                with tempfile.TemporaryDirectory() as directory:
                    publisher = ModelPublisher(config(Path(directory)))
                    publisher.prepare()
                    publisher.publish_runtime(
                        LightweightTrainer(),
                        train_update_id="bootstrap-v0",
                        behavior_model=None,
                        batch_ids=[],
                    )
                    manifest_path = publisher.manifest_path(0)
                    document = read_json(manifest_path)
                    mutate(document)
                    document = finalize_manifest_digest(document)
                    manifest_path.write_text(
                        json.dumps(document, sort_keys=True), encoding="utf-8"
                    )
                    metadata_path = publisher.metadata_path(0)
                    metadata = read_json(metadata_path)
                    metadata["model_identity"] = document["identity"]
                    metadata_path.write_text(
                        json.dumps(metadata, sort_keys=True), encoding="utf-8"
                    )
                    self.assertIsNone(publisher.complete_manifest(0))
