import tempfile
import unittest
from pathlib import Path

import yaml

from src.config.command_line import parse_startup_arguments
from src.config.effective_config import load_effective_config
from src.contracts.identity import training_config_digest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "learner_config.yaml"


class EffectiveConfigTest(unittest.TestCase):
    @staticmethod
    def load(overrides: dict[str, str] | None = None) -> dict:
        return load_effective_config(
            str(CONFIG),
            {
                "RL_MODEL_LINEAGE_ID": "learner-config-chain-test",
                **(overrides or {}),
            },
        )

    def test_default_and_allowed_ppo_overrides_resolve_for_startup(self):
        baseline = self.load()
        overrides = {
            "RL_PPO_LEARNING_RATE": "0.0001",
            "RL_PPO_N_EPOCHS": "2",
            "RL_PPO_MINI_BATCH_SIZE": "32",
            "RL_PPO_TRAIN_BATCH_SIZE": "256",
        }
        first = self.load(overrides)
        second = self.load(dict(reversed(list(overrides.items()))))

        self.assertEqual(first["training"]["learning_rate"], 0.0001)
        self.assertEqual(first["training"]["n_epochs"], 2)
        self.assertEqual(first["training"]["mini_batch_size"], 32)
        self.assertEqual(first["sample_pool"]["train_batch_size"], 256)
        self.assertEqual(
            first["sample_pool"]["max_train_batch_size"], 383
        )
        self.assertNotEqual(
            training_config_digest(first).hex,
            training_config_digest(baseline).hex,
        )
        self.assertEqual(
            training_config_digest(first).hex,
            training_config_digest(second).hex,
        )

    def test_cli_paths_and_endpoints_override_config_for_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "learner.yaml"
            config_path.write_text(
                CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
            )
            initial_model = root / "warm" / "SaveModel.onnx"
            initial_model.parent.mkdir()
            initial_model.write_bytes(b"onnx-chain-placeholder")
            startup = parse_startup_arguments(
                [
                    "--config",
                    str(config_path),
                    "--initial-model",
                    "../warm/SaveModel.onnx",
                    "--model-distributor",
                    "model-service:19200",
                    "--aiserver",
                    "actor-service:19002",
                ]
            )
            config = load_effective_config(
                startup.config_path,
                {"RL_MODEL_LINEAGE_ID": "learner-cli-chain-test"},
                startup.cli_overrides,
            )

            self.assertEqual(
                config["model"]["local_train_dir"],
                str(root / "models" / "train"),
            )
            self.assertEqual(
                config["model"]["initial_model_path"], str(initial_model)
            )
            self.assertEqual(
                config["model_distributor"],
                {"host": "model-service", "port": 19200},
            )
            self.assertEqual(config["aiserver_status"]["host"], "actor-service")
            self.assertEqual(config["aiserver_status"]["port"], 19002)

            inside = (
                root / "models" / "train" / "0000000" / "SaveModel.onnx"
            )
            inside.parent.mkdir(parents=True)
            inside.write_bytes(b"onnx-chain-placeholder")
            invalid = parse_startup_arguments(
                [
                    "--config",
                    str(config_path),
                    "--initial-model",
                    "../models/train/0000000/SaveModel.onnx",
                ]
            )
            with self.assertRaisesRegex(
                ValueError, "outside the fresh training workspace"
            ):
                load_effective_config(
                    invalid.config_path,
                    {"RL_MODEL_LINEAGE_ID": "learner-cli-chain-test"},
                    invalid.cli_overrides,
                )

    def test_invalid_or_platform_owned_overrides_fail_before_startup(self):
        cases = (
            ({"RL_PPO_LEARNING_RATE": "nan"}, "finite"),
            ({"RL_PPO_LEARNNG_RATE": "0.1"}, "unknown"),
            ({"RL_RUN_ID": "infra-owned"}, "platform control identity"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.load(overrides)

    def test_initial_model_ack_wait_default_and_bounded_diagnostic(self):
        self.assertIsNone(
            self.load()["aiserver_status"]["initial_model_ack_timeout_sec"]
        )

        with tempfile.TemporaryDirectory() as temporary:
            document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            document["aiserver_status"][
                "initial_model_ack_timeout_sec"
            ] = 45.5
            config_path = Path(temporary) / "learner.yaml"
            config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
            config = load_effective_config(
                str(config_path),
                {"RL_MODEL_LINEAGE_ID": "learner-config-chain-test"},
            )
            self.assertEqual(
                config["aiserver_status"]["initial_model_ack_timeout_sec"],
                45.5,
            )
