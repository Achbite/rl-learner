import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from proto import training_pb2
from src.training.ppo_trainer import PPOTrainer


ROOT = Path(__file__).resolve().parents[1]
ACTOR_WIRE_GOLDEN = ROOT / "tests" / "fixtures" / "actor_wire_golden_v1.json"


def production_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_config() -> dict:
    document = production_config()
    document["training"]["n_epochs"] = 1
    document["training"]["mini_batch_size"] = 2
    return document


def neutral_training_samples() -> list[dict]:
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


def install_actor_wire_reference_model(trainer: PPOTrainer) -> None:
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
        for index in range(3):
            trainer.model.policy_encoder[0].weight[index, index] = 1.0
            trainer.model.policy_encoder[2].weight[index, index] = 1.0
            trainer.model.policy_head.weight[index, index] = 1.0
        trainer.model.value_encoder[0].weight[0, 3] = 1.0
        trainer.model.value_encoder[2].weight[0, 0] = 1.0
        trainer.model.value_head.weight[0, 0] = 1.0


class ModelParityTest(unittest.TestCase):
    def test_onnx_export_matches_the_pytorch_serving_outputs(self):
        trainer = PPOTrainer(test_config())
        observations = np.asarray(
            [[0.01 * index for index in range(17)]], dtype=np.float32
        )
        with torch.no_grad():
            expected_logits, expected_values = trainer.model(
                torch.from_numpy(observations)
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SaveModel.onnx"
            trainer.export_onnx(str(path))
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(
                [item.name for item in session.get_outputs()],
                ["action_logits", "value"],
            )
            actual_logits, actual_values = session.run(
                ["action_logits", "value"],
                {"observation": observations},
            )

        np.testing.assert_allclose(
            actual_logits, expected_logits.numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            actual_values, expected_values.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_actor_wire_sample_and_onnx_model_share_one_policy_binding(self):
        fixture = json.loads(ACTOR_WIRE_GOLDEN.read_text(encoding="utf-8"))
        trainer = PPOTrainer(production_config())
        install_actor_wire_reference_model(trainer)

        proto_golden = fixture["sample_proto_golden"]
        wire_sample = next(
            item
            for item in fixture["actor_wire_samples"]
            if item["action"] == proto_golden["action"]
        )
        sample = training_pb2.Sample(
            action=proto_golden["action"],
            reward=0.0,
            old_log_probability=wire_sample["old_log_probability"],
            old_value_prediction=wire_sample["old_value_prediction"],
            end_kind=getattr(training_pb2, proto_golden["end_kind"]),
            action_step=proto_golden["action_step"],
        )
        sample.observation.extend(fixture["observation"])
        sample.next_observation.extend(fixture["observation"])
        self.assertEqual(
            sample.SerializeToString(deterministic=True).hex(),
            proto_golden["deterministic_hex"],
        )

        observations = np.asarray(
            [fixture["observation"]] * len(fixture["actor_wire_samples"]),
            dtype=np.float32,
        )
        actions = torch.tensor(
            [item["action"] for item in fixture["actor_wire_samples"]],
            dtype=torch.long,
        )
        old_log_probabilities = torch.tensor(
            [
                item["old_log_probability"]
                for item in fixture["actor_wire_samples"]
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits, values = trainer.model(torch.from_numpy(observations))
            new_log_probabilities, evaluated_values, _ = (
                trainer.model.evaluate_actions(
                    torch.from_numpy(observations), actions
                )
            )
            ratio = trainer._importance_ratio(
                new_log_probabilities, old_log_probabilities
            )

        expected_logits = np.asarray(
            [fixture["actor_logits"]] * len(fixture["actor_wire_samples"]),
            dtype=np.float32,
        )
        expected_values = np.asarray(
            [fixture["expected_pytorch_onnx_output"]["value"]]
            * len(fixture["actor_wire_samples"]),
            dtype=np.float32,
        )
        np.testing.assert_array_equal(logits.numpy(), expected_logits)
        np.testing.assert_array_equal(values.numpy(), expected_values)
        torch.testing.assert_close(
            evaluated_values,
            torch.tensor(
                [
                    item["old_value_prediction"]
                    for item in fixture["actor_wire_samples"]
                ],
                dtype=torch.float32,
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            ratio,
            torch.ones_like(ratio),
            rtol=0.0,
            atol=float(fixture["ratio_absolute_tolerance"]),
        )

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "SaveModel.onnx"
            trainer.export_onnx(str(model_path))
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            onnx_logits, onnx_values = session.run(
                ["action_logits", "value"],
                {"observation": observations},
            )
        np.testing.assert_allclose(
            onnx_logits, expected_logits, rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            onnx_values, expected_values, rtol=0.0, atol=1e-6
        )


class PpoUpdateTest(unittest.TestCase):
    def test_update_changes_model_once_and_exports_the_new_step(self):
        trainer = PPOTrainer(test_config())
        before = copy.deepcopy(trainer.model.state_dict())

        trainer.train_on_batch(
            neutral_training_samples(), behavior_model_step=0
        )

        self.assertEqual(trainer.model_step, 1)
        self.assertTrue(
            any(
                not torch.equal(value, trainer.model.state_dict()[key])
                for key, value in before.items()
            )
        )
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in trainer.model.parameters()
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "SaveModel.onnx"
            trainer.export_onnx(str(model_path))
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(
                [output.name for output in session.get_outputs()],
                ["action_logits", "value"],
            )

    def test_stale_or_non_finite_batch_does_not_mutate_the_model(self):
        trainer = PPOTrainer(test_config())
        for _ in range(3):
            trainer.train_on_batch(
                neutral_training_samples(), behavior_model_step=0
            )
        state = copy.deepcopy(trainer.model.state_dict())

        with self.assertRaisesRegex(ValueError, "max_policy_lag"):
            trainer.train_on_batch(
                neutral_training_samples(), behavior_model_step=0
            )
        invalid = neutral_training_samples()
        invalid[0]["old_log_probability"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid tensors"):
            trainer.train_on_batch(invalid, behavior_model_step=1)

        self.assertEqual(trainer.model_step, 3)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]))

    def test_mixed_behavior_steps_do_not_start_an_update(self):
        trainer = PPOTrainer(test_config())
        trainer.train_on_batch(
            neutral_training_samples(), behavior_model_step=0
        )
        samples = neutral_training_samples()
        samples[0]["behavior_model_step"] = 0
        samples[1]["behavior_model_step"] = 1
        state = copy.deepcopy(trainer.model.state_dict())

        with self.assertRaisesRegex(ValueError, "exactly one behavior"):
            trainer.train_on_batch(samples)

        self.assertEqual(trainer.model_step, 1)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]))
