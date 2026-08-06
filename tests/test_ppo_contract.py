import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from src.training.ppo_trainer import PPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    document = yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    document["training"]["n_epochs"] = 1
    document["training"]["mini_batch_size"] = 2
    return document


def trajectory(terminated=False, truncated=False) -> list[dict]:
    return [
        {
            "observation": [0.0] * 17,
            "next_observation": [0.1] * 17,
            "action": 1,
            "reward": 0.25,
            "old_log_probability": -2.0,
            "old_value_prediction": 0.1,
            "terminated": False,
            "truncated": False,
        },
        {
            "observation": [0.1] * 17,
            "next_observation": [0.2] * 17,
            "action": 2,
            "reward": -0.1,
            "old_log_probability": -2.1,
            "old_value_prediction": 0.2,
            "terminated": terminated,
            "truncated": truncated,
        },
    ]


class GaeContractTest(unittest.TestCase):
    def test_environment_terminal_requires_zero_bootstrap(self):
        trainer = PPOTrainer(config())
        samples = trainer.compute_gae(
            trajectory(terminated=True), 0.0, False
        )
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(math.isfinite(s["td_return"]) for s in samples))
        with self.assertRaisesRegex(ValueError, "must not bootstrap"):
            trainer.compute_gae(trajectory(terminated=True), 0.5, True)

    def test_continuing_and_external_truncation_require_bootstrap(self):
        trainer = PPOTrainer(config())
        for samples in (trajectory(), trajectory(truncated=True)):
            result = trainer.compute_gae(samples, 0.3, True)
            self.assertTrue(all(math.isfinite(s["advantage"]) for s in result))
        with self.assertRaisesRegex(ValueError, "requires bootstrap"):
            trainer.compute_gae(trajectory(), 0.0, False)

    def test_end_flags_are_mutually_exclusive_and_final(self):
        trainer = PPOTrainer(config())
        invalid = trajectory(terminated=True, truncated=True)
        with self.assertRaisesRegex(ValueError, "cannot be terminated"):
            trainer.compute_gae(invalid, 0.0, False)
        invalid = trajectory()
        invalid[0]["terminated"] = True
        with self.assertRaisesRegex(ValueError, "continues after"):
            trainer.compute_gae(invalid, 0.0, False)


class ModelParityTest(unittest.TestCase):
    def test_onnx_exports_logits_and_value_without_softmax(self):
        trainer = PPOTrainer(config())
        observations = np.asarray(
            [[0.01 * index for index in range(17)]], dtype=np.float32
        )
        with torch.no_grad():
            expected_logits, expected_values = trainer.model(
                torch.from_numpy(observations)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
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
        self.assertFalse(np.isclose(actual_logits.sum(), 1.0))


class PpoUpdateTest(unittest.TestCase):
    def _samples(self, trainer: PPOTrainer) -> list[dict]:
        return trainer.compute_gae(trajectory(), 0.25, True)

    def test_update_reports_finite_metrics_and_advances_once(self):
        trainer = PPOTrainer(config())
        stats = trainer.train_on_batch(
            self._samples(trainer), behavior_model_version=0
        )
        self.assertEqual(trainer.model_version, 1)
        for key in (
            "policy_loss",
            "value_loss",
            "total_loss",
            "entropy",
            "approx_kl",
            "gradient_norm",
            "value_pred_mean",
            "return_target_mean",
            "explained_variance",
        ):
            self.assertTrue(math.isfinite(stats[key]), key)

    def test_policy_lag_and_non_finite_input_fail_closed(self):
        trainer = PPOTrainer(config())
        trainer.train_on_batch(self._samples(trainer), behavior_model_version=0)
        trainer.train_on_batch(self._samples(trainer), behavior_model_version=0)
        self.assertEqual(trainer.model_version, 2)
        with self.assertRaisesRegex(ValueError, "max_policy_lag"):
            trainer.train_on_batch(
                self._samples(trainer), behavior_model_version=0
            )
        invalid = self._samples(trainer)
        invalid[0]["old_log_probability"] = float("nan")
        state = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaisesRegex(ValueError, "invalid tensors"):
            trainer.train_on_batch(invalid, behavior_model_version=1)
        self.assertEqual(trainer.model_version, 2)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
