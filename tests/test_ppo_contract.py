import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from main.training_runtime import TrainingRuntime
from proto import maze_pb2
from src.training.ppo_trainer import PPOTrainer


def trainer_config() -> dict:
    return {
        "model": {"obs_dim": 3, "action_dim": 2, "hidden_dim": 8},
        "training": {
            "device": "cpu",
            "seed": 7,
            "learning_rate": 0.001,
            "gamma": 1.0,
            "gae_lambda": 1.0,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 2,
            "mini_batch_size": 2,
            "normalize_advantage": True,
        },
    }


def trajectory(final_terminated: bool = False) -> list[dict]:
    return [
        {
            "obs": [0.0, 0.1, 0.2],
            "action": 0,
            "reward": 1.0,
            "old_log_prob": -0.7,
            "old_vpred": 0.5,
            "terminated": False,
            "truncated": False,
        },
        {
            "obs": [0.2, 0.3, 0.4],
            "action": 1,
            "reward": 2.0,
            "old_log_prob": -0.6,
            "old_vpred": 0.25,
            "terminated": final_terminated,
            "truncated": not final_terminated,
        },
    ]


class GaeContractTest(unittest.TestCase):
    def test_terminated_fragment_uses_zero_bootstrap(self):
        trainer = PPOTrainer(trainer_config())
        result = trainer.compute_gae(
            trajectory(final_terminated=True),
            bootstrap_value=0.0,
            bootstrap_valid=True,
        )
        np.testing.assert_allclose(
            [sample["advantage"] for sample in result],
            [2.5, 1.75],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [sample["td_return"] for sample in result],
            [3.0, 2.0],
            rtol=0,
            atol=1e-6,
        )

    def test_truncated_fragment_uses_behavior_bootstrap(self):
        trainer = PPOTrainer(trainer_config())
        result = trainer.compute_gae(
            trajectory(final_terminated=False),
            bootstrap_value=0.75,
            bootstrap_valid=True,
        )
        np.testing.assert_allclose(
            [sample["advantage"] for sample in result],
            [3.25, 2.5],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            [sample["td_return"] for sample in result],
            [3.75, 2.75],
            rtol=0,
            atol=1e-6,
        )

    def test_tmax_fragment_requires_valid_bootstrap(self):
        trainer = PPOTrainer(trainer_config())
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            trainer.compute_gae(
                trajectory(final_terminated=False),
                bootstrap_value=0.0,
                bootstrap_valid=False,
            )


class ModelParityTest(unittest.TestCase):
    def test_pytorch_and_onnx_outputs_match(self):
        trainer = PPOTrainer(trainer_config())
        observations = np.asarray(
            [[0.0, 0.1, 0.2], [0.4, -0.2, 0.8]],
            dtype=np.float32,
        )
        with torch.no_grad():
            expected_probs, expected_values = trainer.model(
                torch.from_numpy(observations)
            )

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            trainer.export_onnx(str(model_path))
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            actual_probs, actual_values = session.run(
                ["action_probs", "value"], {"obs": observations}
            )

        np.testing.assert_allclose(
            actual_probs, expected_probs.numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            actual_values, expected_values.numpy(), rtol=1e-5, atol=1e-6
        )


class PpoUpdateTest(unittest.TestCase):
    def test_update_reports_finite_metrics(self):
        trainer = PPOTrainer(trainer_config())
        samples = trainer.compute_gae(
            trajectory(final_terminated=False),
            bootstrap_value=0.75,
            bootstrap_valid=True,
        )
        stats = trainer.train_on_batch(samples)
        for field in (
            "policy_loss",
            "value_loss",
            "total_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "gradient_norm",
        ):
            self.assertTrue(np.isfinite(stats[field]), field)
        self.assertEqual(stats["model_version"], 1)


class FragmentIdentityTest(unittest.TestCase):
    @staticmethod
    def batch(checksum: str):
        batch = maze_pb2.SampleBatch(
            protocol_version=3,
            batch_id="batch-0",
            behavior_model_version=0,
            behavior_model_checksum=checksum,
            bootstrap_value=0.5,
            bootstrap_valid=True,
            first_action_frame_id=0,
            last_action_frame_id=1,
        )
        for frame_id in range(2):
            batch.samples.add(
                obs=[0.0, 0.1, 0.2],
                action=frame_id,
                reward=1.0,
                old_log_prob=-0.5,
                old_vpred=0.25,
                action_frame_id=frame_id,
                termination_reason=maze_pb2.TERMINATION_REASON_ACTIVE,
            )
        return batch

    def test_fragment_checksum_must_match_published_behavior_model(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.trainer = PPOTrainer(trainer_config())
        runtime._behavior_checksums = {0: "a" * 64}

        version, batch_ids = runtime._validate_fragments(
            [self.batch("a" * 64)]
        )
        self.assertEqual(version, 0)
        self.assertEqual(batch_ids, ["batch-0"])

        with self.assertRaisesRegex(ValueError, "published model"):
            runtime._validate_fragments([self.batch("b" * 64)])


if __name__ == "__main__":
    unittest.main()
