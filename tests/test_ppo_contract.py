import copy
import json
import math
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
ACTOR_WIRE_GOLDEN = (
    ROOT / "tests" / "fixtures" / "actor_wire_golden_v1.json"
)


def production_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def config() -> dict:
    document = production_config()
    document["training"]["n_epochs"] = 1
    document["training"]["mini_batch_size"] = 2
    return document


def actor_wire_golden() -> dict:
    return json.loads(ACTOR_WIRE_GOLDEN.read_text(encoding="utf-8"))


def install_actor_wire_reference_model(trainer: PPOTrainer) -> None:
    """Install the independently specified identity-path reference MLP."""
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
    @staticmethod
    def _golden_trainer() -> PPOTrainer:
        document = config()
        document["training"]["gamma"] = 0.9
        document["training"]["gae_lambda"] = 0.8
        return PPOTrainer(document)

    @staticmethod
    def _golden_trajectory(
        *, terminated: bool = False, truncated: bool = False
    ) -> list[dict]:
        samples = trajectory(terminated=terminated, truncated=truncated)
        samples[0]["reward"] = 1.0
        samples[0]["old_value_prediction"] = 0.5
        samples[1]["reward"] = 2.0
        samples[1]["old_value_prediction"] = 0.25
        return samples

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

    def test_terminal_gae_matches_hand_calculated_golden(self):
        result = self._golden_trainer().compute_gae(
            self._golden_trajectory(terminated=True),
            0.0,
            False,
        )
        self.assertAlmostEqual(result[1]["advantage"], 1.75, places=5)
        self.assertAlmostEqual(result[1]["td_return"], 2.0, places=5)
        self.assertAlmostEqual(result[0]["advantage"], 1.985, places=5)
        self.assertAlmostEqual(result[0]["td_return"], 2.485, places=5)

    def test_continuing_and_truncated_gae_match_bootstrap_golden(self):
        trainer = self._golden_trainer()
        for final_flags in ({}, {"truncated": True}):
            result = trainer.compute_gae(
                self._golden_trajectory(**final_flags),
                0.4,
                True,
            )
            self.assertAlmostEqual(result[1]["advantage"], 2.11, places=5)
            self.assertAlmostEqual(result[1]["td_return"], 2.36, places=5)
            self.assertAlmostEqual(result[0]["advantage"], 2.2442, places=5)
            self.assertAlmostEqual(result[0]["td_return"], 2.7442, places=5)

    def test_fragment_boundary_resets_gae_trace_at_bootstrap(self):
        trainer = self._golden_trainer()
        first_fragment = [self._golden_trajectory()[0]]
        second_fragment = [self._golden_trajectory(terminated=True)[1]]

        first = trainer.compute_gae(first_fragment, 0.25, True)
        second = trainer.compute_gae(second_fragment, 0.0, False)

        self.assertAlmostEqual(first[0]["advantage"], 0.725, places=5)
        self.assertAlmostEqual(first[0]["td_return"], 1.225, places=5)
        self.assertAlmostEqual(second[0]["advantage"], 1.75, places=5)
        self.assertAlmostEqual(second[0]["td_return"], 2.0, places=5)
        self.assertNotAlmostEqual(first[0]["advantage"], 1.985, places=5)


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

    def test_fixed_actor_wire_golden_matches_pytorch_onnx_and_lag_zero(self):
        fixture = actor_wire_golden()
        self.assertEqual(fixture["fixture_id"], "a3-arch-policy-wire-v1")
        self.assertEqual(fixture["wire_dtype"], "float32")
        self.assertEqual(
            fixture["provenance"]["kind"],
            "actor-side-independent-reference",
        )
        trainer = PPOTrainer(production_config())
        install_actor_wire_reference_model(trainer)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            model_path = Path(directory) / "model.onnx"
            trainer.save_checkpoint(str(checkpoint))

            restored = PPOTrainer(production_config())
            self.assertTrue(restored.load_checkpoint(str(checkpoint)))
            restored.export_onnx(str(model_path))

            wire_samples = fixture["actor_wire_samples"]
            self.assertEqual(
                restored.model_version,
                fixture["behavior_model_version"],
            )
            proto_golden = fixture["sample_proto_golden"]
            action_two_wire = next(
                sample
                for sample in wire_samples
                if sample["action"] == proto_golden["action"]
            )
            actor_sample = training_pb2.Sample(
                action=proto_golden["action"],
                reward=proto_golden["reward"],
                old_log_probability=action_two_wire[
                    "old_log_probability"
                ],
                old_value_prediction=action_two_wire[
                    "old_value_prediction"
                ],
                end_kind=getattr(
                    training_pb2, proto_golden["end_kind"]
                ),
                action_step=proto_golden["action_step"],
            )
            actor_sample.observation.extend(fixture["observation"])
            actor_sample.next_observation.extend(fixture["observation"])
            actor_wire = actor_sample.SerializeToString(deterministic=True)
            self.assertEqual(
                actor_wire.hex(), proto_golden["deterministic_hex"]
            )
            parsed_sample = training_pb2.Sample.FromString(
                bytes.fromhex(proto_golden["deterministic_hex"])
            )
            self.assertEqual(
                parsed_sample.SerializeToString(deterministic=True).hex(),
                proto_golden["deterministic_hex"],
            )
            self.assertEqual(parsed_sample.action, action_two_wire["action"])
            self.assertEqual(
                parsed_sample.old_log_probability,
                np.float32(action_two_wire["old_log_probability"]).item(),
            )
            self.assertEqual(
                parsed_sample.old_value_prediction,
                np.float32(action_two_wire["old_value_prediction"]).item(),
            )
            observations = np.asarray(
                [fixture["observation"]] * len(wire_samples),
                dtype=np.float32,
            )
            actor_logits = np.asarray(
                fixture["actor_logits"], dtype=np.float32
            )
            model_output_logits = np.asarray(
                fixture["expected_pytorch_onnx_output"]["action_logits"],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(
                model_output_logits, actor_logits
            )
            expected_logits = np.asarray(
                [actor_logits] * len(wire_samples),
                dtype=np.float32,
            )
            expected_values = np.asarray(
                [fixture["expected_pytorch_onnx_output"]["value"]]
                * len(wire_samples),
                dtype=np.float32,
            )
            actions = torch.tensor(
                [sample["action"] for sample in wire_samples],
                dtype=torch.long,
            )
            wire_old_log_probs = torch.tensor(
                [sample["old_log_probability"] for sample in wire_samples],
                dtype=torch.float32,
            )
            wire_old_values = torch.tensor(
                [sample["old_value_prediction"] for sample in wire_samples],
                dtype=torch.float32,
            )
            with torch.no_grad():
                torch_logits, torch_values = restored.model(
                    torch.from_numpy(observations)
                )
                new_log_probs, evaluated_values, _ = (
                    restored.model.evaluate_actions(
                        torch.from_numpy(observations), actions
                    )
                )
                ratio = restored._importance_ratio(
                    new_log_probs,
                    wire_old_log_probs,
                )
                parsed_log_prob, parsed_value, _ = (
                    restored.model.evaluate_actions(
                        torch.tensor(
                            [list(parsed_sample.observation)],
                            dtype=torch.float32,
                        ),
                        torch.tensor([parsed_sample.action]),
                    )
                )
                parsed_ratio = restored._importance_ratio(
                    parsed_log_prob,
                    torch.tensor(
                        [parsed_sample.old_log_probability],
                        dtype=torch.float32,
                    ),
                )

            np.testing.assert_array_equal(
                torch_logits.numpy(), expected_logits
            )
            np.testing.assert_array_equal(
                torch_values.numpy(), expected_values
            )
            torch.testing.assert_close(
                evaluated_values,
                wire_old_values,
                rtol=0.0,
                atol=0.0,
            )
            tolerance = float(fixture["ratio_absolute_tolerance"])
            torch.testing.assert_close(
                parsed_value,
                torch.tensor([parsed_sample.old_value_prediction]),
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                parsed_ratio,
                torch.tensor([fixture["expected_lag_zero_ratio"]]),
                rtol=0.0,
                atol=tolerance,
            )
            torch.testing.assert_close(
                new_log_probs,
                wire_old_log_probs,
                rtol=0.0,
                atol=tolerance,
            )
            torch.testing.assert_close(
                ratio,
                torch.full_like(
                    ratio, float(fixture["expected_lag_zero_ratio"])
                ),
                rtol=0.0,
                atol=tolerance,
            )
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
    def _samples(self, trainer: PPOTrainer) -> list[dict]:
        return trainer.compute_gae(trajectory(), 0.25, True)

    def test_update_reports_finite_metrics_and_advances_once(self):
        trainer = PPOTrainer(config())
        before = copy.deepcopy(trainer.model.state_dict())
        stats = trainer.train_on_batch(
            self._samples(trainer), behavior_model_version=0
        )
        self.assertEqual(trainer.model_version, 1)
        self.assertTrue(
            any(
                not torch.equal(value, trainer.model.state_dict()[key])
                for key, value in before.items()
            )
        )
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
        trainer.train_on_batch(self._samples(trainer), behavior_model_version=0)
        self.assertEqual(trainer.model_version, 3)
        with self.assertRaisesRegex(ValueError, "max_policy_lag"):
            trainer.train_on_batch(
                self._samples(trainer), behavior_model_version=0
            )
        invalid = self._samples(trainer)
        invalid[0]["old_log_probability"] = float("nan")
        state = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaisesRegex(ValueError, "invalid tensors"):
            trainer.train_on_batch(invalid, behavior_model_version=1)
        self.assertEqual(trainer.model_version, 3)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, trainer.model.state_dict()[key]))

    def test_mixed_compatible_behavior_versions_train_once(self):
        trainer = PPOTrainer(config())
        trainer.train_on_batch(self._samples(trainer), behavior_model_version=0)
        samples = self._samples(trainer)
        samples[0]["behavior_model_version"] = 0
        samples[1]["behavior_model_version"] = 1
        stats = trainer.train_on_batch(samples)
        self.assertEqual(trainer.model_version, 2)
        self.assertEqual(stats["minimum_behavior_model_version"], 0)
        self.assertEqual(stats["maximum_behavior_model_version"], 1)
        self.assertEqual(stats["policy_lag"], 1)

    def test_reward_gae_and_ppo_move_logits_in_expected_direction(self):
        document = config()
        document["training"]["normalize_advantage"] = False
        document["training"]["entropy_coef"] = 0.0
        document["training"]["value_coef"] = 0.0
        document["training"]["max_grad_norm"] = 100.0
        trainer = PPOTrainer(document)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()

        old_log_probability = -math.log(9.0)
        terminal_samples = []
        for action, reward in ((0, 1.0), (1, -1.0)):
            trajectory_sample = {
                "observation": [0.0] * 17,
                "next_observation": [0.0] * 17,
                "action": action,
                "reward": reward,
                "old_log_probability": old_log_probability,
                "old_value_prediction": 0.0,
                "terminated": True,
                "truncated": False,
            }
            terminal_samples.extend(
                trainer.compute_gae([trajectory_sample], 0.0, False)
            )

        self.assertEqual(terminal_samples[0]["advantage"], 1.0)
        self.assertEqual(terminal_samples[1]["advantage"], -1.0)

        trainer.train_on_batch(terminal_samples, behavior_model_version=0)

        bias = trainer.model.policy_head.bias.detach()
        self.assertGreater(bias[0].item(), 0.0)
        self.assertLess(bias[1].item(), 0.0)
        self.assertEqual(trainer.model_version, 1)

    def test_production_defaults_move_policy_and_critic_then_export_parity(self):
        document = production_config()
        trainer = PPOTrainer(document)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
            for index in range(2):
                trainer.model.policy_encoder[0].weight[index, index] = 1.0
                trainer.model.policy_encoder[2].weight[index, index] = 1.0
                trainer.model.value_encoder[0].weight[index, index] = 1.0
                trainer.model.value_encoder[2].weight[index, index] = 1.0
            for action in range(9):
                scale = 0.01 * (action + 1)
                trainer.model.policy_head.weight[action, 0] = scale
                trainer.model.policy_head.weight[action, 1] = -scale
            trainer.model.value_head.weight[0, 0] = 0.1
            trainer.model.value_head.weight[0, 1] = -0.1

        observations = [
            [1.0, 0.0] + [0.0] * 15,
            [0.0, 1.0] + [0.0] * 15,
        ]
        actions = [0, 1]
        return_targets = [1.0, -1.0]
        observation_tensor = torch.tensor(
            observations, dtype=torch.float32
        )
        with torch.no_grad():
            before_logits, before_values = trainer.model(observation_tensor)
            before_log_probs = torch.log_softmax(before_logits, dim=-1)
        before_policy_encoder = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.policy_encoder.named_parameters()
            if name.endswith("weight")
        }
        before_value_encoder = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.value_encoder.named_parameters()
            if name.endswith("weight")
        }
        terminal_samples = []
        for index, (observation, action, target) in enumerate(
            zip(observations, actions, return_targets)
        ):
            terminal_samples.extend(
                trainer.compute_gae(
                    [
                        {
                            "observation": observation,
                            "next_observation": observation,
                            "action": action,
                            "reward": target,
                            "old_log_probability": before_log_probs[
                                index, action
                            ].item(),
                            "old_value_prediction": before_values[
                                index
                            ].item(),
                            "terminated": True,
                            "truncated": False,
                        }
                    ],
                    0.0,
                    False,
                )
            )

        self.assertEqual(
            [sample["td_return"] for sample in terminal_samples],
            return_targets,
        )
        self.assertGreater(
            terminal_samples[0]["advantage"],
            terminal_samples[1]["advantage"],
        )
        trainer.train_on_batch(terminal_samples, behavior_model_version=0)

        with torch.no_grad():
            after_logits, after_values = trainer.model(observation_tensor)
            after_log_probs = torch.log_softmax(after_logits, dim=-1)
        self.assertGreater(
            after_log_probs[0, actions[0]].item(),
            before_log_probs[0, actions[0]].item(),
        )
        self.assertLess(
            after_log_probs[1, actions[1]].item(),
            before_log_probs[1, actions[1]].item(),
        )
        self.assertTrue(
            all(
                abs(target - after_values[index].item())
                < abs(target - before_values[index].item())
                for index, target in enumerate(return_targets)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(
                    before_policy_encoder[name], parameter
                )
                for name, parameter in (
                    trainer.model.policy_encoder.named_parameters()
                )
                if name.endswith("weight")
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before_value_encoder[name], parameter)
                for name, parameter in (
                    trainer.model.value_encoder.named_parameters()
                )
                if name.endswith("weight")
            )
        )
        self.assertEqual(trainer.model_version, 1)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "trained.onnx"
            trainer.export_onnx(str(model_path))
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            onnx_logits, onnx_values = session.run(
                ["action_logits", "value"],
                {"observation": observation_tensor.numpy()},
            )
        np.testing.assert_allclose(
            onnx_logits, after_logits.numpy(), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            onnx_values, after_values.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_policy_and_value_clip_boundaries_match_known_objectives(self):
        new_log_probs = torch.log(
            torch.tensor([1.5, 0.5, 1.1, 0.9], requires_grad=True)
        )
        new_log_probs.retain_grad()
        old_log_probs = torch.zeros(4)
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

        policy_loss, ratio = PPOTrainer._clipped_policy_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            0.2,
        )
        self.assertTrue(
            torch.allclose(
                ratio,
                torch.tensor([1.5, 0.5, 1.1, 0.9]),
            )
        )
        self.assertAlmostEqual(policy_loss.item(), -0.15, places=6)
        policy_loss.backward()
        self.assertAlmostEqual(new_log_probs.grad[0].item(), 0.0, places=7)
        self.assertAlmostEqual(new_log_probs.grad[1].item(), 0.0, places=7)
        self.assertNotEqual(new_log_probs.grad[2].item(), 0.0)
        self.assertNotEqual(new_log_probs.grad[3].item(), 0.0)

        new_values = torch.tensor(
            [0.1, 0.5, -0.5], requires_grad=True
        )
        value_loss = PPOTrainer._clipped_value_loss(
            new_values,
            torch.zeros(3),
            torch.tensor([1.0, 1.0, -1.0]),
            0.2,
        )
        self.assertAlmostEqual(
            value_loss.item(),
            (0.81 + 0.64 + 0.64) / 3.0,
            places=6,
        )
        value_loss.backward()
        self.assertNotEqual(new_values.grad[0].item(), 0.0)
        self.assertAlmostEqual(new_values.grad[1].item(), 0.0, places=7)
        self.assertAlmostEqual(new_values.grad[2].item(), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
