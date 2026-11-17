import math
import copy
import threading
import unittest

import torch

from src.training.ppo_trainer import PPOTrainer
from tests.test_delivery_contract import _trainer_config


class PPOCalculationTest(unittest.TestCase):
    def test_parallel_batch_matches_one_full_batch_update(self):
        config = _trainer_config()
        config["policy"]["action_mask_mode"] = "disabled"
        config["training"].update(
            train_batch_size=4096, mini_batch_size=512, update_mode="parallel_batch",
        )
        parallel = PPOTrainer(config)
        full_config = copy.deepcopy(config)
        full_config["training"].update(mini_batch_size=4096, update_mode="sequential_minibatch")
        full = PPOTrainer(full_config)
        initial = copy.deepcopy(parallel._model.state_dict())
        obs = torch.randn(4096, 5)
        obs[:, 0] = torch.arange(4096) / 4096
        actions = torch.arange(4096) % 4
        advantages = torch.randn(4096)
        with torch.no_grad():
            old_logs, old_values, _ = parallel._model.evaluate_actions(obs, actions)
        samples = [dict(
            item_id=f"parallel-{index}", observation=obs[index].tolist(), action=int(actions[index]),
            old_log_probability=float(old_logs[index]), old_value_prediction=float(old_values[index]),
            advantage=float(advantages[index]), value_target=float(old_values[index] + advantages[index]),
            behavior_model_step=0, action_mask=[],
        ) for index in range(4096)]
        unchanged_samples = copy.deepcopy(samples)
        barrier = threading.Barrier(8, timeout=15)
        observations = []
        lock = threading.Lock()
        evaluate_actions = parallel._model.evaluate_actions

        def observe(shard_obs, shard_actions, masks=None):
            if torch.is_grad_enabled():
                with lock:
                    observations.append((threading.get_ident(), (shard_obs[:, 0] * 4096).tolist()))
                barrier.wait()
                for key, value in parallel._model.state_dict().items():
                    torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
            return evaluate_actions(shard_obs, shard_actions, masks)

        parallel._model.evaluate_actions = observe
        actual = parallel.train_on_batch(samples)
        expected = full.train_on_batch(samples)
        self.assertEqual(len(observations), 8)
        self.assertEqual(len({thread for thread, _ in observations}), 8)
        self.assertTrue(all(len(ids) == 512 for _, ids in observations))
        self.assertEqual(sorted(index for _, ids in observations for index in ids), list(range(4096)))
        self.assertEqual(samples, unchanged_samples)
        self.assertEqual(actual["sample_evaluation_count"], 4096)
        self.assertEqual(actual["optimizer_step_count"], 1)
        self.assertEqual(actual["gradient_worker_count"], 8)
        self.assertEqual(actual["model_step"], 1)
        self.assertAlmostEqual(actual["importance_ratio_sum"] / 4096, 1.0, places=6)
        self.assertAlmostEqual(actual["approx_kl"], 0.0, places=6)
        self.assertEqual(actual["clip_fraction"], 0.0)
        self.assertGreater(actual["post_update_approx_kl"], 0.0)
        self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in parallel._model.state_dict().items()))
        for key, value in parallel._model.state_dict().items():
            torch.testing.assert_close(value, full._model.state_dict()[key], rtol=1e-5, atol=1e-6)
        for key, state in parallel._optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, full._optimizer.state_dict()["state"][key][name], rtol=1e-4, atol=1e-8)
        for name in ("policy_loss", "value_loss", "entropy", "gradient_norm", "total_loss", "post_update_approx_kl"):
            self.assertAlmostEqual(actual[name], expected[name], places=6)

    def test_clipped_ppo_loss_matches_reference_values(self):
        old_log_probability = torch.tensor([0.0, 0.0])
        new_log_probability = torch.tensor(
            [math.log(1.3), math.log(0.7)]
        )
        advantage = torch.tensor([1.0, -1.0])

        policy_loss, ratio = PPOTrainer._clipped_policy_loss(
            new_log_probability,
            old_log_probability,
            advantage,
            0.2,
        )
        clip_fraction = ((ratio - 1.0).abs() > 0.2).float().mean()

        old_value_prediction = torch.tensor([0.0, 0.0])
        new_value_prediction = torch.tensor([0.5, -0.5])
        value_target = torch.tensor([1.0, -1.0])
        value_loss = PPOTrainer._clipped_value_loss(
            new_value_prediction,
            old_value_prediction,
            value_target,
            0.2,
        )

        torch.testing.assert_close(ratio, torch.tensor([1.3, 0.7]))
        self.assertAlmostEqual(policy_loss.item(), -0.2, places=6)
        self.assertAlmostEqual(clip_fraction.item(), 1.0, places=6)
        self.assertAlmostEqual(value_loss.item(), 0.64, places=6)
