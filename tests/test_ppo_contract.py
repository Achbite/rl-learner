import math
import unittest

import torch

from src.training.ppo_trainer import PPOTrainer


class LearnerDevelopmentTest(unittest.TestCase):
    def test_fixed_processed_transitions_match_ppo_loss(self):
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
