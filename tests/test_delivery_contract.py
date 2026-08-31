import unittest
from pathlib import Path

from main.training_runtime import train_processed_delivery
from proto import training_pb2
from src.training.ppo_trainer import PPOTrainer


def _trainer_config() -> dict:
    return {
        "contract": {
            "training_contract_path": str(
                Path(__file__).resolve().parents[1]
                / "schemas"
                / "training-contract.json"
            )
        },
        "training": {
            "device": "cpu",
            "seed": 0,
            "learning_rate": 0.0003,
            "clip_epsilon": 0.2,
            "value_clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 1,
            "train_batch_size": 2,
            "mini_batch_size": 2,
            "normalize_advantage": True,
        },
    }


class LearnerDevelopmentTest(unittest.TestCase):
    @staticmethod
    def _transition(index: int) -> training_pb2.ProcessedTransition:
        return training_pb2.ProcessedTransition(
            item_id=f"item-{index}",
            observation=[float(index)] * 17,
            action=index,
            behavior_log_probability=-0.5 - index,
            behavior_value=0.2 + 0.1 * index,
            advantage=-0.18515 if index == 0 else -0.3,
            value_target=0.01485 if index == 0 else 0.0,
            behavior_model_step=0,
            created_at_unix_ms=1700000000000 + index,
        )

    def test_processed_transitions_reach_real_trainer(self):
        response = training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="delivery-test",
        )
        for index in range(2):
            response.items.add(
                transition=self._transition(index),
                insert_sequence=index + 1,
                inserted_at_unix_ms=1700000000100 + index,
                draw_count=1,
            )

        trainer = PPOTrainer(_trainer_config())
        batch, stats = train_processed_delivery(response.items, trainer)

        self.assertEqual(len(batch), 2)
        self.assertEqual(stats["sample_evaluation_count"], 2)
        self.assertEqual(
            [sample["item_id"] for sample in batch], ["item-0", "item-1"]
        )
        self.assertEqual([sample["action"] for sample in batch], [0, 1])
        self.assertEqual(
            [sample["behavior_model_step"] for sample in batch], [0, 0]
        )
        for index, sample in enumerate(batch):
            transition = response.items[index].transition
            self.assertEqual(sample["observation"], list(transition.observation))
            self.assertEqual(
                sample["old_log_probability"],
                float(transition.behavior_log_probability),
            )
            self.assertEqual(
                sample["old_value_prediction"],
                float(transition.behavior_value),
            )
            self.assertEqual(sample["advantage"], float(transition.advantage))
            self.assertEqual(
                sample["value_target"], float(transition.value_target)
            )
