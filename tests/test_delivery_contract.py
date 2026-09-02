import unittest
from pathlib import Path

from main.training_runtime import train_processed_delivery
from proto import training_pb2
from src.config.effective_config import load_effective_config
from src.contracts.identity import validate_config
from src.training.ppo_trainer import PPOTrainer


def _trainer_config() -> dict:
    return {
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
        "model": {
            "observation_dimension": 5,
            "action_count": 4,
            "hidden_dimension": 8,
        },
        "policy": {"action_mask_mode": "required"},
    }


class LearnerDevelopmentTest(unittest.TestCase):
    @staticmethod
    def _transition(
        index: int,
        observation_dimension: int,
        action_count: int,
    ) -> training_pb2.ProcessedTransition:
        action_mask = [False] * action_count
        action_mask[index] = True
        return training_pb2.ProcessedTransition(
            item_id=f"item-{index}",
            observation=[float(index)] * observation_dimension,
            action=index,
            behavior_log_probability=-0.5 - index,
            behavior_value=0.2 + 0.1 * index,
            advantage=-0.18515 if index == 0 else -0.3,
            value_target=0.01485 if index == 0 else 0.0,
            behavior_model_step=0,
            created_at_unix_ms=1700000000000 + index,
            action_mask=action_mask,
        )

    def test_processed_transition_data_reaches_real_trainer(self):
        config = _trainer_config()
        response = training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="delivery-test",
        )
        for index in range(2):
            response.items.add(
                transition=self._transition(
                    index,
                    config["model"]["observation_dimension"],
                    config["model"]["action_count"],
                ),
                insert_sequence=index + 1,
                inserted_at_unix_ms=1700000000100 + index,
                draw_count=1,
            )

        trainer = PPOTrainer(config)
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
            self.assertEqual(
                sample["action_mask"], list(transition.action_mask)
            )
            self.assertEqual(sample["advantage"], float(transition.advantage))
            self.assertEqual(
                sample["value_target"], float(transition.value_target)
            )

    def test_local_effective_config_reaches_runtime_validation(self):
        repository = Path(__file__).resolve().parents[1]
        config = load_effective_config(
            str(repository / "configs" / "learner_config.yaml"),
            environment={
                "RL_MODEL_LINEAGE_ID": "maze-model-local-config-test",
                "RL_PPO_TRAIN_BATCH_SIZE": "32",
                "RL_PPO_MINI_BATCH_SIZE": "16",
                "RL_PPO_N_EPOCHS": "1",
                "RL_PPO_TMAX": "16",
            },
        )

        validate_config(config)
        self.assertEqual(config["training"]["train_batch_size"], 32)
        self.assertEqual(config["training"]["mini_batch_size"], 16)
        self.assertEqual(config["training"]["n_epochs"], 1)
        self.assertEqual(config["training"]["tmax"], 16)
