import unittest
from types import SimpleNamespace

from main.training_runtime import TrainingRuntime
from proto import common_pb2, training_pb2


def _digest(hex_value: str) -> common_pb2.ContentDigest:
    return common_pb2.ContentDigest(
        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
        hex=hex_value,
    )


class LearnerDevelopmentTest(unittest.TestCase):
    def _runtime(self) -> TrainingRuntime:
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.semantics = training_pb2.TrainingSemanticsIdentity(
            policy_distribution_schema_id="categorical.logits.v1"
        )
        runtime.policy_digest = _digest("c" * 64)
        runtime.rollout_profile = SimpleNamespace(
            profile_digest=_digest("d" * 64)
        )
        runtime.publisher = SimpleNamespace(
            lineage_id="lineage-fixed",
            obs_dim=17,
            action_dim=9,
        )
        runtime.trainer = SimpleNamespace(model_step=0)
        runtime.train_batch_size = 2
        return runtime

    @staticmethod
    def _transition(
        runtime: TrainingRuntime, index: int
    ) -> training_pb2.ProcessedTransition:
        terminal = index == 1
        transition = training_pb2.ProcessedTransition(
            item_id=f"item-{index}",
            environment_session_id="environment-session-fixed",
            episode_id="episode-fixed",
            agent_id=1,
            segment_id="segment-fixed",
            transition_index=index,
            segment_transition_count=2,
            action_step=index,
            observation=[float(index)] * 17,
            next_observation=[float(index + 1)] * 17,
            action=index,
            reward=0.0,
            behavior_log_probability=-0.5 - index,
            behavior_value=0.2 + 0.1 * index,
            advantage=-0.18515 if index == 0 else -0.3,
            value_target=0.01485 if index == 0 else 0.0,
            environment_terminal=terminal,
            end_kind=(
                training_pb2.TRANSITION_END_KIND_ENVIRONMENT_TERMINATED
                if terminal
                else training_pb2.TRANSITION_END_KIND_CONTINUING
            ),
            segment_close_reason=(
                training_pb2.SEGMENT_CLOSE_REASON_GOAL
                if terminal
                else training_pb2.SEGMENT_CLOSE_REASON_UNSPECIFIED
            ),
            segment_boundary=terminal,
            bootstrap_applied=False,
            behavior_policy=training_pb2.BehaviorPolicyReference(
                model_lineage_id=runtime.publisher.lineage_id,
                model_step=0,
                distribution_schema_id=(
                    runtime.semantics.policy_distribution_schema_id
                ),
                policy_spec_digest=runtime.policy_digest,
                artifact_digest=_digest("a" * 64),
                manifest_digest=_digest("b" * 64),
            ),
            rollout_estimator_profile_digest=(
                runtime.rollout_profile.profile_digest
            ),
            created_at_unix_ms=1700000000000 + index,
        )
        if terminal:
            transition.bootstrap_value = 0.0
        return transition

    def test_fixed_processed_transitions_enter_training_batch(self):
        runtime = self._runtime()
        response = training_pb2.GetBatchRsp(
            ret_code=0,
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="delivery-fixed",
            returned_transitions=2,
            actual_transition_count=2,
            leased_transitions=2,
            minimum_behavior_model_step=0,
            maximum_behavior_model_step=0,
            oldest_transition_created_at_unix_ms=1700000000000,
            newest_transition_created_at_unix_ms=1700000000001,
        )
        for index in range(2):
            response.items.add(
                transition=self._transition(runtime, index),
                insert_sequence=index + 1,
                inserted_at_unix_ms=1700000000100 + index,
                draw_count=1,
            )

        summary = runtime._validate_delivery(response)
        batch = runtime._training_samples(response.items)

        self.assertEqual(summary["minimum_model_step"], 0)
        self.assertEqual(summary["maximum_model_step"], 0)
        self.assertEqual(summary["model_lineage_id"], "lineage-fixed")
        self.assertEqual(
            summary["models"],
            [
                {
                    "model_lineage_id": "lineage-fixed",
                    "model_step": 0,
                    "artifact_digest": "a" * 64,
                    "manifest_digest": "b" * 64,
                }
            ],
        )
        self.assertEqual(len(batch), 2)
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
                sample["next_observation"],
                list(transition.next_observation),
            )
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
