import tempfile
import unittest
from pathlib import Path

from proto import common_pb2, maze_metrics_pb2, training_metrics_pb2, training_pb2
from src.metrics.metric_events import (
    LocalMetricProjector,
    LocalTrainUpdateMetricWriter,
    RawMetricBatchStore,
    _episode_event_statistics,
    _render_episode_statistics,
    _render_train_statistics,
    _train_event_statistics,
)


class LearnerMetricCalculationTest(unittest.TestCase):
    def test_episode_metrics_are_derived_from_raw_agent_facts(self):
        fact = maze_metrics_pb2.EpisodeMetricFact(
            environment_instance_id="environment-fixed",
            episode_id="episode-fixed",
        )
        successful = fact.agents.add(
            agent_id=0,
            episode_return=3.0,
            transition_count=4,
            success=True,
            termination_reason="GOAL_REACHED",
            shortest_action_steps=2,
            unique_cell_count=3,
            blocked_move_count=1,
            attempted_move_count=4,
            minimum_behavior_model_step=1,
            maximum_behavior_model_step=2,
            behavior_model_lineage_id="lineage-fixed",
        )
        successful.reward_components.add(
            field_id="total_reward", sum=3.0, count=4
        )
        unsuccessful = fact.agents.add(
            agent_id=1,
            episode_return=-1.0,
            transition_count=2,
            success=False,
            termination_reason="TIME_LIMIT",
            shortest_action_steps=2,
            unique_cell_count=2,
            blocked_move_count=0,
            attempted_move_count=2,
            minimum_behavior_model_step=0,
            maximum_behavior_model_step=1,
            behavior_model_lineage_id="lineage-fixed",
        )
        unsuccessful.reward_components.add(
            field_id="total_reward", sum=-1.0, count=2
        )

        raw = _episode_event_statistics(fact)
        rendered = _render_episode_statistics(
            raw, status="complete", window_kind="all"
        )

        self.assertEqual(raw["environment_episode_count"], 1)
        self.assertEqual(raw["agent_episode_count"], 2)
        self.assertEqual(raw["successful_agent_count"], 1)
        self.assertEqual(raw["return_sum"], 2.0)
        self.assertEqual(raw["return_count"], 2)
        self.assertEqual(raw["minimum_behavior_model_step"], 0)
        self.assertEqual(raw["maximum_behavior_model_step"], 2)
        values = rendered["values"]
        self.assertEqual(values["mean_agent_return"], 1.0)
        self.assertEqual(values["min_agent_return"], -1.0)
        self.assertEqual(values["max_agent_return"], 3.0)
        self.assertEqual(values["agent_success_rate"], 0.5)
        self.assertEqual(values["any_success_rate"], 1.0)
        self.assertEqual(values["all_success_rate"], 0.0)
        self.assertEqual(values["mean_episode_step"], 3.0)
        self.assertEqual(values["mean_unique_cells"], 2.5)
        self.assertAlmostEqual(values["blocked_move_rate"], 1.0 / 6.0)
        self.assertEqual(values["path_ratio_mean"], 2.0)
        self.assertEqual(
            values["reward_components"]["total_reward"]["episode_mean"],
            1.0,
        )
        self.assertAlmostEqual(
            values["reward_components"]["total_reward"][
                "transition_mean"
            ],
            1.0 / 3.0,
        )

    def test_train_metrics_are_derived_from_raw_sum_counts(self):
        fact = training_metrics_pb2.TrainUpdateMetricFact(
            train_update_id="train-update-00000003",
            train_update_sequence=3,
            delivery_id="delivery-fixed",
            cumulative_trained_samples=10,
            actual_batch_size=2,
            minimum_behavior_model_step=0,
            maximum_behavior_model_step=2,
            behavior_model_lineage_id="lineage-fixed",
        )
        fact.published_model.model_lineage_id = "lineage-fixed"
        fact.published_model.model_step = 4
        fact.ppo_statistics.add(field_id="policy_loss", sum=-0.4, count=2)
        fact.ppo_statistics.add(field_id="value_loss", sum=1.28, count=2)

        raw = _train_event_statistics(fact)
        rendered = _render_train_statistics(
            raw, status="complete", window_kind="all"
        )

        self.assertEqual(raw["train_update_count"], 1)
        self.assertEqual(raw["actual_batch_size_sum"], 2)
        self.assertEqual(raw["latest_train_update_sequence"], 3)
        self.assertEqual(raw["latest_model_step"], 4)
        self.assertEqual(raw["latest_cumulative_trained_samples"], 10)
        self.assertEqual(
            rendered["values"]["ppo"]["policy_loss"]["mean"], -0.2
        )
        self.assertEqual(
            rendered["values"]["ppo"]["value_loss"]["mean"], 0.64
        )

    def test_metric_payloads_follow_typed_transport(self):
        learner = common_pb2.ServiceInstanceIdentity(
            component="learner",
            instance_id="learner-metric-test",
            lifecycle_epoch=1,
        )
        aiserver = common_pb2.ServiceInstanceIdentity(
            component="aiserver",
            instance_id="aiserver-metric-test",
            lifecycle_epoch=1,
        )
        train_fact = training_metrics_pb2.TrainUpdateMetricFact(
            train_update_id="train-update-1",
            train_update_sequence=2,
            published_model=training_pb2.ModelIdentity(
                model_lineage_id="lineage-test",
                model_step=1,
            ),
            delivery_id="delivery-test",
            cumulative_trained_samples=2,
            actual_batch_size=2,
            minimum_behavior_model_step=0,
            maximum_behavior_model_step=0,
            behavior_model_lineage_id="lineage-test",
        )
        train_fact.ppo_statistics.add(
            field_id="policy_loss", sum=-0.4, count=2
        )
        episode_fact = maze_metrics_pb2.EpisodeMetricFact(
            environment_instance_id="environment-test",
            episode_id="episode-test",
        )
        agent = episode_fact.agents.add(
            agent_id=0,
            episode_return=1.0,
            transition_count=2,
            success=True,
            termination_reason="completed",
            shortest_action_steps=1,
            unique_cell_count=2,
            blocked_move_count=0,
            attempted_move_count=2,
            minimum_behavior_model_step=0,
            maximum_behavior_model_step=0,
            behavior_model_lineage_id="lineage-test",
        )
        agent.reward_components.add(
            field_id="goal_reward", sum=1.0, count=2
        )

        with tempfile.TemporaryDirectory() as directory:
            store = RawMetricBatchStore(Path(directory) / "metrics.sqlite3")
            try:
                writer = LocalTrainUpdateMetricWriter(store, learner)
                train_bytes = train_fact.SerializeToString(deterministic=True)
                writer.append(train_fact, observed_at_unix_ms=1700000000000)
                learner_cursor = store.committed_cursor(learner)
                self.assertEqual(learner_cursor.acknowledged_batch_sequence, 2)
                self.assertEqual(learner_cursor.acknowledged_event_sequence, 2)

                episode_bytes = episode_fact.SerializeToString(
                    deterministic=True
                )
                episode_batch = training_pb2.MetricBatch(
                    source=aiserver,
                    batch_sequence=1,
                    created_at_unix_ms=1700000000001,
                    first_event_sequence=1,
                    last_event_sequence=1,
                    events=[
                        training_pb2.MetricEvent(
                            event_sequence=1,
                            observed_at_unix_ms=1700000000001,
                            fact_payload=episode_bytes,
                            fact_kind=(
                                training_pb2.METRIC_FACT_KIND_MAZE_EPISODE
                            ),
                        )
                    ],
                )
                cursor = store.persist_batch("aiserver", episode_batch)
                self.assertEqual(cursor.acknowledged_batch_sequence, 1)
                self.assertEqual(cursor.acknowledged_event_sequence, 1)
                store.mark_acknowledged(episode_batch, cursor)
                writer.finalize()

                batches = store.committed_batches_after(0)
                payloads = {
                    int(batch.events[0].fact_kind): batch.events[0].fact_payload
                    for _, _, _, batch in batches
                    if batch.events
                }
                self.assertEqual(
                    payloads[training_pb2.METRIC_FACT_KIND_TRAIN_UPDATE],
                    train_bytes,
                )
                self.assertEqual(
                    payloads[training_pb2.METRIC_FACT_KIND_MAZE_EPISODE],
                    episode_bytes,
                )
                learner_batches = [
                    batch
                    for _, role, _, batch in batches
                    if role == "learner"
                ]
                self.assertEqual(len(learner_batches), 3)
                self.assertTrue(learner_batches[0].HasField("gap"))
                self.assertEqual(
                    learner_batches[0].gap.first_unavailable_event_sequence,
                    1,
                )
                self.assertEqual(
                    learner_batches[0].gap.last_unavailable_event_sequence,
                    1,
                )
                self.assertTrue(learner_batches[-1].source_final)
                self.assertTrue(learner_batches[-1].heartbeat)
                self.assertEqual(learner_batches[-1].final_event_sequence, 2)
                final_cursor = store.committed_cursor(learner)
                self.assertEqual(final_cursor.acknowledged_batch_sequence, 3)
                self.assertEqual(final_cursor.acknowledged_event_sequence, 2)
                snapshot = LocalMetricProjector(store).snapshot()
                self.assertEqual(
                    snapshot["train_updates"]["windows"]["all"]
                    ["raw"]["train_update_count"],
                    1,
                )
                self.assertEqual(
                    snapshot["episodes"]["windows"]["all"]
                    ["raw"]["environment_episode_count"],
                    1,
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
