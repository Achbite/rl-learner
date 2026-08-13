import tempfile
import unittest
from pathlib import Path

from proto import common_pb2, training_pb2
from src.contracts.identity import service_identity
from src.metrics.metric_events import (
    LocalMetricProjector,
    LocalTrainUpdateMetricWriter,
    MetricSchemaCatalog,
    RawMetricBatchStore,
    default_metric_schema_directory,
)
from tests.test_metric_event_store import batch_digest, contract


def agent(agent_id: int, *, success: bool) -> training_pb2.AgentEpisodeMetricFact:
    episode_return = 10.0 if success else -2.0
    transitions = 10
    return training_pb2.AgentEpisodeMetricFact(
        agent_id=agent_id,
        episode_return=episode_return,
        transition_count=transitions,
        success=success,
        termination_reason="GOAL_REACHED" if success else "TIME_LIMIT",
        shortest_action_steps=5 if success else 0,
        unique_cell_count=8,
        blocked_move_count=1,
        attempted_move_count=10,
        reward_components=[
            training_pb2.RawMetricSumCount(
                field_id="goal_reward",
                sum=episode_return,
                count=transitions,
            )
        ],
        behavior_model_version_min=3,
        behavior_model_version_max=4,
        behavior_model_lineage_id="lineage",
    )


def episode_batch(
    store: RawMetricBatchStore,
    source: common_pb2.ServiceInstanceIdentity,
    *,
    sequence: int,
    committed_at_unix_ms: int,
    successes: int = 2,
) -> training_pb2.MetricBatch:
    event = training_pb2.MetricEvent(
        contract=store.contract,
        schema_identity=store.catalog.schema_identity(),
        source=source,
        event_sequence=sequence,
        committed_at_unix_ms=committed_at_unix_ms,
        episode=training_pb2.EpisodeMetricFact(
            task_id="maze.fixed.single-map.v1",
            environment_instance_id="env-0",
            episode_id=f"episode-{sequence}",
            agents=[
                agent(agent_id, success=agent_id <= successes)
                for agent_id in range(1, 5)
            ],
        ),
    )
    batch = training_pb2.MetricBatch(
        contract=store.contract,
        schema_identity=store.catalog.schema_identity(),
        source=source,
        batch_sequence=sequence,
        created_at_unix_ms=committed_at_unix_ms + 1,
        first_event_sequence=sequence,
        last_event_sequence=sequence,
        events=[event],
        event_time_watermark_unix_ms=committed_at_unix_ms,
    )
    batch_digest(batch)
    return batch


class LocalMetricProjectorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "metric-events.sqlite3"
        self.catalog = MetricSchemaCatalog.load(
            default_metric_schema_directory()
        )
        self.store = RawMetricBatchStore(self.path, contract(), self.catalog)
        self.aiserver = service_identity("rl-aiserver", "aiserver-0", 7)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def commit(self, batch: training_pb2.MetricBatch) -> None:
        cursor = self.store.persist_batch("aiserver", batch)
        self.store.mark_acknowledged(batch, cursor)

    def test_projects_raw_episode_and_train_update_sufficient_statistics(self):
        self.commit(
            episode_batch(
                self.store,
                self.aiserver,
                sequence=1,
                committed_at_unix_ms=10_000,
            )
        )
        heartbeat = training_pb2.MetricBatch(
            contract=self.store.contract,
            schema_identity=self.store.catalog.schema_identity(),
            source=self.aiserver,
            batch_sequence=2,
            created_at_unix_ms=15_001,
            heartbeat=True,
            event_time_watermark_unix_ms=15_000,
        )
        batch_digest(heartbeat)
        self.commit(heartbeat)

        learner = service_identity("rl-learner", "learner-0", 11)
        writer = LocalTrainUpdateMetricWriter(self.store, learner)
        writer.append(
            training_pb2.TrainUpdateMetricFact(
                train_update_id="train-update-00000001",
                train_update_sequence=1,
                published_model=training_pb2.ModelIdentity(
                    model_lineage_id="lineage",
                    model_version=1,
                    artifact_digest=common_pb2.ContentDigest(
                        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
                        hex="4" * 64,
                    ),
                    manifest_digest=common_pb2.ContentDigest(
                        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
                        hex="5" * 64,
                    ),
                ),
                delivery_id="delivery-1",
                cumulative_trained_samples=512,
                actual_batch_size=512,
                behavior_model_version_min=0,
                behavior_model_version_max=0,
                behavior_model_lineage_id="lineage",
                ppo_statistics=[
                    training_pb2.RawMetricSumCount(
                        field_id="policy_loss", sum=-4.0, count=8
                    )
                ],
            ),
            committed_at_unix_ms=12_000,
        )

        projection = LocalMetricProjector(self.store).snapshot()
        window = projection["episodes"]["windows"]["100"]
        self.assertEqual(window["raw"]["environment_episode_count"], 1)
        self.assertEqual(window["raw"]["agent_episode_count"], 4)
        self.assertFalse(window["complete_window"])
        self.assertEqual(window["values"]["mean_agent_return"], 4.0)
        self.assertEqual(window["values"]["agent_success_rate"], 0.5)
        self.assertEqual(window["values"]["any_success_rate"], 1.0)
        self.assertEqual(window["values"]["all_success_rate"], 0.0)
        self.assertEqual(window["values"]["path_ratio_mean"], 2.0)
        self.assertEqual(window["values"]["blocked_move_rate"], 0.1)
        reward = window["values"]["reward_components"]["goal_reward"]
        self.assertEqual(reward["episode_mean"], 4.0)
        self.assertEqual(reward["transition_mean"], 0.4)
        self.assertEqual(
            projection["episodes"]["windows"]["5s"]["raw"][
                "environment_episode_count"
            ],
            1,
        )
        update = projection["train_updates"]["latest"]
        self.assertEqual(update["values"]["latest_model_version"], 1)
        self.assertEqual(update["values"]["ppo"]["policy_loss"]["mean"], -0.5)

    def test_exact_episode_windows_keep_last_25_and_100(self):
        for sequence in range(1, 102):
            self.commit(
                episode_batch(
                    self.store,
                    self.aiserver,
                    sequence=sequence,
                    committed_at_unix_ms=sequence * 5_000,
                    successes=sequence % 5,
                )
            )
        projection = LocalMetricProjector(self.store).snapshot()
        self.assertEqual(
            projection["episodes"]["windows"]["25"]["raw"][
                "environment_episode_count"
            ],
            25,
        )
        self.assertEqual(
            projection["episodes"]["windows"]["100"]["raw"][
                "environment_episode_count"
            ],
            100,
        )
        self.assertEqual(
            projection["episodes"]["windows"]["all"]["raw"][
                "environment_episode_count"
            ],
            101,
        )

    def test_rebuild_is_deterministic_and_no_data_is_omitted(self):
        empty = LocalMetricProjector(self.store).snapshot()
        self.assertEqual(
            empty["episodes"]["windows"]["100"]["status"], "no_data"
        )
        self.assertEqual(
            empty["episodes"]["windows"]["100"]["values"], {}
        )

        self.commit(
            episode_batch(
                self.store,
                self.aiserver,
                sequence=1,
                committed_at_unix_ms=10_000,
            )
        )
        first = LocalMetricProjector(self.store).snapshot()
        second = LocalMetricProjector(self.store).snapshot()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
