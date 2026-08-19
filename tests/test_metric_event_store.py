import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from proto import common_pb2, training_pb2
from src.contracts.identity import service_identity
from src.metrics.metric_events import (
    LocalTrainUpdateMetricWriter,
    MetricEventContractError,
    MetricSchemaCatalog,
    RawMetricBatchStore,
    cursor_for_batch,
    default_metric_schema_directory,
)


def digest(value: str) -> common_pb2.ContentDigest:
    return common_pb2.ContentDigest(
        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
        hex=value * 64,
    )


def contract() -> common_pb2.ContractIdentity:
    return common_pb2.ContractIdentity(
        package_name="rl-contracts",
        package_version="0.13.0",
        source_digest=digest("1"),
        artifact_digest=digest("2"),
        platform="linux/arm64",
        generator_identity="3" * 64,
    )


def batch_digest(batch: training_pb2.MetricBatch) -> None:
    canonical = training_pb2.MetricBatch()
    canonical.CopyFrom(batch)
    canonical.ClearField("batch_digest")
    batch.batch_digest.CopyFrom(
        common_pb2.ContentDigest(
            algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
            hex=hashlib.sha256(
                canonical.SerializeToString(deterministic=True)
            ).hexdigest(),
        )
    )


def episode_batch(
    store: RawMetricBatchStore,
    source: common_pb2.ServiceInstanceIdentity,
    *,
    final: bool = False,
) -> training_pb2.MetricBatch:
    event = training_pb2.MetricEvent(
        contract=store.contract,
        schema_identity=store.catalog.schema_identity(),
        source=source,
        event_sequence=1,
        committed_at_unix_ms=1_000,
        episode=training_pb2.EpisodeMetricFact(
            environment_instance_id="env-0",
            episode_id="episode-1",
            agents=[
                training_pb2.AgentEpisodeMetricFact(
                    agent_id=1,
                    episode_return=0.0,
                    transition_count=2,
                    success=False,
                    termination_reason="CHAIN_TEST_COMPLETE",
                    minimum_behavior_model_step=3,
                    maximum_behavior_model_step=4,
                    behavior_model_lineage_id="lineage",
                )
            ],
        ),
    )
    batch = training_pb2.MetricBatch(
        contract=store.contract,
        schema_identity=store.catalog.schema_identity(),
        source=source,
        batch_sequence=1,
        created_at_unix_ms=2_000,
        first_event_sequence=1,
        last_event_sequence=1,
        events=[event],
        source_final=final,
        final_event_sequence=1 if final else 0,
        event_time_watermark_unix_ms=1_000,
    )
    batch_digest(batch)
    return batch


def train_update_fact(sequence: int) -> training_pb2.TrainUpdateMetricFact:
    return training_pb2.TrainUpdateMetricFact(
        train_update_id=f"train-update-{sequence:08d}",
        train_update_sequence=sequence,
        published_model=training_pb2.ModelIdentity(
            model_lineage_id="lineage",
            model_step=sequence,
            artifact_digest=digest("4"),
            manifest_digest=digest("5"),
        ),
        delivery_id=f"delivery-{sequence}",
        cumulative_trained_samples=sequence * 512,
        actual_batch_size=512,
        minimum_behavior_model_step=max(0, sequence - 2),
        maximum_behavior_model_step=max(0, sequence - 1),
        behavior_model_lineage_id="lineage",
    )


class RawMetricBatchStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "metric-events.sqlite3"
        self.catalog = MetricSchemaCatalog.load(
            default_metric_schema_directory()
        )
        self.source = service_identity("rl-aiserver", "aiserver-0", 7)

    def tearDown(self):
        self.temporary.cleanup()

    def open(self) -> RawMetricBatchStore:
        return RawMetricBatchStore(self.path, contract(), self.catalog)

    def test_raw_bytes_are_durable_before_ack_and_pending_recovers(self):
        store = self.open()
        batch = episode_batch(store, self.source)
        cursor = store.persist_batch("aiserver", batch)
        self.assertEqual(store.snapshot()["pending_batch_count"], 1)
        store.close()

        recovered = self.open()
        self.assertEqual(recovered.pending_batch(self.source), batch)
        self.assertEqual(recovered.pending_cursor(self.source), cursor)
        recovered.mark_acknowledged(batch, cursor)
        self.assertEqual(recovered.committed_cursor(self.source), cursor)
        snapshot = recovered.snapshot()
        self.assertEqual(snapshot["pending_batch_count"], 0)
        self.assertEqual(snapshot["committed_batch_count"], 1)
        recovered.close()

    def test_conflicting_replay_is_rejected_without_moving_cursor(self):
        store = self.open()
        batch = episode_batch(store, self.source)
        cursor = store.persist_batch("aiserver", batch)
        conflicting = training_pb2.MetricBatch()
        conflicting.CopyFrom(batch)
        conflicting.created_at_unix_ms += 1
        batch_digest(conflicting)
        with self.assertRaisesRegex(
            MetricEventContractError, "unacknowledged batch"
        ):
            store.persist_batch("aiserver", conflicting)
        self.assertEqual(store.pending_cursor(self.source), cursor)
        store.close()

    def test_zero_event_final_heartbeat_commits_event_cursor_zero(self):
        store = self.open()
        store.activate_source("aiserver", self.source)
        initial = store.committed_cursor(self.source)
        heartbeat = training_pb2.MetricBatch(
            contract=store.contract,
            schema_identity=store.catalog.schema_identity(),
            source=self.source,
            batch_sequence=1,
            created_at_unix_ms=2_000,
            heartbeat=True,
            source_final=True,
            final_event_sequence=0,
            event_time_watermark_unix_ms=2_000,
        )
        batch_digest(heartbeat)
        expected = cursor_for_batch(heartbeat, initial)
        actual = store.persist_batch("aiserver", heartbeat)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.acknowledged_event_sequence, 0)
        store.mark_acknowledged(heartbeat, actual)
        self.assertTrue(store.is_final(self.source))
        store.close()

    def test_source_replacement_marks_unfinalized_source_incomplete(self):
        store = self.open()
        store.activate_source("aiserver", self.source)
        replacement = service_identity("rl-aiserver", "aiserver-1", 8)
        store.activate_source("aiserver", replacement)
        sources = store.snapshot()["sources"]
        original = next(
            source for source in sources if source["instance_id"] == "aiserver-0"
        )
        self.assertTrue(original["incomplete"])
        self.assertEqual(
            original["incomplete_reason"],
            "source_replaced_before_final",
        )
        store.close()

    def test_event_time_watermark_cannot_regress(self):
        store = self.open()
        first = episode_batch(store, self.source)
        cursor = store.persist_batch("aiserver", first)
        store.mark_acknowledged(first, cursor)
        heartbeat = training_pb2.MetricBatch(
            contract=store.contract,
            schema_identity=store.catalog.schema_identity(),
            source=self.source,
            batch_sequence=2,
            created_at_unix_ms=3_000,
            heartbeat=True,
            event_time_watermark_unix_ms=999,
        )
        batch_digest(heartbeat)
        with self.assertRaisesRegex(
            MetricEventContractError, "watermark regressed"
        ):
            store.persist_batch("aiserver", heartbeat)
        self.assertEqual(
            store.committed_cursor(self.source).acknowledged_batch_sequence,
            1,
        )
        store.close()

    def test_local_writer_commits_exact_update_and_final_heartbeat(self):
        store = self.open()
        learner = service_identity("learner", "learner-0", 1)
        writer = LocalTrainUpdateMetricWriter(
            store, learner, initial_train_update_sequence=40
        )
        writer.append(
            training_pb2.TrainUpdateMetricFact(
                train_update_id="train-update-00000041",
                train_update_sequence=41,
                published_model=training_pb2.ModelIdentity(
                    model_lineage_id="lineage",
                    model_step=41,
                    artifact_digest=digest("4"),
                    manifest_digest=digest("5"),
                ),
                delivery_id="delivery-41",
                cumulative_trained_samples=20_992,
                actual_batch_size=512,
                minimum_behavior_model_step=39,
                maximum_behavior_model_step=40,
                behavior_model_lineage_id="lineage",
            ),
            committed_at_unix_ms=4_000,
        )
        writer.finalize()
        cursor = store.committed_cursor(learner)
        self.assertEqual(cursor.acknowledged_batch_sequence, 2)
        self.assertEqual(cursor.acknowledged_event_sequence, 1)
        self.assertTrue(store.is_final(learner))
        store.close()

    def test_local_writer_clamps_wall_clock_rollback(self):
        store = self.open()
        learner = service_identity("learner", "learner-0", 1)
        writer = LocalTrainUpdateMetricWriter(
            store, learner, initial_train_update_sequence=40
        )
        writer.append(train_update_fact(41), committed_at_unix_ms=4_000)
        writer.append(train_update_fact(42), committed_at_unix_ms=3_990)

        batches = [
            batch
            for _, role, _, batch in store.committed_batches_after(0)
            if role == "learner"
        ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[1].events[0].committed_at_unix_ms, 4_000)
        self.assertEqual(batches[1].event_time_watermark_unix_ms, 4_000)
        self.assertEqual(
            store.committed_cursor(learner).acknowledged_event_sequence, 2
        )
        with mock.patch(
            "src.metrics.metric_events.time.time", return_value=3.990
        ):
            writer.finalize()
        final_batch = store.committed_batches_after(0)[-1][3]
        self.assertTrue(final_batch.source_final)
        self.assertEqual(final_batch.event_time_watermark_unix_ms, 4_000)
        store.close()

    def test_local_writer_records_gap_and_continues_after_missing_update(self):
        store = self.open()
        learner = service_identity("learner", "learner-0", 1)
        writer = LocalTrainUpdateMetricWriter(
            store, learner, initial_train_update_sequence=40
        )
        writer.append(train_update_fact(41), committed_at_unix_ms=4_000)
        writer.append(train_update_fact(43), committed_at_unix_ms=5_000)

        batches = [
            batch
            for _, role, _, batch in store.committed_batches_after(0)
            if role == "learner"
        ]
        self.assertEqual(len(batches), 3)
        self.assertTrue(batches[1].HasField("gap"))
        self.assertEqual(
            batches[1].gap.first_unavailable_event_sequence, 2
        )
        self.assertEqual(
            batches[1].gap.last_unavailable_event_sequence, 2
        )
        self.assertEqual(
            batches[1].gap.oldest_available_event_sequence, 3
        )
        self.assertEqual(batches[2].events[0].event_sequence, 3)
        self.assertEqual(
            batches[2].events[0].train_update.train_update_sequence, 43
        )
        snapshot = store.snapshot()
        source = next(
            item for item in snapshot["sources"] if item["role"] == "learner"
        )
        self.assertTrue(source["incomplete"])
        self.assertEqual(source["incomplete_reason"], "sequence_gap:2-2")
        store.close()
