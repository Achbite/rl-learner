import logging
import grpc
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from http.server import HTTPServer
from urllib.request import urlopen
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from proto.common import identity_pb2 as common_pb2
from proto.metrics import training_pb2 as training_metrics_pb2
from proto.training import training_pb2 as wire
from proto.training import training_pb2_grpc as rpc
from proto.metrics import catalog_pb2 as metric_catalog_pb2
from proto.metrics import catalog_pb2_grpc as metric_catalog_pb2_grpc
from proto.metrics import registry_pb2 as metric_registry_pb2
from proto.metrics import transport_pb2 as metric_transport_pb2
from proto.metrics import transport_pb2_grpc as metric_transport_pb2_grpc
from proto.training import model_identity_pb2
from src.metrics.metric_events import (
    MetricEventCollector, LocalMetricProjector, LocalTrainUpdateMetricWriter,
    MetricEventContractError, RawMetricBatchStore,
)
from src.metrics.registered_metrics import MetricRegistry
from src.metrics.learner_status import LearnerStatusService, MetricCatalogService
from src.metrics.metric_events import LearnerMetricEventService
from src.metrics.metrics_backend import JsonlBackend
from tools import metrics_server
from tools.metrics_server import MetricsFileReader, project_metric_values
from main.training_runtime import TrainingRuntime
from src.preview.topology import training_chain_status
from src.preview.collector import PreviewCollector


def source(instance="server-a"):
    return common_pb2.ServiceInstanceIdentity(
        component="aiserver", instance_id=instance, lifecycle_epoch=1,
    )


def score_registry(unit="reward", category="custom"):
    registry = MetricRegistry()
    registry.register("task.balance.score", display_name="Balance score", unit=unit,
                      scope="episode", value_type=metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT,
                      aggregation=metric_registry_pb2.METRIC_AGGREGATION_MEAN, denominator="transition", category=category)
    return registry


def score_record(registry, value, count):
    return registry.record([metric_registry_pb2.MetricPoint(
        metric_id="task.balance.score", sum_count=metric_registry_pb2.MetricSumCount(sum=value, count=count),
    )])


def persist(store, producer, sequence, payload, kind=metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS):
    batch = metric_transport_pb2.MetricBatch(
        source=producer, batch_sequence=sequence, created_at_unix_ms=1700000000000 + sequence,
        first_event_sequence=sequence, last_event_sequence=sequence,
        events=[metric_transport_pb2.MetricEvent(
            event_sequence=sequence, observed_at_unix_ms=1700000000000 + sequence,
            fact_kind=kind, fact_payload=payload,
        )],
    )
    cursor = store.persist_batch("aiserver", batch)
    store.mark_acknowledged(batch, cursor)
    return batch


class LearnerMetricCalculationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = RawMetricBatchStore(Path(self.directory.name) / "metrics.sqlite3")
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(self.store.close)

    def test_preview_catalog_failure_preserves_status_and_throughput(self):
        pool_source = common_pb2.ServiceInstanceIdentity(
            component="sample-pool", instance_id="pool-metrics", lifecycle_epoch=1)
        status = wire.SamplePoolStatusRsp(sample_pool=pool_source, ready=True,
            timestamp_unix_ms=1700000000000, accepted_unique_transitions=100)
        registry = MetricRegistry()
        registry.register("sample.flow.accepted.total", display_name="Accepted Samples",
            unit="count", category="sample_flow", scope="sample_pool",
            value_type=metric_registry_pb2.METRIC_VALUE_TYPE_UNSIGNED,
            aggregation=metric_registry_pb2.METRIC_AGGREGATION_LATEST)
        catalog = registry.catalog(pool_source)
        catalog.entries[0].status_method = "rl.training.v1.SamplePoolConsumerService/GetStatus"
        catalog.entries[0].status_field = "accepted_unique_transitions"

        class PoolStatus(rpc.SamplePoolConsumerServiceServicer):
            def GetStatus(self, request, context):
                return status

        class Catalog(metric_catalog_pb2_grpc.MetricCatalogServiceServicer):
            available = False

            def GetMetricCatalog(self, request, context):
                if not self.available:
                    context.abort(grpc.StatusCode.UNAVAILABLE, "catalog observation failed")
                return catalog

        catalog_service = Catalog()
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        rpc.add_SamplePoolConsumerServiceServicer_to_server(PoolStatus(), server)
        metric_catalog_pb2_grpc.add_MetricCatalogServiceServicer_to_server(catalog_service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        collector = PreviewCollector(
            endpoints={role: f"127.0.0.1:{port}" for role in ("learner", "aiserver", "sample_pool", "distributor")},
            learner_source=source("learner"), source_id="preview-test", directory=self.directory.name,
            backend="jsonl", logger=logging.getLogger("test"), collect_aiserver=False)
        self.addCleanup(collector.close)
        collector.record()
        status.timestamp_unix_ms += 2000
        status.accepted_unique_transitions += 60
        collector.record()
        reader = MetricsFileReader(self.directory.name, metrics_source_id="preview-test")
        reader.refresh()
        record = reader.latest()
        self.assertTrue(record["sample_pool"]["ready"])
        self.assertEqual(record["sample_pool"]["accepted"], 160)
        self.assertNotIn("error", record["sample_pool"])
        self.assertEqual(record["rates"]["accepted_sps"], 30)
        self.assertNotIn("produced_sps", record["rates"])
        query = reader.query_fields(["sample.throughput.accepted_per_second"])
        self.assertEqual(list(query["latest"]["metric_values"].values()), [30])
        self.assertIn("catalog observation failed", query["latest"]["metric_event_views"]["catalog_errors"]["sample_pool"])
        self.assertFalse(any(field["field_id"] == "sample.flow.accepted.total" for field in reader.catalog()["fields"]))

        catalog_service.available = True
        status.timestamp_unix_ms += 2000
        status.accepted_unique_transitions += 40
        collector.record()
        reader.refresh()
        query = reader.query_fields(["sample.flow.accepted.total", "sample.throughput.accepted_per_second"])
        values = query["latest"]["metric_values"]
        self.assertEqual(sorted(values.values()), [20, 200])
        self.assertEqual(query["latest"]["metric_event_views"]["catalog_errors"], {})

    def test_preview_rates_keep_source_intervals_independent(self):
        collector = PreviewCollector.__new__(PreviewCollector)
        collector._rate_snapshot = {}
        collector._last_rate_time = None
        actor = dict(instance_id="actor", lifecycle_epoch=1, timestamp=100.0, produced=100)
        pool = dict(instance_id="pool", lifecycle_epoch=1, timestamp=100.0, accepted=90, acked=60, trained=60)
        self.assertEqual(collector._rates(actor, pool, 1000)["reason"], "initial_snapshot")
        actor.update(timestamp=102.0, produced=140)
        partial = collector._rates(actor, {"error": "status unavailable"}, 1001)
        self.assertEqual(partial["produced_sps"], 20)
        self.assertEqual(partial["source_intervals"]["produced"], {"start": 100.0, "end": 102.0})
        self.assertNotIn("trained_sps", partial)

        actor.update(timestamp=104.0, produced=160)
        pool.update(timestamp=104.0, accepted=140, acked=100, trained=100)
        recovered = collector._rates(actor, pool, 1002)
        self.assertEqual(recovered["produced_sps"], 10)
        self.assertEqual(recovered["unavailable_counters"]["accepted"], "initial_snapshot")
        actor.update(timestamp=106.0, produced=180)
        pool.update(timestamp=105.0, accepted=150, acked=105, trained=105)
        complete = collector._rates(actor, pool, 1003)
        self.assertTrue(complete["available"])
        self.assertEqual([complete[f"{key}_sps"] for key in ("produced", "accepted", "acked", "trained")], [10, 10, 5, 5])

        actor.update(lifecycle_epoch=2, timestamp=108.0, produced=5)
        pool.update(timestamp=106.0, accepted=160, acked=110, trained=110)
        restarted = collector._rates(actor, pool, 1004)
        self.assertEqual(restarted["unavailable_counters"], {"produced": "source_identity_changed"})
        self.assertEqual(restarted["trained_sps"], 5)
        actor.update(timestamp=108.0, produced=10)
        pool.update(timestamp=107.0, accepted=170, acked=115, trained=115)
        repeated = collector._rates(actor, pool, 1005)
        self.assertEqual(repeated["unavailable_counters"], {"produced": "non_positive_window"})
        self.assertEqual(repeated["accepted_sps"], 10)
        actor.update(timestamp=109.0, produced=20)
        pool.update(timestamp=108.0, accepted=175, acked=120, trained=90)
        regressed = collector._rates(actor, pool, 1006)
        self.assertEqual(regressed["unavailable_counters"], {"trained": "counter_regression"})
        self.assertEqual(regressed["accepted_sps"], 5)
        self.assertNotIn("trained_sps", regressed)

    def test_catalog_is_readable_before_measurement_without_pinning_a_consumer(self):
        learner = source("learner-catalog")
        writer = LocalTrainUpdateMetricWriter(self.store, learner)
        status = LearnerStatusService(learner, lambda: {"model_step": 0, "train_updates": 0,
            "trained_samples": 0, "disposition": "STARTING"})
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        metric_catalog_pb2_grpc.add_MetricCatalogServiceServicer_to_server(MetricCatalogService(writer, status), server)
        rpc.add_LearnerStatusServiceServicer_to_server(status, server)
        metric_transport_pb2_grpc.add_MetricEventServiceServicer_to_server(LearnerMetricEventService(store=self.store, source=learner), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(channel.close)
        catalog_stub = metric_catalog_pb2_grpc.MetricCatalogServiceStub(channel)
        first = catalog_stub.GetMetricCatalog(metric_catalog_pb2.GetMetricCatalogReq())
        second = catalog_stub.GetMetricCatalog(metric_catalog_pb2.GetMetricCatalogReq())
        self.assertEqual(first, second)
        entries = {entry.definition.metric_id: entry for entry in first.entries}
        self.assertIn("learner.ppo.policy_loss", entries)
        self.assertIn("learner.ppo.value_loss", entries)
        self.assertEqual(entries["learner.ppo.entropy"].definition.unit, "nats")
        self.assertEqual(entries["learner.model_step"].status_field, "model_step")
        state = rpc.LearnerStatusServiceStub(channel).GetLearnerStatus(wire.LearnerStatusReq())
        self.assertEqual(state.model_step, 0)
        self.assertFalse(state.HasField("explained_variance"))
        response = metric_transport_pb2_grpc.MetricEventServiceStub(channel).GetMetricBatch(metric_transport_pb2.GetMetricBatchReq(
            consumer=source("independent-consumer"), cursor=metric_transport_pb2.MetricBatchCursor(source=learner),
            max_events=10, max_bytes=1048576))
        self.assertEqual(response.result, metric_transport_pb2.METRIC_BATCH_RESULT_WAIT)
        self.assertFalse(response.HasField("batch"))

    def test_collector_finishes_after_final_ack(self):
        producer = source("finished-producer")
        writer = LocalTrainUpdateMetricWriter(self.store, producer)
        writer.finalize()
        server = grpc.server(ThreadPoolExecutor(max_workers=2))
        metric_transport_pb2_grpc.add_MetricEventServiceServicer_to_server(
            LearnerMetricEventService(store=self.store, source=producer), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.addCleanup(channel.close)
        sink = RawMetricBatchStore(Path(self.directory.name) / "collector.sqlite3")
        self.addCleanup(sink.close)
        collector = MetricEventCollector(store=sink, consumer=source("preview"), role="learner",
            event_stub=metric_transport_pb2_grpc.MetricEventServiceStub(channel), logger=logging.getLogger("test"))
        self.addCleanup(collector.close)
        collector.start(producer)
        collector._thread.join(timeout=2)
        self.assertFalse(collector._thread.is_alive())
        self.assertTrue(sink.is_final(producer))
        self.assertEqual(collector.snapshot()["state"], "final")
        server.stop(0).wait()
        collector.close()
        self.assertEqual(collector.snapshot()["state"], "final")
        self.assertEqual(collector.snapshot()["retry_count"], 0)

    def test_registered_task_metrics_use_raw_denominators_and_source_scope(self):
        registry = score_registry()
        catalog_projector = LocalMetricProjector(self.store)
        catalog_projector.register_catalog("aiserver", registry.catalog(source()))
        catalog_only = next(iter(catalog_projector.snapshot()["sources"].values()))
        self.assertIn("task.balance.score", catalog_only["catalog"])
        self.assertEqual(catalog_only["windows"]["all"], {})
        persist(self.store, source(), 1, score_record(registry, 3.0, 2).SerializeToString())
        persist(self.store, source(), 2, score_record(registry, -1.0, 4).SerializeToString())
        # The same metric ID can have a different task meaning in another source.
        other = score_registry("distance")
        persist(self.store, source("server-b"), 1, score_record(other, 100.0, 1).SerializeToString())
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000005.1).snapshot()
        sources = {s["instance_id"]: s for s in snapshot["sources"].values()}
        metric = sources["server-a"]["windows"]["all"]["task.balance.score"]
        self.assertEqual(metric["sum"], 2.0)
        self.assertEqual(metric["count"], 6)
        self.assertAlmostEqual(metric["value"], 1.0 / 3.0)
        self.assertEqual(sources["server-b"]["windows"]["all"]["task.balance.score"]["value"], 100.0)
        self.assertEqual(sources["server-a"]["catalog"]["task.balance.score"]["denominator"], "transition")
        # A recent window excludes the old outlier and weights actual counts;
        # it must not average means from differently sized episodes.
        interval_record = score_record(registry, 9.0, 3)
        interval_record.points[0].interval_start_unix_ms = 1700000009000
        interval_record.points[0].interval_end_unix_ms = 1700000069000
        batch = metric_transport_pb2.MetricBatch(
            source=source(), batch_sequence=3, created_at_unix_ms=1700000070000,
            first_event_sequence=3, last_event_sequence=3,
            events=[metric_transport_pb2.MetricEvent(event_sequence=3, observed_at_unix_ms=1700000070000,
                fact_kind=metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS,
                fact_payload=interval_record.SerializeToString())])
        cursor = self.store.persist_batch("aiserver", batch)
        self.store.mark_acknowledged(batch, cursor)
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000070.1).snapshot()
        expired = LocalMetricProjector(self.store, clock=lambda: 1700000140.0).snapshot()
        self.assertTrue(all(not item["windows"]["1m"] for item in expired["sources"].values()))
        # Public catalog and value projection require no task-specific entry.
        reader = MetricsFileReader(self.directory.name, metrics_source_id="metric-test", clock=lambda: 1700000070.1)
        document = {"metric_event_views": snapshot, "metrics_source_id": "metric-test",
                    "sequence": 1, "timestamp": 1700000000.0}
        backend = JsonlBackend(self.directory.name)
        try:
            backend.write(document)
        finally:
            backend.close()
        reader.refresh()
        self.assertEqual(reader.latest()["metric_event_views"], snapshot)
        fields = reader.catalog()["fields"]
        field = next(f for f in fields if f.get("metric_id") == "task.balance.score" and f.get("source_instance") == "server-a")
        projected = project_metric_values(document, reader.metric_definitions())
        self.assertEqual(projected["metric_values"][field["series_id"]], 3.0)
        self.assertEqual(field["denominator"], "transition")
        self.assertEqual(field["label"], "Balance score")
        self.assertEqual(field["group"], "custom")
        server = HTTPServer(("127.0.0.1", 0), metrics_server.MetricsHTTPHandler)
        thread = threading.Thread(target=server.serve_forever)
        with patch.object(metrics_server, "metrics_reader", reader):
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/metrics/catalog", timeout=2) as response:
                    catalog = json.load(response)
                self.assertIn(field, catalog["fields"])
                with urlopen(base + "/api/metrics/catalog?category=custom", timeout=2) as response:
                    category = json.load(response)
                self.assertIn(field, category["fields"])
                self.assertTrue(all(f["group"] == "custom" for f in category["fields"]))
                with urlopen(base + "/api/metrics/latest", timeout=2) as response:
                    latest = json.load(response)
                self.assertEqual(latest["record"]["metric_values"][field["series_id"]], 3.0)
                with urlopen(base + "/api/metrics/query?field=task.balance.score&field=task.unknown&window=1m", timeout=2) as response:
                    query = json.load(response)
                self.assertEqual(query["unregistered_fields"], ["task.unknown"])
                self.assertEqual(len(query["series"]), 2)
                expected_statistic = {"value": 3.0, "sum": 9.0, "count": 3,
                    "window_start_unix_ms": 1700000010000, "window_end_unix_ms": 1700000070000,
                    "interval_start_unix_ms": 1700000009000, "interval_end_unix_ms": 1700000069000}
                self.assertEqual(query["latest"]["metric_statistics"][field["series_id"]], expected_statistic)
                self.assertEqual(reader.query_fields(["task.balance.score"])["latest"]
                                 ["metric_statistics"][field["series_id"]], expected_statistic)
                reader._clock = lambda: 1700000140.0
                now = reader.query_fields(["task.balance.score"])["latest"]
                self.assertTrue(all(value is None for value in now["metric_values"].values()))
                self.assertTrue(all(value is None for value in now["metric_statistics"].values()))
                reader._clock = lambda: 1700000070.1
                self.assertEqual({item["field_id"] for item in query["series"]}, {"task.balance.score"})
                self.assertEqual(query["records"][0]["metric_values"][field["series_id"]], 3.0)
                self.assertEqual(query["records"][0]["metric_statistics"][field["series_id"]], expected_statistic)
                self.assertEqual(set(query["records"][0]["metric_values"]),
                                 {item["series_id"] for item in query["series"]})
                with urlopen(base + "/api/metrics/query?field=task.balance.score&window=all&after_sequence=1", timeout=2) as response:
                    self.assertEqual(json.load(response)["records"], [])
                # A final snapshot may precede the bucket boundary of its tail.
                # The query clock must include that tail without another write.
                tail = score_record(registry, 5.0, 1)
                tail.points[0].interval_start_unix_ms = 1700000070000
                tail.points[0].interval_end_unix_ms = 1700000071000
                tail_batch = metric_transport_pb2.MetricBatch(source=source(), batch_sequence=4,
                    created_at_unix_ms=1700000071000, first_event_sequence=4, last_event_sequence=4,
                    events=[metric_transport_pb2.MetricEvent(event_sequence=4, observed_at_unix_ms=1700000071000,
                        fact_kind=metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS,
                        fact_payload=tail.SerializeToString())])
                cursor = self.store.persist_batch("aiserver", tail_batch)
                self.store.mark_acknowledged(tail_batch, cursor)
                final_snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000071.1).snapshot()
                backend = JsonlBackend(self.directory.name)
                try:
                    backend.write({**document, "metric_event_views": final_snapshot,
                                   "sequence": 2, "timestamp": 1700000071.1})
                finally:
                    backend.close()
                reader.refresh()
                reader._clock = lambda: 1700000071.1
                self.assertEqual(reader.query_fields(["task.balance.score"])["latest"]
                                 ["metric_values"][field["series_id"]], 3.0)
                reader._clock = lambda: 1700000075.1
                with urlopen(base + "/api/metrics/query?field=task.balance.score&window=current", timeout=2) as response:
                    matured = json.load(response)["latest"]["metric_statistics"][field["series_id"]]
                self.assertEqual(matured["value"], 3.5)
                self.assertEqual((matured["sum"], matured["count"]), (14.0, 4))
                self.assertEqual(matured["window_end_unix_ms"], 1700000075000)
                self.assertEqual(matured["interval_end_unix_ms"], 1700000071000)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_monitor_views_preserve_registered_values_and_select_training_modules(self):
        learner = common_pb2.ServiceInstanceIdentity(
            component="learner", instance_id="learner-for-monitor", lifecycle_epoch=1)
        writer = LocalTrainUpdateMetricWriter(self.store, learner)
        fact = training_metrics_pb2.TrainUpdateMetricFact(
            train_update_id="update-1", train_update_sequence=1, delivery_id="delivery-1",
            cumulative_trained_samples=4, actual_batch_size=4,
            sample_evaluation_count=4, optimizer_step_count=1,
            published_model=model_identity_pb2.ModelIdentity(model_lineage_id="lineage", model_step=1),
            behavior_model_lineage_id="lineage",
        )
        for name, total in (("policy_loss", -0.8), ("value_loss", 1.2),
                            ("approx_kl", 0.04), ("entropy", 1.6)):
            fact.ppo_statistics.add(field_id=name, sum=total, count=4)
        writer.append(fact, observed_at_unix_ms=1700000000000)
        writer.finalize()

        registry = MetricRegistry()
        points = []
        # New task fields remain discoverable independently of panel defaults.
        for name, unit, denominator, total in (
                ("return", "reward", "agent_episode", 12.0),
                ("success", "ratio", "agent_episode", 3.0),
                ("any_success", "ratio", "environment_episode", 3.0),
                ("all_success", "ratio", "environment_episode", 1.0),
                ("reward.total.per_transition", "reward", "transition", 6.0),
                ("reward.additional_bonus.per_transition", "reward", "transition", 2.0),
                ("reward.additional_bonus.per_episode", "reward", "agent_episode", 2.0)):
            metric_id = "task.maze." + name
            registry.register(metric_id, display_name="Registered " + name, unit=unit,
                              scope="environment_episode" if denominator == "environment_episode" else "agent_episode",
                              denominator=denominator,
                              category="reward" if name == "return" or name.startswith("reward.") else "success" if "success" in name else "episode",
                              value_type=metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT,
                              aggregation=metric_registry_pb2.METRIC_AGGREGATION_MEAN)
            points.append(metric_registry_pb2.MetricPoint(
                metric_id=metric_id, sum_count=metric_registry_pb2.MetricSumCount(sum=total, count=4)))
        persist(self.store, source(), 1, registry.record(points).SerializeToString())
        balance = score_registry(category="balance")
        persist(self.store, source("balance-server"), 1,
                score_record(balance, 9.0, 3).SerializeToString())
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000005.1).snapshot()
        record = {"metric_event_views": snapshot, "metrics_source_id": "monitor-views",
                  "sequence": 1, "timestamp": 1700000000.0}
        backend = JsonlBackend(self.directory.name)
        try:
            backend.write(record)
        finally:
            backend.close()
        reader = MetricsFileReader(
            self.directory.name, metrics_source_id="monitor-views",
            task_views_path=Path(__file__).resolve().parents[1] / "configs" / "monitor_views.json")
        reader.refresh()
        catalog = reader.catalog()
        fields = {f["metric_id"]: f for f in catalog["fields"] if "metric_id" in f}
        self.assertEqual(fields["learner.ppo.policy_loss"]["label"], "Policy Loss")
        self.assertEqual(fields["learner.ppo.value_loss"]["group"], "loss")
        loss_panel = next(p for p in catalog["panels"] if p["category"] == "loss")
        self.assertTrue({"learner.ppo.policy_loss", "learner.ppo.value_loss"}
                        .issubset(loss_panel["required"]))
        ppo_panel = next(p for p in catalog["panels"] if p["id"] == "ppo-stability")
        self.assertIn("learner.ppo.entropy", ppo_panel["required"])
        self.assertIn("learner.ppo.entropy", ppo_panel["defaults"])
        self.assertFalse(any(p["id"] == "entropy" for p in catalog["panels"]))
        self.assertEqual(fields["task.maze.return"]["group"], "reward")
        self.assertEqual(fields["task.maze.return"]["label"], "Registered return")
        reward_panel = next(p for p in catalog["panels"] if p["id"] == "reward")
        self.assertEqual(reward_panel["defaults"], ["task.maze.return"])
        reward_fields = {field["field_id"] for field in reader.catalog(category="reward")["fields"]}
        self.assertTrue({"task.maze.return", "task.maze.reward.total.per_transition",
                         "task.maze.reward.additional_bonus.per_transition",
                         "task.maze.reward.additional_bonus.per_episode"}.issubset(reward_fields))
        self.assertEqual(fields["task.maze.reward.additional_bonus.per_transition"]["denominator"],
                         "transition")
        self.assertEqual(fields["task.maze.reward.additional_bonus.per_episode"]["denominator"],
                         "agent_episode")
        success_panel = next(p for p in catalog["panels"] if p["id"] == "success")
        self.assertEqual(success_panel["defaults"], ["task.maze.any_success", "task.maze.all_success"])
        success = fields["task.maze.success"]
        self.assertEqual(success["unit"], "ratio")
        self.assertEqual(success["display_unit"], "%")
        projected = project_metric_values(record, reader.metric_definitions())
        episode_return = fields["task.maze.return"]
        self.assertEqual(episode_return["denominator"], "agent_episode")
        self.assertEqual(projected["metric_values"][episode_return["series_id"]], 3.0)
        self.assertEqual(projected["metric_values"][fields["task.maze.reward.total.per_transition"]["series_id"]], 1.5)
        return_statistic = projected["metric_statistics"][episode_return["series_id"]]
        self.assertEqual((return_statistic["sum"], return_statistic["count"]), (12.0, 4))
        self.assertEqual(projected["metric_values"][success["series_id"]], 0.75)
        for suffix in ("per_transition", "per_episode"):
            component = fields["task.maze.reward.additional_bonus." + suffix]
            self.assertEqual(projected["metric_values"][component["series_id"]], 0.5)
        self.assertEqual(projected["metric_event_views"], snapshot)
        self.assertEqual(success["denominator"], "agent_episode")
        self.assertEqual(reader.history("all")[0]["metric_values"][success["series_id"]], 0.75)
        success_query = reader.query_fields(
            ["task.maze.success", "task.maze.any_success", "task.maze.all_success"])
        self.assertEqual(success_query["unregistered_fields"], [])
        success_values = success_query["records"][0]["metric_values"]
        self.assertEqual(success_values[success["series_id"]], 0.75)
        self.assertEqual(success_values[fields["task.maze.any_success"]["series_id"]], 0.75)
        self.assertEqual(success_values[fields["task.maze.all_success"]["series_id"]], 0.25)
        for name in ("task.maze.any_success", "task.maze.all_success"):
            self.assertEqual(fields[name]["denominator"], "environment_episode")

        common_reader = MetricsFileReader(self.directory.name, metrics_source_id="monitor-views")
        common_reader.refresh()
        common_catalog = common_reader.catalog()
        self.assertNotEqual(common_catalog["layout_key"], catalog["layout_key"])
        self.assertFalse(any(p["category"] in {"reward", "episode", "success"}
                             for p in common_catalog["panels"]))
        self.assertFalse(any(field.startswith("task.maze.")
                             for p in common_catalog["panels"]
                             for field in p.get("defaults", []) + p.get("required", [])))

        balance_views = Path(self.directory.name) / "balance_views.json"
        balance_views.write_text(json.dumps({
            "layout_key": "monitor-balance",
            "categories": [{"id": "balance", "label": "Balance"}],
            "panels": [{"id": "balance", "title": "Balance", "category": "balance",
                        "defaults": ["task.balance.score"]}],
            "fields": [],
        }), encoding="utf-8")
        balance_reader = MetricsFileReader(
            self.directory.name, metrics_source_id="monitor-views", task_views_path=balance_views)
        balance_reader.refresh()
        balance_catalog = balance_reader.catalog()
        self.assertEqual(balance_catalog["layout_key"], "monitor-balance")
        balance_panel = next(p for p in balance_catalog["panels"] if p["id"] == "balance")
        balance_field = next(f for f in balance_reader.catalog(category="balance")["fields"]
                             if f["field_id"] == balance_panel["defaults"][0])
        self.assertEqual(balance_field["label"], "Balance score")
        self.assertFalse(any(field.startswith("task.maze.")
                             for p in balance_catalog["panels"]
                             for field in p.get("defaults", []) + p.get("required", [])))
        balance_query = balance_reader.query_fields(balance_panel["defaults"])
        self.assertEqual(balance_query["unregistered_fields"], [])
        self.assertEqual(balance_query["records"][0]["metric_values"][balance_field["series_id"]], 3.0)

    def test_train_metrics_are_derived_from_raw_sum_counts(self):
        learner = common_pb2.ServiceInstanceIdentity(component="learner", instance_id="learner-a", lifecycle_epoch=1)
        writer = LocalTrainUpdateMetricWriter(self.store, learner)
        fact = training_metrics_pb2.TrainUpdateMetricFact(
            train_update_id="update-2", train_update_sequence=2, delivery_id="delivery-2",
            cumulative_trained_samples=(1 << 53) + 1, actual_batch_size=2,
            sample_evaluation_count=6, optimizer_step_count=3,
            published_model=model_identity_pb2.ModelIdentity(model_lineage_id="lineage", model_step=2),
            behavior_model_lineage_id="lineage", minimum_behavior_model_step=0, maximum_behavior_model_step=1,
        )
        fact.ppo_statistics.add(field_id="policy_loss", sum=-1.2, count=6)
        fact.ppo_statistics.add(field_id="gradient_norm", sum=6.0, count=3)
        fact.ppo_statistics.add(field_id="policy_lag", sum=3.0, count=2)
        fact.ppo_statistics.add(field_id="raw_advantage", sum=4.0, count=2)
        writer.append(fact, observed_at_unix_ms=1700000000000)
        writer.finalize()
        batches = self.store.committed_batches_after(0)
        self.assertTrue(batches[0][3].HasField("gap"))
        self.assertTrue(batches[-1][3].source_final)
        self.assertEqual(self.store.committed_cursor(learner).acknowledged_event_sequence, 2)
        result = next(iter(LocalMetricProjector(self.store, clock=lambda: 1700000005.1).snapshot()["sources"].values()))
        values = result["windows"]["all"]
        self.assertAlmostEqual(values["learner.ppo.policy_loss"]["value"], -0.2)
        self.assertEqual(values["learner.ppo.gradient_norm"]["value"], 2.0)
        self.assertEqual(values["learner.ppo.policy_lag"]["value"], 1.5)
        self.assertEqual(values["learner.ppo.raw_advantage"]["value"], 2.0)
        self.assertEqual(values["learner.update.cumulative_trained_samples"]["value"], (1 << 53) + 1)
        self.assertEqual(result["catalog"]["learner.ppo.policy_loss"]["denominator"], "sample_evaluation")
        self.assertEqual(result["catalog"]["learner.ppo.gradient_norm"]["denominator"], "optimizer_step")
        self.assertEqual(result["catalog"]["learner.ppo.policy_lag"]["denominator"], "transition")

    def test_metric_content_errors_do_not_rewind_durable_transport(self):
        registry = score_registry()
        valid = score_record(registry, 2.0, 2).SerializeToString()
        persist(self.store, source(), 1, valid)
        bad = persist(self.store, source(), 2, b"invalid protobuf")
        changed = score_record(score_registry("distance"), 999.0, 1).SerializeToString()
        persist(self.store, source(), 3, changed)
        persist(self.store, source(), 4, valid)
        projector = LocalMetricProjector(self.store, clock=lambda: 1700000005.1)
        snapshot = projector.snapshot()
        self.assertEqual(snapshot["status"], "projection_error")
        state = next(iter(snapshot["sources"].values()))
        self.assertEqual(state["error_count"], 2)
        self.assertEqual(state["last_error"]["event_sequence"], 3)
        self.assertIn("definition changed", state["last_error"]["message"])
        self.assertEqual(state["windows"]["all"]["task.balance.score"], {"value": 1.0, "sum": 4.0, "count": 4})
        self.assertEqual(projector.snapshot(), snapshot)
        self.assertEqual(self.store.committed_cursor(source()).acknowledged_event_sequence, 4)
        stored = self.store.committed_batches_after(0)[1][3]
        self.assertEqual(stored.SerializeToString(deterministic=True), bad.SerializeToString(deterministic=True))

    def test_model_feedback_failure_preserves_active_model_state(self):
        identity = {"model_lineage_id": "lineage", "model_step": 0}
        feedback = {"candidate_model": {"model_lineage_id": "lineage", "model_step": 1},
                    "stage": "prepare", "last_error": "invalid model output"}
        actor = {"ready": True, "state": "AISERVER_STATE_READY", "instance_id": "server",
                 "model_identity": identity, "model_feedback": feedback}
        pool = {"ready": True, "ingress_ready": True, "instance_id": "pool"}
        learner = {"model_identity": identity}
        model = {"ready": True, "instance_id": "model", "latest_model_identity": identity,
                 "latest_ack_model_identity": identity, "latest_ack_status": "MODEL_LOAD_STATUS_LOADED"}
        state = training_chain_status(actor, pool, learner, model)
        self.assertEqual(state["model_sync"]["state"], "failed")
        self.assertEqual(state["model_sync"]["feedback"], feedback)
        self.assertTrue(state["server_pod"]["ready"])
        self.assertFalse(state["ready"])

    def test_metric_relay_starts_from_bootstrap_ack_source(self):
        current_source = source("bootstrap-actor")
        learner = common_pb2.ServiceInstanceIdentity(
            component="learner", instance_id="learner", lifecycle_epoch=1)
        status_stub = Mock()
        status_stub.GetAIServerStatus.return_value = wire.AIServerStatusRsp(
            aiserver=source("previous-actor"))
        event_stub = Mock()
        batch = metric_transport_pb2.MetricBatch(
            source=current_source, batch_sequence=1, created_at_unix_ms=1700000000001,
            first_event_sequence=1, last_event_sequence=1,
            events=[metric_transport_pb2.MetricEvent(
                event_sequence=1, observed_at_unix_ms=1700000000001,
                fact_kind=metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS,
                fact_payload=score_record(score_registry(), 6.0, 3).SerializeToString())])
        relay = MetricEventCollector(store=self.store, consumer=learner,
            role="aiserver", event_stub=event_stub, logger=logging.getLogger("test"))
        self.addCleanup(relay.close)

        def get_batch(request, **kwargs):
            if request.cursor.source != current_source:
                return metric_transport_pb2.GetMetricBatchRsp(
                    producer=source("previous-actor"),
                    result=metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID,
                    message="metric journal is pinned to another consumer")
            return metric_transport_pb2.GetMetricBatchRsp(producer=current_source,
                result=metric_transport_pb2.METRIC_BATCH_RESULT_DELIVERED, batch=batch)

        def acknowledge(request, **kwargs):
            relay._stop.set()
            return metric_transport_pb2.AckMetricBatchRsp(producer=current_source,
                result=metric_transport_pb2.METRIC_BATCH_ACK_RESULT_APPLIED, committed_cursor=request.cursor)

        event_stub.GetMetricBatch.side_effect = get_batch
        event_stub.AckMetricBatch.side_effect = acknowledge
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.logger = logging.getLogger("test")
        runtime.metric_event_server_enabled = True
        runtime.metric_event_server = Mock()
        runtime.preview = PreviewCollector.__new__(PreviewCollector)
        runtime.preview._sources = {}
        runtime.preview._binding_lock = threading.Lock()
        runtime.preview.collectors = {"aiserver": relay}
        runtime._publish_metric_ready = Mock()
        runtime._start_metric_events()
        runtime.metric_event_server.start.assert_called_once()
        event_stub.GetMetricBatch.assert_not_called()

        identity = model_identity_pb2.ModelIdentity(model_lineage_id="this-run", model_step=0)
        document = {"manifest": wire.ModelArtifactManifest(identity=identity),
                    "identity": {"model_lineage_id": "this-run", "model_step": 0}}
        runtime.trainer = Mock(model_step=0)
        runtime.publisher = Mock()
        runtime.publisher.publish_runtime.return_value = document
        runtime._startup_mode = "fresh"
        runtime.train_updates = runtime.trained_samples = 0
        runtime.model_manifests = {}
        runtime.initial_model_ack_timeout = 2.0
        runtime.model_stub = Mock()
        runtime.model_stub.GetModelDistributorStatus.return_value = wire.ModelDistributorStatusRsp(
            ready=True, latest_ack_model=identity, latest_ack_status=wire.MODEL_LOAD_STATUS_LOADED,
            latest_ack_aiserver=current_source)
        runtime._model_distributor_authority = Mock()
        runtime._pin_model_distributor_authority = Mock(return_value="distributor")
        runtime._assert_initial_latest_not_newer = Mock(return_value="distributor")
        runtime._initial_model_requires_registration = Mock(return_value=(False, "distributor"))
        runtime._commit_learner_metrics = Mock()
        with patch("main.training_runtime._stop_requested", threading.Event()):
            self.assertTrue(runtime._initialize_models())
        relay._thread.join(timeout=2.0)
        self.assertFalse(relay._thread.is_alive())
        self.assertEqual(relay.snapshot()["state"], "connected")
        self.assertEqual(event_stub.GetMetricBatch.call_count, 1)
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000005.1).snapshot()
        projected = list(snapshot["sources"].values())
        self.assertEqual([item["instance_id"] for item in projected], [current_source.instance_id])
        self.assertEqual(projected[0]["windows"]["all"]["task.balance.score"]["value"], 2.0)

    def test_relay_contract_rejection_is_terminal_and_visible(self):
        status_stub = Mock()
        status_stub.GetAIServerStatus.return_value = wire.AIServerStatusRsp(aiserver=source())
        event_stub = Mock()
        event_stub.GetMetricBatch.return_value = metric_transport_pb2.GetMetricBatchRsp(
            producer=source(), result=metric_transport_pb2.METRIC_BATCH_RESULT_DELIVERED,
        )
        relay = MetricEventCollector(store=self.store, consumer=source("consumer"),
            role="aiserver", event_stub=event_stub, logger=logging.getLogger("test"))
        self.addCleanup(relay.close)
        relay.start(source())
        relay._thread.join(timeout=2.0)
        self.assertFalse(relay._thread.is_alive())
        self.assertEqual(event_stub.GetMetricBatch.call_count, 1)
        self.assertEqual(relay.snapshot()["state"], "rejected")
        self.assertIn("without a batch", relay.snapshot()["last_error"])
        relay.close()
        self.assertEqual(relay.snapshot()["state"], "rejected")


if __name__ == "__main__":
    unittest.main()
