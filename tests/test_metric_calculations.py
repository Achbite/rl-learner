import logging
import json
import threading
from http.server import HTTPServer
from urllib.request import urlopen
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from proto import common_pb2, training_metrics_pb2, training_pb2 as wire
from src.metrics.metric_events import (
    AIServerMetricRelay, LocalMetricProjector, LocalTrainUpdateMetricWriter,
    MetricEventContractError, RawMetricBatchStore,
)
from src.metrics.registered_metrics import MetricRegistry
from src.metrics.metrics_backend import JsonlBackend
from tools import metrics_server
from tools.metrics_server import MetricsFileReader, project_metric_values
from main.training_runtime import TrainingRuntime, training_chain_status


def source(instance="server-a"):
    return common_pb2.ServiceInstanceIdentity(
        component="aiserver", instance_id=instance, lifecycle_epoch=1,
    )


def score_registry(unit="reward"):
    registry = MetricRegistry()
    registry.register("task.balance.score", display_name="Balance score", unit=unit,
                      scope="episode", value_type=wire.METRIC_VALUE_TYPE_SUM_COUNT,
                      aggregation=wire.METRIC_AGGREGATION_MEAN, denominator="transition")
    return registry


def score_record(registry, value, count):
    return registry.record([wire.MetricPoint(
        metric_id="task.balance.score", sum_count=wire.MetricSumCount(sum=value, count=count),
    )])


def persist(store, producer, sequence, payload, kind=wire.METRIC_FACT_KIND_REGISTERED_METRICS):
    batch = wire.MetricBatch(
        source=producer, batch_sequence=sequence, created_at_unix_ms=1700000000000 + sequence,
        first_event_sequence=sequence, last_event_sequence=sequence,
        events=[wire.MetricEvent(
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

    def test_registered_task_metrics_use_raw_denominators_and_source_scope(self):
        registry = score_registry()
        persist(self.store, source(), 1, score_record(registry, 3.0, 2).SerializeToString())
        persist(self.store, source(), 2, score_record(registry, -1.0, 4).SerializeToString())
        # The same metric ID can have a different task meaning in another source.
        other = score_registry("distance")
        persist(self.store, source("server-b"), 1, score_record(other, 100.0, 1).SerializeToString())
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000000.1).snapshot()
        sources = {s["instance_id"]: s for s in snapshot["sources"].values()}
        metric = sources["server-a"]["windows"]["all"]["task.balance.score"]
        self.assertEqual(metric["sum"], 2.0)
        self.assertEqual(metric["count"], 6)
        self.assertAlmostEqual(metric["value"], 1.0 / 3.0)
        self.assertEqual(sources["server-b"]["windows"]["all"]["task.balance.score"]["value"], 100.0)
        self.assertEqual(sources["server-a"]["catalog"]["task.balance.score"]["denominator"], "transition")
        # A recent window excludes the old outlier and weights actual counts;
        # it must not average means from differently sized episodes.
        batch = wire.MetricBatch(
            source=source(), batch_sequence=3, created_at_unix_ms=1700000070000,
            first_event_sequence=3, last_event_sequence=3,
            events=[wire.MetricEvent(event_sequence=3, observed_at_unix_ms=1700000070000,
                fact_kind=wire.METRIC_FACT_KIND_REGISTERED_METRICS,
                fact_payload=score_record(registry, 9.0, 3).SerializeToString())])
        cursor = self.store.persist_batch("aiserver", batch)
        self.store.mark_acknowledged(batch, cursor)
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000070.1).snapshot()
        expired = LocalMetricProjector(self.store, clock=lambda: 1700000140.0).snapshot()
        self.assertTrue(all(not item["windows"]["1m"] for item in expired["sources"].values()))
        # Public catalog and value projection require no task-specific entry.
        reader = MetricsFileReader(self.directory.name, metrics_source_id="metric-test")
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
                self.assertEqual({item["field_id"] for item in query["series"]}, {"task.balance.score"})
                self.assertEqual(query["records"][0]["metric_values"][field["series_id"]], 3.0)
                self.assertEqual(set(query["records"][0]["metric_values"]),
                                 {item["series_id"] for item in query["series"]})
                with urlopen(base + "/api/metrics/query?field=task.balance.score&window=all&after_sequence=1", timeout=2) as response:
                    self.assertEqual(json.load(response)["records"], [])
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
            published_model=wire.ModelIdentity(model_lineage_id="lineage", model_step=1),
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
                              value_type=wire.METRIC_VALUE_TYPE_SUM_COUNT,
                              aggregation=wire.METRIC_AGGREGATION_MEAN)
            points.append(wire.MetricPoint(
                metric_id=metric_id, sum_count=wire.MetricSumCount(sum=total, count=4)))
        persist(self.store, source(), 1, registry.record(points).SerializeToString())
        balance = score_registry()
        persist(self.store, source("balance-server"), 1,
                score_record(balance, 9.0, 3).SerializeToString())
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000000.1).snapshot()
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
        self.assertEqual(fields["task.maze.return"]["group"], "episode")
        self.assertEqual(fields["task.maze.return"]["label"], "Registered return")
        reward_panel = next(p for p in catalog["panels"] if p["id"] == "reward")
        self.assertEqual(reward_panel["defaults"], ["task.maze.reward.total.per_transition"])
        reward_fields = {field["field_id"] for field in reader.catalog(category="reward")["fields"]}
        self.assertTrue({"task.maze.reward.total.per_transition",
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
            "fields": [{"match": "task.balance.score", "category": "balance"}],
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
            published_model=wire.ModelIdentity(model_lineage_id="lineage", model_step=2),
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
        result = next(iter(LocalMetricProjector(self.store, clock=lambda: 1700000000.1).snapshot()["sources"].values()))
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
        projector = LocalMetricProjector(self.store, clock=lambda: 1700000000.1)
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
        batch = wire.MetricBatch(
            source=current_source, batch_sequence=1, created_at_unix_ms=1700000000001,
            first_event_sequence=1, last_event_sequence=1,
            events=[wire.MetricEvent(
                event_sequence=1, observed_at_unix_ms=1700000000001,
                fact_kind=wire.METRIC_FACT_KIND_REGISTERED_METRICS,
                fact_payload=score_record(score_registry(), 6.0, 3).SerializeToString())])
        relay = AIServerMetricRelay(store=self.store, consumer=learner,
            status_stub=status_stub, event_stub=event_stub, logger=logging.getLogger("test"))
        self.addCleanup(relay.close)

        def get_batch(request, **kwargs):
            if request.cursor.source != current_source:
                return wire.GetMetricBatchRsp(
                    producer=source("previous-actor"),
                    result=wire.METRIC_BATCH_RESULT_REJECTED_INVALID,
                    message="metric journal is pinned to another consumer")
            return wire.GetMetricBatchRsp(producer=current_source,
                result=wire.METRIC_BATCH_RESULT_DELIVERED, batch=batch)

        def acknowledge(request, **kwargs):
            relay._stop.set()
            return wire.AckMetricBatchRsp(producer=current_source,
                result=wire.METRIC_BATCH_ACK_RESULT_APPLIED, committed_cursor=request.cursor)

        event_stub.GetMetricBatch.side_effect = get_batch
        event_stub.AckMetricBatch.side_effect = acknowledge
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.logger = logging.getLogger("test")
        runtime.metric_event_server_enabled = True
        runtime.metric_event_server = Mock()
        runtime.metric_event_relay = relay
        runtime._publish_metric_ready = Mock()
        runtime._start_metric_events()
        runtime.metric_event_server.start.assert_called_once()
        event_stub.GetMetricBatch.assert_not_called()

        identity = wire.ModelIdentity(model_lineage_id="this-run", model_step=0)
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
        snapshot = LocalMetricProjector(self.store, clock=lambda: 1700000000.1).snapshot()
        projected = list(snapshot["sources"].values())
        self.assertEqual([item["instance_id"] for item in projected], [current_source.instance_id])
        self.assertEqual(projected[0]["windows"]["all"]["task.balance.score"]["value"], 2.0)

    def test_relay_contract_rejection_is_terminal_and_visible(self):
        status_stub = Mock()
        status_stub.GetAIServerStatus.return_value = wire.AIServerStatusRsp(aiserver=source())
        event_stub = Mock()
        event_stub.GetMetricBatch.return_value = wire.GetMetricBatchRsp(
            producer=source(), result=wire.METRIC_BATCH_RESULT_DELIVERED,
        )
        relay = AIServerMetricRelay(store=self.store, consumer=source("consumer"),
            status_stub=status_stub, event_stub=event_stub, logger=logging.getLogger("test"))
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
