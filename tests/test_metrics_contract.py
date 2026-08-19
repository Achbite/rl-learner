import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from main.training_runtime import TrainingRuntime
from proto import training_pb2
from src.metrics.metrics_backend import DisabledMetricsBackend
from tools.metrics_server import MetricsFileReader


class MetricsContractTest(unittest.TestCase):
    def test_operational_training_depth_uses_the_committed_runtime_state(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime._metrics_lock = threading.RLock()
        runtime._metrics_context = {
            "behavior_model": {
                "minimum_model_step": 1,
                "maximum_model_step": 1,
            },
            "actual_batch_size": 512,
            "disposition": "TRAINED",
            "train_update_id": "train-update-00000002",
            "error": "",
        }
        runtime._committed_learner_metrics = {
            "model_identity": {
                "model_lineage_id": "learner-metrics-chain-test",
                "model_step": 2,
                "artifact_digest": "a" * 64,
                "manifest_digest": "b" * 64,
            },
            "model_step": 2,
            "train_updates": 2,
            "run_train_updates": 2,
            "run_trained_samples": 1024,
            "policy_lag": 0,
            "max_policy_lag": 1,
        }
        runtime.learner_service = SimpleNamespace(
            instance_id="learner-current", lifecycle_epoch=1
        )
        runtime.trainer = SimpleNamespace(model_step=3)
        runtime.initial_model_step = 0
        runtime.train_batch_size = 512
        runtime.max_train_batch_size = 639
        runtime.model_manifests = {}

        snapshot = runtime._learner_metrics_snapshot()

        self.assertEqual(snapshot["model_step"], 2)
        self.assertEqual(snapshot["train_updates"], 2)
        self.assertEqual(snapshot["model_identity"]["model_step"], 2)
        self.assertEqual(snapshot["run_trained_samples"], 1024)
        self.assertEqual(snapshot["initial_model_step"], 0)
        self.assertEqual(snapshot["max_train_batch_size"], 639)

    def test_actor_snapshot_projects_client_and_model_chain_state(self):
        status = training_pb2.AIServerStatusRsp()
        status.contract.package_name = "rl-contracts"
        status.contract.package_version = "0.13.0"
        status.aiserver.component = "rl-aiserver"
        status.aiserver.instance_id = "aiserver-test"
        status.aiserver.lifecycle_epoch = 3
        status.ready = True
        status.state = training_pb2.AISERVER_STATE_READY
        status.active_actor_session_count = 1
        status.active_trajectory_count = 0
        status.metrics.descriptors.add(
            field_id="server.client.session_recent.v1",
            label="Client session recent",
        )
        status.metrics.descriptors.add(
            field_id="server.client.last_activity_unix_ms.v1",
            label="Client last activity",
        )
        status.metrics.values.add(
            field_id="server.client.session_recent.v1", value=1.0
        )
        status.metrics.values.add(
            field_id="server.client.last_activity_unix_ms.v1",
            value=1234.0,
        )

        runtime = object.__new__(TrainingRuntime)
        runtime.contract = type(status.contract)()
        runtime.contract.CopyFrom(status.contract)
        runtime.actor_stub = SimpleNamespace(
            GetAIServerStatus=lambda *_args, **_kwargs: status
        )

        snapshot = runtime._actor_snapshot()

        self.assertEqual(snapshot["active_sessions"], 1)
        self.assertEqual(snapshot["active_trajectories"], 0)
        self.assertTrue(snapshot["client_session_recent"])
        self.assertEqual(snapshot["client_last_activity_unix_ms"], 1234)
        self.assertEqual(snapshot["instance_id"], "aiserver-test")

    def test_status_reports_one_current_metrics_source(self):
        with tempfile.TemporaryDirectory() as directory:
            started_at = time.time()
            root = Path(directory)
            old_path = root / "metrics_old.jsonl"
            old_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "metrics_source_id": "old-source",
                        "sequence": 128,
                        "timestamp": started_at,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            os.utime(
                old_path,
                (started_at - 60.0, started_at - 60.0),
            )
            current_path = root / "metrics_current.jsonl"
            current_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "mode": "training",
                        "metrics_source_id": "current-source",
                        "sequence": 1,
                        "timestamp": started_at,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reader = MetricsFileReader(
                directory,
                metrics_source_id="current-source",
                service_instance_id="learner-metrics-test",
                started_at=started_at,
                runtime_mode="training",
            )

            records = reader.query()
            status = reader.status()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["metrics_source_id"], "current-source")
            self.assertEqual(status["latest_sequence"], 1)
            self.assertEqual(status["mode"], "training")
            self.assertEqual(status["service_instance_id"], "learner-metrics-test")

    def test_metrics_backend_failure_does_not_block_runtime_startup(self):
        runtime = object.__new__(TrainingRuntime)
        runtime.logger = mock.Mock()

        with mock.patch(
            "main.training_runtime.create_backend",
            side_effect=OSError("read-only metrics directory"),
        ):
            backend = runtime._create_metrics_backend(
                "jsonl", "/unavailable/metrics"
            )

        self.assertIsInstance(backend, DisabledMetricsBackend)
        backend.write({"sequence": 1})
        self.assertIsNone(backend.latest())
        self.assertEqual(backend.query(), [])
        self.assertFalse(backend.summary()["enabled"])

    def test_metric_snapshot_preserves_neutral_sample_flow_facts(self):
        snapshot = training_pb2.MetricSnapshot()
        descriptor = snapshot.descriptors.add(
            field_id="sample.flow.produced.total.v1",
            label="Produced Samples",
            scope="aiserver",
            statistic="sum",
            aggregation_kind=training_pb2.METRIC_AGGREGATION_KIND_SUM,
            window_kind=training_pb2.METRIC_WINDOW_KIND_CUMULATIVE,
        )
        snapshot.values.add(
            field_id=descriptor.field_id,
            value=128.0,
            sum=128.0,
            count=1,
            window_end_unix_ms=1234,
        )

        values, labels, statistics, descriptors = (
            TrainingRuntime._metric_snapshot(snapshot)
        )

        self.assertEqual(values[descriptor.field_id], 128.0)
        self.assertEqual(labels[descriptor.field_id], descriptor.label)
        self.assertEqual(statistics[descriptor.field_id]["sum"], 128.0)
        self.assertEqual(statistics[descriptor.field_id]["count"], 1)
        self.assertEqual(descriptors[descriptor.field_id]["scope"], "aiserver")

    def test_component_error_remains_task_neutral(self):
        snapshot = TrainingRuntime._component_error_snapshot(
            "aiserver", "connection refused"
        )
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["error"], "connection refused")
        self.assertEqual(snapshot["component"], "aiserver")

    def test_reader_summary_exposes_sample_and_chain_state_without_platform_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics_current.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "mode": "training",
                        "sequence": 1,
                        "timestamp": time.time() - 10,
                        "interval_ms": 1000,
                        "sample_pool": {
                            "acked": 512,
                            "ready_samples": 128,
                        },
                        "chain": {"ready": True},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reader = MetricsFileReader(directory)
            summary = reader.summary()
            status = reader.status()

            self.assertEqual(summary["consumed"], 512)
            self.assertEqual(summary["queue_size"], 128)
            self.assertTrue(summary["chain_ready"])
            self.assertTrue(status["stale"])
            self.assertNotIn("run_id", summary)
            self.assertNotIn("run_id", status)
