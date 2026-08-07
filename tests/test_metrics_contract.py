import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace

from main.training_runtime import TrainingRuntime
from tools import metrics_server as metrics_server_module
from tools.metrics_server import (
    DASHBOARD_PATH,
    MetricsFileReader,
    MetricsHTTPHandler,
    reward_component_field_id,
)


class MetricsContractTest(unittest.TestCase):
    def _start_metrics_server(self, reader):
        metrics_server_module.metrics_reader = reader
        server = HTTPServer(("127.0.0.1", 0), MetricsHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    @staticmethod
    def _http_get(port, path):
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=2
        )
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_root_redirects_to_monitor_and_api_index_stays_json(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = MetricsFileReader(
                directory,
                metrics_source_id="local-training-source-a",
            )
            port = self._start_metrics_server(reader)

            status, headers, body = self._http_get(port, "/")
            self.assertEqual(status, 302)
            self.assertEqual(headers["Location"], "/monitor")
            self.assertEqual(body, b"")

            status, headers, body = self._http_get(port, "/api")
            self.assertEqual(status, 200)
            self.assertIn("application/json", headers["Content-Type"])
            document = json.loads(body)
            self.assertEqual(document["service"], "learner-metrics")
            self.assertIn("/monitor", document["endpoints"])
            self.assertIn("/api/metrics/catalog", document["endpoints"])

            status, headers, body = self._http_get(port, "/monitor")
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers["Content-Type"])
            self.assertIn("本地训练监控", body.decode("utf-8"))

    def test_status_adds_stable_service_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            started_at = time.time() - 1.0
            metrics_path = Path(directory) / "metrics_current.jsonl"
            timestamp = time.time()
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "mode": "training",
                        "sequence": 7,
                        "timestamp": timestamp,
                        "interval_ms": 1000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            reader = MetricsFileReader(
                directory,
                metrics_source_id="local-training-source-b",
                service_instance_id="learner-metrics-instance-b",
                started_at=started_at,
            )

            first = reader.status()
            second = reader.status()

            self.assertEqual(first["schema_version"], 1)
            self.assertEqual(first["service"], "learner-metrics")
            self.assertEqual(first["stream"], "current")
            self.assertEqual(
                first["service_instance_id"],
                "learner-metrics-instance-b",
            )
            self.assertEqual(
                first["metrics_source_id"], "local-training-source-b"
            )
            self.assertEqual(first["started_at"], started_at)
            self.assertEqual(first["mode"], "training")
            self.assertEqual(first["latest_sequence"], 7)
            self.assertEqual(first["latest_timestamp"], timestamp)
            self.assertFalse(first["stale"])
            self.assertEqual(
                first["service_instance_id"],
                second["service_instance_id"],
            )

    def test_status_exposes_runtime_mode_before_first_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = MetricsFileReader(
                directory,
                metrics_source_id="local-training-source-starting",
                runtime_mode="training",
            )

            status = reader.status()

            self.assertEqual(status["mode"], "training")
            self.assertEqual(status["record_count"], 0)
            self.assertTrue(status["stale"])

    def test_default_service_identity_changes_between_server_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            first_port = self._start_metrics_server(
                MetricsFileReader(directory)
            )
            _, _, first_body = self._http_get(first_port, "/api/status")
            second_port = self._start_metrics_server(
                MetricsFileReader(directory)
            )
            _, _, second_body = self._http_get(second_port, "/api/status")
            first = json.loads(first_body)
            second = json.loads(second_body)

            self.assertNotEqual(
                first["service_instance_id"],
                second["service_instance_id"],
            )
            self.assertNotEqual(
                first["metrics_source_id"], second["metrics_source_id"]
            )

    def test_metric_catalog_and_response_projection_are_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics_contract.jsonl"
            raw_record = {
                "schema_version": 3,
                "mode": "training",
                "sequence": 9,
                "timestamp": time.time(),
                "rates": {
                    "produced_sps": 12.5,
                    "trained_sps": 10.0,
                },
                "actor": {
                    "produced": 100,
                    "push_rpc_mean_ms": 1.25,
                    "metric_values": {
                        "server.environment_step.v1": 37.0,
                        "server.task.curriculum.multiplier.v1": 8.0,
                        "server.episode.max_steps.current.v1": 1504.0,
                        "server.episode.path_ratio.mean.v1": 1.25,
                        "server.episode.step.mean.v1": 235.5,
                        "server.episode.unique_cells.mean.v1": 91.5,
                        "server.episode.blocked_move_rate.v1": 0.125,
                    },
                    "metric_labels": {
                        "server.task.curriculum.multiplier.v1": "8×",
                    },
                    "episodes": {
                        "mean_agent_return": -3.5,
                        "min_agent_return": -5.0,
                        "max_agent_return": -2.0,
                        "agent_success_rate": 0.25,
                        "reward_components": {
                            "geodesic_progress": 0.125,
                            "custom_bonus_2": 0.5,
                            "Bad Reward": 99.0,
                            "double__underscore": 99.0,
                        },
                    },
                },
                "distributor": {"trained": 96},
                "learner": {
                    "model_version": 3,
                    "run_train_updates": 3,
                    "run_trained_samples": 96,
                    "policy_loss": -0.01,
                    "clip_fraction": 0.125,
                    "value_pred_mean": 0.75,
                    "return_target_mean": 0.5,
                    "explained_variance": 0.625,
                },
            }
            original_bytes = (json.dumps(raw_record) + "\n").encode()
            path.write_bytes(original_bytes)
            reader = MetricsFileReader(directory)
            port = self._start_metrics_server(reader)

            status, _, body = self._http_get(
                port, "/api/metrics/catalog"
            )
            self.assertEqual(status, 200)
            catalog = json.loads(body)
            self.assertEqual(catalog["schema_version"], 1)
            self.assertEqual(catalog["catalog_version"], 1)
            required_keys = {
                "field_id",
                "label",
                "group",
                "dimension",
                "unit",
                "scope",
                "statistic",
                "value_kind",
                "owner_component",
                "aggregation_kind",
                "window_kind",
                "schema_identity",
            }
            self.assertTrue(catalog["fields"])
            self.assertTrue(
                all(set(field) == required_keys for field in catalog["fields"])
            )
            self.assertTrue(
                all(
                    field["schema_identity"]["schema_id"]
                    == "maze.metrics.v1"
                    and field["schema_identity"]["schema_version"] == 1
                    and len(
                        field["schema_identity"]["canonical_digest"]["hex"]
                    )
                    == 64
                    for field in catalog["fields"]
                )
            )
            field_ids = [field["field_id"] for field in catalog["fields"]]
            self.assertEqual(len(field_ids), len(set(field_ids)))
            self.assertIn(
                "server.reward.component.custom_bonus_2.transition_mean.v1",
                field_ids,
            )
            self.assertNotIn(
                "server.reward.component.Bad Reward.transition_mean.v1",
                field_ids,
            )
            self.assertFalse(
                any("double__underscore" in field_id for field_id in field_ids)
            )

            status, _, body = self._http_get(
                port, "/api/metrics?after_sequence=0"
            )
            self.assertEqual(status, 200)
            response = json.loads(body)
            projected = response["records"][0]
            self.assertEqual(projected["actor"], raw_record["actor"])
            values = projected["metric_values"]
            self.assertEqual(
                values["server.episode.learning_return.mean.v1"], -3.5
            )
            self.assertEqual(
                values[
                    "server.reward.component.geodesic_progress."
                    "transition_mean.v1"
                ],
                0.125,
            )
            self.assertEqual(
                values["server.episode.success.agent_rate.v1"], 25.0
            )
            self.assertEqual(values["learner.loss.policy.v1"], -0.01)
            self.assertEqual(
                values["learner.ppo.clip_fraction.v1"], 12.5
            )
            self.assertEqual(
                values["server.latency.sample_send.mean_ms.v1"], 1.25
            )
            self.assertEqual(values["server.environment_step.v1"], 37.0)
            self.assertEqual(
                values["server.task.curriculum.multiplier.v1"], 8.0
            )
            self.assertEqual(
                values["server.episode.path_ratio.mean.v1"], 1.25
            )
            self.assertEqual(
                values["server.episode.step.mean.v1"], 235.5
            )
            self.assertEqual(
                values["server.episode.unique_cells.mean.v1"], 91.5
            )
            self.assertEqual(
                values["server.episode.blocked_move_rate.v1"], 12.5
            )
            self.assertEqual(
                values["learner.value.prediction_mean.v1"], 0.75
            )
            self.assertEqual(
                values["learner.value.return_target_mean.v1"], 0.5
            )
            self.assertEqual(
                values["learner.value.explained_variance.v1"], 0.625
            )

            status, _, body = self._http_get(
                port, "/api/metrics/latest"
            )
            self.assertEqual(status, 200)
            latest = json.loads(body)["record"]
            self.assertEqual(latest["metric_values"], values)
            self.assertNotIn("metric_values", reader.latest())
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_reward_component_names_require_canonical_snake_case(self):
        self.assertEqual(
            reward_component_field_id("goal_reward"),
            "server.reward.component.goal_reward.transition_mean.v1",
        )
        for invalid in (
            "GoalReward",
            "goal-reward",
            "goal reward",
            "_goal_reward",
            "goal__reward",
            "goal_reward_",
            "",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    reward_component_field_id(invalid)

    def test_local_sample_monitor_is_self_contained(self):
        document = DASHBOARD_PATH.read_text(encoding="utf-8")
        self.assertIn("本地训练监控", document)
        self.assertIn("/api/metrics?after_sequence=", document)
        self.assertIn("/api/metrics/catalog", document)
        self.assertIn("/api/status", document)
        self.assertIn("metric_values", document)
        self.assertIn("Learner Pod", document)
        self.assertIn("Server Pod", document)
        for panel in (
            "Loss",
            "Episode Return",
            "Reward Components",
            "Episode Success",
            "Sample Throughput",
            "Sample Flow",
            "Latency",
            "PPO Stability",
        ):
            self.assertIn(f'title: "{panel}"', document)
        self.assertIn("catalogCandidates", document)
        self.assertIn("groupSupportsAll", document)
        self.assertIn("state.domains.clear()", document)
        self.assertIn("insideSince", document)
        self.assertIn(">= 30000", document)
        self.assertIn("quantile(sorted, .04)", document)
        self.assertIn("quantile(sorted, .96)", document)
        self.assertNotIn("Total Reward", document)
        self.assertNotIn("Reward Functions", document)
        self.assertIn("server.reward.component.geodesic_progress", document)
        self.assertIn("server.reward.component.timeout_penalty", document)
        self.assertIn("server.reward.component.first_visit_bonus", document)
        self.assertIn(
            "server.reward.component.wasted_action_penalty", document
        )
        self.assertIn("server.episode.path_ratio.mean.v1", document)
        self.assertIn("server.episode.step.mean.v1", document)
        self.assertIn("server.episode.unique_cells.mean.v1", document)
        self.assertIn("server.episode.blocked_move_rate.v1", document)
        self.assertIn("server.task.curriculum.multiplier.v1", document)
        self.assertIn("learner.value.prediction_mean.v1", document)
        self.assertIn("learner.value.return_target_mean.v1", document)
        self.assertIn("learner.value.explained_variance.v1", document)
        self.assertNotIn("potential_reward", document)
        self.assertNotIn("loiter_penalty", document)
        self.assertNotIn("rank_reward", document)
        self.assertNotIn("exploration_reward", document)
        self.assertNotIn(
            'state.focus.reward = "all"', document
        )
        self.assertNotIn("Server Pod Group", document)
        self.assertNotIn("GPU", document)
        self.assertNotIn("https://", document)

    def test_component_error_is_explicit_and_task_neutral(self):
        snapshot = TrainingRuntime._component_error_snapshot(
            "aiserver",
            "connection refused",
        )
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["error"], "connection refused")
        self.assertEqual(snapshot["component"], "aiserver")

    def test_interval_rates_use_counter_deltas(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime._rate_snapshot = {}
        first = runtime._rates(
            {"produced": 100},
            {"accepted": 100, "acked": 80, "trained": 64},
            10.0,
        )
        self.assertEqual(first["produced_sps"], 0.0)
        second = runtime._rates(
            {"produced": 140},
            {"accepted": 130, "acked": 120, "trained": 84},
            12.0,
        )
        self.assertEqual(second["produced_sps"], 20.0)
        self.assertEqual(second["accepted_sps"], 15.0)
        self.assertEqual(second["acked_sps"], 20.0)
        self.assertEqual(second["trained_sps"], 10.0)

    def test_learner_metrics_use_only_the_last_committed_model_state(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime._metrics_lock = threading.RLock()
        runtime._metrics_context = {
            "behavior_model": {"model_version": 1},
            "actual_batch_size": 512,
            "disposition": "TRAINED",
            "train_update_id": "train-update-00000002",
            "error": "",
        }
        runtime._committed_learner_metrics = {
            "model_identity": {
                "model_lineage_id": "maze-fixed-map-seed-0",
                "model_version": 2,
                "artifact_digest": "a" * 64,
                "manifest_digest": "b" * 64,
            },
            "model_version": 2,
            "model_step": 2,
            "run_train_updates": 2,
            "run_trained_samples": 1024,
            "policy_lag": 0,
            "max_policy_lag": 1,
        }
        runtime.learner_service = SimpleNamespace(
            instance_id="learner-current",
            lifecycle_epoch=1,
        )
        runtime.trainer = SimpleNamespace(model_version=3)
        runtime.model_manifests = {}

        snapshot = runtime._learner_metrics_snapshot()

        self.assertEqual(snapshot["model_version"], 2)
        self.assertEqual(snapshot["model_step"], 2)
        self.assertEqual(snapshot["model_identity"]["model_version"], 2)
        self.assertEqual(snapshot["run_trained_samples"], 1024)

    def test_metrics_reader_exposes_schema_three_summary_and_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics_20260730.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "mode": "training",
                        "sequence": 1,
                        "timestamp": time.time() - 10,
                        "interval_ms": 1000,
                        "distributor": {
                            "acked": 512,
                            "ready_samples": 128,
                        },
                        "rates": {"trained_sps": 256.0},
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
            self.assertEqual(summary["consumer_sps"], 256.0)
            self.assertEqual(summary["queue_size"], 128)
            self.assertTrue(summary["chain_ready"])
            self.assertTrue(status["stale"])
            self.assertNotIn("run_id", summary)
            self.assertNotIn("run_id", status)


if __name__ == "__main__":
    unittest.main()
