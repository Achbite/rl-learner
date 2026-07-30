import json
import tempfile
import time
import unittest
from pathlib import Path

from main.training_runtime import TrainingRuntime
from tools.metrics_server import MetricsFileReader


class MetricsContractTest(unittest.TestCase):
    def test_component_error_preserves_last_successful_counters(self):
        snapshot = TrainingRuntime._component_error_snapshot(
            {
                "ready": True,
                "instance_id": "actor-a",
                "produced": 1024,
                "episodes": {"mean_agent_return": 2.79},
            },
            "connection refused",
            {"episodes": {}},
        )

        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["error"], "connection refused")
        self.assertEqual(snapshot["produced"], 1024)
        self.assertEqual(
            snapshot["episodes"]["mean_agent_return"], 2.79
        )

        initial_failure = TrainingRuntime._component_error_snapshot(
            {}, "not ready", {"episodes": {}}
        )
        self.assertFalse(initial_failure["ready"])
        self.assertEqual(initial_failure["episodes"], {})

    def test_interval_rates_reset_when_component_instance_changes(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.trained_samples = 100
        runtime._rate_snapshot = {}
        first = runtime._rate_interval(
            10.0,
            {"instance_id": "actor-a", "produced": 100},
            {"instance_id": "pool-a", "accepted": 100, "acked": 80},
        )
        self.assertEqual(first["rates"]["produced_sps"], 0.0)

        runtime.trained_samples = 120
        second = runtime._rate_interval(
            12.0,
            {"instance_id": "actor-a", "produced": 140},
            {"instance_id": "pool-a", "accepted": 130, "acked": 120},
        )
        self.assertEqual(second["rates"]["produced_sps"], 20.0)
        self.assertEqual(second["rates"]["accepted_sps"], 15.0)
        self.assertEqual(second["rates"]["acked_sps"], 20.0)
        self.assertEqual(second["rates"]["trained_sps"], 10.0)

        reset = runtime._rate_interval(
            13.0,
            {"instance_id": "actor-b", "produced": 1},
            {"instance_id": "pool-b", "accepted": 1, "acked": 1},
        )
        self.assertEqual(reset["rates"]["produced_sps"], 0.0)
        self.assertEqual(reset["rates"]["accepted_sps"], 0.0)
        self.assertEqual(reset["rates"]["acked_sps"], 0.0)

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
