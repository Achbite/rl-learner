import unittest

from scripts.healthcheck import metrics_service_ready


class HealthcheckContractTest(unittest.TestCase):
    def status(self) -> dict:
        return {
            "schema_version": 1,
            "service": "learner-metrics",
            "stream": "current",
            "mode": "training",
            "service_instance_id": "metrics-instance-a",
            "metrics_source_id": "training-source-a",
            "started_at": 100.0,
            "latest_sequence": 0,
            "stale": True,
        }

    def test_startup_liveness_does_not_require_actor_ack(self):
        self.assertTrue(metrics_service_ready(self.status()))

    def test_identity_and_mode_fail_closed(self):
        for field, value in (
            ("service", "other-service"),
            ("stream", "archive"),
            ("mode", "local-test"),
            ("service_instance_id", ""),
            ("metrics_source_id", ""),
            ("started_at", 0.0),
            ("started_at", "not-a-number"),
        ):
            with self.subTest(field=field):
                status = self.status()
                status[field] = value
                self.assertFalse(metrics_service_ready(status))


if __name__ == "__main__":
    unittest.main()
