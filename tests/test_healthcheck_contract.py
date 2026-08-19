import unittest
from unittest.mock import patch

from scripts.healthcheck import main


class HealthcheckContractTest(unittest.TestCase):
    def test_optional_monitor_port_does_not_own_container_health(self):
        with patch(
            "scripts.healthcheck.tcp_ready",
            side_effect=lambda port: port in (9100, 9200),
        ) as ready:
            self.assertEqual(main(), 0)
        self.assertEqual(
            [call.args[0] for call in ready.call_args_list],
            [9100, 9200],
        )

    def test_training_dependency_port_failure_is_unhealthy(self):
        for failed_port in (9100, 9200):
            with self.subTest(failed_port=failed_port), patch(
                "scripts.healthcheck.tcp_ready",
                side_effect=lambda port, failed=failed_port: port != failed,
            ):
                self.assertEqual(main(), 1)
