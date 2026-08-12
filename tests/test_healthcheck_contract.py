import unittest
from pathlib import Path
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

    def test_optional_monitor_uses_dynamic_vm_port_mapping(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "dev_container.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('--publish "127.0.0.1::9005"', script)
        self.assertNotIn('--publish "127.0.0.1:9005:9005"', script)
        self.assertIn('docker port "${container_name}" 9005/tcp', script)
        self.assertIn(
            'command -v ssh >/dev/null 2>&1; then\n        return 1',
            script,
        )
        self.assertIn(
            "Migrating stopped learner-dev from legacy fixed monitor port",
            script,
        )


if __name__ == "__main__":
    unittest.main()
