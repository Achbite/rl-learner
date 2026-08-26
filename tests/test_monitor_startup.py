import unittest
from pathlib import Path

from src.config.command_line import parse_startup_arguments
from src.config.effective_config import load_effective_config


CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "learner_config.yaml"
)
LINEAGE = "maze-model-local-monitor-test"


class LocalMonitorStartupTest(unittest.TestCase):
    def test_run_flags_override_the_local_monitor(self):
        enabled = parse_startup_arguments(["--monitor"])
        disabled = parse_startup_arguments(["--no-monitor"])
        default = parse_startup_arguments([])

        self.assertIs(
            enabled.cli_overrides[("dashboard", "enabled")], True
        )
        self.assertIs(
            disabled.cli_overrides[("dashboard", "enabled")], False
        )
        self.assertNotIn(("dashboard", "enabled"), default.cli_overrides)

    def test_infra_environment_disables_only_the_local_monitor(self):
        environment = {
            "RL_MODEL_LINEAGE_ID": LINEAGE,
            "RL_LEARNER_LOCAL_MONITOR_ENABLED": "false",
        }
        disabled = load_effective_config(CONFIG_PATH, environment=environment)
        enabled = load_effective_config(
            CONFIG_PATH,
            environment=environment,
            cli_overrides={("dashboard", "enabled"): True},
        )

        self.assertIs(disabled["dashboard"]["enabled"], False)
        self.assertIs(enabled["dashboard"]["enabled"], True)
        self.assertEqual(disabled["metric_events"], enabled["metric_events"])
        self.assertEqual(disabled["training"], enabled["training"])
        self.assertEqual(
            disabled["dashboard"]["server_port"],
            enabled["dashboard"]["server_port"],
        )
        with self.assertRaisesRegex(
            ValueError,
            "RL_LEARNER_LOCAL_MONITOR_ENABLED must be exactly true or false",
        ):
            load_effective_config(
                CONFIG_PATH,
                environment={
                    "RL_MODEL_LINEAGE_ID": LINEAGE,
                    "RL_LEARNER_LOCAL_MONITOR_ENABLED": "0",
                },
            )


if __name__ == "__main__":
    unittest.main()
