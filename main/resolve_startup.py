"""Resolve the exact Learner startup configuration for the supervisor."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.config.command_line import parse_startup_arguments
from src.config.effective_config import load_effective_config


def resolve(arguments: Sequence[str]) -> dict:
    startup = parse_startup_arguments(arguments)
    config = load_effective_config(
        startup.config_path,
        cli_overrides=startup.cli_overrides,
    )
    return {
        "config_path": config["_effective_config"]["config_path"],
        "local_train_dir": config["model"]["local_train_dir"],
        "initial_model_path": config["model"]["initial_model_path"] or "",
        "sample_host": config["sample_pool"]["host"],
        "sample_port": int(config["sample_pool"]["port"]),
        "model_host": config["model_distributor"]["host"],
        "model_port": int(config["model_distributor"]["port"]),
        "aiserver_host": config["aiserver_status"]["host"],
        "aiserver_port": int(config["aiserver_status"]["port"]),
        "metrics_port": int(config["dashboard"]["server_port"]),
        "metrics_enabled": 1 if config["dashboard"]["enabled"] else 0,
        "metric_event_port": int(config["metric_events"]["server_port"]),
        "contract_package": config["contract"]["package_name"],
        "contract_version": config["contract"]["package_version"],
        "contract_platform": config["contract"]["platform"],
    }


def main() -> int:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--format", choices=("json", "lines"), default="json")
    wrapper.add_argument("arguments", nargs=argparse.REMAINDER)
    selected = wrapper.parse_args()
    forwarded = list(selected.arguments)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    document = resolve(forwarded)
    if selected.format == "json":
        print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    else:
        for field in (
            "config_path",
            "local_train_dir",
            "initial_model_path",
            "sample_host",
            "sample_port",
            "model_host",
            "model_port",
            "aiserver_host",
            "aiserver_port",
            "metrics_port",
            "metrics_enabled",
            "metric_event_port",
            "contract_package",
            "contract_version",
            "contract_platform",
        ):
            value = str(document[field])
            if "\n" in value or "\r" in value:
                raise ValueError(f"startup value contains a newline: {field}")
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
