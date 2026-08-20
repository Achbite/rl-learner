"""Canonical Learner startup command line.

Business options only select overrides for fields that already exist in the
selected YAML configuration. Both the supervisor and TrainingRuntime use this
module so an option cannot acquire different meanings at the two layers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[2] / "configs" / "learner_config.yaml"
)


@dataclass(frozen=True)
class LearnerStartupArguments:
    config_path: str
    cli_overrides: dict[tuple[str, ...], object]


def _address(option: str, value: str) -> tuple[str, int]:
    if value == "" or value != value.strip() or ":" not in value:
        raise argparse.ArgumentTypeError(f"{option} requires host:port")
    host, raw_port = value.rsplit(":", 1)
    if not host or any(character.isspace() for character in host):
        raise argparse.ArgumentTypeError(f"{option} host is invalid")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option} port must be an integer"
        ) from error
    if str(port) != raw_port or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"{option} port must be in [1, 65535]"
        )
    return host, port


def _port(option: str, value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{option} must be an integer"
        ) from error
    if str(port) != value or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            f"{option} must be in [1, 65535]"
        )
    return port


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Start the task-neutral Learner training runtime"
    )
    result.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="select the YAML config file (meta option)",
    )
    result.add_argument(
        "--initial-model",
        help="override model.initial_model_path",
    )
    result.add_argument(
        "--model-distributor",
        type=lambda value: _address("--model-distributor", value),
        metavar="HOST:PORT",
        help="override model_distributor.host/port",
    )
    result.add_argument(
        "--aiserver",
        type=lambda value: _address("--aiserver", value),
        metavar="HOST:PORT",
        help="override aiserver_status.host/port",
    )
    result.add_argument(
        "--metrics-port",
        type=lambda value: _port("--metrics-port", value),
        metavar="PORT",
        help="override dashboard.server_port",
    )
    return result


def parse_startup_arguments(
    arguments: Sequence[str] | None = None,
) -> LearnerStartupArguments:
    parsed = parser().parse_args(arguments)
    overrides: dict[tuple[str, ...], object] = {}
    if parsed.initial_model is not None:
        overrides[("model", "initial_model_path")] = parsed.initial_model
    if parsed.model_distributor is not None:
        host, port = parsed.model_distributor
        overrides[("model_distributor", "host")] = host
        overrides[("model_distributor", "port")] = port
    if parsed.aiserver is not None:
        host, port = parsed.aiserver
        overrides[("aiserver_status", "host")] = host
        overrides[("aiserver_status", "port")] = port
    if parsed.metrics_port is not None:
        overrides[("dashboard", "server_port")] = parsed.metrics_port
    return LearnerStartupArguments(
        config_path=str(parsed.config),
        cli_overrides=overrides,
    )
