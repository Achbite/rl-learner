#!/usr/bin/env python3

import json
import socket
import sys
import urllib.request


def tcp_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def metrics_service_ready(status: dict) -> bool:
    try:
        started_at = float(status.get("started_at", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        status.get("schema_version") == 1
        and status.get("service") == "learner-metrics"
        and status.get("stream") == "current"
        and status.get("mode") == "training"
        and bool(status.get("service_instance_id"))
        and bool(status.get("metrics_source_id"))
        and started_at > 0.0
    )


def main() -> int:
    if not all(tcp_ready(port) for port in (9100, 9200, 9005)):
        return 1
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:9005/api/status", timeout=1
        ) as response:
            status = json.loads(response.read().decode("utf-8"))
    except (OSError, TypeError, ValueError):
        return 1
    return 0 if isinstance(status, dict) and metrics_service_ready(status) else 1


if __name__ == "__main__":
    sys.exit(main())
