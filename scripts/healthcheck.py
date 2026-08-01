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


def main() -> int:
    if not all(tcp_ready(port) for port in (9100, 9200, 9005)):
        return 1
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:9005/api/metrics/latest", timeout=1
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return 1

    record = payload.get("record")
    if not isinstance(record, dict):
        return 1
    distributor = record.get("distributor", {})
    model = record.get("model", {})
    learner = record.get("learner", {})
    backend_type = distributor.get("backend_type")
    return 0 if (
        distributor.get("service_name") == "LocalSampleService"
        and distributor.get("ingress_ready") is True
        and distributor.get("pool_ready") is True
        and backend_type == "SAMPLE_BACKEND_TYPE_LOCAL_MEMORY"
        and int(distributor.get("max_concurrent_consumers", 0)) == 1
        and model.get("ready") is True
        and int(model.get("latest_version", -1)) >= 0
        and int(learner.get("model_version", -1)) >= 0
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
