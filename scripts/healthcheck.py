#!/usr/bin/env python3

import socket
import sys


def tcp_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> int:
    # Metrics on 9005 are an optional observability sidecar. Container health
    # is owned only by the Sample Pool and Model Distributor required by PPO.
    return 0 if all(tcp_ready(port) for port in (9100, 9200)) else 1


if __name__ == "__main__":
    sys.exit(main())
