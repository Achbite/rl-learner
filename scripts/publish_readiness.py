#!/usr/bin/env python3
"""Publish one bounded business-readiness receipt for the managed container."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


RECEIPT_PATH = Path("/run/rl/readiness.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        required=True,
        choices=("learner", "aiserver", "client"),
    )
    parser.add_argument("--fact", action="append", default=[])
    arguments = parser.parse_args()

    facts: dict[str, str] = {}
    for value in arguments.fact:
        key, separator, selected = value.partition("=")
        if (
            separator == ""
            or key == ""
            or key in facts
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in key)
        ):
            raise SystemExit(f"invalid readiness fact: {value}")
        facts[key] = selected

    document = {
        "schema_version": "rl.component-readiness.v2",
        "component": arguments.component,
        "ready": True,
        "facts": facts,
        "published_at_unix_ms": int(time.time() * 1000),
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT_PATH.with_name(
        f".{RECEIPT_PATH.name}.tmp.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(document, output, separators=(",", ":"), sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, RECEIPT_PATH)


if __name__ == "__main__":
    main()
