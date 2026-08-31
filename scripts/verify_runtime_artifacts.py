#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"[a-f0-9]{64}")
SOURCE_COMMIT = re.compile(r"[a-f0-9]{12}")


def load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Artifact manifest is invalid: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SystemExit(f"Artifact manifest must be an object: {manifest_path}")
    return manifest


def verify_files(root: Path, manifest: dict[str, Any], expected: set[str] | None = None) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"Artifact file inventory is invalid: {root / 'manifest.json'}")
    if expected is not None and set(files) != expected:
        raise SystemExit(f"Artifact file inventory is unexpected: {root / 'manifest.json'}")
    for relative, expected_checksum in files.items():
        if not isinstance(relative, str) or not isinstance(expected_checksum, str):
            raise SystemExit(f"Artifact file inventory is invalid: {root / 'manifest.json'}")
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Artifact file is missing: {path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise SystemExit(f"Artifact checksum mismatch: {path}")


def verify_contract(root: Path, version: str, platform: str) -> dict[str, Any]:
    manifest = load_manifest(root)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("package") != "rl-contracts"
        or manifest.get("version") != version
        or manifest.get("platform") != platform
        or manifest.get("source_tree_state") != "clean"
        or SOURCE_COMMIT.fullmatch(
            str(manifest.get("source_commit", ""))
        ) is None
    ):
        raise SystemExit("Contract artifact identity is invalid")
    verify_files(root, manifest)
    schema = manifest.get("metric_schemas", {}).get("maze.metrics")
    catalog_path = root / "schemas/maze.metrics.json"
    digest_path = root / "schemas/maze.metrics.sha256"
    if not catalog_path.is_file() or not digest_path.is_file():
        raise SystemExit("Contract metric schema artifact is missing")
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    if digest_path.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit("Contract metric schema digest is invalid")
    if schema != {
        "canonical_digest": {"algorithm": "sha256", "hex": digest},
        "digest_path": "schemas/maze.metrics.sha256",
        "path": "schemas/maze.metrics.json",
        "schema_version": 1,
    }:
        raise SystemExit("Contract metric schema manifest binding is invalid")
    return manifest


def verify_component(
    root: Path,
    package: str,
    version: str,
    platform: str,
    contract: dict[str, Any],
    expected_files: set[str],
) -> None:
    manifest = load_manifest(root)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("package") != package
        or manifest.get("version") != version
        or manifest.get("platform") != platform
    ):
        raise SystemExit(f"{package} artifact identity is invalid")

    source_commit = str(manifest.get("source_commit", ""))
    source_id = str(manifest.get("source_id", ""))
    source_sha256 = str(manifest.get("source_sha256", ""))
    if (
        SOURCE_COMMIT.fullmatch(source_commit) is None
        or source_id != source_commit
        or SHA256.fullmatch(source_sha256) is None
    ):
        raise SystemExit(f"{package} artifact provenance is not a clean savepoint")

    expected_contract = {
        "version": contract["version"],
        "source_digest": contract["source_digest"],
        "artifact_digest": contract["artifact_digest"],
        "platform": contract["platform"],
        "generator_identity": contract["generator_identity"],
    }
    if manifest.get("contract") != expected_contract:
        raise SystemExit(f"{package} artifact uses a different contract artifact")
    verify_files(root, manifest, expected_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--sample-pool-dir", type=Path, required=True)
    parser.add_argument("--model-distributor-dir", type=Path, required=True)
    parser.add_argument("--contract-version", required=True)
    parser.add_argument("--sample-pool-version", required=True)
    parser.add_argument("--model-distributor-version", required=True)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()

    contract = verify_contract(args.contract_dir, args.contract_version, args.platform)
    verify_component(
        args.sample_pool_dir,
        "rl-sample-pool",
        args.sample_pool_version,
        args.platform,
        contract,
        {"bin/maze_sample_pool", "config/pool_config.yaml"},
    )
    verify_component(
        args.model_distributor_dir,
        "rl-model-distributor",
        args.model_distributor_version,
        args.platform,
        contract,
        {"bin/maze_model_distributor", "config/model_distributor_config.yaml"},
    )

    print(
        "runtime artifacts verified: "
        f"contracts={args.contract_version}, "
        f"sample-pool={args.sample_pool_version}, "
        f"model-distributor={args.model_distributor_version}, "
        f"platform={args.platform}"
    )


if __name__ == "__main__":
    main()
