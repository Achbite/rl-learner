#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_SCHEMA = "rl.artifact-manifest.v1"
TRAINING_ARTIFACT_PACKAGE = "rl-training-contracts"


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


def artifact_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise SystemExit(f"Artifact path is invalid: {relative!r}")
    return root / candidate


def verify_files(
    root: Path,
    manifest: dict[str, Any],
    *,
    required_files: set[str],
    required_executables: set[str] = frozenset(),
) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files or not all(
        isinstance(value, str) for value in files
    ):
        raise SystemExit(f"Artifact file list is invalid: {root / 'manifest.json'}")
    missing_declarations = required_files - set(files)
    if missing_declarations:
        raise SystemExit(
            f"Artifact does not declare required files: {sorted(missing_declarations)}"
        )
    for relative in files:
        path = artifact_path(root, relative)
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Artifact file is missing or invalid: {path}")
    for relative in required_executables:
        path = artifact_path(root, relative)
        if not os.access(path, os.X_OK):
            raise SystemExit(f"Artifact executable is not executable: {path}")


def verify_component(
    root: Path,
    package: str,
    version: str,
    platform: str,
    training_contract_version: str,
    channel: str,
    required_files: set[str],
    executable: str,
) -> None:
    manifest = load_manifest(root)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("package") != package
        or manifest.get("version") != version
        or manifest.get("platform") != platform
        or manifest.get("artifact_channel") != channel
    ):
        raise SystemExit(f"{package} artifact type is incompatible")
    expected_contract = {
        "package": TRAINING_ARTIFACT_PACKAGE,
        "version": training_contract_version,
    }
    if manifest.get("contract") != expected_contract:
        raise SystemExit(f"{package} uses an incompatible Contracts package")
    verify_files(
        root,
        manifest,
        required_files=required_files,
        required_executables={executable},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-pool-dir", type=Path, required=True)
    parser.add_argument("--model-distributor-dir", type=Path, required=True)
    parser.add_argument("--training-contract-version", required=True)
    parser.add_argument("--sample-pool-version", required=True)
    parser.add_argument("--model-distributor-version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--channel", choices=("production", "development"), required=True
    )
    args = parser.parse_args()

    verify_component(
        args.sample_pool_dir,
        "rl-sample-pool",
        args.sample_pool_version,
        args.platform,
        args.training_contract_version,
        args.channel,
        {"bin/maze_sample_pool", "config/pool_config.yaml"},
        "bin/maze_sample_pool",
    )
    verify_component(
        args.model_distributor_dir,
        "rl-model-distributor",
        args.model_distributor_version,
        args.platform,
        args.training_contract_version,
        args.channel,
        {
            "bin/maze_model_distributor",
            "config/model_distributor_config.yaml",
        },
        "bin/maze_model_distributor",
    )

    print(
        "runtime artifact types verified: "
        f"training-contracts={args.training_contract_version}, "
        f"sample-pool={args.sample_pool_version}, "
        f"model-distributor={args.model_distributor_version}, "
        f"platform={args.platform}"
    )


if __name__ == "__main__":
    main()
