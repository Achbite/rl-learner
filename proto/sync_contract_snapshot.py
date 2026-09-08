#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
from pathlib import Path


SNAPSHOT_FILES = {'training': {'python/proto/common/identity_pb2.py': 'common/identity_pb2.py',
              'python/proto/training/model_identity_pb2.py': 'training/model_identity_pb2.py',
              'python/proto/training/training_pb2.py': 'training/training_pb2.py',
              'python/proto/metrics/registry_pb2.py': 'metrics/registry_pb2.py',
              'python/proto/metrics/catalog_pb2.py': 'metrics/catalog_pb2.py',
              'python/proto/metrics/transport_pb2.py': 'metrics/transport_pb2.py',
              'python/proto/metrics/training_pb2.py': 'metrics/training_pb2.py',
              'python/proto/training/training_pb2_grpc.py': 'training/training_pb2_grpc.py',
              'python/proto/metrics/catalog_pb2_grpc.py': 'metrics/catalog_pb2_grpc.py',
              'python/proto/metrics/transport_pb2_grpc.py': 'metrics/transport_pb2_grpc.py',
              'python/proto/common/__init__.py': 'common/__init__.py',
              'python/proto/training/__init__.py': 'training/__init__.py',
              'python/proto/metrics/__init__.py': 'metrics/__init__.py'}}


def require_regular_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"required protocol file is missing: {path}")


def sync_snapshot(artifact_root: Path, target_root: Path, profile: str) -> None:
    files = SNAPSHOT_FILES[profile]
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".protocol-files-", dir=target_root.parent
    ) as temporary:
        stage = Path(temporary)
        for artifact_name, local_name in files.items():
            source = artifact_root / artifact_name
            require_regular_file(source)
            staged = stage / local_name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)
        for local_name in files.values():
            target = target_root / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / local_name, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly synchronize Learner protocol bindings"
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(SNAPSHOT_FILES),
        required=True,
    )
    args = parser.parse_args()
    sync_snapshot(
        args.artifact_dir.resolve(),
        args.target_dir.resolve(),
        args.profile,
    )
    print(f"{args.profile} Learner protocol files synchronized")


if __name__ == "__main__":
    main()
