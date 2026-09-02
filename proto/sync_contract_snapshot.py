#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
from pathlib import Path


SNAPSHOT_FILES = {
    "training": {
        "python/common_pb2.py": "common_pb2.py",
        "python/training_pb2.py": "training_pb2.py",
        "python/training_pb2_grpc.py": "training_pb2_grpc.py",
        "python/training_metrics_pb2.py": "training_metrics_pb2.py",
    },
    "task-maze": {
        "python/maze_metrics_pb2.py": "maze_metrics_pb2.py",
    },
}


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
            shutil.copyfile(source, stage / local_name)
        for local_name in files.values():
            os.replace(stage / local_name, target_root / local_name)


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
