#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def require_runtime_file(
    root: Path,
    relative: str,
    *,
    executable: bool = False,
) -> None:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Required runtime file is missing: {path}")
    if executable and not os.access(path, os.X_OK):
        raise SystemExit(f"Required runtime executable is not executable: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-pool-dir", type=Path, required=True)
    parser.add_argument("--model-distributor-dir", type=Path, required=True)
    args = parser.parse_args()
    require_runtime_file(
        args.sample_pool_dir,
        "bin/maze_sample_pool",
        executable=True,
    )
    require_runtime_file(args.sample_pool_dir, "config/pool_config.yaml")
    require_runtime_file(
        args.model_distributor_dir,
        "bin/maze_model_distributor",
        executable=True,
    )
    require_runtime_file(
        args.model_distributor_dir,
        "config/model_distributor_config.yaml",
    )
    print("required Sample Pool and Model Distributor runtime files are present")


if __name__ == "__main__":
    main()
