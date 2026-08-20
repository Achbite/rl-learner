"""Reset the exact Learner-owned training workspace for a fresh invocation."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Sequence


def reset_training_workspace(path: str | Path) -> Path:
    """Remove every entry below one validated ``.../train`` directory.

    The configured directory is the ownership boundary.  The directory itself
    is retained, symbolic-link roots are rejected, and symbolic-link children
    are unlinked rather than followed.
    """

    requested = Path(path).expanduser()
    if requested.name != "train":
        raise ValueError("Learner local train root must end with /train")
    if requested.is_symlink():
        raise ValueError("Learner local train root must not be a symbolic link")
    if requested.exists() and not requested.is_dir():
        raise ValueError("Learner local train root must be a directory")

    requested.mkdir(parents=True, exist_ok=True)
    root = requested.resolve(strict=True)
    if root.name != "train" or root.parent == Path(root.anchor):
        raise ValueError("Learner local train root is too broad to reset")

    children = tuple(root.iterdir())
    for directory, names, _files in os.walk(
        root, topdown=True, followlinks=False
    ):
        parent = Path(directory)
        traversable: list[str] = []
        for name in names:
            child = parent / name
            if child.is_symlink():
                continue
            if child.is_mount():
                raise ValueError(
                    "Learner local train root contains a mounted directory: "
                    f"{child}"
                )
            traversable.append(name)
        names[:] = traversable

    for child in children:
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)

    if next(root.iterdir(), None) is not None:
        raise RuntimeError(f"Learner local train root reset is incomplete: {root}")

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return root


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reset one validated Learner training workspace"
    )
    parser.add_argument("--path", required=True)
    selected = parser.parse_args(arguments)
    root = reset_training_workspace(selected.path)
    print(f"Learner fresh workspace reset: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
