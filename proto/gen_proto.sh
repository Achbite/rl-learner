#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-contracts"
output_dir="${repo_dir}/build/python/proto"
source "${repo_dir}/artifact_versions.env"
training_dir="${artifact_root}/${RL_TRAINING_CONTRACTS_VERSION}/training"
task_dir="${artifact_root}/${RL_TRAINING_CONTRACTS_VERSION}/task-maze"
if ! test -f "${training_dir}/python/training_pb2.py" ||
   ! test -f "${task_dir}/python/maze_task_pb2.py"; then
    echo "training and Maze task protocol artifacts are required" >&2
    exit 1
fi
if ! test -f "${task_dir}/schemas/maze.episode.metrics.json" ||
   ! test -f "${training_dir}/schemas/training.metrics.json" ||
   ! test -f "${task_dir}/schemas/training-contract.json"; then
    echo "protocol schema artifacts are incomplete" >&2
    exit 1
fi

mkdir -p "${output_dir}" "${output_dir}/schemas"
cp "${training_dir}"/python/*.py "${output_dir}/"
cp "${task_dir}"/python/*.py "${output_dir}/"
cp "${task_dir}"/schemas/maze.episode.metrics.json \
   "${training_dir}"/schemas/training.metrics.json \
   "${task_dir}"/schemas/training-contract.json \
   "${output_dir}/schemas/"
printf '%s\n' "${output_dir}"
