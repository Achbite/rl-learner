#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-contracts"
output_dir="${repo_dir}/build/python/proto"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
artifact_dir="${artifact_root}/${RL_CONTRACTS_VERSION}/${platform_dir}"
if ! test -f "${artifact_dir}/python/training_pb2.py"; then
    echo "rl-contracts artifact is missing" >&2
    exit 1
fi
if ! test -f "${artifact_dir}/schemas/maze.metrics.json" ||
   ! test -f "${artifact_dir}/schemas/maze.metrics.sha256"; then
    echo "rl-contracts metric schema artifact is missing" >&2
    exit 1
fi

mkdir -p "${output_dir}" "${output_dir}/schemas"
cp "${artifact_dir}"/python/*.py "${output_dir}/"
cp "${artifact_dir}"/schemas/maze.metrics.json \
   "${artifact_dir}"/schemas/maze.metrics.sha256 \
   "${output_dir}/schemas/"
printf '%s\n' "${output_dir}"
