#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"

if ! command -v docker >/dev/null 2>&1; then
    echo "prepare_dev_artifacts.sh is a host-side Docker entrypoint" >&2
    exit 1
fi

RL_TRAINING_WORKSPACE="${workspace_root}" \
    bash "${workspace_root}/rl-contracts/build_dev_artifact.sh" training \
    >/dev/null
RL_TRAINING_WORKSPACE="${workspace_root}" \
    bash "${workspace_root}/rl-sample-pool/build_dev_artifact.sh" \
    >/dev/null
RL_TRAINING_WORKSPACE="${workspace_root}" \
    bash "${workspace_root}/rl-model-distributor/build_dev_artifact.sh" \
    >/dev/null
RL_TRAINING_WORKSPACE="${workspace_root}" \
    bash "${repo_dir}/scripts/sync_runtime_artifacts.sh" --development
