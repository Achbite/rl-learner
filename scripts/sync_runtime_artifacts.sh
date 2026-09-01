#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
source "${repo_dir}/artifact_versions.env"

channel="production"
if [ "$#" -gt 1 ]; then
    echo "usage: bash scripts/sync_runtime_artifacts.sh [--development]" >&2
    exit 2
fi
if [ "$#" -eq 1 ]; then
    if [ "$1" != "--development" ]; then
        echo "usage: bash scripts/sync_runtime_artifacts.sh [--development]" >&2
        exit 2
    fi
    channel="development"
fi

platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
if [ "${channel}" = "development" ]; then
    sample_pool_source="${workspace_root}/.workspace/dev-artifacts/rl-sample-pool/${RL_SAMPLE_POOL_VERSION}/${platform_dir}/current"
    model_distributor_source="${workspace_root}/.workspace/dev-artifacts/rl-model-distributor/${RL_MODEL_DISTRIBUTOR_VERSION}/${platform_dir}/current"
else
    sample_pool_source="${workspace_root}/.workspace/artifacts/rl-sample-pool/${RL_SAMPLE_POOL_VERSION}/${platform_dir}"
    model_distributor_source="${workspace_root}/.workspace/artifacts/rl-model-distributor/${RL_MODEL_DISTRIBUTOR_VERSION}/${platform_dir}"
fi
sample_pool_target="${repo_dir}/sample-pool"
model_distributor_target="${repo_dir}/model-distributor"
mkdir -p "${workspace_root}/.workspace"
temp_root="$(mktemp -d "${workspace_root}/.workspace/runtime-artifacts.XXXXXX")"
trap 'rm -rf "${temp_root}"' EXIT

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_source}" \
    --model-distributor-dir "${model_distributor_source}" \
    --training-contract-version "${RL_TRAINING_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}" \
    --channel "${channel}"

mkdir -p \
    "${temp_root}/sample-pool" \
    "${temp_root}/model-distributor"
cp -R "${sample_pool_source}/." "${temp_root}/sample-pool/"
cp -R "${model_distributor_source}/." "${temp_root}/model-distributor/"

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${temp_root}/sample-pool" \
    --model-distributor-dir "${temp_root}/model-distributor" \
    --training-contract-version "${RL_TRAINING_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}" \
    --channel "${channel}"

mkdir -p \
    "${sample_pool_target}/bin" \
    "${sample_pool_target}/config" \
    "${model_distributor_target}/bin" \
    "${model_distributor_target}/config"
rsync -a --delete "${temp_root}/sample-pool/bin/" "${sample_pool_target}/bin/"
rsync -a --delete "${temp_root}/sample-pool/config/" "${sample_pool_target}/config/"
cp "${temp_root}/sample-pool/manifest.json" "${sample_pool_target}/manifest.json"
rsync -a --delete "${temp_root}/model-distributor/bin/" "${model_distributor_target}/bin/"
rsync -a --delete "${temp_root}/model-distributor/config/" "${model_distributor_target}/config/"
cp "${temp_root}/model-distributor/manifest.json" "${model_distributor_target}/manifest.json"

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_target}" \
    --model-distributor-dir "${model_distributor_target}" \
    --training-contract-version "${RL_TRAINING_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}" \
    --channel "${channel}"

printf 'Learner runtime dependencies synchronized: sample-pool=%s model-distributor=%s channel=%s platform=%s\n' \
    "${RL_SAMPLE_POOL_VERSION}" \
    "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    "${channel}" \
    "${platform}"
