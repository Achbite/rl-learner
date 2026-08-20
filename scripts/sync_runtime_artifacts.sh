#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
source "${repo_dir}/artifact_versions.env"

platform="${RL_RUNTIME_ARTIFACT_PLATFORM}"
platform_dir="${platform//\//-}"
contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${RL_CONTRACTS_VERSION}/${platform_dir}"
sample_pool_source="${workspace_root}/.workspace/artifacts/rl-sample-pool/${RL_SAMPLE_POOL_VERSION}/${platform_dir}"
model_distributor_source="${workspace_root}/.workspace/artifacts/rl-model-distributor/${RL_MODEL_DISTRIBUTOR_VERSION}/${platform_dir}"
sample_pool_target="${repo_dir}/sample-pool"
model_distributor_target="${repo_dir}/model-distributor"
temp_root="$(mktemp -d "${workspace_root}/.workspace/runtime-artifacts.XXXXXX")"
trap 'rm -rf "${temp_root}"' EXIT

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --contract-dir "${contract_dir}" \
    --sample-pool-dir "${sample_pool_source}" \
    --model-distributor-dir "${model_distributor_source}" \
    --contract-version "${RL_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}"

mkdir -p "${temp_root}/sample-pool" "${temp_root}/model-distributor"
cp -R "${sample_pool_source}/." "${temp_root}/sample-pool/"
cp -R "${model_distributor_source}/." "${temp_root}/model-distributor/"

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --contract-dir "${contract_dir}" \
    --sample-pool-dir "${temp_root}/sample-pool" \
    --model-distributor-dir "${temp_root}/model-distributor" \
    --contract-version "${RL_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}"

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
    --contract-dir "${contract_dir}" \
    --sample-pool-dir "${sample_pool_target}" \
    --model-distributor-dir "${model_distributor_target}" \
    --contract-version "${RL_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}"
