#!/usr/bin/env bash

set -euo pipefail

LEARNER_IMAGE_TAG="${RL_LEARNER_IMAGE_TAG:-training-001}"

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
contract_root="${workspace_root}/.workspace/artifacts/rl-contracts"
context_dir="${workspace_root}/.workspace/build-contexts/rl-learner-$$"
image_ref="rl-training/learner:${LEARNER_IMAGE_TAG}"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
if test "${platform}" != "${RL_RUNTIME_ARTIFACT_PLATFORM}"; then
    echo "Docker platform does not match the selected runtime artifact platform:" >&2
    echo "  docker=${platform}" >&2
    echo "  selected=${RL_RUNTIME_ARTIFACT_PLATFORM}" >&2
    exit 1
fi
platform_dir="${RL_RUNTIME_ARTIFACT_PLATFORM//\//-}"
contract_dir="${contract_root}/${RL_CONTRACTS_VERSION}/${platform_dir}"
sample_pool_dir="${repo_dir}/sample-pool"
model_distributor_dir="${repo_dir}/model-distributor"
if ! test -f "${contract_dir}/python/training_pb2.py"; then
    echo "rl-contracts artifact is missing; run: (cd ../rl-contracts && bash build_artifact.sh)" >&2
    exit 1
fi
if ! test -x "${sample_pool_dir}/bin/maze_sample_distributor"; then
    echo "Staged Sample Pool artifact is missing: ${sample_pool_dir}" >&2
    echo "Run: bash scripts/sync_runtime_artifacts.sh" >&2
    exit 1
fi
if ! test -x "${model_distributor_dir}/bin/maze_model_distributor"; then
    echo "Staged Model Distributor artifact is missing: ${model_distributor_dir}" >&2
    echo "Run: bash scripts/sync_runtime_artifacts.sh" >&2
    exit 1
fi

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --contract-dir "${contract_dir}" \
    --sample-pool-dir "${sample_pool_dir}" \
    --model-distributor-dir "${model_distributor_dir}" \
    --contract-version "${RL_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${RL_RUNTIME_ARTIFACT_PLATFORM}"

mkdir -p "${context_dir}/_deps/contracts" \
    "${context_dir}/_deps/sample-pool/bin" \
    "${context_dir}/_deps/sample-pool/config" \
    "${context_dir}/_deps/model-distributor/bin" \
    "${context_dir}/_deps/model-distributor/config" \
    "${context_dir}/_deps/identity"
trap 'rm -rf "${context_dir}"' EXIT
rsync -a \
    --exclude='.git/' \
    --exclude='build/' \
    --exclude='logs/' \
    --exclude='models/' \
    --exclude='sample-pool/' \
    --exclude='model-distributor/' \
    --exclude='__pycache__/' \
    "${repo_dir}/" "${context_dir}/"
cp -R "${contract_dir}/python" "${context_dir}/_deps/contracts/python"
cp "${sample_pool_dir}/bin/maze_sample_distributor" \
    "${context_dir}/_deps/sample-pool/bin/maze_sample_distributor"
cp "${sample_pool_dir}/config/distributor_config.yaml" \
    "${context_dir}/_deps/sample-pool/config/distributor_config.yaml"
cp "${sample_pool_dir}/manifest.json" \
    "${context_dir}/_deps/sample-pool/manifest.json"
cp "${model_distributor_dir}/bin/maze_model_distributor" \
    "${context_dir}/_deps/model-distributor/bin/maze_model_distributor"
cp "${model_distributor_dir}/config/model_distributor_config.yaml" \
    "${context_dir}/_deps/model-distributor/config/model_distributor_config.yaml"
cp "${model_distributor_dir}/manifest.json" \
    "${context_dir}/_deps/model-distributor/manifest.json"
cp "${contract_dir}/manifest.json" \
    "${context_dir}/_deps/identity/contracts.json"
cp "${sample_pool_dir}/manifest.json" \
    "${context_dir}/_deps/identity/sample-pool.json"
cp "${model_distributor_dir}/manifest.json" \
    "${context_dir}/_deps/identity/model-distributor.json"

docker build --tag "${image_ref}" "${context_dir}"
RL_LEARNER_IMAGE_REF="${image_ref}" \
    bash "${repo_dir}/build_smoke_model_artifact.sh" >/dev/null
printf '%s\n' "${image_ref}"
