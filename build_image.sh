#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
context_dir="${workspace_root}/.workspace/build-contexts/rl-learner-$$"
source "${repo_dir}/artifact_versions.env"
image_name="rl-training/learner"
image_tag="${RL_PROJECT_IMAGE_TAG:-maze-tag-001}"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "RL_PROJECT_IMAGE_TAG is not a valid Docker tag" >&2
    exit 2
fi

platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
sample_pool_dir="${repo_dir}/sample-pool"
model_distributor_dir="${repo_dir}/model-distributor"

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_dir}" \
    --model-distributor-dir "${model_distributor_dir}" \
    --training-contract-version "${RL_TRAINING_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${platform}" \
    --channel production

trap 'rm -rf "${context_dir}"' EXIT
mkdir -p "${context_dir}/_deps/sample-pool/bin" \
    "${context_dir}/_deps/sample-pool/config" \
    "${context_dir}/_deps/model-distributor/bin" \
    "${context_dir}/_deps/model-distributor/config"
rsync -a \
    --exclude='.git/' \
    --exclude='build/' \
    --exclude='logs/' \
    --exclude='models/' \
    --exclude='sample-pool/' \
    --exclude='model-distributor/' \
    --exclude='component-contract/' \
    --exclude='__pycache__/' \
    "${repo_dir}/" "${context_dir}/"
cp "${sample_pool_dir}/bin/maze_sample_pool" \
    "${context_dir}/_deps/sample-pool/bin/maze_sample_pool"
cp "${sample_pool_dir}/config/pool_config.yaml" \
    "${context_dir}/_deps/sample-pool/config/pool_config.yaml"
cp "${sample_pool_dir}/manifest.json" \
    "${context_dir}/_deps/sample-pool/manifest.json"
cp "${model_distributor_dir}/bin/maze_model_distributor" \
    "${context_dir}/_deps/model-distributor/bin/maze_model_distributor"
cp "${model_distributor_dir}/config/model_distributor_config.yaml" \
    "${context_dir}/_deps/model-distributor/config/model_distributor_config.yaml"
cp "${model_distributor_dir}/manifest.json" \
    "${context_dir}/_deps/model-distributor/manifest.json"

image_ref="${image_name}:${image_tag}"
docker build \
    --label "org.rl-training.component=learner" \
    --label "org.rl-training.build-profile=p1-modelrepo" \
    --label "org.rl-training.project-image-tag=${image_tag}" \
    --tag "${image_ref}" \
    "${context_dir}"
printf '%s\n' "${image_ref}"
