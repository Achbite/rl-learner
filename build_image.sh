#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
contract_root="${workspace_root}/.workspace/artifacts/rl-contracts"
context_dir="${workspace_root}/.workspace/build-contexts/rl-learner-$$"
source "${repo_dir}/artifact_versions.env"
image_name="rl-training/learner"
image_tag="${RL_PROJECT_IMAGE_TAG:-maze-tag-001}"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "RL_PROJECT_IMAGE_TAG is not a valid Docker tag" >&2
    exit 2
fi

image_ref="${image_name}:${image_tag}"
# 运行产物由各上游仓库的 build_artifact.sh 按当前 Docker 服务端平台生成，
# 因此镜像构建同样以实时平台定位产物目录；artifact_versions.env 中的
# RL_RUNTIME_ARTIFACT_PLATFORM 仅作为主力环境的参考默认值，不参与硬校验，
# 否则跨平台开发环境无法复用同一套构建脚本。
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
contract_dir="${contract_root}/${RL_CONTRACTS_VERSION}/${platform_dir}"
sample_pool_dir="${repo_dir}/sample-pool"
model_distributor_dir="${repo_dir}/model-distributor"
if ! test -f "${contract_dir}/python/training_pb2.py"; then
    echo "rl-contracts artifact is missing; run: (cd ../rl-contracts && bash build_artifact.sh)" >&2
    exit 1
fi
if ! test -x "${sample_pool_dir}/bin/maze_sample_pool"; then
    echo "Staged Sample Pool artifact is missing: ${sample_pool_dir}" >&2
    echo "Run: bash scripts/sync_runtime_artifacts.sh" >&2
    exit 1
fi
if ! test -x "${model_distributor_dir}/bin/maze_model_distributor"; then
    echo "Staged Model Distributor artifact is missing: ${model_distributor_dir}" >&2
    echo "Run: bash scripts/sync_runtime_artifacts.sh" >&2
    exit 1
fi

trap 'rm -rf "${context_dir}"' EXIT
mkdir -p "${context_dir}/_deps/contracts" \
    "${context_dir}/_deps/sample-pool/bin" \
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
    --exclude='__pycache__/' \
    "${repo_dir}/" "${context_dir}/"
cp -R "${contract_dir}/python" "${context_dir}/_deps/contracts/python"
cp -R "${contract_dir}/schemas" "${context_dir}/_deps/contracts/schemas"
cp "${sample_pool_dir}/bin/maze_sample_pool" \
    "${context_dir}/_deps/sample-pool/bin/maze_sample_pool"
cp "${sample_pool_dir}/config/pool_config.yaml" \
    "${context_dir}/_deps/sample-pool/config/pool_config.yaml"
cp "${model_distributor_dir}/bin/maze_model_distributor" \
    "${context_dir}/_deps/model-distributor/bin/maze_model_distributor"
cp "${model_distributor_dir}/config/model_distributor_config.yaml" \
    "${context_dir}/_deps/model-distributor/config/model_distributor_config.yaml"

docker build \
    --label "org.rl-training.component=learner" \
    --label "org.rl-training.contracts-version=${RL_CONTRACTS_VERSION}" \
    --label "org.rl-training.component-contract.path=/opt/rl/component-contract/manifest.json" \
    --label "org.rl-training.project-image-tag=${image_tag}" \
    --tag "${image_ref}" \
    "${context_dir}"
printf '%s\n' "${image_ref}"
