#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"

channel="production"
sample_pool_source=""
model_distributor_source=""
usage="usage: bash scripts/sync_runtime_artifacts.sh [--development] [--sample-pool-dir DIR --model-distributor-dir DIR]"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --development)
            channel="development"
            shift
            ;;
        --sample-pool-dir)
            [ "$#" -ge 2 ] || { echo "${usage}" >&2; exit 2; }
            sample_pool_source="$2"
            shift 2
            ;;
        --model-distributor-dir)
            [ "$#" -ge 2 ] || { echo "${usage}" >&2; exit 2; }
            model_distributor_source="$2"
            shift 2
            ;;
        *)
            echo "${usage}" >&2
            exit 2
            ;;
    esac
done

if [ -n "${sample_pool_source}" ] || [ -n "${model_distributor_source}" ]; then
    if [ -z "${sample_pool_source}" ] || [ -z "${model_distributor_source}" ] ||
       [ "${channel}" != "production" ]; then
        echo "${usage}" >&2
        exit 2
    fi
    channel="explicit"
    platform="external"
else
    sample_pool_version="$(tr -d '[:space:]' < "${workspace_root}/rl-sample-pool/VERSION")"
    model_distributor_version="$(tr -d '[:space:]' < "${workspace_root}/rl-model-distributor/VERSION")"
    platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
    platform_dir="${platform//\//-}"
    if [ "${channel}" = "development" ]; then
        sample_pool_source="${workspace_root}/.workspace/dev-artifacts/rl-sample-pool/${sample_pool_version}/${platform_dir}/current"
        model_distributor_source="${workspace_root}/.workspace/dev-artifacts/rl-model-distributor/${model_distributor_version}/${platform_dir}/current"
    else
        sample_pool_source="${workspace_root}/.workspace/artifacts/rl-sample-pool/${sample_pool_version}/${platform_dir}"
        model_distributor_source="${workspace_root}/.workspace/artifacts/rl-model-distributor/${model_distributor_version}/${platform_dir}"
    fi
fi
sample_pool_target="${repo_dir}/sample-pool"
model_distributor_target="${repo_dir}/model-distributor"
mkdir -p "${workspace_root}/.workspace"
temp_root="$(mktemp -d "${workspace_root}/.workspace/runtime-artifacts.XXXXXX")"
trap 'rm -rf "${temp_root}"' EXIT

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_source}" \
    --model-distributor-dir "${model_distributor_source}"

mkdir -p \
    "${temp_root}/sample-pool/bin" \
    "${temp_root}/sample-pool/config" \
    "${temp_root}/model-distributor/bin" \
    "${temp_root}/model-distributor/config"
cp "${sample_pool_source}/bin/maze_sample_pool" \
    "${temp_root}/sample-pool/bin/maze_sample_pool"
cp "${sample_pool_source}/config/pool_config.yaml" \
    "${temp_root}/sample-pool/config/pool_config.yaml"
cp "${model_distributor_source}/bin/maze_model_distributor" \
    "${temp_root}/model-distributor/bin/maze_model_distributor"
cp "${model_distributor_source}/config/model_distributor_config.yaml" \
    "${temp_root}/model-distributor/config/model_distributor_config.yaml"

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${temp_root}/sample-pool" \
    --model-distributor-dir "${temp_root}/model-distributor"

mkdir -p \
    "${sample_pool_target}/bin" \
    "${sample_pool_target}/config" \
    "${model_distributor_target}/bin" \
    "${model_distributor_target}/config"
rsync -a --delete "${temp_root}/sample-pool/bin/" "${sample_pool_target}/bin/"
if [ ! -f "${sample_pool_target}/config/pool_config.yaml" ]; then
    cp "${temp_root}/sample-pool/config/pool_config.yaml" \
        "${sample_pool_target}/config/pool_config.yaml"
fi
rsync -a --delete "${temp_root}/model-distributor/bin/" "${model_distributor_target}/bin/"
if [ ! -f "${model_distributor_target}/config/model_distributor_config.yaml" ]; then
    cp "${temp_root}/model-distributor/config/model_distributor_config.yaml" \
        "${model_distributor_target}/config/model_distributor_config.yaml"
fi

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_target}" \
    --model-distributor-dir "${model_distributor_target}"

printf 'Learner runtime dependencies synchronized: sample-pool=%s model-distributor=%s channel=%s platform=%s\n' \
    "${sample_pool_source}" \
    "${model_distributor_source}" \
    "${channel}" \
    "${platform}"
