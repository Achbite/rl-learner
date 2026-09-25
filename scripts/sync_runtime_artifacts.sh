#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sample_pool_source=""
model_distributor_source=""
usage="usage: bash scripts/sync_runtime_artifacts.sh --sample-pool-dir DIR --model-distributor-dir DIR"
while [ "$#" -gt 0 ]; do
    case "$1" in
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

if [ -z "${sample_pool_source}" ] || [ -z "${model_distributor_source}" ]; then
    echo "${usage}" >&2
    exit 2
fi
sample_pool_target="${repo_dir}/sample-pool"
model_distributor_target="${repo_dir}/model-distributor"
python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${sample_pool_source}" \
    --model-distributor-dir "${model_distributor_source}"

mkdir -p \
    "${sample_pool_target}/bin" \
    "${sample_pool_target}/config" \
    "${model_distributor_target}/bin" \
    "${model_distributor_target}/config"
rsync -rcp --delete "${sample_pool_source}/bin/" "${sample_pool_target}/bin/"
if [ ! -f "${sample_pool_target}/config/pool_config.yaml" ]; then
    cp "${sample_pool_source}/config/pool_config.yaml" \
        "${sample_pool_target}/config/pool_config.yaml"
fi
rsync -rcp --delete "${model_distributor_source}/bin/" "${model_distributor_target}/bin/"
if [ ! -f "${model_distributor_target}/config/model_distributor_config.yaml" ]; then
    cp "${model_distributor_source}/config/model_distributor_config.yaml" \
        "${model_distributor_target}/config/model_distributor_config.yaml"
fi

printf 'Learner runtime dependencies synchronized: sample-pool=%s model-distributor=%s\n' \
    "${sample_pool_source}" \
    "${model_distributor_source}"
