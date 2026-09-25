#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
image_name="rl-training/learner"
image_tag="${RL_PROJECT_IMAGE_TAG:-maze-tag-001}"

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "RL_PROJECT_IMAGE_TAG is not a valid Docker tag" >&2
    exit 2
fi

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --sample-pool-dir "${repo_dir}/sample-pool" \
    --model-distributor-dir "${repo_dir}/model-distributor"

image_ref="${image_name}:${image_tag}"
docker build --provenance=false \
    --label "org.rl-training.component=learner" \
    --label "org.rl-training.build-profile=p1-modelrepo" \
    --label "org.rl-training.project-image-tag=${image_tag}" \
    --tag "${image_ref}" \
    "${repo_dir}"
printf '%s\n' "${image_ref}"
