#!/usr/bin/env bash

set -euo pipefail

requested_image_tag="${RL_LEARNER_IMAGE_TAG:-}"

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
contract_root="${workspace_root}/.workspace/artifacts/rl-contracts"
context_dir="${workspace_root}/.workspace/build-contexts/rl-learner-$$"
source "${repo_dir}/artifact_versions.env"
if test -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=all)"; then
    echo "refusing to build a Learner runtime image from a dirty worktree" >&2
    exit 1
fi
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

python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
    --contract-dir "${contract_dir}" \
    --sample-pool-dir "${sample_pool_dir}" \
    --model-distributor-dir "${model_distributor_dir}" \
    --contract-version "${RL_CONTRACTS_VERSION}" \
    --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
    --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    --platform "${RL_RUNTIME_ARTIFACT_PLATFORM}"

stack_identity_tool="${workspace_root}/rl-framework/tools/compute_stack_source_id.py"
if [ ! -f "${stack_identity_tool}" ]; then
    echo "stack source identity tool is missing: ${stack_identity_tool}" >&2
    exit 1
fi
stack_identity_json="$(
    python3 "${stack_identity_tool}" --workspace-root "${workspace_root}"
)"
identity_fields="$(
    python3 -c '
import json
import sys

document = json.loads(sys.argv[1])
print("\t".join((
    document["stack_source_id"],
    document["repositories"]["rl-learner"],
    document["artifacts"]["rl-contracts"]["artifact_digest"],
    document["artifacts"]["rl-contracts"]["manifest_sha256"],
    document["artifacts"]["rl-sample-pool"]["manifest_sha256"],
    document["artifacts"]["rl-model-distributor"]["manifest_sha256"],
    document["configs"]["learner"]["sha256"],
)))
' "${stack_identity_json}"
)"
IFS=$'\t' read -r \
    stack_source_id component_commit contracts_artifact_digest \
    contracts_manifest_digest sample_pool_manifest_digest \
    model_distributor_manifest_digest component_config_digest \
    <<< "${identity_fields}"
canonical_image_tag="a3-${RL_CONTRACTS_VERSION}-${stack_source_id:0:12}"
if [ -n "${requested_image_tag}" ] &&
   [ "${requested_image_tag}" != "${canonical_image_tag}" ]; then
    echo "Learner image tag must match the canonical stack identity:" >&2
    echo "  expected=${canonical_image_tag}" >&2
    echo "  requested=${requested_image_tag}" >&2
    exit 1
fi
image_ref="rl-training/learner:${canonical_image_tag}"

if docker image inspect "${image_ref}" >/dev/null 2>&1; then
    existing_identity="$(
        docker image inspect --format \
            '{{index .Config.Labels "org.rl-training.stack-source-id"}}|{{index .Config.Labels "org.rl-training.component"}}|{{index .Config.Labels "org.rl-training.component-commit"}}|{{index .Config.Labels "org.rl-training.contracts-version"}}|{{index .Config.Labels "org.rl-training.contracts-artifact-digest"}}|{{index .Config.Labels "org.rl-training.contracts-manifest-digest"}}|{{index .Config.Labels "org.rl-training.component-config-digest"}}|{{index .Config.Labels "org.rl-training.sample-pool-manifest-digest"}}|{{index .Config.Labels "org.rl-training.model-distributor-manifest-digest"}}' \
            "${image_ref}"
    )"
    expected_identity="${stack_source_id}|learner|${component_commit}|${RL_CONTRACTS_VERSION}|${contracts_artifact_digest}|${contracts_manifest_digest}|${component_config_digest}|${sample_pool_manifest_digest}|${model_distributor_manifest_digest}"
    if [ "${existing_identity}" != "${expected_identity}" ]; then
        echo "refusing to overwrite an existing Learner tag with another identity: ${image_ref}" >&2
        exit 1
    fi
    RL_LEARNER_IMAGE_REF="${image_ref}" \
        bash "${repo_dir}/build_smoke_model_artifact.sh" >/dev/null
    printf '%s\n' "${image_ref}"
    exit 0
fi

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
cp -R "${contract_dir}/schemas" "${context_dir}/_deps/contracts/schemas"
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
cp "${contract_dir}/manifest.json" \
    "${context_dir}/_deps/identity/contracts.json"
cp "${sample_pool_dir}/manifest.json" \
    "${context_dir}/_deps/identity/sample-pool.json"
cp "${model_distributor_dir}/manifest.json" \
    "${context_dir}/_deps/identity/model-distributor.json"
printf '%s\n' "${stack_identity_json}" \
    > "${context_dir}/_deps/identity/stack-source.json"

docker build \
    --label "org.opencontainers.image.revision=${component_commit}" \
    --label "org.rl-training.component=learner" \
    --label "org.rl-training.component-commit=${component_commit}" \
    --label "org.rl-training.stack-source-id=${stack_source_id}" \
    --label "org.rl-training.contracts-version=${RL_CONTRACTS_VERSION}" \
    --label "org.rl-training.contracts-artifact-digest=${contracts_artifact_digest}" \
    --label "org.rl-training.contracts-manifest-digest=${contracts_manifest_digest}" \
    --label "org.rl-training.component-config-digest=${component_config_digest}" \
    --label "org.rl-training.sample-pool-manifest-digest=${sample_pool_manifest_digest}" \
    --label "org.rl-training.model-distributor-manifest-digest=${model_distributor_manifest_digest}" \
    --tag "${image_ref}" \
    "${context_dir}"
RL_LEARNER_IMAGE_REF="${image_ref}" \
    bash "${repo_dir}/build_smoke_model_artifact.sh" >/dev/null
printf '%s\n' "${image_ref}"
