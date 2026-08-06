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
platform_dir="${platform//\//-}"
contract_dir="${contract_root}/${RL_CONTRACTS_VERSION}/${platform_dir}"
sample_pool_dir="${repo_dir}/sample-pool"
model_distributor_dir="${repo_dir}/model-distributor"
if ! test -f "${contract_dir}/python/training_pb2.py"; then
    echo "rl-contracts artifact is missing; run: (cd ../rl-contracts && bash build_artifact.sh)" >&2
    exit 1
fi
if ! test -x "${sample_pool_dir}/bin/maze_sample_distributor"; then
    echo "Staged Sample Pool artifact is missing: ${sample_pool_dir}" >&2
    echo "Build it in rl-sample-pool, then copy the selected version here explicitly:" >&2
    echo "  cp -R ../.workspace/artifacts/rl-sample-pool/${RL_SAMPLE_POOL_VERSION}/${platform_dir}/. sample-pool/" >&2
    exit 1
fi
if ! test -x "${model_distributor_dir}/bin/maze_model_distributor"; then
    echo "Staged Model Distributor artifact is missing: ${model_distributor_dir}" >&2
    echo "Build it in rl-model-distributor, then copy the selected version here explicitly:" >&2
    echo "  cp -R ../.workspace/artifacts/rl-model-distributor/${RL_MODEL_DISTRIBUTOR_VERSION}/${platform_dir}/. model-distributor/" >&2
    exit 1
fi

python3 - \
    "${contract_dir}/manifest.json" \
    "${sample_pool_dir}/manifest.json" \
    "${model_distributor_dir}/manifest.json" \
    "${RL_CONTRACTS_VERSION}" \
    "${RL_SAMPLE_POOL_VERSION}" \
    "${RL_MODEL_DISTRIBUTOR_VERSION}" \
    "${platform}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

contract_path = Path(sys.argv[1])
sample_pool_path = Path(sys.argv[2])
distributor_path = Path(sys.argv[3])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
sample_pool = json.loads(sample_pool_path.read_text(encoding="utf-8"))
distributor = json.loads(distributor_path.read_text(encoding="utf-8"))

def verify_files(root, manifest):
    for relative, expected_checksum in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Artifact file is missing: {path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise SystemExit(f"Artifact checksum mismatch: {path}")

if contract.get("package") != "rl-contracts" or contract.get("version") != sys.argv[4]:
    raise SystemExit("Contract artifact identity is invalid")
if contract.get("platform") != sys.argv[7]:
    raise SystemExit("Contract artifact platform is invalid")
verify_files(contract_path.parent, contract)
if sample_pool.get("package") != "rl-sample-pool" or sample_pool.get("version") != sys.argv[5]:
    raise SystemExit("LocalSampleService artifact identity is invalid")
if sample_pool.get("platform") != sys.argv[7]:
    raise SystemExit("LocalSampleService artifact platform is invalid")
verify_files(sample_pool_path.parent, sample_pool)
if distributor.get("package") != "rl-model-distributor" or distributor.get("version") != sys.argv[6]:
    raise SystemExit("ModelDistributor artifact identity is invalid")
if distributor.get("platform") != sys.argv[7]:
    raise SystemExit("ModelDistributor artifact platform is invalid")
verify_files(distributor_path.parent, distributor)
expected = (
    contract["version"],
    contract["source_digest"]["hex"],
    contract["artifact_digest"]["hex"],
    contract["generator_identity"],
    contract["platform"],
)
for name, manifest in (
    ("Sample Pool", sample_pool),
    ("Model Distributor", distributor),
):
    actual = (
        manifest["contract"]["version"],
        manifest["contract"]["source_digest"]["hex"],
        manifest["contract"]["artifact_digest"]["hex"],
        manifest["contract"]["generator_identity"],
        manifest["contract"]["platform"],
    )
    if actual != expected:
        raise SystemExit(
            f"{name} artifact was built against a different contract artifact"
        )
PY

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
