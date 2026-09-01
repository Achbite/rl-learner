#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd -P)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd -P)}"
image_ref="${RL_LEARNER_IMAGE_REF:-rl-training/learner:${RL_PROJECT_IMAGE_TAG:-maze-tag-001}}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-smoke-model"
source "${repo_dir}/artifact_versions.env"
version="${RL_SMOKE_MODEL_VERSION}"
output_dir="${artifact_root}/${version}/any"
mkdir -p "${workspace_root}/.workspace/artifacts"
temp_dir="$(mktemp -d "${workspace_root}/.workspace/artifacts/.tmp-smoke-model.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT

docker run --rm \
    --entrypoint python3 \
    --env PYTHONPATH=/workspace/rl-learner \
    --workdir /tmp \
    --volume "${repo_dir}:/workspace/rl-learner:ro" \
    --volume "${temp_dir}:/output" \
    "${image_ref}" \
    -m main.model_bootstrap \
    --config /workspace/rl-learner/configs/learner_config.yaml \
    --output-root /output

python3 - "${temp_dir}" "${version}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("SaveModel.onnx", "manifest.pb"):
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"smoke-model output is missing or invalid: {path}")

document = {
    "schema_version": "rl.artifact-manifest.v1",
    "artifact_channel": "production",
    "package": "rl-smoke-model",
    "version": sys.argv[2],
    "platform": "any",
    "files": ["SaveModel.onnx", "manifest.pb"],
}
(root / "artifact.json").write_text(
    json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mkdir -p "$(dirname "${output_dir}")"
rm -rf "${output_dir}"
mv "${temp_dir}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
