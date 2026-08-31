#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
image_ref="${RL_LEARNER_IMAGE_REF:-rl-training/learner:${RL_PROJECT_IMAGE_TAG:-maze-tag-001}}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-smoke-model"
source "${repo_dir}/artifact_versions.env"
version="${RL_SMOKE_MODEL_VERSION}"
source_commit="$(git -C "${repo_dir}" rev-parse --short=12 HEAD 2>/dev/null || printf 'unborn')"
source_sha256="$(
    python3 - "${repo_dir}" \
        artifact_versions.env \
        build_smoke_model_artifact.sh \
        Dockerfile \
        requirements.txt \
        configs/learner_config.yaml \
        main/model_bootstrap.py \
        main/training_runtime.py \
        proto/common_pb2.py \
        proto/training_pb2.py \
        src/contracts/identity.py \
        src/log/logger.py \
        src/training/ppo_trainer.py <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for relative in sorted(sys.argv[2:]):
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"smoke-model source input is missing: {path}")
    digest.update(relative.encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"
# The smoke model is owned by the model-definition inputs above. Unrelated
# monitor, documentation, or launcher edits must not invalidate its identity.
source_id="model-source-${source_sha256:0:16}"
output_dir="${artifact_root}/${version}/any"
if test -d "${output_dir}"; then
    if PACKAGE_VERSION="${version}" \
       SOURCE_ID="${source_id}" \
       SOURCE_SHA256="${source_sha256}" \
       python3 - "${output_dir}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifact_path = root / "artifact.json"
if not artifact_path.is_file():
    raise SystemExit(1)
artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
expected = {
    "package": "rl-smoke-model",
    "version": os.environ["PACKAGE_VERSION"],
    "source_id": os.environ["SOURCE_ID"],
    "source_sha256": os.environ["SOURCE_SHA256"],
    "platform": "any",
}
if any(artifact.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
files = artifact.get("files", {})
if set(files) != {"SaveModel.onnx", "manifest.pb"}:
    raise SystemExit(1)
for name, expected_digest in files.items():
    path = root / name
    if not path.is_file():
        raise SystemExit(1)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise SystemExit(1)
PY
    then
        printf '%s\n' "${output_dir}"
        exit 0
    fi
    echo "smoke-model artifact version already exists with different content: ${output_dir}" >&2
    echo "remove that generated artifact explicitly or increment RL_SMOKE_MODEL_VERSION" >&2
    exit 1
fi

temp_dir="${workspace_root}/.workspace/artifacts/.tmp-smoke-model-$$"
mkdir -p "${temp_dir}"
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

generated="${temp_dir}"
python3 - \
    "${generated}/artifact.json" \
    "${version}" \
    "${source_commit}" \
    "${source_id}" \
    "${source_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = path.parent
files = {}
for name in ("SaveModel.onnx", "manifest.pb"):
    files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
document = {
    "package": "rl-smoke-model",
    "version": sys.argv[2],
    "source_commit": sys.argv[3],
    "source_id": sys.argv[4],
    "source_sha256": sys.argv[5],
    "platform": "any",
    "files": files,
}
path.write_text(
    json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mkdir -p "$(dirname "${output_dir}")"
mv "${generated}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
