#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
image_ref="${LEARNER_IMAGE_REF:-rl-training/learner:${LEARNER_IMAGE_TAG:-test-001}}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-smoke-model"
source "${repo_dir}/artifact_versions.env"
version="${RL_SMOKE_MODEL_VERSION}"
source_commit="$(git -C "${repo_dir}" rev-parse --short=12 HEAD 2>/dev/null || printf 'unborn')"
source_sha256="$(
    python3 - "${repo_dir}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
excluded = {
    ".git",
    "build",
    "logs",
    "models",
    "sample-pool",
    "model-distributor",
    "__pycache__",
    "_deps",
}
for path in sorted(root.rglob("*")):
    if not path.is_file() or excluded.intersection(path.parts):
        continue
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"
source_id="${source_commit}"
if test -n "$(git -C "${repo_dir}" status --porcelain=v1)"; then
    source_id="${source_commit}-dirty-${source_sha256:0:12}"
fi
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
manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "package": "rl-smoke-model",
    "version": os.environ["PACKAGE_VERSION"],
    "source_id": os.environ["SOURCE_ID"],
    "source_sha256": os.environ["SOURCE_SHA256"],
    "platform": "any",
}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
model_path = root / manifest.get("model_file", "")
if not model_path.is_file():
    raise SystemExit(1)
payload = model_path.read_bytes()
if len(payload) != manifest.get("size_bytes"):
    raise SystemExit(1)
if hashlib.sha256(payload).hexdigest() != manifest.get("sha256"):
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
    "${generated}/manifest.json" \
    "${version}" \
    "${source_commit}" \
    "${source_id}" \
    "${source_sha256}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
document = json.loads(path.read_text(encoding="utf-8"))
document.update(
    {
        "package": "rl-smoke-model",
        "version": sys.argv[2],
        "source_commit": sys.argv[3],
        "source_id": sys.argv[4],
        "source_sha256": sys.argv[5],
        "platform": "any",
    }
)
document["artifact_uri"] = (
    "file:///opt/rl/aiserver/models/smoke/" + document["model_file"]
)
path.write_text(
    json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

mkdir -p "$(dirname "${output_dir}")"
mv "${generated}" "${output_dir}"
trap - EXIT
printf '%s\n' "${output_dir}"
