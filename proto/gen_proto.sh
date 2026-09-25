#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [ "$#" -ne 1 ]; then
    echo "usage: bash proto/gen_proto.sh TRAINING_ARTIFACT_DIR" >&2
    exit 2
fi
python3 "${repo_dir}/proto/sync_contract_snapshot.py" \
    --artifact-dir "$1" --target-dir "${repo_dir}/proto" --profile training
printf '%s\n' "${repo_dir}/proto"
