#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
artifact_root="${workspace_root}/.workspace/artifacts/rl-contracts"
output_dir="${repo_dir}/proto"
contracts_version="$(tr -d '[:space:]' < "${workspace_root}/rl-contracts/VERSION")"
profile="${1:-training}"

if [ "$#" -gt 1 ]; then
    echo "usage: bash proto/gen_proto.sh [training]" >&2
    exit 2
fi
case "${profile}" in
    training) ;;
    *)
        echo "usage: bash proto/gen_proto.sh [training]" >&2
        exit 2
        ;;
esac

mkdir -p "${output_dir}"
if [ "${profile}" = "training" ]; then
    python3 "${repo_dir}/proto/sync_contract_snapshot.py" \
        --artifact-dir "${artifact_root}/${contracts_version}/training" \
        --target-dir "${output_dir}" \
        --profile training
fi
printf '%s\n' "${output_dir}"
