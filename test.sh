#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
test_runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-learner-test.XXXXXX")"
trap 'rm -rf "${test_runtime_dir}"' EXIT
cd "${test_runtime_dir}"

python3 -m unittest -v \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_fixed_processed_transitions_enter_training_batch \
    tests.test_ppo_contract.LearnerDevelopmentTest.test_fixed_processed_transitions_match_ppo_loss
