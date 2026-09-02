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
    tests.test_delivery_contract.LearnerDevelopmentTest.test_processed_transition_data_reaches_real_trainer \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_local_effective_config_reaches_runtime_validation \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_episode_metrics_are_derived_from_raw_agent_facts \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_train_metrics_are_derived_from_raw_sum_counts \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_metric_payloads_follow_typed_transport \
    tests.test_ppo_contract.PPOCalculationTest.test_clipped_ppo_loss_matches_reference_values
