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
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_registered_task_metrics_use_raw_denominators_and_source_scope \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_monitor_views_preserve_registered_values_and_select_training_modules \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_train_metrics_are_derived_from_raw_sum_counts \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_metric_content_errors_do_not_rewind_durable_transport \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_metric_relay_starts_from_bootstrap_ack_source \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_relay_contract_rejection_is_terminal_and_visible \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_model_feedback_failure_preserves_active_model_state \
    tests.test_ppo_contract.PPOCalculationTest.test_clipped_ppo_loss_matches_reference_values
