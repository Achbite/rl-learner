#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
    echo "usage: bash ./test.sh" >&2
    exit 2
fi

# The registered bootstrap contract case launches only these explicit test dependencies.
: "${RL_AISERVER_TEST_BINARY:?Set RL_AISERVER_TEST_BINARY to the current production build}"
: "${RL_MODEL_DISTRIBUTOR_TEST_BINARY:?Set RL_MODEL_DISTRIBUTOR_TEST_BINARY to the current production build}"
: "${RL_AISERVER_SOURCE_DIR:?Set RL_AISERVER_SOURCE_DIR for the current config and fixed ONNX fixture}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
test_runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/rl-learner-test.XXXXXX")"
trap 'rm -rf "${test_runtime_dir}"' EXIT
cd "${test_runtime_dir}"

python3 -m unittest -v \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_processed_transition_data_reaches_real_trainer \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_local_effective_config_reaches_runtime_validation \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_get_batch_recovery_uses_request_deadline \
    tests.test_delivery_contract.LearnerDevelopmentTest.test_busy_lease_and_stop_have_distinct_recovery \
    tests.test_model_startup_contract.ModelStartupContractTest.test_exact_failed_ack_preserves_cause \
    tests.test_model_startup_contract.ModelStartupContractTest.test_other_model_failed_ack_does_not_fail_target \
    tests.test_model_startup_contract.ModelStartupContractTest.test_no_ack_waits_until_explicit_stop \
    tests.test_model_startup_contract.ModelStartupContractTest.test_real_aiserver_failed_ack_reaches_real_distributor_and_learner \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_preview_catalog_failure_preserves_status_and_throughput \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_preview_rates_keep_source_intervals_independent \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_catalog_is_readable_before_measurement_without_pinning_a_consumer \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_registered_task_metrics_use_raw_denominators_and_source_scope \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_monitor_views_preserve_registered_values_and_select_training_modules \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_train_metrics_are_derived_from_raw_sum_counts \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_metric_content_errors_do_not_rewind_durable_transport \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_metric_relay_starts_from_bootstrap_ack_source \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_relay_contract_rejection_is_terminal_and_visible \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_collector_finishes_after_final_ack \
    tests.test_metric_calculations.LearnerMetricCalculationTest.test_model_feedback_failure_preserves_active_model_state \
    tests.test_ppo_contract.PPOCalculationTest.test_clipped_ppo_loss_matches_reference_values
