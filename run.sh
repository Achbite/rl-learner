#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
config="${repo_dir}/configs/learner_config.yaml"
repository_sample_pool_dir="${repo_dir}/sample-pool"
runtime_sample_pool_dir="/opt/rl/learner/sample-pool"
repository_distributor_dir="${repo_dir}/model-distributor"
runtime_distributor_dir="/opt/rl/learner/model-distributor"
if [ -x "${repository_sample_pool_dir}/bin/maze_sample_pool" ] &&
   [ -f "${repository_sample_pool_dir}/config/pool_config.yaml" ]; then
    default_sample_pool_dir="${repository_sample_pool_dir}"
elif [ -x "${runtime_sample_pool_dir}/bin/maze_sample_pool" ] &&
     [ -f "${runtime_sample_pool_dir}/config/pool_config.yaml" ]; then
    default_sample_pool_dir="${runtime_sample_pool_dir}"
else
    default_sample_pool_dir="${repository_sample_pool_dir}"
fi
if [ -x "${repository_distributor_dir}/bin/maze_model_distributor" ] &&
   [ -f "${repository_distributor_dir}/config/model_distributor_config.yaml" ]; then
    default_distributor_dir="${repository_distributor_dir}"
elif [ -x "${runtime_distributor_dir}/bin/maze_model_distributor" ] &&
     [ -f "${runtime_distributor_dir}/config/model_distributor_config.yaml" ]; then
    default_distributor_dir="${runtime_distributor_dir}"
else
    default_distributor_dir="${repository_distributor_dir}"
fi
sample_pool_bin="${SAMPLE_POOL_BIN:-${default_sample_pool_dir}/bin/maze_sample_pool}"
sample_pool_config="${SAMPLE_POOL_CONFIG:-${default_sample_pool_dir}/config/pool_config.yaml}"
model_distributor_bin="${MODEL_DISTRIBUTOR_BIN:-${default_distributor_dir}/bin/maze_model_distributor}"
model_distributor_config="${MODEL_DISTRIBUTOR_CONFIG:-${default_distributor_dir}/config/model_distributor_config.yaml}"

for argument in "$@"; do
    if [ "${argument}" = "--help" ] || [ "${argument}" = "-h" ]; then
        cd "${repo_dir}"
        exec python3 -m main.training_runtime "$@"
    fi
done

execution_token="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export RL_MODEL_LINEAGE_ID="maze-model-${execution_token}"
export RL_TRAINING_FINALIZE_REQUEST_PATH="/tmp/rl-training-${execution_token}-finalize"
export RL_TRAINING_FINALIZE_COMPLETE_PATH="/tmp/rl-training-${execution_token}-finalized"
metrics_source_id="${RL_METRICS_SOURCE_ID:-}"
if [ -z "${metrics_source_id}" ]; then
    metrics_source_id="$(python3 -c 'import uuid; print("local-training-" + uuid.uuid4().hex)')"
fi
export RL_METRICS_SOURCE_ID="${metrics_source_id}"

startup_output="$(
    cd "${repo_dir}"
    python3 -m main.resolve_startup --format lines -- "$@"
)"
mapfile -t startup_values <<< "${startup_output}"
if [ "${#startup_values[@]}" -ne 10 ]; then
    echo "Learner effective startup handoff is invalid" >&2
    exit 1
fi
config="${startup_values[0]}"
local_train_root="${startup_values[1]}"
initial_model="${startup_values[2]}"
sample_pool_host="${startup_values[3]}"
sample_pool_port="${startup_values[4]}"
model_distributor_host="${startup_values[5]}"
model_distributor_port="${startup_values[6]}"
aiserver_host="${startup_values[7]}"
aiserver_port="${startup_values[8]}"
metrics_port="${startup_values[9]}"
printf '%s\n' \
    "Learner effective startup: config=${config} train=${local_train_root} initial_model=${initial_model:-<fresh>} sample=${sample_pool_host}:${sample_pool_port} model=${model_distributor_host}:${model_distributor_port} aiserver=${aiserver_host}:${aiserver_port} metrics_port=${metrics_port}"

# Host port ownership belongs to the development launcher. Only advertise its
# URL when it explicitly identifies the container port mapped by that URL.
unset RL_METRICS_PUBLIC_URL
development_monitor_url="${RL_DEVELOPMENT_MONITOR_URL:-}"
development_monitor_container_port="${RL_DEVELOPMENT_MONITOR_CONTAINER_PORT:-}"
if [ -n "${development_monitor_url}" ]; then
    if [ "${development_monitor_container_port}" = "${metrics_port}" ]; then
        export RL_METRICS_PUBLIC_URL="${development_monitor_url}"
    else
        echo "Learner Monitor host URL unavailable: effective metrics port ${metrics_port} is not the published development port ${development_monitor_container_port:-<unset>}; training continues" >&2
    fi
fi

if [ ! -x "${sample_pool_bin}" ]; then
    echo "Sample Pool executable is missing: ${sample_pool_bin}" >&2
    echo "Build rl-sample-pool and stage its artifact in sample-pool/" >&2
    exit 1
fi
if [ ! -f "${sample_pool_config}" ]; then
    echo "Sample Pool config is missing: ${sample_pool_config}" >&2
    echo "Build rl-sample-pool and stage its artifact in sample-pool/" >&2
    exit 1
fi
if [ ! -x "${model_distributor_bin}" ]; then
    echo "Model Distributor executable is missing: ${model_distributor_bin}" >&2
    echo "Build rl-model-distributor and stage its artifact in model-distributor/" >&2
    exit 1
fi
if [ ! -f "${model_distributor_config}" ]; then
    echo "Model Distributor config is missing: ${model_distributor_config}" >&2
    echo "Build rl-model-distributor and stage its artifact in model-distributor/" >&2
    exit 1
fi

if [ "$(basename "${local_train_root}")" != "train" ] ||
   [ "${local_train_root}" = "/train" ]; then
    echo "Learner local train root must be a scoped path ending with /train" >&2
    exit 1
fi
if [ -L "${local_train_root}" ]; then
    echo "Learner local train root must not be a symbolic link" >&2
    exit 1
fi
if [ -e "${local_train_root}" ] && [ ! -d "${local_train_root}" ]; then
    echo "Learner local train root is not a directory: ${local_train_root}" >&2
    exit 1
fi
mkdir -p "${local_train_root}"
local_train_root="$(cd "${local_train_root}" && pwd -P)"
if [ "$(basename "${local_train_root}")" != "train" ] ||
   [ "${local_train_root}" = "/train" ]; then
    echo "Learner local train root must be a scoped path ending with /train" >&2
    exit 1
fi

# Keep process ownership outside the publication namespace. ModelPublisher
# treats every entry under local_train_root as state owned by this invocation.
training_lock="${local_train_root}.learner.lock"
if ! mkdir "${training_lock}" 2>/dev/null; then
    echo "Learner training is already active or its lock remains: ${training_lock}" >&2
    exit 1
fi
printf '%s\n' "$$" > "${training_lock}/pid"
cleanup_training_lock() {
    rm -f -- "${training_lock}/pid"
    rmdir "${training_lock}" 2>/dev/null || true
}
trap cleanup_training_lock EXIT

if ! python3 -m main.reset_workspace --path "${local_train_root}"; then
    echo "Learner fresh workspace reset failed: ${local_train_root}" >&2
    exit 1
fi

mkdir -p "${local_train_root}"
mkdir -p \
    "${local_train_root}/runtime/checkpoints" \
    "${local_train_root}/runtime/receipts"
if ! mkdir -p "${local_train_root}/metrics"; then
    echo "Metrics directory unavailable; training continues without local metrics" >&2
fi

export PYTHONUNBUFFERED=1
metrics_pid=""
metrics_failure_reported=0
training_pid=""
training_child_status=""
model_distributor_pid=""
sample_pool_pid=""
stopping=0
quiesced=0
quiesce_marker="${RL_QUIESCE_MARKER:-/tmp/rl-training-${execution_token}-quiesced}"
quiesce_failure_marker="${RL_QUIESCE_FAILURE_MARKER:-/tmp/rl-training-${execution_token}-quiesce-failed}"
rm -f "${quiesce_marker}" "${quiesce_failure_marker}"

terminate_process() {
    local pid="$1"
    local timeout_seconds="$2"
    if [ -z "${pid}" ]; then
        return 125
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null
        return $?
    fi
    kill -TERM "${pid}" 2>/dev/null || true
    if [ "${timeout_seconds}" -eq 0 ]; then
        wait "${pid}" 2>/dev/null
        return $?
    fi
    local waited=0
    while kill -0 "${pid}" 2>/dev/null &&
          [ "${waited}" -lt "${timeout_seconds}" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null
}

shutdown() {
    if [ "${stopping}" -eq 1 ]; then
        return
    fi
    stopping=1
    if [ -n "${training_pid}" ]; then
        if terminate_process "${training_pid}" 0; then
            training_child_status=0
        else
            training_child_status=$?
        fi
        training_pid=""
    fi
    terminate_process "${metrics_pid}" 3 || true
    metrics_pid=""
    terminate_process "${model_distributor_pid}" 3 || true
    model_distributor_pid=""
    terminate_process "${sample_pool_pid}" 10 || true
    sample_pool_pid=""
    cleanup_training_lock
}

on_signal() {
    shutdown
    exit "${training_child_status:-0}"
}

quiesce() {
    if [ "${quiesced}" -eq 1 ] || [ "${stopping}" -eq 1 ]; then
        return
    fi
    quiesced=1
    local child_status="${training_child_status}"
    if [ -n "${training_pid}" ]; then
        if terminate_process "${training_pid}" 0; then
            child_status=0
        else
            child_status=$?
        fi
    elif [ -z "${child_status}" ]; then
        child_status=125
    fi
    if [ -n "${training_pid}" ]; then
        training_child_status="${child_status}"
    fi
    training_pid=""
    if [ "${child_status}" -eq 0 ]; then
        : > "${quiesce_marker}"
    else
        printf '%s\n' "${child_status}" > "${quiesce_failure_marker}"
    fi
}

trap quiesce USR1
trap shutdown EXIT
trap on_signal TERM INT

cd "${repo_dir}"
RL_SAMPLE_POOL_PORT="${sample_pool_port}" \
    "${sample_pool_bin}" "${sample_pool_config}" &
sample_pool_pid=$!

ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${sample_pool_pid}" 2>/dev/null; then
        wait "${sample_pool_pid}"
        exit $?
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${sample_pool_port}") \
        2>/dev/null; then
        exec 3>&-
        exec 3<&-
        ready=1
        break
    fi
    sleep 0.1
done
if [ "${ready}" -ne 1 ]; then
    echo "Sample Pool readiness timeout" >&2
    exit 1
fi

RL_MODEL_DISTRIBUTOR_PORT="${model_distributor_port}" \
RL_MODEL_ARTIFACT_ROOT="${local_train_root}" \
    "${model_distributor_bin}" "${model_distributor_config}" &
model_distributor_pid=$!

ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${model_distributor_pid}" 2>/dev/null; then
        wait "${model_distributor_pid}"
        exit $?
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${model_distributor_port}") \
        2>/dev/null; then
        exec 3>&-
        exec 3<&-
        ready=1
        break
    fi
    sleep 0.1
done
if [ "${ready}" -ne 1 ]; then
    echo "Model Distributor readiness timeout" >&2
    exit 1
fi

python3 tools/metrics_server.py \
    --dir "${local_train_root}/metrics" \
    --port "${metrics_port}" \
    --source-id "${metrics_source_id}" \
    --mode training &
metrics_pid=$!

python3 -m main.training_runtime "$@" &
training_pid=$!

while [ "${stopping}" -eq 0 ]; do
    for process in \
        "${sample_pool_pid}" \
        "${model_distributor_pid}" \
        "${training_pid}"; do
        if [ -z "${process}" ]; then
            continue
        fi
        if ! kill -0 "${process}" 2>/dev/null; then
            if wait "${process}"; then
                child_status=0
            else
                child_status=$?
            fi
            if [ "${process}" = "${training_pid}" ]; then
                training_child_status="${child_status}"
                training_pid=""
            fi
            shutdown
            exit "${child_status}"
        fi
    done
    if [ -n "${metrics_pid}" ] &&
       ! kill -0 "${metrics_pid}" 2>/dev/null; then
        if wait "${metrics_pid}"; then
            metrics_status=0
        else
            metrics_status=$?
        fi
        metrics_pid=""
        if [ "${metrics_failure_reported}" -eq 0 ]; then
            echo "Learner Monitor unavailable (exit=${metrics_status}); training continues" >&2
            metrics_failure_reported=1
        fi
    fi
    sleep 0.2
done
