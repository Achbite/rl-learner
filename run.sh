#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workload="${1:-training}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "${workload}" != "training" ]; then
    echo "Learner supports only the training workload" >&2
    exit 2
fi

config="${RL_LEARNER_CONFIG:-${repo_dir}/configs/learner_config.yaml}"
repository_sample_pool_dir="${repo_dir}/sample-pool"
runtime_sample_pool_dir="/opt/rl/learner/sample-pool"
repository_distributor_dir="${repo_dir}/model-distributor"
runtime_distributor_dir="/opt/rl/learner/model-distributor"
if [ -x "${repository_sample_pool_dir}/bin/maze_sample_distributor" ] &&
   [ -f "${repository_sample_pool_dir}/config/distributor_config.yaml" ]; then
    default_sample_pool_dir="${repository_sample_pool_dir}"
elif [ -x "${runtime_sample_pool_dir}/bin/maze_sample_distributor" ] &&
     [ -f "${runtime_sample_pool_dir}/config/distributor_config.yaml" ]; then
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
sample_pool_bin="${SAMPLE_DISTRIBUTOR_BIN:-${default_sample_pool_dir}/bin/maze_sample_distributor}"
sample_pool_config="${SAMPLE_DISTRIBUTOR_CONFIG:-${default_sample_pool_dir}/config/distributor_config.yaml}"
model_distributor_bin="${MODEL_DISTRIBUTOR_BIN:-${default_distributor_dir}/bin/maze_model_distributor}"
model_distributor_config="${MODEL_DISTRIBUTOR_CONFIG:-${default_distributor_dir}/config/model_distributor_config.yaml}"
local_train_root="${RL_LOCAL_TRAIN_ROOT:-${repo_dir}/models/local-train}"
initial_checkpoint="${RL_INITIAL_CHECKPOINT:-}"
metrics_port="${RL_METRICS_PORT:-9005}"
metrics_source_id="${RL_METRICS_SOURCE_ID:-}"
if [ -z "${metrics_source_id}" ]; then
    metrics_source_id="$(python3 -c 'import uuid; print("local-training-" + uuid.uuid4().hex)')"
fi
export RL_METRICS_SOURCE_ID="${metrics_source_id}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            config="${2:?--config requires a value}"
            shift 2
            ;;
        --initial-checkpoint)
            initial_checkpoint="${2:?--initial-checkpoint requires a value}"
            shift 2
            ;;
        --model-distributor)
            address="${2:?--model-distributor requires host:port}"
            export RL_MODEL_DISTRIBUTOR_HOST="${address%:*}"
            export RL_MODEL_DISTRIBUTOR_PORT="${address##*:}"
            shift 2
            ;;
        --aiserver)
            address="${2:?--aiserver requires host:port}"
            export RL_AISERVER_HOST="${address%:*}"
            export RL_AISERVER_PORT="${address##*:}"
            shift 2
            ;;
        --metrics-port)
            metrics_port="${2:?--metrics-port requires a value}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

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

if [ "$(basename "${local_train_root}")" != "local-train" ]; then
    echo "Learner local-train path must end with /local-train" >&2
    exit 1
fi
if [ -L "${local_train_root}" ]; then
    echo "Learner local-train path must not be a symbolic link" >&2
    exit 1
fi
mkdir -p "$(dirname "${local_train_root}")"
local_train_parent="$(
    cd "$(dirname "${local_train_root}")" && pwd -P
)"
local_train_root="${local_train_parent}/local-train"
expected_local_train_root="${repo_dir}/models/local-train"
if [ "${local_train_root}" != "${expected_local_train_root}" ]; then
    echo "Unsafe Learner local-train path: ${local_train_root}" >&2
    exit 1
fi
if [ -n "${initial_checkpoint}" ]; then
    if [ ! -f "${initial_checkpoint}" ]; then
        echo "Initial checkpoint does not exist: ${initial_checkpoint}" >&2
        exit 1
    fi
    checkpoint_parent="$(
        cd "$(dirname "${initial_checkpoint}")" && pwd -P
    )"
    initial_checkpoint="${checkpoint_parent}/$(basename "${initial_checkpoint}")"
    case "${initial_checkpoint}" in
        "${local_train_root}"|"${local_train_root}"/*)
            echo "Initial checkpoint must be outside local-train" >&2
            exit 1
            ;;
    esac
fi

training_lock="${RL_TRAIN_LOCK_DIR:-${local_train_parent}/.learner-local-train.lock}"
if ! mkdir "${training_lock}" 2>/dev/null; then
    echo "Learner training is already active or its lock remains: ${training_lock}" >&2
    exit 1
fi
printf '%s\n' "$$" > "${training_lock}/pid"

if [ -d "${local_train_root}" ]; then
    find "${local_train_root}" -mindepth 1 -maxdepth 1 \
        -exec rm -rf -- {} +
else
    mkdir -p "${local_train_root}"
fi
mkdir -p \
    "${local_train_root}/runtime/serving" \
    "${local_train_root}/runtime/checkpoints" \
    "${local_train_root}/runtime/receipts" \
    "${local_train_root}/archive" \
    "${local_train_root}/metrics"

export PYTHONUNBUFFERED=1
export RL_LOCAL_TRAIN_ROOT="${local_train_root}"
export RL_SAMPLE_POOL_HOST="127.0.0.1"
export RL_SAMPLE_POOL_PORT="${RL_SAMPLE_POOL_PORT:-9100}"
export RL_MODEL_DISTRIBUTOR_PORT="${RL_MODEL_DISTRIBUTOR_PORT:-9200}"
if [ -n "${initial_checkpoint}" ]; then
    export RL_INITIAL_CHECKPOINT="${initial_checkpoint}"
fi

metrics_pid=""
training_pid=""
model_distributor_pid=""
sample_pool_pid=""
stopping=0
quiesced=0
quiesce_marker="${RL_QUIESCE_MARKER:-/tmp/rl-training-quiesced}"
rm -f "${quiesce_marker}"

terminate_process() {
    local pid="$1"
    local timeout_seconds="$2"
    if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
        return
    fi
    kill -TERM "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null &&
          [ "${waited}" -lt "${timeout_seconds}" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

shutdown() {
    if [ "${stopping}" -eq 1 ]; then
        return
    fi
    stopping=1
    terminate_process "${training_pid}" 120
    training_pid=""
    terminate_process "${metrics_pid}" 3
    metrics_pid=""
    terminate_process "${model_distributor_pid}" 3
    model_distributor_pid=""
    terminate_process "${sample_pool_pid}" 10
    sample_pool_pid=""
    rm -rf -- "${training_lock}"
}

quiesce() {
    if [ "${quiesced}" -eq 1 ] || [ "${stopping}" -eq 1 ]; then
        return
    fi
    quiesced=1
    terminate_process "${training_pid}" 120
    training_pid=""
    : > "${quiesce_marker}"
}

trap quiesce USR1
trap shutdown EXIT TERM INT

cd "${repo_dir}"
"${sample_pool_bin}" "${sample_pool_config}" &
sample_pool_pid=$!

ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${sample_pool_pid}" 2>/dev/null; then
        wait "${sample_pool_pid}"
        exit $?
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${RL_SAMPLE_POOL_PORT}") \
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

"${model_distributor_bin}" "${model_distributor_config}" &
model_distributor_pid=$!

ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${model_distributor_pid}" 2>/dev/null; then
        wait "${model_distributor_pid}"
        exit $?
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${RL_MODEL_DISTRIBUTOR_PORT}") \
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

training_args=(--config "${config}")
if [ -n "${initial_checkpoint}" ]; then
    training_args+=(--initial-checkpoint "${initial_checkpoint}")
fi
python3 -m main.training_runtime "${training_args[@]}" &
training_pid=$!

while [ "${stopping}" -eq 0 ]; do
    for process in \
        "${sample_pool_pid}" \
        "${model_distributor_pid}" \
        "${metrics_pid}" \
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
            shutdown
            exit "${child_status}"
        fi
    done
    sleep 0.2
done
