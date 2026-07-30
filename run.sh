#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workload="${1:-training}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "${workload}" != "training" ]; then
    echo "Learner supports only the training workload" >&2
    exit 2
fi

config="${MAZE_LEARNER_CONFIG:-${repo_dir}/configs/learner_config.yaml}"
model_distributor_bin="${MODEL_DISTRIBUTOR_BIN:-/opt/rl/learner/model-distributor/bin/maze_model_distributor}"
model_distributor_config="${MODEL_DISTRIBUTOR_CONFIG:-/opt/rl/learner/model-distributor/config/model_distributor_config.yaml}"
model_root="${MAZE_MODEL_ARTIFACT_ROOT:-${repo_dir}/models/published}"
checkpoint_root="${MAZE_CHECKPOINT_ROOT:-${repo_dir}/models/checkpoints}"
update_root="${MAZE_UPDATE_RECEIPT_ROOT:-${repo_dir}/models/updates}"
run_id="${MAZE_RUN_ID:-local-run}"
metrics_port="${MAZE_DASHBOARD_PORT:-9005}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            config="${2:?--config requires a value}"
            shift 2
            ;;
        --run-id)
            run_id="${2:?--run-id requires a value}"
            shift 2
            ;;
        --model-root)
            model_root="${2:?--model-root requires a value}"
            shift 2
            ;;
        --checkpoint-root)
            checkpoint_root="${2:?--checkpoint-root requires a value}"
            shift 2
            ;;
        --update-root)
            update_root="${2:?--update-root requires a value}"
            shift 2
            ;;
        --model-distributor)
            address="${2:?--model-distributor requires host:port}"
            export MAZE_MODEL_DISTRIBUTOR_HOST="${address%:*}"
            export MAZE_MODEL_DISTRIBUTOR_PORT="${address##*:}"
            shift 2
            ;;
        --sample-distributor)
            address="${2:?--sample-distributor requires host:port}"
            export MAZE_SAMPLE_DISTRIBUTOR_HOST="${address%:*}"
            export MAZE_SAMPLE_DISTRIBUTOR_PORT="${address##*:}"
            shift 2
            ;;
        --aiserver)
            address="${2:?--aiserver requires host:port}"
            export MAZE_AISERVER_HOST="${address%:*}"
            export MAZE_AISERVER_PORT="${address##*:}"
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

if [ ! -x "${model_distributor_bin}" ]; then
    echo "ModelDistributor executable is missing: ${model_distributor_bin}" >&2
    exit 1
fi
if [ ! -f "${model_distributor_config}" ]; then
    echo "ModelDistributor config is missing: ${model_distributor_config}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export MAZE_RUN_ID="${run_id}"
export MAZE_MODEL_ARTIFACT_ROOT="${model_root}"
export MAZE_CHECKPOINT_ROOT="${checkpoint_root}"
export MAZE_UPDATE_RECEIPT_ROOT="${update_root}"
export MAZE_MODEL_DISTRIBUTOR_PORT="${MAZE_MODEL_DISTRIBUTOR_PORT:-9200}"

metrics_pid=""
training_pid=""
model_distributor_pid=""
stopping=0
quiesced=0
quiesce_marker="${MAZE_QUIESCE_MARKER:-/tmp/rl-training-quiesced}"
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
"${model_distributor_bin}" "${model_distributor_config}" &
model_distributor_pid=$!

ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "${model_distributor_pid}" 2>/dev/null; then
        wait "${model_distributor_pid}"
        exit $?
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${MAZE_MODEL_DISTRIBUTOR_PORT}") \
        2>/dev/null; then
        exec 3>&-
        exec 3<&-
        ready=1
        break
    fi
    sleep 0.1
done
if [ "${ready}" -ne 1 ]; then
    echo "ModelDistributor readiness timeout" >&2
    exit 1
fi

python3 tools/metrics_server.py \
    --dir logs/metrics \
    --port "${metrics_port}" &
metrics_pid=$!

python3 -m main.training_runtime \
    --config "${config}" \
    --run-id "${run_id}" &
training_pid=$!

while [ "${stopping}" -eq 0 ]; do
    for process in \
        "${model_distributor_pid}" \
        "${metrics_pid}" \
        "${training_pid}"; do
        if [ -z "${process}" ]; then
            continue
        fi
        if ! kill -0 "${process}" 2>/dev/null; then
            wait "${process}"
            exit $?
        fi
    done
    sleep 0.2
done
