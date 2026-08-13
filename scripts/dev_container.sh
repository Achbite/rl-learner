#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
if [ "$#" -gt 0 ]; then
    shift
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
container_name="learner-dev"
network_name="rl-training-dev"
monitor_host_port=9005
monitor_tunnel_socket="${TMPDIR:-/tmp}/rl-training-learner-dev-9005.sock"
monitor_tunnel_marker="${TMPDIR:-/tmp}/rl-training-learner-dev-9005.json"
colima_ssh_config="${RL_COLIMA_SSH_CONFIG:-${HOME}/.colima/_lima/colima/ssh.config}"
tag="${LEARNER_DEV_IMAGE_TAG:-test-001}"
runtime_image="rl-training/learner:${LEARNER_IMAGE_TAG:-test-001}"
dev_image="rl-training/learner-dev:${tag}"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"

tcp_ready() {
    nc -z 127.0.0.1 "${monitor_host_port}" >/dev/null 2>&1
}

colima_ssh_target() {
    awk '$1 == "Host" { print $2; exit }' "${colima_ssh_config}"
}

monitor_marker_owned() {
    [ -f "${monitor_tunnel_marker}" ] || return 1
    python3 - "${monitor_tunnel_marker}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        marker = json.load(stream)
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(
    0
    if marker.get("owner") == "learner-dev"
    and marker.get("port") == 9005
    else 1
)
PY
}

monitor_tunnel_running() {
    local target
    monitor_marker_owned || return 1
    [ -S "${monitor_tunnel_socket}" ] || return 1
    [ -f "${colima_ssh_config}" ] || return 1
    target="$(colima_ssh_target)"
    [ -n "${target}" ] || return 1
    ssh \
        -F "${colima_ssh_config}" \
        -o "ControlPath=${monitor_tunnel_socket}" \
        -O check \
        "${target}" >/dev/null 2>&1
}

write_monitor_marker() {
    local target="$1"
    python3 - \
        "${monitor_tunnel_marker}" \
        "${monitor_tunnel_socket}" \
        "${colima_ssh_config}" \
        "${target}" <<'PY'
import json
import os
import sys
import tempfile
import time

marker_path, socket_path, config_path, target = sys.argv[1:]
document = {
    "schema_version": 1,
    "owner": "learner-dev",
    "port": 9005,
    "socket_path": socket_path,
    "config_path": config_path,
    "target": target,
    "created_at": time.time(),
}
directory = os.path.dirname(marker_path) or "."
descriptor, temporary = tempfile.mkstemp(
    prefix=".learner-monitor-", suffix=".json", dir=directory
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, marker_path)
except BaseException:
    try:
        os.unlink(temporary)
    except OSError:
        pass
    raise
PY
}

remove_owned_monitor_files() {
    monitor_marker_owned || return 0
    python3 - "${monitor_tunnel_marker}" "${monitor_tunnel_socket}" <<'PY'
import os
import sys

for value in sys.argv[1:]:
    try:
        os.unlink(value)
    except FileNotFoundError:
        pass
PY
}

stop_monitor_transport() {
    local target
    if ! monitor_marker_owned; then
        return
    fi
    if monitor_tunnel_running; then
        target="$(colima_ssh_target)"
        ssh \
            -F "${colima_ssh_config}" \
            -o "ControlPath=${monitor_tunnel_socket}" \
            -O exit \
            "${target}" >/dev/null 2>&1 || true
    fi
    remove_owned_monitor_files
}

prepare_monitor_transport() {
    local target
    local upstream_port
    if [ ! -f "${colima_ssh_config}" ] || ! command -v ssh >/dev/null 2>&1; then
        return 1
    fi
    if monitor_tunnel_running; then
        return
    fi
    if [ -e "${monitor_tunnel_marker}" ] && ! monitor_marker_owned; then
        echo "PORT_IDENTITY_CONFLICT: unowned monitor marker ${monitor_tunnel_marker}" >&2
        return 1
    fi
    if [ -e "${monitor_tunnel_socket}" ] && ! monitor_marker_owned; then
        echo "PORT_IDENTITY_CONFLICT: unowned monitor socket ${monitor_tunnel_socket}" >&2
        return 1
    fi
    if tcp_ready; then
        if python3 - 2>/dev/null <<'PY'
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:9005/api/status", timeout=1
) as response:
    response.read()
PY
        then
            echo "PORT_IDENTITY_CONFLICT: host port 9005 is not owned by learner-dev" >&2
        else
            echo "MONITOR_TARGET_UNAVAILABLE: host port 9005 accepts TCP but not the metrics API" >&2
        fi
        return 1
    fi
    if monitor_marker_owned; then
        remove_owned_monitor_files
    fi
    target="$(colima_ssh_target)"
    if [ -z "${target}" ]; then
        echo "Colima SSH config has no Host entry: ${colima_ssh_config}" >&2
        return 1
    fi
    upstream_port="$(
        docker port "${container_name}" 9005/tcp 2>/dev/null |
            awk -F: 'NR == 1 {print $NF}'
    )"
    if ! [[ "${upstream_port}" =~ ^[0-9]+$ ]]; then
        echo "MONITOR_TARGET_UNAVAILABLE: learner-dev has no metrics port mapping" >&2
        return 1
    fi
    ssh \
        -F "${colima_ssh_config}" \
        -o ControlMaster=yes \
        -o "ControlPath=${monitor_tunnel_socket}" \
        -o ControlPersist=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -L "127.0.0.1:${monitor_host_port}:127.0.0.1:${upstream_port}" \
        -N \
        -f \
        "${target}"
    write_monitor_marker "${target}"
}

fetch_host_monitor_status() {
    python3 - <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:9005/api/status", timeout=1
) as response:
    sys.stdout.write(response.read().decode("utf-8"))
PY
}

fetch_container_monitor_status() {
    docker exec -i "${container_name}" python3 - <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(
    "http://127.0.0.1:9005/api/status", timeout=1
) as response:
    sys.stdout.write(response.read().decode("utf-8"))
PY
}

verify_monitor_identity() {
    local host_status
    local container_status
    host_status="$(fetch_host_monitor_status 2>/dev/null)" || return 1
    container_status="$(fetch_container_monitor_status 2>/dev/null)" || return 1
    python3 - "${host_status}" "${container_status}" <<'PY'
import json
import sys

host = json.loads(sys.argv[1])
container = json.loads(sys.argv[2])
required = (
    "service_instance_id",
    "metrics_source_id",
    "started_at",
    "latest_sequence",
    "latest_timestamp",
)
for document in (host, container):
    if document.get("schema_version") != 1:
        raise SystemExit(1)
    if document.get("service") != "learner-metrics":
        raise SystemExit(1)
    if document.get("stream") != "current":
        raise SystemExit(1)
    if document.get("mode") != "training" or document.get("stale") is not False:
        raise SystemExit(1)
    if any(document.get(field) in (None, "") for field in required):
        raise SystemExit(1)
    if int(document["latest_sequence"]) < 1:
        raise SystemExit(1)
    if float(document["latest_timestamp"]) < float(document["started_at"]):
        raise SystemExit(1)
    if not str(document.get("metrics_dir", "")).endswith(
        "/models/local-train/metrics"
    ):
        raise SystemExit(1)
for field in (
    "service_instance_id",
    "metrics_source_id",
    "started_at",
    "metrics_dir",
):
    if host.get(field) != container.get(field):
        raise SystemExit(1)
PY
}

wait_monitor_identity() {
    local attempts="${1:-${RL_MONITOR_READY_ATTEMPTS:-90}}"
    local attempt
    for attempt in $(seq 1 "${attempts}"); do
        if verify_monitor_identity; then
            return
        fi
        sleep 1
    done
    return 1
}

build_image() {
    docker build \
        --file "${repo_dir}/Dockerfile.dev" \
        --build-arg "LEARNER_RUNTIME_IMAGE=${runtime_image}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

create_container() {
    local contract_dir="$1"
    docker run --detach \
        --name "${container_name}" \
        --network "${network_name}" \
        --network-alias "${container_name}" \
        --network-alias "maze-learner" \
        --publish "127.0.0.1::9005" \
        --volume "${repo_dir}:/workspace/rl-learner" \
        --volume "${contract_dir}/python/common_pb2.py:/workspace/rl-learner/proto/common_pb2.py:ro" \
        --volume "${contract_dir}/python/training_pb2.py:/workspace/rl-learner/proto/training_pb2.py:ro" \
        --volume "${contract_dir}/python/training_pb2_grpc.py:/workspace/rl-learner/proto/training_pb2_grpc.py:ro" \
        --volume "${contract_dir}/schemas:/workspace/rl-learner/schemas:ro" \
        "${dev_image}" >/dev/null
}

container_mount_source() {
    local destination="$1"
    docker inspect \
        --format "{{range .Mounts}}{{if eq .Destination \"${destination}\"}}{{.Source}}{{end}}{{end}}" \
        "${container_name}"
}

container_uses_contract_artifact() {
    local contract_dir="$1"
    local relative
    for relative in common_pb2.py training_pb2.py training_pb2_grpc.py; do
        if [ "$(container_mount_source "/workspace/rl-learner/proto/${relative}")" != \
             "${contract_dir}/python/${relative}" ]; then
            return 1
        fi
    done
    if [ "$(container_mount_source "/workspace/rl-learner/schemas")" != \
         "${contract_dir}/schemas" ]; then
        return 1
    fi
}

container_has_training_processes() {
    local process_status
    [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ] ||
        return 1
    set +e
    docker exec "${container_name}" sh -lc \
        "pgrep -f '[m]ain.training_runtime|[r]un.sh training|[m]etrics_server.py|[m]aze_sample_distributor|[m]aze_model_distributor' >/dev/null"
    process_status=$?
    set -e
    if [ "${process_status}" -eq 0 ]; then
        return 0
    fi
    if [ "${process_status}" -eq 1 ]; then
        return 1
    fi
    echo "Unable to inspect learner-dev training processes" >&2
    return 2
}

verify_contract_mount_permissions() {
    local contract_dir="$1"
    python3 - \
        "${contract_dir}/python/common_pb2.py" \
        "${contract_dir}/python/training_pb2.py" \
        "${contract_dir}/python/training_pb2_grpc.py" \
        "${contract_dir}/schemas/maze.metrics.v2.json" \
        "${contract_dir}/schemas/maze.metrics.v2.sha256" <<'PY'
import os
import stat
import sys

for raw in sys.argv[1:]:
    mode = os.stat(raw).st_mode
    if not stat.S_ISREG(mode) or not mode & stat.S_IROTH:
        raise SystemExit(f"Contract artifact is not container-readable: {raw}")
PY
}

verify_training_runtime_artifacts() {
    local contract_dir
    contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${RL_CONTRACTS_VERSION}/${platform_dir}"
    if ! python3 "${repo_dir}/scripts/verify_runtime_artifacts.py" \
        --contract-dir "${contract_dir}" \
        --sample-pool-dir "${repo_dir}/sample-pool" \
        --model-distributor-dir "${repo_dir}/model-distributor" \
        --contract-version "${RL_CONTRACTS_VERSION}" \
        --sample-pool-version "${RL_SAMPLE_POOL_VERSION}" \
        --model-distributor-version "${RL_MODEL_DISTRIBUTOR_VERSION}" \
        --platform "${RL_RUNTIME_ARTIFACT_PLATFORM}"; then
        echo "Training runtime artifacts do not match artifact_versions.env" >&2
        echo "Build reviewed artifacts, then run: bash scripts/sync_runtime_artifacts.sh" >&2
        return 1
    fi
}

ensure_container() {
    local contract_dir
    local legacy_binding
    local process_state
    local start_error
    contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${RL_CONTRACTS_VERSION}/${platform_dir}"
    if [ ! -f "${contract_dir}/python/common_pb2.py" ] ||
       [ ! -f "${contract_dir}/python/training_pb2.py" ]; then
        echo "Contract artifact is missing. Run ../rl-contracts/build_artifact.sh" >&2
        exit 1
    fi
    verify_contract_mount_permissions "${contract_dir}"
    if ! docker image inspect "${dev_image}" >/dev/null 2>&1; then
        build_image
    fi
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
    if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
        create_container "${contract_dir}"
    elif ! container_uses_contract_artifact "${contract_dir}"; then
        process_state=0
        container_has_training_processes || process_state=$?
        if [ "${process_state}" -eq 0 ]; then
            echo "learner-dev uses a stale contract artifact while training processes are active" >&2
            echo "Stop the active training chain before recreating learner-dev" >&2
            exit 1
        elif [ "${process_state}" -ne 1 ]; then
            exit 1
        fi
        echo "Migrating learner-dev to contract artifact ${RL_CONTRACTS_VERSION}" >&2
        docker rm --force "${container_name}" >/dev/null
        create_container "${contract_dir}"
    elif [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" != "true" ]; then
        legacy_binding="$(docker port "${container_name}" 9005/tcp 2>/dev/null || true)"
        if ! start_error="$(docker start "${container_name}" 2>&1)"; then
            if [[ "${legacy_binding}" =~ :9005$ ]] &&
               { [[ "${start_error}" == *"address already in use"* ]] ||
                 [[ "${start_error}" == *"port is already allocated"* ]]; }; then
                echo "Migrating stopped learner-dev from legacy fixed monitor port" >&2
                docker rm "${container_name}" >/dev/null
                create_container "${contract_dir}"
            else
                echo "${start_error}" >&2
                return 1
            fi
        fi
    fi
}

case "${action}" in
    image)
        build_image
        ;;
    shell)
        ensure_container
        exec docker exec -it "${container_name}" bash
        ;;
    build)
        ensure_container
        docker exec "${container_name}" sh -lc \
            "cd /workspace/rl-learner && python3 -m compileall -q main proto src tools"
        ;;
    test)
        ensure_container
        docker exec "${container_name}" sh -lc \
            "cd /workspace/rl-learner && python3 -m unittest discover -s tests -v"
        ;;
    training)
        ensure_container
        verify_training_runtime_artifacts
        monitor_transport_ready=0
        if prepare_monitor_transport; then
            monitor_transport_ready=1
        else
            echo "Learner Monitor transport unavailable; training continues" >&2
        fi
        trap '[ "${monitor_transport_ready}" -eq 0 ] || stop_monitor_transport' EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        if [ "${monitor_transport_ready}" -eq 1 ]; then
            printf 'Learner Monitor (available after startup): http://127.0.0.1:%s/\n' \
                "${monitor_host_port}"
        fi
        docker_exec_args=(exec -i)
        if [ -t 0 ] && [ -t 1 ]; then
            docker_exec_args+=(--tty)
        fi
        set +e
        docker "${docker_exec_args[@]}" "${container_name}" bash -lc \
            'cd /workspace/rl-learner && exec bash ./run.sh "$@"' \
            bash training "$@"
        training_status=$?
        set -e
        trap - EXIT INT TERM
        if [ "${monitor_transport_ready}" -eq 1 ]; then
            stop_monitor_transport
        fi
        exit "${training_status}"
        ;;
    monitor)
        ensure_container
        prepare_monitor_transport
        if ! wait_monitor_identity; then
            stop_monitor_transport
            echo "MONITOR_TARGET_UNAVAILABLE: Learner metrics identity did not become ready" >&2
            exit 1
        fi
        printf 'Learner Monitor: http://127.0.0.1:%s/\n' \
            "${monitor_host_port}"
        ;;
    monitor-stop)
        stop_monitor_transport
        ;;
    clean)
        stop_monitor_transport
        if docker container inspect "${container_name}" >/dev/null 2>&1; then
            if [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ]; then
                docker stop --time 5 "${container_name}" >/dev/null
            fi
            docker rm "${container_name}" >/dev/null
        fi
        ;;
    *)
        echo "unknown dev action: ${action}" >&2
        exit 2
        ;;
esac
