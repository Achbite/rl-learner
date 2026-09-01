#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
if [ "$#" -gt 0 ]; then
    shift
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_name="learner-dev"
network_name="rl-training-dev"
monitor_host_port=9005
monitor_tunnel_socket="${TMPDIR:-/tmp}/rl-training-learner-dev-9005.sock"
colima_ssh_config="${RL_COLIMA_SSH_CONFIG:-${HOME}/.colima/_lima/colima/ssh.config}"
tag="${LEARNER_DEV_IMAGE_TAG:-test-001}"
dev_image="rl-training/learner-dev:${tag}"
python_dev_base_image="${LEARNER_DEV_BASE_IMAGE:-python:3.11-slim}"
torch_version="${LEARNER_DEV_TORCH_VERSION:-2.12.1+cpu}"
if [ -f "/.dockerenv" ]; then
    echo "make shell is a host-side Docker entrypoint; leave the component container first" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "make shell is a host-side Docker entrypoint; Docker CLI is unavailable" >&2
    exit 1
fi
if ! platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}' 2>/dev/null)" ||
   [ -z "${platform}" ]; then
    echo "make shell cannot reach the host Docker daemon" >&2
    exit 1
fi
tcp_ready() {
    nc -z 127.0.0.1 "${monitor_host_port}" >/dev/null 2>&1
}

colima_ssh_target() {
    awk '$1 == "Host" { print $2; exit }' "${colima_ssh_config}"
}

monitor_tunnel_running() {
    local target
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

remove_monitor_socket() {
    python3 - "${monitor_tunnel_socket}" <<'PY'
import os
import sys

try:
    os.unlink(sys.argv[1])
except FileNotFoundError:
    pass
PY
}

stop_monitor_transport() {
    local target
    if [ "$(monitor_transport_mode)" = "direct" ]; then
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
    remove_monitor_socket
}

published_monitor_port() {
    docker port "${container_name}" 9005/tcp 2>/dev/null |
        awk -F: 'NR == 1 {print $NF}'
}

# Colima 在 macOS 上把 Docker 运行在 Lima 虚拟机内，宿主无法直达容器发布端口，
# 因此需要 SSH 隧道把固定端口转发进去。Linux 与 WSL 的 Docker 直接运行在宿主
# 内核上，--publish 的映射端口即可访问，无需隧道。
monitor_transport_mode() {
    if [ -f "${colima_ssh_config}" ] && command -v ssh >/dev/null 2>&1; then
        echo "tunnel"
    else
        echo "direct"
    fi
}

prepare_monitor_transport() {
    local target
    local upstream_port
    if [ "$(monitor_transport_mode)" = "direct" ]; then
        upstream_port="$(published_monitor_port)"
        if ! [[ "${upstream_port}" =~ ^[0-9]+$ ]]; then
            echo "MONITOR_TARGET_UNAVAILABLE: learner-dev has no metrics port mapping" >&2
            return 1
        fi
        monitor_host_port="${upstream_port}"
        return 0
    fi
    if [ ! -f "${colima_ssh_config}" ] || ! command -v ssh >/dev/null 2>&1; then
        return 1
    fi
    if monitor_tunnel_running; then
        stop_monitor_transport
    elif [ -e "${monitor_tunnel_socket}" ]; then
        remove_monitor_socket
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
    target="$(colima_ssh_target)"
    if [ -z "${target}" ]; then
        echo "Colima SSH config has no Host entry: ${colima_ssh_config}" >&2
        return 1
    fi
    upstream_port="$(published_monitor_port)"
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
}

fetch_host_monitor_status() {
    python3 - "${monitor_host_port}" <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/api/status", timeout=1
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
    metrics_dir = str(document.get("metrics_dir", ""))
    if "/models/train/" not in metrics_dir or not metrics_dir.endswith(
        "/metrics"
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
        --build-arg "PYTHON_DEV_BASE_IMAGE=${python_dev_base_image}" \
        --build-arg "TORCH_VERSION=${torch_version}" \
        --label "org.rl-training.component=learner-dev" \
        --label "org.rl-training.dev-platform=${platform}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

ensure_dev_image() {
    echo "Building Learner development image" >&2
    build_image
}

container_exists() {
    docker container inspect "${container_name}" >/dev/null 2>&1
}

container_running() {
    [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" = "true" ]
}

container_uses_current_image() {
    docker image inspect "${dev_image}" >/dev/null 2>&1 || return 1
    [ "$(docker inspect --format '{{.Image}}' "${container_name}")" = \
      "$(docker image inspect --format '{{.Id}}' "${dev_image}")" ]
}

container_has_legacy_artifact_mounts() {
    local destinations
    destinations="$(docker inspect \
        --format '{{range .Mounts}}{{println .Destination}}{{end}}' \
        "${container_name}")"
    case "${destinations}" in
        *"/workspace/rl-learner/proto/common_pb2.py"*|\
        *"/workspace/rl-learner/proto/training_pb2.py"*|\
        *"/workspace/rl-learner/proto/training_pb2_grpc.py"*|\
        *"/workspace/rl-learner/schemas"*|\
        *"/workspace/rl-learner/sample-pool"*|\
        *"/workspace/rl-learner/model-distributor"*)
            return 0
            ;;
    esac
    return 1
}

warn_container_drift() {
    if docker image inspect "${dev_image}" >/dev/null 2>&1 &&
       ! container_uses_current_image; then
        echo "learner-dev uses an older local image; run make dev-refresh when ready" >&2
    fi
    if container_has_legacy_artifact_mounts; then
        echo "learner-dev still has retired external artifact mounts; run make dev-refresh to use repository-owned inputs" >&2
    fi
}

create_container() {
    docker run --detach \
        --name "${container_name}" \
        --network "${network_name}" \
        --network-alias "${container_name}" \
        --network-alias "maze-learner" \
        --publish "127.0.0.1::9005" \
        --volume "${repo_dir}:/workspace/rl-learner" \
        "${dev_image}" >/dev/null
}

container_has_training_processes() {
    local process_status
    container_running || return 1
    set +e
    docker exec "${container_name}" sh -lc \
        "pgrep -f '[m]ain.training_runtime|[/]run.sh|[m]etrics_server.py|[m]aze_sample_pool|[m]aze_model_distributor' >/dev/null"
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

ensure_container_resources() {
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
}

ensure_container() {
    if container_exists; then
        if ! container_running; then
            docker start "${container_name}" >/dev/null
        fi
        warn_container_drift
        return
    fi

    ensure_dev_image
    ensure_container_resources
    stop_monitor_transport
    create_container
}

refresh_container() {
    local process_state
    if container_exists && container_running; then
        process_state=0
        container_has_training_processes || process_state=$?
        if [ "${process_state}" -eq 0 ]; then
            echo "learner-dev has active Learner, Sample Pool, Model Distributor, or Monitor processes" >&2
            echo "Stop the active training chain before refreshing learner-dev" >&2
            exit 1
        elif [ "${process_state}" -ne 1 ]; then
            exit 1
        fi
    fi

    ensure_dev_image
    ensure_container_resources
    stop_monitor_transport
    if container_exists; then
        if container_running; then
            docker stop --time 5 "${container_name}" >/dev/null
        fi
        docker rm "${container_name}" >/dev/null
    fi
    create_container
    echo "Learner development container refreshed: ${container_name}"
}

case "${action}" in
    image)
        build_image
        ;;
    refresh)
        refresh_container
        ;;
    shell)
        ensure_container
        monitor_url=""
        if prepare_monitor_transport; then
            monitor_url="http://127.0.0.1:${monitor_host_port}/monitor"
        else
            echo "Learner Monitor host forwarding unavailable; shell continues" >&2
        fi
        if [ -n "${monitor_url}" ]; then
            exec docker exec -it \
                --env "RL_DEVELOPMENT_MONITOR_URL=${monitor_url}" \
                --env "RL_DEVELOPMENT_MONITOR_CONTAINER_PORT=9005" \
                "${container_name}" bash
        fi
        exec docker exec -it "${container_name}" bash
        ;;
    build)
        ensure_container
        docker exec "${container_name}" sh -lc \
            "cd /workspace/rl-learner && python3 -m compileall -q main proto src tools"
        ;;
    monitor)
        ensure_container
        prepare_monitor_transport
        if ! wait_monitor_identity; then
            stop_monitor_transport
            echo "MONITOR_TARGET_UNAVAILABLE: Learner metrics identity did not become ready" >&2
            exit 1
        fi
        printf 'Learner Monitor: http://127.0.0.1:%s/monitor\n' \
            "${monitor_host_port}"
        ;;
    monitor-stop)
        stop_monitor_transport
        ;;
    clean)
        stop_monitor_transport
        if container_exists; then
            if container_running; then
                process_state=0
                container_has_training_processes || process_state=$?
                if [ "${process_state}" -eq 0 ]; then
                    echo "learner-dev has active Learner business processes" >&2
                    echo "Stop the active training chain before make dev-clean" >&2
                    exit 1
                elif [ "${process_state}" -ne 1 ]; then
                    exit 1
                fi
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
