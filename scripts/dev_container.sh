#!/usr/bin/env bash

set -euo pipefail

action="${1:-shell}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${RL_TRAINING_WORKSPACE:-$(cd "${repo_dir}/.." && pwd)}"
container_name="learner-dev"
network_name="rl-training-dev"
tag="${LEARNER_DEV_IMAGE_TAG:-test-001}"
runtime_image="rl-training/learner:${LEARNER_IMAGE_TAG:-test-001}"
dev_image="rl-training/learner-dev:${tag}"
source "${repo_dir}/artifact_versions.env"
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"

build_image() {
    docker build \
        --file "${repo_dir}/Dockerfile.dev" \
        --build-arg "LEARNER_RUNTIME_IMAGE=${runtime_image}" \
        --tag "${dev_image}" \
        "${repo_dir}"
}

ensure_container() {
    contract_dir="${workspace_root}/.workspace/artifacts/rl-contracts/${RL_CONTRACTS_VERSION}/${platform_dir}"
    if [ ! -f "${contract_dir}/python/maze_pb2.py" ]; then
        echo "Contract artifact is missing. Run ../rl-contracts/build_artifact.sh" >&2
        exit 1
    fi
    if ! docker image inspect "${dev_image}" >/dev/null 2>&1; then
        build_image
    fi
    if ! docker network inspect "${network_name}" >/dev/null 2>&1; then
        docker network create "${network_name}" >/dev/null
    fi
    if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
        docker run --detach \
            --name "${container_name}" \
            --network "${network_name}" \
            --network-alias "${container_name}" \
            --network-alias "maze-learner" \
            --publish "127.0.0.1:9005:9005" \
            --volume "${repo_dir}:/workspace/rl-learner" \
            --volume "${contract_dir}/python/maze_pb2.py:/workspace/rl-learner/proto/maze_pb2.py:ro" \
            --volume "${contract_dir}/python/maze_pb2_grpc.py:/workspace/rl-learner/proto/maze_pb2_grpc.py:ro" \
            "${dev_image}" >/dev/null
    elif [ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" != "true" ]; then
        docker start "${container_name}" >/dev/null
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
    clean)
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
