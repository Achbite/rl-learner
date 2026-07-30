#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/learner
exec ./run.sh "${MAZE_WORKLOAD:-training}"
