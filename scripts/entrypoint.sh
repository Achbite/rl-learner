#!/usr/bin/env bash

set -euo pipefail

cd /opt/rl/learner
exec ./run.sh "${RL_WORKLOAD:-training}"
