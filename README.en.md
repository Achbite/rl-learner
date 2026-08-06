# RL Learner

English | [简体中文](README.md)

RL Learner is a containerized PPO training service. By default, one Learner
container runs Sample Pool, Model Distributor, Training Runtime, and the
read-only metrics service.

| Service | Port | Purpose |
| --- | ---: | --- |
| Sample Pool | `9100` | Ingest, lease, and dispose of training samples |
| Model Distributor | `9200` | Register and distribute models and receive load acknowledgements |
| Learner Monitor | `9005` | Display training status and serve the read-only metrics API |

## Requirements

- Docker; Colima is recommended on macOS.
- A checkout inside the RL-Training-Framework workspace, next to
  `rl-contracts`, `rl-sample-pool`, and `rl-model-distributor`.
- Matching Contracts, Sample Pool, and Model Distributor artifacts.

## Prepare Artifacts

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-sample-pool && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-sample-pool/0.8.0/linux-arm64/. \
  sample-pool/
cp -R ../.workspace/artifacts/rl-model-distributor/0.8.0/linux-arm64/. \
  model-distributor/
```

Build the runtime image:

```bash
LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

## Local Training

### Recommended Startup

Run from a host terminal:

```bash
bash scripts/dev_container.sh training
```

All training processes and dependencies run inside the `learner-dev` container.
The host launcher only creates the development container, invokes
`run.sh training` in that container, and manages its own Colima forwarding for
port `9005` when required. It removes only that forwarding when training exits
or receives a signal.

Open the monitor after startup:

```text
http://127.0.0.1:9005/
```

The page reads the current training instance's read-only metrics every second
and uses separate panels for Loss, Episode Return, Reward Components, Episode
Success, Sample Throughput, Sample Flow, Latency, and PPO Stability.

Each fresh training run clears the Learner-owned `models/local-train` working
directory.

### Interactive Container

To run commands manually inside the development container:

```bash
make shell
```

Start training after entering the container:

```bash
bash ./run.sh training
```

`run.sh` manages only the in-container workload. To view this manually started
training run from the host browser, create and verify the monitor forwarding in
another host terminal:

```bash
bash scripts/dev_container.sh monitor
```

Stop that forwarding after the manual training run ends:

```bash
bash scripts/dev_container.sh monitor-stop
```

## Monitor Endpoints

| Path | Content |
| --- | --- |
| `/` | A `302` redirect to `/monitor` |
| `/monitor` | Local Learner training monitor |
| `/api` | API index |
| `/api/status` | Current service and training identity plus freshness |
| `/api/metrics/catalog` | Versioned metric field catalog |
| `/api/metrics/latest` | Latest metrics record |
| `/api/metrics` | Paginated metrics records |
| `/api/metrics/summary` | Current training summary |

`/api/metrics` and `/api/metrics/latest` preserve the original nested record
and add a `metric_values` projection keyed by `field_id` to the response copy.
Use the catalog for labels, dimensions, units, scopes, and statistics. This does
not rewrite the on-disk JSONL stream.

When the launcher finds an unknown listener on host port `9005`, it reports
`PORT_IDENTITY_CONFLICT` or `MONITOR_TARGET_UNAVAILABLE` and never terminates
the unknown process.

## Start From a Checkpoint

The checkpoint must be outside `models/local-train` and readable from inside the
container:

```bash
bash scripts/dev_container.sh training \
  --initial-checkpoint /workspace/rl-learner/models/checkpoints/000200/checkpoint.pt
```

Long-term savepoints use a fixed layout:

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
```

## Complete Local Training

Use Framework to start Learner, AIServer, and Client:

```bash
../rl-framework/framework local-test --profile training --json
```

## Common Commands

| Command | Purpose |
| --- | --- |
| `make shell` | Enter the source-mounted development container |
| `make test` | Run Learner tests inside the development container |
| `make build` | Run the Python compile smoke check inside the development container |
| `make dev-image` | Build the Learner development image |
| `make dev-clean` | Stop the owned forwarding and remove `learner-dev` |
| `bash scripts/dev_container.sh monitor` | Create and verify host monitor forwarding for manual training |
| `bash scripts/dev_container.sh monitor-stop` | Stop only the forwarding owned by the Learner launcher |

## Tests

```bash
make test
```

## License

[MIT License](LICENSE)
