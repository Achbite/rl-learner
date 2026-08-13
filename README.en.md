# RL Learner

[简体中文](README.md) | English

The Learner container runs PPO, Sample Pool, Model Distributor, and the optional monitor. Start the full three-component chain from [rl-framework](../rl-framework/README.en.md).

## 1. Prepare artifacts

From the directory containing all repositories, run:

```bash
(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

Build the Learner image. This command also creates the matching smoke model:

```bash
RL_LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

## 2. Component development environment

```bash
# Build the development image from the training-001 runtime image
LEARNER_IMAGE_TAG=training-001 make dev-image

# Enter the source-mounted container
make shell

# Run the Python compile check
make build

# Run tests
make test
```

## 3. Start Learner-side services

A new Run clears the Learner-owned `models/local-train` and starts at model version zero:

```bash
bash scripts/dev_container.sh training --new-run
```

To continue the same Run after a normal stop, omit `--new-run`:

```bash
bash scripts/dev_container.sh training
```

This starts only Learner-side services and waits when no AIServer or Client is connected. Use Framework for full training.

The launcher prints the monitor URL for that invocation. Port `9005` is optional observability; a monitor or port failure does not stop PPO.

## 4. Start a new Run from an existing model

Place the complete savepoint outside `models/local-train` at a path visible inside the container, for example:

```text
models/import/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

Start the Run:

```bash
bash scripts/dev_container.sh training \
  --new-run \
  --initial-model-dir /workspace/rl-learner/models/import/000200
```

This inherits only model weights and provenance. The new lineage starts model version, optimizer, RNG, update count, and sample count from zero.

## 5. Archive and resume

Every complete publication uses one layout:

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

A normal stop preserves the archive and runtime state. A later start without `--new-run` resumes the same Run. Only explicit `--new-run` removes Learner-local Run data.

## 6. Ports

| Port | Service |
| ---: | --- |
| `9100` | Sample Pool |
| `9200` | Model Distributor |
| `9005` | Optional Learner Monitor |

Monitor endpoints: `/monitor`, `/api/status`, `/api/metrics/catalog`, `/api/metrics/latest`, and `/api/metrics`.

## 7. Remove the development container

```bash
make dev-clean
```

This removes `learner-dev` and launcher-owned forwarding only. It never terminates unknown host processes.

## License

[MIT License](LICENSE)
