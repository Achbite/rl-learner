# RL Learner

[简体中文](README.md) | English

The Learner container runs PPO, SamplePoolService, ModelDistributor, and the
optional monitor. For local training, start Learner first, followed by AIServer
and Client.

Local training runs only three containers: Learner, AIServer, and Client.
`make shell` is a host command that prepares development artifacts from sibling
source repositories; it does not download those repositories. A fresh workspace
therefore needs at least these sibling directories:

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

The first three repositories supply development artifacts only and do not add
runtime containers. See [rl-framework](https://github.com/Achbite/rl-framework)
for the complete three-container startup order.

## 1. Component development environment

```bash
# Host: prepare dirty-capable dependencies, build or reuse the dev image, and enter it
make shell

# Host: reuse the same development image and dependency identity for builds
make build

# Container: unified test entrypoint
bash ./test.sh
```

The development path does not depend on an old runtime image, prebuilt formal
artifacts, or clean source. Development dependencies remain under
`.workspace/dev-artifacts` and cannot feed a formal image. Run `make shell` only
on the host.

## 2. Start Learner-side services

Open the first host terminal, enter the development container, and start Learner
directly. Learner has only the training workload and accepts no positional
workload:

```bash
# Host
cd /path/to/workspace/rl-learner
make shell

# Run the following commands inside the Learner container
./run.sh --help
./run.sh --monitor --config configs/learner_config.yaml
# Disable only the local three-container preview, not raw metrics or training
./run.sh --no-monitor --config configs/learner_config.yaml
```

`--help` prints the supported overrides and their config fields without
starting Sample Pool, Model Distributor, PPO, or the monitor.

`configs/learner_config.yaml` is the complete default source. Startup applies
allowlisted PPO/training environment overrides and then CLI overrides before
validation. `--initial-model`, `--model-distributor`, `--aiserver`,
`--metrics-port`, `--monitor`, and `--no-monitor` only override existing config
fields. The supervisor and PPO
runtime share the same parser/loader, and relative paths are resolved against
the selected config file.

Every invocation is a new task-neutral training. Learner sees only its direct
`models/train`, receives no platform `task_id/run_id`, and starts `model_step`,
updates, sample count, and optimizer state at zero. After taking the sibling
workspace lock, `run.sh` clears the configured `model.local_train_dir`, creates
a new internal lineage, and publishes random `0000000`. The directory must end
in `/train` and must not be a symlink; cleanup is restricted to its children.
Copy any model that must survive or seed the next invocation outside this
directory before starting again.

Without AIServer, Learner keeps Sample Pool, Model Distributor, and monitoring
alive while waiting without a deadline for the exact bootstrap-model ACK. The
wait ends only on that ACK, explicit `SIGINT/SIGTERM`, or a positive
`aiserver_status.initial_model_ack_timeout_sec` configured for a bounded
diagnostic. Client can start after AIServer is ready.

`dashboard.enabled: true` is the local default. Infra can inject the strict
boolean environment variable `RL_LEARNER_LOCAL_MONITOR_ENABLED=false`; CLI
`--monitor/--no-monitor` takes precedence. Disabling the preview skips only the
HTTP/HTML MetricsServer and does not disable JSONL, MetricEvent, SQLite,
AIServer relay, the projector, or training.

`run.sh` prints a browser URL only when the development launcher supplied one
for the effective container port; otherwise it reports a container-only URL or
that the host URL is unavailable. Linux/WSL direct mode uses the actual host
port returned by `docker port`, for example
`http://127.0.0.1:32793/monitor`; macOS/Colima keeps
`http://127.0.0.1:9005/monitor` through an SSH tunnel. MetricsServer uses a
dedicated background thread to tail the current JSONL continuously: it drains
backlog without waiting and switches to periodic polling only after catching
up. HTTP requests read the in-memory projection and do not control file-read
progress. `/api/status` exposes the tail, backlog, and error facts. Port `9005`
is optional observability; a monitor or port failure does not stop PPO.

Learner does not consume raw trajectories or compute GAE. It
asks SamplePool to draw `training.train_batch_size` READY processed transitions
uniformly without replacement, normalizes advantages once over the full batch,
then runs PPO/optimizer work according to `mini_batch_size` and `n_epochs`. A
batch may contain multiple behavior-model steps; each transition retains exact
lineage, step, and digest provenance for lag observation.

## 3. Start fresh training from an existing model

Select an explicit `SaveModel.onnx` file:

```text
models/save/0002355/
  SaveModel.onnx
```

Start the Run:

```bash
./run.sh --config configs/learner_config.yaml \
  --initial-model /workspace/rl-learner/models/save/0002355/SaveModel.onnx
```

This reads only the selected file's weights. It is still an independent fresh training invocation, and the launcher generates a new internal model lineage; `model_step`, optimizer, RNG, update count, and sample count start at zero.
`model.initial_model_path` defaults to `null`. Setting it in config or overriding
it with `--initial-model` enables the same weight-only warm start. The value
must name a regular, non-symlink `SaveModel.onnx` outside the fresh training
workspace.

## 4. Training model package

Every complete publication uses one public layout. Checkpoints remain only in
that training invocation's private runtime:

```text
models/train/0000200/
  SaveModel.onnx
  manifest.pb

models/train/runtime/checkpoints/
  publication-0000200.checkpoint.pt
```

`manifest.pb` is the sole authoritative model-package manifest. Learner-private
checkpoints retain update metadata; the public model directory no longer keeps
parallel JSON manifest or metadata projections.

A normal stop preserves public model packages until the next `run.sh` clears the
same `model.local_train_dir`. A private checkpoint is not part of a model package
and cannot start later training. To preserve or inherit a model, first place its
`SaveModel.onnx` outside the training directory, then select it explicitly in
config or with `--initial-model`. Later training never restores the previous
update counter automatically.

## 5. Build the runtime image

The runtime image packages the current Learner source and the synchronized
Contracts, Sample Pool, and Model Distributor runtime artifacts. Synchronize the
runtime dependencies and build it with an explicit project tag from the host:

```bash
bash scripts/sync_runtime_artifacts.sh
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

The scripts never consume `.workspace/dev-artifacts` or a mutable development
container build directory. The build does not derive a cross-repository stack
source identity. Without an override it uses `maze-tag-001`; a later tuning build
may overwrite the same tag.

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
