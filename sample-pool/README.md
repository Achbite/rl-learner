# SamplePoolService Runtime Artifact

This directory stages the SamplePoolService artifact selected for the Learner
image. The artifact is copied here explicitly after `rl-sample-pool` publishes
it to the workspace artifact store.

To stage the current sibling repository build, run from `rl-learner`:

```bash
bash scripts/sync_runtime_artifacts.sh
```

`bin/` is a generated input and is not committed. The sync always replaces the
binary, but copies the default `config/pool_config.yaml` only when no target
config exists. The runtime check verifies only that the required regular files
exist and that the binary is executable; it performs no package, version,
platform, repository, Contracts, or hash admission.
