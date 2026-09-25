# ModelDistributor Runtime Artifact

This directory is the staging location for the ModelDistributor artifact
selected for the Learner image. The artifact is copied here explicitly after
`rl-model-distributor` publishes it to the workspace artifact store.

To stage explicitly selected artifacts, run from `rl-learner`:

```bash
bash scripts/sync_runtime_artifacts.sh \
  --sample-pool-dir /path/to/pool-artifact \
  --model-distributor-dir /path/to/distributor-artifact
```

`bin/` is a generated input and is not committed. The sync updates changed binary content, but copies the default `config/model_distributor_config.yaml` only when
no target config exists. The runtime check verifies only that the required
regular files exist and that the binary is executable; it performs no package,
version, platform, repository, Contracts, or hash admission.
