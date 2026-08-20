# SamplePoolService Runtime Artifact

This directory stages the SamplePoolService artifact selected for the Learner
image. The artifact is copied here explicitly after `rl-sample-pool` publishes
it to the workspace artifact store.

For the versions in `../artifact_versions.env`, run from `rl-learner`:

```bash
bash scripts/sync_runtime_artifacts.sh
```

`bin/`, `config/`, and `manifest.json` are generated inputs and are not
committed. The sync command and `build_image.sh` verify their package, version,
platform, clean provenance, Contracts identity, and SHA-256 values.
