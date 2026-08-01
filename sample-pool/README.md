# LocalSampleService Runtime Artifact

This directory stages the LocalSampleService artifact selected for the Learner
image. The artifact is copied here explicitly after `rl-sample-pool` publishes
it to the workspace artifact store.

For the versions in `../artifact_versions.env`, run from `rl-learner`:

```bash
cp -R ../.workspace/artifacts/rl-sample-pool/0.6.0/linux-arm64/. \
    sample-pool/
```

`bin/`, `config/`, and `manifest.json` are generated inputs and are not
committed. `build_image.sh` verifies their package, version, platform,
Contracts identity, and SHA-256 values before building the image.
