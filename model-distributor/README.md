# ModelDistributor Runtime Artifact

This directory is the staging location for the ModelDistributor artifact
selected for the Learner image. The artifact is copied here explicitly after
`rl-model-distributor` publishes it to the workspace artifact store.

For the versions in `../artifact_versions.env`, run from `rl-learner`:

```bash
cp -R ../.workspace/artifacts/rl-model-distributor/0.2.0/linux-arm64/. \
    model-distributor/
```

`bin/`, `config/`, and `manifest.json` are generated inputs and are not
committed. `build_image.sh` verifies their package, version, platform,
Contracts identity, and SHA-256 values before building the image; an older
staged artifact is rejected.
