# ModelDistributor Runtime Artifact

This directory is the staging location for the ModelDistributor artifact
selected for the Learner image. The artifact is copied here explicitly after
`rl-model-distributor` publishes it to the workspace artifact store.

For the versions in `../artifact_versions.env`, run from `rl-learner`:

```bash
source artifact_versions.env
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
platform_dir="${platform//\//-}"
cp -R "../.workspace/artifacts/rl-model-distributor/${RL_MODEL_DISTRIBUTOR_VERSION}/${platform_dir}/." \
    model-distributor/
```

`bin/`, `config/`, and `manifest.json` are generated inputs and are not
committed. `build_image.sh` verifies their package, version, platform,
Contracts identity, and SHA-256 values before building the image; an older
staged artifact is rejected.
