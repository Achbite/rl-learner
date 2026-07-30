# RL Learner

English | [简体中文](README.md)

Local PPO training service. It consumes samples, updates models, starts ModelDistributor, and exposes the training metrics API on `9005`.

## Quick Start

Build Contracts and ModelDistributor, then stage the binary:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-model-distributor/0.4.0/linux-arm64/. \
  model-distributor/
```

Build the image:

```bash
LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

Enter the development container and start the service:

```bash
make shell
bash ./run.sh training
```

This clears the Learner-owned `models/local-train` directory and starts from zero. Load training state only from an explicitly selected complete checkpoint:

```bash
bash ./run.sh training \
  --initial-checkpoint /absolute/path/to/checkpoint_v000200.pt
```

Use `rl-framework` to start the complete three-container training workflow:

```bash
../rl-framework/framework local-test --profile training --json
```

## Tests

```bash
make test
```

## License

[MIT License](LICENSE)
