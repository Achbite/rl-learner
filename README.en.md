# RL Learner

English | [简体中文](README.md)

Local PPO training service. It supervises LocalSampleService, ModelDistributor, the training runtime, and the metrics API. LocalSampleService accepts samples on `9100`, ModelDistributor publishes models on `9200`, and the metrics API uses `9005`.

## Quick Start

Build Contracts, LocalSampleService, and ModelDistributor, then explicitly stage both runtime artifacts:

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-sample-pool && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-sample-pool/0.6.0/linux-arm64/. \
  sample-pool/
cp -R ../.workspace/artifacts/rl-model-distributor/0.5.0/linux-arm64/. \
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
  --initial-checkpoint /absolute/path/to/archive/000200/checkpoint.pt
```

Long-term savepoints use a fixed layout:

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
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
