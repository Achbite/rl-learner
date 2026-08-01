# RL Learner

简体中文 | [English](README.en.md)

本地 PPO 训练服务，负责监管 LocalSampleService、ModelDistributor、训练进程和指标 API。LocalSampleService 在 `9100` 接收样本，ModelDistributor 在 `9200` 发布模型，指标 API 使用 `9005`。

## 快速开始

先构建 Contracts、LocalSampleService 和 ModelDistributor，再显式装配两个运行制品：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-sample-pool && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-sample-pool/0.6.0/linux-arm64/. \
  sample-pool/
cp -R ../.workspace/artifacts/rl-model-distributor/0.5.0/linux-arm64/. \
  model-distributor/
```

构建镜像：

```bash
LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

进入开发容器并启动：

```bash
make shell
bash ./run.sh training
```

该命令会清理 Learner 专属的 `models/local-train` 并从零开始。只有显式指定完整 checkpoint 才加载训练状态：

```bash
bash ./run.sh training \
  --initial-checkpoint /absolute/path/to/archive/000200/checkpoint.pt
```

长期保存点使用固定布局：

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
```

完整三容器训练建议从 `rl-framework` 启动：

```bash
../rl-framework/framework local-test --profile training --json
```

## 测试

```bash
make test
```

## License

[MIT License](LICENSE)
