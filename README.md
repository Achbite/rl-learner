# RL Learner

简体中文 | [English](README.en.md)

本地 PPO 训练服务，负责消费样本、更新模型、启动 ModelDistributor，并在 `9005` 提供训练指标 API。

## 快速开始

先构建 Contracts 和 ModelDistributor，并装配二进制：

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
cp -R ../.workspace/artifacts/rl-model-distributor/0.4.0/linux-arm64/. \
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
  --initial-checkpoint /absolute/path/to/checkpoint_v000200.pt
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
