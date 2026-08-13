# RL Learner

简体中文 | [English](README.en.md)

Learner 容器内运行 PPO、Sample Pool、Model Distributor 和可选监控。完整三端链路请从 [rl-framework](../rl-framework/README.md) 启动。

## 1. 准备制品

在八仓同级目录中依次执行：

```bash
(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
```

构建 Learner 镜像；该命令同时生成匹配版本的 smoke model：

```bash
RL_LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

## 2. 组件开发环境

```bash
# 从已经构建的 training-001 运行镜像构建开发镜像
LEARNER_IMAGE_TAG=training-001 make dev-image

# 进入源码挂载容器
make shell

# Python 编译检查
make build

# 运行测试
make test
```

## 3. 启动 Learner 侧服务

新 Run 会清空 Learner 自己的 `models/local-train`，从模型版本 0 开始：

```bash
bash scripts/dev_container.sh training --new-run
```

正常停止后继续同一个 Run 时不要传 `--new-run`：

```bash
bash scripts/dev_container.sh training
```

这只启动 Learner 侧服务；没有 AIServer 和 Client 时会等待链路。完整训练请使用 Framework。

launcher 会打印本次可用的监控 URL。`9005` 只属于可选观测，端口或监控失败不会终止 PPO。

## 4. 从已有模型开始新 Run

将完整保存点放在 `models/local-train` 之外、容器可读取的位置，例如：

```text
models/import/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

启动：

```bash
bash scripts/dev_container.sh training \
  --new-run \
  --initial-model-dir /workspace/rl-learner/models/import/000200
```

该入口只继承模型权重和来源信息；新 lineage 的模型版本、优化器、RNG、更新数和样本计数从 0 开始。

## 5. 归档与恢复

每次完整发布使用同一布局：

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

正常停止保留归档和运行状态；不带 `--new-run` 的下一次启动恢复同一个 Run。只有显式 `--new-run` 才清理 Learner 本地 Run 数据。

## 6. 端口

| 端口 | 服务 |
| ---: | --- |
| `9100` | Sample Pool |
| `9200` | Model Distributor |
| `9005` | 可选 Learner Monitor |

监控 API：`/monitor`、`/api/status`、`/api/metrics/catalog`、`/api/metrics/latest`、`/api/metrics`。

## 7. 清理开发容器

```bash
make dev-clean
```

该命令删除 `learner-dev` 和 launcher 自己创建的转发，不删除未知宿主进程。

## License

[MIT License](LICENSE)
