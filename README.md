# RL Learner

简体中文 | [English](README.en.md)

RL Learner 是容器化 PPO 训练服务。默认训练工作负载在同一 Learner 容器内启动
Sample Pool、Model Distributor、Training Runtime 和只读指标服务。

| 服务 | 端口 | 用途 |
| --- | ---: | --- |
| Sample Pool | `9100` | 接收、租约和处置训练样本 |
| Model Distributor | `9200` | 注册、分发模型并接收加载 ACK |
| Learner Monitor | `9005` | 展示训练状态并提供只读指标 API |

## 运行要求

- Docker；macOS 推荐使用 Colima。
- 仓库位于 RL-Training-Framework 工作区中，与 `rl-contracts`、
  `rl-sample-pool` 和 `rl-model-distributor` 同级。
- 已生成匹配版本的 Contracts、Sample Pool 和 Model Distributor 制品。

## 准备制品

```bash
(cd ../rl-contracts && bash build_artifact.sh)
(cd ../rl-sample-pool && bash build_artifact.sh)
(cd ../rl-model-distributor && bash build_artifact.sh)
bash scripts/sync_runtime_artifacts.sh
```

`artifact_versions.env` 固定选择 Contracts、Sample Pool、Model Distributor 及目标平台。
同步入口会在复制前后校验版本、平台、clean savepoint、Contracts 身份和全部文件的
SHA-256；不读取 `latest`，也不会改写工作区 artifact store。

构建运行镜像：

```bash
LEARNER_IMAGE_TAG=training-001 bash build_image.sh
```

## 本地训练

### 推荐启动方式

在宿主终端执行：

```bash
bash scripts/dev_container.sh training
```

训练进程和依赖全部运行在 `learner-dev` 容器内。宿主 launcher 只负责创建开发容器、
执行容器内的 `run.sh training`，以及在需要时管理本轮拥有的 Colima `9005` 转发。
训练退出或收到信号后，launcher 会精确清理自己的转发。

启动后打开：

```text
http://127.0.0.1:9005/
```

页面每秒读取当前训练实例的只读指标，分面展示 Loss、Episode Return、
Reward Components、Episode Success、Sample Throughput、Sample Flow、Latency 和
PPO Stability 曲线。

每次 fresh training 都会清空 Learner 专属的 `models/local-train` 工作目录。

### 交互式容器

需要在开发容器内手工运行命令时：

```bash
make shell
```

进入容器后启动训练：

```bash
bash ./run.sh training
```

`run.sh` 只管理容器内工作负载。若需要从宿主浏览器访问这次手工训练，请在另一个
宿主终端创建并核验监控转发：

```bash
bash scripts/dev_container.sh monitor
```

手工训练结束后停止该转发：

```bash
bash scripts/dev_container.sh monitor-stop
```

## 监控接口

| 路径 | 内容 |
| --- | --- |
| `/` | `302` 跳转至 `/monitor` |
| `/monitor` | Learner 本地训练监控页面 |
| `/api` | API 索引 |
| `/api/status` | 当前服务、训练实例和 freshness |
| `/api/metrics/catalog` | 版本化指标字段目录 |
| `/api/metrics/latest` | 最新指标记录 |
| `/api/metrics` | 分页指标记录 |
| `/api/metrics/summary` | 当前训练摘要 |

`/api/metrics` 和 `/api/metrics/latest` 保留原始嵌套记录，并在响应副本中增加按
`field_id` 索引的 `metric_values`。字段的 label、dimension、unit、scope 和统计口径
以 catalog 为准；磁盘上的 JSONL 不会因此改写。

launcher 遇到未知的宿主 `9005` listener 时只报告
`PORT_IDENTITY_CONFLICT` 或 `MONITOR_TARGET_UNAVAILABLE`，不会终止未知进程。

## 从 Checkpoint 启动

Checkpoint 必须位于 `models/local-train` 之外，并且路径必须能从容器内读取：

```bash
bash scripts/dev_container.sh training \
  --initial-checkpoint /workspace/rl-learner/models/checkpoints/000200/checkpoint.pt
```

长期保存点采用固定布局：

```text
models/local-train/archive/000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
```

## 完整本地训练

从 Framework 启动 Learner、AIServer 和 Client：

```bash
../rl-framework/framework local-test --profile training --json
```

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `make shell` | 进入源码挂载的开发容器 |
| `make test` | 在开发容器内运行 Learner 测试 |
| `make build` | 在开发容器内执行 Python compile smoke |
| `make dev-image` | 构建 Learner 开发镜像 |
| `make dev-clean` | 停止自有转发并删除 `learner-dev` 容器 |
| `bash scripts/dev_container.sh monitor` | 为手工训练创建并核验宿主监控转发 |
| `bash scripts/dev_container.sh monitor-stop` | 仅停止 Learner launcher 自有转发 |

## 测试

```bash
make test
```

## License

[MIT License](LICENSE)
