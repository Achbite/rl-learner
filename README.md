# RL Learner

简体中文 | [English](README.en.md)

Learner 容器内运行 PPO、LocalSampleService、ModelDistributor 和可选监控。本地完整三端链路由开发者分别启动 Learner、AIServer 和 Client；Framework 只提供只读诊断，不参与运行编排。

## 1. 组件开发环境

```bash
# 宿主机：自动准备 dirty-capable 开发依赖、构建或复用开发镜像并进入容器
make shell

# 宿主机：复用同一开发镜像和依赖身份进行编译
make build

# 容器内：唯一测试入口（测试清单受 TCR 管理）
bash ./test.sh
```

开发入口不依赖旧 runtime image，也不要求正式 0.13 artifact 或 clean source。开发依赖只写入
`.workspace/dev-artifacts`，不得用于正式镜像。`make shell` 只能在宿主机执行。

## 2. 启动 Learner 侧服务

进入容器后修改 config，并直接启动 Learner。Learner 只有 training workload，不接受位置 workload：

```bash
./run.sh --help
./run.sh --config configs/learner_config.yaml
```

`--help` 只打印实际支持的覆盖项及对应 config 字段，不启动 Sample Pool、Model Distributor、
PPO 或监控进程。

`configs/learner_config.yaml` 是完整默认事实源；启动时只执行一次
`config -> RL_PPO_*/RL_TRAINING_* 白名单环境覆盖 -> CLI 覆盖 -> 校验`。支持的业务 CLI
只有 `--initial-model`、`--model-distributor`、`--aiserver` 和 `--metrics-port`，分别覆盖
config 中已经存在的 warm-start、Distributor、AIServer 和 Dashboard 字段。`run.sh` 与 PPO
Runtime 复用同一解析器，shell 不保存第二套训练目录、端点或端口默认值。相对路径统一相对
所选 config 文件所在目录解析。

每次 invocation 都是新的 task-neutral 训练。Learner 只看到自己的直接 `models/train`，不接收
也不理解平台 `task_id/run_id`；`model_step`、Update、样本计数和优化器状态均从 0 开始。
`run.sh` 取得同级 workspace lock 后会清空 config 指定的 `model.local_train_dir`，然后生成
新的内部 lineage 并发布随机 `0000000`。该目录必须以 `/train` 结尾且不能是符号链接；清理
范围严格限制在这个目录的子项。需要保留或继承的模型必须在下一次启动前复制到该目录之外。

没有 AIServer 时，Learner 默认无限等待 AIServer 对 bootstrap 模型的 exact ACK，并保持
Sample Pool、Model Distributor 和监控存活；等待只会因 exact ACK、显式 `SIGINT/SIGTERM`
或 config 中显式设置的正数 `aiserver_status.initial_model_ack_timeout_sec` 结束。Client 可以在
AIServer ready 后再启动。

launcher 会打印本次可用的监控 URL。`9005` 只属于可选观测，端口或监控失败不会终止 PPO。

## 3. 从已有模型开始全新训练

指定要继承的 `SaveModel.onnx` 文件：

```text
models/save/0002355/
  SaveModel.onnx
```

启动：

```bash
./run.sh --config configs/learner_config.yaml \
  --initial-model /workspace/rl-learner/models/save/0002355/SaveModel.onnx
```

该入口只读取显式文件的权重；它仍是全新独立训练，launcher 自动生成新的内部模型 lineage，`model_step`、优化器、RNG、更新数和样本计数从 0 开始。
config 的 `model.initial_model_path` 默认为 `null`；在 config 中显式设置路径，或使用
`--initial-model` 覆盖，都会触发同一种权重继承。两种方式都必须直接指向常规、非符号链接的
`SaveModel.onnx`，且不能位于本次新的 `model.local_train_dir` 中。

## 4. 训练模型包

每次完整发布使用同一公开布局；checkpoint 只存在于本次训练的私有 runtime：

```text
models/train/0000200/
  SaveModel.onnx
  manifest.json
  metadata.json

models/train/runtime/checkpoints/
  publication-0000200.checkpoint.pt
```

正常停止不会立即删除公开模型包；它们会保留到下一次 `run.sh` 清空同一个
`model.local_train_dir`。私有 checkpoint 不属于模型包，也不能作为后续训练的入口；要保留或
继承某个模型，必须先将 `SaveModel.onnx` 放到训练目录之外，再通过 config 或
`--initial-model` 显式读取。后续训练不会自动恢复旧 Update。

## 5. 正式制品与镜像

只有 Level 1/2 通过、用户 Review 并形成 clean savepoint 后，才在宿主机依次构建正式
Contracts/Pool/Distributor artifact、同步 runtime 依赖并运行 `bash build_image.sh`。正式脚本
不读取 `.workspace/dev-artifacts` 或开发容器的可变 build 目录。

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
