# RL Learner

简体中文 | [English](README.en.md)

Learner 容器内运行 PPO、SamplePoolService、ModelDistributor 和可选监控。本地训练时先启动
Learner，再启动 AIServer 和 Client。

本地训练只运行 Learner、AIServer 和 Client 三个容器。`make shell` 是宿主机命令，会从同一父目录的
源码准备开发制品，但不会自动下载依赖仓库。冷启动工作区至少需要以下同级目录：

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

前三个依赖仓只提供开发制品，不会增加运行容器。完整的三容器启动顺序也可参阅
[rl-framework](https://github.com/Achbite/rl-framework)。

## 1. 组件开发环境

```bash
# 宿主机：自动准备 dirty-capable 开发依赖、构建或复用开发镜像并进入容器
make shell

# 宿主机：复用同一开发镜像和依赖身份进行编译
make build

# 容器内：统一测试入口
bash ./test.sh
```

开发入口不依赖旧 runtime image，也不要求正式 0.14 artifact 或 clean source。开发依赖只写入
`.workspace/dev-artifacts`，不得用于正式镜像。`make shell` 只能在宿主机执行。

## 2. 启动 Learner 侧服务

打开第一个宿主终端，进入开发容器后直接启动 Learner。Learner 只有 training workload，不接受位置
workload：

```bash
# 宿主机
cd /path/to/workspace/rl-learner
make shell

# 以下命令在 Learner 容器内执行
./run.sh --help
./run.sh --monitor --config configs/learner_config.yaml
# 只关闭本地三容器预览，不关闭原始指标或训练
./run.sh --no-monitor --config configs/learner_config.yaml
```

`--help` 只打印实际支持的覆盖项及对应 config 字段，不启动 Sample Pool、Model Distributor、
PPO 或监控进程。

`configs/learner_config.yaml` 是完整默认事实源；启动时只执行一次
`config -> RL_PPO_*/RL_TRAINING_* 白名单环境覆盖 -> CLI 覆盖 -> 校验`。支持的业务 CLI
只有 `--initial-model`、`--model-distributor`、`--aiserver`、`--metrics-port`、`--monitor` 和
`--no-monitor`，分别覆盖 config 中已经存在的 warm-start、Distributor、AIServer 和 Dashboard
字段。`run.sh` 与 PPO
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

`dashboard.enabled: true` 是本地默认值。Infra 可注入严格布尔环境变量
`RL_LEARNER_LOCAL_MONITOR_ENABLED=false` 关闭预览，CLI `--monitor/--no-monitor` 优先于环境变量。
关闭预览只跳过 HTTP/HTML MetricsServer，不关闭 JSONL、MetricEvent、SQLite、AIServer relay、
projector 或训练线程。

`run.sh` 只在开发 launcher 已提供与当前容器端口一致的 host URL 时打印可由浏览器访问的地址；
否则明确报告仅容器内可用或 host URL unavailable。Linux/WSL direct 模式使用 `docker port`
返回的真实宿主发布端口，例如 `http://127.0.0.1:32793/monitor`；macOS/Colima 模式通过
SSH tunnel 保持 `http://127.0.0.1:9005/monitor`。MetricsServer 使用独立后台线程持续 tail 当前 JSONL；存在
积压时连续追赶，追平后再按固定间隔检查。HTTP 请求只读取内存 projection，不参与控制磁盘读取
进度。`/api/status` 会报告 tail 运行、backlog 和错误事实。`9005` 只属于可选观测，端口或监控
失败不会终止 PPO。

Learner 不读取 raw trajectory 或计算 GAE。它请求 SamplePool 从 READY 集合随机无放回
抽取 `training.train_batch_size` 条 processed transition，对整批 advantage 做一次归一化，再按
`mini_batch_size` 与 `n_epochs` 执行 PPO/optimizer。一个 batch 可以包含多个 behavior model step；
每条 transition 的 lineage/step/digest 仍作为真实 provenance 和 lag 指标保留。

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

## 5. 构建运行镜像

运行镜像封装当前 Learner 源码和已同步的 Contracts、Sample Pool、Model Distributor 运行产物。
在宿主机完成依赖同步后，以明确的项目 tag 构建：

```bash
bash scripts/sync_runtime_artifacts.sh
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

脚本不读取 `.workspace/dev-artifacts` 或开发容器的可变 build 目录，不计算跨仓 stack source
identity。未指定时使用 `maze-tag-001`；同名 tag 允许由后续微调构建直接覆盖。

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
