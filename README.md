# RL Learner

简体中文 | [English](README.en.md)

Learner 容器内运行 PPO、SamplePoolService、ModelDistributor 和可选监控。本地训练时先启动
Learner，再启动 AIServer 和 Client。

本地训练只运行 Learner、AIServer 和 Client 三个容器。`make shell` 是宿主机命令：没有
`learner-dev` 时构建开发镜像、创建并启动容器；容器存在时只在必要时启动它，然后直接进入。
它不会构建或同步任何依赖。完整三容器工作区包含以下同级目录：

```text
workspace/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
```

前三个仓库不增加运行容器。Sample Pool 与 Model Distributor 是 Learner 必需的二进制依赖，
使用显式 `make deps` 构建并同步；该命令不会覆盖 Learner 本地 Proto 或 schema。完整启动顺序参阅
[rl-framework](https://github.com/Achbite/rl-framework)。

## 1. 组件开发环境

```bash
# 宿主机：首次使用或依赖变化后，显式构建并同步两个服务制品
make deps

# 宿主机：容器不存在时构建并创建；存在时直接复用并进入
make shell

# 宿主机：复用同一容器进行 Python 编译检查
make build

# Dockerfile.dev、工具链、端口、环境变量或挂载变化后显式刷新
make dev-refresh

# 容器内：统一测试入口
bash ./test.sh
```

`make deps` 使用工作区已显式生成的 Training Proto 编译输入构建 Sample Pool/Model Distributor
开发制品，并只把两个二进制与配置装配到 Learner；二进制始终更新，已存在的目标配置不会被覆盖。
Learner 的 Proto 始终由本仓维护。
开发制品只写入 `.workspace/dev-artifacts`，不得用于正式镜像。`make shell` 不调用 `make deps`，
也不要求源码 clean；它只能在宿主机执行。`make dev-refresh` 才会替换常驻容器，活动训练链存在时
会明确失败。

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

`dashboard.enabled: true` 是本地默认值。严格布尔环境变量
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
每条 transition 的 lineage/step 仍作为真实 provenance 和 lag 指标保留。Action mask 由
`policy.action_mask_mode` 控制：`disabled` 时样本必须不带 mask，`required` 时 Learner 按
`model.action_count` 校验并在 PPO logits 上应用；它不是所有任务的必选项。

## 3. 从已有模型开始全新训练

指定要继承的 ONNX 模型文件；文件名不参与训练合同：

```text
models/save/
  seed-policy.onnx
```

启动：

```bash
./run.sh --config configs/learner_config.yaml \
  --initial-model /workspace/rl-learner/models/save/seed-policy.onnx
```

该入口只读取显式文件的权重；它仍是全新独立训练，launcher 自动生成新的内部模型 lineage，`model_step`、优化器、RNG、更新数和样本计数从 0 开始。
config 的 `model.initial_model_path` 默认为 `null`；在 config 中显式设置路径，或使用
`--initial-model` 覆盖，都会触发同一种权重继承。两种方式都必须直接指向常规、非符号链接的
非空 ONNX 模型文件，且不能位于本次新的 `model.local_train_dir` 中。实际 tensor 名称、dtype 与
shape 由 Learner 按 `model.observation_dimension`、`model.action_count` 和当前模型实现检查。

## 4. 训练模型包

每次完整发布使用同一公开布局；checkpoint 只存在于本次训练的私有 runtime：

```text
models/train/0000200/
  SaveModel.onnx
  manifest.pb

models/train/runtime/checkpoints/
  publication-0000200.checkpoint.pt
```

`manifest.pb` 是模型包的唯一权威清单；Learner 私有 checkpoint 保存本次 Update
的运行元数据，公开模型目录不再维护一份 JSON manifest 或 metadata 副本。

正常停止不会立即删除公开模型包；它们会保留到下一次 `run.sh` 清空同一个
`model.local_train_dir`。私有 checkpoint 不属于模型包，也不能作为后续训练的入口；要保留或
继承某个模型，必须先将模型文件放到训练目录之外，再通过 config 或
`--initial-model` 显式读取。后续训练不会自动恢复旧 Update。

## 5. 构建运行镜像

运行镜像封装当前 Learner 源码、本仓维护的 Proto，以及已同步的 Sample Pool 与
Model Distributor 运行产物。正式同步脚本更新两个本地服务的二进制，只在目标配置不存在时复制
默认配置，不会从
Contracts 覆盖 Learner 源码。首次构建或依赖版本变化后，在宿主机显式同步并使用项目 tag 构建：

```bash
bash scripts/sync_runtime_artifacts.sh
RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh
```

正式脚本不读取 `.workspace/dev-artifacts` 或开发容器的可变 build 目录。装配检查只确认两个必需
二进制和配置存在、二进制可执行，不读取包版本、平台、manifest、仓库身份或哈希。构建不计算
跨仓 stack source identity。未指定时使用
`maze-tag-001`；同名 tag 允许由后续微调构建直接覆盖。

## 6. 端口

| 端口 | 服务 |
| ---: | --- |
| `9100` | Sample Pool |
| `9200` | Model Distributor |
| `9005` | 可选 Learner Monitor |

监控 API：`/monitor`、`/api/status`、`/api/metrics/catalog`、`/api/metrics/latest`、`/api/metrics`。

## 7. 刷新与清理开发容器

```bash
make dev-refresh
make dev-clean
```

`dev-refresh` 只在没有活动训练链时重建镜像与容器；`dev-clean` 删除 `learner-dev` 和 launcher
自己创建的转发，不删除未知宿主进程。两者都不会同步依赖或协议。

## License

[MIT License](LICENSE)
