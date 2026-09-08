# Component Metrics 与本地 Preview

## 组件出口

接口定义的唯一源为 `rl-contracts/proto/`：公共指标注册、目录、传输分别位于 `metrics/registry.proto`、`catalog.proto`、`transport.proto`；业务状态位于 `training/training.proto`。所有 RPC 使用显式 endpoint；`GetMetricCatalog` 返回当前组件的 ServiceInstanceIdentity 和完整字段定义，读取不需要 consumer、cursor 或 ACK。

| Producer | 默认端口 | 数值出口 |
| --- | --- | --- |
| Learner | 9006 | MetricEventService + LearnerStatusService/GetLearnerStatus |
| AIServer | 9002 | MetricEventService + AIServerTrainingStatusService/GetAIServerStatus |
| SamplePool | 9100 | SamplePoolConsumerService/GetStatus |
| ModelDistributor | 9200 | ModelDistributorService/GetModelDistributorStatus |

四者都提供 `rl.training.v1.MetricCatalogService/GetMetricCatalog`。状态目录中的 `status_method` 是完整 RPC 名，`status_field` 是响应的字段路径；MEAN 用 `status_count_field` 给出 sum 的分母，最大值等标量也可用 count 表示尚无测量。事件定义的 status_method 为空，数值来自该 source 的 MetricEventService。目录仅包含生产者显式声明的当前监控字段，不反射导出任意内部变量。

示例（在安装当前生成 Python bindings 的环境中）：

```python
import grpc
from proto.metrics import catalog_pb2, catalog_pb2_grpc
from proto.training import training_pb2, training_pb2_grpc

with grpc.insecure_channel("127.0.0.1:9006") as channel:
    catalog = catalog_pb2_grpc.MetricCatalogServiceStub(channel).GetMetricCatalog(
        catalog_pb2.GetMetricCatalogReq(), timeout=2)
    for entry in catalog.entries:
        print(entry.definition.metric_id, entry.definition.category,
              entry.definition.unit, entry.definition.denominator,
              entry.status_method, entry.status_field)
    status = training_pb2_grpc.LearnerStatusServiceStub(channel).GetLearnerStatus(
        training_pb2.LearnerStatusReq(), timeout=2)
```

字段 ID、display_name、category、unit、scope、value_type、aggregation、denominator、description 由事实 owner 声明。注册先于测量，不发送假的零值事件；同一 source lifecycle 的定义不可变。每条事件仍携带所用定义，可独立解释历史。

## 采集与预览

`src/preview/collector.py` 拥有 RPC 轮询、状态快照、速率、RawMetricBatchStore 和 LocalMetricProjector。它不导入 TrainingRuntime、Client Proto、Reward 或 trainer。TrainingRuntime 只提供自身已提交状态、事件生产和组装接线；模型 ACK、样本交付等训练业务调用仍由原 owner 负责。

Learner 的生产者 journal 为 `metric-events.sqlite3`；Preview 的消费存储为 `preview-events.sqlite3`。事件批次先完整落盘，再 ACK。最终批次 ACK 完成后，采集器进入 final 并停止该 source 的读取；Producer 随后正常退出不会触发重连。状态快照保留 producer 身份与观测时间，不冒充事件。速率由 Preview 声明原始计数字段，按同一 source lifecycle 两次 producer 状态观测时间求差；结果保留 `rates.source_intervals`，不能以浏览器刷新周期作为分母。

状态 RPC 与目录 RPC 的结果分别保留。目录读取失败不会抹掉已读取的状态，错误通过 `metric_event_views.catalog_errors` 暴露给查询接口与页面；未取得的目录不由 Preview 猜测生成。Produced 由 AIServer 的累计产出计算，Accepted / Acknowledged / Trained 分别由 Pool 的对应累计计数计算。每条速率独立维护基线；缺测、source lifecycle 变化、非正时间区间或计数回退只影响对应速率，原因在 `rates.unavailable_counters` 中保留。缺测后须重新取得两次有效观测，不能借用其他来源补值。

Preview 只从业务 bootstrap ACK 得到本轮 AIServer source，随后严格读取该 source。它不会按端口找一个最新实例或抢占其他 consumer。页面只是 HTTP 查询者，关闭或刷新页面不创建 ACK consumer，也不控制指标生产。

配置：

```yaml
metric_events:
  server_enabled: true
  server_port: 9006
  aiserver_relay_enabled: true

dashboard:
  enabled: true
  server_port: 9005
  backend: jsonl
  mean_window_ms: 60000
  time_bucket_ms: 5000
```

独立采集方接管时，在新的 source lifecycle 启动配置中关闭本地 Preview / AIServer relay，保留生产者事件服务。当前没有在线消费方交接、游标迁移、历史接力或双消费。

## Reward 和 Episode

AIServer 的 MazeReward 在初始化时注册 Total Reward、全部奖励分量及 Episode 聚合。每次确认 transition 只计算一次结果；GAE 使用 total，指标记录同一结果。周期窗口使用 `metrics.reward_interval_ms`（默认 5000）。奖励是否显示不影响 reward 或 GAE。

Reward 区间以单调时钟测量从首个 transition 到关闭窗口的实际时长，结束位置使用关闭时的 Unix 时间，开始位置为结束时间减去该时长。周期 flush 与 EndEpisode 尾段使用同一规则；系统时钟回拨不会使区间反转。事件观测时间仍保留关闭时的系统时钟值，journal 的时钟回退诊断与消费者对非法历史区间的校验继续生效。

每个 per-transition 分量的 count 都是区间内确认 transition 数，包括该分量为零的 transition。total 和各分量保持相同 count，MEAN 为 sum / count。周期区间带实际起止时间；短尾部保持原长度。

EndEpisode 先验证事实，把 Episode 记录和未输出的 Reward 尾部放入同一 RegisteredMetricRecord，经同一次本地 journal 接收后提交命令。已经输出的 Reward 增量不在 Episode 汇总中重复。该步骤不等待远端 ACK，也不改变 journal 的有界内存语义。

查询默认采用当前时钟最近一个已结束的 5 s 边界，向前取 60 s，按 producer 区间结束时间合并 sum/count。跨边界区间不按比例拆分；响应保留实际覆盖区间。最终快照保留已上报、尚未进入查询边界的尾部，查询时钟到达边界后纳入统计，无需生产者再写一条事件。无新数据时旧区间继续过期，空窗口返回缺失值。Episode Length / Return 和 Maze Any / All Success 使用完成 Episode 的原分母；PPO 保持最近 Train Update 统计。

## 字段查询和页面

- `GET /api/metrics/catalog`：全部注册字段及当前可用性、默认订阅、显示格式。
- `GET /api/metrics/query?field=<field_id>&window=current`：指定字段及当前 source 系列；多个 field 参数选择多个字段。
- 其余 History / Latest / Stream 端点沿现有 Preview 路由，原始事件服务独立可用。

Query 的 `series` 给出字段定义和来源身份；`records` 与 `latest` 中的 `metric_values`、`metric_statistics` 均按 `series_id` 对应。MEAN 的统计结果保留原始 `sum/count`，时间窗口结果另含 `window_start_unix_ms` / `window_end_unix_ms` 与实际观测的 `interval_start_unix_ms` / `interval_end_unix_ms`。实际区间表示所含统计段的起止范围，不保证中间没有缺口；来源的 gap / incomplete 状态仍独立保留。空窗口的值和统计结果均为 null，不提供虚构的零分母。

实际 HTTP 路径以 `tools/metrics_server.py` 路由为准。注册字段选择与展示分组分开：筛选类别不会成为注册白名单；字段没有当前测量仍可添加，详情显示 unavailable / no measurement / window empty 等状态。

默认 Loss 保留 Policy Loss / Value Loss；Entropy 在 PPO Stability；Reward 单独面板默认只选 Total Reward，加号列出全部注册分量；Episode 独立。Maze Success 是任务模板，默认 Any / All，已有 Agent Success 字段仍可选。`metrics_views.json` 和任务 `monitor_views.json` 只保存布局、订阅和显示格式。字段单位、类别、分母不由模板覆盖。浏览器已保存的选择不会被默认配置覆盖，操作支持中文 / English。

## 后续 Infra 边界

当前仍是三个容器：Learner 容器承载 Learner、Pool、Distributor 和 Preview；AIServer 与 Client 组成 Server Pod。后续基于 Kubernetes 的资源调度、启动/停止、Pod / Attempt 绑定、跨 Pod 存储和聚合由 Infra 接管。本批未实现 Infra 或 P2P。

Infra 可以维护指标协议和查询设施；训练/任务组件仍拥有测量及原始语义。MEAN 跨源合并需要相同语义的 sum/count；LATEST、模型 step 和生命周期计数不能任意相加。组件不会解析 K8s 对象或猜测 Pod 归属。
