#!/usr/bin/env python3
"""Serve metrics for the currently active Learner training process."""

import argparse
from collections import deque
import copy
import glob
import json
import math
import os
import re
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse


DASHBOARD_PATH = Path(__file__).with_name("metrics_dashboard.html")

_CATALOG_FIELD_KEYS = (
    "field_id",
    "label",
    "group",
    "dimension",
    "unit",
    "scope",
    "statistic",
    "value_kind",
    "owner_component",
    "aggregation_kind",
    "window_kind",
)
_REWARD_COMPONENT_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_KNOWN_REWARD_COMPONENTS = (
    "goal_reward",
    "timeout_penalty",
    "geodesic_progress",
    "first_visit_bonus",
    "wasted_action_penalty",
)


def _metric_definition(
    field_id,
    label,
    group,
    dimension,
    unit,
    scope,
    statistic,
    value_kind,
    path,
    scale=1.0,
):
    if field_id.startswith("learner."):
        owner_component = "rl-learner"
    elif field_id.startswith("server.episode.") or field_id.startswith(
        "server.reward."
    ) or field_id.startswith(
        "server.training."
    ) or field_id.startswith("server.task."):
        owner_component = "maze-task-adapter"
    elif field_id.startswith("server.") or field_id.startswith(
        "sample.flow.produced"
    ) or field_id.startswith("sample.flow.outbound") or field_id.startswith(
        "sample.flow.final_drop"
    ):
        owner_component = "rl-aiserver"
    elif field_id.startswith("sample.flow."):
        owner_component = "rl-sample-pool"
    else:
        owner_component = "learner-monitor"

    if value_kind == "counter":
        aggregation_kind = "sum"
        window_kind = "cumulative"
    elif statistic == "mean":
        aggregation_kind = "weighted_mean"
        window_kind = "rolling"
    elif statistic == "latest":
        aggregation_kind = "latest"
        window_kind = "instant"
    else:
        aggregation_kind = "not_mergeable"
        window_kind = "rolling"
    return {
        "descriptor": {
            "field_id": field_id,
            "label": label,
            "group": group,
            "dimension": dimension,
            "unit": unit,
            "scope": scope,
            "statistic": statistic,
            "value_kind": value_kind,
            "owner_component": owner_component,
            "aggregation_kind": aggregation_kind,
            "window_kind": window_kind,
        },
        "path": path,
        "scale": scale,
    }


_STATIC_METRIC_DEFINITIONS = (
    _metric_definition(
        "learner.model_step", "Model Step", "training_depth",
        "model_step", "step", "learner", "latest", "gauge",
        ("learner", "model_step"),
    ),
    _metric_definition(
        "learner.train_update.total", "Train Update", "training_depth",
        "train_update", "update", "learner", "total", "counter",
        ("learner", "train_updates"),
    ),
    _metric_definition(
        "learner.trained_samples.total", "Trained Samples",
        "training_depth", "sample_count", "samples", "learner", "total",
        "counter",
        ("learner", "trained_samples"),
    ),
    _metric_definition(
        "server.episode.max_steps.current", "Episode Max Steps",
        "training_depth", "environment_step", "step", "server",
        "latest", "gauge",
        ("actor", "metric_values", "server.episode.max_steps.current"),
    ),
    _metric_definition(
        "learner.loss.policy", "Policy Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "policy_loss", "mean"),
    ),
    _metric_definition(
        "learner.loss.value", "Value Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "value_loss", "mean"),
    ),
    _metric_definition(
        "learner.loss.total", "Total Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "total_loss", "mean"),
    ),
    _metric_definition(
        "server.episode.learning_return.mean", "Mean Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_agent_return"),
    ),
    _metric_definition(
        "server.training.episode.learning_return.mean",
        "Mean Training Agent Return", "episode_return", "episode_return",
        "reward", "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_agent_return"),
    ),
    _metric_definition(
        "server.training.episode.learning_return.latest_mean",
        "Latest Training Agent Return", "episode_return", "episode_return",
        "reward", "latest_training_environment_episode", "latest", "gauge",
        ("metric_event_views", "episodes", "latest", "values",
         "mean_agent_return"),
    ),
    _metric_definition(
        "server.training.episode.completed.total",
        "Completed Training Episodes", "training_depth", "episode_count",
        "episodes", "server", "total", "counter",
        ("metric_event_views", "episodes", "windows", "all", "raw",
         "environment_episode_count"),
    ),
    _metric_definition(
        "server.episode.learning_return.min", "Min Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "min", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "min_agent_return"),
    ),
    _metric_definition(
        "server.episode.learning_return.max", "Max Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "max", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "max_agent_return"),
    ),
    _metric_definition(
        "server.episode.success.agent_rate", "Agent Success",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "agent_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.agent_rate",
        "Training Agent Success", "episode_success", "percentage", "%",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "agent_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.any_rate",
        "Training Any Success", "episode_success", "percentage", "%",
        "training_environment_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "any_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.all_rate",
        "Training All Success", "episode_success", "percentage", "%",
        "training_environment_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "all_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.any_rate", "Any Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "any_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.all_rate", "All Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "all_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.path_ratio.mean", "Path Ratio",
        "episode_success", "ratio", "1", "successful_agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "path_ratio_mean"),
    ),
    _metric_definition(
        "server.episode.step.mean", "Episode Step",
        "episode_success", "environment_step", "step",
        "agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_episode_step"),
    ),
    _metric_definition(
        "server.episode.unique_cells.mean", "Unique Cells",
        "episode_success", "cell_count", "cells", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_unique_cells"),
    ),
    _metric_definition(
        "server.episode.blocked_move_rate", "Blocked Move Rate",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "blocked_move_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.path_ratio.mean",
        "Training Path Ratio", "episode_success", "ratio", "1",
        "successful_training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "path_ratio_mean"),
    ),
    _metric_definition(
        "server.training.episode.step.mean", "Training Episode Step",
        "episode_success", "environment_step", "step",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_episode_step"),
    ),
    _metric_definition(
        "server.training.episode.unique_cells.mean",
        "Training Unique Cells", "episode_success", "cell_count", "cells",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_unique_cells"),
    ),
    _metric_definition(
        "server.training.episode.blocked_move_rate",
        "Training Blocked Move Rate", "episode_success", "percentage", "%",
        "training_transition_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "blocked_move_rate"), scale=100.0,
    ),
    _metric_definition(
        "sample.throughput.produced_per_second", "Produced / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "produced_sps"),
    ),
    _metric_definition(
        "sample.throughput.accepted_per_second", "Accepted / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "accepted_sps"),
    ),
    _metric_definition(
        "sample.throughput.acknowledged_per_second", "Acknowledged / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "acked_sps"),
    ),
    _metric_definition(
        "sample.throughput.trained_per_second", "Trained / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "trained_sps"),
    ),
    _metric_definition(
        "sample.flow.produced.total", "Produced Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("actor", "produced"),
    ),
    _metric_definition(
        "sample.flow.accepted.total", "Accepted Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("sample_pool", "accepted"),
    ),
    _metric_definition(
        "sample.flow.acknowledged.total", "Acknowledged Samples",
        "sample_flow", "sample_count", "samples", "sample_chain", "total",
        "counter", ("sample_pool", "acked"),
    ),
    _metric_definition(
        "sample.flow.trained.total", "Trained Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("sample_pool", "trained"),
    ),
    _metric_definition(
        "sample.flow.invalid.total", "Invalid Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("sample_pool", "invalid"),
    ),
    _metric_definition(
        "sample.flow.shutdown_untrained.total", "Shutdown Untrained",
        "sample_flow", "sample_count", "samples", "sample_chain", "total",
        "counter", ("sample_pool", "shutdown_untrained"),
    ),
    _metric_definition(
        "sample.flow.ready.total", "Ready Transitions", "sample_flow",
        "sample_count", "transitions", "sample_chain", "latest", "gauge",
        ("sample_pool", "ready_transitions"),
    ),
    _metric_definition(
        "sample.flow.leased.total", "Leased Transitions", "sample_flow",
        "sample_count", "transitions", "sample_chain", "latest", "gauge",
        ("sample_pool", "leased_transitions"),
    ),
    _metric_definition(
        "sample.flow.outbound_pending.total", "Outbound Pending",
        "sample_flow", "sample_count", "samples", "server", "latest",
        "gauge", ("actor", "outbound_queue_transitions"),
    ),
    _metric_definition(
        "sample.flow.final_drop.total", "Final Drop", "sample_flow",
        "sample_count", "samples", "server", "total", "counter",
        ("actor", "final_drop"),
    ),
    _metric_definition(
        "server.latency.sample_send.mean_ms", "Sample Send Latency",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "push_rpc_mean_ms"),
    ),
    _metric_definition(
        "server.latency.inference.mean_ms", "Inference Latency Mean",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "inference_mean_ms"),
    ),
    _metric_definition(
        "server.latency.inference.max_ms", "Inference Latency Max",
        "latency", "duration", "ms", "server", "max", "gauge",
        ("actor", "inference_max_ms"),
    ),
    _metric_definition(
        "server.latency.update_rpc.mean_ms", "Update RPC Latency Mean",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "update_rpc_mean_ms"),
    ),
    _metric_definition(
        "server.latency.update_rpc.max_ms", "Update RPC Latency Max",
        "latency", "duration", "ms", "server", "max", "gauge",
        ("actor", "update_rpc_max_ms"),
    ),
    _metric_definition(
        "learner.ppo.entropy", "Policy Entropy", "ppo_stability",
        "entropy", "1", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "entropy", "mean"),
    ),
    _metric_definition(
        "learner.ppo.approx_kl", "Approx. KL", "ppo_stability",
        "divergence", "1", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "approx_kl", "mean"),
    ),
    _metric_definition(
        "learner.ppo.clip_fraction", "Clip Fraction", "ppo_stability",
        "percentage", "%", "train_update", "mean", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "clip_fraction", "mean"), scale=100.0,
    ),
    _metric_definition(
        "learner.ppo.gradient_norm", "Gradient Norm", "ppo_stability",
        "norm", "1", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "gradient_norm", "mean"),
    ),
    _metric_definition(
        "learner.ppo.max_importance_ratio", "Max Importance Ratio",
        "ppo_stability", "ratio", "1", "train_update", "max", "gauge",
        ("learner", "max_importance_ratio"),
    ),
    _metric_definition(
        "learner.ppo.policy_lag", "Policy Lag", "ppo_stability",
        "model_step", "step", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "policy_lag", "mean"),
    ),
    _metric_definition(
        "learner.value.prediction_mean", "Value Prediction Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "value_prediction", "mean"),
    ),
    _metric_definition(
        "learner.value.return_target_mean", "Return Target Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "return_target", "mean"),
    ),
    _metric_definition(
        "learner.value.explained_variance", "Explained Variance",
        "ppo_stability", "ratio", "1", "train_update", "latest",
        "gauge", ("learner", "explained_variance"),
    ),
)


def reward_component_field_id(name: str) -> str:
    if not isinstance(name, str) or not _REWARD_COMPONENT_NAME.fullmatch(name):
        raise ValueError("reward component name must be canonical snake_case")
    return f"server.reward.component.{name}.transition_mean"


def training_reward_component_field_id(name: str, statistic: str) -> str:
    if not isinstance(name, str) or not _REWARD_COMPONENT_NAME.fullmatch(name):
        raise ValueError("reward component name must be canonical snake_case")
    if statistic not in {
        "episode_mean",
        "transition_mean",
        "latest_episode_mean",
    }:
        raise ValueError("unsupported training reward component statistic")
    return f"server.training.reward.component.{name}.{statistic}"


def _reward_component_definition(name: str):
    field_id = reward_component_field_id(name)
    return _metric_definition(
        field_id,
        " ".join(part.capitalize() for part in name.split("_")),
        "reward_components",
        "transition_reward",
        "reward/transition",
        "server_transition_window",
        "mean",
        "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "reward_components", name, "transition_mean"),
    )


def _training_reward_component_definition(name: str, statistic: str):
    field_id = training_reward_component_field_id(name, statistic)
    label = " ".join(part.capitalize() for part in name.split("_"))
    if statistic == "episode_mean":
        return _metric_definition(
            field_id, f"{label} / Agent Episode", "reward_components",
            "episode_reward", "reward/agent episode",
            "training_agent_episode_window", "mean", "gauge",
            ("metric_event_views", "episodes", "windows", "100", "values",
             "reward_components", name, "episode_mean"),
        )
    if statistic == "latest_episode_mean":
        return _metric_definition(
            field_id, f"Latest {label} / Agent Episode", "reward_components",
            "episode_reward", "reward/agent episode",
            "latest_training_environment_episode", "latest", "gauge",
            ("metric_event_views", "episodes", "latest", "values",
             "reward_components", name, "episode_mean"),
        )
    return _metric_definition(
        field_id, f"{label} / Transition", "reward_components",
        "transition_reward", "reward/transition",
        "training_transition_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "reward_components", name, "transition_mean"),
    )


def _nested(document, path):
    value = document
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


_EVENT_WINDOWS = {"25", "100", "5s", "1m", "1h", "24h", "all"}

# 粗粒度历史层支持的时间范围；秒级实时层仅保留 max_records 条，
# 超出该窗口的 6h/24h/all 视图由按桶降采样的历史层提供。
_HISTORY_RANGE_SECONDS = {
    "6h": 6.0 * 3600.0,
    "24h": 24.0 * 3600.0,
    "all": None,
}
# 历史响应的目标点数上限：超过该点数时按均匀步长做二级抽取，
# 使响应体与前端解析开销不随保留时长线性膨胀。
_HISTORY_DEFAULT_MAX_POINTS = 1500
_HISTORY_MAX_POINTS_LIMIT = 6000


def _project_metric_value(record, definition, event_window="100"):
    if event_window not in _EVENT_WINDOWS:
        raise ValueError("unsupported metric event window")
    path = definition["path"]
    selected_path = path
    if (
        len(path) >= 5
        and path[:3] == ("metric_event_views", "episodes", "windows")
        and path[3] == "100"
    ):
        selected_path = (*path[:3], event_window, *path[4:])
    value = _finite_number(_nested(record, selected_path))
    return None if value is None else value * definition["scale"]


def project_metric_values(record, definitions, event_window="100"):
    projected = copy.deepcopy(record)
    projected["metric_values"] = {
        definition["descriptor"]["field_id"]: _project_metric_value(
            record, definition, event_window
        )
        for definition in definitions
    }
    statistics = _nested(record, ("actor", "metric_statistics"))
    projected["metric_statistics"] = (
        copy.deepcopy(statistics) if isinstance(statistics, dict) else {}
    )
    descriptors = _nested(record, ("actor", "metric_descriptors"))
    projected["metric_descriptors"] = (
        copy.deepcopy(descriptors) if isinstance(descriptors, dict) else {}
    )
    return projected


class MetricsFileReader:
    def __init__(
        self,
        metrics_dir: str,
        *,
        metrics_source_id: str = "",
        runtime_mode: str = "",
        service_instance_id: str = "",
        started_at: float | None = None,
        max_records: int = 4096,
        read_chunk_bytes: int = 256 * 1024,
        max_pending_bytes: int = 1024 * 1024,
        tail_interval_seconds: float = 0.05,
        history_bucket_seconds: float = 15.0,
        history_max_records: int = 6000,
    ):
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if read_chunk_bytes <= 0 or max_pending_bytes <= 0:
            raise ValueError("metrics read bounds must be positive")
        if (
            not math.isfinite(tail_interval_seconds)
            or tail_interval_seconds <= 0
        ):
            raise ValueError("metrics tail interval must be positive and finite")
        if history_max_records <= 0:
            raise ValueError("history_max_records must be positive")
        if (
            not math.isfinite(history_bucket_seconds)
            or history_bucket_seconds <= 0
        ):
            raise ValueError("history bucket seconds must be positive and finite")
        self._metrics_dir = os.path.abspath(metrics_dir)
        self._metrics_source_id = metrics_source_id or (
            f"local-training-{uuid.uuid4().hex}"
        )
        self._runtime_mode = runtime_mode
        self._service_instance_id = service_instance_id or (
            f"learner-metrics-{uuid.uuid4().hex}"
        )
        self._started_at = (
            time.time() if started_at is None else float(started_at)
        )
        # 历史层摄入在 refresh() 持锁路径内会间接调用 metric_definitions()，
        # 需要可重入锁避免同线程二次获取造成死锁。
        self._lock = threading.RLock()
        self._max_records = int(max_records)
        self._read_chunk_bytes = int(read_chunk_bytes)
        self._max_pending_bytes = int(max_pending_bytes)
        self._tail_interval_seconds = float(tail_interval_seconds)
        self._tail_stop = threading.Event()
        self._tail_thread = None
        self._tail_last_refresh_timestamp = None
        self._tail_refresh_count = 0
        self._tail_backlog_remaining = False
        self._tail_error = None
        self._records = deque(maxlen=self._max_records)
        self._history_bucket_seconds = float(history_bucket_seconds)
        # 粗粒度历史层：桶内末值瘦身记录，重启后随磁盘回读自动重建。
        self._history = deque(maxlen=int(history_max_records))
        self._history_pending_bucket = None
        self._history_pending_record = None
        self._history_definitions = None
        self._history_definitions_at = 0.0
        self._total_record_count = 0
        self._files = {}
        self._corrupt_lines = 0
        self._last_scan_time = 0.0
        os.makedirs(self._metrics_dir, exist_ok=True)
        print(f"[MetricsServer] 监控目录: {self._metrics_dir}")

    def start(self):
        if self._tail_thread is not None:
            raise RuntimeError("metrics background tail is already started")
        self._tail_stop.clear()
        self._tail_thread = threading.Thread(
            target=self._tail_loop,
            name="learner-metrics-tail",
            daemon=True,
        )
        self._tail_thread.start()
        print(
            "[MetricsServer] continuous background tail: "
            f"interval={self._tail_interval_seconds * 1000.0:g}ms "
            f"chunk={self._read_chunk_bytes} bytes"
        )

    def close(self):
        thread = self._tail_thread
        if thread is None:
            return
        self._tail_stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            print("[MetricsServer] background tail did not stop within 2 seconds")
        self._tail_thread = None

    def _tail_loop(self):
        while not self._tail_stop.is_set():
            try:
                backlog_remaining = self.refresh()
            except Exception as exc:
                with self._lock:
                    self._tail_error = f"{type(exc).__name__}: {exc}"
                print(f"[MetricsServer] background tail failed: {self._tail_error}")
                return
            with self._lock:
                self._tail_last_refresh_timestamp = time.time()
                self._tail_refresh_count += 1
                self._tail_backlog_remaining = backlog_remaining
            if not backlog_remaining:
                self._tail_stop.wait(self._tail_interval_seconds)

    def refresh(self):
        backlog_remaining = False
        with self._lock:
            now = time.monotonic()
            if now - self._last_scan_time >= 0.5:
                self._last_scan_time = now
                for path in glob.glob(
                    os.path.join(self._metrics_dir, "metrics_*.jsonl")
                ):
                    self._files.setdefault(
                        path,
                        {
                            "offset": 0,
                            "pending": b"",
                            "discarding_oversize_line": False,
                            "corrupt": 0,
                        },
                    )
            for path, state in list(self._files.items()):
                backlog_remaining = (
                    self._read_file(path, state) or backlog_remaining
                )
        return backlog_remaining

    def _read_file(self, path: str, state: dict):
        try:
            file_size = os.path.getsize(path)
            if file_size < state["offset"]:
                state["offset"] = 0
                state["pending"] = b""
                state["discarding_oversize_line"] = False
            if file_size == state["offset"]:
                return False
            with open(path, "rb") as stream:
                stream.seek(state["offset"])
                chunk = stream.read(self._read_chunk_bytes)
                state["offset"] = stream.tell()
            data = chunk
            if state.get("discarding_oversize_line"):
                newline = data.find(b"\n")
                if newline < 0:
                    return state["offset"] < file_size
                data = data[newline + 1 :]
                state["discarding_oversize_line"] = False
            data = state["pending"] + data
            lines = data.split(b"\n")
            state["pending"] = lines.pop()
            if len(state["pending"]) > self._max_pending_bytes:
                state["pending"] = b""
                state["discarding_oversize_line"] = True
                state["corrupt"] += 1
                self._corrupt_lines += 1
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    record_source_id = record.get("metrics_source_id")
                    if record_source_id != self._metrics_source_id:
                        continue
                    sequence = record.get("sequence")
                    if (
                        isinstance(sequence, bool)
                        or not isinstance(sequence, int)
                        or sequence <= 0
                    ):
                        state["corrupt"] += 1
                        self._corrupt_lines += 1
                        continue
                    self._records.append(record)
                    self._total_record_count += 1
                    self._ingest_history(record)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    state["corrupt"] += 1
                    self._corrupt_lines += 1
            return state["offset"] < file_size
        except OSError as exc:
            state["error"] = str(exc)
            return False

    def query(self, after_sequence: int = 0, limit: int = 0):
        with self._lock:
            records = [
                record
                for record in self._records
                if int(record["sequence"]) > after_sequence
            ]
            if limit > 0:
                records = records[:limit]
            return records

    def _ingest_history(self, record):
        # 按固定秒桶归并，桶内只保留末值原始记录；跨桶时再投影落盘，
        # 避免每条秒级记录都触发一次全量字段投影。
        timestamp = _finite_number(record.get("timestamp"))
        if timestamp is None:
            return
        bucket = int(timestamp // self._history_bucket_seconds)
        if self._history_pending_bucket is None:
            self._history_pending_bucket = bucket
            self._history_pending_record = record
            return
        if bucket == self._history_pending_bucket:
            self._history_pending_record = record
            return
        if bucket < self._history_pending_bucket:
            # 多文件回读可能出现轻微乱序，丢弃过期桶保持历史层时间单调。
            return
        self._flush_history()
        self._history_pending_bucket = bucket
        self._history_pending_record = record

    def _definitions_for_history(self):
        now = time.monotonic()
        if (
            self._history_definitions is None
            or now - self._history_definitions_at >= 60.0
        ):
            self._history_definitions = self.metric_definitions()
            self._history_definitions_at = now
        return self._history_definitions

    def _project_history_record(self, record):
        # 历史层固定以 100 局事件窗口投影，与实时层默认视图保持一致；
        # 瘦身记录仅保留绘图所需字段，控制常驻内存。
        return {
            "sequence": record.get("sequence"),
            "timestamp": record.get("timestamp"),
            "metrics_source_id": record.get("metrics_source_id"),
            "metric_values": {
                definition["descriptor"]["field_id"]: _project_metric_value(
                    record, definition, "100"
                )
                for definition in self._definitions_for_history()
            },
        }

    def _flush_history(self):
        if self._history_pending_record is None:
            return
        self._history.append(
            self._project_history_record(self._history_pending_record)
        )

    @staticmethod
    def _select_history_fields(record, fields):
        # 按调用方声明的字段白名单裁剪响应；面板通常只消费全部字段的少数几个，
        # 裁剪后响应体与解析开销显著下降。
        if fields is None:
            return record
        values = record.get("metric_values") or {}
        return {
            "sequence": record.get("sequence"),
            "timestamp": record.get("timestamp"),
            "metrics_source_id": record.get("metrics_source_id"),
            "metric_values": {
                field_id: values[field_id]
                for field_id in fields
                if field_id in values
            },
        }

    @staticmethod
    def _thin_history(records, max_points):
        # 均匀步长抽取并强制保留末点，避免大范围响应点数随保留时长线性膨胀。
        if max_points <= 0 or len(records) <= max_points:
            return records
        stride = -(-len(records) // max_points)
        thinned = records[::stride]
        if thinned[-1] is not records[-1]:
            thinned.append(records[-1])
        return thinned

    def history(
        self,
        range_key: str = "6h",
        *,
        fields=None,
        after_sequence: int = 0,
        max_points: int = _HISTORY_DEFAULT_MAX_POINTS,
    ):
        if range_key not in _HISTORY_RANGE_SECONDS:
            raise ValueError("unsupported history range")
        if after_sequence < 0:
            raise ValueError("history cursor must not be negative")
        if max_points < 0 or max_points > _HISTORY_MAX_POINTS_LIMIT:
            raise ValueError("history max_points is out of range")
        seconds = _HISTORY_RANGE_SECONDS[range_key]
        with self._lock:
            records = list(self._history)
            if self._history_pending_record is not None:
                records.append(
                    self._project_history_record(self._history_pending_record)
                )
        if seconds is not None:
            cutoff = time.time() - seconds
            records = [
                record
                for record in records
                if (_finite_number(record.get("timestamp")) or 0.0) >= cutoff
            ]
        if after_sequence > 0:
            records = [
                record
                for record in records
                if _finite_number(record.get("sequence")) is not None
                and int(record["sequence"]) > after_sequence
            ]
        records = self._thin_history(records, max_points)
        return [
            self._select_history_fields(record, fields) for record in records
        ]

    def latest(self):
        with self._lock:
            return self._records[-1] if self._records else {}

    def metric_definitions(self):
        with self._lock:
            reward_names = set(_KNOWN_REWARD_COMPONENTS)
            for record in self._records:
                for component_kind in (
                    "reward_components",
                    "transition_reward_components",
                    "latest_reward_components",
                ):
                    components = _nested(
                        record, ("actor", "episodes", component_kind)
                    )
                    if not isinstance(components, dict):
                        continue
                    reward_names.update(
                        name
                        for name in components
                        if isinstance(name, str)
                        and _REWARD_COMPONENT_NAME.fullmatch(name)
                    )
        definitions = list(_STATIC_METRIC_DEFINITIONS)
        definitions.extend(
            _reward_component_definition(name)
            for name in sorted(reward_names)
        )
        definitions.extend(
            _training_reward_component_definition(name, statistic)
            for name in sorted(reward_names)
            for statistic in (
                "episode_mean",
                "latest_episode_mean",
                "transition_mean",
            )
        )
        return definitions

    def catalog(self):
        definitions = self.metric_definitions()
        return {
            "schema_version": 1,
            "catalog_version": 1,
            "fields": [
                {
                    key: definition["descriptor"][key]
                    for key in _CATALOG_FIELD_KEYS
                }
                for definition in definitions
            ],
        }

    def status(self):
        latest = self.latest()
        with self._lock:
            timestamp = _finite_number(latest.get("timestamp"))
            interval_ms = _finite_number(latest.get("interval_ms"))
            configured_interval_ms = _finite_number(
                latest.get("configured_poll_interval_ms")
            )
            interval_basis_ms = (
                interval_ms
                if interval_ms is not None and interval_ms > 0.0
                else configured_interval_ms
                if configured_interval_ms is not None
                and configured_interval_ms > 0.0
                else None
            )
            stale_after = (
                None
                if interval_basis_ms is None
                else max(5.0, 3.0 * interval_basis_ms / 1000.0)
            )
            age_seconds = (
                None if timestamp is None else time.time() - timestamp
            )
            stale = (
                None
                if age_seconds is None or stale_after is None
                else age_seconds > stale_after
            )
            return {
                "schema_version": 1,
                "service": "learner-metrics",
                "stream": "current",
                "service_instance_id": self._service_instance_id,
                "metrics_source_id": self._metrics_source_id,
                "started_at": self._started_at,
                "metrics_dir": self._metrics_dir,
                "mode": latest.get("mode") or self._runtime_mode,
                "record_count": len(self._records),
                "total_record_count": self._total_record_count,
                "retained_record_limit": self._max_records,
                "history_record_count": len(self._history),
                "history_oldest_timestamp": (
                    self._history[0].get("timestamp") if self._history else None
                ),
                "history_bucket_seconds": self._history_bucket_seconds,
                "latest_sequence": latest.get("sequence"),
                "latest_timestamp": timestamp,
                "latest_interval_ms": interval_ms,
                "configured_poll_interval_ms": configured_interval_ms,
                "age_seconds": age_seconds,
                "stale_after_seconds": stale_after,
                "stale": stale,
                "stale_status": (
                    "unavailable" if stale is None else "stale" if stale else "fresh"
                ),
                "corrupt_line_count": self._corrupt_lines,
                "file_count": len(self._files),
                "tail_mode": "continuous_background",
                "tail_running": (
                    self._tail_thread is not None
                    and self._tail_thread.is_alive()
                ),
                "tail_interval_ms": self._tail_interval_seconds * 1000.0,
                "tail_last_refresh_timestamp": (
                    self._tail_last_refresh_timestamp
                ),
                "tail_refresh_count": self._tail_refresh_count,
                "tail_backlog_remaining": self._tail_backlog_remaining,
                "tail_error": self._tail_error,
                "file_errors": {
                    os.path.basename(path): state["error"]
                    for path, state in self._files.items()
                    if state.get("error")
                },
            }

    def summary(self):
        latest = self.latest()
        sample_pool = latest.get("sample_pool", {})
        rates = latest.get("rates", {})
        chain = latest.get("chain", {})
        return {
            "mode": latest.get("mode"),
            "sequence": latest.get("sequence"),
            "consumed": sample_pool.get("acked"),
            "consumer_sps": rates.get("trained_sps"),
            "consumer_rate_available": rates.get("available"),
            "queue_size": sample_pool.get("ready_transitions"),
            "chain_ready": chain.get("ready"),
        }


metrics_reader = None


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._redirect("/monitor")
        elif path == "/api":
            self._json_response(
                {
                    "schema_version": 1,
                    "service": "learner-metrics",
                    "stream": "current",
                    "endpoints": [
                        "/api",
                        "/api/metrics",
                        "/api/metrics/catalog",
                        "/api/metrics/history",
                        "/api/metrics/latest",
                        "/api/metrics/summary",
                        "/api/status",
                        "/monitor",
                    ],
                }
            )
        elif path == "/monitor":
            try:
                body = DASHBOARD_PATH.read_bytes()
            except OSError:
                self.send_error(500, "Metrics dashboard is unavailable")
                return
            self._response(
                body, "text/html; charset=utf-8", status=200
            )
        elif path == "/api/metrics":
            try:
                after_sequence = int(
                    params.get(
                        "after_sequence", params.get("since", ["0"])
                    )[0]
                )
                limit = int(params.get("limit", ["0"])[0])
                if after_sequence < 0 or limit < 0:
                    raise ValueError
                event_window = params.get("window", ["100"])[0]
                if event_window not in _EVENT_WINDOWS:
                    raise ValueError
            except ValueError:
                self._json_response(
                    {"schema_version": 1, "error": "invalid query cursor"},
                    status=400,
                )
                return
            records = metrics_reader.query(after_sequence, limit)
            definitions = metrics_reader.metric_definitions()
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "event_window": event_window,
                    "records": [
                        project_metric_values(
                            record, definitions, event_window
                        )
                        for record in records
                    ],
                    "total": len(records),
                }
            )
        elif path == "/api/metrics/history":
            range_key = params.get("range", ["6h"])[0]
            try:
                after_sequence = int(params.get("after_sequence", ["0"])[0])
                max_points = int(
                    params.get(
                        "max_points", [str(_HISTORY_DEFAULT_MAX_POINTS)]
                    )[0]
                )
                raw_fields = params.get("fields", [""])[0].strip()
                fields = (
                    tuple(
                        field_id
                        for field_id in raw_fields.split(",")
                        if field_id
                    )
                    if raw_fields
                    else None
                )
                records = metrics_reader.history(
                    range_key,
                    fields=fields,
                    after_sequence=after_sequence,
                    max_points=max_points,
                )
            except ValueError:
                self._json_response(
                    {
                        "schema_version": 1,
                        "error": "invalid history query",
                    },
                    status=400,
                )
                return
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "range": range_key,
                    "after_sequence": after_sequence,
                    "records": records,
                    "total": len(records),
                }
            )
        elif path == "/api/metrics/catalog":
            self._json_response(metrics_reader.catalog())
        elif path == "/api/metrics/latest":
            latest = metrics_reader.latest()
            definitions = metrics_reader.metric_definitions()
            event_window = params.get("window", ["100"])[0]
            if event_window not in _EVENT_WINDOWS:
                self._json_response(
                    {"schema_version": 1, "error": "invalid event window"},
                    status=400,
                )
                return
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "event_window": event_window,
                    "record": (
                        project_metric_values(
                            latest, definitions, event_window
                        )
                        if latest
                        else {}
                    ),
                }
            )
        elif path == "/api/metrics/summary":
            self._json_response(metrics_reader.summary())
        elif path == "/api/status":
            self._json_response(metrics_reader.status())
        else:
            self.send_error(404, "Not Found")

    def _redirect(self, location: str):
        try:
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json_response(self, document, status=200):
        try:
            body = json.dumps(
                document, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self._response(
                body, "application/json; charset=utf-8", status=status
            )
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _response(self, body: bytes, content_type: str, status: int):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        if "200" not in str(args):
            super().log_message(format_string, *args)


def main():
    global metrics_reader
    parser = argparse.ArgumentParser(
        description="Serve the current training metrics API"
    )
    parser.add_argument("--dir", "-d", required=True)
    parser.add_argument("--port", "-p", type=int, default=9005)
    parser.add_argument(
        "--source-id",
        default=os.environ.get("RL_METRICS_SOURCE_ID", ""),
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("RL_METRICS_MODE", ""),
    )
    args = parser.parse_args()
    metrics_reader = MetricsFileReader(
        args.dir,
        metrics_source_id=args.source_id,
        runtime_mode=args.mode,
    )

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MetricsHTTPHandler)

    def handle_stop_signal(_signal_number, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_stop_signal)
    try:
        metrics_reader.start()
        print(f"[MetricsServer] container listener: 0.0.0.0:{args.port}")
        public_url = os.environ.get("RL_METRICS_PUBLIC_URL", "").strip()
        if public_url:
            print(f"[MetricsServer] host browser: {public_url}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        metrics_reader.close()


if __name__ == "__main__":
    main()
