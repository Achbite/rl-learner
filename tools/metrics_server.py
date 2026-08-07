#!/usr/bin/env python3
"""Serve metrics for the currently active Learner training process."""

import argparse
import copy
import glob
import json
import math
import os
import re
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
    "schema_identity",
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
    *paths,
    scale=1.0,
):
    if field_id.startswith("learner."):
        owner_component = "rl-learner"
    elif field_id.startswith("server.episode.") or field_id.startswith(
        "server.reward."
    ) or field_id.startswith("server.task.") or field_id.startswith(
        "server.evaluation."
    ):
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
            "schema_identity": {
                "schema_id": "maze.metrics.v1",
                "schema_version": 1,
                "canonical_digest": {
                    "algorithm": "sha256",
                    "hex": (
                        "2ab9434f8c80b4651b0f51f65f3e94e29b0dd0a55803ed4"
                        "c7d888f44f4604ce4"
                    ),
                },
            },
        },
        "paths": paths,
        "scale": scale,
    }


_STATIC_METRIC_DEFINITIONS = (
    _metric_definition(
        "learner.model_step.v1", "Model Step", "training_depth",
        "model_step", "step", "learner", "latest", "gauge",
        ("learner", "model_version"), ("model", "latest_version"),
        ("model_version",),
    ),
    _metric_definition(
        "learner.train_update.total.v1", "Train Update", "training_depth",
        "train_update", "update", "learner", "total", "counter",
        ("learner", "run_train_updates"), ("learner", "train_updates"),
        ("train_step",),
    ),
    _metric_definition(
        "learner.trained_samples.total.v1", "Trained Samples",
        "training_depth", "sample_count", "samples", "learner", "total",
        "counter", ("learner", "run_trained_samples"),
        ("learner", "trained_samples"), ("trained_samples",),
    ),
    _metric_definition(
        "server.environment_step.v1", "Environment Step", "training_depth",
        "environment_step", "step", "server", "latest", "gauge",
        ("actor", "metric_values", "server.environment_step.v1"),
        ("actor", "environment_step"),
    ),
    _metric_definition(
        "server.task.curriculum.multiplier.v1", "Curriculum",
        "training_depth", "curriculum_multiplier", "×", "server",
        "latest", "gauge",
        ("actor", "metric_values",
         "server.task.curriculum.multiplier.v1"),
    ),
    _metric_definition(
        "server.task.stage_produced_samples.v1", "Stage Samples",
        "training_depth", "sample_count", "samples", "server_stage",
        "total", "counter", ("actor", "metric_values",
                               "server.task.stage_produced_samples.v1"),
    ),
    _metric_definition(
        "server.task.stage_sample_budget.v1", "Stage Sample Budget",
        "training_depth", "sample_count", "samples", "server_stage",
        "latest", "gauge", ("actor", "metric_values",
                              "server.task.stage_sample_budget.v1"),
    ),
    _metric_definition(
        "server.task.next_evaluation_trained_samples.v1", "Next Evaluation",
        "training_depth", "sample_count", "samples", "server_task",
        "latest", "gauge", ("actor", "metric_values",
                              "server.task.next_evaluation_trained_samples.v1"),
    ),
    _metric_definition(
        "server.evaluation.episode_in_round.v1", "Evaluation Episode",
        "training_depth", "episode_count", "episode", "evaluation_round",
        "latest", "gauge", ("actor", "metric_values",
                              "server.evaluation.episode_in_round.v1"),
    ),
    _metric_definition(
        "server.episode.max_steps.current.v1", "Episode Max Steps",
        "training_depth", "environment_step", "step", "server",
        "latest", "gauge",
        ("actor", "metric_values", "server.episode.max_steps.current.v1"),
    ),
    _metric_definition(
        "learner.loss.policy.v1", "Policy Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge", ("learner", "policy_loss"),
        ("policy_loss",),
    ),
    _metric_definition(
        "learner.loss.value.v1", "Value Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge", ("learner", "value_loss"),
        ("value_loss",),
    ),
    _metric_definition(
        "learner.loss.total.v1", "Total Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge", ("learner", "total_loss"),
        ("total_loss",),
    ),
    _metric_definition(
        "server.episode.learning_return.mean.v1", "Mean Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "mean", "gauge", ("actor", "episodes", "mean_agent_return"),
        ("mean_episode_reward",),
    ),
    _metric_definition(
        "server.episode.learning_return.min.v1", "Min Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "min", "gauge", ("actor", "episodes", "min_agent_return"),
    ),
    _metric_definition(
        "server.episode.learning_return.max.v1", "Max Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "max", "gauge", ("actor", "episodes", "max_agent_return"),
    ),
    _metric_definition(
        "server.episode.success.agent_rate.v1", "Agent Success",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge", ("actor", "episodes", "agent_success_rate"),
        ("pass_rate",), scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.any_rate.v1", "Any Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge", ("actor", "episodes", "any_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.all_rate.v1", "All Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge", ("actor", "episodes", "all_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.path_ratio.mean.v1", "Path Ratio",
        "episode_success", "ratio", "1", "successful_agent_episode_window",
        "mean", "gauge",
        ("actor", "metric_values", "server.episode.path_ratio.mean.v1"),
    ),
    _metric_definition(
        "server.episode.step.mean.v1", "Episode Step",
        "episode_success", "environment_step", "step",
        "agent_episode_window", "mean", "gauge",
        ("actor", "metric_values", "server.episode.step.mean.v1"),
    ),
    _metric_definition(
        "server.episode.unique_cells.mean.v1", "Unique Cells",
        "episode_success", "cell_count", "cells", "agent_episode_window",
        "mean", "gauge",
        ("actor", "metric_values", "server.episode.unique_cells.mean.v1"),
    ),
    _metric_definition(
        "server.episode.blocked_move_rate.v1", "Blocked Move Rate",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge",
        ("actor", "metric_values", "server.episode.blocked_move_rate.v1"),
        scale=100.0,
    ),
    _metric_definition(
        "server.evaluation.argmax_round_1_success_rate.v1",
        "Argmax Round 1 Success", "episode_success", "percentage", "%",
        "evaluation_agent_episodes", "mean", "gauge",
        ("actor", "metric_values",
         "server.evaluation.argmax_round_1_success_rate.v1"), scale=100.0,
    ),
    _metric_definition(
        "server.evaluation.argmax_round_2_success_rate.v1",
        "Argmax Round 2 Success", "episode_success", "percentage", "%",
        "evaluation_agent_episodes", "mean", "gauge",
        ("actor", "metric_values",
         "server.evaluation.argmax_round_2_success_rate.v1"), scale=100.0,
    ),
    _metric_definition(
        "server.evaluation.stochastic_success_rate.v1",
        "Stochastic Success", "episode_success", "percentage", "%",
        "diagnostic_agent_episodes", "mean", "gauge",
        ("actor", "metric_values",
         "server.evaluation.stochastic_success_rate.v1"), scale=100.0,
    ),
    _metric_definition(
        "server.evaluation.path_ratio_median.v1", "Argmax Path Ratio Median",
        "episode_success", "ratio", "1", "successful_evaluation_episodes",
        "median", "gauge", ("actor", "metric_values",
                              "server.evaluation.path_ratio_median.v1"),
    ),
    _metric_definition(
        "server.evaluation.path_ratio_p95.v1", "Argmax Path Ratio p95",
        "episode_success", "ratio", "1", "successful_evaluation_episodes",
        "p95", "gauge", ("actor", "metric_values",
                           "server.evaluation.path_ratio_p95.v1"),
    ),
    _metric_definition(
        "sample.throughput.produced_per_second.v1", "Produced / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "produced_sps"),
    ),
    _metric_definition(
        "sample.throughput.accepted_per_second.v1", "Accepted / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "accepted_sps"),
    ),
    _metric_definition(
        "sample.throughput.acknowledged_per_second.v1", "Acknowledged / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "acked_sps"),
    ),
    _metric_definition(
        "sample.throughput.trained_per_second.v1", "Trained / sec",
        "sample_throughput", "sample_rate", "samples/s", "sample_chain",
        "rate", "gauge", ("rates", "trained_sps"),
    ),
    _metric_definition(
        "sample.flow.produced.total.v1", "Produced Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("actor", "produced"),
    ),
    _metric_definition(
        "sample.flow.accepted.total.v1", "Accepted Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("distributor", "accepted"),
    ),
    _metric_definition(
        "sample.flow.acknowledged.total.v1", "Acknowledged Samples",
        "sample_flow", "sample_count", "samples", "sample_chain", "total",
        "counter", ("distributor", "acked"),
    ),
    _metric_definition(
        "sample.flow.trained.total.v1", "Trained Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("distributor", "trained"), ("trained_samples",),
    ),
    _metric_definition(
        "sample.flow.invalid.total.v1", "Invalid Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("distributor", "invalid"),
    ),
    _metric_definition(
        "sample.flow.stale.total.v1", "Stale Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "total", "counter",
        ("distributor", "stale"),
    ),
    _metric_definition(
        "sample.flow.shutdown_untrained.total.v1", "Shutdown Untrained",
        "sample_flow", "sample_count", "samples", "sample_chain", "total",
        "counter", ("distributor", "shutdown_untrained"),
    ),
    _metric_definition(
        "sample.flow.ready.total.v1", "Ready Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "latest", "gauge",
        ("distributor", "ready_samples"),
    ),
    _metric_definition(
        "sample.flow.leased.total.v1", "Leased Samples", "sample_flow",
        "sample_count", "samples", "sample_chain", "latest", "gauge",
        ("distributor", "leased_samples"),
    ),
    _metric_definition(
        "sample.flow.outbound_pending.total.v1", "Outbound Pending",
        "sample_flow", "sample_count", "samples", "server", "latest",
        "gauge", ("actor", "outbound_pending"),
    ),
    _metric_definition(
        "sample.flow.final_drop.total.v1", "Final Drop", "sample_flow",
        "sample_count", "samples", "server", "total", "counter",
        ("actor", "final_drop"),
    ),
    _metric_definition(
        "server.latency.sample_send.mean_ms.v1", "Sample Send Latency",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "push_rpc_mean_ms"),
    ),
    _metric_definition(
        "server.latency.inference.mean_ms.v1", "Inference Latency Mean",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "inference_mean_ms"),
    ),
    _metric_definition(
        "server.latency.inference.max_ms.v1", "Inference Latency Max",
        "latency", "duration", "ms", "server", "max", "gauge",
        ("actor", "inference_max_ms"),
    ),
    _metric_definition(
        "server.latency.update_rpc.mean_ms.v1", "Update RPC Latency Mean",
        "latency", "duration", "ms", "server", "mean", "gauge",
        ("actor", "update_rpc_mean_ms"),
    ),
    _metric_definition(
        "server.latency.update_rpc.max_ms.v1", "Update RPC Latency Max",
        "latency", "duration", "ms", "server", "max", "gauge",
        ("actor", "update_rpc_max_ms"),
    ),
    _metric_definition(
        "learner.ppo.entropy.v1", "Policy Entropy", "ppo_stability",
        "entropy", "1", "train_update", "latest", "gauge",
        ("learner", "entropy"), ("entropy",),
    ),
    _metric_definition(
        "learner.ppo.approx_kl.v1", "Approx. KL", "ppo_stability",
        "divergence", "1", "train_update", "latest", "gauge",
        ("learner", "approx_kl"), ("approx_kl",),
    ),
    _metric_definition(
        "learner.ppo.clip_fraction.v1", "Clip Fraction", "ppo_stability",
        "percentage", "%", "train_update", "mean", "gauge",
        ("learner", "clip_fraction"), ("clip_fraction",), scale=100.0,
    ),
    _metric_definition(
        "learner.ppo.gradient_norm.v1", "Gradient Norm", "ppo_stability",
        "norm", "1", "train_update", "latest", "gauge",
        ("learner", "gradient_norm"), ("gradient_norm",),
    ),
    _metric_definition(
        "learner.ppo.max_importance_ratio.v1", "Max Importance Ratio",
        "ppo_stability", "ratio", "1", "train_update", "max", "gauge",
        ("learner", "max_importance_ratio"),
        ("max_importance_ratio",),
    ),
    _metric_definition(
        "learner.ppo.policy_lag.v1", "Policy Lag", "ppo_stability",
        "model_step", "version", "train_update", "latest", "gauge",
        ("learner", "policy_lag"), ("policy_lag",),
    ),
    _metric_definition(
        "learner.value.prediction_mean.v1", "Value Prediction Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge", ("learner", "value_pred_mean"),
        ("value_pred_mean",),
    ),
    _metric_definition(
        "learner.value.return_target_mean.v1", "Return Target Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge", ("learner", "return_target_mean"),
        ("return_target_mean",),
    ),
    _metric_definition(
        "learner.value.explained_variance.v1", "Explained Variance",
        "ppo_stability", "ratio", "1", "train_update", "latest",
        "gauge", ("learner", "explained_variance"),
        ("explained_variance",),
    ),
)


def reward_component_field_id(name: str) -> str:
    if not isinstance(name, str) or not _REWARD_COMPONENT_NAME.fullmatch(name):
        raise ValueError("reward component name must be canonical snake_case")
    return f"server.reward.component.{name}.transition_mean.v1"


def _reward_component_definition(name: str):
    return _metric_definition(
        reward_component_field_id(name),
        " ".join(part.capitalize() for part in name.split("_")),
        "reward_components",
        "transition_reward",
        "reward/transition",
        "server_transition_window",
        "mean",
        "gauge",
        ("actor", "episodes", "reward_components", name),
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


def _project_metric_value(record, definition):
    for path in definition["paths"]:
        value = _finite_number(_nested(record, path))
        if value is not None:
            return value * definition["scale"]
    return None


def project_metric_values(record, definitions):
    projected = copy.deepcopy(record)
    projected["metric_values"] = {
        definition["descriptor"]["field_id"]: _project_metric_value(
            record, definition
        )
        for definition in definitions
    }
    return projected


class MetricsFileReader:
    def __init__(
        self,
        metrics_dir: str,
        *,
        metrics_source_id: str = "",
        service_instance_id: str = "",
        started_at: float | None = None,
    ):
        self._metrics_dir = os.path.abspath(metrics_dir)
        self._metrics_source_id = metrics_source_id or (
            f"local-training-{uuid.uuid4().hex}"
        )
        self._service_instance_id = service_instance_id or (
            f"learner-metrics-{uuid.uuid4().hex}"
        )
        self._started_at = (
            time.time() if started_at is None else float(started_at)
        )
        self._lock = threading.Lock()
        self._records = []
        self._files = {}
        self._corrupt_lines = 0
        self._last_scan_time = 0.0
        os.makedirs(self._metrics_dir, exist_ok=True)
        print(f"[MetricsServer] 监控目录: {self._metrics_dir}")

    def refresh(self):
        with self._lock:
            now = time.monotonic()
            if now - self._last_scan_time >= 0.5:
                self._last_scan_time = now
                for path in glob.glob(
                    os.path.join(self._metrics_dir, "metrics_*.jsonl")
                ):
                    self._files.setdefault(
                        path, {"offset": 0, "pending": b"", "corrupt": 0}
                    )
            for path, state in list(self._files.items()):
                self._read_file(path, state)

    def _read_file(self, path: str, state: dict):
        try:
            file_size = os.path.getsize(path)
            if file_size < state["offset"]:
                state["offset"] = 0
                state["pending"] = b""
            if file_size == state["offset"]:
                return
            with open(path, "rb") as stream:
                stream.seek(state["offset"])
                data = state["pending"] + stream.read()
                state["offset"] = stream.tell()
            lines = data.split(b"\n")
            state["pending"] = lines.pop()
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                try:
                    self._records.append(
                        json.loads(raw_line.decode("utf-8"))
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    state["corrupt"] += 1
                    self._corrupt_lines += 1
        except OSError as exc:
            state["error"] = str(exc)

    def query(self, after_sequence: int = 0, limit: int = 0):
        self.refresh()
        with self._lock:
            records = [
                record
                for record in self._records
                if int(record.get("sequence", record.get("train_step", 0)))
                > after_sequence
            ]
            if limit > 0:
                records = records[:limit]
            return records

    def latest(self):
        self.refresh()
        with self._lock:
            return self._records[-1] if self._records else {}

    def metric_definitions(self):
        self.refresh()
        with self._lock:
            reward_names = set(_KNOWN_REWARD_COMPONENTS)
            for record in self._records:
                components = _nested(
                    record, ("actor", "episodes", "reward_components")
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
            timestamp = float(latest.get("timestamp", 0.0))
            interval_seconds = max(
                float(latest.get("interval_ms", 0.0)) / 1000.0, 0.0
            )
            stale_after = max(5.0, 3.0 * interval_seconds)
            age_seconds = (
                max(0.0, time.time() - timestamp) if timestamp else None
            )
            return {
                "schema_version": 1,
                "service": "learner-metrics",
                "stream": "current",
                "service_instance_id": self._service_instance_id,
                "metrics_source_id": self._metrics_source_id,
                "started_at": self._started_at,
                "metrics_dir": self._metrics_dir,
                "mode": latest.get("mode", ""),
                "record_count": len(self._records),
                "latest_sequence": latest.get(
                    "sequence", latest.get("train_step", 0)
                ),
                "latest_timestamp": timestamp,
                "age_seconds": age_seconds,
                "stale_after_seconds": stale_after,
                "stale": age_seconds is None or age_seconds > stale_after,
                "corrupt_line_count": self._corrupt_lines,
                "file_count": len(self._files),
                "file_errors": {
                    os.path.basename(path): state["error"]
                    for path, state in self._files.items()
                    if state.get("error")
                },
            }

    def summary(self):
        latest = self.latest()
        distributor = latest.get("distributor", {})
        rates = latest.get("rates", {})
        chain = latest.get("chain", {})
        return {
            "mode": latest.get("mode", ""),
            "sequence": latest.get("sequence", 0),
            "consumed": distributor.get("acked", 0),
            "consumer_sps": rates.get("trained_sps", 0.0),
            "queue_size": distributor.get("ready_samples", 0),
            "chain_ready": chain.get("ready", False),
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
                    "records": [
                        project_metric_values(record, definitions)
                        for record in records
                    ],
                    "total": len(records),
                }
            )
        elif path == "/api/metrics/catalog":
            self._json_response(metrics_reader.catalog())
        elif path == "/api/metrics/latest":
            latest = metrics_reader.latest()
            definitions = metrics_reader.metric_definitions()
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "record": (
                        project_metric_values(latest, definitions)
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
    parser.add_argument("--dir", "-d", default="models/local-train/metrics")
    parser.add_argument("--port", "-p", type=int, default=9005)
    parser.add_argument(
        "--source-id",
        default=os.environ.get("RL_METRICS_SOURCE_ID", ""),
    )
    args = parser.parse_args()
    metrics_reader = MetricsFileReader(
        args.dir, metrics_source_id=args.source_id
    )
    metrics_reader.refresh()

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MetricsHTTPHandler)
    print(f"[MetricsServer] http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
