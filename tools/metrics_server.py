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
        ("metric_event_views", "train_updates", "latest", "values",
         "latest_model_version"),
        ("learner", "model_version"), ("model", "latest_version"),
        ("model_version",),
    ),
    _metric_definition(
        "learner.train_update.total.v1", "Train Update", "training_depth",
        "train_update", "update", "learner", "total", "counter",
        ("metric_event_views", "train_updates", "latest", "values",
         "latest_train_update_sequence"),
        ("learner", "run_train_updates"), ("learner", "train_updates"),
        ("train_step",),
    ),
    _metric_definition(
        "learner.trained_samples.total.v1", "Trained Samples",
        "training_depth", "sample_count", "samples", "learner", "total",
        "counter",
        ("metric_event_views", "train_updates", "latest", "values",
         "latest_cumulative_trained_samples"),
        ("learner", "run_trained_samples"),
        ("learner", "trained_samples"), ("trained_samples",),
    ),
    _metric_definition(
        "server.episode.max_steps.current.v1", "Episode Max Steps",
        "training_depth", "environment_step", "step", "server",
        "latest", "gauge",
        ("actor", "metric_values", "server.episode.max_steps.current.v1"),
    ),
    _metric_definition(
        "learner.loss.policy.v1", "Policy Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "policy_loss", "mean"), ("learner", "policy_loss"),
        ("policy_loss",),
    ),
    _metric_definition(
        "learner.loss.value.v1", "Value Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "value_loss", "mean"), ("learner", "value_loss"),
        ("value_loss",),
    ),
    _metric_definition(
        "learner.loss.total.v1", "Total Loss", "loss", "loss", "1",
        "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "total_loss", "mean"), ("learner", "total_loss"),
        ("total_loss",),
    ),
    _metric_definition(
        "server.episode.learning_return.mean.v1", "Mean Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_agent_return"), ("actor", "episodes", "mean_agent_return"),
        ("mean_episode_reward",),
    ),
    _metric_definition(
        "server.training.episode.learning_return.mean.v1",
        "Mean Training Agent Return", "episode_return", "episode_return",
        "reward", "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_agent_return"),
        ("actor", "metric_values",
         "server.training.episode.learning_return.mean.v1"),
        ("actor", "episodes", "mean_agent_return"),
    ),
    _metric_definition(
        "server.training.episode.learning_return.latest_mean.v1",
        "Latest Training Agent Return", "episode_return", "episode_return",
        "reward", "latest_training_environment_episode", "latest", "gauge",
        ("metric_event_views", "episodes", "latest", "values",
         "mean_agent_return"),
        ("actor", "metric_values",
         "server.training.episode.learning_return.latest_mean.v1"),
        ("actor", "episodes", "latest_agent_return"),
    ),
    _metric_definition(
        "server.training.episode.completed.total.v1",
        "Completed Training Episodes", "training_depth", "episode_count",
        "episodes", "server", "total", "counter",
        ("metric_event_views", "episodes", "windows", "all", "raw",
         "environment_episode_count"),
        ("actor", "metric_values",
         "server.training.episode.completed.total.v1"),
    ),
    _metric_definition(
        "server.episode.learning_return.min.v1", "Min Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "min", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "min_agent_return"), ("actor", "episodes", "min_agent_return"),
    ),
    _metric_definition(
        "server.episode.learning_return.max.v1", "Max Learning Return",
        "episode_return", "episode_return", "reward", "agent_episode_window",
        "max", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "max_agent_return"), ("actor", "episodes", "max_agent_return"),
    ),
    _metric_definition(
        "server.episode.success.agent_rate.v1", "Agent Success",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "agent_success_rate"), ("actor", "episodes", "agent_success_rate"),
        ("pass_rate",), scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.agent_rate.v1",
        "Training Agent Success", "episode_success", "percentage", "%",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "agent_success_rate"),
        ("actor", "metric_values",
         "server.training.episode.success.agent_rate.v1"),
        ("actor", "episodes", "agent_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.any_rate.v1",
        "Training Any Success", "episode_success", "percentage", "%",
        "training_environment_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "any_success_rate"),
        ("actor", "metric_values",
         "server.training.episode.success.any_rate.v1"),
        ("actor", "episodes", "any_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.success.all_rate.v1",
        "Training All Success", "episode_success", "percentage", "%",
        "training_environment_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "all_success_rate"),
        ("actor", "metric_values",
         "server.training.episode.success.all_rate.v1"),
        ("actor", "episodes", "all_success_rate"), scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.any_rate.v1", "Any Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "any_success_rate"), ("actor", "episodes", "any_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.success.all_rate.v1", "All Success",
        "episode_success", "percentage", "%", "environment_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "all_success_rate"), ("actor", "episodes", "all_success_rate"),
        scale=100.0,
    ),
    _metric_definition(
        "server.episode.path_ratio.mean.v1", "Path Ratio",
        "episode_success", "ratio", "1", "successful_agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "path_ratio_mean"),
        ("actor", "metric_values", "server.episode.path_ratio.mean.v1"),
    ),
    _metric_definition(
        "server.episode.step.mean.v1", "Episode Step",
        "episode_success", "environment_step", "step",
        "agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_episode_step"),
        ("actor", "metric_values", "server.episode.step.mean.v1"),
    ),
    _metric_definition(
        "server.episode.unique_cells.mean.v1", "Unique Cells",
        "episode_success", "cell_count", "cells", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_unique_cells"),
        ("actor", "metric_values", "server.episode.unique_cells.mean.v1"),
    ),
    _metric_definition(
        "server.episode.blocked_move_rate.v1", "Blocked Move Rate",
        "episode_success", "percentage", "%", "agent_episode_window",
        "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "blocked_move_rate"),
        ("actor", "metric_values", "server.episode.blocked_move_rate.v1"),
        scale=100.0,
    ),
    _metric_definition(
        "server.training.episode.path_ratio.mean.v1",
        "Training Path Ratio", "episode_success", "ratio", "1",
        "successful_training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "path_ratio_mean"),
        ("actor", "metric_values",
         "server.training.episode.path_ratio.mean.v1"),
    ),
    _metric_definition(
        "server.training.episode.step.mean.v1", "Training Episode Step",
        "episode_success", "environment_step", "step",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_episode_step"),
        ("actor", "metric_values",
         "server.training.episode.step.mean.v1"),
    ),
    _metric_definition(
        "server.training.episode.unique_cells.mean.v1",
        "Training Unique Cells", "episode_success", "cell_count", "cells",
        "training_agent_episode_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "mean_unique_cells"),
        ("actor", "metric_values",
         "server.training.episode.unique_cells.mean.v1"),
    ),
    _metric_definition(
        "server.training.episode.blocked_move_rate.v1",
        "Training Blocked Move Rate", "episode_success", "percentage", "%",
        "training_transition_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "blocked_move_rate"),
        ("actor", "metric_values",
         "server.training.episode.blocked_move_rate.v1"), scale=100.0,
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
        "sample.flow.producer_stale_before_ingress.total.v1",
        "Producer Stale Before Ingress", "sample_flow", "sample_count",
        "samples", "server", "total", "counter",
        ("actor", "producer_stale_before_ingress"),
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
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "entropy", "mean"),
        ("learner", "entropy"), ("entropy",),
    ),
    _metric_definition(
        "learner.ppo.approx_kl.v1", "Approx. KL", "ppo_stability",
        "divergence", "1", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "approx_kl", "mean"),
        ("learner", "approx_kl"), ("approx_kl",),
    ),
    _metric_definition(
        "learner.ppo.clip_fraction.v1", "Clip Fraction", "ppo_stability",
        "percentage", "%", "train_update", "mean", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "clip_fraction", "mean"),
        ("learner", "clip_fraction"), ("clip_fraction",), scale=100.0,
    ),
    _metric_definition(
        "learner.ppo.gradient_norm.v1", "Gradient Norm", "ppo_stability",
        "norm", "1", "train_update", "latest", "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "gradient_norm", "mean"),
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
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "policy_lag", "mean"),
        ("learner", "policy_lag"), ("policy_lag",),
    ),
    _metric_definition(
        "learner.value.prediction_mean.v1", "Value Prediction Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "value_prediction", "mean"),
        ("learner", "value_pred_mean"),
        ("value_pred_mean",),
    ),
    _metric_definition(
        "learner.value.return_target_mean.v1", "Return Target Mean",
        "ppo_stability", "value", "reward", "train_update", "mean",
        "gauge",
        ("metric_event_views", "train_updates", "latest", "values", "ppo",
         "return_target", "mean"),
        ("learner", "return_target_mean"),
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


def training_reward_component_field_id(name: str, statistic: str) -> str:
    if not isinstance(name, str) or not _REWARD_COMPONENT_NAME.fullmatch(name):
        raise ValueError("reward component name must be canonical snake_case")
    if statistic not in {
        "episode_mean",
        "transition_mean",
        "latest_episode_mean",
    }:
        raise ValueError("unsupported training reward component statistic")
    return f"server.training.reward.component.{name}.{statistic}.v1"


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
        ("actor", "metric_values", field_id),
        ("actor", "episodes", "transition_reward_components", name),
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
            ("actor", "metric_values", field_id),
            ("actor", "episodes", "reward_components", name),
        )
    if statistic == "latest_episode_mean":
        return _metric_definition(
            field_id, f"Latest {label} / Agent Episode", "reward_components",
            "episode_reward", "reward/agent episode",
            "latest_training_environment_episode", "latest", "gauge",
            ("metric_event_views", "episodes", "latest", "values",
             "reward_components", name, "episode_mean"),
            ("actor", "metric_values", field_id),
            ("actor", "episodes", "latest_reward_components", name),
        )
    return _metric_definition(
        field_id, f"{label} / Transition", "reward_components",
        "transition_reward", "reward/transition",
        "training_transition_window", "mean", "gauge",
        ("metric_event_views", "episodes", "windows", "100", "values",
         "reward_components", name, "transition_mean"),
        ("actor", "metric_values", field_id),
        ("actor", "episodes", "transition_reward_components", name),
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


def _project_metric_value(record, definition, event_window="100"):
    if event_window not in _EVENT_WINDOWS:
        raise ValueError("unsupported metric event window")
    event_views = record.get("metric_event_views")
    event_views_authoritative = (
        isinstance(event_views, dict)
        and event_views.get("status") != "unavailable"
    )
    for path in definition["paths"]:
        selected_path = path
        if (
            len(path) >= 5
            and path[:3] == ("metric_event_views", "episodes", "windows")
            and path[3] == "100"
        ):
            selected_path = (*path[:3], event_window, *path[4:])
        value = _finite_number(_nested(record, selected_path))
        if value is not None:
            return value * definition["scale"]
        if path and path[0] == "metric_event_views" and event_views_authoritative:
            return None
    return None


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
    ):
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if read_chunk_bytes <= 0 or max_pending_bytes <= 0:
            raise ValueError("metrics read bounds must be positive")
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
        self._lock = threading.Lock()
        self._max_records = int(max_records)
        self._read_chunk_bytes = int(read_chunk_bytes)
        self._max_pending_bytes = int(max_pending_bytes)
        self._records = deque(maxlen=self._max_records)
        self._total_record_count = 0
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
                        path,
                        {
                            "offset": 0,
                            "pending": b"",
                            "discarding_oversize_line": False,
                            "corrupt": 0,
                        },
                    )
            for path, state in list(self._files.items()):
                self._read_file(path, state)

    def _read_file(self, path: str, state: dict):
        try:
            file_size = os.path.getsize(path)
            if file_size < state["offset"]:
                state["offset"] = 0
                state["pending"] = b""
                state["discarding_oversize_line"] = False
            if file_size == state["offset"]:
                return
            with open(path, "rb") as stream:
                stream.seek(state["offset"])
                chunk = stream.read(self._read_chunk_bytes)
                state["offset"] = stream.tell()
            data = chunk
            if state.get("discarding_oversize_line"):
                newline = data.find(b"\n")
                if newline < 0:
                    return
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
                    self._records.append(
                        json.loads(raw_line.decode("utf-8"))
                    )
                    self._total_record_count += 1
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
                "mode": latest.get("mode") or self._runtime_mode,
                "record_count": len(self._records),
                "total_record_count": self._total_record_count,
                "retained_record_limit": self._max_records,
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
    parser.add_argument("--dir", "-d", default="models/local-train/metrics")
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
