"""Producer registration and task-independent reading of metric records."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque

from google.protobuf.message import DecodeError
from proto import training_pb2 as wire


class MetricDefinitionError(ValueError):
    """Invalid metric content; independent of durable transport/ACK errors."""


def _validate_definition(definition):
    if not all((definition.metric_id, definition.display_name,
                definition.unit, definition.scope)):
        raise MetricDefinitionError("metric definition is incomplete")
    if definition.value_type not in (
        wire.METRIC_VALUE_TYPE_SCALAR, wire.METRIC_VALUE_TYPE_UNSIGNED,
        wire.METRIC_VALUE_TYPE_SUM_COUNT,
    ) or definition.aggregation not in (
        wire.METRIC_AGGREGATION_LATEST, wire.METRIC_AGGREGATION_SUM,
        wire.METRIC_AGGREGATION_MIN, wire.METRIC_AGGREGATION_MAX,
        wire.METRIC_AGGREGATION_MEAN,
    ):
        raise MetricDefinitionError(f"unsupported metric definition: {definition.metric_id}")
    mean = definition.aggregation == wire.METRIC_AGGREGATION_MEAN
    if mean != (definition.value_type == wire.METRIC_VALUE_TYPE_SUM_COUNT):
        raise MetricDefinitionError("MEAN requires SUM_COUNT; scalar means are invalid")
    if mean != bool(definition.denominator):
        raise MetricDefinitionError("only MEAN requires a denominator")


class MetricRegistry:
    """Local to a producer lifecycle; no remote registration handshake."""

    def __init__(self):
        self._definitions = {}

    def register(self, metric_id, *, unit, scope, value_type, aggregation,
                 denominator="", display_name=None):
        definition = wire.MetricDefinition(
            metric_id=metric_id, display_name=display_name or metric_id,
            unit=unit, scope=scope, value_type=value_type,
            aggregation=aggregation, denominator=denominator,
        )
        previous = self._definitions.get(metric_id)
        if previous is not None:
            if previous != definition:
                raise MetricDefinitionError(f"metric definition changed: {metric_id}")
            return metric_id
        _validate_definition(definition)
        self._definitions[metric_id] = definition
        return metric_id

    def record(self, points, *, attributes=None):
        points = list(points)
        try:
            definitions = [self._definitions[name]
                           for name in sorted({p.metric_id for p in points})]
        except KeyError as error:
            raise MetricDefinitionError(f"unregistered metric: {error.args[0]}") from error
        record = wire.RegisteredMetricRecord(
            definitions=definitions, points=points, attributes=attributes or {},
        )
        _validate_points(points, self._definitions)
        return record


def validate_record(record, registered):
    """Decode boundary: validate the entire record before advancing any totals."""
    definitions = {}
    for definition in record.definitions:
        _validate_definition(definition)
        name = definition.metric_id
        if name in definitions:
            raise MetricDefinitionError(f"duplicate metric definition: {name}")
        if name in registered and registered[name] != definition:
            raise MetricDefinitionError(f"metric definition changed: {name}")
        definitions[name] = definition
    _validate_points(record.points, definitions)
    return definitions


def _validate_points(points, definitions):
    if not points:
        raise MetricDefinitionError("metric record has no points")
    expected = {
        wire.METRIC_VALUE_TYPE_SCALAR: "scalar",
        wire.METRIC_VALUE_TYPE_UNSIGNED: "unsigned_value",
        wire.METRIC_VALUE_TYPE_SUM_COUNT: "sum_count",
    }
    for point in points:
        definition = definitions.get(point.metric_id)
        if definition is None:
            raise MetricDefinitionError(f"record lacks definition: {point.metric_id}")
        kind = point.WhichOneof("value")
        if kind != expected[definition.value_type]:
            raise MetricDefinitionError(f"metric value type differs: {point.metric_id}")
        if kind == "scalar" and not math.isfinite(point.scalar):
            raise MetricDefinitionError(f"non-finite metric: {point.metric_id}")
        if kind == "sum_count" and (
            not math.isfinite(point.sum_count.sum) or point.sum_count.count == 0
        ):
            raise MetricDefinitionError(f"invalid sum/count: {point.metric_id}")


def _merge_value(totals, name, operation, value):
    if operation == wire.METRIC_AGGREGATION_MEAN:
        raw = totals.setdefault(name, {"sum": 0.0, "count": 0})
        raw["sum"] += value["sum"]
        raw["count"] += value["count"]
    elif name not in totals or operation == wire.METRIC_AGGREGATION_LATEST:
        totals[name] = value
    elif operation == wire.METRIC_AGGREGATION_SUM:
        totals[name] += value
    elif operation == wire.METRIC_AGGREGATION_MIN:
        totals[name] = min(totals[name], value)
    elif operation == wire.METRIC_AGGREGATION_MAX:
        totals[name] = max(totals[name], value)


def _merge(totals, definitions, points):
    for point in points:
        value = ({"sum": point.sum_count.sum, "count": int(point.sum_count.count)}
                 if point.WhichOneof("value") == "sum_count"
                 else getattr(point, point.WhichOneof("value")))
        _merge_value(totals, point.metric_id, definitions[point.metric_id].aggregation, value)


def _render(totals):
    return {name: {"value": raw["sum"] / raw["count"], **raw}
            if isinstance(raw, dict) else {"value": raw}
            for name, raw in totals.items()}


class LocalMetricProjector:
    """Reads any producer's registered metrics without importing task protocols."""

    TIME_WINDOWS_MS = {"5s": 5_000, "1m": 60_000, "1h": 3_600_000, "24h": 86_400_000}

    def __init__(self, store, *, clock=time.time):
        self.store = store
        self._clock = clock
        self._lock = threading.Lock()
        self._row_id = 0
        self._sources = {}

    def snapshot(self):
        with self._lock:
            snapshot_at = int(self._clock() * 1000)
            for row_id, role, source_key, batch in self.store.committed_batches_after(self._row_id):
                state = self._sources.setdefault(source_key, {
                    "role": role, "instance_id": batch.source.instance_id,
                    "lifecycle_epoch": int(batch.source.lifecycle_epoch),
                    "definitions": {}, "totals": {}, "latest": {}, "recent": deque(maxlen=100),
                    "buckets": {}, "maximum_observed_at": 0,
                    "event_count": 0, "error_count": 0, "last_error": None,
                })
                for event in batch.events:
                    try:
                        if event.fact_kind != wire.METRIC_FACT_KIND_REGISTERED_METRICS:
                            raise MetricDefinitionError("unsupported metric fact kind")
                        record = wire.RegisteredMetricRecord.FromString(event.fact_payload)
                        definitions = validate_record(record, state["definitions"])
                    except (DecodeError, MetricDefinitionError) as error:
                        state["error_count"] += 1
                        state["last_error"] = {
                            "event_sequence": int(event.event_sequence), "message": str(error),
                        }
                        continue
                    state["definitions"].update(definitions)
                    _merge(state["totals"], definitions, record.points)
                    state["recent"].append((int(event.event_sequence),
                        int(event.observed_at_unix_ms), record))
                    latest = {}
                    _merge(latest, definitions, record.points)
                    state["latest"].update(latest)
                    observed_at = int(event.observed_at_unix_ms)
                    state["maximum_observed_at"] = max(state["maximum_observed_at"], observed_at)
                    bucket = observed_at // 5000 * 5000
                    bucket_state = state["buckets"].setdefault(bucket, {"totals": {}, "sequences": {}})
                    _merge(bucket_state["totals"], definitions, record.points)
                    bucket_state["sequences"].update({p.metric_id: int(event.event_sequence) for p in record.points})
                    floor = (state["maximum_observed_at"] - self.TIME_WINDOWS_MS["24h"]) // 5000 * 5000
                    state["buckets"] = {key: value for key, value in state["buckets"].items() if key >= floor}
                    state["event_count"] += 1
                # Content errors are retained and reported once. They do not
                # rewind transport cursors or repeatedly poison later snapshots.
                self._row_id = row_id
            store_snapshot = self.store.snapshot()
            sources = {}
            for key, state in self._sources.items():
                windows = {"all": _render(state["totals"]), "latest": _render(state["latest"])}
                for size in (25, 100):
                    totals = {}
                    for _, _, record in list(state["recent"])[-size:]:
                        _merge(totals, state["definitions"], record.points)
                    windows[str(size)] = _render(totals)
                for label, duration in self.TIME_WINDOWS_MS.items():
                    totals = {}
                    # A quiet producer must age out of the current window;
                    # anchoring to its last event would keep old rewards fresh.
                    floor = (snapshot_at - duration + 1) // 5000 * 5000
                    latest_sequences = {}
                    for bucket, bucket_state in state["buckets"].items():
                        if bucket < floor or bucket > snapshot_at:
                            continue
                        for name, value in bucket_state["totals"].items():
                            operation = state["definitions"][name].aggregation
                            sequence = bucket_state["sequences"][name]
                            if operation == wire.METRIC_AGGREGATION_LATEST:
                                if sequence <= latest_sequences.get(name, 0):
                                    continue
                                latest_sequences[name] = sequence
                            _merge_value(totals, name, operation, value)
                    windows[label] = _render(totals)
                catalog = {}
                for name, definition in state["definitions"].items():
                    catalog[name] = {
                        field: getattr(definition, field) for field in (
                            "metric_id", "display_name", "unit", "scope", "denominator")
                    }
                    catalog[name]["value_type"] = wire.MetricValueType.Name(definition.value_type).removeprefix("METRIC_VALUE_TYPE_").lower()
                    catalog[name]["aggregation"] = wire.MetricAggregation.Name(definition.aggregation).removeprefix("METRIC_AGGREGATION_").lower()
                sources[key] = {
                    field: state[field] for field in (
                        "role", "instance_id", "lifecycle_epoch", "event_count",
                        "error_count", "last_error")
                }
                sources[key].update({
                    "catalog": catalog,
                    "status": "projection_error" if state["error_count"] else "ok",
                    "latest_event_sequence": state["recent"][-1][0] if state["recent"] else None,
                    "latest_observed_at_unix_ms": state["recent"][-1][1] if state["recent"] else None,
                    "windows": windows,
                    "window_kind": "source_events",
                    "time_bucket_ms": 5000,
                    "recent_event_count": len(state["recent"]),
                })
            status = "provisional"
            if any(s["error_count"] for s in sources.values()):
                status = "projection_error"
            elif store_snapshot["incomplete_source_count"]:
                status = "incomplete"
            elif store_snapshot["sources"] and all(s["final_acknowledged"] for s in store_snapshot["sources"]):
                status = "final"
            elif not sources:
                status = "no_data"
            return copy.deepcopy({"status": status, "sources": sources})
