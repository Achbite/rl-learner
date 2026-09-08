"""Producer registration and task-independent reading of metric records."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import deque

from google.protobuf.message import DecodeError
from proto.metrics import catalog_pb2 as metric_catalog_pb2
from proto.metrics import registry_pb2 as metric_registry_pb2
from proto.metrics import transport_pb2 as metric_transport_pb2


class MetricDefinitionError(ValueError):
    """Invalid metric content; independent of durable transport/ACK errors."""


def _validate_definition(definition):
    if not all((definition.metric_id, definition.display_name,
                definition.unit, definition.scope)):
        raise MetricDefinitionError("metric definition is incomplete")
    if definition.value_type not in (
        metric_registry_pb2.METRIC_VALUE_TYPE_SCALAR, metric_registry_pb2.METRIC_VALUE_TYPE_UNSIGNED,
        metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT,
    ) or definition.aggregation not in (
        metric_registry_pb2.METRIC_AGGREGATION_LATEST, metric_registry_pb2.METRIC_AGGREGATION_SUM,
        metric_registry_pb2.METRIC_AGGREGATION_MIN, metric_registry_pb2.METRIC_AGGREGATION_MAX,
        metric_registry_pb2.METRIC_AGGREGATION_MEAN,
    ):
        raise MetricDefinitionError(f"unsupported metric definition: {definition.metric_id}")
    mean = definition.aggregation == metric_registry_pb2.METRIC_AGGREGATION_MEAN
    if mean != (definition.value_type == metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT):
        raise MetricDefinitionError("MEAN requires SUM_COUNT; scalar means are invalid")
    if mean != bool(definition.denominator):
        raise MetricDefinitionError("only MEAN requires a denominator")


class MetricRegistry:
    """Local to a producer lifecycle; no remote registration handshake."""

    def __init__(self):
        self._definitions = {}

    def register(self, metric_id, *, unit, scope, value_type, aggregation,
                 denominator="", display_name=None, category="custom", description=""):
        definition = metric_registry_pb2.MetricDefinition(
            metric_id=metric_id, display_name=display_name or metric_id,
            unit=unit, scope=scope, value_type=value_type,
            aggregation=aggregation, denominator=denominator,
            category=category, description=description,
        )
        previous = self._definitions.get(metric_id)
        if previous is not None:
            if previous != definition:
                raise MetricDefinitionError(f"metric definition changed: {metric_id}")
            return metric_id
        _validate_definition(definition)
        self._definitions[metric_id] = definition
        return metric_id

    def catalog(self, source):
        return metric_catalog_pb2.GetMetricCatalogRsp(source=source, entries=[
            metric_catalog_pb2.MetricCatalogEntry(definition=self._definitions[name])
            for name in sorted(self._definitions)
        ])

    def record(self, points, *, attributes=None):
        points = list(points)
        try:
            definitions = [self._definitions[name]
                           for name in sorted({p.metric_id for p in points})]
        except KeyError as error:
            raise MetricDefinitionError(f"unregistered metric: {error.args[0]}") from error
        record = metric_registry_pb2.RegisteredMetricRecord(
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
        metric_registry_pb2.METRIC_VALUE_TYPE_SCALAR: "scalar",
        metric_registry_pb2.METRIC_VALUE_TYPE_UNSIGNED: "unsigned_value",
        metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT: "sum_count",
    }
    for point in points:
        definition = definitions.get(point.metric_id)
        if definition is None:
            raise MetricDefinitionError(f"record lacks definition: {point.metric_id}")
        has_start = point.HasField("interval_start_unix_ms")
        has_end = point.HasField("interval_end_unix_ms")
        if has_start != has_end or (has_start and point.interval_start_unix_ms > point.interval_end_unix_ms):
            raise MetricDefinitionError(f"invalid metric interval: {point.metric_id}")
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
    if operation == metric_registry_pb2.METRIC_AGGREGATION_MEAN:
        raw = totals.setdefault(name, {"sum": 0.0, "count": 0})
        raw["sum"] += value["sum"]
        raw["count"] += value["count"]
    elif name not in totals or operation == metric_registry_pb2.METRIC_AGGREGATION_LATEST:
        totals[name] = value
    elif operation == metric_registry_pb2.METRIC_AGGREGATION_SUM:
        totals[name] += value
    elif operation == metric_registry_pb2.METRIC_AGGREGATION_MIN:
        totals[name] = min(totals[name], value)
    elif operation == metric_registry_pb2.METRIC_AGGREGATION_MAX:
        totals[name] = max(totals[name], value)


def _merge(totals, definitions, points):
    for point in points:
        value = ({"sum": point.sum_count.sum, "count": int(point.sum_count.count)}
                 if point.WhichOneof("value") == "sum_count"
                 else getattr(point, point.WhichOneof("value")))
        _merge_value(totals, point.metric_id, definitions[point.metric_id].aggregation, value)


def _render(totals, coverage=None):
    rendered = {name: {"value": raw["sum"] / raw["count"], **raw}
                if isinstance(raw, dict) else {"value": raw}
                for name, raw in totals.items()}
    if coverage is not None:
        for name, value in rendered.items():
            value.update(interval_start_unix_ms=coverage[name][0],
                         interval_end_unix_ms=coverage[name][1])
    return rendered


class LocalMetricProjector:
    """Reads any producer's registered metrics without importing task protocols."""

    TIME_WINDOWS_MS = {"5s": 5_000, "1m": 60_000, "1h": 3_600_000, "24h": 86_400_000}

    def __init__(self, store, *, clock=time.time, bucket_ms=5000, mean_window_ms=60000):
        self.store = store
        self.bucket_ms = bucket_ms
        self.mean_window_ms = mean_window_ms
        self._clock = clock
        self._lock = threading.Lock()
        self._row_id = 0
        self._sources = {}

    def _state(self, key, role, source):
        return self._sources.setdefault(key, {
            "role": role, "component": source.component, "instance_id": source.instance_id,
            "lifecycle_epoch": int(source.lifecycle_epoch), "definitions": {}, "origins": {},
            "totals": {}, "latest": {}, "recent": deque(maxlen=100), "buckets": {},
            "maximum_observed_at": 0, "event_count": 0, "error_count": 0, "last_error": None,
        })

    def register_catalog(self, role, catalog):
        key = json.dumps([catalog.source.component, catalog.source.instance_id,
                          int(catalog.source.lifecycle_epoch)], separators=(",", ":"))
        with self._lock:
            state = self._state(key, role, catalog.source)
            for entry in catalog.entries:
                definition = entry.definition
                _validate_definition(definition)
                previous = state["definitions"].get(definition.metric_id)
                if previous is not None and previous != definition:
                    raise MetricDefinitionError(f"metric definition changed: {definition.metric_id}")
                state["definitions"][definition.metric_id] = copy.deepcopy(definition)
                state["origins"][definition.metric_id] = {
                    "status_method": entry.status_method,
                    "status_field": entry.status_field,
                    "status_count_field": entry.status_count_field,
                }
        return key

    def snapshot(self):
        with self._lock:
            snapshot_at = int(self._clock() * 1000)
            for row_id, role, source_key, batch in self.store.committed_batches_after(self._row_id):
                state = self._state(source_key, role, batch.source)
                for event in batch.events:
                    try:
                        if event.fact_kind != metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS:
                            raise MetricDefinitionError("unsupported metric fact kind")
                        record = metric_registry_pb2.RegisteredMetricRecord.FromString(event.fact_payload)
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
                    for point in record.points:
                        end = (int(point.interval_end_unix_ms) if point.HasField("interval_end_unix_ms")
                               else observed_at)
                        start = (int(point.interval_start_unix_ms) if point.HasField("interval_start_unix_ms")
                                 else end)
                        bucket = (end + self.bucket_ms - 1) // self.bucket_ms * self.bucket_ms
                        bucket_state = state["buckets"].setdefault(bucket, {
                            "totals": {}, "sequences": {}, "coverage": {}})
                        _merge(bucket_state["totals"], definitions, [point])
                        bucket_state["sequences"][point.metric_id] = int(event.event_sequence)
                        old = bucket_state["coverage"].get(point.metric_id, (start, end))
                        bucket_state["coverage"][point.metric_id] = (min(start, old[0]), max(end, old[1]))
                    floor = (state["maximum_observed_at"] - self.TIME_WINDOWS_MS["24h"]) // self.bucket_ms * self.bucket_ms
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
                for label, duration in {**self.TIME_WINDOWS_MS, "current": self.mean_window_ms}.items():
                    totals = {}
                    # A quiet producer must age out of the current window;
                    # anchoring to its last event would keep old rewards fresh.
                    boundary = snapshot_at // self.bucket_ms * self.bucket_ms
                    floor = boundary - duration
                    coverage = {}
                    latest_sequences = {}
                    for bucket, bucket_state in state["buckets"].items():
                        if bucket <= floor or bucket > boundary:
                            continue
                        for name, value in bucket_state["totals"].items():
                            operation = state["definitions"][name].aggregation
                            sequence = bucket_state["sequences"][name]
                            if operation == metric_registry_pb2.METRIC_AGGREGATION_LATEST:
                                if sequence <= latest_sequences.get(name, 0):
                                    continue
                                latest_sequences[name] = sequence
                            _merge_value(totals, name, operation, value)
                            start, end = bucket_state["coverage"][name]
                            old = coverage.get(name, (start, end))
                            coverage[name] = (min(start, old[0]), max(end, old[1]))
                    windows[label] = _render(totals, coverage)
                    for value in windows[label].values():
                        value.update(window_start_unix_ms=floor, window_end_unix_ms=boundary)
                catalog = {}
                for name, definition in state["definitions"].items():
                    catalog[name] = {
                        field: getattr(definition, field) for field in (
                            "metric_id", "display_name", "unit", "scope", "denominator", "category", "description")
                    }
                    catalog[name].update(state["origins"].get(name, {}))
                    catalog[name]["value_type"] = metric_registry_pb2.MetricValueType.Name(definition.value_type).removeprefix("METRIC_VALUE_TYPE_").lower()
                    catalog[name]["aggregation"] = metric_registry_pb2.MetricAggregation.Name(definition.aggregation).removeprefix("METRIC_AGGREGATION_").lower()
                sources[key] = {
                    field: state[field] for field in (
                        "role", "component", "instance_id", "lifecycle_epoch", "event_count",
                        "error_count", "last_error")
                }
                sources[key].update({
                    "catalog": catalog,
                    "status": "projection_error" if state["error_count"] else "ok",
                    "latest_event_sequence": state["recent"][-1][0] if state["recent"] else None,
                    "latest_observed_at_unix_ms": state["recent"][-1][1] if state["recent"] else None,
                    "windows": windows,
                    "window_kind": "source_events",
                    "time_bucket_ms": self.bucket_ms,
                    "mean_window_ms": self.mean_window_ms,
                    "query_time_unix_ms": snapshot_at,
                    "current_intervals": [
                        {"end_unix_ms": bucket, "values": _render(value["totals"], value["coverage"])}
                        for bucket, value in sorted(state["buckets"].items())
                        # Keep the pending boundary for queries after the final snapshot.
                        if snapshot_at // self.bucket_ms * self.bucket_ms - self.mean_window_ms < bucket],
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
