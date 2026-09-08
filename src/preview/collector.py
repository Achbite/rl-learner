"""Optional local Preview: RPC collection, durable receipt, projection and JSONL output.

This module has no TrainingRuntime, trainer, Client or task-protocol dependency.
"""
from __future__ import annotations
import copy
import math
import resource
import sys
import threading
import time
import uuid
from pathlib import Path
import grpc
from proto.common import identity_pb2 as common_pb2
from proto.training import training_pb2
from proto.training import training_pb2_grpc
from proto.metrics import catalog_pb2 as metric_catalog_pb2
from proto.metrics import catalog_pb2_grpc as metric_catalog_pb2_grpc
from proto.metrics import registry_pb2 as metric_registry_pb2
from proto.metrics import transport_pb2_grpc as metric_transport_pb2_grpc
from src.contracts.identity import model_identity_document
from src.metrics.metric_events import RawMetricBatchStore, MetricEventCollector, _source_key
from src.metrics.registered_metrics import LocalMetricProjector
from src.metrics.metrics_backend import create_backend
from .topology import training_chain_status

class PreviewCollector:
    def __init__(self, *, endpoints, learner_source, source_id, directory, backend, logger,
                 collect_aiserver=True, mean_window_ms=60000, bucket_ms=5000):
        self.logger = logger
        self.source_id = source_id
        self.channels = {role: grpc.insecure_channel(address) for role, address in endpoints.items()}
        self.catalog_stubs = {role: metric_catalog_pb2_grpc.MetricCatalogServiceStub(channel)
                              for role, channel in self.channels.items()}
        self._status_readers = {
            "learner": (training_pb2_grpc.LearnerStatusServiceStub(self.channels["learner"]).GetLearnerStatus,
                        training_pb2.LearnerStatusReq, "learner"),
            "aiserver": (training_pb2_grpc.AIServerTrainingStatusServiceStub(self.channels["aiserver"]).GetAIServerStatus,
                         training_pb2.AIServerStatusReq, "aiserver"),
            "sample_pool": (training_pb2_grpc.SamplePoolConsumerServiceStub(self.channels["sample_pool"]).GetStatus,
                            training_pb2.SamplePoolStatusReq, "sample_pool"),
            "distributor": (training_pb2_grpc.ModelDistributorServiceStub(self.channels["distributor"]).GetModelDistributorStatus,
                            training_pb2.ModelDistributorStatusReq, "distributor"),
        }
        self._status = {}
        self._catalogs = {}
        self._catalog_errors = {}
        self._sources = {"learner": copy.deepcopy(learner_source)}
        self._binding_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.sequence = 0
        self._rate_snapshot = {}
        self._last_rate_time = None
        self._last_resource_time = time.monotonic()
        self._last_process_cpu = time.process_time()
        self.backend = create_backend(backend, str(directory))
        self.store = RawMetricBatchStore(Path(directory) / "preview-events.sqlite3")
        self.projector = LocalMetricProjector(self.store, mean_window_ms=mean_window_ms, bucket_ms=bucket_ms)
        self.consumer = common_pb2.ServiceInstanceIdentity(
            component="local-preview", instance_id="preview-" + uuid.uuid4().hex, lifecycle_epoch=1)
        roles = ("learner", "aiserver") if collect_aiserver else ("learner",)
        self.collectors = {role: MetricEventCollector(store=self.store, consumer=self.consumer, role=role,
            event_stub=metric_transport_pb2_grpc.MetricEventServiceStub(self.channels[role]), logger=logger) for role in roles}

    def bind_aiserver(self, source):
        # Only the business bootstrap ACK supplies this identity. Never discover a replacement by port.
        _source_key(source)
        with self._binding_lock:
            current = self._sources.get("aiserver")
            if current is not None and current != source:
                raise ValueError("Preview AIServer source was already bound for this lifecycle")
            self._sources["aiserver"] = copy.deepcopy(source)
            if "aiserver" in self.collectors:
                self.collectors["aiserver"].start(source)

    @staticmethod
    def _has_field(message, name):
        field = message.DESCRIPTOR.fields_by_name[name]
        return not field.has_presence or message.HasField(name)

    def _read_status(self, role):
        reader, request, identity_field = self._status_readers[role]
        with self._binding_lock:
            expected = self._sources.get(role)
        if role == "aiserver" and expected is None:
            raise RuntimeError("waiting for the bootstrap AIServer source")
        status = reader(request(), timeout=1.5)
        source = getattr(status, identity_field)
        key = _source_key(source)
        if expected is not None and expected != source:
            raise RuntimeError(f"{role} status source differs from the configured lifecycle")
        catalog = self._catalogs.get(key)
        if catalog is None:
            try:
                response = self.catalog_stubs[role].GetMetricCatalog(metric_catalog_pb2.GetMetricCatalogReq(), timeout=1.5)
                if response.source != source:
                    raise RuntimeError(f"{role} catalog and status describe different sources")
                self.projector.register_catalog(role, response)
                self._catalogs[key] = catalog = response
            except (grpc.RpcError, RuntimeError, ValueError) as error:
                self._catalog_errors[role] = str(error)
        self._status[key] = (status, catalog)
        return status

    def _learner_snapshot(self):
        try:
            status = self._read_status("learner")
            document = {field.name: getattr(status, field.name)
                        for field in status.DESCRIPTOR.fields
                        if field.name not in ("learner", "model") and self._has_field(status, field.name)}
            document.update(instance_id=status.learner.instance_id,
                lifecycle_epoch=int(status.learner.lifecycle_epoch),
                model_identity=model_identity_document(status.model), error=status.last_error)
            return document
        except (grpc.RpcError, RuntimeError, ValueError) as error:
            return self._component_error_snapshot("learner", str(error))

    @staticmethod
    def _field(status, path):
        value = status
        for name in path.split("."):
            field = value.DESCRIPTOR.fields_by_name.get(name)
            if field is None:
                raise ValueError(f"catalog status field is absent: {path}")
            if field.has_presence and not value.HasField(name):
                return None
            value = getattr(value, name)
        return value

    def _views(self):
        view = self.projector.snapshot()
        view["catalog_errors"] = dict(self._catalog_errors)
        for key, source in view["sources"].items():
            # No previous status value is carried across an unavailable observation.
            for name, definition in source["catalog"].items():
                if definition.get("status_method"):
                    source["windows"]["latest"].pop(name, None)
            current = self._status.get(key)
            source["catalog_available"] = key in self._catalogs
            source["status_available"] = current is not None
            if current is None:
                continue
            status, catalog = current
            if catalog is None:
                continue
            for entry in catalog.entries:
                if not entry.status_method:
                    continue
                value = self._field(status, entry.status_field)
                count = self._field(status, entry.status_count_field) if entry.status_count_field else None
                if value is None or (entry.status_count_field and not count):
                    continue
                item = {"value": value}
                if entry.definition.value_type == metric_registry_pb2.METRIC_VALUE_TYPE_SUM_COUNT:
                    item = {"sum": value, "count": count, "value": value / count}
                item["observed_at_unix_ms"] = int(status.timestamp_unix_ms)
                source["windows"]["latest"][entry.definition.metric_id] = item
        return view

    def record(self):
        with self._poll_lock:
            self._status = {}
            self._catalog_errors = {}
            actor = self._actor_snapshot()
            sample_pool = self._sample_pool_snapshot()
            model = self._model_snapshot()
            learner = self._learner_snapshot()
            now = time.time()
            rates = self._rates(actor, sample_pool, now)
            self.sequence += 1
            transport = self.store.snapshot()
            transport.update({role + "_relay": collector.snapshot() for role, collector in self.collectors.items()})
            self.backend.write({
                "mode": "training", "metrics_source_id": self.source_id,
                "sequence": self.sequence, "timestamp": now,
                "interval_ms": None if rates.get("window_seconds") is None else rates["window_seconds"] * 1000,
                "configured_poll_interval_ms": 1000, "learner": learner, "actor": actor,
                "sample_pool": sample_pool, "model": model,
                "chain": training_chain_status(actor, sample_pool, learner, model, learner.get("error", "")),
                "rates": rates, "resources": {"learner": self._resource_snapshot()},
                "metric_events": transport, "metric_event_views": self._views(),
            })

    def record_best_effort(self, phase):
        try:
            self.record()
        except Exception as error:
            self.logger.error("Preview snapshot failed during %s: %s", phase, error)

    def start(self):
        self.collectors["learner"].start(self._sources["learner"])
        def run():
            while not self._stop.is_set():
                self.record_best_effort("poll")
                self._stop.wait(1.0)
        self._thread = threading.Thread(target=run, name="local-preview", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            # Every RPC has a deadline; finish the current read before closing
            # its channels and stores, even when several endpoints are slow.
            self._thread.join()
        for collector in self.collectors.values():
            collector.close()
        self.record_best_effort("final receipt")
        self.backend.close()
        self.store.close()
        for channel in self.channels.values():
            channel.close()

    @staticmethod
    def _component_error_snapshot(component: str, error: str) -> dict:
        return {
            "component": component,
            "ready": False,
            "error": error,
            "timestamp": time.time(),
        }

    @staticmethod
    def _raw_mean(raw_sum: float, count: int) -> float | None:
        if count < 0 or not math.isfinite(raw_sum):
            raise ValueError("raw sum/count metric is invalid")
        return None if count == 0 else raw_sum / count

    def _actor_snapshot(self) -> dict:
        try:
            status = self._read_status("aiserver")
            inference_count = int(status.inference_count)
            push_rpc_count = int(status.push_rpc_count)
            segment_close_counts = {
                training_pb2.SegmentCloseReason.Name(item.reason): int(
                    item.count
                )
                for item in status.segment_close_counts
            }
            return {
                "ready": bool(status.ready),
                "state": training_pb2.AIServerState.Name(status.state),
                "instance_id": status.aiserver.instance_id,
                "lifecycle_epoch": int(status.aiserver.lifecycle_epoch),
                "active_sessions": int(
                    status.active_actor_session_count
                ),
                "active_segments": int(status.active_segment_count),
                "model_identity": model_identity_document(status.loaded_model),
                "model_feedback": {
                    "candidate_model": model_identity_document(status.model_feedback.candidate_model),
                    "stage": status.model_feedback.stage,
                    "last_error": status.model_feedback.last_error,
                },
                "staged_model_identity": model_identity_document(
                    status.staged_model
                ),
                "produced": int(status.produced_unique_transitions),
                "produced_envelopes": int(
                    status.produced_unique_envelopes
                ),
                "accepted": int(status.accepted_unique_transitions),
                "push_attempts": int(status.push_attempt_count),
                "duplicate_push_attempts": int(
                    status.duplicate_push_attempt_count
                ),
                "rejected_push_attempts": int(
                    status.rejected_push_attempt_count
                ),
                "retry_attempts": int(status.retry_attempt_count),
                "final_drop": int(status.final_drop_unique_transitions),
                "outbound_queue_envelopes": int(
                    status.outbound_queue_envelopes
                ),
                "outbound_queue_transitions": int(
                    status.outbound_queue_transitions
                ),
                "outbound_queue_estimated_bytes": int(
                    status.outbound_queue_estimated_bytes
                ),
                "outbound_queue_high_watermark": int(
                    status.outbound_queue_high_watermark
                ),
                "inference_count": inference_count,
                "inference_latency_sum_ms": float(
                    status.inference_latency_sum_ms
                ),
                "inference_mean_ms": self._raw_mean(
                    float(status.inference_latency_sum_ms), inference_count
                ),
                "inference_max_ms": (
                    None
                    if inference_count == 0
                    else float(status.inference_latency_max_ms)
                ),
                "push_rpc_count": push_rpc_count,
                "push_rpc_latency_sum_ms": float(
                    status.push_rpc_latency_sum_ms
                ),
                "push_rpc_mean_ms": self._raw_mean(
                    float(status.push_rpc_latency_sum_ms), push_rpc_count
                ),
                "push_rpc_max_ms": (
                    None
                    if push_rpc_count == 0
                    else float(status.push_rpc_latency_max_ms)
                ),
                "closed_segment_count": int(status.closed_segment_count),
                "segment_close_counts": segment_close_counts,
                "pending_action_excluded_count": int(
                    status.pending_action_excluded_count
                ),
                "rollout_estimator_failure_count": int(
                    status.rollout_estimator_failure_count
                ),
                "per_agent_model_activation_count": int(
                    status.per_agent_model_activation_count
                ),
                "superseded_without_agent_activation_count": int(
                    status.superseded_without_agent_activation_count
                ),
                "quarantined_transition_count": int(
                    status.quarantined_transition_count
                ),
                "quarantined_envelope_count": int(
                    status.quarantined_envelope_count
                ),
                "error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError, ValueError) as error:
            return self._component_error_snapshot("aiserver", str(error))

    def _sample_pool_snapshot(self) -> dict:
        try:
            status = self._read_status("sample_pool")
            authority = status.sample_pool
            return {
                "ready": bool(status.ready),
                "ingress_ready": bool(status.ingress_ready),
                "pool_ready": bool(status.pool_ready),
                "component": authority.component,
                "instance_id": authority.instance_id,
                "lifecycle_epoch": int(authority.lifecycle_epoch),
                "backend_type": training_pb2.SampleBackendType.Name(
                    status.backend_type
                ),
                "push_attempt_count": int(status.push_attempt_count),
                "accepted": int(status.accepted_unique_transitions),
                "accepted_envelopes": int(
                    status.accepted_unique_envelopes
                ),
                "duplicate_push_attempt_count": int(
                    status.duplicate_push_attempt_count
                ),
                "duplicate_transition_attempts": int(
                    status.duplicate_transition_attempts
                ),
                "rejected_push_attempt_count": int(
                    status.rejected_push_attempt_count
                ),
                "rejected_transition_attempts": int(
                    status.rejected_transition_attempts
                ),
                "acked": int(status.acked_unique_transitions),
                "acked_deliveries": int(status.acked_unique_deliveries),
                "trained": int(status.trained_transition_count),
                "invalid": int(status.invalid_transition_count),
                "shutdown_untrained": int(
                    status.shutdown_untrained_transition_count
                ),
                "finalized": bool(status.finalized),
                "finalization_id": status.finalization_id,
                "finalized_at_unix_ms": (
                    int(status.finalized_at_unix_ms)
                    if self._has_field(status, "finalized_at_unix_ms")
                    else None
                ),
                "finalized_transitions": int(
                    status.finalized_transition_count
                ),
                "ready_transitions": int(status.ready_transitions),
                "leased_transitions": int(status.leased_transitions),
                "resident_transitions": int(status.resident_transitions),
                "resident_envelopes": int(status.resident_envelopes),
                "resident_estimated_bytes": int(
                    status.resident_estimated_bytes
                ),
                "capacity_transitions": int(status.capacity_transitions),
                "capacity_bytes": int(status.capacity_bytes),
                "pressure_state": training_pb2.PressureState.Name(
                    status.pressure_state
                ),
                "evicted_transitions": int(
                    status.evicted_transition_count
                ),
                "evicted_envelopes": int(status.evicted_envelope_count),
                "unsampled_evicted_transitions": int(
                    status.unsampled_evicted_transition_count
                ),
                "previously_drawn_evicted_transitions": int(
                    status.previously_drawn_evicted_transition_count
                ),
                "draw_attempt_count": int(status.draw_attempt_count),
                "drawn_transition_slot_count": int(
                    status.drawn_transition_slot_count
                ),
                "target_hit_count": int(status.target_hit_count),
                "partial_get_count": int(status.partial_get_count),
                "empty_timeout_count": int(status.empty_timeout_count),
                "redelivery_count": int(status.redelivery_count),
                "nack_count": int(status.nack_count),
                "expired_lease_count": int(status.expired_lease_count),
                "lease_renew_count": int(status.lease_renew_count),
                "oldest_ready_transition_age_ms": (
                    int(status.oldest_ready_transition_age_ms)
                    if self._has_field(
                        status, "oldest_ready_transition_age_ms"
                    )
                    else None
                ),
                "minimum_ready_model_step": (
                    int(status.minimum_ready_model_step)
                    if self._has_field(status, "minimum_ready_model_step")
                    else None
                ),
                "maximum_ready_model_step": (
                    int(status.maximum_ready_model_step)
                    if self._has_field(status, "maximum_ready_model_step")
                    else None
                ),
                "last_error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError) as error:
            return self._component_error_snapshot(
                "sample-pool", str(error)
            )

    def _model_snapshot(self) -> dict:
        try:
            status = self._read_status("distributor")
            authority = status.distributor
            available_range = ((int(status.available_floor_model_step), int(status.latest_available_model_step))
                if status.HasField("available_floor_model_step") and status.HasField("latest_available_model_step") else None)
            return {
                "ready": bool(status.ready),
                "component": authority.component,
                "instance_id": authority.instance_id,
                "lifecycle_epoch": int(authority.lifecycle_epoch),
                "registered_model_count": int(status.registered_model_count),
                "latest_model_identity": model_identity_document(
                    status.latest_model
                ),
                "latest_ack_model_identity": model_identity_document(
                    status.latest_ack_model
                ),
                "latest_ack_status": training_pb2.ModelLoadStatus.Name(
                    status.latest_ack_status
                ),
                "latest_ack_aiserver": status.latest_ack_aiserver.instance_id,
                "available_floor_model_step": (
                    None if available_range is None else available_range[0]
                ),
                "latest_available_model_step": (
                    None if available_range is None else available_range[1]
                ),
                "last_error": status.last_error,
                "timestamp": int(status.timestamp_unix_ms) / 1000.0,
            }
        except (grpc.RpcError, RuntimeError, ValueError) as error:
            return self._component_error_snapshot(
                "model-distributor", str(error)
            )

    def _resource_snapshot(self) -> dict:
        now = time.monotonic()
        process_cpu = time.process_time()
        elapsed = now - self._last_resource_time
        cpu_delta = process_cpu - self._last_process_cpu
        cpu = None if elapsed <= 0.0 else cpu_delta / elapsed * 100.0
        self._last_resource_time = now
        self._last_process_cpu = process_cpu
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = float(usage.ru_maxrss) / 1024.0
        if sys.platform == "darwin":
            rss_mb /= 1024.0
        return {
            "cpu_percent": cpu,
            "process_cpu_seconds_delta": cpu_delta,
            "window_seconds": elapsed,
            "memory_mb": rss_mb,
        }

    def _rates(self, actor: dict, sample_pool: dict, timestamp: float) -> dict:
        sources = {
            "produced": actor,
            "accepted": sample_pool,
            "acked": sample_pool,
            "trained": sample_pool,
        }
        elapsed = None if self._last_rate_time is None else timestamp - self._last_rate_time
        self._last_rate_time = timestamp
        result = {"window_seconds": elapsed, "source_intervals": {}}
        unavailable = {}
        for name, source in sources.items():
            if source.get("error") or name not in source:
                unavailable[name] = "missing_counter"
                self._rate_snapshot.pop(name, None)
                continue
            current = {
                "source": (source.get("instance_id"), source.get("lifecycle_epoch")),
                "timestamp": source["timestamp"],
                "value": source[name],
            }
            previous = self._rate_snapshot.get(name)
            self._rate_snapshot[name] = current
            if previous is None:
                unavailable[name] = "initial_snapshot"
                continue
            if previous["source"] != current["source"]:
                unavailable[name] = "source_identity_changed"
                continue
            duration = current["timestamp"] - previous["timestamp"]
            if duration <= 0:
                unavailable[name] = "non_positive_window"
                continue
            delta = current["value"] - previous["value"]
            result[f"{name}_delta"] = delta
            if delta < 0:
                unavailable[name] = "counter_regression"
                continue
            result["source_intervals"][name] = {
                "start": previous["timestamp"], "end": current["timestamp"],
            }
            result[f"{name}_sps"] = delta / duration
        reasons = set(unavailable.values())
        reason = ""
        if unavailable:
            if len(unavailable) < len(sources):
                reason = "partial"
            else:
                reason = next(iter(reasons)) if len(reasons) == 1 else "unavailable"
        result.update(
            available=not unavailable,
            reason=reason,
            missing_counters=[name for name, reason in unavailable.items() if reason == "missing_counter"],
            unavailable_counters=unavailable,
        )
        return result
