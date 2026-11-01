"""Learner-side SamplePool consumption, lease settlement and finalization."""
from __future__ import annotations

import threading
import time
from typing import Callable

import grpc

from proto.common import identity_pb2 as common_pb2
from proto.training import training_pb2, training_pb2_grpc
from src.contracts.identity import has_field, same_service_instance
from src.training.update_commit import MAX_ATTEMPTS, RETRY_DELAY_SEC, UpdateCommitOutcomeUncertain

class _SamplePoolUnavailable(RuntimeError):
    """The current Sample Pool authority is temporarily unable to serve."""



def training_samples(items) -> list[dict]:
    """Project each draw slot, including repeated items, into a PPO input."""
    return [
        {
            "observation": list(item.transition.observation),
            "action": int(item.transition.action),
            "old_log_probability": float(
                item.transition.behavior_log_probability
            ),
            "old_value_prediction": float(item.transition.behavior_value),
            "advantage": float(item.transition.advantage),
            "value_target": float(item.transition.value_target),
            "action_mask": list(item.transition.action_mask),
            "behavior_model_step": int(
                item.transition.behavior_model_step
            ),
            "item_id": item.transition.item_id,
        }
        for item in items
    ]


class LeaseRenewer:
    def __init__(
        self,
        stub,
        consumer: common_pb2.ServiceInstanceIdentity,
        delivery_id: str,
        lease_timeout_ms: int,
        authority_check: Callable[[], None],
    ):
        self.stub = stub
        self.request = training_pb2.RenewLeaseReq(
            consumer=consumer,
            delivery_id=delivery_id,
            lease_timeout_ms=lease_timeout_ms,
        )
        self.authority_check = authority_check
        self._renew_lock = threading.Lock()
        self.interval = max(0.2, lease_timeout_ms / 3000.0)
        self.failure_deadline = max(1.0, lease_timeout_ms / 1000.0 * 0.8)
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.renew_now()
        self._thread.start()
        return self

    def renew_now(self) -> None:
        with self._renew_lock:
            response = self.stub.RenewLease(self.request, timeout=2.0)
            positive_result = response.result in (
                training_pb2.DELIVERY_RESULT_APPLIED,
                training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
            )
            if not positive_result:
                raise RuntimeError(response.message or "lease renewal rejected")
            if response.delivery_id != self.request.delivery_id:
                raise RuntimeError("lease renewal returned another delivery")
            if int(response.lease_deadline_unix_ms) <= int(time.time() * 1000):
                raise RuntimeError("lease renewal returned a non-future deadline")
            self.authority_check()

    def _run(self) -> None:
        last_success = time.monotonic()
        while not self._stop.wait(self.interval):
            try:
                self.renew_now()
                last_success = time.monotonic()
            except grpc.RpcError as error:
                if time.monotonic() - last_success >= self.failure_deadline:
                    self.error = error.details() or str(error)
                    return
            except Exception as error:
                self.error = str(error)
                return

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


class SamplePoolConsumer:
    SAMPLE_RETRY_INITIAL_SEC = 0.05
    SAMPLE_RETRY_MAX_SEC = 1.0
    GET_BATCH_RECONCILE_POLL_SEC = 0.5
    SHUTDOWN_RECONCILE_MARGIN_SEC = 5.0

    def __init__(self, address, settings, consumer, batch_size, stop_event, logger, report_wait):
        self.channel = grpc.insecure_channel(address)
        self.stub = training_pb2_grpc.SamplePoolConsumerServiceStub(self.channel)
        self.learner_service = consumer
        self.train_batch_size = batch_size
        self.get_timeout_ms = int(settings["get_timeout_ms"])
        self.lease_timeout_ms = int(settings["lease_timeout_ms"])
        self.shutdown_drain_timeout_ms = int(settings["shutdown_drain_timeout_ms"])
        self.stop_event = stop_event
        self.logger = logger
        self.report_wait = report_wait
        self.ready_authority = None

    def next_batch(self):
        """Wait for the Pool and return one leased batch, or yield on timeout/stop."""
        self.ready_authority = self._wait_for_sample_pool()
        if self.ready_authority is None:
            return None
        response = self._get_batch_recovering(ready_authority=self.ready_authority)
        if response is None or response.result == training_pb2.GET_BATCH_RESULT_TIMEOUT:
            return None
        if response.result != training_pb2.GET_BATCH_RESULT_LEASED:
            raise RuntimeError(f"sample GetBatch failed: {response.message}")
        return response

    def delivery_provenance(self, response, validate_training_item) -> dict:
        if (
            response.result != training_pb2.GET_BATCH_RESULT_LEASED
            or len(response.items) != self.train_batch_size
            or not response.delivery_id
        ):
            raise ValueError("sample delivery violates exact transition batch size")

        item_snapshots: dict[str, training_pb2.SamplePoolItem] = {}
        behavior_steps: set[int] = set()
        behavior_models: dict[int, dict] = {}
        producers: dict[tuple[str, str, int], dict] = {}
        transition_created_at: list[int] = []
        inserted_at: list[int] = []
        draw_counts: list[int] = []
        for item in response.items:
            transition = item.transition
            identity = validate_training_item(item)
            if (
                int(item.insert_sequence) <= 0
                or int(item.inserted_at_unix_ms) <= 0
                or int(item.draw_count) <= 0
            ):
                raise ValueError("sample pool item storage facts are invalid")
            previous = item_snapshots.get(transition.item_id)
            if previous is not None and previous != item:
                raise ValueError("repeated sample item has conflicting snapshot facts")
            item_snapshots[transition.item_id] = item
            step = int(transition.behavior_model_step)
            behavior_steps.add(step)
            behavior_models[step] = identity
            producer = item.producer
            if not producer.component or not producer.instance_id or not producer.lifecycle_epoch:
                raise ValueError(f"processed transition producer is invalid: item={transition.item_id}")
            key = (producer.component, producer.instance_id, producer.lifecycle_epoch)
            producers[key] = {"component": key[0], "instance_id": key[1], "lifecycle_epoch": key[2]}
            transition_created_at.append(int(transition.created_at_unix_ms))
            inserted_at.append(int(item.inserted_at_unix_ms))
            draw_counts.append(int(item.draw_count))

        minimum_step = min(behavior_steps)
        maximum_step = max(behavior_steps)
        oldest_created_at = min(transition_created_at)
        newest_created_at = max(transition_created_at)
        return {
            "model_lineage_id": behavior_models[minimum_step]["model_lineage_id"],
            "minimum_model_step": minimum_step,
            "maximum_model_step": maximum_step,
            "models": [behavior_models[step] for step in sorted(behavior_steps)],
            "producers": [producers[key] for key in sorted(producers)],
            "oldest_transition_created_at_unix_ms": oldest_created_at,
            "newest_transition_created_at_unix_ms": newest_created_at,
            "oldest_inserted_at_unix_ms": min(inserted_at),
            "newest_inserted_at_unix_ms": max(inserted_at),
            "draw_count_sum": sum(draw_counts),
            "draw_count_minimum": min(draw_counts),
            "draw_count_maximum": max(draw_counts),
        }

    def renew_during_update(self, delivery):
        return LeaseRenewer(
            self.stub, self.learner_service, delivery.delivery_id, self.lease_timeout_ms,
            lambda: self._assert_lease_authority(delivery.sample_pool),
        ).start()

    def acknowledge_trained(self, delivery, update_id):
        return self._ack_idempotently(
            delivery.delivery_id, training_pb2.ACK_DISPOSITION_TRAINED,
            update_id, delivery.sample_pool,
        )

    def acknowledge_invalid(self, delivery):
        self._ack(delivery.delivery_id, training_pb2.ACK_DISPOSITION_INVALID,
                  expected_authority=delivery.sample_pool)

    def close(self):
        self.channel.close()

    @staticmethod
    def _sample_pool_authority(
        authority: common_pb2.ServiceInstanceIdentity,
    ) -> common_pb2.ServiceInstanceIdentity:
        try:
            valid = (
                bool(authority.component)
                and bool(authority.instance_id)
                and int(authority.lifecycle_epoch) > 0
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            raise RuntimeError("sample pool authority is invalid")
        result = common_pb2.ServiceInstanceIdentity()
        try:
            result.CopyFrom(authority)
        except TypeError as error:
            raise RuntimeError(
                "sample pool authority has an invalid wire type"
            ) from error
        return result

    def _sample_pool_status(self):
        return self.stub.GetStatus(
            training_pb2.SamplePoolStatusReq(), timeout=2.0
        )

    @staticmethod
    def _is_retryable_sample_rpc(error: grpc.RpcError) -> bool:
        try:
            code = error.code()
        except Exception:
            return False
        return code in (
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.CANCELLED,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.UNKNOWN,
            grpc.StatusCode.UNAVAILABLE,
        )

    @staticmethod
    def _rpc_error_text(error: grpc.RpcError) -> str:
        try:
            details = error.details()
        except Exception:
            details = ""
        return details or str(error)


    def _mark_sample_wait(self, disposition, operation, error, attempt):
        message = f"{operation}: {error}"
        self.report_wait(disposition, message)
        if attempt == 1 or (attempt & (attempt - 1)) == 0:
            self.logger.warning("%s; waiting for the same Sample Pool authority (attempt=%d)", message, attempt)

    def _clear_sample_wait(self):
        self.report_wait("", "")

    def _wait_sample_retry(
        self,
        attempt: int,
        deadline: float | None = None,
        *,
        ignore_stop: bool = False,
    ) -> bool:
        delay = min(
            self.SAMPLE_RETRY_MAX_SEC,
            self.SAMPLE_RETRY_INITIAL_SEC * (2 ** min(attempt - 1, 8)),
        )
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
            if delay <= 0:
                return True
        if ignore_stop:
            time.sleep(delay)
            return False
        return self.stop_event.wait(delay)

    def _ready_sample_pool_authority(
        self, status
    ) -> common_pb2.ServiceInstanceIdentity:
        authority = self._sample_pool_authority(status.sample_pool)
        if not status.ready or not status.ingress_ready or status.finalized:
            raise _SamplePoolUnavailable(
                "sample pool service is not accepting samples"
            )
        return authority

    def _sample_status_for_authority(
        self,
        expected: common_pb2.ServiceInstanceIdentity,
        operation: str,
    ):
        status = self._sample_pool_status()
        actual = self._ready_sample_pool_authority(status)
        if not same_service_instance(actual, expected):
            raise RuntimeError(
                f"sample pool authority changed during {operation}"
            )
        return status

    def _assert_lease_authority(
        self, expected: common_pb2.ServiceInstanceIdentity
    ) -> None:
        self._sample_status_for_authority(expected, "the lease")

    def _assert_sample_pool_ready(
        self,
    ) -> common_pb2.ServiceInstanceIdentity:
        return self._ready_sample_pool_authority(self._sample_pool_status())

    def _wait_for_sample_pool(
        self,
    ) -> common_pb2.ServiceInstanceIdentity | None:
        attempt = 0
        while not self.stop_event.is_set():
            try:
                authority = self._assert_sample_pool_ready()
                self._clear_sample_wait()
                return authority
            except _SamplePoolUnavailable as error:
                attempt += 1
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "sample pool readiness",
                    error,
                    attempt,
                )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                attempt += 1
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "sample pool transport",
                    self._rpc_error_text(error),
                    attempt,
                )
            if self._wait_sample_retry(attempt):
                return None
        return None

    def _get_batch_recovery_status(
        self, expected: common_pb2.ServiceInstanceIdentity
    ):
        status = self._sample_pool_status()
        actual = self._sample_pool_authority(status.sample_pool)
        if not same_service_instance(actual, expected):
            raise RuntimeError(
                "sample pool authority changed while GetBatch outcome "
                "was unknown"
            )
        if int(status.max_concurrent_consumers) != 1:
            raise RuntimeError(
                "GetBatch recovery requires the declared single-consumer "
                "delivery contract"
            )
        leased_transitions = int(status.leased_transitions)
        active_consumers = int(status.active_consumer_count)
        if (
            leased_transitions < 0
            or active_consumers < 0
            or (leased_transitions == 0 and active_consumers != 0)
            or (leased_transitions > 0 and active_consumers != 1)
        ):
            raise RuntimeError(
                "sample pool returned contradictory lease status while "
                "GetBatch outcome was unknown"
            )
        return status

    def _reconcile_get_batch_outcome(
        self,
        expected: common_pb2.ServiceInstanceIdentity,
        reason: str,
        deadline: float | None = None,
        *,
        ignore_stop: bool = False,
        request_deadline: float | None = None,
    ) -> bool:
        expected = self._sample_pool_authority(expected)
        attempt = 0
        while (ignore_stop or not self.stop_event.is_set()) and (
            deadline is None or time.monotonic() < deadline
        ):
            # A transport failure can arrive before cancellation reaches Pool.
            # Wait out this exact RPC's deadline before observing lease state;
            # repeated zero-lease snapshots cannot prove the request has ended.
            if request_deadline is not None and time.monotonic() < request_deadline:
                wait_until = request_deadline if deadline is None else min(request_deadline, deadline)
                delay = max(0.0, wait_until - time.monotonic())
                if ignore_stop:
                    time.sleep(delay)
                elif self.stop_event.wait(delay):
                    return False
                continue
            attempt += 1
            try:
                status = self._get_batch_recovery_status(expected)
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "GET_BATCH_OUTCOME_UNKNOWN",
                    "GetBatch reconciliation transport",
                    self._rpc_error_text(error),
                    attempt,
                )
                if self._wait_sample_retry(
                    attempt, deadline, ignore_stop=ignore_stop
                ):
                    return False
                continue

            if not status.ready or int(status.leased_transitions) > 0:
                state = (
                    "hidden lease transitions="
                    f"{int(status.leased_transitions)}"
                    if int(status.leased_transitions) > 0
                    else "sample pool service not ready"
                )
                self._mark_sample_wait(
                    "GET_BATCH_OUTCOME_UNKNOWN",
                    "GetBatch reconciliation",
                    state,
                    attempt,
                )
            else:
                # Pool checks cancellation/deadline before committing a lease;
                # GetStatus observes the same lease state under the same lock.
                self._clear_sample_wait()
                self.logger.info(
                    "GetBatch outcome reconciled without training: %s", reason
                )
                return True
            poll_delay = self.GET_BATCH_RECONCILE_POLL_SEC
            if deadline is not None:
                poll_delay = min(
                    poll_delay, max(0.0, deadline - time.monotonic())
                )
                if poll_delay <= 0:
                    return False
            if ignore_stop:
                time.sleep(poll_delay)
            elif self.stop_event.wait(poll_delay):
                return False
        return False

    def _get_batch(
        self,
        ready_authority: common_pb2.ServiceInstanceIdentity | None = None,
        *,
        request_deadline: float | None = None,
    ):
        if request_deadline is None:
            request_deadline = time.monotonic() + max(
                2.0, self.get_timeout_ms / 1000.0 + 1.0
            )
        response = self.stub.GetBatch(
            training_pb2.GetBatchReq(
                requested_transitions=self.train_batch_size,
                timeout_ms=self.get_timeout_ms,
                consumer=self.learner_service,
                lease_timeout_ms=self.lease_timeout_ms,
            ),
            timeout=max(0.0, request_deadline - time.monotonic()),
        )
        if response.result == training_pb2.GET_BATCH_RESULT_BUSY:
            response_authority = self._sample_pool_authority(
                response.sample_pool
            )
            if ready_authority is None:
                ready_authority = self._ready_sample_pool_authority(
                    self._sample_pool_status()
                )
            else:
                ready_authority = self._sample_pool_authority(
                    ready_authority
                )
            if (
                not same_service_instance(
                    response_authority, ready_authority
                )
                or bool(response.delivery_id)
                or int(response.lease_deadline_unix_ms) != 0
                or len(response.items) != 0
            ):
                raise RuntimeError(
                    "sample pool returned an incoherent busy delivery"
                )
            return response
        if response.result != training_pb2.GET_BATCH_RESULT_LEASED:
            response_authority = self._sample_pool_authority(
                response.sample_pool
            )
            if ready_authority is not None and not same_service_instance(
                response_authority,
                self._sample_pool_authority(ready_authority),
            ):
                raise RuntimeError(
                    "sample pool authority changed during GetBatch"
                )
            if (
                response.delivery_id
                or int(response.lease_deadline_unix_ms) != 0
                or response.items
            ):
                raise RuntimeError(
                    "sample pool returned a non-leased result with lease payload"
                )
            return response
        if ready_authority is None:
            ready_authority = self._ready_sample_pool_authority(
                self._sample_pool_status()
            )
        else:
            ready_authority = self._sample_pool_authority(
                ready_authority
            )
        response_authority = self._sample_pool_authority(
            response.sample_pool
        )
        if (
            not same_service_instance(
                response_authority, ready_authority
            )
            or not response.delivery_id
            or len(response.items) != self.train_batch_size
            or int(response.lease_deadline_unix_ms)
            <= int(time.time() * 1000)
        ):
            raise RuntimeError(
                "sample pool returned an incoherent leased delivery"
            )
        return response

    def _get_batch_recovering(
        self,
        *,
        ready_authority: common_pb2.ServiceInstanceIdentity,
        deadline: float | None = None,
        ignore_stop: bool = False,
    ):
        expected = self._sample_pool_authority(ready_authority)
        request_deadline = time.monotonic() + max(
            2.0, self.get_timeout_ms / 1000.0 + 1.0
        )
        try:
            response = self._get_batch(
                ready_authority=expected, request_deadline=request_deadline
            )
        except grpc.RpcError as error:
            if not self._is_retryable_sample_rpc(error):
                raise
            reason = self._rpc_error_text(error)
            self._mark_sample_wait(
                "GET_BATCH_OUTCOME_UNKNOWN",
                "GetBatch transport outcome unknown",
                reason,
                1,
            )
            self._reconcile_get_batch_outcome(
                expected,
                reason,
                deadline=deadline,
                ignore_stop=ignore_stop,
                request_deadline=request_deadline,
            )
            return None
        if response.result == training_pb2.GET_BATCH_RESULT_BUSY:
            reason = response.message or "Sample Pool reports an active lease"
            self._mark_sample_wait(
                "GET_BATCH_OUTCOME_UNKNOWN",
                "GetBatch busy recovery",
                reason,
                1,
            )
            self._reconcile_get_batch_outcome(
                expected,
                reason,
                deadline=deadline,
                ignore_stop=ignore_stop,
            )
            return None
        return response


    def _ack(
        self,
        delivery_id: str,
        disposition: int,
        train_update_id: str = "",
        expected_authority: common_pb2.ServiceInstanceIdentity | None = None,
    ) -> None:
        response = self.stub.AckBatch(
            training_pb2.AckBatchReq(
                consumer=self.learner_service,
                delivery_id=delivery_id,
                disposition=disposition,
                train_update_id=train_update_id,
            ),
            timeout=3.0,
        )
        positive_result = response.result in (
            training_pb2.DELIVERY_RESULT_APPLIED,
            training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
        )
        if not positive_result:
            if response.delivery_id != delivery_id:
                raise UpdateCommitOutcomeUncertain(
                    "sample ACK negative response does not identify the "
                    "requested delivery"
                )
            if expected_authority is None:
                raise UpdateCommitOutcomeUncertain(
                    "sample ACK negative response has no pinned lease authority"
                )
            try:
                self._assert_lease_authority(expected_authority)
            except (grpc.RpcError, RuntimeError) as error:
                raise UpdateCommitOutcomeUncertain(
                    "sample ACK negative response cannot be bound to the "
                    "pinned lease authority"
                ) from error
            raise RuntimeError(f"sample ACK failed: {response.message}")
        if (
            response.delivery_id != delivery_id
            or int(response.disposition) != int(disposition)
            or response.train_update_id != train_update_id
        ):
            raise UpdateCommitOutcomeUncertain(
                "sample ACK positive response does not echo the exact request"
            )

    def _ack_idempotently(
        self,
        delivery_id: str,
        disposition: int,
        train_update_id: str,
        expected_authority: common_pb2.ServiceInstanceIdentity,
    ) -> int:
        last_error: Exception | None = None
        saw_uncertain_outcome = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self._ack(
                    delivery_id,
                    disposition,
                    train_update_id,
                    expected_authority,
                )
                return attempt
            except Exception as error:
                last_error = error
                saw_uncertain_outcome = saw_uncertain_outcome or isinstance(
                    error, (grpc.RpcError, UpdateCommitOutcomeUncertain)
                )
                if attempt == MAX_ATTEMPTS:
                    break
                self.logger.warning(
                    "sample ACK attempt %d failed; retrying exact delivery/update: %s",
                    attempt,
                    error,
                )
                time.sleep(RETRY_DELAY_SEC)
        if not saw_uncertain_outcome:
            raise RuntimeError(
                "sample ACK was not applied after "
                f"{MAX_ATTEMPTS} attempts: {last_error}"
            ) from last_error
        raise UpdateCommitOutcomeUncertain(
            "sample ACK outcome is uncertain after "
            f"{MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def release(self, delivery_id: str, reason: str) -> None:
        try:
            response = self.stub.NackBatch(
                training_pb2.NackBatchReq(
                    consumer=self.learner_service,
                    delivery_id=delivery_id,
                    reason=reason[:256],
                ),
                timeout=3.0,
            )
            if response.result not in (
                training_pb2.DELIVERY_RESULT_APPLIED,
                training_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
            ):
                self.logger.error("sample NACK rejected: %s", response.message)
        except grpc.RpcError as error:
            self.logger.error("sample NACK failed: %s", error)


    def _sample_pool_finalization_id(self) -> str:
        identity = (
            f"{self.learner_service.component}:"
            f"{self.learner_service.instance_id}:"
            f"{int(self.learner_service.lifecycle_epoch)}"
        )
        return f"learner-shutdown-{identity.encode('utf-8').hex()}"

    def _validate_finalized_sample_pool_status(
        self,
        status,
        expected_authority: common_pb2.ServiceInstanceIdentity,
        finalization_id: str,
        response,
    ) -> None:
        actual = self._sample_pool_authority(status.sample_pool)
        if not same_service_instance(actual, expected_authority):
            raise RuntimeError(
                "sample pool authority changed after finalization"
            )
        if (
            not status.finalized
            or status.finalization_id != finalization_id
            or not has_field(status, "finalized_at_unix_ms")
            or not has_field(response, "finalized_at_unix_ms")
            or int(status.finalized_at_unix_ms) <= 0
            or int(status.finalized_at_unix_ms)
            != int(response.finalized_at_unix_ms)
        ):
            raise RuntimeError(
                "sample pool returned contradictory finalization identity"
            )
        counts = {
            "accepted": int(status.accepted_unique_transitions),
            "acked": int(status.acknowledged_transition_slot_count),
            "released": int(status.released_transition_slot_count),
            "drawn": int(status.drawn_transition_slot_count),
            "trained": int(status.trained_transition_count),
            "invalid": int(status.invalid_transition_count),
            "shutdown_untrained": int(
                status.shutdown_untrained_transition_count
            ),
            "ready": int(status.ready_transitions),
            "leased": int(status.leased_transitions),
            "resident": int(status.resident_transitions),
            "resident_bytes": int(status.resident_estimated_bytes),
            "evicted": int(status.evicted_transition_count),
            "finalized": int(status.finalized_transition_count),
        }
        if min(counts.values()) < 0:
            raise RuntimeError(
                "sample pool returned negative finalized accounting"
            )
        if (
            status.ingress_ready
            or status.pool_ready
            or counts["ready"] != 0
            or counts["leased"] != 0
            or counts["resident"] != 0
            or counts["resident_bytes"] != 0
        ):
            raise RuntimeError(
                "sample pool retained live data after finalization"
            )
        if counts["acked"] != (
            counts["trained"] + counts["invalid"] + counts["shutdown_untrained"]
        ):
            raise RuntimeError(
                "sample pool returned contradictory finalized disposition "
                "accounting"
            )
        if counts["drawn"] != counts["acked"] + counts["released"]:
            raise RuntimeError(
                "sample pool returned contradictory finalized draw-slot accounting"
            )
        if counts["accepted"] != counts["evicted"] + counts["finalized"]:
            raise RuntimeError(
                "sample pool returned contradictory finalized transition "
                "accounting"
            )

    def _finalize_sample_pool(
        self,
        expected_authority: common_pb2.ServiceInstanceIdentity,
        deadline: float,
    ) -> None:
        expected = self._sample_pool_authority(expected_authority)
        finalization_id = self._sample_pool_finalization_id()
        request = training_pb2.FinalizeSamplePoolReq(
            consumer=self.learner_service,
            expected_sample_pool=expected,
            finalization_id=finalization_id,
        )
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            timeout = min(2.0, max(0.001, deadline - time.monotonic()))
            try:
                response = self.stub.FinalizeSamplePool(
                    request, timeout=timeout
                )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown SamplePool finalization",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue

            actual = self._sample_pool_authority(response.sample_pool)
            if not same_service_instance(actual, expected):
                raise RuntimeError(
                    "sample pool authority changed during finalization"
                )
            if response.finalization_id != finalization_id:
                raise RuntimeError(
                    "sample pool returned another finalization identity"
                )
            if (
                response.result
                == training_pb2.SAMPLE_POOL_FINALIZE_RESULT_REJECTED_ACTIVE_LEASE
            ):
                if not self._reconcile_get_batch_outcome(
                    expected,
                    "SamplePool finalization observed an active delivery",
                    deadline=deadline,
                    ignore_stop=True,
                ):
                    break
                continue

            if response.result not in (
                training_pb2.SAMPLE_POOL_FINALIZE_RESULT_FINALIZED,
                training_pb2.SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED,
            ):
                raise RuntimeError(
                    "shutdown SamplePool finalization failed: "
                    f"{response.message}"
                )
            if (
                not has_field(response, "finalized_at_unix_ms")
                or int(response.finalized_at_unix_ms) <= 0
            ):
                raise RuntimeError(
                    "sample pool returned a non-empty finalization result"
                )
            try:
                status = self._sample_pool_status()
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "finalized SamplePool status",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue
            self._validate_finalized_sample_pool_status(
                status, expected, finalization_id, response
            )
            self._clear_sample_wait()
            self.logger.info(
                "SamplePool finalized: id=%s cleared_resident_transitions=%d",
                finalization_id,
                int(status.finalized_transition_count),
            )
            return
        raise RuntimeError(
            "shutdown SamplePool finalization did not settle before timeout"
        )

    def finalize(self) -> None:
        minimum_reconcile_sec = (
            self.lease_timeout_ms / 1000.0
            + self.SHUTDOWN_RECONCILE_MARGIN_SEC
        )
        budget_sec = max(
            self.shutdown_drain_timeout_ms / 1000.0,
            minimum_reconcile_sec,
        )
        deadline = time.monotonic() + budget_sec
        pinned = self.ready_authority
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                status = self._sample_pool_status()
                authority = self._sample_pool_authority(
                    status.sample_pool
                )
                if pinned is None:
                    pinned = authority
                elif not same_service_instance(authority, pinned):
                    raise RuntimeError(
                        "sample pool authority changed during shutdown"
                    )
            except grpc.RpcError as error:
                if not self._is_retryable_sample_rpc(error):
                    raise
                self._mark_sample_wait(
                    "WAITING_FOR_SAMPLE_POOL",
                    "shutdown sample status",
                    self._rpc_error_text(error),
                    attempt,
                )
                self._wait_sample_retry(
                    attempt, deadline, ignore_stop=True
                )
                continue

            ready_transitions = int(status.ready_transitions)
            leased_transitions = int(status.leased_transitions)
            if min(ready_transitions, leased_transitions) < 0:
                raise RuntimeError(
                    "sample pool returned contradictory shutdown counts"
                )
            if leased_transitions > 0:
                reconciled = self._reconcile_get_batch_outcome(
                    pinned,
                    "shutdown observed an unresolved delivery",
                    deadline=deadline,
                    ignore_stop=True,
                )
                if not reconciled:
                    break
                continue
            self._finalize_sample_pool(pinned, deadline)
            return
        raise RuntimeError(
            "shutdown SamplePool lease reconciliation did not settle before "
            "timeout"
        )
