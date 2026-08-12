import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import grpc
import yaml

from main.training_runtime import TrainingRuntime
from proto import training_pb2
from src.contracts.identity import (
    contract_identity,
    policy_spec_digest,
    service_identity,
    training_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )


class RpcFault(grpc.RpcError):
    def __init__(self, code, details):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class SampleTransportRecoveryTest(unittest.TestCase):
    def runtime(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"]
        )
        runtime.trainer = SimpleNamespace(
            model_version=17,
            max_policy_lag=2,
            train_on_batch=mock.Mock(),
        )
        runtime._effective_max_policy_lag = lambda: 2
        runtime.learner_service = service_identity(
            "learner", "learner-transport-test", 1
        )
        runtime.train_batch_size = 2
        runtime.max_train_batch_size = 3
        runtime.max_sample_age_ms = 120000
        runtime.get_timeout_ms = 1000
        runtime.lease_timeout_ms = 30000
        runtime.train_updates = 9
        runtime.trained_samples = 4608
        runtime._metrics_lock = threading.RLock()
        runtime._metrics_context = {
            "disposition": "TRAINED",
            "error": "",
        }
        runtime.logger = SimpleNamespace(
            warning=mock.Mock(),
            info=mock.Mock(),
            error=mock.Mock(),
            exception=mock.Mock(),
        )
        runtime.SAMPLE_RETRY_INITIAL_SEC = 0.001
        runtime.SAMPLE_RETRY_MAX_SEC = 0.001
        runtime.GET_BATCH_RECONCILE_POLL_SEC = 0.001
        runtime.GET_BATCH_RECONCILE_STABLE_WINDOW_SEC = 0.002
        runtime.GET_BATCH_RECONCILE_CONFIRMATIONS = 2
        runtime.DEMAND_RELEASE_RETRY_TIMEOUT_SEC = 0.01
        runtime.SHUTDOWN_RECONCILE_MARGIN_SEC = 0.005
        return runtime

    @staticmethod
    def authority(instance_id="sample-distributor-transport-test", epoch=1):
        return service_identity("sample-distributor", instance_id, epoch)

    def status(self, runtime, authority, *, leased_samples=0, ready=True):
        return training_pb2.DistributorStatusRsp(
            contract=runtime.contract,
            distributor=authority,
            ready=ready,
            ingress_ready=ready,
            pool_ready=ready,
            leased_samples=leased_samples,
            leased_fragments=1 if leased_samples else 0,
            max_concurrent_consumers=1,
            active_consumer_count=1 if leased_samples else 0,
            timestamp_unix_ms=int(time.time() * 1000),
        )

    @staticmethod
    def sequence_then_last(items):
        values = iter(items)
        last = items[-1]

        def next_value(*_args, **_kwargs):
            nonlocal last
            try:
                last = next(values)
            except StopIteration:
                pass
            return last

        return next_value

    def test_deadline_with_hidden_lease_waits_for_expiry_before_retry(self):
        runtime = self.runtime()
        authority = self.authority()
        get_status = mock.Mock(
            side_effect=self.sequence_then_last(
                [
                    self.status(runtime, authority, leased_samples=2),
                    self.status(runtime, authority),
                    self.status(runtime, authority),
                    self.status(runtime, authority),
                ]
            )
        )
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected GetBatch response loss",
                )
            ),
            GetStatus=get_status,
        )
        progress = (
            runtime.train_updates,
            runtime.trained_samples,
            runtime.trainer.model_version,
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            response = runtime._get_batch_recovering(
                ready_authority=authority
            )

        self.assertIsNone(response)
        self.assertGreaterEqual(get_status.call_count, 4)
        self.assertEqual(
            (
                runtime.train_updates,
                runtime.trained_samples,
                runtime.trainer.model_version,
            ),
            progress,
        )
        runtime.trainer.train_on_batch.assert_not_called()
        self.assertEqual(runtime._metrics_context["disposition"], "TRAINED")
        self.assertEqual(runtime._metrics_context["error"], "")

    def test_deadline_without_lease_requires_stable_zero_confirmations(self):
        runtime = self.runtime()
        authority = self.authority()
        get_status = mock.Mock(
            return_value=self.status(runtime, authority)
        )
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected pre-forward timeout",
                )
            ),
            GetStatus=get_status,
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            self.assertIsNone(
                runtime._get_batch_recovering(ready_authority=authority)
            )

        self.assertGreaterEqual(
            get_status.call_count,
            runtime.GET_BATCH_RECONCILE_CONFIRMATIONS,
        )
        runtime.trainer.train_on_batch.assert_not_called()

    def test_reconciliation_survives_transient_status_failure(self):
        runtime = self.runtime()
        authority = self.authority()
        stable = self.status(runtime, authority)
        get_status = mock.Mock(
            side_effect=self.sequence_then_last(
                [
                    RpcFault(
                        grpc.StatusCode.UNAVAILABLE,
                        "injected status outage",
                    ),
                    stable,
                    stable,
                    stable,
                ]
            )
        )

        def status_or_fault(*args, **kwargs):
            result = get_status(*args, **kwargs)
            if isinstance(result, Exception):
                raise result
            return result

        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected GetBatch timeout",
                )
            ),
            GetStatus=status_or_fault,
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            self.assertIsNone(
                runtime._get_batch_recovering(ready_authority=authority)
            )

        self.assertGreaterEqual(get_status.call_count, 3)
        runtime.trainer.train_on_batch.assert_not_called()

    def test_ambiguous_internal_and_unknown_get_batch_reconcile(self):
        for code in (
            grpc.StatusCode.INTERNAL,
            grpc.StatusCode.UNKNOWN,
        ):
            with self.subTest(code=code):
                runtime = self.runtime()
                authority = self.authority()
                stable = self.status(runtime, authority)
                runtime.sample_stub = SimpleNamespace(
                    GetBatch=mock.Mock(
                        side_effect=RpcFault(
                            code,
                            "injected post-commit transport ambiguity",
                        )
                    ),
                    GetStatus=mock.Mock(return_value=stable),
                )

                with mock.patch(
                    "main.training_runtime._stop_requested",
                    threading.Event(),
                ):
                    self.assertIsNone(
                        runtime._get_batch_recovering(
                            ready_authority=authority
                        )
                    )

                runtime.trainer.train_on_batch.assert_not_called()
                self.assertEqual(
                    runtime._metrics_context["disposition"], "TRAINED"
                )

    def test_reconciliation_fails_closed_on_authority_change(self):
        runtime = self.runtime()
        expected = self.authority()
        changed = self.authority("sample-distributor-restarted", 2)
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected GetBatch timeout",
                )
            ),
            GetStatus=mock.Mock(
                return_value=self.status(runtime, changed)
            ),
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            with self.assertRaisesRegex(RuntimeError, "authority changed"):
                runtime._get_batch_recovering(ready_authority=expected)

        runtime.trainer.train_on_batch.assert_not_called()

    def test_active_demand_readiness_fails_closed_on_authority_change(self):
        runtime = self.runtime()
        expected = self.authority()
        changed = self.authority("sample-distributor-restarted", 2)
        runtime._demand_active = True
        runtime._demand_authority = expected
        runtime._demand_epoch = 18
        status = self.status(runtime, changed)
        status.active_demand_count = 1
        status.active_demand_epoch = 18
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(return_value=status)
        )

        with self.assertRaisesRegex(RuntimeError, "authority changed"):
            runtime._assert_sample_pool_ready()

    def test_active_demand_refresh_does_not_repin_a_new_authority(self):
        runtime = self.runtime()
        expected = self.authority()
        changed = self.authority("sample-distributor-restarted", 2)
        runtime._demand_active = True
        runtime._demand_authority = expected
        runtime._demand_epoch = 18
        runtime._last_demand_refresh = 0.0
        runtime._demand_message = mock.Mock(
            return_value=training_pb2.SampleDemand(demand_epoch=18)
        )
        upsert = mock.Mock()
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(
                return_value=self.status(runtime, changed)
            ),
            UpsertSampleDemand=upsert,
        )

        with self.assertRaisesRegex(RuntimeError, "authority changed"):
            runtime._upsert_demand(force=True)

        upsert.assert_not_called()

    def test_hidden_lease_status_requires_at_least_one_fragment(self):
        runtime = self.runtime()
        authority = self.authority()
        status = self.status(runtime, authority, leased_samples=2)
        status.leased_fragments = 0
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(return_value=status)
        )

        with self.assertRaisesRegex(RuntimeError, "contradictory lease status"):
            runtime._get_batch_recovery_status(authority)

    def test_busy_delivery_enters_the_same_hidden_lease_recovery(self):
        runtime = self.runtime()
        authority = self.authority()
        busy = training_pb2.GetBatchRsp(
            ret_code=2,
            result=training_pb2.GET_BATCH_RESULT_BUSY,
            message="another delivery is still leased",
            leased_samples=2,
            distributor=authority,
        )
        stable = self.status(runtime, authority)
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(return_value=busy),
            GetStatus=mock.Mock(
                side_effect=self.sequence_then_last(
                    [
                        self.status(runtime, authority, leased_samples=2),
                        stable,
                        stable,
                        stable,
                    ]
                )
            ),
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            self.assertIsNone(
                runtime._get_batch_recovering(ready_authority=authority)
            )

        runtime.trainer.train_on_batch.assert_not_called()

    def test_non_retryable_get_batch_transport_error_is_terminal(self):
        runtime = self.runtime()
        authority = self.authority()
        get_status = mock.Mock()
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "injected terminal transport status",
                )
            ),
            GetStatus=get_status,
        )

        with self.assertRaises(RpcFault):
            runtime._get_batch_recovering(ready_authority=authority)

        get_status.assert_not_called()
        runtime.trainer.train_on_batch.assert_not_called()

    def test_preflight_and_upsert_transport_faults_wait_then_recover(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime._upsert_demand = mock.Mock(
            side_effect=[
                RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected demand upsert timeout",
                ),
                None,
                None,
            ]
        )
        runtime._assert_sample_pool_ready = mock.Mock(
            side_effect=[
                RpcFault(
                    grpc.StatusCode.UNAVAILABLE,
                    "injected readiness outage",
                ),
                authority,
            ]
        )

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            actual = runtime._wait_for_sample_pool(force_demand=True)

        self.assertTrue(TrainingRuntime._same_authority(actual, authority))
        self.assertEqual(runtime._upsert_demand.call_count, 3)
        self.assertEqual(runtime._assert_sample_pool_ready.call_count, 2)
        self.assertEqual(runtime._metrics_context["disposition"], "TRAINED")

    def test_startup_failure_uses_standard_metrics_and_channel_cleanup(self):
        runtime = self.runtime()
        runtime._start_metrics = mock.Mock()
        runtime._record_metrics = mock.Mock()
        runtime._initialize_models = mock.Mock(
            side_effect=RuntimeError("injected startup failure")
        )
        runtime._release_demand = mock.Mock()
        runtime._stop_metrics = mock.Mock()
        runtime.metrics_backend = SimpleNamespace(close=mock.Mock())
        runtime.actor_channel = SimpleNamespace(close=mock.Mock())
        runtime.model_channel = SimpleNamespace(close=mock.Mock())
        runtime.sample_channel = SimpleNamespace(close=mock.Mock())

        with mock.patch(
            "main.training_runtime._stop_requested", threading.Event()
        ):
            result = runtime.run()

        self.assertEqual(result, 1)
        runtime._start_metrics.assert_called_once()
        runtime._record_metrics.assert_called()
        runtime._release_demand.assert_called_once_with(required=False)
        runtime._stop_metrics.assert_called_once()
        runtime.metrics_backend.close.assert_called_once()
        runtime.actor_channel.close.assert_called_once()
        runtime.model_channel.close.assert_called_once()
        runtime.sample_channel.close.assert_called_once()

    def test_shutdown_with_stop_set_waits_out_a_hidden_lease(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 20
        runtime.lease_timeout_ms = 1
        runtime._ack = mock.Mock()
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(
                side_effect=self.sequence_then_last(
                    [
                        self.status(runtime, authority, leased_samples=2),
                        self.status(runtime, authority, leased_samples=2),
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                    ]
                )
            )
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            runtime._shutdown_drain(authority)

        runtime._ack.assert_not_called()
        self.assertEqual(runtime._metrics_context["error"], "")

    def test_shutdown_authority_comes_from_the_active_demand(self):
        runtime = self.runtime()
        pinned = self.authority()
        runtime._demand_active = True
        runtime._demand_authority = pinned

        selected = runtime._shutdown_sample_authority(None)

        self.assertTrue(runtime._same_authority(selected, pinned))

    def test_shutdown_fails_closed_on_authority_change(self):
        runtime = self.runtime()
        expected = self.authority()
        changed = self.authority("sample-distributor-restarted", 2)
        runtime.shutdown_drain_timeout_ms = 5
        runtime.lease_timeout_ms = 1
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(
                return_value=self.status(runtime, changed)
            )
        )

        with self.assertRaisesRegex(RuntimeError, "authority changed"):
            runtime._shutdown_drain(expected)

    def test_shutdown_busy_reconciles_without_switching_operation(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 30
        runtime.lease_timeout_ms = 1
        runtime._ack = mock.Mock()
        ready = self.status(runtime, authority)
        ready.ready_queue_samples = 2
        busy = training_pb2.GetBatchRsp(
            ret_code=2,
            result=training_pb2.GET_BATCH_RESULT_BUSY,
            message="response from the original lease is unresolved",
            leased_samples=2,
            distributor=authority,
        )
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(return_value=busy),
            GetStatus=mock.Mock(
                side_effect=self.sequence_then_last(
                    [
                        ready,
                        self.status(runtime, authority, leased_samples=2),
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                    ]
                )
            ),
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            runtime._shutdown_drain(authority)

        runtime.sample_stub.GetBatch.assert_called_once()
        runtime._ack.assert_not_called()

    def test_shutdown_transport_unknown_reconciles_with_stop_set(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 30
        runtime.lease_timeout_ms = 1
        runtime._ack = mock.Mock()
        ready = self.status(runtime, authority)
        ready.ready_queue_samples = 2
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected shutdown response loss",
                )
            ),
            GetStatus=mock.Mock(
                side_effect=self.sequence_then_last(
                    [
                        ready,
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                        self.status(runtime, authority),
                    ]
                )
            ),
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            runtime._shutdown_drain(authority)

        runtime.sample_stub.GetBatch.assert_called_once()
        runtime._ack.assert_not_called()

    def test_shutdown_unsettled_reservation_is_a_failure(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 5
        runtime.lease_timeout_ms = 1
        reserved = self.status(runtime, authority)
        reserved.reserved_samples = 1
        reserved.reserved_fragments = 1
        reserved.reserved_estimated_bytes = 128
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(return_value=reserved)
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            with self.assertRaisesRegex(
                RuntimeError, "did not settle.*reserved=1"
            ):
                runtime._shutdown_drain(authority)

    def test_required_demand_release_retries_exactly_after_response_loss(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.demand_id = "demand-release-transport-test"
        runtime._demand_epoch = 18
        runtime._demand_active = True
        runtime._demand_authority = authority
        released = training_pb2.SampleDemandRsp(
            ret_code=0,
            result=training_pb2.SAMPLE_DEMAND_RESULT_NOT_FOUND,
            distributor=authority,
        )
        release = mock.Mock(
            side_effect=[
                RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected applied release response loss",
                ),
                released,
            ]
        )
        runtime.sample_stub = SimpleNamespace(
            ReleaseSampleDemand=release,
            GetStatus=mock.Mock(
                return_value=self.status(runtime, authority)
            ),
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            runtime._release_demand(required=True)

        self.assertEqual(release.call_count, 2)
        first = release.call_args_list[0].args[0].SerializeToString(
            deterministic=True
        )
        second = release.call_args_list[1].args[0].SerializeToString(
            deterministic=True
        )
        self.assertEqual(first, second)
        self.assertFalse(runtime._demand_active)
        self.assertIsNone(runtime._demand_authority)

    def test_required_demand_release_persistent_timeout_is_failure(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.DEMAND_RELEASE_RETRY_TIMEOUT_SEC = 0.003
        runtime.demand_id = "demand-release-timeout-test"
        runtime._demand_epoch = 18
        runtime._demand_active = True
        runtime._demand_authority = authority
        runtime.sample_stub = SimpleNamespace(
            ReleaseSampleDemand=mock.Mock(
                side_effect=RpcFault(
                    grpc.StatusCode.UNAVAILABLE,
                    "injected persistent release outage",
                )
            )
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                runtime._release_demand(required=True)

        self.assertTrue(runtime._demand_active)
        self.assertTrue(
            runtime._same_authority(runtime._demand_authority, authority)
        )


if __name__ == "__main__":
    unittest.main()
