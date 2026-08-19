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
            model_step=17,
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
        runtime.SHUTDOWN_RECONCILE_MARGIN_SEC = 0.005
        return runtime

    @staticmethod
    def authority(instance_id="sample-pool-transport-test", epoch=1):
        return service_identity("sample-pool", instance_id, epoch)

    def status(
        self,
        runtime,
        authority,
        *,
        leased_samples=0,
        ready_samples=0,
        ready=True,
        finalized=False,
        finalization_id="",
        accepted_samples=None,
        accepted_batches=None,
        acked_samples=0,
        acked_batches=0,
        shutdown_untrained_samples=0,
        evicted_samples=0,
        evicted_fragments=0,
        finalized_samples=0,
        finalized_fragments=0,
        finalized_at_unix_ms=0,
    ):
        leased_fragments = 1 if leased_samples else 0
        ready_fragments = 1 if ready_samples else 0
        if accepted_samples is None:
            accepted_samples = (
                acked_samples
                + evicted_samples
                + leased_samples
                + ready_samples
            )
        if accepted_batches is None:
            accepted_batches = (
                acked_batches
                + evicted_fragments
                + leased_fragments
                + ready_fragments
            )
        return training_pb2.SamplePoolStatusRsp(
            contract=runtime.contract,
            sample_pool=authority,
            ready=ready,
            ingress_ready=ready,
            pool_ready=ready,
            accepted_unique_samples=accepted_samples,
            accepted_unique_batches=accepted_batches,
            acked_unique_samples=acked_samples,
            acked_unique_batches=acked_batches,
            ready_queue_samples=ready_samples,
            ready_queue_fragments=ready_fragments,
            leased_samples=leased_samples,
            leased_fragments=leased_fragments,
            resident_samples=ready_samples + leased_samples,
            resident_fragments=ready_fragments + leased_fragments,
            shutdown_untrained_sample_count=(
                shutdown_untrained_samples
            ),
            evicted_sample_count=evicted_samples,
            evicted_fragment_count=evicted_fragments,
            max_concurrent_consumers=1,
            active_consumer_count=1 if leased_samples else 0,
            finalized=finalized,
            finalization_id=finalization_id,
            finalized_at_unix_ms=finalized_at_unix_ms,
            finalized_sample_count=finalized_samples,
            finalized_fragment_count=finalized_fragments,
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
            runtime.trainer.model_step,
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
                runtime.trainer.model_step,
            ),
            progress,
        )
        runtime.trainer.train_on_batch.assert_not_called()
        self.assertEqual(runtime._metrics_context["disposition"], "TRAINED")
        self.assertEqual(runtime._metrics_context["error"], "")

    def test_reconciliation_fails_closed_on_authority_change(self):
        runtime = self.runtime()
        expected = self.authority()
        changed = self.authority("sample-pool-restarted", 2)
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

    def test_sample_pool_readiness_transport_faults_wait_then_recover(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime._assert_sample_pool_ready = mock.Mock(
            side_effect=[
                RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected status timeout",
                ),
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
            actual = runtime._wait_for_sample_pool()

        self.assertTrue(TrainingRuntime._same_authority(actual, authority))
        self.assertEqual(runtime._assert_sample_pool_ready.call_count, 3)
        self.assertEqual(runtime._metrics_context["disposition"], "TRAINED")


    def test_shutdown_with_stop_set_waits_out_a_hidden_lease(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 50
        runtime.lease_timeout_ms = 1
        runtime.GET_BATCH_RECONCILE_STABLE_WINDOW_SEC = 0
        runtime._ack = mock.Mock()
        runtime._get_batch_recovering = mock.Mock()
        finalization_id = runtime._sample_pool_finalization_id()
        finalized_at = int(time.time() * 1000)
        finalized = self.status(
            runtime,
            authority,
            ready=False,
            finalized=True,
            finalization_id=finalization_id,
            accepted_samples=4,
            accepted_batches=2,
            acked_samples=4,
            acked_batches=2,
            shutdown_untrained_samples=4,
            finalized_samples=4,
            finalized_fragments=2,
            finalized_at_unix_ms=finalized_at,
        )
        finalize = mock.Mock(
            side_effect=[
                RpcFault(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "injected FinalizeSamplePool response loss",
                ),
                training_pb2.FinalizeSamplePoolRsp(
                    ret_code=0,
                    result=(
                        training_pb2.SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED
                    ),
                    finalization_id=finalization_id,
                    sample_pool=authority,
                    settled_samples=4,
                    settled_fragments=2,
                    finalized_at_unix_ms=finalized_at,
                ),
            ]
        )
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(
                side_effect=[
                    self.status(runtime, authority, leased_samples=2),
                    self.status(runtime, authority, leased_samples=2),
                    self.status(runtime, authority),
                    self.status(runtime, authority),
                    self.status(runtime, authority),
                    finalized,
                ]
            ),
            FinalizeSamplePool=finalize,
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            runtime._shutdown_finalize_sample_pool(authority)

        runtime._ack.assert_not_called()
        runtime._get_batch_recovering.assert_not_called()
        self.assertEqual(finalize.call_count, 2)
        first_request = finalize.call_args_list[0].args[0]
        second_request = finalize.call_args_list[1].args[0]
        self.assertEqual(
            first_request.SerializeToString(deterministic=True),
            second_request.SerializeToString(deterministic=True),
        )
        self.assertEqual(first_request.finalization_id, finalization_id)
        self.assertTrue(
            TrainingRuntime._same_authority(
                first_request.expected_sample_pool, authority
            )
        )
        self.assertEqual(
            first_request.consumer.SerializeToString(deterministic=True),
            runtime.learner_service.SerializeToString(deterministic=True),
        )
        self.assertEqual(runtime._metrics_context["error"], "")

    def test_shutdown_unsettled_ready_samples_is_a_failure(self):
        runtime = self.runtime()
        authority = self.authority()
        runtime.shutdown_drain_timeout_ms = 20
        runtime.lease_timeout_ms = 1
        finalization_id = runtime._sample_pool_finalization_id()
        finalized_at = int(time.time() * 1000)
        unsettled = self.status(
            runtime, authority, ready_samples=1
        )
        contradictory = self.status(
            runtime,
            authority,
            ready=False,
            finalized=True,
            finalization_id=finalization_id,
            accepted_samples=2,
            accepted_batches=2,
            acked_samples=1,
            acked_batches=1,
            shutdown_untrained_samples=1,
            finalized_samples=1,
            finalized_fragments=1,
            finalized_at_unix_ms=finalized_at,
        )
        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(side_effect=[unsettled, contradictory]),
            FinalizeSamplePool=mock.Mock(
                return_value=training_pb2.FinalizeSamplePoolRsp(
                    ret_code=0,
                    result=(
                        training_pb2.SAMPLE_POOL_FINALIZE_RESULT_FINALIZED
                    ),
                    finalization_id=finalization_id,
                    sample_pool=authority,
                    settled_samples=1,
                    settled_fragments=1,
                    finalized_at_unix_ms=finalized_at,
                )
            ),
        )
        stop = threading.Event()
        stop.set()

        with mock.patch(
            "main.training_runtime._stop_requested", stop
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "contradictory finalized sample accounting",
            ):
                runtime._shutdown_finalize_sample_pool(authority)
