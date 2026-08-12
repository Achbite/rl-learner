import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from main.training_runtime import LeaseRenewer, TrainingRuntime
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


def sample_authority(
    instance_id: str = "sample-distributor-coherence-test",
    lifecycle_epoch: int = 1,
):
    return service_identity(
        "sample-distributor", instance_id, lifecycle_epoch
    )


class RpcResponseCoherenceTest(unittest.TestCase):
    def get_batch_runtime(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"]
        )
        runtime.trainer = SimpleNamespace(model_version=1, max_policy_lag=2)
        runtime._effective_max_policy_lag = lambda: 2
        runtime.learner_service = service_identity(
            "learner", "learner-coherence-test", 1
        )
        runtime.train_batch_size = 2
        runtime.max_train_batch_size = 3
        runtime.max_sample_age_ms = 120000
        runtime.get_timeout_ms = 1000
        runtime.lease_timeout_ms = 30000
        return runtime

    def demand_runtime(self):
        runtime = self.get_batch_runtime()
        runtime.demand_ttl_ms = 10000
        runtime.demand_refresh_interval_ms = 3000
        runtime.demand_max_fragments = 3
        runtime.demand_max_estimated_bytes = 8 * 1024 * 1024
        runtime.demand_id = "learner-coherence-test-demand"
        runtime._demand_epoch = 0
        runtime._demand_active = False
        runtime._demand_authority = None
        runtime._last_demand_refresh = 0.0
        runtime.logger = SimpleNamespace(error=mock.Mock())
        return runtime

    @staticmethod
    def leased_response(authority=None):
        return training_pb2.GetBatchRsp(
            ret_code=0,
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="delivery-coherence-1",
            returned_samples=2,
            actual_batch_size=2,
            returned_fragments=0,
            leased_samples=2,
            lease_deadline_unix_ms=int(time.time() * 1000) + 30000,
            distributor=authority or sample_authority(),
        )

    @staticmethod
    def ready_status(
        runtime,
        authority=None,
        *,
        ready=True,
        active_demand_count=0,
        active_demand_epoch=0,
        reserved_samples=0,
        reserved_fragments=0,
        reserved_estimated_bytes=0,
    ):
        return training_pb2.DistributorStatusRsp(
            ready=ready,
            contract=runtime.contract,
            distributor=authority or sample_authority(),
            active_demand_count=active_demand_count,
            active_demand_epoch=active_demand_epoch,
            reserved_samples=reserved_samples,
            reserved_fragments=reserved_fragments,
            reserved_estimated_bytes=reserved_estimated_bytes,
        )

    def test_leased_get_batch_requires_coherent_ready_authority_envelope(self):
        cases = (
            "ret_code",
            "invalid_authority",
            "changed_authority",
            "leased_samples",
            "deadline",
            "not_ready",
        )
        for case in cases:
            with self.subTest(case=case):
                runtime = self.get_batch_runtime()
                expected_authority = sample_authority()
                response = self.leased_response(expected_authority)
                status = self.ready_status(runtime, expected_authority)
                if case == "ret_code":
                    response.ret_code = -1
                elif case == "invalid_authority":
                    response.distributor.component = "rl-sample-pool"
                elif case == "changed_authority":
                    response.distributor.CopyFrom(
                        sample_authority("sample-distributor-restarted", 2)
                    )
                elif case == "leased_samples":
                    response.leased_samples = 1
                elif case == "deadline":
                    response.lease_deadline_unix_ms = (
                        int(time.time() * 1000) - 1
                    )
                elif case == "not_ready":
                    status.ready = False

                runtime.sample_stub = SimpleNamespace(
                    GetBatch=mock.Mock(return_value=response),
                    GetStatus=mock.Mock(return_value=status),
                    AckBatch=mock.Mock(),
                )
                ready_authority = (
                    None if case == "not_ready" else expected_authority
                )
                with self.assertRaises(RuntimeError):
                    runtime._get_batch(
                        ready_authority=ready_authority
                    )

                runtime.sample_stub.AckBatch.assert_not_called()

    def test_coherent_leased_get_batch_preserves_exact_authority(self):
        runtime = self.get_batch_runtime()
        authority = sample_authority()
        response = self.leased_response(authority)
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(return_value=response),
            GetStatus=mock.Mock(
                return_value=self.ready_status(runtime, authority)
            ),
        )

        actual = runtime._get_batch()

        self.assertIs(actual, response)
        runtime.sample_stub.GetStatus.assert_called_once()

    def test_malformed_leased_envelope_never_reaches_invalid_ack(self):
        runtime = self.get_batch_runtime()
        authority = sample_authority()
        response = self.leased_response(authority)
        response.ret_code = -1
        runtime.sample_stub = SimpleNamespace(
            GetBatch=mock.Mock(return_value=response)
        )
        runtime._initialize_models = mock.Mock()
        runtime._upsert_demand = mock.Mock()
        runtime._start_metrics = mock.Mock()
        runtime._record_metrics = mock.Mock()
        runtime._assert_sample_pool_ready = mock.Mock(
            return_value=authority
        )
        runtime._train_delivery = mock.Mock()
        runtime._ack = mock.Mock()
        runtime._release_demand = mock.Mock()
        runtime._stop_metrics = mock.Mock()
        runtime._metrics_lock = threading.RLock()
        runtime._metrics_context = {}
        runtime.logger = SimpleNamespace(exception=mock.Mock())
        runtime.metrics_backend = SimpleNamespace(close=mock.Mock())
        runtime.actor_channel = SimpleNamespace(close=mock.Mock())
        runtime.model_channel = SimpleNamespace(close=mock.Mock())
        runtime.sample_channel = SimpleNamespace(close=mock.Mock())

        with tempfile.TemporaryDirectory() as directory:
            runtime.finalize_request_path = Path(directory) / "finalize"
            with mock.patch(
                "main.training_runtime._stop_requested",
                threading.Event(),
            ):
                result = runtime.run()

        self.assertEqual(result, 1)
        runtime._train_delivery.assert_not_called()
        runtime._ack.assert_not_called()

    def test_demand_upsert_requires_exact_echo_and_ready_terminal_authority(self):
        authority = sample_authority()
        runtime = self.demand_runtime()
        expected_epoch = runtime.trainer.model_version + 1

        def upsert(request, timeout):
            return training_pb2.SampleDemandRsp(
                ret_code=0,
                result=training_pb2.SAMPLE_DEMAND_RESULT_APPLIED,
                demand=request.demand,
                distributor=authority,
            )

        runtime.sample_stub = SimpleNamespace(
            GetStatus=mock.Mock(
                side_effect=(
                    self.ready_status(runtime, authority),
                    self.ready_status(
                        runtime,
                        authority,
                        active_demand_count=1,
                        active_demand_epoch=expected_epoch,
                    ),
                )
            ),
            UpsertSampleDemand=mock.Mock(side_effect=upsert),
        )

        runtime._upsert_demand(force=True)

        self.assertTrue(runtime._demand_active)
        self.assertEqual(runtime._demand_epoch, expected_epoch)
        self.assertTrue(
            TrainingRuntime._same_authority(
                runtime._demand_authority, authority
            )
        )

    def test_demand_upsert_faults_do_not_publish_local_active_state(self):
        cases = (
            "contradictory_result",
            "changed_authority_negative",
            "wrong_demand_echo",
            "missing_terminal_epoch",
        )
        for case in cases:
            with self.subTest(case=case):
                runtime = self.demand_runtime()
                old_authority = sample_authority()
                new_authority = sample_authority(
                    "sample-distributor-restarted", 2
                )
                expected_epoch = runtime.trainer.model_version + 1

                def upsert(request, timeout):
                    response = training_pb2.SampleDemandRsp(
                        ret_code=0,
                        result=training_pb2.SAMPLE_DEMAND_RESULT_APPLIED,
                        demand=request.demand,
                        distributor=old_authority,
                    )
                    if case == "contradictory_result":
                        response.ret_code = -1
                    elif case == "changed_authority_negative":
                        response.ret_code = -1
                        response.result = (
                            training_pb2.SAMPLE_DEMAND_RESULT_REJECTED_IDENTITY
                        )
                        response.distributor.CopyFrom(new_authority)
                        response.ClearField("demand")
                    elif case == "wrong_demand_echo":
                        response.demand.max_buffered_samples += 1
                    return response

                post_status = self.ready_status(
                    runtime,
                    old_authority,
                    active_demand_count=(
                        0 if case == "missing_terminal_epoch" else 1
                    ),
                    active_demand_epoch=(
                        0 if case == "missing_terminal_epoch" else expected_epoch
                    ),
                )
                runtime.sample_stub = SimpleNamespace(
                    GetStatus=mock.Mock(
                        side_effect=(
                            self.ready_status(runtime, old_authority),
                            post_status,
                        )
                    ),
                    UpsertSampleDemand=mock.Mock(side_effect=upsert),
                )

                with self.assertRaises(RuntimeError):
                    runtime._upsert_demand(force=True)

                self.assertFalse(runtime._demand_active)
                self.assertEqual(runtime._demand_epoch, 0)
                self.assertIsNone(runtime._demand_authority)
                self.assertEqual(runtime._last_demand_refresh, 0.0)

    def test_demand_release_binds_absence_to_pinned_authority_and_empty_state(self):
        cases = (
            ("same_authority_not_found", True),
            ("changed_authority_released", False),
            ("changed_authority_not_found", False),
            ("contradictory_released", False),
            ("nonempty_terminal_response", False),
            ("nonempty_terminal_status", False),
        )
        for case, succeeds in cases:
            with self.subTest(case=case):
                runtime = self.demand_runtime()
                pinned = sample_authority()
                changed = sample_authority(
                    "sample-distributor-restarted", 2
                )
                runtime._demand_active = True
                runtime._demand_epoch = runtime.trainer.model_version + 1
                runtime._demand_authority = pinned
                response_authority = (
                    changed if case.startswith("changed_authority") else pinned
                )
                result = (
                    training_pb2.SAMPLE_DEMAND_RESULT_NOT_FOUND
                    if case.endswith("not_found")
                    else training_pb2.SAMPLE_DEMAND_RESULT_RELEASED
                )
                response = training_pb2.SampleDemandRsp(
                    ret_code=0,
                    result=result,
                    distributor=response_authority,
                )
                if case == "contradictory_released":
                    response.ret_code = -1
                elif case == "nonempty_terminal_response":
                    response.demand.CopyFrom(runtime._demand_message())
                post_status = self.ready_status(
                    runtime,
                    response_authority,
                    active_demand_count=(
                        1 if case == "nonempty_terminal_status" else 0
                    ),
                    active_demand_epoch=(
                        runtime._demand_epoch
                        if case == "nonempty_terminal_status"
                        else 0
                    ),
                )
                runtime.sample_stub = SimpleNamespace(
                    ReleaseSampleDemand=mock.Mock(return_value=response),
                    GetStatus=mock.Mock(return_value=post_status),
                )

                if succeeds:
                    runtime._release_demand(required=True)
                    self.assertFalse(runtime._demand_active)
                    self.assertIsNone(runtime._demand_authority)
                else:
                    with self.assertRaisesRegex(RuntimeError, "release failed"):
                        runtime._release_demand(required=True)
                    self.assertTrue(runtime._demand_active)
                    self.assertTrue(
                        TrainingRuntime._same_authority(
                            runtime._demand_authority, pinned
                        )
                    )

    def test_active_lease_requires_same_ready_authority_epoch(self):
        for case in ("not_ready", "changed_authority"):
            with self.subTest(case=case):
                runtime = self.get_batch_runtime()
                expected = sample_authority()
                actual = expected
                ready = True
                if case == "not_ready":
                    ready = False
                else:
                    actual = sample_authority(
                        "sample-distributor-restarted", 2
                    )
                runtime.sample_stub = SimpleNamespace(
                    GetStatus=mock.Mock(
                        return_value=self.ready_status(
                            runtime, actual, ready=ready
                        )
                    )
                )

                with self.assertRaises(RuntimeError):
                    runtime._assert_lease_authority(expected)

    def test_renew_lease_validates_response_before_authority_success(self):
        cases = (
            "error_code_with_applied_result",
            "success_code_with_rejected_result",
            "delivery_id",
            "deadline",
            "authority",
        )
        for case in cases:
            with self.subTest(case=case):
                response = training_pb2.DeliveryRsp(
                    ret_code=0,
                    result=training_pb2.DELIVERY_RESULT_APPLIED,
                    delivery_id="delivery-coherence-1",
                    lease_deadline_unix_ms=(
                        int(time.time() * 1000) + 30000
                    ),
                )
                if case == "error_code_with_applied_result":
                    response.ret_code = -1
                elif case == "success_code_with_rejected_result":
                    response.result = training_pb2.DELIVERY_RESULT_REJECTED
                elif case == "delivery_id":
                    response.delivery_id = "another-delivery"
                elif case == "deadline":
                    response.lease_deadline_unix_ms = (
                        int(time.time() * 1000) - 1
                    )

                authority_check = mock.Mock()
                if case == "authority":
                    authority_check.side_effect = RuntimeError(
                        "sample distributor authority changed during the lease"
                    )
                renewer = LeaseRenewer(
                    SimpleNamespace(
                        RenewLease=mock.Mock(return_value=copy.deepcopy(response))
                    ),
                    service_identity("learner", "learner-coherence-test", 1),
                    "delivery-coherence-1",
                    30000,
                    authority_check,
                )

                with self.assertRaises(RuntimeError):
                    renewer.renew_now()

                if case == "authority":
                    authority_check.assert_called_once()
                else:
                    authority_check.assert_not_called()

    def test_coherent_renew_lease_checks_authority_once(self):
        authority_check = mock.Mock()
        response = training_pb2.DeliveryRsp(
            ret_code=0,
            result=training_pb2.DELIVERY_RESULT_APPLIED,
            delivery_id="delivery-coherence-1",
            lease_deadline_unix_ms=int(time.time() * 1000) + 30000,
        )
        renewer = LeaseRenewer(
            SimpleNamespace(RenewLease=mock.Mock(return_value=response)),
            service_identity("learner", "learner-coherence-test", 1),
            "delivery-coherence-1",
            30000,
            authority_check,
        )

        renewer.renew_now()

        authority_check.assert_called_once()

    def test_lease_renewer_checks_remote_lease_synchronously_on_start(self):
        events = []
        response = training_pb2.DeliveryRsp(
            ret_code=0,
            result=training_pb2.DELIVERY_RESULT_APPLIED,
            delivery_id="delivery-coherence-1",
            lease_deadline_unix_ms=int(time.time() * 1000) + 30000,
        )

        def renew(request, timeout):
            events.append("renew")
            return response

        renewer = LeaseRenewer(
            SimpleNamespace(RenewLease=renew),
            service_identity("learner", "learner-coherence-test", 1),
            "delivery-coherence-1",
            30000,
            lambda: events.append("authority"),
        )

        renewer.start()
        try:
            self.assertEqual(events, ["renew", "authority"])
        finally:
            renewer.close()

    def test_lease_renewer_does_not_start_after_synchronous_authority_failure(self):
        response = training_pb2.DeliveryRsp(
            ret_code=0,
            result=training_pb2.DELIVERY_RESULT_APPLIED,
            delivery_id="delivery-coherence-1",
            lease_deadline_unix_ms=int(time.time() * 1000) + 30000,
        )
        renewer = LeaseRenewer(
            SimpleNamespace(RenewLease=mock.Mock(return_value=response)),
            service_identity("learner", "learner-coherence-test", 1),
            "delivery-coherence-1",
            30000,
            mock.Mock(side_effect=RuntimeError("authority changed")),
        )

        with self.assertRaisesRegex(RuntimeError, "authority changed"):
            renewer.start()

        self.assertFalse(renewer._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
