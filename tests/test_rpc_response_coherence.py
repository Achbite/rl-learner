import copy
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


def sample_pool_authority(
    instance_id: str = "sample-pool-coherence-test",
    lifecycle_epoch: int = 1,
):
    return service_identity("sample-pool", instance_id, lifecycle_epoch)


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
        runtime.trainer = SimpleNamespace(model_step=1, max_policy_lag=2)
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
            sample_pool=authority or sample_pool_authority(),
        )

    @staticmethod
    def ready_status(
        runtime,
        authority=None,
        *,
        ready=True,
    ):
        return training_pb2.SamplePoolStatusRsp(
            ready=ready,
            pool_ready=ready,
            contract=runtime.contract,
            sample_pool=authority or sample_pool_authority(),
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
                expected_authority = sample_pool_authority()
                response = self.leased_response(expected_authority)
                status = self.ready_status(runtime, expected_authority)
                if case == "ret_code":
                    response.ret_code = -1
                elif case == "invalid_authority":
                    response.sample_pool.component = "rl-sample-pool"
                elif case == "changed_authority":
                    response.sample_pool.CopyFrom(
                        sample_pool_authority("sample-pool-restarted", 2)
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
        authority = sample_pool_authority()
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
                        "sample pool authority changed during the lease"
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
