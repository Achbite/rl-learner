import unittest
from pathlib import Path
from types import SimpleNamespace

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


class RuntimeSelectionTest(unittest.TestCase):
    def test_get_batch_requests_bounded_freshness_window(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"]
        )
        runtime.trainer = SimpleNamespace(model_version=2, max_policy_lag=2)
        runtime.learner_service = service_identity("learner", "learner-test")
        runtime.train_batch_size = 512
        runtime.max_train_batch_size = 639
        runtime.max_sample_age_ms = 120000
        runtime.get_timeout_ms = 1000
        runtime.lease_timeout_ms = 30000

        class Stub:
            request = None

            def GetBatch(self, request, timeout):
                self.request = request
                return training_pb2.GetBatchRsp()

        runtime.sample_stub = Stub()
        runtime._get_batch()
        request = runtime.sample_stub.request
        self.assertEqual(request.assembly.target_samples, 512)
        self.assertEqual(request.assembly.max_samples, 639)
        self.assertEqual(
            request.assembly.mode,
            training_pb2.BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
        )
        self.assertEqual(request.freshness.reference_model_version, 2)
        self.assertEqual(request.freshness.max_version_lag, 2)
        self.assertEqual(request.freshness.max_sample_age_ms, 120000)

    def test_rejects_pool_contract_drift(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime._sample_pool_status = lambda: training_pb2.DistributorStatusRsp(
            ready=True, ingress_ready=True, pool_ready=True
        )
        with self.assertRaisesRegex(RuntimeError, "exact contract"):
            runtime._assert_sample_pool_ready()


if __name__ == "__main__":
    unittest.main()
