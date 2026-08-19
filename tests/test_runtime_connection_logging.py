import tempfile
import threading
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
        runtime.trainer = SimpleNamespace(model_step=2, max_policy_lag=2)
        runtime._effective_max_policy_lag = lambda: 2
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
        self.assertEqual(request.freshness.reference_model_step, 2)
        self.assertEqual(request.freshness.max_model_step_lag, 2)
        self.assertEqual(request.freshness.max_sample_age_ms, 120000)

    def test_rejects_pool_contract_drift(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime._sample_pool_status = lambda: training_pb2.SamplePoolStatusRsp(
            ready=True, ingress_ready=True, pool_ready=True
        )
        with self.assertRaisesRegex(RuntimeError, "contract identity"):
            runtime._assert_sample_pool_ready()


    def test_explicit_finalize_trains_available_tail_before_completion(self):
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.finalize_drain_timeout_ms = 1000
        runtime.train_updates = 6
        runtime.trained_samples = 3072
        runtime._metrics_lock = threading.RLock()
        runtime._metrics_context = {"disposition": "TRAINED"}
        runtime._finalized = False
        statuses = iter(
            (
                training_pb2.SamplePoolStatusRsp(
                    ready_queue_samples=496,
                ),
                training_pb2.SamplePoolStatusRsp(),
            )
        )
        runtime._sample_pool_status = lambda: next(statuses)
        runtime._ready_sample_pool_authority = lambda status: (
            service_identity(
                "sample-pool", "sample-pool-finalize-test", 1
            )
        )
        modes = []
        delivery = training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="tail-delivery",
        )
        runtime._get_batch_recovering = lambda **kwargs: (
            modes.append(kwargs["mode"]) or delivery
        )
        trained = []
        runtime._train_delivery = lambda response, allow_partial=False: (
            trained.append((response.delivery_id, allow_partial))
        )
        recorded = []
        runtime._record_metrics = lambda: recorded.append(True)

        with tempfile.TemporaryDirectory() as directory:
            runtime.finalize_complete_path = (
                Path(directory) / "training-finalized"
            )
            runtime._finalize_training()
            self.assertEqual(
                runtime.finalize_complete_path.read_text(encoding="utf-8"),
                "6 3072\n",
            )

        self.assertEqual(
            modes, [training_pb2.BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE]
        )
        self.assertEqual(trained, [("tail-delivery", True)])
        # Final sample settlement is authoritative; optional observation is
        # emitted only by the background metrics thread and cannot delay or
        # fail this control-path operation.
        self.assertEqual(recorded, [])
        self.assertEqual(runtime._metrics_context["disposition"], "FINALIZED")
        self.assertTrue(runtime._finalized)
