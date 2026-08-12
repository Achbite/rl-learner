import inspect
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
    def demand_runtime(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"]
        )
        runtime.trainer = SimpleNamespace(model_version=2, max_policy_lag=2)
        runtime._effective_max_policy_lag = lambda: 2
        runtime.learner_service = service_identity("learner", "learner-test")
        runtime.train_batch_size = 512
        runtime.max_train_batch_size = 639
        runtime.max_sample_age_ms = 120000
        runtime.demand_ttl_ms = 10000
        runtime.demand_refresh_interval_ms = 3000
        runtime.demand_max_fragments = 639
        runtime.demand_max_estimated_bytes = 8 * 1024 * 1024
        runtime.demand_id = "learner-test-training-demand"
        runtime._demand_epoch = 0
        runtime._demand_active = False
        runtime._demand_authority = None
        runtime._last_demand_refresh = 0.0
        runtime.logger = SimpleNamespace(error=lambda *args: None)
        return runtime

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
        self.assertEqual(request.freshness.reference_model_version, 2)
        self.assertEqual(request.freshness.max_version_lag, 2)
        self.assertEqual(request.freshness.max_sample_age_ms, 120000)

    def test_exact_serving_ack_is_bootstrap_only(self):
        bootstrap_source = inspect.getsource(
            TrainingRuntime._initialize_models
        )
        update_source = inspect.getsource(TrainingRuntime._train_delivery)

        self.assertIn("_wait_initial_model_loaded", bootstrap_source)
        self.assertNotIn("_wait_initial_model_loaded", update_source)

    def test_rejects_pool_contract_drift(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime._sample_pool_status = lambda: training_pb2.DistributorStatusRsp(
            ready=True, ingress_ready=True, pool_ready=True
        )
        runtime._demand_epoch = 1
        with self.assertRaisesRegex(RuntimeError, "exact demand"):
            runtime._assert_sample_pool_ready()

    def test_demand_epoch_refresh_and_release_are_explicit(self):
        runtime = self.demand_runtime()

        class Stub:
            upserts = []
            releases = []
            reject_release = False
            active = False
            active_epoch = 0
            authority = service_identity(
                "sample-distributor", "sample-distributor-test", 1
            )

            def GetStatus(self, request, timeout):
                return training_pb2.DistributorStatusRsp(
                    ready=True,
                    contract=runtime.contract,
                    distributor=self.authority,
                    active_demand_count=1 if self.active else 0,
                    active_demand_epoch=(
                        self.active_epoch if self.active else 0
                    ),
                )

            def UpsertSampleDemand(self, request, timeout):
                self.upserts.append(request)
                self.active = True
                self.active_epoch = request.demand.demand_epoch
                return training_pb2.SampleDemandRsp(
                    ret_code=0,
                    result=training_pb2.SAMPLE_DEMAND_RESULT_APPLIED,
                    demand=request.demand,
                    distributor=self.authority,
                )

            def ReleaseSampleDemand(self, request, timeout):
                self.releases.append(request)
                if self.reject_release:
                    return training_pb2.SampleDemandRsp(
                        ret_code=-1,
                        result=(
                            training_pb2.SAMPLE_DEMAND_RESULT_REJECTED_STALE_EPOCH
                        ),
                        message="stale",
                        distributor=self.authority,
                    )
                self.active = False
                self.active_epoch = 0
                return training_pb2.SampleDemandRsp(
                    ret_code=0,
                    result=training_pb2.SAMPLE_DEMAND_RESULT_RELEASED,
                    distributor=self.authority,
                )

        stub = Stub()
        runtime.sample_stub = stub
        runtime._upsert_demand(force=True)
        first = stub.upserts[-1].demand
        self.assertEqual(first.demand_epoch, 3)
        self.assertEqual(first.assembly.target_samples, 512)
        self.assertEqual(first.assembly.max_samples, 639)
        self.assertEqual(first.max_buffered_samples, 639)
        self.assertEqual(first.max_buffered_fragments, 639)
        runtime._upsert_demand()
        self.assertEqual(len(stub.upserts), 1)

        runtime.trainer.model_version = 3
        runtime._upsert_demand()
        self.assertEqual(stub.upserts[-1].demand.demand_epoch, 4)
        self.assertEqual(runtime._demand_epoch, 4)

        stub.reject_release = True
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            runtime._release_demand(required=True)
        self.assertTrue(runtime._demand_active)
        stub.reject_release = False
        runtime._release_demand(required=False)
        self.assertFalse(runtime._demand_active)
        self.assertEqual(stub.releases[-1].demand_epoch, 4)

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
                training_pb2.DistributorStatusRsp(
                    ready_queue_samples=496,
                ),
                training_pb2.DistributorStatusRsp(),
            )
        )
        runtime._sample_pool_status = lambda: next(statuses)
        runtime._ready_sample_distributor_authority = lambda status: (
            service_identity(
                "sample-distributor", "sample-distributor-finalize-test", 1
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
        released = []
        runtime._release_demand = lambda required: released.append(required)
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
        self.assertEqual(released, [True])
        # Final sample settlement is authoritative; optional observation is
        # emitted only by the background metrics thread and cannot delay or
        # fail this control-path operation.
        self.assertEqual(recorded, [])
        self.assertEqual(runtime._metrics_context["disposition"], "FINALIZED")
        self.assertTrue(runtime._finalized)


if __name__ == "__main__":
    unittest.main()
