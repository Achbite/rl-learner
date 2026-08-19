import hashlib
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from main.training_runtime import TrainingRuntime
from proto import common_pb2, training_pb2
from src.contracts.identity import (
    contract_document,
    contract_identity,
    finalize_manifest_digest,
    policy_spec_digest,
    schema_document,
    semantics_document,
    service_identity,
    training_config_digest,
    training_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load(
        (ROOT / "configs" / "learner_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def model_document(cfg, version):
    semantics = training_semantics(cfg)
    artifact_digest = hashlib.sha256(
        f"artifact-{version}".encode("utf-8")
    ).hexdigest()
    return finalize_manifest_digest(
        {
            "manifest_schema_version": 2,
            "contract": contract_document(contract_identity(cfg)),
            "identity": {
                "model_lineage_id": cfg["identity"]["model_lineage_id"],
                "model_step": version,
                "artifact_digest": artifact_digest,
                "manifest_digest": "0" * 64,
            },
            "observation_schema": schema_document(
                semantics.observation_schema
            ),
            "action_schema": schema_document(semantics.action_schema),
            "model_architecture_id": semantics.model_architecture_id,
            "tensor_dtype": "float32",
            "input_shape": [1, 17],
            "action_shape": [1, 9],
            "value_shape": [1, 1],
            "artifact_uri": f"file:///tmp/model_v{version}.onnx",
            "model_file": f"model_v{version}.onnx",
            "size_bytes": 1,
            "seed": 0,
            "train_updates": version,
            "trained_samples": version,
            "training_config_digest": training_config_digest(cfg).hex,
            "training_semantics": semantics_document(semantics),
            "published_at_unix_ms": 1,
            "ready": True,
        }
    )


class _GetBatchStub:
    def __init__(self):
        self.request = None

    def GetBatch(self, request, timeout):
        self.request = request
        return training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_TIMEOUT
        )


class EffectivePolicyLagTest(unittest.TestCase):
    def runtime(self, current_version=7):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"],
            training_digest=training_config_digest(cfg),
        )
        runtime.trainer = SimpleNamespace(
            model_step=current_version, max_policy_lag=2
        )
        runtime.learner_service = service_identity(
            "learner", "learner-effective-lag-test", 1
        )
        runtime.train_batch_size = 1
        runtime.max_train_batch_size = 1
        runtime.max_sample_age_ms = 120000
        runtime.get_timeout_ms = 1000
        runtime.lease_timeout_ms = 30000
        runtime.model_manifests = {
            current_version: model_document(cfg, current_version)
        }
        runtime.sample_stub = _GetBatchStub()
        return cfg, runtime

    def assert_freshness_surfaces(self, runtime, expected_lag):
        self.assertEqual(
            runtime._effective_max_policy_lag(), expected_lag
        )
        runtime._get_batch()
        self.assertEqual(
            runtime.sample_stub.request.freshness.max_model_step_lag,
            expected_lag,
        )

    @staticmethod
    def delivery(runtime, version):
        created_at = int(time.time() * 1000)
        batch = training_pb2.SampleBatch(
            batch_id=f"batch-{version}",
            actor_session_id="session-1",
            trajectory_id=f"trajectory-{version}",
            actor_id=1,
            fragment_id=1,
            fragment_sequence=1,
            trajectory_end=False,
            bootstrap_value=0.25,
            bootstrap_valid=True,
            behavior_policy=training_pb2.BehaviorPolicyReference(
                model_lineage_id=runtime.publisher.lineage_id,
                model_step=version,
                distribution_schema_id=(
                    runtime.semantics.policy_distribution_schema_id
                ),
                policy_spec_digest=runtime.policy_digest,
            ),
            training_semantics=runtime.semantics,
            producer=common_pb2.ServiceInstanceIdentity(
                component="aiserver",
                instance_id="aiserver-effective-lag-test",
                lifecycle_epoch=1,
            ),
            contract=runtime.contract,
            created_at_unix_ms=created_at,
            first_action_step=1,
            last_action_step=1,
        )
        sample = batch.samples.add(
            action=1,
            reward=0.0,
            old_log_probability=-2.0,
            old_value_prediction=0.1,
            end_kind=training_pb2.TRANSITION_END_KIND_CONTINUING,
            action_step=1,
        )
        sample.observation.extend([0.0] * 17)
        sample.next_observation.extend([0.1] * 17)
        digest_copy = training_pb2.SampleBatch()
        digest_copy.CopyFrom(batch)
        digest_copy.ClearField("payload_digest")
        batch.payload_digest.algorithm = common_pb2.DIGEST_ALGORITHM_SHA256
        batch.payload_digest.hex = hashlib.sha256(
            digest_copy.SerializeToString(deterministic=True)
        ).hexdigest()
        return training_pb2.GetBatchRsp(
            ret_code=0,
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id=f"delivery-{version}",
            returned_samples=1,
            actual_batch_size=1,
            returned_fragments=1,
            minimum_behavior_model_step=version,
            maximum_behavior_model_step=version,
            oldest_sample_created_at_unix_ms=created_at,
            newest_sample_created_at_unix_ms=created_at,
            batches=[batch],
        )

    def test_checkpoint_restart_starts_with_lag_zero(self):
        _, runtime = self.runtime(current_version=7)

        self.assert_freshness_surfaces(runtime, 0)
        summary = runtime._validate_delivery(self.delivery(runtime, 7))

        self.assertEqual(summary["minimum_model_step"], 7)
        self.assertEqual(summary["maximum_model_step"], 7)

    def test_contiguous_publications_expand_to_configured_cap(self):
        cfg, runtime = self.runtime(current_version=7)
        self.assert_freshness_surfaces(runtime, 0)

        runtime.model_manifests[8] = model_document(cfg, 8)
        runtime.trainer.model_step = 8
        self.assert_freshness_surfaces(runtime, 1)

        runtime.model_manifests[9] = model_document(cfg, 9)
        runtime.trainer.model_step = 9
        self.assert_freshness_surfaces(runtime, 2)

        runtime.model_manifests[10] = model_document(cfg, 10)
        runtime.trainer.model_step = 10
        self.assert_freshness_surfaces(runtime, 2)

    def test_manifest_gap_does_not_widen_freshness(self):
        cfg, runtime = self.runtime(current_version=9)
        runtime.model_manifests[7] = model_document(cfg, 7)

        self.assert_freshness_surfaces(runtime, 0)

        runtime.model_manifests[8] = model_document(cfg, 8)
        self.assert_freshness_surfaces(runtime, 2)
        runtime.model_manifests.pop(7)
        self.assert_freshness_surfaces(runtime, 1)

    def test_contiguous_old_samples_remain_valid_after_publication(self):
        cfg, runtime = self.runtime(current_version=7)
        runtime.model_manifests[8] = model_document(cfg, 8)
        runtime.trainer.model_step = 8

        summary = runtime._validate_delivery(self.delivery(runtime, 7))
        self.assertEqual(summary["minimum_model_step"], 7)

        runtime.model_manifests[9] = model_document(cfg, 9)
        runtime.trainer.model_step = 9
        summary = runtime._validate_delivery(self.delivery(runtime, 7))
        self.assertEqual(summary["minimum_model_step"], 7)

    def test_unresolvable_manifest_stops_the_contiguous_window(self):
        cfg, runtime = self.runtime(current_version=9)
        runtime.model_manifests[8] = model_document(cfg, 8)
        runtime.model_manifests[7] = model_document(cfg, 7)
        runtime.model_manifests[8]["identity"]["model_lineage_id"] = (
            "wrong-lineage"
        )

        self.assert_freshness_surfaces(runtime, 0)

    def test_unresolvable_current_manifest_fails_before_remote_work(self):
        for current_document in (None, {"identity": {}}):
            with self.subTest(current_document=current_document):
                _, runtime = self.runtime(current_version=7)
                runtime.model_manifests = (
                    {}
                    if current_document is None
                    else {7: current_document}
                )
                response = self.delivery(runtime, 7)

                with self.assertRaisesRegex(
                    RuntimeError, "current model manifest"
                ):
                    runtime._get_batch()
                self.assertIsNone(runtime.sample_stub.request)
                with self.assertRaisesRegex(
                    RuntimeError, "current model manifest"
                ):
                    runtime._validate_delivery(response)
