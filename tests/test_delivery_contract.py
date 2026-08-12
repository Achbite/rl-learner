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
    manifest_message,
    policy_spec_digest,
    schema_document,
    semantics_document,
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


def model_document(cfg, version=0):
    semantics = training_semantics(cfg)
    return finalize_manifest_digest(
        {
            "manifest_schema_version": 1,
            "contract": contract_document(contract_identity(cfg)),
            "identity": {
                "model_lineage_id": cfg["identity"]["model_lineage_id"],
                "model_version": version,
                "artifact_digest": chr(ord("a") + version) * 64,
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
            "artifact_uri": "file:///tmp/model.onnx",
            "model_file": "model.onnx",
            "size_bytes": 1,
            "seed": 0,
            "train_updates": 0,
            "trained_samples": 0,
            "training_config_digest": training_config_digest(cfg).hex,
            "training_semantics": semantics_document(semantics),
            "published_at_unix_ms": 1,
            "ready": True,
        }
    )


def _batch(runtime, version, first_step, sample_count):
    policy = training_pb2.BehaviorPolicyReference(
        model_lineage_id=runtime.publisher.lineage_id,
        model_version=version,
        distribution_schema_id=(
            runtime.semantics.policy_distribution_schema_id
        ),
        policy_spec_digest=runtime.policy_digest,
    )
    created_at = int(time.time() * 1000)
    batch = training_pb2.SampleBatch(
        batch_id=f"batch-{version}",
        actor_session_id="session-1",
        trajectory_id=f"trajectory-{version}",
        actor_id=version,
        fragment_id=version,
        fragment_sequence=version,
        trajectory_end=False,
        bootstrap_value=0.25,
        bootstrap_valid=True,
        behavior_policy=policy,
        training_semantics=runtime.semantics,
        producer=common_pb2.ServiceInstanceIdentity(
            component="aiserver", instance_id="aiserver-1", lifecycle_epoch=1
        ),
        contract=runtime.contract,
        created_at_unix_ms=created_at,
        first_action_step=first_step,
        last_action_step=first_step + sample_count - 1,
    )
    for step in range(first_step, first_step + sample_count):
        sample = batch.samples.add(
            action=step % 9,
            reward=0.01,
            old_log_probability=-2.0,
            old_value_prediction=0.1,
            end_kind=training_pb2.TRANSITION_END_KIND_CONTINUING,
            action_step=step,
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
    return batch


def delivery(runtime):
    batches = [_batch(runtime, 0, 1, 256), _batch(runtime, 1, 257, 256)]
    created = [batch.created_at_unix_ms for batch in batches]
    return training_pb2.GetBatchRsp(
        result=training_pb2.GET_BATCH_RESULT_LEASED,
        delivery_id="delivery-1",
        returned_samples=512,
        actual_batch_size=512,
        returned_fragments=2,
        minimum_behavior_model_version=0,
        maximum_behavior_model_version=1,
        oldest_sample_created_at_unix_ms=min(created),
        newest_sample_created_at_unix_ms=max(created),
        batches=batches,
    )


def partial_delivery(runtime):
    batches = [_batch(runtime, 0, 1, 248), _batch(runtime, 1, 249, 248)]
    created = [batch.created_at_unix_ms for batch in batches]
    return training_pb2.GetBatchRsp(
        result=training_pb2.GET_BATCH_RESULT_LEASED,
        delivery_id="delivery-final",
        returned_samples=496,
        actual_batch_size=496,
        returned_fragments=2,
        minimum_behavior_model_version=0,
        maximum_behavior_model_version=1,
        oldest_sample_created_at_unix_ms=min(created),
        newest_sample_created_at_unix_ms=max(created),
        batches=batches,
    )


class DeliveryContractTest(unittest.TestCase):
    def runtime(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.publisher = SimpleNamespace(
            lineage_id=cfg["identity"]["model_lineage_id"],
            training_digest=training_config_digest(cfg),
        )
        runtime.trainer = SimpleNamespace(model_version=1, max_policy_lag=2)
        runtime.train_batch_size = 512
        runtime.max_train_batch_size = 639
        runtime.max_sample_age_ms = 120000
        runtime.model_manifests = {
            version: model_document(cfg, version) for version in (0, 1)
        }
        return runtime

    def test_bounded_multi_version_delivery_passes(self):
        runtime = self.runtime()
        summary = runtime._validate_delivery(delivery(runtime))
        self.assertEqual(summary["minimum_model_version"], 0)
        self.assertEqual(summary["maximum_model_version"], 1)

    def test_partial_delivery_is_accepted_only_for_explicit_final_drain(self):
        runtime = self.runtime()
        response = partial_delivery(runtime)
        with self.assertRaisesRegex(ValueError, "bounded batch assembly"):
            runtime._validate_delivery(response)
        summary = runtime._validate_delivery(response, allow_partial=True)
        self.assertEqual(summary["minimum_model_version"], 0)
        self.assertEqual(summary["maximum_model_version"], 1)

    def test_payload_or_policy_identity_drift_fails(self):
        runtime = self.runtime()
        response = delivery(runtime)
        response.batches[0].samples[0].reward = 9.0
        with self.assertRaisesRegex(ValueError, "payload digest"):
            runtime._validate_delivery(response)
        response = delivery(runtime)
        response.batches[0].behavior_policy.policy_spec_digest.hex = "b" * 64
        with self.assertRaisesRegex(ValueError, "sample batch identity"):
            runtime._validate_delivery(response)

    def test_non_canonical_producer_component_fails(self):
        runtime = self.runtime()
        response = delivery(runtime)
        response.batches[0].producer.component = "rl-aiserver"
        with self.assertRaisesRegex(ValueError, "sample batch identity"):
            runtime._validate_delivery(response)


if __name__ == "__main__":
    unittest.main()
