import hashlib
import unittest
from pathlib import Path

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


def model_document(cfg):
    semantics = training_semantics(cfg)
    return finalize_manifest_digest(
        {
            "manifest_schema_version": 1,
            "contract": contract_document(contract_identity(cfg)),
            "identity": {
                "model_lineage_id": cfg["identity"]["model_lineage_id"],
                "model_version": 0,
                "artifact_digest": "a" * 64,
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


def delivery(runtime, document):
    model = manifest_message(runtime._manifest_for_wire(document)).identity
    policy = training_pb2.BehaviorPolicyIdentity(
        model=model,
        distribution_schema_id=runtime.semantics.policy_distribution_schema_id,
        policy_spec_digest=runtime.policy_digest,
    )
    batch = training_pb2.SampleBatch(
        batch_id="batch-1",
        actor_session_id="session-1",
        trajectory_id="trajectory-1",
        actor_id=1,
        fragment_id=1,
        fragment_sequence=1,
        trajectory_end=False,
        bootstrap_value=0.25,
        bootstrap_valid=True,
        behavior_policy=policy,
        training_semantics=runtime.semantics,
        producer=common_pb2.ServiceInstanceIdentity(
            component="aiserver", instance_id="aiserver-1", lifecycle_epoch=1
        ),
        contract=runtime.contract,
        created_at_unix_ms=1,
        first_action_step=1,
        last_action_step=512,
    )
    for step in range(1, 513):
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
    return training_pb2.GetBatchRsp(
        result=training_pb2.GET_BATCH_RESULT_LEASED,
        delivery_id="delivery-1",
        returned_samples=512,
        actual_batch_size=512,
        behavior_policy=policy,
        batches=[batch],
    )


class DeliveryContractTest(unittest.TestCase):
    def runtime(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.semantics = training_semantics(cfg)
        runtime.policy_digest = policy_spec_digest(cfg)
        runtime.train_batch_size = 512
        return runtime, model_document(cfg)

    def test_exact_delivery_passes(self):
        runtime, document = self.runtime()
        runtime._validate_delivery(delivery(runtime, document), document)

    def test_payload_or_policy_identity_drift_fails(self):
        runtime, document = self.runtime()
        response = delivery(runtime, document)
        response.batches[0].samples[0].reward = 9.0
        with self.assertRaisesRegex(ValueError, "payload digest"):
            runtime._validate_delivery(response, document)
        response = delivery(runtime, document)
        response.behavior_policy.policy_spec_digest.hex = "b" * 64
        with self.assertRaisesRegex(ValueError, "behavior policy"):
            runtime._validate_delivery(response, document)

    def test_non_canonical_producer_component_fails(self):
        runtime, document = self.runtime()
        response = delivery(runtime, document)
        response.batches[0].producer.component = "rl-aiserver"
        with self.assertRaisesRegex(ValueError, "sample batch identity"):
            runtime._validate_delivery(response, document)


if __name__ == "__main__":
    unittest.main()
