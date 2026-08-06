import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from main.training_runtime import TrainingRuntime
from proto import training_pb2
from src.contracts.identity import (
    contract_document,
    contract_identity,
    finalize_manifest_digest,
    manifest_message,
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


def model_document(cfg, version: int, marker: str) -> dict:
    semantics = training_semantics(cfg)
    return finalize_manifest_digest(
        {
            "manifest_schema_version": 1,
            "contract": contract_document(contract_identity(cfg)),
            "identity": {
                "model_lineage_id": cfg["identity"]["model_lineage_id"],
                "model_version": version,
                "artifact_digest": marker * 64,
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
            "artifact_uri": f"file:///tmp/model-{version}.onnx",
            "model_file": f"model-{version}.onnx",
            "size_bytes": 1,
            "seed": 0,
            "train_updates": version,
            "trained_samples": version * 512,
            "training_config_digest": training_config_digest(cfg).hex,
            "training_semantics": semantics_document(semantics),
            "published_at_unix_ms": 1,
            "ready": True,
        }
    )


class RuntimeSelectionTest(unittest.TestCase):
    def test_prefers_current_policy_then_allows_one_version_lag(self):
        cfg = config()
        previous = model_document(cfg, 1, "a")
        current = model_document(cfg, 2, "b")
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.trainer = SimpleNamespace(model_version=2, max_policy_lag=1)
        runtime.train_batch_size = 512
        runtime.model_manifests = {1: previous, 2: current}
        status = training_pb2.DistributorStatusRsp(
            contract=runtime.contract,
            ready=True,
            ingress_ready=True,
            pool_ready=True,
        )
        entry = status.behavior_versions.add()
        entry.behavior_model.CopyFrom(
            manifest_message(runtime._manifest_for_wire(previous)).identity
        )
        entry.ready_samples = 512
        runtime._sample_pool_status = lambda: status
        self.assertEqual(
            runtime._select_behavior_document()["identity"]["model_version"],
            1,
        )
        current_entry = status.behavior_versions.add()
        current_entry.behavior_model.CopyFrom(
            manifest_message(runtime._manifest_for_wire(current)).identity
        )
        current_entry.ready_samples = 512
        self.assertEqual(
            runtime._select_behavior_document()["identity"]["model_version"],
            2,
        )

    def test_rejects_pool_contract_drift(self):
        cfg = config()
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.contract = contract_identity(cfg)
        runtime.trainer = SimpleNamespace(model_version=0, max_policy_lag=1)
        runtime.train_batch_size = 512
        runtime.model_manifests = {0: model_document(cfg, 0, "a")}
        runtime._sample_pool_status = lambda: training_pb2.DistributorStatusRsp(
            ready=True, ingress_ready=True, pool_ready=True
        )
        with self.assertRaisesRegex(RuntimeError, "exact contract"):
            runtime._select_behavior_document()


if __name__ == "__main__":
    unittest.main()
