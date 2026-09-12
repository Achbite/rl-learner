import unittest
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import grpc

from main.training_runtime import TrainingRuntime, train_processed_delivery, _stop_requested
from proto.common import identity_pb2
from proto.training import training_pb2
from src.config.effective_config import load_effective_config
from src.contracts.identity import validate_config
from src.training.ppo_trainer import PPOTrainer


def _trainer_config() -> dict:
    return {
        "training": {
            "device": "cpu",
            "seed": 0,
            "learning_rate": 0.0003,
            "clip_epsilon": 0.2,
            "value_clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 1,
            "train_batch_size": 2,
            "mini_batch_size": 2,
            "normalize_advantage": True,
        },
        "model": {
            "observation_dimension": 5,
            "action_count": 4,
            "hidden_dimension": 8,
        },
        "policy": {"action_mask_mode": "required"},
    }


class LearnerDevelopmentTest(unittest.TestCase):
    def setUp(self):
        _stop_requested.clear()
        self.addCleanup(_stop_requested.clear)

    @staticmethod
    def runtime(config):
        # Configure the real consumer functions without launching child services.
        runtime = TrainingRuntime.__new__(TrainingRuntime)
        runtime.trainer = PPOTrainer(config)
        runtime.publisher = SimpleNamespace(obs_dim=config["model"]["observation_dimension"],
            action_dim=config["model"]["action_count"], lineage_id="training-lineage")
        runtime.train_batch_size = config["training"]["train_batch_size"]
        manifest = training_pb2.ModelArtifactManifest()
        manifest.identity.model_lineage_id = runtime.publisher.lineage_id
        manifest.identity.model_step = 0
        runtime.model_manifests = {0: {"manifest": manifest}}
        runtime.logger = logging.getLogger("delivery-contract")
        runtime._metrics_lock = threading.Lock()
        runtime._metrics_context = {"disposition": "READY"}
        runtime.get_timeout_ms = 10
        runtime.lease_timeout_ms = 1000
        runtime.learner_service = identity_pb2.ServiceInstanceIdentity(
            component="learner", instance_id="learner-test", lifecycle_epoch=1)
        return runtime

    @staticmethod
    def _transition(
        index: int,
        observation_dimension: int,
        action_count: int,
    ) -> training_pb2.ProcessedTransition:
        action_mask = [False] * action_count
        action_mask[index] = True
        return training_pb2.ProcessedTransition(
            item_id=f"item-{index}",
            observation=[float(index)] * observation_dimension,
            action=index,
            behavior_log_probability=-0.5 - index,
            behavior_value=0.2 + 0.1 * index,
            advantage=-0.18515 if index == 0 else -0.3,
            value_target=0.01485 if index == 0 else 0.0,
            behavior_model_step=0,
            created_at_unix_ms=1700000000000 + index,
            action_mask=action_mask,
        )

    def test_processed_transition_data_reaches_real_trainer(self):
        config = _trainer_config()
        runtime = self.runtime(config)
        response = training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_LEASED,
            delivery_id="delivery-test",
        )
        for index in range(2):
            response.items.add(
                transition=self._transition(
                    index,
                    config["model"]["observation_dimension"],
                    config["model"]["action_count"],
                ),
                insert_sequence=index + 1,
                inserted_at_unix_ms=1700000000100 + index,
                draw_count=1,
                behavior_model=runtime.model_manifests[0]["manifest"].identity,
                producer=identity_pb2.ServiceInstanceIdentity(
                    component="aiserver", instance_id="producer-test", lifecycle_epoch=1),
            )

        provenance = runtime._validate_delivery(response)
        self.assertEqual(provenance["model_lineage_id"], "training-lineage")
        self.assertEqual(provenance["producers"][0]["instance_id"], "producer-test")
        trainer = runtime.trainer
        batch, stats = train_processed_delivery(response.items, trainer)

        self.assertEqual(len(batch), 2)
        self.assertEqual(stats["sample_evaluation_count"], 2)
        raw = trainer.raw_metric_sum_counts()
        self.assertEqual(raw["raw_advantage"]["count"], len(response.items))
        self.assertAlmostEqual(raw["raw_advantage"]["sum"],
                               sum(item.transition.advantage for item in response.items))
        self.assertEqual(
            [sample["item_id"] for sample in batch], ["item-0", "item-1"]
        )
        self.assertEqual([sample["action"] for sample in batch], [0, 1])
        self.assertEqual(
            [sample["behavior_model_step"] for sample in batch], [0, 0]
        )
        for index, sample in enumerate(batch):
            transition = response.items[index].transition
            self.assertEqual(sample["observation"], list(transition.observation))
            self.assertEqual(
                sample["old_log_probability"],
                float(transition.behavior_log_probability),
            )
            self.assertEqual(
                sample["old_value_prediction"],
                float(transition.behavior_value),
            )
            self.assertEqual(
                sample["action_mask"], list(transition.action_mask)
            )
            self.assertEqual(sample["advantage"], float(transition.advantage))
            self.assertEqual(
                sample["value_target"], float(transition.value_target)
            )
        for corruption, error in [("other-lineage", "source model"),
                                  ("missing-model", "source model"),
                                  ("missing-producer", "producer is invalid")]:
            with self.subTest(corruption=corruption):
                invalid = training_pb2.GetBatchRsp()
                invalid.CopyFrom(response)
                if corruption == "other-lineage":
                    invalid.items[0].behavior_model.model_lineage_id = "other-lineage"
                else:
                    invalid.items[0].ClearField("behavior_model" if corruption == "missing-model" else "producer")
                with self.assertRaisesRegex(ValueError, error):
                    runtime._validate_delivery(invalid)

    def test_local_effective_config_reaches_runtime_validation(self):
        repository = Path(__file__).resolve().parents[1]
        config = load_effective_config(
            str(repository / "configs" / "learner_config.yaml"),
            environment={
                "RL_MODEL_LINEAGE_ID": "config-test-lineage",
                "RL_PPO_TRAIN_BATCH_SIZE": "32",
                "RL_PPO_MINI_BATCH_SIZE": "16",
                "RL_PPO_N_EPOCHS": "1",
            },
        )

        validate_config(config)
        self.assertEqual(config["training"]["train_batch_size"], 32)
        self.assertEqual(config["training"]["mini_batch_size"], 16)
        self.assertEqual(config["training"]["n_epochs"], 1)

    def test_get_batch_recovery_uses_request_deadline(self):
        runtime = self.runtime(_trainer_config())
        authority = identity_pb2.ServiceInstanceIdentity(
            component="sample-pool", instance_id="pool-test", lifecycle_epoch=1)
        queries = []
        rpc_deadlines = []

        class Unavailable(grpc.RpcError):
            def code(self): return grpc.StatusCode.UNAVAILABLE
            def details(self): return "transport lost after request was sent"

        def get_batch(request, timeout):
            rpc_deadlines.append(time.monotonic() + timeout)
            raise Unavailable()

        def get_status(request, timeout):
            queries.append(time.monotonic())
            return training_pb2.SamplePoolStatusRsp(sample_pool=authority, ready=True,
                max_concurrent_consumers=1, leased_transitions=0, active_consumer_count=0)

        runtime.sample_stub = SimpleNamespace(GetBatch=get_batch, GetStatus=get_status)
        result = runtime._get_batch_recovering(ready_authority=authority, deadline=time.monotonic() + 5)
        self.assertIsNone(result)
        self.assertTrue(queries)
        self.assertGreaterEqual(min(queries), rpc_deadlines[0] - 0.005)
        self.assertEqual(runtime._metrics_context["disposition"], "READY")

    def test_busy_lease_and_stop_have_distinct_recovery(self):
        runtime = self.runtime(_trainer_config())
        authority = identity_pb2.ServiceInstanceIdentity(
            component="sample-pool", instance_id="pool-test", lifecycle_epoch=1)
        lease_expires = time.monotonic() + 0.1
        query_times = []

        def status(request, timeout):
            now = time.monotonic()
            query_times.append(now)
            leased = now < lease_expires
            return training_pb2.SamplePoolStatusRsp(sample_pool=authority, ready=True,
                max_concurrent_consumers=1, leased_transitions=2 if leased else 0,
                active_consumer_count=1 if leased else 0)

        runtime.sample_stub = SimpleNamespace(
            GetBatch=lambda request, timeout: training_pb2.GetBatchRsp(
                result=training_pb2.GET_BATCH_RESULT_BUSY, sample_pool=authority), GetStatus=status)
        self.assertIsNone(runtime._get_batch_recovering(ready_authority=authority, deadline=time.monotonic() + 3))
        self.assertLess(query_times[0], lease_expires)
        self.assertGreaterEqual(query_times[-1], lease_expires)
        self.assertEqual(runtime._metrics_context["disposition"], "READY")

        for stop_first in (True, False):
            with self.subTest(stop=stop_first):
                query_times.clear()
                runtime._mark_sample_wait("GET_BATCH_OUTCOME_UNKNOWN", "GetBatch", "uncertain", 1)
                if stop_first: _stop_requested.set()
                self.assertFalse(runtime._reconcile_get_batch_outcome(authority, "uncertain",
                    request_deadline=time.monotonic() + 2, deadline=time.monotonic() + 0.02))
                self.assertEqual(query_times, [])
                self.assertEqual(runtime._metrics_context["disposition"], "GET_BATCH_OUTCOME_UNKNOWN")
                _stop_requested.clear()
