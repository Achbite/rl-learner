import unittest
import logging
import threading
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import grpc
import yaml

from main.training_runtime import TrainingRuntime, _stop_requested
from src.training.sample_pool_consumer import SamplePoolConsumer, training_samples
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

    def runtime(self, config):
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
        runtime.learner_service = identity_pb2.ServiceInstanceIdentity(
            component="learner", instance_id="learner-test", lifecycle_epoch=1)
        runtime.pool = SamplePoolConsumer(
            "localhost:1", {"get_timeout_ms": 10, "lease_timeout_ms": 1000, "shutdown_drain_timeout_ms": 1000},
            runtime.learner_service, runtime.train_batch_size, _stop_requested, runtime.logger, runtime._report_sample_wait,
        )
        self.addCleanup(runtime.pool.close)
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

        provenance = runtime.pool.delivery_provenance(response, runtime._validate_training_item)
        self.assertEqual(provenance["model_lineage_id"], "training-lineage")
        self.assertEqual(provenance["producers"][0]["instance_id"], "producer-test")
        trainer = runtime.trainer
        batch = training_samples(response.items)
        stats = trainer.train_on_batch(batch)

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
                    runtime.pool.delivery_provenance(invalid, runtime._validate_training_item)

    def test_replay_slots_reach_real_trainer_without_deduplication(self):
        config = _trainer_config()
        config["training"].update(train_batch_size=4, n_epochs=2)
        runtime = self.runtime(config)
        response = training_pb2.GetBatchRsp(
            result=training_pb2.GET_BATCH_RESULT_LEASED, delivery_id="replay-batch")
        for index in (0, 0, 1, 1):
            response.items.add(
                transition=self._transition(index, 5, 4),
                insert_sequence=index + 1, inserted_at_unix_ms=1700000000100 + index,
                draw_count=2, behavior_model=runtime.model_manifests[0]["manifest"].identity,
                producer=identity_pb2.ServiceInstanceIdentity(
                    component="aiserver", instance_id="producer-test", lifecycle_epoch=1))
        runtime.pool.delivery_provenance(response, runtime._validate_training_item)
        batch = training_samples(response.items)
        stats = runtime.trainer.train_on_batch(batch)
        self.assertEqual([item["item_id"] for item in batch], ["item-0", "item-0", "item-1", "item-1"])
        self.assertEqual(stats["sample_evaluation_count"], 8)
        self.assertEqual(stats["optimizer_step_count"], 4)
        self.assertEqual(runtime.trainer.raw_metric_sum_counts()["raw_advantage"]["count"], 4)
        # Reuse the unchanged snapshots in another update, not another Push.
        runtime.pool.delivery_provenance(response, runtime._validate_training_item)
        repeated = runtime.trainer.train_on_batch(training_samples(response.items))
        self.assertEqual(repeated["model_step"], 2)
        for changed in ("payload", "producer", "storage"):
            with self.subTest(changed=changed):
                conflict = training_pb2.GetBatchRsp()
                conflict.CopyFrom(response)
                if changed == "payload": conflict.items[1].transition.advantage += 1
                elif changed == "producer": conflict.items[1].producer.instance_id = "other-source"
                else: conflict.items[1].insert_sequence += 1
                with self.assertRaisesRegex(ValueError, "conflicting snapshot"):
                    runtime.pool.delivery_provenance(conflict, runtime._validate_training_item)

    def test_replay_finalization_keeps_storage_and_draw_accounts_separate(self):
        runtime = self.runtime(_trainer_config())
        authority = identity_pb2.ServiceInstanceIdentity(
            component="sample-pool", instance_id="pool-test", lifecycle_epoch=1)
        response = training_pb2.FinalizeSamplePoolRsp(
            result=training_pb2.SAMPLE_POOL_FINALIZE_RESULT_FINALIZED,
            finalization_id="replay-final", finalized_at_unix_ms=1700000000000,
            sample_pool=authority)
        status = training_pb2.SamplePoolStatusRsp(
            sample_pool=authority, ready=True, finalized=True,
            finalization_id=response.finalization_id, finalized_at_unix_ms=response.finalized_at_unix_ms,
            accepted_unique_transitions=6, evicted_transition_count=2, finalized_transition_count=4,
            drawn_transition_slot_count=28, acknowledged_transition_slot_count=24,
            released_transition_slot_count=4, trained_transition_count=16,
            invalid_transition_count=4, shutdown_untrained_transition_count=4)
        runtime.pool._validate_finalized_sample_pool_status(status, authority, response.finalization_id, response)
        for field in ("accepted_unique_transitions", "drawn_transition_slot_count", "trained_transition_count"):
            with self.subTest(field=field):
                broken = training_pb2.SamplePoolStatusRsp()
                broken.CopyFrom(status)
                setattr(broken, field, getattr(broken, field) + 1)
                with self.assertRaisesRegex(RuntimeError, "accounting"):
                    runtime.pool._validate_finalized_sample_pool_status(broken, authority, response.finalization_id, response)

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

        # The managed supervisor starts Learner before any actor endpoint exists.
        template = yaml.safe_load((repository / "configs/learner_config.yaml").read_text())
        del template["aiserver_status"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "managed.yaml"
            path.write_text(yaml.safe_dump(template))
            managed = load_effective_config(str(path), environment={
                "RL_MODEL_LINEAGE_ID": "managed-lineage", "RL_INFRA_MANAGED": "true",
                "RL_INFRA_DATA_ROOT": directory, "RL_INFRA_RUNTIME_ROOT": directory,
                "RL_INFRA_ENDPOINT_SAMPLE_POOL_HOST": "pool", "RL_INFRA_ENDPOINT_SAMPLE_POOL_PORT": "9100",
                "RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_HOST": "models", "RL_INFRA_ENDPOINT_MODEL_DISTRIBUTOR_PORT": "9200",
                "RL_INFRA_ENDPOINT_LEARNER_METRICS_PORT": "9006",
            })
        self.assertFalse(managed["dashboard"]["enabled"])
        self.assertNotIn("aiserver_status", managed)

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

        runtime.pool.stub = SimpleNamespace(GetBatch=get_batch, GetStatus=get_status)
        result = runtime.pool._get_batch_recovering(ready_authority=authority, deadline=time.monotonic() + 5)
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

        runtime.pool.stub = SimpleNamespace(
            GetBatch=lambda request, timeout: training_pb2.GetBatchRsp(
                result=training_pb2.GET_BATCH_RESULT_BUSY, sample_pool=authority), GetStatus=status)
        self.assertIsNone(runtime.pool._get_batch_recovering(ready_authority=authority, deadline=time.monotonic() + 3))
        self.assertLess(query_times[0], lease_expires)
        self.assertGreaterEqual(query_times[-1], lease_expires)
        self.assertEqual(runtime._metrics_context["disposition"], "READY")

        for stop_first in (True, False):
            with self.subTest(stop=stop_first):
                query_times.clear()
                runtime.pool._mark_sample_wait("GET_BATCH_OUTCOME_UNKNOWN", "GetBatch", "uncertain", 1)
                if stop_first: _stop_requested.set()
                self.assertFalse(runtime.pool._reconcile_get_batch_outcome(authority, "uncertain",
                    request_deadline=time.monotonic() + 2, deadline=time.monotonic() + 0.02))
                self.assertEqual(query_times, [])
                self.assertEqual(runtime._metrics_context["disposition"], "GET_BATCH_OUTCOME_UNKNOWN")
                _stop_requested.clear()
