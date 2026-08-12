import unittest

from main.training_runtime import training_chain_status


def identity(version: int, artifact: str) -> dict:
    return {
        "model_lineage_id": "maze-fixed-map-seed-0",
        "model_version": version,
        "artifact_digest": artifact * 64,
        "manifest_digest": chr(ord(artifact) + 1) * 64,
    }


class TrainingChainReadinessTest(unittest.TestCase):
    def healthy_components(self):
        learner_model = identity(2, "b")
        actor_model = identity(1, "a")
        return (
            {
                "ready": True,
                "instance_id": "actor-current",
                "client_session_recent": True,
                "model_identity": actor_model,
            },
            {
                "ready": True,
                "instance_id": "pool-current",
                "ingress_ready": True,
                "pool_ready": True,
            },
            {
                "model_identity": learner_model,
                "actual_batch_size": 512,
                "policy_lag": 1,
                "max_policy_lag": 1,
            },
            {
                "ready": True,
                "instance_id": "model-distributor-current",
                "latest_model_identity": learner_model,
                "latest_ack_model_identity": actor_model,
                "latest_ack_status": "MODEL_LOAD_STATUS_LOADED",
            },
        )

    def test_allows_one_version_lag_with_exact_identities(self):
        chain = training_chain_status(*self.healthy_components())
        self.assertTrue(chain["ready"])
        self.assertEqual(chain["model_lag"], 1)

    def test_rejects_digest_or_ack_identity_mismatch(self):
        actor, pool, learner, model = self.healthy_components()
        actor["model_identity"] = identity(1, "c")
        model["latest_ack_status"] = "MODEL_LOAD_STATUS_FAILED"
        chain = training_chain_status(actor, pool, learner, model)
        self.assertFalse(chain["ready"])
        self.assertIn("actor_model_ack_mismatch", chain["reasons"])
        self.assertIn("actor_model_ack_not_loaded", chain["reasons"])

    def test_requires_service_instances_and_pool_readiness(self):
        actor, pool, learner, model = self.healthy_components()
        actor["instance_id"] = ""
        pool["pool_ready"] = False
        chain = training_chain_status(actor, pool, learner, model)
        self.assertIn("actor_instance_missing", chain["reasons"])
        self.assertIn("sample_pool_pool_ready_false", chain["reasons"])

    def test_requires_recent_client_activity_not_only_a_session_record(self):
        actor, pool, learner, model = self.healthy_components()
        actor["active_sessions"] = 1
        actor["client_session_recent"] = False
        chain = training_chain_status(actor, pool, learner, model)
        self.assertFalse(chain["ready"])
        self.assertIn("client_session_not_recent", chain["reasons"])

    def test_rejects_policy_lag_outside_bound(self):
        actor, pool, learner, model = self.healthy_components()
        learner["policy_lag"] = 2
        chain = training_chain_status(actor, pool, learner, model)
        self.assertIn("training_policy_lag_invalid", chain["reasons"])


if __name__ == "__main__":
    unittest.main()
