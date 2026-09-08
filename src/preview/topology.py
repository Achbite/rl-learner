"""Read-only local training topology projection."""

def _identity_dict(document: dict | None) -> dict:
    if not document:
        return {}
    identity = document.get("identity", document)
    return {
        "model_lineage_id": str(identity.get("model_lineage_id", "")),
        "model_step": int(identity.get("model_step", -1)),
    }


def _identity_equal(left: dict | None, right: dict | None) -> bool:
    return bool(left and right) and _identity_dict(left) == _identity_dict(right)


def training_chain_status(
    actor: dict,
    sample_pool: dict,
    learner: dict,
    model: dict,
    error: str = "",
) -> dict:
    """Return task-neutral readiness from exact service/model identities."""
    reasons: list[str] = []
    if error:
        reasons.append("learner_update_error")
    for name, document in (
        ("actor", actor),
        ("sample_pool", sample_pool),
        ("model_distributor", model),
    ):
        if document.get("error"):
            reasons.append(f"{name}_status_error")
        if not document.get("ready"):
            reasons.append(f"{name}_not_ready")
        if not document.get("instance_id"):
            reasons.append(f"{name}_instance_missing")
    if sample_pool.get("ingress_ready") is not True:
        reasons.append("sample_pool_ingress_ready_false")

    learner_model = learner.get("model_identity", {})
    actor_model = actor.get("model_identity", {})
    published_model = model.get("latest_model_identity", {})
    acknowledged_model = model.get("latest_ack_model_identity", {})
    if not _identity_equal(learner_model, published_model):
        reasons.append("published_model_identity_mismatch")
    if not _identity_equal(actor_model, acknowledged_model):
        reasons.append("actor_model_ack_mismatch")
    if model.get("latest_ack_status") != "MODEL_LOAD_STATUS_LOADED":
        reasons.append("actor_model_ack_not_loaded")

    learner_step = int(_identity_dict(learner_model).get("model_step", -1))
    actor_step = int(_identity_dict(actor_model).get("model_step", -1))
    model_lag = (
        None
        if learner_step < 0 or actor_step < 0
        else learner_step - actor_step
    )

    feedback = actor.get("model_feedback", {})
    if feedback.get("last_error"):
        reasons.append("actor_model_feedback_error")
    actor_state = str(actor.get("state", ""))
    actor_lifecycle_ready = actor_state == "AISERVER_STATE_READY"
    server_pod_reasons: list[str] = []
    if actor.get("error"):
        server_pod_reasons.append("actor_status_error")
    if not actor.get("instance_id"):
        server_pod_reasons.append("actor_instance_missing")
    if not actor_lifecycle_ready:
        server_pod_reasons.append("actor_lifecycle_not_ready")
    if actor_step < 0:
        server_pod_reasons.append("active_model_missing")

    staged_step = int(
        _identity_dict(actor.get("staged_model_identity", {})).get(
            "model_step", -1
        )
    )
    latest_step = int(
        _identity_dict(published_model).get("model_step", -1)
    )
    if actor_step < 0:
        model_sync_state = "waiting_for_initial_model"
        model_sync_lag = None
    elif latest_step < 0:
        model_sync_state = "unknown"
        model_sync_lag = None
    elif actor_step < latest_step:
        model_sync_state = "catching_up"
        model_sync_lag = latest_step - actor_step
    elif actor_step == latest_step:
        model_sync_state = "synchronized"
        model_sync_lag = 0
    else:
        model_sync_state = "actor_ahead"
        model_sync_lag = actor_step - latest_step
    if feedback.get("last_error"):
        model_sync_state = "failed"
    return {
        "ready": not reasons,
        "state": "ready" if not reasons else "degraded",
        "reasons": reasons,
        "model_lag": model_lag,
        "server_pod": {
            "ready": not server_pod_reasons,
            "state": (
                "running" if not server_pod_reasons else "not_ready"
            ),
            "reasons": server_pod_reasons,
        },
        "model_sync": {
            "state": model_sync_state,
            "feedback": feedback,
            "active_model_step": (
                None if actor_step < 0 else actor_step
            ),
            "staged_model_step": (
                None if staged_step < 0 else staged_step
            ),
            "latest_model_step": (
                None if latest_step < 0 else latest_step
            ),
            "lag": model_sync_lag,
        },
        "error": error,
    }
