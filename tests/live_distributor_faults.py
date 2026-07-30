import argparse
import json
import os
import sys
import time

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import maze_pb2, maze_pb2_grpc


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_batch(run_id, batch_id, sample_count, sequence):
    batch = maze_pb2.SampleBatch(
        protocol_version=2,
        run_id=run_id,
        aiserver_id="aiserver-0",
        env_id="env-0",
        session_id=0,
        episode_id=0,
        agent_id=0,
        producer_instance_id="fault-producer-0",
        fragment_seq=sequence,
        batch_id=batch_id,
        behavior_model_version=0,
        behavior_model_checksum="a" * 64,
        bootstrap_value=0.25,
        bootstrap_valid=True,
        first_action_frame_id=0,
        last_action_frame_id=sample_count - 1,
    )
    for index in range(sample_count):
        batch.samples.add(
            obs=[float(index)] * 13,
            action=index % 9,
            reward=0.1,
            old_log_prob=-0.5,
            old_vpred=0.25,
            termination_reason=maze_pb2.TERMINATION_REASON_ACTIVE,
            action_frame_id=index,
        )
    return batch


def get_batch(stub, run_id, consumer_id, lease_timeout_ms):
    return stub.GetBatch(
        maze_pb2.GetBatchReq(
            run_id=run_id,
            consumer_instance_id=consumer_id,
            batch_size=8,
            timeout_ms=100,
            lease_timeout_ms=lease_timeout_ms,
            behavior_model_version=0,
            selection_policy=(
                maze_pb2.BATCH_SELECTION_POLICY_TARGET_ONLY
            ),
        ),
        timeout=2.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--run-id", default="live-fault-run")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.target)
    grpc.channel_ready_future(channel).result(timeout=5.0)
    stub = maze_pb2_grpc.SampleDistributorServiceStub(channel)

    accepted = stub.PushSamples(
        make_batch(args.run_id, "fault-batch-0", 8, 1),
        timeout=2.0,
    )
    duplicate = stub.PushSamples(
        make_batch(args.run_id, "fault-batch-0", 8, 1),
        timeout=2.0,
    )
    capacity = stub.PushSamples(
        make_batch(args.run_id, "fault-batch-1", 1, 2),
        timeout=2.0,
    )
    require(
        accepted.result == maze_pb2.PUSH_RESULT_ACCEPTED,
        "initial batch was not accepted",
    )
    require(
        duplicate.result == maze_pb2.PUSH_RESULT_DUPLICATE,
        "duplicate batch was not deduplicated",
    )
    require(
        capacity.result == maze_pb2.PUSH_RESULT_REJECTED_CAPACITY,
        "full pool did not reject a unique batch",
    )

    first = get_batch(stub, args.run_id, "consumer-exits", 300)
    require(
        first.result == maze_pb2.GET_BATCH_RESULT_LEASED,
        "first consumer did not obtain a lease",
    )
    first_delivery_id = first.delivery_id
    channel.close()

    time.sleep(0.45)
    recovery_channel = grpc.insecure_channel(args.target)
    grpc.channel_ready_future(recovery_channel).result(timeout=5.0)
    recovery = maze_pb2_grpc.SampleDistributorServiceStub(recovery_channel)
    second = get_batch(recovery, args.run_id, "consumer-recovery", 1000)
    require(
        second.result == maze_pb2.GET_BATCH_RESULT_LEASED,
        "expired lease was not redelivered",
    )
    require(
        second.delivery_id != first_delivery_id,
        "redelivery reused the expired delivery identity",
    )
    require(
        [batch.batch_id for batch in second.batches] == ["fault-batch-0"],
        "redelivery did not preserve the original fragment identity",
    )

    ack = recovery.AckBatch(
        maze_pb2.AckBatchReq(
            run_id=args.run_id,
            consumer_instance_id="consumer-recovery",
            delivery_id=second.delivery_id,
            disposition=maze_pb2.ACK_DISPOSITION_TRAINED,
            train_update_id="live-fault-update-0",
        ),
        timeout=2.0,
    )
    duplicate_ack = recovery.AckBatch(
        maze_pb2.AckBatchReq(
            run_id=args.run_id,
            consumer_instance_id="consumer-recovery",
            delivery_id=second.delivery_id,
            disposition=maze_pb2.ACK_DISPOSITION_TRAINED,
            train_update_id="live-fault-update-0",
        ),
        timeout=2.0,
    )
    require(
        ack.result == maze_pb2.DELIVERY_RESULT_APPLIED,
        "recovered delivery Ack was not applied",
    )
    require(
        duplicate_ack.result == maze_pb2.DELIVERY_RESULT_ALREADY_APPLIED,
        "Ack retry was not idempotent",
    )

    status = recovery.GetStatus(
        maze_pb2.DistributorStatusReq(run_id=args.run_id),
        timeout=2.0,
    )
    require(status.accepted_unique_samples == 8, "accepted count drifted")
    require(status.acked_unique_samples == 8, "Ack count drifted")
    require(status.ready_queue_samples == 0, "ready queue was not drained")
    require(status.leased_samples == 0, "lease was not cleared")
    require(status.expired_lease_count == 1, "lease expiry was not counted")
    require(status.redelivery_count == 1, "redelivery was not counted")
    require(
        status.duplicate_push_attempt_count == 1,
        "duplicate Push attempt count drifted",
    )
    require(
        status.rejected_push_attempt_count == 1,
        "capacity rejection count drifted",
    )

    print(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "ok": True,
                "accepted_result": maze_pb2.PushResult.Name(
                    accepted.result
                ),
                "duplicate_result": maze_pb2.PushResult.Name(
                    duplicate.result
                ),
                "capacity_result": maze_pb2.PushResult.Name(
                    capacity.result
                ),
                "first_delivery_id": first_delivery_id,
                "redelivery_id": second.delivery_id,
                "acked_unique_samples": status.acked_unique_samples,
                "expired_lease_count": status.expired_lease_count,
                "redelivery_count": status.redelivery_count,
                "duplicate_push_attempt_count": (
                    status.duplicate_push_attempt_count
                ),
                "rejected_push_attempt_count": (
                    status.rejected_push_attempt_count
                ),
                "ack_retry_result": maze_pb2.DeliveryResult.Name(
                    duplicate_ack.result
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    recovery_channel.close()


if __name__ == "__main__":
    main()
