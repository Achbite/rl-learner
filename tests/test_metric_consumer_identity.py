import unittest

from proto import common_pb2, training_pb2
from src.metrics.metric_events import LearnerMetricEventService


def _service_identity(component: str, instance_id: str, epoch: int = 1):
    return common_pb2.ServiceInstanceIdentity(
        component=component,
        instance_id=instance_id,
        lifecycle_epoch=epoch,
    )


class _MetricStore:
    def __init__(self):
        self.bound_consumer = None
        self.bind_calls = 0

    def export_availability(self, _source):
        return 0, 0, False

    def bind_export_consumer(self, _source, consumer):
        self.bind_calls += 1
        identity = consumer.SerializeToString(deterministic=True)
        if self.bound_consumer is None:
            self.bound_consumer = identity
        return self.bound_consumer == identity

    def export_cursor(self, source):
        return training_pb2.MetricBatchCursor(source=source)

    def next_export_batch(self, _source, _cursor):
        return None

    def wait_for_export_change(self, _timeout):
        return None

    def acknowledge_export(self, _source, _cursor):
        raise AssertionError("zero cursor must be already applied")


class _ActiveContext:
    @staticmethod
    def is_active():
        return True


class MetricConsumerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.contract = common_pb2.ContractIdentity(
            package_name="rl-contracts",
            package_version="0.14.0",
        )
        self.source = _service_identity("learner", "fresh-learner", 2)
        self.old_source = _service_identity("learner", "stopped-learner", 1)
        self.old_consumer = _service_identity("rl-infra", "old-binding", 1)
        self.current_consumer = _service_identity(
            "rl-infra", "current-binding", 2
        )

    def _service(self):
        store = _MetricStore()
        return store, LearnerMetricEventService(
            store=store,
            contract=self.contract,
            source=self.source,
        )

    def test_wrong_get_cursor_source_cannot_pin_fresh_journal(self):
        store, service = self._service()
        rejected = service.GetMetricBatch(
            training_pb2.GetMetricBatchReq(
                contract=self.contract,
                consumer=self.old_consumer,
                cursor=training_pb2.MetricBatchCursor(
                    source=self.old_source
                ),
                max_events=1,
                max_bytes=1024,
                wait_timeout_ms=0,
            ),
            _ActiveContext(),
        )
        self.assertEqual(
            rejected.result,
            training_pb2.METRIC_BATCH_RESULT_REJECTED_IDENTITY,
        )
        self.assertEqual(store.bind_calls, 0)

        accepted = service.GetMetricBatch(
            training_pb2.GetMetricBatchReq(
                contract=self.contract,
                consumer=self.current_consumer,
                cursor=training_pb2.MetricBatchCursor(source=self.source),
                max_events=1,
                max_bytes=1024,
                wait_timeout_ms=0,
            ),
            _ActiveContext(),
        )
        self.assertEqual(accepted.result, training_pb2.METRIC_BATCH_RESULT_WAIT)
        self.assertEqual(store.bind_calls, 1)

    def test_wrong_ack_cursor_source_cannot_pin_fresh_journal(self):
        store, service = self._service()
        rejected = service.AckMetricBatch(
            training_pb2.AckMetricBatchReq(
                contract=self.contract,
                consumer=self.old_consumer,
                cursor=training_pb2.MetricBatchCursor(
                    source=self.old_source
                ),
            ),
            _ActiveContext(),
        )
        self.assertEqual(
            rejected.result,
            training_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_IDENTITY,
        )
        self.assertEqual(store.bind_calls, 0)

        accepted = service.AckMetricBatch(
            training_pb2.AckMetricBatchReq(
                contract=self.contract,
                consumer=self.current_consumer,
                cursor=training_pb2.MetricBatchCursor(source=self.source),
            ),
            _ActiveContext(),
        )
        self.assertEqual(
            accepted.result,
            training_pb2.METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED,
        )
        self.assertEqual(store.bind_calls, 1)


if __name__ == "__main__":
    unittest.main()
