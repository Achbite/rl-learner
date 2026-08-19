import tempfile
import unittest
from pathlib import Path

from proto import training_pb2
from src.contracts.identity import service_identity
from src.metrics.metric_events import (
    AIServerMetricRelay,
    MetricSchemaCatalog,
    RawMetricBatchStore,
    default_metric_schema_directory,
)
from tests.test_metric_event_store import contract, episode_batch


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    @staticmethod
    def _format(message, args):
        return message % args if args else message

    def info(self, message, *args, **_kwargs):
        self.infos.append(self._format(message, args))

    def warning(self, message, *args, **_kwargs):
        self.warnings.append(self._format(message, args))

    def error(self, message, *args, **_kwargs):
        self.errors.append(self._format(message, args))


class _StatusStub:
    def __init__(self, contract_identity, source):
        self.contract_identity = contract_identity
        self.source = source

    def GetAIServerStatus(self, _request, timeout):
        self.timeout = timeout
        return training_pb2.AIServerStatusRsp(
            contract=self.contract_identity,
            aiserver=self.source,
        )


class _EventStub:
    def __init__(self, source, batch):
        self.source = source
        self.batch = batch
        self.acks = []
        self.gets = []

    def GetMetricBatch(self, request, timeout):
        self.gets.append((request, timeout))
        return training_pb2.GetMetricBatchRsp(
            ret_code=0,
            result=training_pb2.METRIC_BATCH_RESULT_DELIVERED,
            batch=self.batch,
            producer=self.source,
        )

    def AckMetricBatch(self, request, timeout):
        self.acks.append((request, timeout))
        return training_pb2.AckMetricBatchRsp(
            ret_code=0,
            result=training_pb2.METRIC_BATCH_ACK_RESULT_APPLIED,
            producer=self.source,
            committed_cursor=request.cursor,
        )


class AIServerMetricRelayTest(unittest.TestCase):
    def test_ack_is_sent_only_after_raw_batch_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            contract_identity = contract()
            catalog = MetricSchemaCatalog.load(
                default_metric_schema_directory()
            )
            store = RawMetricBatchStore(
                Path(directory) / "metric-events.sqlite3",
                contract_identity,
                catalog,
            )
            source = service_identity("rl-aiserver", "aiserver-0", 7)
            store.activate_source("aiserver", source)
            batch = episode_batch(store, source)
            event_stub = _EventStub(source, batch)
            relay = AIServerMetricRelay(
                store=store,
                contract=contract_identity,
                consumer=service_identity("learner", "learner-0", 1),
                status_stub=_StatusStub(contract_identity, source),
                event_stub=event_stub,
                logger=_Logger(),
            )

            observed = []
            original_ack = event_stub.AckMetricBatch

            def observing_ack(request, timeout):
                observed.append(store.pending_batch(source) == batch)
                return original_ack(request, timeout)

            event_stub.AckMetricBatch = observing_ack
            relay._pull_once(source)

            self.assertEqual(observed, [True])
            self.assertEqual(len(event_stub.acks), 1)
            self.assertEqual(store.snapshot()["pending_batch_count"], 0)
            self.assertEqual(store.snapshot()["committed_batch_count"], 1)
            store.close()

    def test_recovered_pending_batch_is_acked_without_get(self):
        with tempfile.TemporaryDirectory() as directory:
            contract_identity = contract()
            catalog = MetricSchemaCatalog.load(
                default_metric_schema_directory()
            )
            path = Path(directory) / "metric-events.sqlite3"
            source = service_identity("rl-aiserver", "aiserver-0", 7)
            first = RawMetricBatchStore(path, contract_identity, catalog)
            batch = episode_batch(first, source)
            first.persist_batch("aiserver", batch)
            first.close()

            recovered = RawMetricBatchStore(path, contract_identity, catalog)
            event_stub = _EventStub(source, batch)
            relay = AIServerMetricRelay(
                store=recovered,
                contract=contract_identity,
                consumer=service_identity("learner", "learner-0", 1),
                status_stub=_StatusStub(contract_identity, source),
                event_stub=event_stub,
                logger=_Logger(),
            )
            relay._pull_once(source)
            self.assertEqual(len(event_stub.acks), 1)
            self.assertEqual(event_stub.gets, [])
            self.assertEqual(recovered.snapshot()["pending_batch_count"], 0)
            recovered.close()
