import unittest
from types import SimpleNamespace
from unittest import mock

import grpc

from main.training_runtime import TrainingRuntime


class UnavailableRpcError(grpc.RpcError):
    def details(self):
        return "connection refused"


class StopEvent:
    def __init__(self):
        self.stopped = False
        self.wait_count = 0

    def is_set(self):
        return self.stopped

    def wait(self, _timeout):
        self.wait_count += 1
        return self.stopped


def runtime_for_logging():
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.trainer = SimpleNamespace(model_version=0)
    runtime.train_batch_size = 512
    runtime.logger = mock.Mock()
    runtime.metrics = mock.Mock()
    runtime.publisher = SimpleNamespace(
        archive_on_graceful_shutdown=False
    )
    runtime._startup_mode = "fresh"
    runtime._initialize_models = mock.Mock()
    runtime._metrics_stop = mock.Mock()
    runtime._metrics_stop.wait.return_value = True
    runtime._metrics_thread = None
    runtime._metrics_loop = mock.Mock()
    runtime._record_metrics = mock.Mock()
    runtime._drain_stale = mock.Mock()
    runtime._reconcile_receipts = mock.Mock()
    runtime._process_delivery = mock.Mock()
    runtime._drain_shutdown = mock.Mock()
    runtime.sample_channel = mock.Mock()
    runtime.model_channel = mock.Mock()
    runtime.aiserver_channel = mock.Mock()
    return runtime


class RuntimeConnectionLoggingTest(unittest.TestCase):
    def test_startup_wait_is_logged_once_at_info_then_connection_is_logged(self):
        runtime = runtime_for_logging()
        stop_event = StopEvent()
        attempts = iter(
            [
                UnavailableRpcError(),
                UnavailableRpcError(),
                None,
            ]
        )

        def select_batch():
            result = next(attempts)
            if isinstance(result, grpc.RpcError):
                raise result
            stop_event.stopped = True
            return result

        runtime._select_batch = select_batch

        with mock.patch(
            "main.training_runtime._stop_requested", stop_event
        ):
            runtime.run()

        self.assertEqual(
            [
                call
                for call in runtime.logger.info.call_args_list
                if call.args[0] == "等待 LocalSampleService: %s"
            ],
            [
                mock.call(
                    "等待 LocalSampleService: %s",
                    "connection refused",
                )
            ],
        )
        runtime.logger.info.assert_any_call(
            "LocalSampleService 连接%s", "已建立"
        )
        runtime.logger.warning.assert_not_called()
        self.assertEqual(stop_event.wait_count, 2)
        runtime._drain_shutdown.assert_called_once()
        runtime.metrics.close.assert_called_once()

    def test_runtime_disconnect_is_warned_once_then_recovery_is_logged(self):
        runtime = runtime_for_logging()
        stop_event = StopEvent()
        attempts = iter(
            [
                None,
                UnavailableRpcError(),
                UnavailableRpcError(),
                None,
            ]
        )
        attempt_count = 0

        def select_batch():
            nonlocal attempt_count
            result = next(attempts)
            attempt_count += 1
            if isinstance(result, grpc.RpcError):
                raise result
            if attempt_count == 4:
                stop_event.stopped = True
            return result

        runtime._select_batch = select_batch

        with mock.patch(
            "main.training_runtime._stop_requested", stop_event
        ):
            runtime.run()

        self.assertEqual(
            runtime.logger.warning.call_args_list,
            [
                mock.call(
                    "等待 LocalSampleService: %s",
                    "connection refused",
                )
            ],
        )
        runtime.logger.info.assert_any_call(
            "LocalSampleService 连接%s", "已恢复"
        )
        self.assertEqual(stop_event.wait_count, 2)
        runtime._drain_shutdown.assert_called_once()
        runtime.metrics.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
