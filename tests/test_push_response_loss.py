import os
import socket
import subprocess
import time
import unittest
from concurrent import futures

import grpc

from proto import maze_pb2, maze_pb2_grpc


def available_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class ResponseLossProxy(maze_pb2_grpc.SampleDistributorServiceServicer):
    def __init__(self, backend):
        self._backend = backend
        self._drop_next_push_response = True

    def PushSamples(self, request, context):
        response = self._backend.PushSamples(request, timeout=2.0)
        if self._drop_next_push_response:
            self._drop_next_push_response = False
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "test proxy dropped accepted response",
            )
        return response

    def GetStatus(self, request, context):
        del context
        return self._backend.GetStatus(request, timeout=2.0)


class PushResponseLossTest(unittest.TestCase):
    def test_retry_is_deduplicated_after_accepted_response_is_lost(self):
        backend_target = os.environ.get("MAZE_TEST_BACKEND_TARGET", "")
        process = None
        if not backend_target:
            executable = os.environ.get(
                "MAZE_TEST_DISTRIBUTOR_EXECUTABLE", ""
            )
            config = os.environ.get("MAZE_TEST_DISTRIBUTOR_CONFIG", "")
            if not executable or not config:
                self.skipTest(
                    "set MAZE_TEST_DISTRIBUTOR_EXECUTABLE and "
                    "MAZE_TEST_DISTRIBUTOR_CONFIG for the live fault test"
                )
            backend_port = available_port()
            backend_target = f"127.0.0.1:{backend_port}"
            environment = os.environ.copy()
            environment["MAZE_SAMPLE_DISTRIBUTOR_PORT"] = str(backend_port)
            process = subprocess.Popen(
                [executable, config],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        backend_channel = grpc.insecure_channel(
            backend_target
        )
        proxy_server = None
        proxy_channel = None
        try:
            grpc.channel_ready_future(backend_channel).result(timeout=5.0)
            backend = maze_pb2_grpc.SampleDistributorServiceStub(
                backend_channel
            )
            proxy_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=2)
            )
            maze_pb2_grpc.add_SampleDistributorServiceServicer_to_server(
                ResponseLossProxy(backend), proxy_server
            )
            proxy_port = proxy_server.add_insecure_port("127.0.0.1:0")
            proxy_server.start()
            proxy_channel = grpc.insecure_channel(
                f"127.0.0.1:{proxy_port}"
            )
            grpc.channel_ready_future(proxy_channel).result(timeout=5.0)
            proxy = maze_pb2_grpc.SampleDistributorServiceStub(
                proxy_channel
            )

            batch = maze_pb2.SampleBatch(
                protocol_version=3,
                aiserver_id="aiserver-0",
                env_id="env-0",
                session_id=0,
                episode_id=0,
                agent_id=0,
                producer_instance_id="producer-0",
                fragment_seq=1,
                batch_id="response-loss-batch",
                behavior_model_version=0,
                behavior_model_checksum="a" * 64,
                bootstrap_value=0.25,
                bootstrap_valid=True,
                first_action_frame_id=0,
                last_action_frame_id=7,
            )
            for frame_id in range(8):
                batch.samples.add(
                    obs=[0.0] * 13,
                    action=0,
                    reward=0.1,
                    old_log_prob=-0.5,
                    old_vpred=0.25,
                    termination_reason=maze_pb2.TERMINATION_REASON_ACTIVE,
                    action_frame_id=frame_id,
                )

            with self.assertRaises(grpc.RpcError) as first_call:
                proxy.PushSamples(batch, timeout=2.0)
            self.assertEqual(
                first_call.exception.code(), grpc.StatusCode.UNAVAILABLE
            )

            retry = proxy.PushSamples(batch, timeout=2.0)
            self.assertEqual(retry.result, maze_pb2.PUSH_RESULT_DUPLICATE)
            status = proxy.GetStatus(
                maze_pb2.DistributorStatusReq(),
                timeout=2.0,
            )
            self.assertEqual(status.accepted_unique_samples, 8)
            self.assertEqual(status.duplicate_push_attempt_count, 1)
        finally:
            if proxy_channel is not None:
                proxy_channel.close()
            if proxy_server is not None:
                proxy_server.stop(grace=0).wait()
            backend_channel.close()
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
