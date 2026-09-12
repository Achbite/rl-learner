"""Bootstrap result propagation; no training loop or training-effect oracle."""
import concurrent.futures
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import grpc
import yaml

from main.training_runtime import TrainingRuntime, _stop_requested
from proto.common import identity_pb2 as common
from proto.training import model_identity_pb2 as identity
from proto.training import training_pb2 as wire
from proto.training import training_pb2_grpc as rpc


def service_identity(component, instance):
    return common.ServiceInstanceIdentity(component=component, instance_id=instance, lifecycle_epoch=1)


def runtime_with(stub, timeout=None):
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.model_stub = stub
    runtime.initial_model_ack_timeout = timeout
    runtime.logger = logging.getLogger("test.bootstrap")
    return runtime


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class ReadyPool(rpc.SamplePoolIngressServiceServicer):
    def GetStatus(self, request, context):
        return wire.SamplePoolStatusRsp(
            sample_pool=service_identity("sample-pool", "bootstrap-test-pool"),
            ready=True, ingress_ready=True, pool_ready=True,
            backend_type=wire.SAMPLE_BACKEND_TYPE_LOCAL_MEMORY)


class ModelStartupContractTest(unittest.TestCase):
    def setUp(self):
        _stop_requested.clear()
        self.addCleanup(_stop_requested.clear)
        self.model = identity.ModelIdentity(model_lineage_id="bootstrap-test", model_step=0)
        self.manifest = wire.ModelArtifactManifest(identity=self.model)
        self.authority = service_identity("model-distributor", "distributor-test")
        self.producer = service_identity("aiserver", "aiserver-test")

    def status(self, model=None, state=wire.MODEL_LOAD_STATUS_FAILED, message="ONNX prepare: shape mismatch"):
        return wire.ModelDistributorStatusRsp(
            ready=True, distributor=self.authority, latest_ack_model=model or self.model,
            latest_ack_aiserver=self.producer, latest_ack_status=state, last_error=message)

    def test_exact_failed_ack_preserves_cause(self):
        observed = self.status()
        runtime = runtime_with(SimpleNamespace(GetModelDistributorStatus=lambda *a, **k: observed))
        with self.assertRaises(RuntimeError) as raised:
            runtime._wait_initial_model_loaded({"manifest": self.manifest})
        self.assertIn(observed.last_error, str(raised.exception))
        self.assertIn("model_step=0", str(raised.exception))
        self.assertIn(self.producer.instance_id, str(raised.exception))

    def test_other_model_failed_ack_does_not_fail_target(self):
        other = identity.ModelIdentity(model_lineage_id="other-target", model_step=0)
        replies = iter([self.status(other), self.status(state=wire.MODEL_LOAD_STATUS_LOADED, message="loaded")])
        runtime = runtime_with(SimpleNamespace(GetModelDistributorStatus=lambda *a, **k: next(replies)), timeout=2)
        self.assertEqual(runtime._wait_initial_model_loaded({"manifest": self.manifest}), self.producer)

    def test_no_ack_waits_until_explicit_stop(self):
        observed = wire.ModelDistributorStatusRsp(ready=True, distributor=self.authority)
        queried = threading.Event()

        def status(*args, **kwargs):
            queried.set()
            return observed

        runtime = runtime_with(SimpleNamespace(GetModelDistributorStatus=status))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(runtime._wait_initial_model_loaded, {"manifest": self.manifest})
            try:
                self.assertTrue(queried.wait(2))
                self.assertFalse(pending.done(), "absence of ACK remains a legitimate wait")
            finally:
                _stop_requested.set()
            self.assertIsNone(pending.result(timeout=2))
        self.assertEqual(observed.latest_ack_status, wire.MODEL_LOAD_STATUS_UNSPECIFIED)

    def test_real_aiserver_failed_ack_reaches_real_distributor_and_learner(self):
        # Test dependencies are explicit; no search for old binaries or running services.
        ai_binary = Path(os.environ["RL_AISERVER_TEST_BINARY"])
        distributor_binary = Path(os.environ["RL_MODEL_DISTRIBUTOR_TEST_BINARY"])
        ai_source = Path(os.environ["RL_AISERVER_SOURCE_DIR"])
        for binary in (ai_binary, distributor_binary):
            self.assertTrue(binary.is_file() and os.access(binary, os.X_OK), str(binary))
        with tempfile.TemporaryDirectory(prefix="bootstrap-contract-") as directory:
            root = Path(directory)
            model_path = root / "published" / "policy.onnx"
            model_path.parent.mkdir()
            shutil.copyfile(ai_source / "tests/fixtures/fixed_policy.onnx", model_path)
            manifest = wire.ModelArtifactManifest(
                identity=self.model, size_bytes=model_path.stat().st_size,
                trained_samples=0, published_at_unix_ms=int(time.time() * 1000))
            port = available_port()
            distributor_config = root / "distributor.yaml"
            distributor_config.write_text(yaml.safe_dump({
                "server": {"listen_port": port},
                "model": {"artifact_root": str(root), "chunk_bytes": 1048576, "retention_steps": 101},
            }))
            processes = []
            logs = []
            pool = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
            rpc.add_SamplePoolIngressServiceServicer_to_server(ReadyPool(), pool)
            pool_port = pool.add_insecure_port("127.0.0.1:0")
            pool.start()
            channel = grpc.insecure_channel(f"127.0.0.1:{port}")

            def launch(command, name):
                log = open(root / f"{name}.log", "w+")
                logs.append(log)
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                return process

            try:
                launch([str(distributor_binary), str(distributor_config)], "distributor")
                grpc.channel_ready_future(channel).result(timeout=5)
                stub = rpc.ModelDistributorServiceStub(channel)
                registered = stub.RegisterModel(wire.RegisterModelReq(
                    manifest=manifest, local_artifact_path=str(model_path)), timeout=2)
                self.assertEqual(registered.result, wire.MODEL_REGISTER_RESULT_REGISTERED, registered.message)
                config = yaml.safe_load((ai_source / "configs/server_config.yaml").read_text())
                config["server"].update(run_mode="training", listen_port=available_port())
                config["model"].update(local_train_dir=str(root / "cache" / "train"),
                                       expected_obs_dim=18, startup_timeout_ms=3000)
                config["model_distribution"].update(host="127.0.0.1", port=port, rpc_timeout_ms=500)
                config["sample_distributor"].update(host="127.0.0.1", port=pool_port, health_timeout_ms=1000)
                ai_config = root / "aiserver.yaml"
                ai_config.write_text(yaml.safe_dump(config))
                ai = launch([str(ai_binary), "--config", str(ai_config)], "aiserver")
                runtime = runtime_with(stub, timeout=6)
                with self.assertRaises(RuntimeError) as raised:
                    runtime._wait_initial_model_loaded({"manifest": manifest})
                status = stub.GetModelDistributorStatus(wire.ModelDistributorStatusReq(), timeout=2)
                self.assertEqual(status.latest_ack_model, self.model)
                self.assertEqual(status.latest_ack_status, wire.MODEL_LOAD_STATUS_FAILED)
                self.assertIn("ONNX", status.last_error)
                self.assertIn("shape", status.last_error)
                self.assertIn(status.last_error, str(raised.exception))
                self.assertIn(status.latest_ack_aiserver.instance_id, str(raised.exception))
                # Startup exits before serving task RPC. Its existing final-metric
                # shutdown wait remains bounded; this test does not alter that policy.
                self.assertEqual(ai.wait(timeout=15), 1)
                self.assertIn(status.last_error, (root / "aiserver.log").read_text())
            except Exception:
                for log in logs:
                    log.flush()
                    log.seek(0)
                    print(f"\n{Path(log.name).name}:\n{log.read()}")
                raise
            finally:
                for process in reversed(processes):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                channel.close()
                pool.stop(0).wait(2)
                for log in logs:
                    log.close()
