import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunLauncherTest(unittest.TestCase):
    def test_shell_entrypoints_are_syntactically_valid(self):
        for relative in (
            "run.sh",
            "build_image.sh",
            "build_smoke_model_artifact.sh",
            "scripts/dev_container.sh",
            "scripts/entrypoint.sh",
        ):
            result = subprocess.run(
                ["bash", "-n", str(ROOT / relative)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_uses_task_neutral_deployment_variables(self):
        document = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("RL_LOCAL_TRAIN_ROOT", document)
        self.assertIn("RL_SAMPLE_POOL_PORT", document)
        self.assertIn("RL_MODEL_DISTRIBUTOR_PORT", document)
        self.assertIn("-m main.training_runtime", document)
        self.assertNotIn("MAZE_", document)

    def test_launcher_preserves_owned_lifecycle_boundaries(self):
        document = (ROOT / "run.sh").read_text(encoding="utf-8")
        self.assertIn("training_lock", document)
        self.assertIn('trap shutdown EXIT TERM INT', document)
        self.assertIn('rm -rf -- "${training_lock}"', document)
        self.assertIn('exit "${child_status}"', document)
        self.assertIn('if [ -z "${pid}" ]; then\n        return 125', document)
        self.assertIn('training_child_status="${child_status}"', document)
        self.assertNotIn('wait "${process}"\n            exit $?', document)
        self.assertIn("expected_local_train_root", document)
        self.assertNotIn("docker rm", document)

    def test_quiesce_does_not_publish_success_for_failed_training_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/learner_config.yaml").write_text(
                "training: {}\n", encoding="utf-8"
            )
            (root / "models/local-train").mkdir(parents=True)
            (root / "main").mkdir()
            (root / "main/__init__.py").write_text("", encoding="utf-8")
            (root / "main/training_runtime.py").write_text(
                """\
import os
import signal
import sys
import time
from pathlib import Path

Path(os.environ["FAKE_TRAINING_STARTED"]).write_text("ready", encoding="utf-8")

def stop(_signum, _frame):
    raise SystemExit(int(os.environ["FAKE_TRAIN_TERM_STATUS"]))

signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.05)
""",
                encoding="utf-8",
            )
            (root / "tools").mkdir()
            (root / "tools/metrics_server.py").write_text(
                "import time\nwhile True: time.sleep(1)\n",
                encoding="utf-8",
            )
            launcher = root / "run.sh"
            launcher.write_text(
                (ROOT / "run.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            def reserve_port():
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    return listener.getsockname()[1]

            def write_listener(path, port):
                path.write_text(
                    f"""\
#!/usr/bin/env python3
import socket

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", {port}))
listener.listen()
while True:
    connection, _ = listener.accept()
    connection.close()
""",
                    encoding="utf-8",
                )
                path.chmod(0o755)

            sample_port = reserve_port()
            model_port = reserve_port()
            sample_bin = root / "sample-pool"
            model_bin = root / "model-distributor"
            write_listener(sample_bin, sample_port)
            write_listener(model_bin, model_port)
            sample_config = root / "sample-pool.yaml"
            model_config = root / "model-distributor.yaml"
            sample_config.write_text("{}\n", encoding="utf-8")
            model_config.write_text("{}\n", encoding="utf-8")
            started = root / "training-started"
            success = root / "quiesced"
            failure = root / "quiesce-failed"
            environment = {
                **os.environ,
                "SAMPLE_DISTRIBUTOR_BIN": str(sample_bin),
                "SAMPLE_DISTRIBUTOR_CONFIG": str(sample_config),
                "MODEL_DISTRIBUTOR_BIN": str(model_bin),
                "MODEL_DISTRIBUTOR_CONFIG": str(model_config),
                "RL_SAMPLE_POOL_PORT": str(sample_port),
                "RL_MODEL_DISTRIBUTOR_PORT": str(model_port),
                "RL_METRICS_PORT": str(reserve_port()),
                "RL_QUIESCE_MARKER": str(success),
                "RL_QUIESCE_FAILURE_MARKER": str(failure),
                "FAKE_TRAINING_STARTED": str(started),
                "FAKE_TRAIN_TERM_STATUS": "17",
            }
            process = subprocess.Popen(
                ["bash", str(launcher), "training"],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll())
                    time.sleep(0.02)
                self.assertTrue(started.exists())
                os.kill(process.pid, signal.SIGUSR1)
                deadline = time.monotonic() + 5
                while not failure.exists() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll())
                    time.sleep(0.02)
                self.assertFalse(success.exists())
                self.assertEqual(failure.read_text(encoding="utf-8"), "17\n")
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
