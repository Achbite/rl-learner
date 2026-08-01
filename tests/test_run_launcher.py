import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class RunLauncherTest(unittest.TestCase):
    @staticmethod
    def prepare_minimal_root(root: Path) -> Path:
        source = Path(__file__).resolve().parents[1] / "run.sh"
        shutil.copy2(source, root / "run.sh")
        (root / "configs").mkdir()
        (root / "configs/learner_config.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        sample_pool_root = root / "sample-pool"
        (sample_pool_root / "bin").mkdir(parents=True)
        (sample_pool_root / "config").mkdir()
        (sample_pool_root / "config" / "distributor_config.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        write_executable(
            sample_pool_root / "bin" / "maze_sample_distributor",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        distributor_root = root / "model-distributor"
        (distributor_root / "bin").mkdir(parents=True)
        (distributor_root / "config").mkdir()
        (distributor_root / "config" / "model_distributor_config.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        write_executable(
            distributor_root / "bin" / "maze_model_distributor",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        return distributor_root

    def test_repository_local_services_are_started_before_training(self):
        source = Path(__file__).resolve().parents[1] / "run.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(source, root / "run.sh")
            (root / "configs").mkdir()
            (root / "configs/learner_config.yaml").write_text(
                "{}\n", encoding="utf-8"
            )

            sample_pool_root = root / "sample-pool"
            (sample_pool_root / "bin").mkdir(parents=True)
            (sample_pool_root / "config").mkdir()
            sample_pool_config = (
                sample_pool_root / "config" / "distributor_config.yaml"
            )
            sample_pool_config.write_text("{}\n", encoding="utf-8")

            distributor_root = root / "model-distributor"
            (distributor_root / "bin").mkdir(parents=True)
            (distributor_root / "config").mkdir()
            distributor_config = (
                distributor_root
                / "config"
                / "model_distributor_config.yaml"
            )
            distributor_config.write_text("{}\n", encoding="utf-8")

            server_script = root / "fake_distributor.py"
            server_script.write_text(
                "\n".join(
                    [
                        "import os",
                        "import signal",
                        "import socket",
                        "import sys",
                        "",
                        "running = True",
                        "",
                        "def stop(*_):",
                        "    global running",
                        "    running = False",
                        "",
                        "signal.signal(signal.SIGTERM, stop)",
                        "listener = socket.socket()",
                        "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)",
                        "listener.bind(('127.0.0.1', int(sys.argv[1])))",
                        "listener.listen()",
                        "listener.settimeout(0.1)",
                        "while running:",
                        "    try:",
                        "        connection, _ = listener.accept()",
                        "        connection.close()",
                        "    except socket.timeout:",
                        "        pass",
                        "listener.close()",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            startup_order = root / "startup-order.txt"
            sample_pool_marker = root / "sample-pool-argument.txt"
            write_executable(
                sample_pool_root / "bin" / "maze_sample_distributor",
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%s\\n" "$1" > "${SAMPLE_POOL_ARGUMENT_MARKER}"',
                        'printf "sample-pool\\n" >> "${STARTUP_ORDER_MARKER}"',
                        'exec "${REAL_PYTHON}" "${FAKE_DISTRIBUTOR_SERVER}" "${MAZE_SAMPLE_DISTRIBUTOR_PORT}"',
                    ]
                )
                + "\n",
            )

            distributor_marker = root / "distributor-argument.txt"
            write_executable(
                distributor_root / "bin" / "maze_model_distributor",
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%s\\n" "$1" > "${DISTRIBUTOR_ARGUMENT_MARKER}"',
                        'printf "model-distributor\\n" >> "${STARTUP_ORDER_MARKER}"',
                        'exec "${REAL_PYTHON}" "${FAKE_DISTRIBUTOR_SERVER}" "${MAZE_MODEL_DISTRIBUTOR_PORT}"',
                    ]
                )
                + "\n",
            )

            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            python_calls = root / "python-calls.txt"
            local_train = root / "models" / "local-train"
            local_train.mkdir(parents=True)
            stale_file = local_train / "stale-model.onnx"
            stale_file.write_text("stale", encoding="utf-8")
            initial_checkpoint = root / "savepoints" / "initial.pt"
            initial_checkpoint.parent.mkdir()
            initial_checkpoint.write_text(
                "external-checkpoint", encoding="utf-8"
            )
            write_executable(
                fake_bin / "python3",
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        'printf "%s\\n" "$*" >> "${PYTHON_CALLS_MARKER}"',
                        "sleep 0.2",
                    ]
                )
                + "\n",
            )

            environment = os.environ.copy()
            environment.pop("MODEL_DISTRIBUTOR_BIN", None)
            environment.pop("MODEL_DISTRIBUTOR_CONFIG", None)
            environment.pop("SAMPLE_DISTRIBUTOR_BIN", None)
            environment.pop("SAMPLE_DISTRIBUTOR_CONFIG", None)
            environment.update(
                {
                    "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                    "REAL_PYTHON": sys.executable,
                    "FAKE_DISTRIBUTOR_SERVER": str(server_script),
                    "SAMPLE_POOL_ARGUMENT_MARKER": str(sample_pool_marker),
                    "DISTRIBUTOR_ARGUMENT_MARKER": str(distributor_marker),
                    "STARTUP_ORDER_MARKER": str(startup_order),
                    "PYTHON_CALLS_MARKER": str(python_calls),
                    "MAZE_SAMPLE_DISTRIBUTOR_PORT": str(available_port()),
                    "MAZE_MODEL_DISTRIBUTOR_PORT": str(available_port()),
                    "MAZE_QUIESCE_MARKER": str(root / "quiesced"),
                    "MAZE_LOCAL_TRAIN_ROOT": str(local_train),
                    "MAZE_INITIAL_CHECKPOINT": str(initial_checkpoint),
                }
            )

            result = subprocess.run(
                ["bash", str(root / "run.sh")],
                cwd=root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertEqual(
                Path(sample_pool_marker.read_text(encoding="utf-8").strip()),
                sample_pool_config.resolve(),
            )
            self.assertEqual(
                Path(distributor_marker.read_text(encoding="utf-8").strip()),
                distributor_config.resolve(),
            )
            self.assertEqual(
                startup_order.read_text(encoding="utf-8").splitlines(),
                ["sample-pool", "model-distributor"],
            )
            calls = python_calls.read_text(encoding="utf-8")
            self.assertIn("tools/metrics_server.py", calls)
            self.assertIn("-m main.training_runtime", calls)
            self.assertIn(
                f"--initial-checkpoint {initial_checkpoint.resolve()}", calls
            )
            self.assertNotIn("--run-id", calls)
            self.assertFalse(stale_file.exists())
            self.assertEqual(
                initial_checkpoint.read_text(encoding="utf-8"),
                "external-checkpoint",
            )

    def test_rejects_symbolic_link_local_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_minimal_root(root)
            target = root / "outside"
            target.mkdir()
            local_train = root / "models" / "local-train"
            local_train.parent.mkdir()
            local_train.symlink_to(target, target_is_directory=True)
            result = subprocess.run(
                ["bash", str(root / "run.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "MAZE_LOCAL_TRAIN_ROOT": str(local_train),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be a symbolic link", result.stderr)

    def test_existing_lock_preserves_local_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_minimal_root(root)
            local_train = root / "models" / "local-train"
            local_train.mkdir(parents=True)
            marker = local_train / "active-data"
            marker.write_text("keep", encoding="utf-8")
            lock = root / "models" / ".learner-local-train.lock"
            lock.mkdir()
            result = subprocess.run(
                ["bash", str(root / "run.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "MAZE_LOCAL_TRAIN_ROOT": str(local_train),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("training is already active", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_rejects_uncontrolled_local_train_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare_minimal_root(root)
            local_train = root / "outside" / "local-train"
            local_train.mkdir(parents=True)
            marker = local_train / "keep"
            marker.write_text("keep", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(root / "run.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "MAZE_LOCAL_TRAIN_ROOT": str(local_train),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsafe Learner local-train path", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
