import subprocess
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
        self.assertIn("expected_local_train_root", document)
        self.assertNotIn("docker rm", document)


if __name__ == "__main__":
    unittest.main()
