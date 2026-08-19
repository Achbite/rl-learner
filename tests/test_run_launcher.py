import subprocess
import tempfile
import unittest
from pathlib import Path

from main.reset_workspace import reset_training_workspace


ROOT = Path(__file__).resolve().parents[1]


class RunLauncherTest(unittest.TestCase):
    def test_shell_entrypoints_are_syntactically_valid(self):
        for relative in (
            "test.sh",
            "run.sh",
            "build_image.sh",
            "build_smoke_model_artifact.sh",
            "scripts/dev_container.sh",
            "scripts/entrypoint.sh",
            "scripts/prepare_dev_artifacts.sh",
        ):
            result = subprocess.run(
                ["bash", "-n", str(ROOT / relative)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_workspace_reset_clears_only_the_owned_train_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "owned" / "train"
            publication = train / "0000000"
            publication.mkdir(parents=True)
            (publication / "SaveModel.onnx").write_bytes(b"old-model")
            (train / "runtime").mkdir()
            outside = root / "preserved.onnx"
            outside.write_bytes(b"preserved")
            (train / "outside-link").symlink_to(outside)

            resolved = reset_training_workspace(train)

            self.assertEqual(resolved, train.resolve())
            self.assertEqual(list(train.iterdir()), [])
            self.assertEqual(outside.read_bytes(), b"preserved")

    def test_workspace_reset_rejects_unowned_or_symlink_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unowned = root / "models"
            with self.assertRaisesRegex(ValueError, "end with /train"):
                reset_training_workspace(unowned)
            self.assertFalse(unowned.exists())

            owned = root / "owned" / "train"
            owned.mkdir(parents=True)
            alias = root / "alias" / "train"
            alias.parent.mkdir()
            alias.symlink_to(owned, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                reset_training_workspace(alias)
