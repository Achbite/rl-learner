import unittest

from main.training_runtime import ModelPublisher


class ModelRetentionMetadataTest(unittest.TestCase):
    def test_public_metadata_uses_interval_classification_and_separate_window(self):
        publisher = ModelPublisher.__new__(ModelPublisher)
        publisher.archive_interval_updates = 200
        publisher.publication_retention_steps = 101

        self.assertIn("retention", ModelPublisher.LOCAL_METADATA_KEYS)
        self.assertEqual(
            publisher.retention_for_updates(200),
            {"class": "permanent", "reason": "interval"},
        )
        self.assertEqual(
            publisher.retention_for_updates(201),
            {"class": "rolling", "reason": ""},
        )
        self.assertEqual(publisher.publication_retention_steps, 101)


if __name__ == "__main__":
    unittest.main()
