import unittest

from map_net_a5.model import MODEL_ARCHITECTURE, LEGACY_MODEL_ARCHITECTURE, checkpoint_matches, TRAIN_OBJECTIVE


class CheckpointIdentityTests(unittest.TestCase):
    def test_current_and_legacy_identity(self):
        for name in (MODEL_ARCHITECTURE, LEGACY_MODEL_ARCHITECTURE):
            state = {"model_name": name, "training_objective": TRAIN_OBJECTIVE}
            self.assertTrue(checkpoint_matches(state))

    def test_reject_other_variants(self):
        self.assertFalse(checkpoint_matches({"model_name": "MAP-Net"}))

    def test_reject_missing_objective(self):
        self.assertFalse(checkpoint_matches({"model_name": LEGACY_MODEL_ARCHITECTURE}))
