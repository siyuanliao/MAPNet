import unittest

from map_net_a3.model import MODEL_ARCHITECTURE, LEGACY_MODEL_ARCHITECTURE, checkpoint_matches


class CheckpointIdentityTests(unittest.TestCase):
    def test_current_and_legacy_identity(self):
        for name in (MODEL_ARCHITECTURE, LEGACY_MODEL_ARCHITECTURE):
            state = {"model_name": name}
            self.assertTrue(checkpoint_matches(state))

    def test_reject_other_variants(self):
        self.assertFalse(checkpoint_matches({"model_name": "MAP-Net"}))
