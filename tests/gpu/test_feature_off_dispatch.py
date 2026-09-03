import inspect
import unittest

try:
    import pytest
except ImportError:
    pytestmark = None
else:
    pytestmark = pytest.mark.gpu

from reliability.config import CoreConfig
from train import build_checkpoint_payload, select_training_path, training


class FeatureOffDispatchTests(unittest.TestCase):
    def test_train_module_exposes_legacy_dispatch(self):
        self.assertEqual(select_training_path(CoreConfig()), "legacy")

    def test_train_module_preserves_legacy_checkpoint_tuple(self):
        gaussian_state = object()

        payload = build_checkpoint_payload(gaussian_state, 7001, CoreConfig())

        self.assertIsInstance(payload, tuple)
        self.assertEqual(payload, (gaussian_state, 7001))

    def test_training_accepts_explicit_core_config(self):
        parameters = inspect.signature(training).parameters

        self.assertIn("core_config", parameters)
        self.assertIsNone(parameters["core_config"].default)


if __name__ == "__main__":
    unittest.main()
