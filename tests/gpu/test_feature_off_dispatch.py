import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import pytest
except ImportError:
    pytestmark = None
else:
    pytestmark = pytest.mark.gpu

from reliability.config import CoreConfig
from train import (
    build_checkpoint_payload,
    prepare_output_and_logger,
    select_training_path,
    training,
)


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

    def test_feature_off_logger_writes_resolved_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = SimpleNamespace(
                model_path=temporary_directory,
                source_path="/synthetic/source",
                sh_degree=3,
            )
            optimization = SimpleNamespace(iterations=1, seed=0)
            pipeline = SimpleNamespace(debug=False)

            writer = prepare_output_and_logger(
                dataset, optimization, pipeline, CoreConfig()
            )
            if writer is not None:
                writer.close()

            resolved = json.loads(
                (Path(temporary_directory) / "resolved_config.json").read_text()
            )
            identity = json.loads(
                (Path(temporary_directory) / "run_identity.json").read_text()
            )

        self.assertEqual(resolved["training_path"], "legacy")
        self.assertEqual(resolved["core"]["enabled_features"], [])
        self.assertEqual(len(identity["git_commit"]), 40)


if __name__ == "__main__":
    unittest.main()
