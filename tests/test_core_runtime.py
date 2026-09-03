from types import SimpleNamespace
import unittest

from reliability.config import CoreConfig
from reliability.runtime import (
    build_checkpoint_payload,
    core_config_from_namespace,
    select_training_path,
)


class CoreRuntimeTests(unittest.TestCase):
    def test_core_off_selects_legacy_training_path(self):
        self.assertEqual(select_training_path(CoreConfig()), "legacy")

    def test_shadow_and_first_method_stage_select_core_path(self):
        shadow = CoreConfig(core_shadow_mode=True)
        c1 = CoreConfig(enable_observation_calibration=True)

        self.assertEqual(select_training_path(shadow), "core")
        self.assertEqual(select_training_path(c1), "core")

    def test_invalid_stage_is_rejected_before_dispatch(self):
        config = CoreConfig(enable_gradient_projection=True)

        with self.assertRaisesRegex(ValueError, "parameter routing"):
            select_training_path(config)

    def test_feature_off_checkpoint_uses_legacy_tuple_schema(self):
        gaussian_state = object()

        payload = build_checkpoint_payload(gaussian_state, 7001, CoreConfig())

        self.assertIsInstance(payload, tuple)
        self.assertEqual(len(payload), 2)
        self.assertIs(payload[0], gaussian_state)
        self.assertEqual(payload[1], 7001)

    def test_core_checkpoint_uses_versioned_mapping(self):
        gaussian_state = object()
        core_state = {"refresh_count": 3}
        config = CoreConfig(enable_observation_calibration=True)

        payload = build_checkpoint_payload(
            gaussian_state, 7001, config, core_state=core_state
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertIs(payload["gaussian_state"], gaussian_state)
        self.assertEqual(payload["iteration"], 7001)
        self.assertIs(payload["core_state"], core_state)

    def test_namespace_conversion_ignores_unrelated_options(self):
        namespace = SimpleNamespace(
            seed=19,
            core_shadow_mode=True,
            enable_observation_calibration=False,
            enable_dual_reliability=False,
            enable_abstention=False,
            enable_parameter_routing=False,
            enable_gradient_projection=False,
            enable_reliability_lifecycle=False,
            iterations=30_000,
        )

        config = core_config_from_namespace(namespace)

        self.assertEqual(config, CoreConfig(seed=19, core_shadow_mode=True))


if __name__ == "__main__":
    unittest.main()
