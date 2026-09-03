import dataclasses
import unittest

from reliability.config import CoreConfig


class CoreConfigTests(unittest.TestCase):
    def test_all_core_flags_default_off(self):
        config = CoreConfig()

        self.assertEqual(config.seed, 0)
        self.assertEqual(config.enabled_features(), ())
        self.assertFalse(config.uses_core_path())
        config.validate()

    def test_shadow_mode_uses_core_path_without_enabling_method(self):
        config = dataclasses.replace(CoreConfig(), core_shadow_mode=True)

        self.assertEqual(config.enabled_features(), ("shadow_diagnostics",))
        self.assertTrue(config.uses_core_path())
        config.validate()

    def test_each_strictly_nested_core_stage_is_valid(self):
        stages = (
            {"enable_observation_calibration": True},
            {
                "enable_observation_calibration": True,
                "enable_dual_reliability": True,
            },
            {
                "enable_observation_calibration": True,
                "enable_dual_reliability": True,
                "enable_abstention": True,
            },
            {
                "enable_observation_calibration": True,
                "enable_dual_reliability": True,
                "enable_abstention": True,
                "enable_parameter_routing": True,
            },
            {
                "enable_observation_calibration": True,
                "enable_dual_reliability": True,
                "enable_abstention": True,
                "enable_parameter_routing": True,
                "enable_gradient_projection": True,
            },
            {
                "enable_observation_calibration": True,
                "enable_dual_reliability": True,
                "enable_abstention": True,
                "enable_parameter_routing": True,
                "enable_gradient_projection": True,
                "enable_reliability_lifecycle": True,
            },
        )

        for values in stages:
            with self.subTest(values=values):
                config = dataclasses.replace(CoreConfig(), **values)
                config.validate()
                self.assertTrue(config.uses_core_path())

    def test_nonnested_core_flags_are_rejected(self):
        invalid_cases = (
            ("enable_dual_reliability", "observation calibration"),
            ("enable_abstention", "dual reliability"),
            ("enable_parameter_routing", "abstention"),
            ("enable_gradient_projection", "parameter routing"),
            ("enable_reliability_lifecycle", "gradient projection"),
        )

        for field, required_name in invalid_cases:
            with self.subTest(field=field):
                config = dataclasses.replace(CoreConfig(), **{field: True})
                with self.assertRaisesRegex(ValueError, required_name):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
