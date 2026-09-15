from argparse import ArgumentParser
import unittest
from unittest.mock import patch

from arguments import OptimizationParams
from reliability.config import CoreConfig
from train import training


class D0TrainingDispatchTests(unittest.TestCase):
    def test_shadow_path_reaches_training_setup_instead_of_e0_guard(self):
        class StopAtSetup(Exception):
            pass

        with patch(
            "train.prepare_output_and_logger", side_effect=StopAtSetup
        ):
            with self.assertRaises(StopAtSetup):
                training(
                    None,
                    None,
                    None,
                    [],
                    [],
                    [],
                    None,
                    -1,
                    core_config=CoreConfig(core_shadow_mode=True),
                )

    def test_d0_refresh_interval_defaults_to_1000_and_accepts_smoke_override(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)

        default = group.extract(parser.parse_args([]))
        smoke = group.extract(
            parser.parse_args(["--d0_refresh_interval", "100"])
        )

        self.assertEqual(default.d0_refresh_interval, 1000)
        self.assertEqual(smoke.d0_refresh_interval, 100)


if __name__ == "__main__":
    unittest.main()
