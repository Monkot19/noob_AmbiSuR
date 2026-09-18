from argparse import ArgumentParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arguments import OptimizationParams
from reliability.config import CoreConfig
from reliability.shadow import D0ShadowRuntime
from tests.test_d0_shadow_runtime import refresh_inputs
from train import persist_d0_snapshot, training


class D0TrainingDispatchTests(unittest.TestCase):
    def test_training_snapshot_bridge_persists_runtime_transition_summary(self):
        runtime = D0ShadowRuntime(
            2,
            cfg=CoreConfig(core_shadow_mode=True),
            device="cpu",
            refresh_interval=1000,
        )
        snapshot = runtime.maybe_refresh(1000, refresh_inputs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = persist_d0_snapshot(
                temporary_directory,
                1000,
                snapshot,
                runtime,
            )
            event = json.loads(
                Path(paths["events"])
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

        self.assertEqual(event["schema_version"], 2)
        self.assertEqual(event["iteration"], 1000)
        self.assertEqual(
            event["transition_count_matrix"],
            runtime.latest_transition_diagnostics[
                "transition_count_matrix"
            ],
        )

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
