import importlib.util
from pathlib import Path
import random
import sys
import types
import unittest
from unittest import mock

import numpy as np

from arguments import OptimizationParams
from argparse import ArgumentParser


class _FakeCuda:
    def __init__(self):
        self.devices = []

    def set_device(self, device):
        self.devices.append(device)


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = _FakeCuda()
        self.manual_seed_calls = []

    def manual_seed(self, seed):
        self.manual_seed_calls.append(seed)

    @staticmethod
    def device(value):
        return value


def _load_general_utils(fake_torch):
    module_path = Path(__file__).parents[1] / "utils" / "general_utils.py"
    spec = importlib.util.spec_from_file_location("_e0_general_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


class SeedContractTests(unittest.TestCase):
    def test_cli_seed_defaults_to_zero_and_accepts_override(self):
        parser = ArgumentParser()
        parameters = OptimizationParams(parser)

        default_args = parameters.extract(parser.parse_args([]))
        override_args = parameters.extract(parser.parse_args(["--seed", "17"]))

        self.assertEqual(default_args.seed, 0)
        self.assertEqual(override_args.seed, 17)

    def test_safe_state_replays_python_numpy_and_torch_seed(self):
        fake_torch = _FakeTorch()
        general_utils = _load_general_utils(fake_torch)
        original_stdout = sys.stdout

        try:
            general_utils.safe_state(True, seed=17)
            first = (random.random(), float(np.random.random()))
            general_utils.safe_state(True, seed=17)
            second = (random.random(), float(np.random.random()))
        finally:
            sys.stdout = original_stdout

        self.assertEqual(first, second)
        self.assertEqual(fake_torch.manual_seed_calls, [17, 17])
        self.assertEqual(fake_torch.cuda.devices, ["cuda:0", "cuda:0"])


if __name__ == "__main__":
    unittest.main()
