import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.diagnostics.create_g1_prior_transfer_confirmation import (
    build_parser,
    run_create,
)


class PriorTransferConfirmationCliTests(unittest.TestCase):
    def test_parser_exposes_no_scientific_override(self):
        options = {action.dest for action in build_parser()._actions}
        self.assertEqual(options, {"help", "request", "output"})
        forbidden = {
            "seed", "iteration", "model", "solver", "fold", "bootstrap",
            "threshold", "candidate", "crop", "transform",
        }
        self.assertTrue(options.isdisjoint(forbidden))

    def test_cli_fails_closed_on_invalid_request_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            request = root / "request.json"
            output = root / "confirmation.json"
            request.write_text(json.dumps({"repository": {"clean": False}}), encoding="utf-8")
            code, result = run_create(
                argparse.Namespace(request=str(request), output=str(output))
            )
            self.assertEqual(code, 2)
            self.assertIn("error", result)
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".sha256").exists())


if __name__ == "__main__":
    unittest.main()
