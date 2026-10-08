import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.diagnostics.audit_prior_transfer_run import build_parser, run_audit
from tests.test_prior_transfer_assets import RunIntegrationFixture, asset_metadata_fixture, file_sha
from reliability.prior_transfer_assets import input_fingerprints


class PriorTransferAssetParserTests(unittest.TestCase):
    def test_parser_has_no_gt_mesh_training_or_scientific_override(self):
        options = {option for action in build_parser()._actions for option in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--confirmation", "--confirmation-sha", "--seed"})
        result = subprocess.run([sys.executable, "-B", "scripts/diagnostics/audit_prior_transfer_run.py", "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PublicationBoundaryTests(unittest.TestCase):
    """Exercise real publication I/O; only expensive checkpoint qualification is substituted."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.record, self.path, self.digest, self.row = asset_metadata_fixture(self.root)
        self.args = build_parser().parse_args(["--confirmation", str(self.path), "--confirmation-sha", self.digest, "--seed", "0"])
        self.report = {"outcome": "QUALIFIED", "confirmation_sha256": self.digest,
                       "input_fingerprints": input_fingerprints(self.record, self.row)}

    def test_exclusive_json_sha_publication_and_no_input_writes(self):
        before = input_fingerprints(self.record, self.row)
        with patch("scripts.diagnostics.audit_prior_transfer_run.qualify_prior_transfer_run", return_value=self.report):
            code, result = run_audit(self.args)
            self.assertEqual(code, 0, result)
            output = Path(self.row["qualification_path"])
            original = output.read_bytes()
            self.assertEqual(json.loads(original)["confirmation_sha256"], self.digest)
            self.assertEqual(Path(str(output) + ".sha256").read_text().strip(), file_sha(output))
            self.assertEqual(run_audit(self.args)[0], 2)
            self.assertEqual(output.read_bytes(), original)
        self.assertEqual(before, input_fingerprints(self.record, self.row))

    def test_post_read_mutation_and_partial_publish_failure_leave_no_result(self):
        target = Path(self.row["qualification_path"])
        original = (Path(self.row["run_dir"]) / "train.log").read_bytes()
        (Path(self.row["run_dir"]) / "train.log").write_text("changed")
        with patch("scripts.diagnostics.audit_prior_transfer_run.qualify_prior_transfer_run", return_value=self.report):
            self.assertEqual(run_audit(self.args)[0], 2)
        self.assertFalse(target.exists())
        (Path(self.row["run_dir"]) / "train.log").write_bytes(original)
        import os
        real_link = os.link
        calls = []
        def fail_second(source, output):
            calls.append(output)
            if len(calls) == 2:
                raise OSError("injected publication failure")
            real_link(source, output)
        with patch("scripts.diagnostics.audit_prior_transfer_run.qualify_prior_transfer_run", return_value=self.report), patch("scripts.diagnostics.audit_prior_transfer_run.os.link", side_effect=fail_second):
            self.assertEqual(run_audit(self.args)[0], 2)
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".sha256").exists())
        self.assertFalse(list(target.parent.glob(".*.tmp-*")))


class PriorTransferAssetCliTests(RunIntegrationFixture):
    def args(self):
        return build_parser().parse_args(["--confirmation", str(self.path), "--confirmation-sha", self.digest, "--seed", "0"])

    def test_publishes_canonical_json_sha_outside_run_without_overwrite(self):
        code, result = run_audit(self.args())
        self.assertEqual(code, 0, result)
        output = Path(self.row["qualification_path"])
        report = json.loads(output.read_text())
        self.assertEqual(report["outcome"], "QUALIFIED")
        self.assertEqual(report["confirmation_sha256"], self.digest)
        self.assertEqual(Path(str(output) + ".sha256").read_text().strip(), file_sha(output))
        original = output.read_bytes()
        self.assertEqual(run_audit(self.args())[0], 2)
        self.assertEqual(output.read_bytes(), original)

    def test_failure_or_post_read_mutation_never_publishes(self):
        target = Path(self.row["qualification_path"])
        (self.run / ".training_active").touch()
        self.assertEqual(run_audit(self.args())[0], 2)
        self.assertFalse(target.exists())
        (self.run / ".training_active").unlink()
        from reliability.prior_transfer_assets import qualify_prior_transfer_run
        def mutate(*args):
            result = qualify_prior_transfer_run(*args)
            (self.run / "train.log").write_text("changed after qualification\n")
            return result
        with patch("scripts.diagnostics.audit_prior_transfer_run.qualify_prior_transfer_run", side_effect=mutate):
            self.assertEqual(run_audit(self.args())[0], 2)
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".sha256").exists())
        self.assertFalse(list(target.parent.glob(".*.tmp-*")))
