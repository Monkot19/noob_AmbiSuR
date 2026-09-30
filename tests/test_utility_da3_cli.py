import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.diagnostics.finalize_utility_da3_snapshot import (
    build_parser as build_finalize_parser,
    run_finalizer,
)
from scripts.diagnostics.prepare_utility_da3_confirmation import (
    build_parser as build_prepare_parser,
    run_prepare,
)
from tests.test_utility_snapshot import _confirmation, _copy_source_and_write_derived


class UtilityDa3CliTests(unittest.TestCase):
    def test_parsers_have_no_gt_mesh_or_execution_override(self):
        for parser in (build_prepare_parser(), build_finalize_parser()):
            help_text = parser.format_help().lower()
            self.assertNotIn("gt", help_text)
            self.assertNotIn("mesh", help_text)
            self.assertNotIn("execute", help_text)

    def test_prepare_writes_confirmation_only_and_never_invokes_da3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, _staging, _snapshot_record, record = _confirmation(root)
            request = root / "request.json"
            request.write_text(json.dumps(record), encoding="utf-8")
            output = root / "confirmation.json"
            with mock.patch("subprocess.run") as run:
                code, result = run_prepare(argparse.Namespace(request=str(request.resolve()), output=str(output.resolve())))
            self.assertEqual(code, 0)
            run.assert_not_called()
            self.assertTrue(output.is_file())
            self.assertTrue(Path(result["sha256_path"]).is_file())

    def test_finalizer_audits_completed_tree_and_cleans_failed_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, staging, snapshot_record, record = _confirmation(root)
            confirmation_path = root / "confirmation.json"
            from reliability.utility_snapshot import write_da3_confirmation
            publication = write_da3_confirmation(record, confirmation_path)
            _copy_source_and_write_derived(source, staging, names=tuple(record["source"]["image_names"]))
            code, result = run_finalizer(argparse.Namespace(
                confirmation=str(confirmation_path.resolve()),
                expected_confirmation_sha=publication["sha256"],
                snapshot_root=str(staging.resolve()),
                output=str(snapshot_record.resolve()),
            ))
            self.assertEqual(code, 0)
            self.assertTrue(Path(result["output"]).is_file())
            bad_output = root / "records/bad.json"
            (staging / "estimated_depths/frame000.jpg.npy").unlink()
            code, _ = run_finalizer(argparse.Namespace(
                confirmation=str(confirmation_path.resolve()),
                expected_confirmation_sha=publication["sha256"],
                snapshot_root=str(staging.resolve()),
                output=str(bad_output.resolve()),
            ))
            self.assertEqual(code, 2)
            self.assertFalse(bad_output.exists())
            self.assertEqual(list(bad_output.parent.glob(".*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
