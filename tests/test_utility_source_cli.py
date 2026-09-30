import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_utility_snapshot import _write_source
from scripts.diagnostics.audit_utility_source import build_parser, run_audit


class UtilitySourceCliTests(unittest.TestCase):
    def test_parser_requires_only_source_contract_arguments_and_has_no_gt_surface(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertEqual(
            option_strings,
            {"-h", "--help", "--source-root", "--expected-count", "--expected-source-sha", "--output"},
        )
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        help_text = parser.format_help().lower()
        self.assertNotIn("gt", help_text)
        self.assertNotIn("mesh", help_text)

    def test_cli_writes_canonical_json_and_detached_sha_without_touching_decoys(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = _write_source(base / "source")
            decoy = source / "do-not-read.bin"
            decoy.write_bytes(b"secret")
            output = base / "records" / "source.json"
            output.parent.mkdir()

            from reliability.utility_snapshot import audit_utility_source, source_manifest
            expected = source_manifest(audit_utility_source(source, expected_count=2))["source_sha256"]
            before = decoy.stat().st_mtime_ns
            code, result = run_audit(argparse.Namespace(
                source_root=str(source.resolve()),
                expected_count=2,
                expected_source_sha=expected,
                output=str(output.resolve()),
            ))
            self.assertEqual(code, 0)
            self.assertEqual(result["source_sha256"], expected)
            self.assertEqual(decoy.stat().st_mtime_ns, before)
            payload = output.read_bytes()
            self.assertEqual(payload, (json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            self.assertEqual(
                output.with_suffix(output.suffix + ".sha256").read_text(encoding="ascii").strip(),
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertFalse(any("do-not-read" in item["path"] for item in json.loads(payload)["files"]))

    def test_cli_rejects_relative_unsafe_overwrite_or_wrong_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = _write_source(base / "source")
            output = base / "source" / "record.json"
            common = dict(source_root=str(source.resolve()), expected_count=2, expected_source_sha="0" * 64, output=str(output.resolve()))
            for label, changes in (
                ("relative source", {"source_root": "relative"}),
                ("relative output", {"output": "relative.json"}),
                ("inside source", {}),
                ("malformed sha", {"expected_source_sha": "bad"}),
            ):
                with self.subTest(label=label):
                    args = argparse.Namespace(**(common | changes))
                    code, _ = run_audit(args)
                    self.assertEqual(code, 2)
            outside = base / "outside.json"
            outside.write_text("existing", encoding="utf-8")
            args = argparse.Namespace(**(common | {"output": str(outside.resolve())}))
            code, _ = run_audit(args)
            self.assertEqual(code, 2)

    def test_cli_rechecks_source_and_cleans_staging_on_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = _write_source(base / "source")
            output = base / "record.json"
            from reliability.utility_snapshot import audit_utility_source, source_manifest
            audit = audit_utility_source(source, expected_count=2)
            manifest = source_manifest(audit)
            changed = dict(manifest)
            changed["source_sha256"] = "f" * 64
            with mock.patch(
                "scripts.diagnostics.audit_utility_source.source_manifest",
                side_effect=(manifest, changed),
            ):
                code, result = run_audit(argparse.Namespace(
                    source_root=str(source.resolve()),
                    expected_count=2,
                    expected_source_sha=manifest["source_sha256"],
                    output=str(output.resolve()),
                ))
            self.assertEqual(code, 2)
            self.assertIn("mutated", result["error"].lower())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(output.suffix + ".sha256").exists())
            self.assertEqual(list(base.glob(".*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
