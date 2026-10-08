"""Checks use temporary synthetic files only; never touch real results."""
import hashlib
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CleanupChecks(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        path = Path(__file__).with_name('2026-10-08-clean-storage.py')
        self.assertTrue(path.is_file(), 'approved cleanup implementation missing')
        spec = importlib.util.spec_from_file_location('cleanup_operation', path)
        self.op = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.op)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'results'
        self.root.mkdir()
        self.receipt = Path(self.tmp.name) / 'offload.jsonl'
        self.file = self.root / 'A.ply'
        self.file.write_bytes(b'colored visualization')
        self.report = self.root / 'report.json'
        self.report.write_bytes(b'preserve this frozen report')
        self.entries = [{'path': 'A.ply', 'bytes': 21,
                         'sha256': hashlib.sha256(self.file.read_bytes()).hexdigest()}]

    def test_success_preserves_unlisted_report_and_durable_receipt(self):
        self.op.remove_verified(self.root, self.entries, self.receipt)
        self.assertFalse(self.file.exists())
        self.assertEqual(self.report.read_bytes(), b'preserve this frozen report')
        records = [json.loads(line) for line in self.receipt.read_text().splitlines()]
        self.assertEqual([r['event'] for r in records], ['plan', 'removed', 'complete'])

    def test_corrupt_last_target_prevents_every_removal(self):
        other = self.root / 'B.ply'
        other.write_bytes(b'bad')
        entries = self.entries + [{'path': 'B.ply', 'bytes': 3, 'sha256': '0' * 64}]
        with self.assertRaises(RuntimeError):
            self.op.remove_verified(self.root, entries, self.receipt)
        self.assertTrue(self.file.exists())
        self.assertTrue(other.exists())
        self.assertFalse(self.receipt.exists())

    def test_existing_receipt_prevents_removal(self):
        self.receipt.write_text('existing receipt')
        with self.assertRaises(FileExistsError):
            self.op.remove_verified(self.root, self.entries, self.receipt)
        self.assertTrue(self.file.exists())

    def test_escape_is_rejected_before_removal(self):
        entries = self.entries + [dict(self.entries[0], path='../outside.ply')]
        with self.assertRaises(RuntimeError):
            self.op.remove_verified(self.root, entries, self.receipt)
        self.assertTrue(self.file.exists())

    def test_symlink_is_rejected_before_read_or_removal(self):
        link = self.root / 'linked.ply'
        try:
            link.symlink_to(self.file)
        except OSError:
            self.skipTest('host cannot create symlinks; required on AutoDL')
        with self.assertRaises(RuntimeError):
            self.op.remove_verified(self.root, [dict(self.entries[0], path='linked.ply')], self.receipt)
        self.assertTrue(self.file.exists())

    def test_changed_pinned_manifest_rejects_plan(self):
        name = next(iter(self.op.OUTPUTS))
        directory = self.root / name
        directory.mkdir()
        (directory / 'manifest.json').write_text('{"files": []}')
        with self.assertRaises(RuntimeError):
            self.op.build_plan(self.root)
        self.assertTrue(self.file.exists())

    def test_hardlink_is_rejected_before_removal(self):
        link = self.root / 'hardlink.ply'
        try:
            os.link(self.file, link)
        except OSError:
            self.skipTest('host cannot create hard links; required on AutoDL')
        with self.assertRaises(RuntimeError):
            self.op.remove_verified(self.root, self.entries, self.receipt)
        self.assertTrue(self.file.exists())

    def test_ancestor_symlink_is_rejected_before_read(self):
        link = self.root / 'linked-directory'
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest('host cannot create directory symlinks; required on AutoDL')
        with self.assertRaises(RuntimeError):
            self.op.remove_verified(self.root, [dict(self.entries[0], path='linked-directory/A.ply')], self.receipt)
        self.assertTrue(self.file.exists())

    def test_mutation_after_final_path_check_is_not_deleted(self):
        original = self.op.safe_path

        def mutate_after_check(root, relative):
            result = original(root, relative)
            if self.receipt.exists() and relative == 'A.ply':
                self.file.write_bytes(b'changed visualization')
            return result

        with patch.object(self.op, 'safe_path', side_effect=mutate_after_check):
            with self.assertRaises(RuntimeError):
                self.op.remove_verified(self.root, self.entries, self.receipt)
        self.assertTrue(self.file.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
