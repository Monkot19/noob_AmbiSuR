"""Synthetic prior-only access contracts; use the existing orchestration fixture."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from tests import test_g1_prior_transfer_cli as cli_tests
from scripts.diagnostics.evaluate_g1_prior_transfer import run_evaluator


class AccessAmendmentTests(unittest.TestCase):
    def test_frozen_core_paths_are_real_original_git_objects(self):
        from reliability.prior_transfer_access_amendment import CORE_PATHS, git_blob
        repository = Path(__file__).resolve().parents[1]
        for name in CORE_PATHS:
            with self.subTest(path=name):
                self.assertTrue(git_blob(repository, "c701424c1b1f5a9006e6f19776769ee7bc8cb299", name))

    def setUp(self):
        self.fixture = cli_tests.PriorTransferCliTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture

    def prepare(self):
        from reliability import prior_transfer_access_amendment as access
        self.access = access
        self.f.args.expected_commit = "c" * 40
        self.f.dependencies.git_identity = lambda p: {"commit": "c" * 40, "clean": True}
        self.original = {}
        for name in access.CORE_PATHS:
            path = self.f.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("frozen " + name).encode())
            self.original[name] = path.read_bytes()
        self.addCleanup(mock.patch.stopall)
        name = "reliability/utility_gt_firewall.py"
        content = Path(name).read_bytes()
        (self.f.repository / name).write_bytes(content)
        self.original[name] = content
        mock.patch.object(access, "git_blob", side_effect=lambda repo, commit, name: self.original[name]).start()
        mock.patch.object(access, "git_identity", return_value={"commit": "c" * 40, "clean": True}).start()
        self.spec = self.f.firewall.publish("access-spec.md", "approved synthetic specification")
        approval = access.approval_record(
            self.f.firewall.prior_identity, self.f.qualifications,
            {"root": str(self.f.repository), "commit": "c" * 40, "clean": True}, self.spec)
        self.approval = self.f.firewall.publish("access-approval.json", approval)
        record = access.build_amendment(self.f.firewall.prior_identity,
                                       self.f.qualifications, self.approval, protected=[Path(self.f.args.gt_mesh)])
        self.handle = self.f.firewall.publish("access-amendment.json", record)
        self.f.args.access_amendment = self.handle["path"]
        self.f.args.access_amendment_sha = self.handle["sha256"]
        self.f.args.geometry_release = self.f.args.geometry_release_sha = None
        return record

    def test_prior_only_path_logs_before_mesh_and_reuses_six_frozen_measurements(self):
        self.prepare()
        code, publication = run_evaluator(self.f.args, self.f.dependencies)
        self.assertEqual(code, 0)
        self.assertEqual(self.f.dependencies.events[:2], ["admit", "mesh"])
        self.assertEqual(sum(e[0] == "evaluate" for e in self.f.dependencies.events if isinstance(e, tuple)), 6)
        log = json.loads(self.f.firewall.log.read_bytes())
        self.assertEqual(log["kind"], "utility_prior_transfer_first_gt_access")
        self.assertNotIn("geometry_release", log)
        self.assertEqual(log["access_amendment"], self.handle)
        self.assertEqual(len(publication["manifest"]["files"]), 6)
        self.assertEqual(publication["report"]["provenance"]["training_commit"], "a" * 40)
        with self.assertRaises(FileExistsError):
            run_evaluator(self.f.args, self.f.dependencies)

    def test_changed_contract_authority_or_qualification_blocks_access(self):
        record = self.prepare()
        mutations = [("protocol", {"candidate": "r_g"}), ("geometry_status", "NO_ACTION_SPECIFIC_SIGNAL"),
                     ("geometry_candidate_admitted", True), ("utility_feedback_prohibited", False),
                     ("repository", {"root": str(self.f.repository), "commit": "d" * 40, "clean": True}),
                     ("qualification_records", list(reversed(self.f.qualifications))),
                     ("probe_targets", {**record["probe_targets"], "output_dir": str(self.f.root / "other")})]
        for key, value in mutations:
            changed = copy.deepcopy(record)
            changed[key] = value
            identity = self.f.firewall.publish("changed.json", changed)
            self.f.args.access_amendment, self.f.args.access_amendment_sha = identity["path"], identity["sha256"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                run_evaluator(self.f.args, self.f.dependencies)
            self.assertFalse(self.f.firewall.log.exists())
        Path(self.approval["path"]).write_bytes(b"unapproved mutation")
        with self.assertRaises(ValueError):
            self.access.load_amendment(self.handle, protected=[Path(self.f.args.gt_mesh)])

    def test_current_core_mutation_blocks_before_gt_and_old_path_stays_closed(self):
        self.prepare()
        path = self.f.repository / self.access.CORE_PATHS[0]
        path.write_bytes(b"new statistics")
        with self.assertRaises(ValueError):
            run_evaluator(self.f.args, self.f.dependencies)
        self.assertFalse(self.f.firewall.log.exists())
        self.f.args.access_amendment = None
        with self.assertRaises(ValueError):
            run_evaluator(self.f.args, self.f.dependencies)

    def test_amendment_and_sidecar_gt_aliases_never_read_gt(self):
        self.prepare()
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.exists() and path.samefile(Path(self.f.args.gt_mesh)):
                raise AssertionError("GT opened before access log")
            return original_open(path, *args, **kwargs)
        for suffix in ("", ".sha256"):
            path = Path(self.handle["path"] + suffix)
            old = path.read_bytes()
            path.unlink()
            os.link(self.f.args.gt_mesh, path)
            with mock.patch.object(Path, "open", guarded_open), self.assertRaises(ValueError):
                run_evaluator(self.f.args, self.f.dependencies)
            path.unlink()
            path.write_bytes(old)
        self.assertFalse(self.f.firewall.log.exists())

    def test_mesh_failure_keeps_one_shot_log_without_loading_checkpoints(self):
        self.prepare()
        from types import SimpleNamespace
        self.f.dependencies.admit_mesh = lambda *a, **k: SimpleNamespace(
            outcome="INCONCLUSIVE", reasons=("alignment failed",), summary={})
        code, publication = run_evaluator(self.f.args, self.f.dependencies)
        self.assertEqual(code, 2)
        self.assertEqual(self.f.dependencies.events, [])
        self.assertTrue(self.f.firewall.log.exists())
        self.assertEqual(publication["report"]["outcome"], "INCONCLUSIVE")

    def test_amendment_mutation_after_admission_prevents_publication(self):
        self.prepare()
        original = self.f.dependencies.admit_mesh
        def changed(*a, **k):
            result = original(*a, **k)
            Path(self.handle["path"]).write_bytes(b"changed amendment")
            return result
        self.f.dependencies.admit_mesh = changed
        with self.assertRaises((ValueError, RuntimeError)):
            run_evaluator(self.f.args, self.f.dependencies)
        self.assertFalse(self.f.target().exists())
        self.assertTrue(self.f.firewall.log.exists())

    def test_operational_wrapper_reuses_original_targets_and_writes_compact_receipt(self):
        self.prepare()
        # Start from the original completed assets, before any amendment exists.
        for handle in (self.handle, self.approval):
            Path(handle["path"]).unlink()
            Path(handle["path"] + ".sha256").unlink()
        from scripts.diagnostics import run_approved_prior_transfer as runner
        spec = self.f.repository / "docs/superpowers/specs/2026-10-09-utility-prior-transfer-gt-decoupling-amendment.md"
        spec.parent.mkdir(parents=True)
        spec.write_bytes(b"synthetic approved spec")
        with mock.patch.object(runner, "git_identity", return_value={"commit": "c" * 40, "clean": True}), \
             mock.patch.object(runner, "run_evaluator", side_effect=lambda request: run_evaluator(request, self.f.dependencies)):
            self.assertEqual(runner.prepare_and_run(self.f.args), 0)
        receipt = self.f.root / (self.f.prior["confirmation_id"] + ".evaluation-receipt.json")
        record = json.loads(receipt.read_bytes())
        self.assertEqual(record["output_dir"], str(self.f.target()))
        self.assertFalse(record["training_started"])
        self.assertFalse(record["c1_started"])
        self.assertEqual(sum(e[0] == "evaluate" for e in self.f.dependencies.events if isinstance(e, tuple)), 6)

    def test_wrapper_rejects_gt_alias_specification_before_gt_read(self):
        self.prepare()
        for handle in (self.handle, self.approval):
            Path(handle["path"]).unlink()
            Path(handle["path"] + ".sha256").unlink()
        from scripts.diagnostics import run_approved_prior_transfer as runner
        spec = self.f.repository / "docs/superpowers/specs/2026-10-09-utility-prior-transfer-gt-decoupling-amendment.md"
        spec.parent.mkdir(parents=True)
        os.link(self.f.args.gt_mesh, spec)
        original = Path.open
        def guarded(path, *a, **k):
            if path.exists() and path.samefile(Path(self.f.args.gt_mesh)):
                raise AssertionError("GT alias opened by wrapper")
            return original(path, *a, **k)
        with mock.patch.object(runner, "git_identity", return_value={"commit": "c" * 40, "clean": True}), \
             mock.patch.object(Path, "open", guarded), self.assertRaises(ValueError):
            runner.prepare_and_run(self.f.args)
        self.assertFalse(self.f.firewall.log.exists())

    def test_firewall_replacement_at_open_rejects_descriptor_before_any_gt_byte(self):
        self.prepare()
        target = self.f.repository / "reliability/utility_gt_firewall.py"
        original_open = Path.open
        class NoGtRead:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, *args): raise AssertionError("GT bytes consumed before access log")
        def replaced(path, *a, **k):
            if path == target:
                target.unlink()
                os.link(self.f.args.gt_mesh, target)
                return NoGtRead(original_open(path, *a, **k))
            return original_open(path, *a, **k)
        with mock.patch.object(Path, "open", replaced), self.assertRaises(ValueError):
            self.access.load_amendment(self.handle, protected=[Path(self.f.args.gt_mesh)])
        self.assertFalse(self.f.firewall.log.exists())

    def test_output_race_during_access_publication_never_reaches_mesh(self):
        self.prepare()
        original_link = os.link
        def raced(source, destination, *args, **kwargs):
            original_link(source, destination, *args, **kwargs)
            if Path(destination) == self.f.firewall.log:
                self.f.target().mkdir()
        with mock.patch.object(os, "link", raced), self.assertRaises(ValueError):
            run_evaluator(self.f.args, self.f.dependencies)
        self.assertEqual(self.f.dependencies.events, [])
        self.assertTrue(self.f.firewall.log.exists())

    def test_dirty_checkout_cannot_issue_a_token_even_without_cli(self):
        self.prepare()
        approval = json.loads(Path(self.approval["path"]).read_bytes())
        approval["repository"]["clean"] = False
        dirty_approval = self.f.firewall.publish("dirty-approval.json", approval)
        with mock.patch.object(self.access, "git_identity", return_value={"commit": "c" * 40, "clean": False}), \
             self.assertRaises(ValueError):
            self.access.build_amendment(self.f.firewall.prior_identity, self.f.qualifications,
                                        dirty_approval, protected=[Path(self.f.args.gt_mesh)])
        self.assertFalse(self.f.firewall.log.exists())


if __name__ == "__main__":
    unittest.main()
