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
    def test_real_operation_probe_suffix_reaches_existing_wrapper_and_evaluator(self):
        prefix = self.f.prior["confirmation_id"]
        self.f.prior["probe_targets"] = {
            "output_dir": str(self.f.root / (prefix + ".probe")),
            "staging_dir": str(self.f.root / ("." + prefix + ".probe-staging")),
            "access_log_path": str(self.f.root / (prefix + ".first-gt-access.json")),
        }
        self.f.firewall.log = Path(self.f.prior["probe_targets"]["access_log_path"])
        self.f.args.diagnostic_id = prefix + ".probe"
        self.use_real_producer_serialization()
        self.run_original_wrapper_case()

    def test_probe_suffix_does_not_admit_traversal_or_unbound_output(self):
        self.prepare()
        for value in ("../escape.probe", ".hidden.probe", "bad/part.probe", "bad.probe.extra", "other.probe"):
            self.f.args.diagnostic_id = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_evaluator(self.f.args, self.f.dependencies)
            self.assertFalse(self.f.firewall.log.exists())
        self.assertEqual(self.f.dependencies.events, [])

    def use_real_producer_serialization(self):
        from scripts.diagnostics.audit_utility_source import _canonical_bytes as source_bytes
        from reliability.utility_snapshot import _canonical_bytes as snapshot_bytes
        for key, serialize in (("source_record", source_bytes), ("snapshot_record", snapshot_bytes)):
            path = Path(self.f.prior[key]["path"])
            raw = serialize(json.loads(path.read_bytes()))
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.f.prior[key]["sha256"] = digest
            Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
        self.f.firewall.prior_identity = self.f.firewall.publish("evaluator-prior.json", self.f.prior)
        self.f.args.confirmation_sha = self.f.firewall.prior_identity["sha256"]
        for index, handle in enumerate(self.f.qualifications):
            record = json.loads(Path(handle["path"]).read_bytes())
            record["confirmation_sha256"] = self.f.args.confirmation_sha
            for key in ("source_record", "snapshot_record"):
                path = Path(self.f.prior[key]["path"])
                record["input_fingerprints"][str(path)].update(
                    bytes=path.stat().st_size, sha256=self.f.prior[key]["sha256"])
            self.f.qualifications[index] = self.f.firewall.publish(Path(handle["path"]).name, record)
        self.f.args.qualification_record = [[q["path"], q["sha256"]] for q in self.f.qualifications]

    def test_real_source_snapshot_producer_bytes_reach_existing_evaluator_without_rewrite(self):
        self.use_real_producer_serialization()
        self.prepare()
        before = {key: Path(self.f.prior[key]["path"]).read_bytes() for key in ("source_record", "snapshot_record")}
        code, _ = run_evaluator(self.f.args, self.f.dependencies)
        self.assertEqual(code, 0)
        for key, raw in before.items():
            self.assertEqual(Path(self.f.prior[key]["path"]).read_bytes(), raw)
        self.assertEqual(self.f.dependencies.events[:2], ["admit", "mesh"])

    def wrapper_recovery_fixture(self):
        self.prepare()
        spec = self.f.repository / "docs/superpowers/specs/2026-10-09-utility-prior-transfer-gt-decoupling-amendment.md"
        spec.parent.mkdir(parents=True)
        spec.write_bytes(Path(self.spec["path"]).read_bytes())
        self.spec = {"path": str(spec), "sha256": hashlib.sha256(spec.read_bytes()).hexdigest()}
        approval = self.access.approval_record(self.f.firewall.prior_identity, self.f.qualifications,
            {"root": str(self.f.repository), "commit": "c" * 40, "clean": True}, self.spec)
        self.approval = self.f.firewall.publish("access-approval.json", approval)
        record = self.access.build_amendment(self.f.firewall.prior_identity,
            self.f.qualifications, self.approval, protected=[Path(self.f.args.gt_mesh)])
        self.handle = self.f.firewall.publish("access-amendment.json", record)
        self.f.args.expected_commit = "d" * 40
        self.f.dependencies.git_identity = lambda p: {"commit": "d" * 40, "clean": True}
        self.f.args.pre_gt_recovery_from = [self.handle["path"], self.handle["sha256"]]
        self.old = {Path(h["path"] + suffix): Path(h["path"] + suffix).read_bytes()
                    for h in (self.handle, self.approval) for suffix in ("", ".sha256")}
        return record

    def run_wrapper_recovery(self):
        from scripts.diagnostics import run_approved_prior_transfer as runner
        current = {"commit": self.f.args.expected_commit, "clean": True}
        with mock.patch.object(self.access, "git_identity", return_value=current), \
             mock.patch.object(runner, "git_identity", return_value=current), \
             mock.patch.object(runner, "run_evaluator", side_effect=lambda request: run_evaluator(request, self.f.dependencies)):
            return runner.prepare_and_run(self.f.args)

    def id_recovery_fixture(self):
        prefix = "utility_prior_transfer_softv4_20261008_v3"
        self.f.prior["probe_targets"] = {
            "output_dir": str(self.f.root / (prefix + ".probe")),
            "staging_dir": str(self.f.root / ("." + prefix + ".probe-staging")),
            "access_log_path": str(self.f.root / (prefix + ".first-gt-access.json")),
        }
        self.f.firewall.log = Path(self.f.prior["probe_targets"]["access_log_path"])
        self.use_real_producer_serialization()
        self.wrapper_recovery_fixture()
        self.root_attempt = self.handle
        with mock.patch.object(self.access, "git_identity", return_value={"commit": "d" * 40, "clean": True}):
            approval = self.access.approval_record(self.f.firewall.prior_identity,
                self.f.qualifications, {"root": str(self.f.repository), "commit": "d" * 40, "clean": True},
                self.spec, recovery_from=self.handle)
            previous_approval = self.f.firewall.publish("failed-format-approval.json", approval)
            previous = self.access.build_amendment(self.f.firewall.prior_identity,
                self.f.qualifications, previous_approval)
            previous_handle = self.f.firewall.publish("failed-format-amendment.json", previous)
        receipt = {"kind": "prior_transfer_operational_failure",
            "error": "ValueError: unsafe diagnostic ID", "access_amendment": previous_handle,
            "wall_seconds": 1.0, "automatic_retry": False}
        self.failure = self.f.firewall.publish("failed-format-receipt.json", receipt)
        for handle in (previous_handle, previous_approval, self.failure):
            for suffix in ("", ".sha256"):
                path = Path(handle["path"] + suffix)
                self.old[path] = path.read_bytes()
        self.handle = previous_handle
        self.f.args.expected_commit = "e" * 40
        self.f.dependencies.git_identity = lambda p: {"commit": "e" * 40, "clean": True}
        self.f.args.pre_gt_recovery_from = [self.handle["path"], self.handle["sha256"]]
        self.f.args.pre_gt_failure_receipt = [self.failure["path"], self.failure["sha256"]]
        return previous

    def test_receipt_bound_id_recovery_preserves_two_attempts_and_reuses_real_probe_contract(self):
        previous = self.id_recovery_fixture()
        self.assertEqual(self.run_wrapper_recovery(), 0)
        log = json.loads(self.f.firewall.log.read_bytes())
        new = json.loads(Path(log["access_amendment"]["path"]).read_bytes())
        self.assertEqual(new["pre_gt_recovery_from"], self.handle)
        self.assertEqual(new["pre_gt_failure_receipt"], self.failure)
        self.assertEqual(new["repository"]["commit"], "e" * 40)
        for key in ("prior_confirmation", "qualification_records", "probe_targets", "protocol",
                    "mesh_admission", "core_sha256", "source_record", "snapshot_record", "training_commit"):
            self.assertEqual(new[key], previous[key])
        for path, raw in self.old.items():
            self.assertEqual(path.read_bytes(), raw)
        inputs = json.loads((self.f.target() / "inputs.json").read_bytes())
        fingerprints = inputs["input_fingerprints"]
        for path in self.old:
            self.assertIn(str(path), fingerprints)
        self.assertEqual(sum(e[0] == "evaluate" for e in self.f.dependencies.events if isinstance(e, tuple)), 6)
        with self.assertRaises(FileExistsError):
            self.run_wrapper_recovery()

    def test_id_recovery_requires_exact_operational_failure_receipt_before_new_records(self):
        self.id_recovery_fixture()
        receipt = json.loads(Path(self.failure["path"]).read_bytes())
        bad = [None, [self.failure["path"], "0" * 64]]
        for key, value in (("error", "ValueError: metric failed"),
                           ("kind", "prior_transfer_evaluation_completion"),
                           ("access_amendment", self.root_attempt),
                           ("automatic_retry", True), ("wall_seconds", -1)):
            changed = {**receipt, key: value}
            handle = self.f.firewall.publish("wrong-failure-" + key + ".json", changed)
            bad.append([handle["path"], handle["sha256"]])
        for pair in bad:
            self.f.args.pre_gt_failure_receipt = pair
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                self.run_wrapper_recovery()
            self.assertFalse(self.f.firewall.log.exists())
            self.assertFalse(list(self.f.root.glob("*.id-recovery1*")))
        self.assertEqual(self.f.dependencies.events, [])

    def test_id_recovery_stops_on_consumed_targets_without_new_records(self):
        self.id_recovery_fixture()
        for name in self.f.prior["probe_targets"].values():
            path = Path(name)
            path.write_bytes(b"consumed")
            with self.subTest(path=path), self.assertRaises(FileExistsError):
                self.run_wrapper_recovery()
            path.unlink()
        self.assertFalse(list(self.f.root.glob("*.id-recovery1*")))
        self.assertEqual(self.f.dependencies.events, [])

    def test_explicit_pre_gt_recovery_preserves_old_records_and_frozen_targets(self):
        old = self.wrapper_recovery_fixture()
        self.assertEqual(self.run_wrapper_recovery(), 0)
        log = json.loads(self.f.firewall.log.read_bytes())
        new = json.loads(Path(log["access_amendment"]["path"]).read_bytes())
        self.assertEqual(new["pre_gt_recovery_from"], self.handle)
        self.assertEqual(new["repository"]["commit"], "d" * 40)
        for key in ("prior_confirmation", "protocol", "mesh_admission", "core_sha256", "probe_targets",
                    "source_record", "snapshot_record", "qualification_records", "training_commit"):
            self.assertEqual(new[key], old[key])
        for path, raw in self.old.items():
            self.assertEqual(path.read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            self.run_wrapper_recovery()

    def test_pre_gt_recovery_rejects_bad_old_sha_or_changed_contract_before_new_records(self):
        self.wrapper_recovery_fixture()
        self.f.args.pre_gt_recovery_from[1] = "0" * 64
        with self.assertRaises(ValueError):
            self.run_wrapper_recovery()
        self.f.args.pre_gt_recovery_from[1] = self.handle["sha256"]
        previous = json.loads(Path(self.handle["path"]).read_bytes())
        previous["protocol"] = {"candidate": "r_g"}
        changed = self.f.firewall.publish("changed-old-amendment.json", previous)
        self.f.args.pre_gt_recovery_from = [changed["path"], changed["sha256"]]
        with self.assertRaises(ValueError):
            self.run_wrapper_recovery()
        self.assertFalse(self.f.firewall.log.exists())
        self.assertFalse(list(self.f.root.glob("*.format-recovery1*")))

    def test_pre_gt_recovery_refuses_every_consumed_probe_target_before_gt(self):
        self.wrapper_recovery_fixture()
        for path in self.f.prior["probe_targets"].values():
            path = Path(path)
            path.write_bytes(b"already consumed")
            with self.subTest(path=path), self.assertRaises(FileExistsError):
                self.run_wrapper_recovery()
            path.unlink()
        self.assertEqual(self.f.dependencies.events, [])
        self.assertFalse(list(self.f.root.glob("*.format-recovery1*")))

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

    def run_original_wrapper_case(self):
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

    def test_operational_wrapper_reuses_original_targets_and_writes_compact_receipt(self):
        self.run_original_wrapper_case()

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
