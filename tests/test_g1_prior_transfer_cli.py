"""Synthetic orchestration contracts; never access a real Utility asset."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tests import test_utility_gt_firewall as firewall_tests
from tests.test_g1_complementarity_cli import iteration_summary
from reliability.g1_prior_transfer_confirmation import build_prior_transfer_confirmation
from scripts.diagnostics.evaluate_g1_prior_transfer import build_parser, run_evaluator

canonical = firewall_tests.canonical


class Dependencies:
    """Only expensive mesh/checkpoint/statistical boundaries are substituted."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.events = []
        self.error = None
        self.mutate = False

    def git_identity(self, repository):
        return {"commit": "a" * 40, "clean": True}

    def admit_mesh(self, mesh_path, source_root, confirmation, *, access_token):
        # Real disk token MUST already exist, not just an in-memory sentinel.
        token = json.loads(Path(access_token["path"]).read_text())
        assert token["prior_confirmation"] == self.fixture.firewall.prior_identity
        self.events.append("admit")
        return SimpleNamespace(outcome="ADMITTED", reasons=(), summary={"synthetic": True})

    def load_mesh(self, path):
        self.events.append("mesh")
        return SimpleNamespace(synthetic=True)

    def load_iteration(self, run, iteration, *, expected_evidence_version):
        assert expected_evidence_version == 4
        self.events.append(("load", Path(run).name, iteration))
        if self.error:
            raise ValueError(self.error)
        validity = np.ones(103, dtype=bool)
        validity[:10] = False
        return SimpleNamespace(centers=np.zeros((103, 3)), evidence={"V_p": validity})

    def distances(self, centers, mesh):
        self.events.append(("distance", len(centers)))
        # Invalid-prior rows must still have a distance queried.
        assert len(centers) == 103
        return np.linspace(0., .2, 103)

    def evaluate_iteration(self, joined, distances, *, training_seed, iteration):
        self.events.append(("evaluate", training_seed, iteration))
        assert distances.shape == (103,)
        assert np.count_nonzero(joined.evidence["V_p"]) == 93
        if self.mutate:
            self.fixture.checkpoints[0].write_bytes(b"mutated checkpoint")
        result = iteration_summary(iteration, passing=True)
        result["domain"].update(eligible_count=93, finite_center_count=103,
                                coverage=93/103, positive_count=23, negative_count=70)
        return result


class PriorTransferCliTests(unittest.TestCase):
    def setUp(self):
        self.firewall = firewall_tests.UtilityGtFirewallTests()
        self.firewall.setUp()
        self.addCleanup(self.firewall.doCleanups)
        self.root = self.firewall.root
        self.source = self.root / "source"
        self.snapshot = self.root / "snapshot"
        self.source.mkdir()
        self.snapshot.mkdir()
        (self.source / "source.txt").write_bytes(b"synthetic source")
        (self.snapshot / "prior.npy").write_bytes(b"synthetic frozen prior")
        f = self.firewall.fixture
        f.source_record.write_bytes(canonical({"kind": "utility_source", "source_sha256": "1"*64,
                                              "source_root": str(self.source)}))
        f.snapshot_record.write_bytes(canonical({"kind": "utility_da3_snapshot", "snapshot_sha256": "2"*64,
                                                "source_sha256": "1"*64, "gt_access": "NONE",
                                                "snapshot_root": str(self.snapshot)}))
        self.repository = self.root / "repository"
        self.repository.mkdir()
        mesh_path = self.root / "mesh_aligned_0.05.ply"
        mesh_path.write_bytes(b"synthetic mesh")
        request = f.request()
        request["repository"]["root"] = str(self.repository)
        request["gt_mesh"].update(bytes=mesh_path.stat().st_size,
                                sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest())
        self.prior = build_prior_transfer_confirmation(**request)
        self.firewall.prior_identity = self.firewall.publish("evaluator-prior.json", self.prior)
        release = self.firewall.release()
        release["repository"]["root"] = str(self.repository)
        approval = json.loads(Path(release["approval"]["path"]).read_text())
        approval["repository"]["root"] = str(self.repository)
        release["approval"] = self.firewall.publish("approval.json", approval)
        self.geometry = self.firewall.publish("release.json", release)
        self.checkpoints = []
        self.qualifications = []
        for row in self.prior["runs"]:
            for name in ("run_dir", "view_dir", "launcher_dir"):
                Path(row[name]).mkdir(parents=True)
            Path(row["state_file"]).write_bytes(b"synthetic state")
            for iteration in (3000, 7000):
                cp = Path(row["run_dir"]) / f"chkpnt{iteration}.pth"
                cp.write_bytes(f"synthetic checkpoint {iteration}".encode())
                self.checkpoints.append(cp)
                snap = Path(row["run_dir"]) / "d0_evidence" / f"iteration_{iteration:06d}.npz"
                snap.parent.mkdir(exist_ok=True)
                snap.write_bytes(b"synthetic snapshot")
            paths = [Path(row["state_file"]), f.source_record, f.snapshot_record]
            for root in (self.source, self.snapshot, *(Path(row[name]) for name in ("run_dir", "view_dir", "launcher_dir"))):
                paths.extend(path for path in root.rglob("*") if path.is_file())
            fingerprints = {str(path): {"resolved_path": str(path.resolve()), "bytes": path.stat().st_size,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths}
            binding = {name: row[name] for name in ("seed", "run_dir", "view_dir", "qualification_path")}
            binding.update(repository_commit="a"*40, snapshot_sha256="2"*64, evidence_version=4)
            report = {"schema_version": 1, "kind": "prior_transfer_run_qualification", "outcome": "QUALIFIED",
                      "gt_access": "NONE", "training_started": False, "c1_authorized": False,
                      "confirmation_sha256": self.firewall.prior_identity["sha256"], "completion": {},
                      "point_counts": [100, 110, 120, 130, 140, 150, 160], "source_sha256": "1"*64,
                      "checkpoints": [], "input_fingerprints": fingerprints, "run_binding": binding}
            self.qualifications.append(self.firewall.publish(Path(row["qualification_path"]).name, report))
        self.args = SimpleNamespace(repository=str(self.repository), expected_commit="a"*40,
            confirmation=self.firewall.prior_identity["path"], confirmation_sha=self.firewall.prior_identity["sha256"],
            qualification_record=[[q["path"], q["sha256"]] for q in self.qualifications],
            geometry_release=self.geometry["path"], geometry_release_sha=self.geometry["sha256"],
            source_root=str(self.source), gt_mesh=self.prior["gt_mesh"]["path"],
            output_root=str(self.root), diagnostic_id="probe-output")
        # A synthetic mesh is not read at all by orchestration doubles.
        Path(self.args.gt_mesh).write_bytes(b"synthetic mesh")
        self.dependencies = Dependencies(self)

    def target(self):
        return Path(self.prior["probe_targets"]["output_dir"])

    def test_parser_has_no_scientific_or_execution_override(self):
        options = {option for action in build_parser()._actions for option in action.option_strings}
        self.assertEqual(options, {"-h", "--help", "--repository", "--expected-commit", "--confirmation",
            "--confirmation-sha", "--qualification-record", "--geometry-release", "--geometry-release-sha",
            "--source-root", "--gt-mesh", "--output-root", "--diagnostic-id"})

    def test_binds_all_runs_before_first_access_and_queries_all_finite_rows(self):
        from reliability.utility_gt_firewall import authorize_first_gt_access
        calls = []
        def authorize(*args, **kwargs):
            calls.append("access")
            return authorize_first_gt_access(*args, **kwargs)
        with mock.patch("scripts.diagnostics.evaluate_g1_prior_transfer.authorize_first_gt_access", side_effect=authorize):
            code, publication = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["access"])
        self.assertEqual(self.dependencies.events[:2], ["admit", "mesh"])
        self.assertEqual([e for e in self.dependencies.events if isinstance(e, tuple) and e[0] == "load"],
            [("load", f"run-{seed}", it) for seed in range(3) for it in (3000, 7000)])
        for index, event in enumerate(self.dependencies.events):
            if isinstance(event, tuple) and event[0] == "evaluate":
                self.assertEqual(self.dependencies.events[index-1], ("distance", 103))
        report = publication["report"]
        self.assertEqual(report["outcome"], "PRIOR_RISK_TRANSFER_SUPPORTED")
        self.assertEqual(report["provenance"]["confirmation_sha256"], self.args.confirmation_sha)
        self.assertFalse(report["c1_authorized"])
        self.assertFalse(report["routing_authorized"])
        self.assertIsNone(report["causal_claim"])

    def test_rejects_wrong_binding_before_consuming_first_access(self):
        path = Path(self.qualifications[1]["path"])
        record = json.loads(path.read_text())
        record["run_binding"]["repository_commit"] = "b"*40
        bad = self.firewall.publish(path.name, record)
        self.args.qualification_record[1] = [bad["path"], bad["sha256"]]
        with self.assertRaises(ValueError):
            run_evaluator(self.args, self.dependencies)
        self.assertFalse(self.firewall.log.exists())
        self.assertFalse(self.target().exists())
        self.assertEqual(self.dependencies.events, [])

    def test_rejects_qualification_mutation_and_missing_seed_before_access(self):
        for change in ("sha", "missing", "path"):
            with self.subTest(change=change):
                args = copy.deepcopy(self.args)
                if change == "sha":
                    args.qualification_record[0][1] = "0"*64
                elif change == "missing":
                    args.qualification_record.pop()
                else:
                    args.qualification_record[0][0] = args.gt_mesh
                with self.assertRaises((ValueError, OSError)):
                    run_evaluator(args, self.dependencies)
                self.assertFalse(self.firewall.log.exists())

    def test_unsafe_request_or_existing_target_never_publishes(self):
        for change in ("id", "output", "source", "gt", "commit"):
            with self.subTest(change=change):
                args = copy.deepcopy(self.args)
                if change == "id": args.diagnostic_id = "../escape"
                if change == "output": args.output_root = str(self.source)
                if change == "source": args.source_root = str(self.snapshot)
                if change == "gt": args.gt_mesh = str(self.root / "other.ply")
                if change == "commit": args.expected_commit = "b"*40
                with self.assertRaises((ValueError, OSError)):
                    run_evaluator(args, self.dependencies)
                self.assertFalse(self.firewall.log.exists())
        self.target().mkdir()
        marker = self.target() / "user.txt"
        marker.write_bytes(b"preserve")
        with self.assertRaises((ValueError, OSError)):
            run_evaluator(self.args, self.dependencies)
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_compact_manifest_has_exact_files_and_seed_csv_inventory(self):
        code, publication = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 0)
        self.assertEqual({p.name for p in self.target().iterdir()}, {"inputs.json", "mesh_admission.json",
            "report.json", "seed_folds.csv", "risk_bins.csv", "bootstrap.csv", "manifest.json"})
        manifest = json.loads((self.target()/"manifest.json").read_text())
        self.assertEqual([entry["path"] for entry in manifest["files"]], sorted([
            "inputs.json", "mesh_admission.json", "report.json", "seed_folds.csv", "risk_bins.csv", "bootstrap.csv"]))
        for entry in manifest["files"]:
            payload = (self.target()/entry["path"]).read_bytes()
            self.assertEqual(len(payload), entry["bytes"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
        with (self.target()/"seed_folds.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertEqual({row["training_seed"] for row in rows}, {"0", "1", "2"})
        self.assertFalse(Path(self.prior["probe_targets"]["staging_dir"]).exists())
        self.assertEqual(list(self.root.glob("*.tar.gz")), [])
        with self.assertRaises((ValueError, OSError)):
            run_evaluator(self.args, self.dependencies)

    def test_bad_computation_publishes_inconclusive_not_a_fallback(self):
        self.dependencies.error = "synthetic damaged checkpoint"
        code, publication = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 2)
        self.assertEqual(publication["report"]["outcome"], "INCONCLUSIVE")
        self.assertIn("damaged checkpoint", publication["report"]["inconclusive_reasons"][0])
        self.assertTrue(self.firewall.log.exists())

    def test_mesh_failure_stops_before_loading_any_checkpoint(self):
        self.dependencies.admit_mesh = lambda *a, **k: SimpleNamespace(outcome="INCONCLUSIVE",
            reasons=("synthetic alignment failed",), summary={})
        code, publication = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 2)
        self.assertEqual(self.dependencies.events, [])
        self.assertEqual(publication["report"]["seeds"], [])

    def test_input_mutation_aborts_publication_but_keeps_access_log(self):
        self.dependencies.mutate = True
        with self.assertRaises((ValueError, RuntimeError)):
            run_evaluator(self.args, self.dependencies)
        self.assertFalse(self.target().exists())
        self.assertFalse(Path(self.prior["probe_targets"]["staging_dir"]).exists())
        self.assertTrue(self.firewall.log.exists())

    def test_publish_failure_cleans_only_own_staging(self):
        with mock.patch("scripts.diagnostics.evaluate_g1_prior_transfer.build_manifest", side_effect=RuntimeError("synthetic write failure")):
            with self.assertRaises(RuntimeError):
                run_evaluator(self.args, self.dependencies)
        self.assertFalse(self.target().exists())
        self.assertFalse(Path(self.prior["probe_targets"]["staging_dir"]).exists())
        self.assertTrue(self.firewall.log.exists())

    def test_bad_geometry_release_stops_before_any_mesh_or_checkpoint(self):
        self.args.geometry_release_sha = "0"*64
        with self.assertRaises(ValueError):
            run_evaluator(self.args, self.dependencies)
        self.assertEqual(self.dependencies.events, [])
        self.assertFalse(self.firewall.log.exists())

    def test_prior_and_qualification_gt_aliases_are_rejected_before_byte_read(self):
        import os
        from reliability import utility_gt_firewall
        for which in ("confirmation", "qualification"):
            args = copy.deepcopy(self.args)
            alias = self.root / (which + "-alias.json")
            os.link(self.args.gt_mesh, alias)
            if which == "confirmation":
                args.confirmation = str(alias)
            else:
                args.qualification_record[0][0] = str(alias)
            original = Path.open
            def reject_gt(path, *a, **k):
                if path.exists() and path.samefile(Path(self.args.gt_mesh)):
                    raise AssertionError("GT alias read before first-access log")
                return original(path, *a, **k)
            with mock.patch.object(Path, "open", reject_gt):
                with self.assertRaises(ValueError):
                    run_evaluator(args, self.dependencies)
            self.assertFalse(self.firewall.log.exists())

    def test_protected_binding_does_not_reopen_general_reference_loader(self):
        with mock.patch("reliability.g1_prior_transfer_confirmation._verify_record_file",
                        side_effect=AssertionError("unguarded reference read")):
            code, _ = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 0)

    def test_atomic_publication_preserves_concurrent_output_and_own_log(self):
        from scripts.diagnostics.evaluate_g1_prior_transfer import _rename_exclusive
        def racing_rename(staging, target):
            target.mkdir()
            (target / "user.txt").write_bytes(b"preserve concurrent result")
            _rename_exclusive(staging, target)
        with mock.patch("scripts.diagnostics.evaluate_g1_prior_transfer._rename_exclusive", side_effect=racing_rename):
            with self.assertRaises(OSError):
                run_evaluator(self.args, self.dependencies)
        self.assertEqual((self.target()/"user.txt").read_bytes(), b"preserve concurrent result")
        self.assertFalse(Path(self.prior["probe_targets"]["staging_dir"]).exists())
        self.assertTrue(self.firewall.log.exists())

    def test_transient_checkpoint_write_restore_aborts_publication(self):
        original_evaluate = self.dependencies.evaluate_iteration
        def temporary_change(*a, **k):
            path = self.checkpoints[0]
            before = path.read_bytes()
            path.write_bytes(b"temporary mutation")
            path.write_bytes(before)
            return original_evaluate(*a, **k)
        self.dependencies.evaluate_iteration = temporary_change
        with self.assertRaises(RuntimeError):
            run_evaluator(self.args, self.dependencies)
        self.assertFalse(self.target().exists())
        self.assertTrue(self.firewall.log.exists())

    def test_gt_replacement_after_admission_cannot_become_new_baseline(self):
        original_admit = self.dependencies.admit_mesh
        def replace_after_admission(*a, **k):
            admitted = original_admit(*a, **k)
            Path(self.args.gt_mesh).write_bytes(b"different post-admission mesh")
            return admitted
        self.dependencies.admit_mesh = replace_after_admission
        with self.assertRaises((ValueError, RuntimeError)):
            run_evaluator(self.args, self.dependencies)
        self.assertEqual(self.dependencies.events, ["admit"])
        self.assertFalse(self.target().exists())
        self.assertTrue(self.firewall.log.exists())

    def test_initial_gt_identity_failure_is_inconclusive_without_parsing(self):
        Path(self.args.gt_mesh).write_bytes(b"wrong initial mesh identity")
        code, publication = run_evaluator(self.args, self.dependencies)
        self.assertEqual(code, 2)
        self.assertEqual(publication["report"]["outcome"], "INCONCLUSIVE")
        self.assertEqual(self.dependencies.events, [])
        self.assertTrue(self.firewall.log.exists())

    def test_gt_write_restore_during_admission_aborts_before_evaluation(self):
        original_admit = self.dependencies.admit_mesh
        def mutate_restore(*a, **k):
            admitted = original_admit(*a, **k)
            path = Path(self.args.gt_mesh)
            before = path.read_bytes()
            path.write_bytes(b"transient GT mutation")
            path.write_bytes(before)
            return admitted
        self.dependencies.admit_mesh = mutate_restore
        with self.assertRaises((ValueError, RuntimeError)):
            run_evaluator(self.args, self.dependencies)
        self.assertEqual(self.dependencies.events, ["admit"])
        self.assertFalse(self.target().exists())

    def test_prior_replaced_by_gt_alias_at_authorizer_reload_is_not_opened(self):
        import os
        from reliability.utility_gt_firewall import authorize_first_gt_access
        original_open = Path.open
        def guarded_open(path, *a, **k):
            if path.exists() and path.samefile(Path(self.args.gt_mesh)):
                raise AssertionError("GT inode opened during confirmation reload")
            return original_open(path, *a, **k)
        def replaced_authorize(*a, **k):
            path = Path(self.args.confirmation)
            path.unlink()
            os.link(self.args.gt_mesh, path)
            with mock.patch.object(Path, "open", guarded_open):
                return authorize_first_gt_access(*a, **k)
        with mock.patch("scripts.diagnostics.evaluate_g1_prior_transfer.authorize_first_gt_access",
                        side_effect=replaced_authorize):
            with self.assertRaises(ValueError):
                run_evaluator(self.args, self.dependencies)
        self.assertFalse(self.firewall.log.exists())
        self.assertEqual(self.dependencies.events, [])


if __name__ == "__main__":
    unittest.main()
