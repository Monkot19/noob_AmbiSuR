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

from tests.test_utility_gt_firewall import UtilityGtFirewallTests, canonical
from tests.test_g1_complementarity_cli import iteration_summary
from reliability.g1_prior_transfer_confirmation import build_prior_transfer_confirmation
from scripts.diagnostics.evaluate_g1_prior_transfer import build_parser, run_evaluator


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
        self.firewall = UtilityGtFirewallTests()
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
        self.prior = build_prior_transfer_confirmation(**f.request())
        self.firewall.prior_identity = self.firewall.publish("evaluator-prior.json", self.prior)
        self.geometry = self.firewall.publish("release.json", self.firewall.release())
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
        self.args = SimpleNamespace(repository=str(self.root), expected_commit="a"*40,
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


if __name__ == "__main__":
    unittest.main()
