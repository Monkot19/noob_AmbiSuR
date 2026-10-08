"""Firewall tests use synthetic records only; no real mesh or release is created."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from tests import test_g1_prior_transfer_confirmation as prior_tests
from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation, write_prior_transfer_confirmation,
)
from reliability.utility_gt_firewall import (
    authorize_first_gt_access, load_geometry_release, validate_geometry_release,
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class UtilityGtFirewallTests(unittest.TestCase):
    def setUp(self):
        self.fixture = prior_tests.PriorTransferConfirmationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.prior = build_prior_transfer_confirmation(**self.fixture.request())
        published = write_prior_transfer_confirmation(self.prior, self.root / "prior.json")
        self.prior_identity = {"path": str(published["path"]), "sha256": published["sha256"]}
        self.log = Path(self.prior["probe_targets"]["access_log_path"])
        self.charter = self.publish("charter.md", "synthetic approved charter")
        self.spec = self.publish("stage-spec.md", "synthetic approved stage specification")
        self.evidence = self.publish("evidence.json", {"synthetic": True, "gt_access": "NONE"})

    def publish(self, name, value):
        path = self.root / name
        payload = value.encode() if isinstance(value, str) else canonical(value)
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")
        return {"path": str(path), "sha256": digest}

    def release(self, candidate=False):
        contract = None
        if candidate:
            # Opaque synthetic values, NOT a proposed geometry action/formula.
            contract = {name: {"synthetic_frozen_value": name} for name in (
                "action", "cluster_unit", "affected_parameters", "outcome", "horizon",
                "formula", "constants", "validity", "state", "topology_lineage",
                "thresholds", "metrics", "coverage", "compute_budget", "stop_rules",
            )}
        record = {
            "schema_version": 1, "kind": "geometry_release", "release_id": "synthetic_release",
            "created_utc": "2026-09-30T11:00:00Z",
            "repository": {"root": str(self.root), "commit": "b" * 40, "clean": True},
            "stage": "G-C" if candidate else "G-A",
            "outcome": "STAGE_G_C_CANDIDATE" if candidate else "NO_ACTION_SPECIFIC_SIGNAL",
            "specifications": {"charter": self.charter, "stage_specification": self.spec},
            "evidence": [self.evidence], "frozen_contract": contract,
            "candidate_admitted_to_utility": candidate, "utility_cannot_reopen": True,
            "utility_gt_access": "METADATA_ONLY",
        }
        approval = {"schema_version": 1, "kind": "geometry_release_approval",
                    **{key: record[key] for key in (
                        "release_id", "stage", "outcome", "repository", "specifications",
                        "evidence", "candidate_admitted_to_utility", "utility_cannot_reopen",
                        "utility_gt_access")},
                    "frozen_contract_sha256": hashlib.sha256(canonical(contract)).hexdigest()}
        record["approval"] = self.publish("approval.json", approval)
        return record

    def authorize(self, geometry):
        return authorize_first_gt_access(self.prior_identity, geometry, access_log_path=self.log)

    def test_candidate_binds_reviewed_contract_and_exact_disk_identities(self):
        record = self.release(candidate=True)
        self.assertEqual(validate_geometry_release(record), "STAGE_G_C_CANDIDATE")
        identity = self.publish("release.json", record)
        self.assertEqual(load_geometry_release(Path(identity["path"]), identity["sha256"]), record)
        token = self.authorize(identity)
        self.assertEqual(token["record"]["geometry_release"], identity)

    def test_candidate_rejects_draft_missing_fields_or_post_review_changes(self):
        record = self.release(candidate=True)
        mutations = []
        for stage in ("G-A", "G-B"):
            changed = copy.deepcopy(record)
            changed["stage"] = stage
            mutations.append(changed)
        for name in record["frozen_contract"]:
            changed = copy.deepcopy(record)
            del changed["frozen_contract"][name]
            mutations.append(changed)
        for name in ("formula", "constants", "action"):
            changed = copy.deepcopy(record)
            changed["frozen_contract"][name] = {"changed": True}
            mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_geometry_release(changed)

    def test_terminal_requires_permanent_reviewed_no_candidate_outcome(self):
        record = self.release()
        self.assertEqual(validate_geometry_release(record), "NO_ACTION_SPECIFIC_SIGNAL")
        for key, value in (
            ("outcome", "NO_SEMANTIC_REPAIR_JUSTIFIED"), ("outcome", "PASS"),
            ("candidate_admitted_to_utility", True), ("utility_cannot_reopen", False),
            ("utility_gt_access", "EVALUATED"), ("stage", "UNKNOWN"),
            ("frozen_contract", {"repackaged_r_g": True}),
        ):
            changed = copy.deepcopy(record)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_geometry_release(changed)

    def test_missing_or_mutated_release_blocks_parser(self):
        identity = self.publish("release.json", self.release())
        parser = mock.Mock()
        for mutation in ("missing", "bytes", "detached", "evidence", "spec", "approval"):
            with self.subTest(mutation=mutation):
                original = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
                if mutation == "missing":
                    Path(identity["path"]).unlink()
                else:
                    target = {"bytes": identity["path"], "detached": identity["path"] + ".sha256",
                              "evidence": self.evidence["path"], "spec": self.spec["path"],
                              "approval": str(self.root / "approval.json")}[mutation]
                    Path(target).write_bytes(b"changed")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.authorize(identity)
                    parser()
                parser.assert_not_called()
                self.assertFalse(self.log.exists())
                for path, payload in original.items():
                    path.write_bytes(payload)

    def test_prior_reload_detects_mutation_and_never_reads_mesh(self):
        identity = self.publish("release.json", self.release())
        mesh = Path(self.prior["gt_mesh"]["path"])
        real_read = Path.read_bytes
        def guarded_read(path):
            self.assertNotEqual(path, mesh, "Task 6 must not read mesh bytes")
            return real_read(path)
        with mock.patch.object(Path, "read_bytes", guarded_read):
            token = self.authorize(identity)
        self.assertEqual(token["record"]["gt_mesh"], self.prior["gt_mesh"])
        self.log.unlink()
        Path(self.prior_identity["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.authorize(identity)
        self.assertFalse(self.log.exists())

    def test_log_precedes_parser_and_is_exclusive_even_with_same_identity(self):
        identity = self.publish("release.json", self.release())
        def parse():
            record = json.loads(self.log.read_text())
            self.assertEqual(record["prior_confirmation"], self.prior_identity)
            self.assertEqual(record["geometry_release"], identity)
            self.assertEqual(record["kind"], "utility_first_gt_access")
        parser = mock.Mock(side_effect=parse)
        token = self.authorize(identity)
        parser()
        self.assertEqual(token["sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        before = self.log.read_bytes()
        with self.assertRaises(FileExistsError):
            self.authorize(identity)
        self.assertEqual(self.log.read_bytes(), before)
        parser.assert_called_once()

    def test_conflicting_log_output_staging_and_wrong_target_fail_closed(self):
        identity = self.publish("release.json", self.release())
        for name in ("output_dir", "staging_dir", "access_log_path"):
            path = Path(self.prior["probe_targets"][name])
            path.write_text("existing GT-derived artifact")
            with self.subTest(name=name), self.assertRaises((ValueError, FileExistsError)):
                self.authorize(identity)
            path.unlink()
        with self.assertRaises(ValueError):
            authorize_first_gt_access(self.prior_identity, identity,
                                      access_log_path=self.root / "unregistered.json")
        self.assertFalse(self.log.exists())

    def test_future_naive_or_pre_approval_chronology_is_rejected(self):
        for timestamp in ("2099-01-01T00:00:00Z", "2026-09-30T11:00:00", "invalid"):
            record = self.release()
            record["created_utc"] = timestamp
            identity = self.publish("release.json", record)
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.authorize(identity)
            self.assertFalse(self.log.exists())

    def test_in_memory_records_are_not_accepted_as_verified_identity_handles(self):
        with self.assertRaises(ValueError):
            authorize_first_gt_access(self.prior, self.release(), access_log_path=self.log)

    def test_publication_race_never_overwrites_a_concurrent_log(self):
        identity = self.publish("release.json", self.release())
        import os
        real_link = os.link
        def competing_link(source, target):
            Path(target).write_bytes(b"concurrent access")
            return real_link(source, target)
        with mock.patch("reliability.utility_gt_firewall.os.link", side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self.authorize(identity)
        self.assertEqual(self.log.read_bytes(), b"concurrent access")
        self.assertFalse(list(self.root.glob(".*.tmp-*")))

    def test_mutation_at_publication_blocks_return_and_retains_consumed_log(self):
        identity = self.publish("release.json", self.release())
        import os
        real_link = os.link
        def mutate_after_link(source, target):
            real_link(source, target)
            Path(identity["path"]).write_bytes(b"changed during publication")
        with mock.patch("reliability.utility_gt_firewall.os.link", side_effect=mutate_after_link):
            with self.assertRaises(ValueError):
                self.authorize(identity)
        self.assertTrue(self.log.exists(), "Do not erase a possibly consumed access record")

    def test_release_reference_aliasing_gt_is_rejected_before_any_mesh_read(self):
        record = self.release()
        mesh = Path(self.prior["gt_mesh"]["path"])
        record["evidence"] = [{"path": str(mesh), "sha256": "3" * 64}]
        identity = self.publish("release.json", record)
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path, mesh, "Firewall must reject before opening GT")
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded_open), self.assertRaises(ValueError):
            self.authorize(identity)
        self.assertFalse(self.log.exists())

    def test_post_publication_probe_artifact_blocks_parser_return(self):
        identity = self.publish("release.json", self.release())
        import os
        real_link = os.link
        def artifact_after_link(source, target):
            real_link(source, target)
            Path(self.prior["probe_targets"]["staging_dir"]).mkdir()
        with mock.patch("reliability.utility_gt_firewall.os.link", side_effect=artifact_after_link):
            with self.assertRaises(ValueError):
                self.authorize(identity)
        self.assertTrue(self.log.exists())


if __name__ == "__main__":
    unittest.main()
