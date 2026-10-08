"""Firewall tests use synthetic records only; no real mesh or release is created."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace
import importlib.util

import numpy as np

from tests import test_g1_prior_transfer_confirmation as prior_tests
from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation, write_prior_transfer_confirmation,
)
from reliability.utility_gt_firewall import (
    authorize_first_gt_access, load_geometry_release, validate_geometry_release,
    audit_utility_mesh, select_admission_points, utility_camera_rays,
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
                self.authorize(self.publish("release.json", changed))

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

    def test_public_release_validation_never_opens_references(self):
        record = self.release()
        mesh = Path(self.prior["gt_mesh"]["path"])
        record["evidence"] = [{"path": str(mesh), "sha256": "3" * 64}]
        identity = self.publish("release.json", record)
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path, mesh, "Early structural validation cannot read GT")
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded_open):
            self.assertEqual(validate_geometry_release(record), "NO_ACTION_SPECIFIC_SIGNAL")
            self.assertEqual(load_geometry_release(Path(identity["path"]), identity["sha256"]), record)
        self.assertFalse(self.log.exists())

    def test_hardlink_alias_is_rejected_without_reading_protected_inode(self):
        import os
        record = self.release()
        mesh = Path(self.prior["gt_mesh"]["path"])
        mesh.write_bytes(Path(self.evidence["path"]).read_bytes())
        alias = self.root / "hardlink.json"
        os.link(mesh, alias)
        record["evidence"] = [{"path": str(alias), "sha256": self.evidence["sha256"]}]
        identity = self.publish("release.json", record)
        real_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.is_file():
                self.assertFalse(path.samefile(mesh), "Do not open a protected hardlink")
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded_open), self.assertRaises(ValueError):
            self.authorize(identity)
        self.assertFalse(self.log.exists())

    def test_reference_replaced_at_open_is_rejected_before_read(self):
        import os
        record = self.release()
        mesh = Path(self.prior["gt_mesh"]["path"])
        evidence = Path(self.evidence["path"])
        mesh.write_bytes(evidence.read_bytes())
        identity = self.publish("release.json", record)
        real_open = Path.open
        reads = []
        class WatchedStream:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def fileno(self):
                return self.stream.fileno()
            def read(self, *args):
                reads.append("protected bytes read")
                return self.stream.read(*args)
        def replace_at_open(path, *args, **kwargs):
            if path == evidence:
                path.unlink()
                os.link(mesh, path)
                return WatchedStream(real_open(path, *args, **kwargs))
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", replace_at_open), self.assertRaises(ValueError):
            self.authorize(identity)
        self.assertEqual(reads, [])
        self.assertFalse(self.log.exists())


class UtilityMeshAdmissionTests(unittest.TestCase):
    """New admission contracts only; statistical probe remains untouched."""

    def setUp(self):
        from tests.test_utility_snapshot import _write_source
        from reliability.utility_snapshot import audit_utility_source, source_manifest
        self.firewall = UtilityGtFirewallTests()
        self.firewall.setUp()
        self.addCleanup(self.firewall.doCleanups)
        self.root = self.firewall.root
        self.source = _write_source(self.root / "source", tuple(f"frame{i:03d}.jpg" for i in range(147)))
        sparse = self.source / "sparse/0"
        (sparse / "images.txt").write_text("".join(
            f"{i+1} 1 0 0 0 0 0 0 1 frame{i:03d}.jpg\n\n" for i in range(147)))
        (sparse / "points3D.txt").write_text("".join(
            f"{i} 0 0 1 10 20 30 0.1 1 0\n" for i in range(1, 11)))
        self.mesh = self.root / "mesh_aligned_0.05.ply"
        self.mesh.write_text(
            "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
            "property float z\nelement face 3\nproperty list uchar int vertex_indices\nend_header\n"
            "-10 -10 1\n10 -10 1\n10 10 1\n-10 10 1\n3 0 1 2\n3 0 2 3\n3 0 0 0\n")
        manifest = source_manifest(audit_utility_source(self.source, 147))
        manifest.update(audit_kind="utility_source", gt_access="NONE")
        f = self.firewall.fixture
        f.source_record.write_bytes(canonical(manifest))
        request = f.request()
        request["gt_mesh"].update(bytes=self.mesh.stat().st_size,
                                  sha256=hashlib.sha256(self.mesh.read_bytes()).hexdigest())
        self.prior = build_prior_transfer_confirmation(**request)
        published = self.firewall.publish("mesh-prior.json", self.prior)
        self.firewall.prior_identity = published
        self.token = self.firewall.authorize(self.firewall.publish("release.json", self.firewall.release()))
        self.surface = SimpleNamespace(
            vertices=np.array([[-10., -10., 1.], [10., -10., 1.], [10., 10., 1.], [-10., 10., 1.]]),
            triangles=np.array([[0, 1, 2], [0, 2, 3]]), source_vertex_count=4,
            source_triangle_count=3, nonfinite_vertex_count=0,
            rejected_nonfinite_triangle_count=0, rejected_degenerate_triangle_count=1,
            rejected_triangle_count=1)

    def run_admission(self, *, distances=None, depths=None, loader=None, token=None, prior=None, mesh=None, source=None):
        with mock.patch("reliability.utility_gt_firewall._load_admission_mesh", return_value=self.surface) as load, \
             mock.patch("reliability.utility_gt_firewall._admission_distances", return_value=(
                 np.zeros(10) if distances is None else distances)) as query, \
             mock.patch("reliability.utility_gt_firewall._admission_ray_depths", return_value=(
                 np.ones((147, 48)) if depths is None else depths)) as cast:
            if loader is not None:
                load.side_effect = loader
            result = audit_utility_mesh(mesh or self.mesh, source or self.source, prior or self.prior,
                                        access_token=self.token if token is None else token)
            return result, load, query, cast

    def test_requires_verified_disk_token_before_parser(self):
        for token in ({}, {**self.token, "sha256": "0"*64},
                      {**self.token, "record": {}}):
            with self.subTest(token=token):
                result, load, query, cast = self.run_admission(token=token)
                self.assertEqual(result.outcome, "INCONCLUSIVE")
                load.assert_not_called()
                query.assert_not_called()
                cast.assert_not_called()
        Path(self.token["path"]).write_bytes(b"changed")
        result, load, _, _ = self.run_admission()
        self.assertEqual(result.outcome, "INCONCLUSIVE")
        load.assert_not_called()

    def test_mesh_identity_and_frozen_contract_fail_before_parse(self):
        wrong = self.root / "wrong.ply"
        wrong.write_bytes(self.mesh.read_bytes())
        changed = copy.deepcopy(self.prior)
        changed["mesh_admission"]["coordinate_transform"][0][0] = 2.0
        for kwargs in ({"mesh": wrong}, {"prior": changed}, {"source": self.root}):
            with self.subTest(kwargs=kwargs):
                result, load, _, _ = self.run_admission(**kwargs)
                self.assertEqual(result.outcome, "INCONCLUSIVE")
                load.assert_not_called()
        self.mesh.write_bytes(b"invalid replacement")
        result, load, _, _ = self.run_admission()
        self.assertEqual(result.outcome, "INCONCLUSIVE")
        load.assert_not_called()

    def test_all_valid_faces_retained_and_summary_has_no_domain_selector(self):
        result, load, query, cast = self.run_admission()
        self.assertEqual(result.outcome, "ADMITTED")
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.summary["surface"]["valid_triangle_count"], 2)
        self.assertEqual(result.summary["surface"]["rejected_degenerate_triangle_count"], 1)
        self.assertEqual(result.summary["coordinate_transform"], np.eye(4).tolist())
        self.assertIs(query.call_args.args[1], self.surface)
        self.assertIs(cast.call_args.args[1], self.surface)
        self.assertEqual(cast.call_args.args[0].shape, (147, 48, 6))
        self.assertEqual(set(vars(result)), {"outcome", "reasons", "summary"})
        self.assertNotIn("mask", json.dumps(result.summary))
        self.assertNotIn("selected_rows", json.dumps(result.summary))
        load.assert_called_once_with(self.mesh)

    def test_unloadable_nonfinite_or_empty_surface_is_inconclusive(self):
        for loader in (ValueError("no valid triangles"), OSError("unloadable mesh")):
            result, _, query, cast = self.run_admission(loader=loader)
            self.assertEqual(result.outcome, "INCONCLUSIVE")
            query.assert_not_called()
            cast.assert_not_called()
        self.surface.nonfinite_vertex_count = 1
        result, _, query, _ = self.run_admission()
        self.assertEqual(result.outcome, "INCONCLUSIVE")
        query.assert_not_called()

    def test_sha_minhash_sampling_is_byte_defined_order_independent_and_bounded(self):
        points = {i: SimpleNamespace(xyz=np.array([i/100., 0., 1.])) for i in range(50002)}
        expected = sorted(points, key=lambda i: (
            hashlib.sha256(np.array([i], dtype="<u8").tobytes() +
                           np.asarray(points[i].xyz, dtype="<f8").tobytes()).digest(), i))[:50000]
        ids, xyz = select_admission_points(points)
        self.assertEqual(ids.tolist(), expected)
        reverse_ids, reverse_xyz = select_admission_points(dict(reversed(list(points.items()))))
        np.testing.assert_array_equal(ids, reverse_ids)
        np.testing.assert_array_equal(xyz, reverse_xyz)
        ids, xyz = select_admission_points({2: points[2], 1: points[1]})
        self.assertEqual(len(ids), 2)
        np.testing.assert_array_equal(xyz, [points[i].xyz for i in ids])

    def test_point_distance_thresholds_are_inclusive_and_fail_closed(self):
        admitted = np.array([0.]*8 + [0.15]*2)
        result, _, _, _ = self.run_admission(distances=admitted)
        self.assertEqual(result.outcome, "ADMITTED")
        self.assertEqual(result.summary["alignment"]["fraction_within_0_10_m"], 0.8)
        for distances in (np.full(10, .050001), np.array([0.]*8 + [.150001]*2),
                          np.array([0.]*7 + [.100001]*3), np.full(10, np.nan),
                          np.full(10, -1.), np.zeros(9)):
            with self.subTest(distances=distances):
                result, _, _, cast = self.run_admission(distances=distances)
                self.assertEqual(result.outcome, "INCONCLUSIVE")
                cast.assert_not_called()

    def test_full_frame_grid_and_inverse_pose_match_pixel_centers(self):
        from scripts.preprocess.read_write_model import Camera, Image
        camera = Camera(1, "PINHOLE", 80, 60, np.array([40., 40., 40., 30.]))
        image = Image(9, np.array([1., 0., 0., 0.]), np.array([-2., -3., -4.]), 1,
                      "synthetic.jpg", np.empty((0, 2)), np.empty(0, dtype=int))
        rays = utility_camera_rays(camera, image)
        self.assertEqual(rays.shape, (48, 6))
        np.testing.assert_allclose(rays[:, :3], np.tile([2., 3., 4.], (48, 1)))
        np.testing.assert_allclose(rays[0, 3:], [-.875, -.625, 1.])
        np.testing.assert_allclose(rays[-1, 3:], [.875, .625, 1.])
        simple = camera._replace(model="SIMPLE_PINHOLE", params=np.array([40., 40., 30.]))
        np.testing.assert_allclose(utility_camera_rays(simple, image), rays)

    def test_coverage_aggregate_and_per_camera_gates_are_both_required(self):
        admitted = np.ones((147, 48))
        admitted[100:, 24:] = np.inf
        self.assertEqual(self.run_admission(depths=admitted)[0].outcome, "ADMITTED")
        poor_aggregate = np.ones((147, 48))
        poor_aggregate[:, 38:] = np.inf
        poor_cameras = np.ones((147, 48))
        poor_cameras[132:] = np.inf
        for depths in (poor_aggregate, poor_cameras, np.ones((146, 48)),
                       np.full((147, 48), np.nan), np.full((147, 48), -1.)):
            with self.subTest(shape=depths.shape):
                self.assertEqual(self.run_admission(depths=depths)[0].outcome, "INCONCLUSIVE")

    def test_source_mutation_camera_inventory_and_invalid_ray_stop_before_gt_parse(self):
        camera = self.source / "sparse/0/cameras.txt"
        camera.write_text("1 PINHOLE 8 6 0 4 4 3\n")
        result, load, _, _ = self.run_admission()
        self.assertEqual(result.outcome, "INCONCLUSIVE")
        load.assert_not_called()

    def test_post_read_input_mutation_cannot_return_admitted(self):
        def mutate(path):
            self.mesh.write_bytes(b"changed after parse")
            return self.surface
        result, _, _, _ = self.run_admission(loader=mutate)
        self.assertEqual(result.outcome, "INCONCLUSIVE")

    def test_transient_mesh_rewrite_and_restore_is_not_admitted(self):
        original = self.mesh.read_bytes()
        def mutate_restore(path):
            self.mesh.write_bytes(original.replace(b"-10 -10 1", b"-10 -10 9"))
            self.mesh.write_bytes(original)
            return self.surface
        result, _, _, _ = self.run_admission(loader=mutate_restore)
        self.assertEqual(self.mesh.read_bytes(), original)
        self.assertEqual(result.outcome, "INCONCLUSIVE")

    def test_transient_colmap_rewrite_and_restore_stops_before_mesh_parser(self):
        from scripts.preprocess.read_write_model import read_model
        path = self.source / "sparse/0/points3D.txt"
        original = path.read_bytes()
        def mutate_restore(*args, **kwargs):
            path.write_bytes(original.replace(b" 0 0 1 ", b" 9 0 1 "))
            try:
                return read_model(*args, **kwargs)
            finally:
                path.write_bytes(original)
        with mock.patch("scripts.preprocess.read_write_model.read_model", side_effect=mutate_restore):
            result, load, query, _ = self.run_admission()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(result.outcome, "INCONCLUSIVE")
        load.assert_not_called()
        query.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("open3d"),
                         "real mesh backend requires Torch/Open3D; qualify on AutoDL")
    def test_real_synthetic_mesh_backend_keeps_full_surface_and_casts_all_cameras(self):
        result = audit_utility_mesh(self.mesh, self.source, self.prior, access_token=self.token)
        self.assertEqual(result.outcome, "ADMITTED", result.reasons)
        self.assertEqual(result.summary["surface"]["valid_triangle_count"], 2)
        self.assertEqual(result.summary["coverage"]["aggregate_hit_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
