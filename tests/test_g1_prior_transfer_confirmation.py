import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation,
    load_prior_transfer_confirmation,
    validate_completed_run_binding,
    validate_preregistration_targets,
    write_prior_transfer_confirmation,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PriorTransferConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source_record = self.root / "source.json"
        self.snapshot_record = self.root / "snapshot.json"
        self.source_record.write_text(
            json.dumps({"kind": "utility_source", "source_sha256": "1" * 64}),
            encoding="utf-8",
        )
        self.snapshot_record.write_text(
            json.dumps(
                {
                    "kind": "utility_da3_snapshot",
                    "snapshot_sha256": "2" * 64,
                    "source_sha256": "1" * 64,
                    "gt_access": "NONE",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        runs = []
        for seed in (0, 1, 2):
            run = self.root / f"run-{seed}"
            view = self.root / f"view-{seed}" / "colmap_undistorted"
            argv = [
                "/env/python",
                "train.py",
                "--source_path",
                str(view),
                "--model_path",
                str(run),
                "-r",
                "2",
                "--iterations",
                "7000",
                "--seed",
                str(seed),
                "--core_shadow_mode",
                "--d0_refresh_interval",
                "1000",
                "--test_iterations",
                "1000",
                "2000",
                "3000",
                "4000",
                "5000",
                "6000",
                "7000",
                "--save_iterations",
                "7000",
                "--checkpoint_iterations",
                "3000",
                "7000",
            ]
            runs.append(
                {
                    "seed": seed,
                    "run_dir": str(run),
                    "view_dir": str(view),
                    "state_file": str(self.root / f"state-{seed}.env"),
                    "launcher_dir": str(self.root / f"launcher-{seed}"),
                    "qualification_path": str(self.root / f"qualified-{seed}.json"),
                    "training_argv": argv,
                }
            )
        return {
            "confirmation_id": "utility_prior_transfer_v1",
            "created_utc": "2026-09-30T10:00:00Z",
            "repository": {
                "root": str(self.root),
                "commit": "a" * 40,
                "clean": True,
            },
            "source_record": {
                "path": str(self.source_record),
                "sha256": _sha(self.source_record),
            },
            "snapshot_record": {
                "path": str(self.snapshot_record),
                "sha256": _sha(self.snapshot_record),
                "snapshot_sha256": "2" * 64,
            },
            "gt_mesh": {
                "path": str(self.root / "mesh_aligned_0.05.ply"),
                "bytes": 123456,
                "sha256": "3" * 64,
                "content_accessed": False,
            },
            "runs": runs,
            "probe_targets": {
                "output_dir": str(self.root / "probe-output"),
                "staging_dir": str(self.root / "probe-staging"),
                "access_log_path": str(self.root / "first-gt-access.json"),
            },
        }

    def test_builds_exact_frozen_record_before_any_target(self):
        record = build_prior_transfer_confirmation(**self.request())
        self.assertEqual(record["protocol"]["models"], {
            "M0": ["A", "1-S"], "M1": ["A", "1-S", "1-r_p"]
        })
        self.assertEqual(record["protocol"]["training_seeds"], [0, 1, 2])
        self.assertEqual(record["protocol"]["evidence_version"], 4)
        self.assertEqual(record["protocol"]["performance_iteration"], 7000)
        self.assertEqual(record["mesh_admission"]["camera_ray_grid"], [8, 6])
        self.assertEqual(record["mesh_admission"]["sparse_point_limit"], 50000)
        self.assertEqual(record["accepted_geometry_releases"], [
            "STAGE_G_C_CANDIDATE", "NO_ACTION_SPECIFIC_SIGNAL"
        ])
        self.assertEqual(record["gt_mesh"]["content_accessed"], False)
        validate_preregistration_targets(record)

    def test_write_load_is_canonical_and_later_binding_is_exact(self):
        record = build_prior_transfer_confirmation(**self.request())
        output = self.root / "confirmation.json"
        published = write_prior_transfer_confirmation(record, output)
        loaded = load_prior_transfer_confirmation(output, published["sha256"])
        self.assertEqual(loaded, record)
        before = output.read_bytes()
        run_records = []
        for row in record["runs"]:
            run_records.append({
                "seed": row["seed"],
                "run_dir": row["run_dir"],
                "view_dir": row["view_dir"],
                "repository_commit": "a" * 40,
                "snapshot_sha256": "2" * 64,
                "evidence_version": 4,
                "qualification_path": row["qualification_path"],
            })
        validate_completed_run_binding(loaded, run_records)
        self.assertEqual(output.read_bytes(), before)
        changed = copy.deepcopy(run_records)
        changed[1]["snapshot_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "snapshot"):
            validate_completed_run_binding(loaded, changed)
        with self.assertRaises(FileExistsError):
            write_prior_transfer_confirmation(record, output)

    def test_rejects_dirty_wrong_inventory_gt_argv_or_existing_targets(self):
        cases = []
        dirty = self.request()
        dirty["repository"]["clean"] = False
        cases.append(dirty)
        duplicate = self.request()
        duplicate["runs"][2]["seed"] = 1
        cases.append(duplicate)
        gt_argv = self.request()
        gt_argv["runs"][0]["training_argv"].extend(["--gt_mesh", "/secret.ply"])
        cases.append(gt_argv)
        bad_snapshot = self.request()
        bad_snapshot["snapshot_record"]["sha256"] = "f" * 64
        cases.append(bad_snapshot)
        for request in cases:
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    build_prior_transfer_confirmation(**request)

        existing = self.request()
        Path(existing["runs"][0]["run_dir"]).mkdir()
        with self.assertRaisesRegex(ValueError, "target already exists"):
            build_prior_transfer_confirmation(**existing)


if __name__ == "__main__":
    unittest.main()
