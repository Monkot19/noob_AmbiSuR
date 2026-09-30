import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from reliability.utility_snapshot import (
    UtilitySourceAudit,
    audit_da3_snapshot,
    audit_utility_source,
    build_da3_confirmation,
    load_da3_confirmation,
    source_manifest,
    write_da3_confirmation,
    write_snapshot_record,
)


def _write_source(root: Path, names=("frame000.jpg", "frame001.JPG"), *, model="PINHOLE"):
    images = root / "images"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    for index, name in enumerate(names):
        Image.new("RGB", (8, 6), (index, 10, 20)).save(images / name)
    (sparse / "cameras.txt").write_text(
        f"1 {model} 8 6 4 4 4 3\n",
        encoding="utf-8",
    )
    image_lines = []
    for index, name in enumerate(names, start=1):
        image_lines.extend((f"{index} 1 0 0 0 {index - 1} 0 0 1 {name}", ""))
    (sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text(
        "1 0 0 1 10 20 30 0.1 1 0\n2 1 2 3 30 20 10 0.2 2 0\n",
        encoding="utf-8",
    )
    return root


def _confirmation(root: Path, *, expected_count=2):
    source = root / "source"
    names = tuple(f"frame{index:03d}.jpg" for index in range(expected_count))
    _write_source(source, names)
    manifest = source_manifest(audit_utility_source(source, expected_count))
    checkpoint = root / "da3.ckpt"
    checkpoint.write_bytes(b"frozen-da3")
    staging = root / "staging"
    snapshot_record = root / "records" / "snapshot.json"
    snapshot_record.parent.mkdir()
    command = [
        "bash",
        str((root / "repo/scripts/run_da3_single.sh").resolve()),
        str(staging.resolve()),
        "500000",
        "0.05",
    ]
    record = build_da3_confirmation(
        repository_root=(root / "repo").resolve(),
        repository_commit="a" * 40,
        repository_clean=True,
        source_manifest=manifest,
        da3_checkpoint={
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        environment={
            "python": "/opt/conda/bin/python",
            "python_version": "3.10.21",
            "torch_version": "2.7.1+cu128",
            "cuda_version": "12.8",
        },
        command=command,
        max_points=500000,
        ransac_thresh=0.05,
        staging_root=staging.resolve(),
        snapshot_record_path=snapshot_record.resolve(),
        created_utc="2026-09-30T00:00:00Z",
    )
    return source, staging, snapshot_record, record


def _copy_source_and_write_derived(source: Path, staging: Path, *, names):
    import shutil

    shutil.copytree(source, staging)
    depth = staging / "estimated_depths"
    conf = staging / "estimated_confs"
    raw = staging / "sparse_da3" / "0"
    aligned = staging / "sparse_da3_aligned" / "0"
    depth.mkdir()
    conf.mkdir()
    raw.mkdir(parents=True)
    aligned.mkdir(parents=True)
    for name in names:
        np.save(depth / f"{name}.npy", np.ones((6, 8), dtype=np.float32))
        np.save(conf / f"{name}.npy", np.full((6, 8), 0.5, dtype=np.float32))
        Image.new("L", (8, 6)).save(depth / f"{name}.jpg")
    for model in (raw, aligned):
        (model / "cameras.txt").write_text("1 PINHOLE 8 6 4 4 4 3\n", encoding="utf-8")
        lines = []
        for index, name in enumerate(names, start=1):
            lines.extend((f"{index} 1 0 0 0 0 0 0 1 {name}", ""))
        (model / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (model / "points3D.txt").write_text("1 0 0 1 1 2 3 0.1\n", encoding="utf-8")
    (aligned / "trans.json").write_text(json.dumps({
        "scale": 1.25,
        "T_matrix_scene2_from_scene1": [
            [1.25, 0, 0, 0],
            [0, 1.25, 0, 0],
            [0, 0, 1.25, 0],
            [0, 0, 0, 1],
        ],
    }), encoding="utf-8")


class UtilitySourceAuditTests(unittest.TestCase):
    def test_valid_source_has_exact_registration_and_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_source(Path(directory))
            audit = audit_utility_source(root, expected_count=2)
            self.assertIsInstance(audit, UtilitySourceAudit)
            self.assertEqual(audit.image_names, ("frame000.jpg", "frame001.JPG"))
            self.assertEqual(audit.camera_count, 1)
            self.assertEqual(audit.point_count, 2)
            first = source_manifest(audit)
            second = source_manifest(audit_utility_source(root, expected_count=2))
            self.assertEqual(first, second)
            self.assertEqual(first["schema_version"], 1)
            self.assertEqual(first["source_sha256"], hashlib.sha256(
                (json.dumps(first["files"], sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            ).hexdigest())
            self.assertEqual(
                [entry["path"] for entry in first["files"]],
                [
                    "images/frame000.jpg",
                    "images/frame001.JPG",
                    "sparse/0/cameras.txt",
                    "sparse/0/images.txt",
                    "sparse/0/points3D.txt",
                ],
            )

    def test_exact_147_name_inventory_is_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            names = tuple(f"DSC{index:05d}.JPG" for index in range(147))
            root = _write_source(Path(directory), names)
            audit = audit_utility_source(root)
            self.assertEqual(audit.image_count, 147)
            self.assertEqual(audit.registered_image_count, 147)

    def test_rejects_registration_case_extra_missing_and_duplicate_names(self):
        mutations = {
            "case": lambda p: p.write_text(p.read_text(encoding="utf-8").replace("frame000.jpg", "FRAME000.jpg"), encoding="utf-8"),
            "missing": lambda p: p.write_text(p.read_text(encoding="utf-8").replace("2 1 0 0 0 1 0 0 1 frame001.JPG\n\n", ""), encoding="utf-8"),
            "duplicate": lambda p: p.write_text(p.read_text(encoding="utf-8") + "3 1 0 0 0 0 0 0 1 frame000.jpg\n\n", encoding="utf-8"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = _write_source(Path(directory))
                mutate(root / "sparse" / "0" / "images.txt")
                with self.assertRaises(ValueError):
                    audit_utility_source(root, expected_count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = _write_source(Path(directory))
            Image.new("RGB", (8, 6)).save(root / "images" / "extra.jpg")
            with self.assertRaises(ValueError):
                audit_utility_source(root, expected_count=3)

    def test_rejects_bad_camera_model_reference_or_dimensions(self):
        for label, mutate in (
            ("fisheye", lambda root: (root / "sparse/0/cameras.txt").write_text("1 OPENCV_FISHEYE 8 6 1 1 1 1 0 0 0 0\n", encoding="utf-8")),
            ("camera", lambda root: (root / "sparse/0/images.txt").write_text((root / "sparse/0/images.txt").read_text(encoding="utf-8").replace(" 1 frame000.jpg", " 9 frame000.jpg"), encoding="utf-8")),
            ("dimensions", lambda root: Image.new("RGB", (9, 6)).save(root / "images/frame000.jpg")),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = _write_source(Path(directory))
                mutate(root)
                with self.assertRaises(ValueError):
                    audit_utility_source(root, expected_count=2)

    def test_rejects_nonfinite_or_non_normalized_pose_and_nonfinite_points(self):
        for label, replacement in (
            ("pose_nan", "1 nan 0 0 0 0 0 0 1 frame000.jpg"),
            ("pose_norm", "1 2 0 0 0 0 0 0 1 frame000.jpg"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = _write_source(Path(directory))
                path = root / "sparse/0/images.txt"
                lines = path.read_text(encoding="utf-8").splitlines()
                lines[0] = replacement
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_utility_source(root, expected_count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = _write_source(Path(directory))
            (root / "sparse/0/points3D.txt").write_text("1 inf 0 1 1 2 3 0.1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_utility_source(root, expected_count=2)

    def test_rejects_unreadable_image_duplicate_ids_and_existing_derived_tree(self):
        for label, mutate in (
            ("image", lambda root: (root / "images/frame000.jpg").write_bytes(b"not-an-image")),
            ("camera_id", lambda root: (root / "sparse/0/cameras.txt").write_text("1 PINHOLE 8 6 4 4 4 3\n1 PINHOLE 8 6 4 4 4 3\n", encoding="utf-8")),
            ("point_id", lambda root: (root / "sparse/0/points3D.txt").write_text("1 0 0 1 1 2 3 0.1\n1 0 0 2 1 2 3 0.1\n", encoding="utf-8")),
            ("derived", lambda root: (root / "estimated_depths").mkdir()),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = _write_source(Path(directory))
                mutate(root)
                with self.assertRaises((ValueError, OSError)):
                    audit_utility_source(root, expected_count=2)


class UtilityDa3SnapshotTests(unittest.TestCase):
    def test_confirmation_binds_explicit_clean_inputs_command_and_absent_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, staging, snapshot_record, record = _confirmation(root)
            self.assertEqual(record["repository"]["commit"], "a" * 40)
            self.assertTrue(record["repository"]["clean"])
            self.assertEqual(record["source"]["source_sha256"], source_manifest(audit_utility_source(source, 2))["source_sha256"])
            self.assertEqual(record["preprocessing"], {"max_points": 500000, "ransac_thresh": 0.05})
            self.assertEqual(record["command"][1].replace("\\", "/").split("/")[-3:], ["repo", "scripts", "run_da3_single.sh"])
            self.assertFalse(staging.exists())
            self.assertFalse(snapshot_record.exists())

    def test_confirmation_rejects_implicit_or_unsafe_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, staging, _snapshot_record, record = _confirmation(root)
            bad = dict(record)
            for label, mutate in (
                ("dirty", lambda value: value["repository"].update(clean=False)),
                ("sha", lambda value: value["da3_checkpoint"].update(sha256="bad")),
                ("max", lambda value: value["preprocessing"].update(max_points=None)),
                ("gt", lambda value: value.update(command=["bash", "/repo/scripts/run_da3_single.sh", "/gt/mesh.ply", "1", "0.1"])),
            ):
                with self.subTest(label=label):
                    candidate = json.loads(json.dumps(record))
                    mutate(candidate)
                    with self.assertRaises(ValueError):
                        write_da3_confirmation(candidate, root / f"{label}.json")
            staging.mkdir()
            with self.assertRaises(ValueError):
                write_da3_confirmation(record, root / "existing-target.json")

    def test_confirmation_write_load_is_canonical_immutable_and_detects_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _staging, _snapshot_record, record = _confirmation(root)
            path = root / "confirmation.json"
            publication = write_da3_confirmation(record, path)
            loaded = load_da3_confirmation(path, publication["sha256"])
            self.assertEqual(loaded, record)
            with self.assertRaises(FileExistsError):
                write_da3_confirmation(record, path)
            (source / "images/frame000.jpg").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_da3_confirmation(path, publication["sha256"], verify_source=True)

    def test_confirmation_rehashes_checkpoint_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, _staging, _snapshot_record, record = _confirmation(root)
            (root / "da3.ckpt").write_bytes(b"tamper-da3")
            with self.assertRaises(ValueError):
                write_da3_confirmation(record, root / "confirmation.json")

    def test_confirmation_rejects_missing_checkpoint_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, _staging, _snapshot_record, record = _confirmation(root)
            (root / "da3.ckpt").unlink()
            with self.assertRaises(FileNotFoundError):
                write_da3_confirmation(record, root / "confirmation.json")

    def test_confirmation_reload_can_reverify_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, _staging, _snapshot_record, record = _confirmation(root)
            path = root / "confirmation.json"
            publication = write_da3_confirmation(record, path)
            (root / "da3.ckpt").write_bytes(b"tamper-da3")
            with self.assertRaises(ValueError):
                load_da3_confirmation(
                    path,
                    publication["sha256"],
                    verify_checkpoint=True,
                )

    def test_snapshot_admits_exact_arrays_models_alignment_and_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, staging, snapshot_record, record = _confirmation(root)
            names = tuple(record["source"]["image_names"])
            _copy_source_and_write_derived(source, staging, names=names)
            audit = audit_da3_snapshot(staging, record)
            self.assertEqual(audit["array_count"], {"depth": 2, "confidence": 2})
            self.assertEqual(audit["alignment"]["scale"], 1.25)
            manifest_paths = [entry["path"] for entry in audit["derived_manifest"]]
            self.assertIn("estimated_depths/frame000.jpg.npy", manifest_paths)
            self.assertIn("estimated_depths/frame000.jpg.jpg", manifest_paths)
            self.assertIn("sparse_da3_aligned/0/trans.json", manifest_paths)
            written = write_snapshot_record(audit, record, snapshot_record)
            self.assertEqual(load_da3_confirmation(snapshot_record, written["sha256"], expected_kind="utility_da3_snapshot")["snapshot_sha256"], audit["snapshot_sha256"])
            with self.assertRaises(FileExistsError):
                write_snapshot_record(audit, record, snapshot_record)

    def test_snapshot_rejects_array_model_alignment_and_source_contract_violations(self):
        mutations = {
            "missing": lambda stage: (stage / "estimated_depths/frame000.jpg.npy").unlink(),
            "extra": lambda stage: np.save(stage / "estimated_confs/extra.jpg.npy", np.ones((6, 8))),
            "case": lambda stage: (stage / "estimated_depths/frame000.jpg.npy").rename(stage / "estimated_depths/FRAME000.jpg.npy"),
            "nonfinite": lambda stage: np.save(stage / "estimated_depths/frame000.jpg.npy", np.full((6, 8), np.nan)),
            "shape": lambda stage: np.save(stage / "estimated_confs/frame000.jpg.npy", np.ones((5, 8))),
            "model": lambda stage: (stage / "sparse_da3/0/cameras.txt").unlink(),
            "scale": lambda stage: (stage / "sparse_da3_aligned/0/trans.json").write_text(json.dumps({"scale": 0, "T_matrix_scene2_from_scene1": np.eye(4).tolist()}), encoding="utf-8"),
            "source": lambda stage: (stage / "images/frame000.jpg").write_bytes(b"changed"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, staging, _snapshot_record, record = _confirmation(root)
                _copy_source_and_write_derived(source, staging, names=tuple(record["source"]["image_names"]))
                mutate(staging)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    audit_da3_snapshot(staging, record)


if __name__ == "__main__":
    unittest.main()
