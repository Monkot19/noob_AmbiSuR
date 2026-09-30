import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from reliability.utility_snapshot import (
    UtilitySourceAudit,
    audit_utility_source,
    source_manifest,
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


if __name__ == "__main__":
    unittest.main()
