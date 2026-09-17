import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from reliability.g1_visualization import required_artifacts
from scripts.diagnostics.evaluate_d0_g1 import (
    assert_inputs_unchanged,
    build_manifest,
    fingerprint_inputs,
    publication_exit_code,
    publish_atomically,
    validate_archive,
    validate_provenance,
    validate_publication_request,
    write_deterministic_archive,
)


class G1PublicationTests(unittest.TestCase):
    @staticmethod
    def materialize_artifacts(root, report=None):
        root = Path(root)
        for index, name in enumerate(required_artifacts()):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "report.json":
                payload = json.dumps(report or {"g1_evaluable": True, "g1_pass": True})
            elif name == "inputs.json":
                payload = json.dumps({"schema_version": 1})
            else:
                payload = f"artifact-{index}"
            path.write_text(payload + "\n", encoding="utf-8")

    def test_atomic_publication_writes_directory_archive_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "immutable.bin"
            source.write_bytes(b"immutable")
            output_root = root / "output"

            result = publish_atomically(
                output_root,
                "confirmation-a",
                {"source": source},
                lambda staging: self.materialize_artifacts(staging),
            )

            self.assertTrue(result["output_dir"].is_dir())
            self.assertTrue(result["archive_path"].is_file())
            self.assertTrue(result["archive_sha256_path"].is_file())
            self.assertEqual(
                result["archive_sha256"],
                hashlib.sha256(result["archive_path"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["archive_sha256_path"].read_text().split()[0],
                result["archive_sha256"],
            )
            manifest = json.loads((result["output_dir"] / "manifest.json").read_text())
            validate_archive(result["archive_path"], manifest)

    def test_atomic_publication_cleans_staging_on_failure_or_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "immutable.bin"
            source.write_bytes(b"immutable")
            output_root = root / "output"

            def fail(_staging):
                raise RuntimeError("producer failed")

            with self.assertRaisesRegex(RuntimeError, "producer failed"):
                publish_atomically(
                    output_root, "failed", {"source": source}, fail
                )

            def mutate(staging):
                self.materialize_artifacts(staging)
                source.write_bytes(b"mutated")

            with self.assertRaisesRegex(RuntimeError, "input mutated"):
                publish_atomically(
                    output_root, "mutated", {"source": source}, mutate
                )
            self.assertFalse((output_root / "failed").exists())
            self.assertFalse((output_root / "mutated").exists())
            self.assertEqual(list(output_root.glob("*.tmp-*")), [])
            self.assertEqual(list(output_root.glob("*.tar.gz")), [])

    def test_publication_exit_code_preserves_formal_and_exploratory_roles(self):
        self.assertEqual(publication_exit_code({"g1_evaluable": True, "g1_pass": True}), 0)
        self.assertEqual(publication_exit_code({"g1_evaluable": True, "g1_pass": False}), 1)
        self.assertEqual(publication_exit_code({"g1_evaluable": False, "g1_pass": False}), 2)
        self.assertEqual(
            publication_exit_code(
                {"g1_evaluable": None, "g1_pass": None}, exploratory=True
            ),
            0,
        )

    def test_cli_help_exposes_every_frozen_argument(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "diagnostics"
            / "evaluate_d0_g1.py"
        )
        result = subprocess.run(
            [sys.executable, "-B", str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in (
            "--run-dir",
            "--source-root",
            "--gt-mesh",
            "--output-root",
            "--confirmation-id",
            "--iterations",
            "--expected-commit",
            "--expected-dataset-sha",
            "--expected-gt-sha",
            "--exploratory",
        ):
            self.assertIn(flag, result.stdout)

    def make_contract(self, root):
        root = Path(root)
        run = root / "run"
        source = root / "source"
        output = root / "output"
        evidence = run / "d0_evidence"
        evidence.mkdir(parents=True)
        source.mkdir()
        gt = root / "mesh.ply"
        gt.write_bytes(b"ply\n")
        for iteration in range(1000, 7001, 1000):
            (evidence / f"iteration_{iteration:06d}.npz").write_bytes(
                str(iteration).encode("ascii")
            )
        return run, source, gt, output

    def test_publication_request_rejects_overwrite_gt_leak_and_wrong_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, source, gt, output = self.make_contract(root)
            target = validate_publication_request(
                run, source, gt, output, "formal-a", (3000, 7000), exploratory=False
            )
            self.assertEqual(target, output / "formal-a")

            target.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                validate_publication_request(
                    run, source, gt, output, "formal-a", (3000, 7000), exploratory=False
                )
            target.rmdir()

            leaked_gt = run / "mesh.ply"
            leaked_gt.write_bytes(b"ply\n")
            with self.assertRaisesRegex(ValueError, "GT mesh must be outside"):
                validate_publication_request(
                    run, source, leaked_gt, output, "formal-b", (3000, 7000), exploratory=False
                )
            with self.assertRaisesRegex(ValueError, "formal iterations"):
                validate_publication_request(
                    run, source, gt, output, "formal-c", (500,), exploratory=False
                )
            exploratory_target = validate_publication_request(
                run, source, gt, output, "smoke-a", (500,), exploratory=True
            )
            self.assertEqual(exploratory_target, output / "smoke-a")

    def test_publication_request_requires_complete_formal_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, source, gt, output = self.make_contract(root)
            (run / "d0_evidence" / "iteration_004000.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing D0 timeline snapshot: 4000"):
                validate_publication_request(
                    run, source, gt, output, "formal", (3000, 7000), exploratory=False
                )

    def test_provenance_requires_exact_commit_dataset_and_gt_hashes(self):
        expected = {"commit": "a" * 40, "dataset_sha": "b" * 64, "gt_sha": "c" * 64}
        validate_provenance(**expected, expected_commit=expected["commit"],
                            expected_dataset_sha=expected["dataset_sha"],
                            expected_gt_sha=expected["gt_sha"])
        for key, message in (
            ("commit", "commit mismatch"),
            ("dataset_sha", "dataset SHA256 mismatch"),
            ("gt_sha", "GT SHA256 mismatch"),
        ):
            actual = dict(expected)
            actual[key] = "0" * len(actual[key])
            with self.assertRaisesRegex(ValueError, message):
                validate_provenance(
                    **actual,
                    expected_commit=expected["commit"],
                    expected_dataset_sha=expected["dataset_sha"],
                    expected_gt_sha=expected["gt_sha"],
                )

    def test_input_fingerprints_detect_post_read_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"before")
            before = fingerprint_inputs({"checkpoint": path})
            self.assertEqual(before["checkpoint"]["bytes"], 6)
            path.write_bytes(b"after")
            after = fingerprint_inputs({"checkpoint": path})
            with self.assertRaisesRegex(RuntimeError, "input mutated during evaluation"):
                assert_inputs_unchanged(before, after)

    def test_manifest_and_archive_are_exact_safe_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "published"
            root.mkdir()
            for index, name in enumerate(required_artifacts()):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"artifact-{index}".encode("ascii"))

            manifest = build_manifest(root, required_artifacts())
            entries = manifest["files"]
            self.assertEqual([entry["path"] for entry in entries], list(required_artifacts()))
            for entry in entries:
                payload = (root / entry["path"]).read_bytes()
                self.assertEqual(entry["bytes"], len(payload))
                self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            (root / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )

            archive_a = root.parent / "a.tar.gz"
            archive_b = root.parent / "b.tar.gz"
            write_deterministic_archive(root, archive_a, manifest)
            write_deterministic_archive(root, archive_b, manifest)
            self.assertEqual(
                hashlib.sha256(archive_a.read_bytes()).hexdigest(),
                hashlib.sha256(archive_b.read_bytes()).hexdigest(),
            )
            members = validate_archive(archive_a, manifest)
            self.assertEqual(
                members,
                tuple(sorted(("manifest.json",) + required_artifacts())),
            )

    def test_manifest_rejects_missing_artifact_and_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required artifact inventory mismatch"):
                build_manifest(root, required_artifacts())

            malicious = root / "malicious.tar.gz"
            with tarfile.open(malicious, "w:gz") as archive:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"escape"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                validate_archive(malicious, {"schema_version": 1, "files": []})


if __name__ == "__main__":
    unittest.main()
