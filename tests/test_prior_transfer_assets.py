"""Run qualification tests: corrupt evidence must never receive a PASS record."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reliability.prior_transfer_assets import (
    expected_training_config,
    qualify_prior_transfer_run,
    validate_completion,
    validate_optimizer_state,
)
from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation, write_prior_transfer_confirmation,
)
from tests.test_g1_prior_transfer_confirmation import PriorTransferConfirmationTests
from tests.test_utility_snapshot import _confirmation, _copy_source_and_write_derived
from reliability.utility_snapshot import audit_da3_snapshot, write_snapshot_record

try:
    import torch
except ImportError:
    torch = None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimizer_fixture(n=2, step=2975):
    shapes = [(n, 3), (n, 6), (n, 1, 3), (n, 15, 3), (n, 1), (n, 3), (n, 4)]
    names = ["xyz", "knn_f", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
    parameters = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    state = {index: {"step": np.array(step, dtype=np.float32),
                     "exp_avg": value.copy(), "exp_avg_sq": value.copy()}
             for index, value in enumerate(parameters) if index != 1}
    return parameters, {"state": state, "param_groups": [
        {"name": name, "params": [index]} for index, name in enumerate(names)
    ]}


def completion_fixture(run, row, digest):
    launch = Path(row["launcher_dir"])
    launch.mkdir(parents=True)
    values = {"exit_code.txt": "0", "start_utc.txt": "2026-10-08T03:00:00Z",
              "end_utc.txt": "2026-10-08T03:10:00Z", "wall_seconds.txt": "600",
              "gpu_peak_mib.txt": "9000", "launcher.pid": "99999999",
              "training.pid": "99999998", "confirmation_sha256.txt": digest}
    for name, value in values.items():
        (launch / name).write_text(value + "\n", encoding="utf-8")
    write_json(launch / "launch_record.json", {
        "schema_version": 1, "seed": row["seed"], "attempt": 1,
        "resumed": False, "replaces_completed_run": False,
        "run_dir": row["run_dir"], "view_dir": row["view_dir"],
        "state_file": row["state_file"], "launcher_dir": row["launcher_dir"],
        "training_argv": row["training_argv"], "confirmation_sha256": digest,
    })
    fields = {"RUN_DIR": row["run_dir"], "VIEW_DIR": row["view_dir"],
              "LAUNCH_DIR": row["launcher_dir"], "SEED": str(row["seed"]),
              "CONFIRMATION_SHA256": digest}
    Path(row["state_file"]).write_text("".join(
        f"{key}='{value}'\n" for key, value in fields.items()), encoding="utf-8")
    (run / "train.log").write_text("\n".join(
        f"[ITER {iteration}] Evaluating train: L1 0.1 PSNR 20.0"
        for iteration in range(1000, 7001, 1000)) + "\nTraining complete.\n", encoding="utf-8")
    (launch / "launcher.log").write_text("train_return_code=0\n", encoding="utf-8")


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run"
        self.run.mkdir()
        self.row = {"seed": 0, "run_dir": str(self.run),
                    "view_dir": str(self.root / "view"),
                    "state_file": str(self.root / "state.env"),
                    "launcher_dir": str(self.root / "launch"), "training_argv": ["python", "train.py"]}
        completion_fixture(self.run, self.row, "f" * 64)

    def test_completion_is_read_only_and_requires_one_exact_launch(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = validate_completion(self.run, self.row, "f" * 64, processes=[])
        self.assertEqual(result["wall_seconds"], 600)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        launch = Path(self.row["launcher_dir"]) / "launch_record.json"
        value = json.loads(launch.read_text())
        for key, wrong in (("attempt", 2), ("resumed", True), ("replaces_completed_run", True),
                           ("confirmation_sha256", "e" * 64), ("seed", 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(value)
                changed[key] = wrong
                write_json(launch, changed)
                with self.assertRaises(ValueError):
                    validate_completion(self.run, self.row, "f" * 64, processes=[])

    def test_bad_completion_logs_processes_state_and_sentinel_fail_closed(self):
        for text in ("Training complete.\nTraining complete.\n", "Training complete.\nNaN\n",
                     "Training complete.\nTraceback\n", "Training complete.\n/secret/gt/mesh.ply\n"):
            (self.run / "train.log").write_text(text)
            with self.assertRaises(ValueError):
                validate_completion(self.run, self.row, "f" * 64, processes=[])
        completion_fixture_reset = self.run / "train.log"
        completion_fixture_reset.write_text("Training complete.\n")
        with self.assertRaises(ValueError):
            validate_completion(self.run, self.row, "f" * 64,
                                processes=[(10, "S", "/env/python -B train.py --iterations 7000")])
        (self.run / ".training_active").touch()
        with self.assertRaises(ValueError):
            validate_completion(self.run, self.row, "f" * 64, processes=[])
        (self.run / ".training_active").unlink()
        Path(self.row["state_file"]).write_text("RUN_DIR=$(touch /tmp/not-executed)\n")
        with self.assertRaises(ValueError):
            validate_completion(self.run, self.row, "f" * 64, processes=[])


class OptimizerTests(unittest.TestCase):
    def test_schedule_uses_parameter_replacement_not_global_iteration(self):
        schedule = {"iterations": 7000, "densify_from_iter": 500,
                    "densify_until_iter": 15000, "densification_interval": 100}
        for iteration, want in ((3000, 2975), (7000, 6935)):
            with self.subTest(iteration=iteration):
                params, optimizer = optimizer_fixture(step=want)
                report = validate_optimizer_state(optimizer, params, schedule, iteration)
                self.assertEqual(report["expected_step"], want)
                self.assertEqual(report["dormant_groups"], ["knn_f"])
                optimizer["state"][0]["step"] = np.array(iteration - 1.0)
                with self.assertRaises(ValueError):
                    validate_optimizer_state(optimizer, params, schedule, iteration)

    def test_only_dormant_knn_may_lack_state_and_moments_must_match(self):
        schedule = {"iterations": 7000, "densify_from_iter": 500,
                    "densify_until_iter": 15000, "densification_interval": 100}
        for defect in ("missing_active", "partial_knn", "shape", "nonfinite", "duplicate"):
            params, optimizer = optimizer_fixture()
            if defect == "missing_active":
                optimizer["state"].pop(0)
            elif defect == "partial_knn":
                optimizer["state"][1] = {"step": np.array(2975.0)}
            elif defect == "shape":
                optimizer["state"][0]["exp_avg"] = np.zeros((1, 3))
            elif defect == "nonfinite":
                optimizer["state"][0]["exp_avg_sq"][0, 0] = np.nan
            else:
                optimizer["param_groups"][1]["params"] = [0]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_optimizer_state(optimizer, params, schedule, 3000)


def completed_run_fixture(root):
    """Real source admission, DA3 manifest, runtime refresh/migration and checkpoint I/O."""
    source, snapshot, snapshot_path, da3 = _confirmation(root, expected_count=147)
    _copy_source_and_write_derived(source, snapshot, names=da3["source"]["image_names"])
    write_snapshot_record(audit_da3_snapshot(snapshot, da3), da3, snapshot_path)
    source_path = root / "source.json"
    write_json(source_path, {**da3["source"], "audit_kind": "utility_source", "gt_access": "NONE"})
    # Reuse the preregistration fixture, not a mock of the qualification behavior.
    maker = PriorTransferConfirmationTests()
    maker.root, maker.source_record, maker.snapshot_record = root, source_path, snapshot_path
    request = maker.request()
    request["snapshot_record"]["snapshot_sha256"] = json.loads(snapshot_path.read_text())["snapshot_sha256"]
    repo = root / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-qm", "fixture"], cwd=repo, check=True)
    request["repository"] = {"root": str(repo), "clean": True,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    confirmation = build_prior_transfer_confirmation(**request)
    confirmation_path = root / "confirmation.json"
    published = write_prior_transfer_confirmation(confirmation, confirmation_path)
    row = confirmation["runs"][0]
    run, view = Path(row["run_dir"]), Path(row["view_dir"])
    run.mkdir()
    shutil.copytree(snapshot, view)
    completion_fixture(run, row, published["sha256"])
    write_json(run / "resolved_config.json", expected_training_config(row))
    write_json(run / "run_identity.json", {"git_commit": request["repository"]["commit"],
        "git_dirty": False, "seed": 0, "argv": row["training_argv"], "cwd": str(repo)})
    from reliability.config import CoreConfig
    from reliability.shadow import D0ShadowRuntime
    from reliability.topology import TopologyChange
    from reliability.diagnostics import write_snapshot
    from tests.test_d0_shadow_runtime import refresh_inputs
    runtime = D0ShadowRuntime(2, cfg=CoreConfig(core_shadow_mode=True), device="cpu", refresh_interval=1000)
    for iteration in range(1000, 7001, 1000):
        if iteration == 4000:
            runtime.on_topology_change(TopologyChange(
                torch.tensor([0, 1, 0]), torch.tensor([False, False, True])))
        inputs = refresh_inputs()
        if iteration >= 4000:
            from dataclasses import fields
            for field in fields(inputs):
                value = getattr(inputs, field.name)
                if isinstance(value, torch.Tensor):
                    dim = 1 if field.name == "pixel_hits" else 0
                    if value.shape[dim] == 2:
                        setattr(inputs, field.name, torch.cat((value, value.narrow(dim, 0, 1)), dim=dim))
        result = runtime.maybe_refresh(iteration, lambda: inputs)
        write_snapshot(run, iteration, result, transition_diagnostics=runtime.latest_transition_diagnostics)
        if iteration in (3000, 7000):
            params, optimizer = optimizer_fixture(n=runtime.accumulator.point_count,
                                                  step=2975 if iteration == 3000 else 6935)
            params = [torch.from_numpy(value) for value in params]
            for state in optimizer["state"].values():
                for key, value in state.items():
                    state[key] = torch.from_numpy(value)
            capture = (0, *params, *[torch.zeros(len(params[0])) for _ in range(6)], optimizer, 1.0)
            torch.save({"schema_version": 1, "iteration": iteration, "gaussian_state": capture,
                        "core_state": runtime.state_dict()}, run / f"chkpnt{iteration}.pth")
    return confirmation, confirmation_path, published["sha256"], row


@unittest.skipIf(torch is None, "Torch checkpoint/runtime integration requires AutoDL")
class PriorTransferAssetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.confirmation, self.path, self.digest, self.row = completed_run_fixture(self.root)
        self.run = Path(self.row["run_dir"])
        # Only the OS process enumeration is substituted; all data parsers are real.
        self.process = patch("reliability.prior_transfer_assets._processes", return_value=[])
        self.process.start()
        self.addCleanup(self.process.stop)

    def test_real_runtime_topology_checkpoint_join_without_opening_gt(self):
        before = {p: file_sha(p) for p in self.root.rglob("*") if p.is_file()}
        report = qualify_prior_transfer_run(self.run, 0, self.confirmation)
        self.assertEqual(report["outcome"], "QUALIFIED")
        self.assertEqual(report["point_counts"], [2, 2, 2, 3, 3, 3, 3])
        self.assertEqual(report["run_binding"]["evidence_version"], 4)
        self.assertEqual([item["optimizer"]["expected_step"] for item in report["checkpoints"]], [2975, 6935])
        self.assertEqual(report["gt_access"], "NONE")
        self.assertEqual(before, {p: file_sha(p) for p in before})
        # The preregistered GT path intentionally does not exist: no stat/hash/parse is needed.
        self.assertFalse(Path(self.confirmation["gt_mesh"]["path"]).exists())

    def test_rejects_wrong_seed_commit_config_or_mutated_snapshot(self):
        with self.assertRaises(ValueError):
            qualify_prior_transfer_run(self.run, 1, self.confirmation)
        for filename, field, wrong in (("run_identity.json", "git_commit", "b" * 40),
                                       ("resolved_config.json", "training_path", "legacy")):
            path = self.run / filename
            original = path.read_bytes()
            value = json.loads(original)
            value[field] = wrong
            write_json(path, value)
            with self.assertRaises(ValueError):
                qualify_prior_transfer_run(self.run, 0, self.confirmation)
            path.write_bytes(original)
        depth = Path(json.loads(Path(self.confirmation["snapshot_record"]["path"]).read_text())["snapshot_root"]) / "estimated_depths/frame000.jpg.npy"
        depth.write_bytes(b"mutated")
        with self.assertRaises(ValueError):
            qualify_prior_transfer_run(self.run, 0, self.confirmation)

    def test_rejects_old_evidence_bad_temporal_nonfinite_and_row_permutation(self):
        path = self.run / "chkpnt7000.pth"
        original = path.read_bytes()
        for defect in ("version", "temporal", "nonfinite", "row"):
            payload = torch.load(path, weights_only=False)
            evidence = payload["core_state"]["evidence"]
            if defect == "version":
                evidence["version"] = 3
            elif defect == "temporal":
                evidence["temporal_transition_diagnostics"]["previous_stable"][0] = 4
            elif defect == "nonfinite":
                payload["gaussian_state"][1][0, 0] = float("nan")
            else:
                evidence["a_value"][0] += 0.1
            torch.save(payload, path)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                qualify_prior_transfer_run(self.run, 0, self.confirmation)
            path.write_bytes(original)

