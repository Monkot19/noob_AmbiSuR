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
    _snapshot_binding,
    input_fingerprints,
    validate_checkpoint_metadata,
    validate_checkpoint_lineage,
)
from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation, write_prior_transfer_confirmation,
)
from tests import test_g1_prior_transfer_confirmation as confirmation_tests
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
        original_log = (self.run / "train.log").read_text()
        for text in ("Training complete.\nTraining complete.\n", "Training complete.\nNaN\n",
                     "Training complete.\nTraceback\n", "Training complete.\n/secret/gt/mesh.ply\n"):
            (self.run / "train.log").write_text(original_log.replace("Training complete.\n", text))
            with self.assertRaises(ValueError):
                validate_completion(self.run, self.row, "f" * 64, processes=[])
        (self.run / "train.log").write_text(original_log)
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


class CheckpointContractTests(unittest.TestCase):
    def fixture(self):
        centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        evidence = {"previous_centers": centers.copy(), "history_valid": np.ones(3, dtype=bool),
            "stable_state": np.array([0, 0, 4], dtype=np.int8),
            "temporal_transition_diagnostics": {
                "previous_stable": np.array([0, 0, 4], dtype=np.int8),
                "stable_age_refreshes": np.array([1, 2, 3], dtype=np.int64),
                "stable_transition_count": np.array([2, 1, 0], dtype=np.int64)}}
        event = {"mean_stable_age_refreshes": 2.0, "mean_stable_transition_count": 1.0,
            "mean_stable_age_refreshes_by_state": {"Bypass": 1.5, "Consensus": None,
                "Prior-led": None, "Geometry-led": None, "Abstain": 3.0},
            "mean_stable_transition_count_by_state": {"Bypass": 1.5, "Consensus": None,
                "Prior-led": None, "Geometry-led": None, "Abstain": 0.0}}
        return centers, evidence, event

    def test_rejects_permuted_centers_and_missing_current_history(self):
        centers, evidence, event = self.fixture()
        validate_checkpoint_lineage(centers, evidence, event, 3)
        with self.assertRaisesRegex(ValueError, "center"):
            validate_checkpoint_lineage(centers[::-1], evidence, event, 3)
        evidence["history_valid"][0] = False
        with self.assertRaisesRegex(ValueError, "history"):
            validate_checkpoint_lineage(centers, evidence, event, 3)

    def test_counter_bounds_and_event_means_are_checked_not_just_round_trip(self):
        for name in ("stable_age_refreshes", "stable_transition_count"):
            for value in (100000, 0):
                centers, evidence, event = self.fixture()
                evidence["temporal_transition_diagnostics"][name][0] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_checkpoint_lineage(centers, evidence, event, 3)

    def test_exact_checkpoint_root_and_nested_gt_metadata_fail_closed(self):
        payload = {"schema_version": 1, "iteration": 3000, "gaussian_state": (), "core_state": {}}
        validate_checkpoint_metadata(payload)
        for key, value in (("gt_mesh", "/secret/gt/mesh.ply"), ("gt_distance", np.zeros(3))):
            changed = copy.deepcopy(payload)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_metadata(changed)
            changed = copy.deepcopy(payload)
            changed["core_state"][key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(changed)


def asset_metadata_fixture(root):
    """Real source/DA3 admission and completed launcher metadata (no checkpoints)."""
    source, snapshot, snapshot_path, da3 = _confirmation(root, expected_count=147)
    _copy_source_and_write_derived(source, snapshot, names=da3["source"]["image_names"])
    write_snapshot_record(audit_da3_snapshot(snapshot, da3), da3, snapshot_path)
    source_path = root / "source.json"
    write_json(source_path, {**da3["source"], "audit_kind": "utility_source", "gt_access": "NONE"})
    # Reuse the preregistration fixture, not a mock of the qualification behavior.
    maker = confirmation_tests.PriorTransferConfirmationTests()
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
    return confirmation, confirmation_path, published["sha256"], row


def completed_run_fixture(root):
    """Add real runtime refresh/migration and checkpoint I/O to the data fixture."""
    confirmation, confirmation_path, digest, row = asset_metadata_fixture(root)
    run = Path(row["run_dir"])
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
        inputs = refresh_inputs(centers=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        if iteration >= 4000:
            from dataclasses import fields, replace
            changed = {}
            for field in fields(inputs):
                value = getattr(inputs, field.name)
                if isinstance(value, torch.Tensor):
                    dim = 1 if field.name == "pixel_hits" else 0
                    if value.shape[dim] == 2:
                        changed[field.name] = torch.cat((value, value.narrow(dim, 0, 1)), dim=dim)
            inputs = replace(inputs, **changed)
        result = runtime.maybe_refresh(iteration, lambda: inputs)
        write_snapshot(run, iteration, result, transition_diagnostics=runtime.latest_transition_diagnostics)
        if iteration in (3000, 7000):
            params, optimizer = optimizer_fixture(n=runtime.accumulator.point_count,
                                                  step=2975 if iteration == 3000 else 6935)
            params = [torch.from_numpy(value) for value in params]
            params[0] = inputs.centers.clone()
            for state in optimizer["state"].values():
                for key, value in state.items():
                    state[key] = torch.from_numpy(value)
            capture = (0, *[params[index] for index in (0, 1, 2, 3, 5, 6, 4)],
                       *[torch.zeros(len(params[0])) for _ in range(6)], optimizer, 1.0)
            torch.save({"schema_version": 1, "iteration": iteration, "gaussian_state": capture,
                        "core_state": runtime.state_dict()}, run / f"chkpnt{iteration}.pth")
    return confirmation, confirmation_path, digest, row


class AssetMetadataTests(unittest.TestCase):
    def test_actual_loader_cache_is_allowed_only_in_private_view(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _, row = asset_metadata_fixture(Path(directory).resolve())
            cache = Path(row["view_dir"]) / "sparse_da3_aligned/0/points3D.ply"
            cache.write_bytes(b"private-loader-generated-point-cloud")
            _snapshot_binding(confirmation, row)
            snapshot_root = Path(json.loads(Path(confirmation["snapshot_record"]["path"]).read_text())["snapshot_root"])
            (snapshot_root / "sparse_da3_aligned/0/points3D.ply").write_bytes(b"changed-canonical")
            with self.assertRaises(ValueError):
                _snapshot_binding(confirmation, row)

    def test_added_split_or_shadowing_colmap_model_is_not_a_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _, row = asset_metadata_fixture(Path(directory).resolve())
            view = Path(row["view_dir"])
            for relative in ("split.json", "sparse/0/images.bin", "images/extra.jpg"):
                added = view / relative
                added.write_bytes(b"unexpected-loader-input")
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    _snapshot_binding(confirmation, row)
                added.unlink()

    def test_invalid_identity_config_and_manifest_fail_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            confirmation, _, _, row = asset_metadata_fixture(Path(directory).resolve())
            run = Path(row["run_dir"])
            with patch("reliability.prior_transfer_assets._processes", return_value=[]):
                for filename, field, wrong in (("run_identity.json", "git_commit", "b" * 40),
                                               ("resolved_config.json", "training_path", "legacy")):
                    path = run / filename
                    original = path.read_bytes()
                    value = json.loads(original)
                    value[field] = wrong
                    write_json(path, value)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        qualify_prior_transfer_run(run, 0, confirmation)
                    path.write_bytes(original)
                view_depth = Path(row["view_dir"]) / "estimated_depths/frame000.jpg.npy"
                view_depth.write_bytes(b"corrupt-depth")
                with self.assertRaisesRegex(ValueError, "immutable input changed"):
                    qualify_prior_transfer_run(run, 0, confirmation)


class RunIntegrationFixture(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("Torch checkpoint/runtime integration requires AutoDL")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.confirmation, self.path, self.digest, self.row = completed_run_fixture(self.root)
        self.run = Path(self.row["run_dir"])
        # Only the OS process enumeration is substituted; all data parsers are real.
        self.process = patch("reliability.prior_transfer_assets._processes", return_value=[])
        self.process.start()
        self.addCleanup(self.process.stop)


@unittest.skipIf(torch is None, "Torch checkpoint/runtime integration requires AutoDL")
class PriorTransferAssetIntegrationTests(RunIntegrationFixture):
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
        for defect in ("version", "temporal", "age", "transitions", "nonfinite", "row", "centers", "gt"):
            payload = torch.load(path, weights_only=False)
            evidence = payload["core_state"]["evidence"]
            if defect == "version":
                evidence["version"] = 3
            elif defect == "temporal":
                evidence["temporal_transition_diagnostics"]["previous_stable"][0] = 4
            elif defect == "nonfinite":
                payload["gaussian_state"][1][0, 0] = float("nan")
            elif defect == "age":
                evidence["temporal_transition_diagnostics"]["stable_age_refreshes"][0] = 100000
            elif defect == "transitions":
                evidence["temporal_transition_diagnostics"]["stable_transition_count"][0] += 1
            elif defect == "centers":
                center = payload["gaussian_state"][1]
                center[[0, 1]] = center[[1, 0]]
            elif defect == "gt":
                payload["gt_distance"] = torch.zeros(3)
            else:
                evidence["a_value"][0] += 0.1
            torch.save(payload, path)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                qualify_prior_transfer_run(self.run, 0, self.confirmation)
            path.write_bytes(original)
