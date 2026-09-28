import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from reliability.g1_confirmation import (
    build_confirmation_record,
    load_confirmation_record,
    validate_formal_admission,
    write_confirmation_record,
)


FORMULA_COMMIT = "a" * 40
DATASET_SHA = "b" * 64
PRIOR_SHA = "c" * 64
GT_SHA = "d" * 64
def training_argv(root):
    root = Path(root).resolve()
    return [
        "/root/miniconda3/envs/ambisur/bin/python",
        "train.py",
        "--source_path",
        str(root / "view"),
        "--model_path",
        str(root / "run"),
        "-r",
        "2",
        "--iterations",
        "7000",
        "--seed",
        "0",
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


def resolved_config():
    return {
        "schema_version": 1,
        "training_path": "core",
        "model": {"resolution": 2},
        "optimization": {"iterations": 7000, "seed": 0},
        "core": {
            "seed": 0,
            "core_shadow_mode": True,
            "d0_refresh_interval": 1000,
            "enabled_features": ["shadow_diagnostics"],
        },
    }


def confirmation_kwargs(root, confirmation_id="soft-calibration-formal-v1"):
    root = Path(root)
    return {
        "confirmation_id": confirmation_id,
        "preflight_utc": "2026-09-28T08:00:00Z",
        "formula_commit": FORMULA_COMMIT,
        "training_argv": training_argv(root),
        "scene": "Tool_Room",
        "seed": 0,
        "resolution": 2,
        "iterations": 7000,
        "refresh_interval": 1000,
        "evaluation_iterations": tuple(range(1000, 7001, 1000)),
        "checkpoint_iterations": (3000, 7000),
        "save_iterations": (7000,),
        "g1_iterations": (3000, 7000),
        "g1_decision_iteration": 7000,
        "dataset_sha256": DATASET_SHA,
        "prior_sha256": PRIOR_SHA,
        "gt_sha256": GT_SHA,
        "run_dir": root / "run",
        "view_dir": root / "view",
        "report_path": root / "formal-output" / "report.json",
        "output_dir": root / "formal-output",
        "archive_path": root / "formal-output.tar.gz",
        "expected_resolved_config": resolved_config(),
        "auroc_threshold": 0.60,
        "gain_threshold": 0.03,
    }


def materialize_admission_run(run, argv):
    run = Path(run)
    run.mkdir()
    (run / "run_identity.json").write_text(
        json.dumps(
            {
                "git_commit": FORMULA_COMMIT,
                "git_dirty": False,
                "argv": argv,
                "seed": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "resolved_config.json").write_text(
        json.dumps(resolved_config(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, value in (
        ("dataset_manifest_before.sha256", DATASET_SHA),
        ("dataset_manifest_after.sha256", DATASET_SHA),
        ("aligned_prior_before.sha256", PRIOR_SHA),
        ("aligned_prior_after.sha256", PRIOR_SHA),
    ):
        (run / name).write_text(value + "\n", encoding="utf-8")
    return run


class G1ConfirmationTests(unittest.TestCase):
    def test_builds_the_frozen_soft_calibration_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = confirmation_kwargs(root)
            record = build_confirmation_record(**kwargs)

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["confirmation_id"], "soft-calibration-formal-v1")
        self.assertEqual(record["formula_commit"], FORMULA_COMMIT)
        self.assertEqual(
            record["formula"],
            {
                "count": "M/(M+K_c)",
                "angle": "D/(D+D_c)",
                "sufficiency": "sqrt(S_count*S_angle)",
                "need": "1-S*(1-A)",
                "k_c": 5.0,
                "theta_c_degrees": 30.0,
                "d_c": (1.0 - math.cos(math.radians(30.0))) / 2.0,
            },
        )
        self.assertEqual(record["training"]["argv"], kwargs["training_argv"])
        self.assertEqual(
            record["training"]["evaluation_iterations"],
            list(range(1000, 7001, 1000)),
        )
        self.assertEqual(record["training"]["checkpoint_iterations"], [3000, 7000])
        self.assertEqual(record["evaluation"]["iterations"], [3000, 7000])
        self.assertEqual(record["evaluation"]["decision_iteration"], 7000)
        self.assertEqual(record["evaluation"]["gt_mesh_sha256"], GT_SHA)
        self.assertEqual(
            record["g1_gate"],
            {
                "auroc_n_strictly_greater_than": 0.60,
                "gain_over_best_component_at_least": 0.03,
            },
        )
        self.assertNotIn("gt", " ".join(record["training"]["argv"]).lower())
        self.assertNotIn(
            "gt", json.dumps(record["training"]["resolved_config"]).lower()
        )
        self.assertTrue(all(item["absent"] for item in record["targets"].values()))

    def test_canonical_write_is_deterministic_and_has_detached_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = build_confirmation_record(**confirmation_kwargs(root))
            record_path = root / "confirmation.json"

            actual_path, sidecar, digest = write_confirmation_record(
                record, record_path
            )

            expected_bytes = (
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            self.assertEqual(actual_path.read_bytes(), expected_bytes)
            self.assertEqual(digest, hashlib.sha256(expected_bytes).hexdigest())
            self.assertEqual(sidecar, Path(f"{record_path}.sha256"))
            self.assertEqual(sidecar.read_text(encoding="utf-8").split()[0], digest)
            self.assertEqual(load_confirmation_record(record_path, digest), record)

    def test_write_rejects_overwrite_and_detects_later_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = build_confirmation_record(**confirmation_kwargs(root))
            path = root / "confirmation.json"
            _, sidecar, digest = write_confirmation_record(record, path)

            with self.assertRaises(FileExistsError):
                write_confirmation_record(record, path)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "confirmation SHA256 mismatch"):
                load_confirmation_record(path, digest)
            self.assertTrue(sidecar.is_file())

    def test_write_rechecks_that_every_frozen_target_is_still_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = confirmation_kwargs(root)
            record = build_confirmation_record(**kwargs)
            Path(kwargs["run_dir"]).mkdir(parents=True)
            record_path = root / "confirmation.json"

            with self.assertRaisesRegex(FileExistsError, "target"):
                write_confirmation_record(record, record_path)

            self.assertFalse(record_path.exists())
            self.assertFalse(Path(f"{record_path}.sha256").exists())

    def test_build_rejects_argv_or_config_outside_the_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for flag, wrong in (
                ("--seed", "1"),
                ("-r", "4"),
                ("--iterations", "6999"),
                ("--d0_refresh_interval", "500"),
            ):
                with self.subTest(flag=flag):
                    kwargs = confirmation_kwargs(root / flag.lstrip("-"))
                    argv = list(kwargs["training_argv"])
                    argv[argv.index(flag) + 1] = wrong
                    kwargs["training_argv"] = argv
                    with self.assertRaisesRegex(ValueError, "training argv"):
                        build_confirmation_record(**kwargs)

            kwargs = confirmation_kwargs(root / "mode")
            kwargs["training_argv"] = [
                token
                for token in kwargs["training_argv"]
                if token != "--core_shadow_mode"
            ]
            with self.assertRaisesRegex(ValueError, "training argv"):
                build_confirmation_record(**kwargs)

            kwargs = confirmation_kwargs(root / "scene")
            kwargs["scene"] = "Utility_Room"
            with self.assertRaisesRegex(ValueError, "training scene"):
                build_confirmation_record(**kwargs)

            for keys, wrong in (
                (("model", "resolution"), 4),
                (("optimization", "iterations"), 6999),
                (("optimization", "seed"), 1),
                (("core", "seed"), 1),
                (("core", "d0_refresh_interval"), 500),
            ):
                with self.subTest(config_keys=keys):
                    kwargs = confirmation_kwargs(root / "-".join(keys))
                    config = copy.deepcopy(kwargs["expected_resolved_config"])
                    config[keys[0]][keys[1]] = wrong
                    kwargs["expected_resolved_config"] = config
                    with self.assertRaisesRegex(
                        ValueError, "resolved config"
                    ):
                        build_confirmation_record(**kwargs)

    def test_rejects_unsafe_ids_existing_targets_and_malformed_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "confirmation ID"):
                build_confirmation_record(
                    **confirmation_kwargs(root, confirmation_id="../unsafe")
                )

            for target_name in (
                "run_dir",
                "view_dir",
                "report_path",
                "output_dir",
                "archive_path",
            ):
                with self.subTest(target=target_name):
                    kwargs = confirmation_kwargs(root / target_name)
                    target = Path(kwargs[target_name])
                    if target.suffix:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(b"exists")
                    else:
                        target.mkdir(parents=True)
                    with self.assertRaisesRegex(FileExistsError, "target"):
                        build_confirmation_record(**kwargs)

            clean_root = root / "clean"
            record = build_confirmation_record(**confirmation_kwargs(clean_root))
            path, _sidecar, _digest = write_confirmation_record(
                record, clean_root / "confirmation.json"
            )
            with self.assertRaisesRegex(ValueError, "expected confirmation SHA256"):
                load_confirmation_record(path, "not-a-sha")

    def test_write_rejects_missing_fields_changed_constants_or_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = build_confirmation_record(**confirmation_kwargs(root))
            mutations = []
            missing = copy.deepcopy(record)
            missing.pop("formula_commit")
            mutations.append(missing)
            changed_constant = copy.deepcopy(record)
            changed_constant["formula"]["k_c"] = 6.0
            mutations.append(changed_constant)
            changed_gate = copy.deepcopy(record)
            changed_gate["g1_gate"]["auroc_n_strictly_greater_than"] = 0.59
            mutations.append(changed_gate)

            for index, mutated in enumerate(mutations):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        write_confirmation_record(
                            mutated, root / f"invalid-{index}.json"
                        )

    def test_formal_admission_matches_run_identity_config_and_hash_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = confirmation_kwargs(root)
            record = build_confirmation_record(**kwargs)
            run = materialize_admission_run(
                kwargs["run_dir"], kwargs["training_argv"]
            )

            validate_formal_admission(
                record,
                run_dir=run,
                confirmation_id=record["confirmation_id"],
                evaluator_commit=FORMULA_COMMIT,
                dataset_sha256=DATASET_SHA,
                prior_sha256=PRIOR_SHA,
                gt_sha256=GT_SHA,
                report_path=kwargs["report_path"],
                output_dir=kwargs["output_dir"],
                archive_path=kwargs["archive_path"],
            )

            for keyword, wrong in (
                ("confirmation_id", "wrong-id"),
                ("evaluator_commit", "0" * 40),
                ("dataset_sha256", "0" * 64),
                ("prior_sha256", "0" * 64),
                ("gt_sha256", "0" * 64),
            ):
                arguments = {
                    "run_dir": run,
                    "confirmation_id": record["confirmation_id"],
                    "evaluator_commit": FORMULA_COMMIT,
                    "dataset_sha256": DATASET_SHA,
                    "prior_sha256": PRIOR_SHA,
                    "gt_sha256": GT_SHA,
                    "report_path": kwargs["report_path"],
                    "output_dir": kwargs["output_dir"],
                    "archive_path": kwargs["archive_path"],
                }
                arguments[keyword] = wrong
                with self.subTest(keyword=keyword):
                    with self.assertRaises(ValueError):
                        validate_formal_admission(record, **arguments)

            for keyword in ("report_path", "output_dir", "archive_path"):
                arguments = {
                    "run_dir": run,
                    "confirmation_id": record["confirmation_id"],
                    "evaluator_commit": FORMULA_COMMIT,
                    "dataset_sha256": DATASET_SHA,
                    "prior_sha256": PRIOR_SHA,
                    "gt_sha256": GT_SHA,
                    "report_path": kwargs["report_path"],
                    "output_dir": kwargs["output_dir"],
                    "archive_path": kwargs["archive_path"],
                }
                arguments[keyword] = root / f"wrong-{keyword}"
                with self.subTest(keyword=keyword):
                    with self.assertRaisesRegex(ValueError, "target"):
                        validate_formal_admission(record, **arguments)

    def test_formal_admission_rejects_run_identity_config_or_hash_mutation(self):
        mutations = (
            ("dirty", "run_identity.json", ("git_dirty",), True),
            (
                "argv",
                "run_identity.json",
                ("argv",),
                training_argv(Path("/changed")) + ["--changed"],
            ),
            ("identity_seed", "run_identity.json", ("seed",), 1),
            ("identity_commit", "run_identity.json", ("git_commit",), "0" * 40),
            ("resolution", "resolved_config.json", ("model", "resolution"), 4),
            (
                "iterations",
                "resolved_config.json",
                ("optimization", "iterations"),
                6999,
            ),
            ("config_seed", "resolved_config.json", ("optimization", "seed"), 1),
            ("core_seed", "resolved_config.json", ("core", "seed"), 1),
            ("core_mode", "resolved_config.json", ("core", "core_shadow_mode"), False),
            ("refresh", "resolved_config.json", ("core", "d0_refresh_interval"), 500),
            ("features", "resolved_config.json", ("core", "enabled_features"), []),
            ("dataset_before", "dataset_manifest_before.sha256", (), "0" * 64),
            ("dataset_after", "dataset_manifest_after.sha256", (), "0" * 64),
            ("prior_before", "aligned_prior_before.sha256", (), "0" * 64),
            ("prior_after", "aligned_prior_after.sha256", (), "0" * 64),
        )
        for label, filename, keys, value in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    kwargs = confirmation_kwargs(root)
                    record = build_confirmation_record(**kwargs)
                    run = materialize_admission_run(
                        kwargs["run_dir"], kwargs["training_argv"]
                    )
                    path = run / filename
                    if keys:
                        document = json.loads(path.read_text(encoding="utf-8"))
                        target = document
                        for key in keys[:-1]:
                            target = target[key]
                        target[keys[-1]] = value
                        path.write_text(
                            json.dumps(document, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    else:
                        path.write_text(value + "\n", encoding="utf-8")

                    with self.assertRaises(ValueError):
                        validate_formal_admission(
                            record,
                            run_dir=run,
                            confirmation_id=record["confirmation_id"],
                            evaluator_commit=FORMULA_COMMIT,
                            dataset_sha256=DATASET_SHA,
                            prior_sha256=PRIOR_SHA,
                            gt_sha256=GT_SHA,
                            report_path=kwargs["report_path"],
                            output_dir=kwargs["output_dir"],
                            archive_path=kwargs["archive_path"],
                        )


if __name__ == "__main__":
    unittest.main()
