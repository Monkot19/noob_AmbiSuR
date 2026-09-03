import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from reliability.config import CoreConfig
from reliability.runtime import (
    build_resolved_config,
    collect_run_identity,
    write_run_metadata,
)


class RunMetadataTests(unittest.TestCase):
    def test_resolved_config_separates_argument_groups_and_core(self):
        dataset = SimpleNamespace(source_path=Path("data/scene"), sh_degree=3)
        optimization = SimpleNamespace(iterations=500, seed=0)
        pipeline = SimpleNamespace(debug=False)
        core = CoreConfig()

        resolved = build_resolved_config(dataset, optimization, pipeline, core)

        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(
            resolved["model"],
            {"sh_degree": 3, "source_path": str(Path("data/scene"))},
        )
        self.assertEqual(
            resolved["optimization"], {"iterations": 500, "seed": 0}
        )
        self.assertEqual(resolved["pipeline"], {"debug": False})
        self.assertEqual(resolved["core"]["seed"], 0)
        self.assertEqual(resolved["core"]["enabled_features"], [])
        self.assertEqual(resolved["training_path"], "legacy")

    def test_metadata_writer_creates_only_named_json_files(self):
        resolved = {"schema_version": 1, "training_path": "legacy"}
        identity = {"git_commit": "abc", "git_dirty": False}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_run_metadata(output, resolved, identity)

            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["resolved_config.json", "run_identity.json"],
            )
            self.assertEqual(
                json.loads((output / "resolved_config.json").read_text()), resolved
            )
            self.assertEqual(
                json.loads((output / "run_identity.json").read_text()), identity
            )

    @unittest.skipUnless(shutil.which("git"), "git executable is required")
    def test_run_identity_reports_commit_and_dirty_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=E0 Test",
                    "-c",
                    "user.email=e0@example.invalid",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=repository,
                check=True,
            )

            clean_identity = collect_run_identity(repository)
            tracked.write_text("changed\n", encoding="utf-8")
            dirty_identity = collect_run_identity(repository)

        self.assertEqual(len(clean_identity["git_commit"]), 40)
        self.assertFalse(clean_identity["git_dirty"])
        self.assertTrue(dirty_identity["git_dirty"])


if __name__ == "__main__":
    unittest.main()
