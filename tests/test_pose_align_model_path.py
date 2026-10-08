"""Real filesystem coverage for the preprocessing COLMAP lookup boundary."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "multi_view_priors"), *sys.path]):
    from multi_view_priors.pose_align import find_model_path, read_model


class PoseAlignModelPathTests(unittest.TestCase):
    def test_zero_text_model_loads_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            sparse = Path(temporary) / "sparse"
            model = sparse / "0"
            model.mkdir(parents=True)
            fixtures = {
                "cameras.txt": "1 PINHOLE 8 6 4 4 4 3\n",
                "images.txt": "1 1 0 0 0 0 0 0 1 frame.jpg\n\n",
                "points3D.txt": "1 0 0 1 10 20 30 0.1 1 0\n",
            }
            for name, content in fixtures.items():
                (model / name).write_text(content, encoding="utf-8")
            before = {p.name: p.read_bytes() for p in model.iterdir()}
            selected, extension = find_model_path(str(sparse))
            self.assertEqual((selected, extension), (str(model), ".txt"))
            cameras, images, points = read_model(selected, extension)
            self.assertEqual(set(cameras), {1})
            self.assertEqual(set(images), {1})
            self.assertEqual(images[1].name, "frame.jpg")
            self.assertEqual(set(points), {1})
            self.assertEqual(before, {p.name: p.read_bytes() for p in model.iterdir()})

    def test_existing_layouts_and_binary_precedence(self):
        for subdirectory, extension in (("0", ".bin"), ("1", ".txt"), ("", ".bin"), ("", ".txt")):
            with self.subTest(subdirectory=subdirectory, extension=extension), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                model = base / subdirectory
                model.mkdir(exist_ok=True)
                (model / ("images" + extension)).touch()
                if extension == ".bin":
                    (model / "images.txt").touch()
                self.assertEqual(find_model_path(str(base)), (str(model), extension))

    def test_zero_text_precedes_legacy_one_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ("0", "1"):
                (base / name).mkdir()
                (base / name / "images.txt").touch()
            self.assertEqual(find_model_path(str(base)), (str(base / "0"), ".txt"))

    def test_missing_model_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                find_model_path(temporary)
