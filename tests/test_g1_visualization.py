from pathlib import Path
import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from reliability.g1_visualization import (
    camera_quartile_indices,
    cast_gt_depth,
    required_artifacts,
    scalar_colors,
    select_static_cameras,
    state_colors,
    write_colored_ply,
    write_gt_overlay,
    write_metric_figures,
)
from reliability.offline_g1 import ValidatedMesh


HAS_OPEN3D = importlib.util.find_spec("open3d") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class G1VisualizationTests(unittest.TestCase):
    def test_required_artifacts_freeze_complete_relative_inventory(self):
        names = required_artifacts()

        self.assertEqual(names, tuple(sorted(names)))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(not Path(name).is_absolute() for name in names))
        self.assertTrue(all(".." not in Path(name).parts for name in names))
        self.assertIn("report.json", names)
        self.assertIn("inputs.json", names)
        for iteration in (3000, 7000):
            root = f"iteration_{iteration:06d}"
            for field in ("A", "S", "N", "T_p", "T_g", "K", "state", "gt_distance"):
                self.assertIn(f"{root}/fields/{field}.ply", names)
                for view in ("q25", "q50", "q75"):
                    self.assertIn(f"{root}/views/{field}_{view}.png", names)
            for view in ("q25", "q50", "q75"):
                self.assertIn(f"{root}/overlays/gt_{view}.png", names)
                self.assertIn(f"{root}/overlays/gt_{view}.json", names)
        for stem in ("primary_curves", "risk_coverage", "state_error"):
            for suffix in ("png", "svg", "pdf", "csv", "json"):
                self.assertIn(f"metrics/iteration_007000_{stem}.{suffix}", names)
        for stem in ("state_proportion", "state_transition", "joint_coverage"):
            for suffix in ("png", "svg", "pdf", "csv", "json"):
                self.assertIn(f"timeline/{stem}.{suffix}", names)

        self.assertEqual(len(names), 108)

    def test_exploratory_500_inventory_is_exact_and_has_no_formal_timeline(self):
        names = required_artifacts(iterations=(500,), exploratory=True)

        self.assertEqual(len(names), 55)
        self.assertEqual(names, tuple(sorted(names)))
        self.assertIn("iteration_000500/fields/N.ply", names)
        self.assertIn("iteration_000500/views/N_q50.png", names)
        self.assertIn("metrics/iteration_000500_primary_curves.png", names)
        self.assertFalse(any(name.startswith("timeline/") for name in names))
        self.assertFalse(any("003000" in name or "007000" in name for name in names))

        with self.assertRaisesRegex(ValueError, "exploratory iterations"):
            required_artifacts(iterations=(3000, 7000), exploratory=True)
        with self.assertRaisesRegex(ValueError, "formal iterations"):
            required_artifacts(iterations=(500,), exploratory=False)

    def test_camera_indices_are_frozen_for_406_views(self):
        self.assertEqual(camera_quartile_indices(406), (101, 202, 303))
        with self.assertRaisesRegex(ValueError, "at least four"):
            camera_quartile_indices(3)

    def test_static_cameras_are_selected_after_colmap_id_sort(self):
        cameras = [
            SimpleNamespace(colmap_id=value, image_name=f"cam-{value}")
            for value in (8, 2, 7, 1, 6, 3, 5, 4)
        ]

        selected = select_static_cameras(cameras)

        self.assertEqual([camera.colmap_id for camera in selected], [2, 4, 6])

    def test_gt_colors_saturate_above_ten_centimeters(self):
        colors = scalar_colors(
            np.array([0.0, 0.05, 0.10, 1.0], dtype=np.float64),
            "gt_distance",
        )

        self.assertEqual(colors.dtype, np.uint8)
        self.assertEqual(colors.shape, (4, 3))
        np.testing.assert_array_equal(colors[2], colors[3])
        self.assertFalse(np.array_equal(colors[0], colors[1]))

    def test_unit_interval_fields_clip_without_data_dependent_rescaling(self):
        colors = scalar_colors(
            np.array([-2.0, 0.0, 0.5, 1.0, 9.0], dtype=np.float64),
            "N",
        )

        np.testing.assert_array_equal(colors[0], colors[1])
        np.testing.assert_array_equal(colors[3], colors[4])
        self.assertFalse(np.array_equal(colors[1], colors[2]))

    def test_state_colors_use_the_frozen_five_state_palette(self):
        colors = state_colors(np.arange(5, dtype=np.int8))

        np.testing.assert_array_equal(
            colors,
            np.array(
                [
                    [127, 127, 127],
                    [44, 160, 44],
                    [31, 119, 180],
                    [255, 127, 14],
                    [214, 39, 40],
                ],
                dtype=np.uint8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "state"):
            state_colors(np.array([5], dtype=np.int8))

    def test_colored_ply_keeps_every_finite_center_and_color(self):
        centers = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-1.0, -2.0, -3.0]],
            dtype=np.float64,
        )
        colors = np.array(
            [[1, 2, 3], [40, 50, 60], [253, 254, 255]], dtype=np.uint8
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "all-finite-gaussians.ply"
            metadata = write_colored_ply(path, centers, colors)
            text = path.read_text(encoding="ascii")

        self.assertEqual(metadata["vertex_count"], 3)
        self.assertIn("element vertex 3", text)
        self.assertIn("0 1 2 1 2 3", text)
        self.assertIn("3 4 5 40 50 60", text)
        self.assertIn("-1 -2 -3 253 254 255", text)

    def test_colored_ply_rejects_nonfinite_or_misaligned_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.ply"
            with self.assertRaisesRegex(ValueError, "finite"):
                write_colored_ply(
                    path,
                    np.array([[np.nan, 0.0, 0.0]]),
                    np.array([[1, 2, 3]], dtype=np.uint8),
                )
            with self.assertRaisesRegex(ValueError, "row count"):
                write_colored_ply(
                    path,
                    np.zeros((2, 3), dtype=np.float64),
                    np.zeros((1, 3), dtype=np.uint8),
                )

    @unittest.skipUnless(HAS_OPEN3D, "Open3D is required")
    def test_gt_depth_casts_full_frame_pixel_center_rays(self):
        camera = self.full_frame_camera()
        mesh = self.front_plane_mesh()

        depth, metadata = cast_gt_depth(camera, mesh)

        self.assertEqual(depth.shape, (4, 4))
        np.testing.assert_allclose(depth, 2.0, rtol=0, atol=1e-5)
        self.assertEqual(metadata["camera_image_name"], "frozen-camera")
        self.assertEqual(metadata["camera_colmap_id"], 17)
        self.assertEqual(metadata["image_shape"], [4, 4])
        self.assertEqual(metadata["hit_fraction"], 1.0)
        self.assertAlmostEqual(metadata["depth_min_m"], 2.0, places=5)
        self.assertAlmostEqual(metadata["depth_max_m"], 2.0, places=5)

    @unittest.skipUnless(HAS_OPEN3D, "Open3D is required")
    def test_gt_overlay_uses_frozen_camera_without_crop(self):
        camera = self.full_frame_camera()
        gaussian_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        gaussian_rgb[..., 1] = 64

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "gt-overlay.png"
            metadata = write_gt_overlay(
                camera,
                self.front_plane_mesh(),
                gaussian_rgb,
                path,
            )
            from PIL import Image

            with Image.open(path) as image:
                saved_shape = [image.height, image.width, len(image.getbands())]
            sidecar = json.loads(path.with_suffix(".json").read_text())

        self.assertEqual(saved_shape, [4, 4, 3])
        self.assertEqual(metadata["image_shape"], [4, 4, 3])
        self.assertEqual(metadata["camera_image_name"], "frozen-camera")
        self.assertEqual(sidecar, metadata)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required")
    def test_metric_figures_export_png_svg_pdf_and_source_data(self):
        report = self.metric_report()

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_metric_figures(
                report,
                Path(temporary_directory),
                prefix="iteration_007000",
            )
            names = sorted(path.name for path in paths)
            sizes = {path.name: path.stat().st_size for path in paths}

        expected_stems = (
            "iteration_007000_primary_curves",
            "iteration_007000_risk_coverage",
            "iteration_007000_state_error",
        )
        expected_names = sorted(
            f"{stem}.{suffix}"
            for stem in expected_stems
            for suffix in ("csv", "json", "pdf", "png", "svg")
        )
        self.assertEqual(names, expected_names)
        self.assertTrue(all(size > 0 for size in sizes.values()))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required")
    def test_metric_figures_still_publish_when_exploratory_curves_are_unavailable(self):
        report = self.metric_report()
        report["iteration"] = 500
        report["primary"]["curves"] = {
            "N": None,
            "A": None,
            "one_minus_S": None,
        }
        report["risk_coverage"] = {"N": None, "r_p": None, "r_g": None}

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_metric_figures(
                report,
                Path(temporary_directory),
                prefix="iteration_000500",
            )
            metadata = json.loads(
                (Path(temporary_directory) / "iteration_000500_primary_curves.json")
                .read_text(encoding="utf-8")
            )

        self.assertEqual(len(paths), 15)
        self.assertFalse(metadata["curves_available"])

    @staticmethod
    def full_frame_camera():
        intrinsic = np.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        camera_to_world = np.eye(4, dtype=np.float32)
        return SimpleNamespace(
            image_width=4,
            image_height=4,
            image_name="frozen-camera",
            colmap_id=17,
            get_calib_matrix_nerf=lambda scale=1.0: (
                intrinsic,
                camera_to_world,
            ),
        )

    @staticmethod
    def front_plane_mesh():
        return ValidatedMesh(
            vertices=np.array(
                [
                    [-10.0, -10.0, 2.0],
                    [10.0, -10.0, 2.0],
                    [10.0, 10.0, 2.0],
                    [-10.0, 10.0, 2.0],
                ],
                dtype=np.float64,
            ),
            triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            source_vertex_count=4,
            source_triangle_count=2,
            nonfinite_vertex_count=0,
            rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )

    @staticmethod
    def metric_report():
        curve = {
            "fpr": np.array([0.0, 0.0, 1.0]),
            "tpr": np.array([0.0, 1.0, 1.0]),
            "recall": np.array([0.0, 1.0, 1.0]),
            "precision": np.array([1.0, 1.0, 0.5]),
            "auroc": 1.0,
            "auprc": 1.0,
        }
        risk_points = [
            {
                "coverage": coverage,
                "count": index + 1,
                "mean_distance": 0.01 + 0.01 * index,
                "high_error_rate": 0.1 * index,
            }
            for index, coverage in enumerate((0.25, 0.5, 0.75, 1.0))
        ]
        return {
            "iteration": 7000,
            "primary": {
                "curves": {"N": curve, "A": curve, "one_minus_S": curve}
            },
            "risk_coverage": {
                name: {"direction": direction, "points": risk_points}
                for name, direction in (
                    ("N", "retain_low"),
                    ("r_p", "retain_high"),
                    ("r_g", "retain_high"),
                )
            },
            "state": {
                "per_state": {
                    name: {
                        "count": count,
                        "fraction": count / 100.0,
                        "mean_distance": 0.01 * state_id,
                        "high_error_rate": 0.05 * state_id,
                    }
                    for state_id, (name, count) in enumerate(
                        (
                            ("Bypass", 10),
                            ("Consensus", 20),
                            ("Prior-led", 30),
                            ("Geometry-led", 25),
                            ("Abstain", 15),
                        )
                    )
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
