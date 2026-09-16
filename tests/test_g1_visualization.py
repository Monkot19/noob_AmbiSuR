from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from reliability.g1_visualization import (
    camera_quartile_indices,
    scalar_colors,
    select_static_cameras,
    state_colors,
    write_colored_ply,
)


class G1VisualizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
