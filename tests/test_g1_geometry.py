from pathlib import Path
import tempfile
import unittest

import numpy as np

from reliability.offline_g1 import (
    ValidatedMesh,
    _writable_open3d_arrays,
    closest_triangle_distances,
    load_valid_mesh,
    validate_mesh_arrays,
)

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


def one_triangle_mesh():
    return ValidatedMesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        source_vertex_count=3,
        source_triangle_count=1,
        nonfinite_vertex_count=0,
        rejected_nonfinite_triangle_count=0,
        rejected_degenerate_triangle_count=0,
    )


class G1GeometryTests(unittest.TestCase):
    def test_open3d_boundary_copies_read_only_validated_mesh_arrays(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2]], dtype=np.int64)
        vertices.setflags(write=False)
        triangles.setflags(write=False)
        mesh = ValidatedMesh(
            vertices=vertices,
            triangles=triangles,
            source_vertex_count=3,
            source_triangle_count=1,
            nonfinite_vertex_count=0,
            rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )

        writable_vertices, writable_triangles = _writable_open3d_arrays(mesh)

        self.assertTrue(writable_vertices.flags.writeable)
        self.assertTrue(writable_triangles.flags.writeable)
        self.assertFalse(np.shares_memory(writable_vertices, vertices))
        self.assertFalse(np.shares_memory(writable_triangles, triangles))
        np.testing.assert_array_equal(writable_vertices, vertices)
        np.testing.assert_array_equal(writable_triangles, triangles)

    def test_validation_rejects_nonfinite_and_zero_area_triangles(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [np.nan, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        triangles = np.array(
            [
                [0, 1, 2],  # valid
                [0, 1, 3],  # nonfinite vertex
                [0, 1, 4],  # zero area
            ],
            dtype=np.int64,
        )

        mesh = validate_mesh_arrays(vertices, triangles)

        self.assertEqual(mesh.source_vertex_count, 5)
        self.assertEqual(mesh.source_triangle_count, 3)
        self.assertEqual(mesh.nonfinite_vertex_count, 1)
        self.assertEqual(mesh.rejected_nonfinite_triangle_count, 1)
        self.assertEqual(mesh.rejected_degenerate_triangle_count, 1)
        self.assertEqual(mesh.triangles.shape, (1, 3))
        self.assertTrue(np.isfinite(mesh.vertices).all())

    def test_validation_rejects_out_of_range_triangle_index(self):
        with self.assertRaisesRegex(ValueError, "triangle index out of range"):
            validate_mesh_arrays(
                np.zeros((3, 3), dtype=np.float64),
                np.array([[0, 1, 3]], dtype=np.int64),
            )

    def test_validation_rejects_mesh_without_valid_triangles(self):
        with self.assertRaisesRegex(ValueError, "no valid GT triangles"):
            validate_mesh_arrays(
                np.array(
                    [[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [2.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array([[0, 1, 2]], dtype=np.int64),
            )

    @unittest.skipIf(o3d is None, "Open3D is required for triangle queries")
    def test_closest_distance_covers_face_edge_and_vertex(self):
        points = np.array(
            [
                [0.25, 0.25, 2.0],
                [0.50, -2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        distances = closest_triangle_distances(
            points, one_triangle_mesh(), chunk_size=2
        )

        np.testing.assert_allclose(
            distances,
            [2.0, 2.0, np.sqrt(4.5)],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(distances.dtype, np.float64)

    @unittest.skipIf(o3d is None, "Open3D is required for triangle queries")
    def test_chunk_size_does_not_change_distances(self):
        points = np.array(
            [[0.25, 0.25, z] for z in (0.1, 0.2, 0.3, 0.4, 0.5)],
            dtype=np.float64,
        )

        one = closest_triangle_distances(points, one_triangle_mesh(), chunk_size=1)
        all_at_once = closest_triangle_distances(
            points, one_triangle_mesh(), chunk_size=100
        )

        np.testing.assert_array_equal(one, all_at_once)

    def test_query_rejects_nonfinite_points_before_open3d(self):
        with self.assertRaisesRegex(ValueError, "query points must be finite"):
            closest_triangle_distances(
                np.array([[np.inf, 0.0, 0.0]], dtype=np.float64),
                one_triangle_mesh(),
            )

    @unittest.skipIf(o3d is None, "Open3D is required for mesh I/O")
    def test_load_valid_mesh_reads_triangle_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            legacy = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(one_triangle_mesh().vertices),
                o3d.utility.Vector3iVector(one_triangle_mesh().triangles),
            )
            self.assertTrue(o3d.io.write_triangle_mesh(str(path), legacy))

            loaded = load_valid_mesh(path)

            self.assertEqual(loaded.triangles.shape, (1, 3))
            np.testing.assert_allclose(loaded.vertices, one_triangle_mesh().vertices)


if __name__ == "__main__":
    unittest.main()
