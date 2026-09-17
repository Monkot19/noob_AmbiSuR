import unittest

import numpy as np

from reliability.offline_g1 import G1IterationInputs, ValidatedMesh
from scripts.diagnostics.evaluate_d0_g1 import evaluate_iteration


class G1OrchestrationTests(unittest.TestCase):
    def test_evaluate_iteration_uses_every_finite_center_and_matching_rows(self):
        snapshot = {
            "A": np.array([0.0, 0.2, 0.9]),
            "S": np.array([1.0, 0.8, 0.1]),
            "N": np.array([0.0, 0.36, 0.99]),
            "r_p": np.array([0.2, 0.4, 0.8]),
            "V_p": np.array([True, True, True]),
            "r_g": np.array([0.1, 0.3, 0.7]),
            "V_g": np.array([True, True, True]),
            "stable": np.array([0, 2, 4], dtype=np.int8),
        }
        inputs = G1IterationInputs(
            iteration=500,
            original_point_count=3,
            centers=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            finite_row_indices=np.array([0, 2]),
            rejected_center_indices=np.array([1]),
            snapshot=snapshot,
            checkpoint_sha256="a" * 64,
            snapshot_sha256="b" * 64,
        )
        mesh = ValidatedMesh(
            vertices=np.zeros((3, 3)),
            triangles=np.array([[0, 1, 2]]),
            source_vertex_count=3,
            source_triangle_count=1,
            nonfinite_vertex_count=0,
            rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )
        calls = []

        def distance_fn(points, received_mesh):
            calls.append((points.copy(), received_mesh))
            return np.array([0.01, 0.20])

        result = evaluate_iteration(inputs, mesh, distance_fn=distance_fn)

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], inputs.centers)
        self.assertIs(calls[0][1], mesh)
        np.testing.assert_array_equal(result["snapshot"]["N"], [0.0, 0.99])
        np.testing.assert_array_equal(result["distances"], [0.01, 0.20])
        self.assertEqual(result["report"]["iteration"], 500)
        self.assertIsNone(result["report"]["g1_evaluable"])
        self.assertIsNone(result["report"]["g1_pass"])
        self.assertEqual(result["rejected_center_count"], 1)

    def test_evaluate_iteration_rejects_distance_contract_violation(self):
        base = G1IterationInputs(
            iteration=500,
            original_point_count=1,
            centers=np.zeros((1, 3)),
            finite_row_indices=np.array([0]),
            rejected_center_indices=np.array([], dtype=np.int64),
            snapshot={
                "A": np.zeros(1), "S": np.ones(1), "N": np.zeros(1),
                "r_p": np.ones(1), "V_p": np.ones(1, dtype=bool),
                "r_g": np.ones(1), "V_g": np.ones(1, dtype=bool),
                "stable": np.zeros(1, dtype=np.int8),
            },
            checkpoint_sha256="a" * 64,
            snapshot_sha256="b" * 64,
        )
        mesh = ValidatedMesh(
            vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2]]),
            source_vertex_count=3, source_triangle_count=1,
            nonfinite_vertex_count=0, rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )
        with self.assertRaisesRegex(ValueError, "distance row count"):
            evaluate_iteration(
                base,
                mesh,
                distance_fn=lambda _points, _mesh: np.zeros(2),
            )


if __name__ == "__main__":
    unittest.main()
