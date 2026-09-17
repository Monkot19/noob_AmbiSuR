import unittest

import torch

from reliability.topology import (
    append_topology_change,
    compose_topology_changes,
    identity_topology_change,
    prune_topology_change,
)


class TopologyCompositionTests(unittest.TestCase):
    def test_append_marks_every_appended_row_as_new(self):
        change = append_topology_change(2, 3, device="cpu")

        self.assertEqual(change.new_to_old.tolist(), [0, 1, -1, -1, -1])
        self.assertEqual(
            change.is_new.tolist(), [False, False, True, True, True]
        )

    def test_append_records_parent_identity_for_clone_children(self):
        change = append_topology_change(
            3,
            2,
            parent_indices=torch.tensor([2, 0], dtype=torch.int64),
            device="cpu",
        )

        self.assertEqual(change.new_to_old.tolist(), [0, 1, 2, 2, 0])
        self.assertEqual(
            change.is_new.tolist(), [False, False, False, True, True]
        )

    def test_composition_preserves_mapped_child_identity_and_newness(self):
        append = append_topology_change(
            2,
            1,
            parent_indices=torch.tensor([1], dtype=torch.int64),
            device="cpu",
        )
        prune = prune_topology_change(
            torch.tensor([True, False, False], dtype=torch.bool)
        )

        composed = compose_topology_changes(append, prune)

        self.assertEqual(composed.new_to_old.tolist(), [1, 1])
        self.assertEqual(composed.is_new.tolist(), [False, True])

    def test_prune_maps_survivors_to_pre_prune_rows(self):
        change = prune_topology_change(
            torch.tensor([False, True, False, True])
        )

        self.assertEqual(change.new_to_old.tolist(), [0, 2])
        self.assertFalse(change.is_new.any())

    def test_composition_preserves_original_survivors_and_new_rows(self):
        append = append_topology_change(2, 1, device="cpu")
        prune = prune_topology_change(torch.tensor([False, True, False]))

        composed = compose_topology_changes(append, prune)

        self.assertEqual(composed.new_to_old.tolist(), [0, -1])
        self.assertEqual(composed.is_new.tolist(), [False, True])

    def test_identity_composes_without_changing_mapping(self):
        identity = identity_topology_change(3, device="cpu")
        prune = prune_topology_change(torch.tensor([True, False, False]))

        composed = compose_topology_changes(identity, prune)

        self.assertEqual(composed.new_to_old.tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
