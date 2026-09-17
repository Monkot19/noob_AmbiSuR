import unittest

import torch

from reliability.topology import (
    TopologyChange,
    migrate_lineage_tensor,
    migrate_tensor,
)


class TopologyMigrationTests(unittest.TestCase):
    def test_survivors_follow_new_to_old_and_new_rows_are_reset(self):
        change = TopologyChange(
            new_to_old=torch.tensor([2, 0, -1], dtype=torch.int64),
            is_new=torch.tensor([False, False, True]),
        )
        old = torch.tensor([[30.0, 31.0], [40.0, 41.0], [50.0, 51.0]])

        migrated = migrate_tensor(old, change, fill_value=0.0)

        torch.testing.assert_close(
            migrated,
            torch.tensor([[50.0, 51.0], [30.0, 31.0], [0.0, 0.0]]),
        )

    def test_boolean_validity_does_not_propagate_to_new_gaussian(self):
        change = TopologyChange(
            new_to_old=torch.tensor([1, -1], dtype=torch.int64),
            is_new=torch.tensor([False, True]),
        )

        migrated = migrate_tensor(torch.tensor([False, True]), change, fill_value=False)

        self.assertEqual(migrated.tolist(), [True, False])

    def test_mapped_new_child_still_resets_generic_evidence_state(self):
        change = TopologyChange(
            new_to_old=torch.tensor([0, 0, -1], dtype=torch.int64),
            is_new=torch.tensor([False, True, True]),
        )

        migrated = migrate_tensor(
            torch.tensor([7.0]), change, fill_value=0.0
        )

        self.assertEqual(migrated.tolist(), [7.0, 0.0, 0.0])

    def test_mapped_new_child_inherits_explicit_lineage_state(self):
        change = TopologyChange(
            new_to_old=torch.tensor([0, 0, -1], dtype=torch.int64),
            is_new=torch.tensor([False, True, True]),
        )

        migrated = migrate_lineage_tensor(
            torch.tensor([7]), change, fill_value=0
        )

        self.assertEqual(migrated.tolist(), [7, 7, 0])

    def test_change_rejects_unmapped_survivor(self):
        with self.assertRaisesRegex(ValueError, "is_new"):
            TopologyChange(
                new_to_old=torch.tensor([0, -1], dtype=torch.int64),
                is_new=torch.tensor([False, False]),
            )

    def test_change_rejects_out_of_range_source_index_during_migration(self):
        change = TopologyChange(
            new_to_old=torch.tensor([3], dtype=torch.int64),
            is_new=torch.tensor([False]),
        )

        with self.assertRaisesRegex(ValueError, "out of range"):
            migrate_tensor(torch.tensor([1.0, 2.0]), change)


if __name__ == "__main__":
    unittest.main()
