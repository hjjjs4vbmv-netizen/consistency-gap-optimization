import copy
import json
import unittest
from pathlib import Path

from scripts import validate_gap_lr_seed_replication_receipt as validator
from scripts import verify_gap_lr_seed_replication_group as group_verifier


ROOT = Path(__file__).resolve().parents[1]


class SeedReplicationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.matrix = json.loads(
            (ROOT / "configs" / "gap_lr_matched_seed_replication_matrix.json").read_text()
        )
        self.source_receipt = json.loads(
            (ROOT / "results" / "gap_lr_matched" / "final_gap_lr_audit_receipt.json").read_text()
        )

    def test_frozen_matrix_passes(self):
        validator.validate_matrix(self.matrix)

    def test_source_receipt_passes_as_inherited_mechanism_gate(self):
        validator.validate_source_receipt(self.source_receipt)

    def test_removing_paired_baseline_fails_closed(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["arms"] = matrix["arms"][1:]
        with self.assertRaisesRegex(SystemExit, "exactly A/B/C"):
            validator.validate_matrix(matrix)

    def test_seed_extension_cannot_add_seed_six(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["new_formal_seeds"] = [4, 5, 6]
        with self.assertRaisesRegex(SystemExit, "exactly \[4, 5\]"):
            validator.validate_matrix(matrix)

    def test_schedule_change_fails_closed(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["shared_training"]["mapping"] = "sigmoid"
        with self.assertRaisesRegex(SystemExit, "mapping changed"):
            validator.validate_matrix(matrix)

    def test_within_seed_normalization_allows_only_gap_lr_and_path(self):
        base = {
            "seed": 4,
            "run_dir": "/a",
            "loss_kwargs": {"global_gap_scale": 1.0, "q": 128.0},
            "optimizer_kwargs": {"lr": 0.0001, "class_name": "torch.optim.RAdam"},
        }
        candidate = copy.deepcopy(base)
        candidate["run_dir"] = "/b"
        candidate["loss_kwargs"]["global_gap_scale"] = 1.3
        candidate["optimizer_kwargs"]["lr"] = validator.C_LR
        self.assertEqual(
            group_verifier.normalized_within_seed(base),
            group_verifier.normalized_within_seed(candidate),
        )
        candidate["loss_kwargs"]["q"] = 256.0
        self.assertNotEqual(
            group_verifier.normalized_within_seed(base),
            group_verifier.normalized_within_seed(candidate),
        )

    def test_between_seed_normalization_allows_only_seed_and_path(self):
        first = {"seed": 4, "run_dir": "/s4", "batch_size": 128}
        second = {"seed": 5, "run_dir": "/s5", "batch_size": 128}
        self.assertEqual(
            group_verifier.normalized_between_seeds(first),
            group_verifier.normalized_between_seeds(second),
        )
        second["batch_size"] = 64
        self.assertNotEqual(
            group_verifier.normalized_between_seeds(first),
            group_verifier.normalized_between_seeds(second),
        )


if __name__ == "__main__":
    unittest.main()
