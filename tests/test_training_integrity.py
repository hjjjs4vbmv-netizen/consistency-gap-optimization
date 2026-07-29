import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts import check_training_integrity


class TrainingIntegrityTest(unittest.TestCase):
    def make_run(self, root: Path, finite: bool = True) -> Path:
        run = root / "run"
        run.mkdir()
        (run / "network-snapshot-latest.pkl").write_bytes(b"checkpoint")
        (run / "training_options.json").write_text(
            json.dumps({"total_kimg": 16, "seed": 7}), encoding="utf-8"
        )
        (run / "stats.jsonl").write_text(json.dumps({
            "Loss/loss": {"mean": 1.0},
            "Progress/kimg": {"mean": 16.0},
        }) + "\n", encoding="utf-8")
        with (run / "train_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["processed_kimg", "loss"])
            writer.writeheader()
            writer.writerow({"processed_kimg": "16", "loss": "1.0" if finite else "nan"})
        (run / "log.txt").write_text("training complete\nExiting...\n", encoding="utf-8")
        torch.save({"cur_nimg": 16000, "weight": torch.tensor([1.0])}, run / "training-state-latest.pt")
        (run / "commit_sha.txt").write_text("training abcdef\n", encoding="utf-8")
        return run

    def test_emits_passed_receipt_for_consistent_run(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root)
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": "abcdef",
                "checker_version": "1",
            })()
            receipt = check_training_integrity.build_receipt(args)
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(receipt["finite_loss_state_passed"])
            self.assertEqual(receipt["evidence"]["training_state"]["cur_nimg"], 16000)

    def test_rejects_non_finite_loss(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = self.make_run(root, finite=False)
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": None,
                "checker_version": "1",
            })()
            with self.assertRaisesRegex(SystemExit, "non-finite loss"):
                check_training_integrity.build_receipt(args)


if __name__ == "__main__":
    unittest.main()
