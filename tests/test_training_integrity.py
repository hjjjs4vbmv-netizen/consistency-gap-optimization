import csv
import json
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts import check_training_integrity


class CheckpointSchedule:
    def __init__(self, name: str, global_gap_scale: float = 1.0):
        self.name = name
        self.global_gap_scale = global_gap_scale


class CheckpointLoss:
    def __init__(self, schedule: CheckpointSchedule):
        self.schedule = schedule


class CheckpointEMA(torch.nn.Module):
    def __init__(self, finite: bool = True):
        super().__init__()
        value = 1.0 if finite else float("nan")
        self.weight = torch.nn.Parameter(torch.tensor([value]))
        self.register_buffer("running_value", torch.tensor([value]))


class TrainingIntegrityTest(unittest.TestCase):
    def test_checker_adds_repository_root_to_import_path(self):
        self.assertEqual(sys.path[0], str(check_training_integrity.REPO_ROOT))

    def make_run(
        self, root: Path, finite: bool = True, method: str = "fixed",
        checkpoint_finite: bool = True,
    ) -> Path:
        run = root / "run"
        run.mkdir()
        schedule, scale = (
            ("sigmoid", 1.0) if method == "fixed" else ("global_sigmoid", 1.10)
        )
        checkpoint = {
            "ema": CheckpointEMA(checkpoint_finite),
            "loss_fn": CheckpointLoss(CheckpointSchedule(schedule, scale)),
        }
        # The evaluator stores snapshots with pickle; keep the fixture format
        # identical to the actual network-snapshot-latest.pkl artifact.
        with (run / "network-snapshot-latest.pkl").open("wb") as handle:
            pickle.dump(checkpoint, handle)
        (run / "training_options.json").write_text(
            json.dumps({
                "total_kimg": 16, "seed": 7,
                "loss_kwargs": {"adj": schedule, "global_gap_scale": scale},
            }), encoding="utf-8"
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
            self.assertTrue(receipt["checkpoint_load_passed"])
            self.assertTrue(receipt["ema_present"])
            self.assertTrue(receipt["ema_finite_passed"])
            self.assertTrue(receipt["schedule_identity_passed"])
            self.assertTrue(receipt["global_gap_scale_identity_passed"])
            self.assertTrue(receipt["method_identity_passed"])
            self.assertEqual(receipt["evidence"]["training_state"]["cur_nimg"], 16000)

    def test_allows_float32_progress_rounding_when_integer_state_agrees(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir))
            (run / "stats.jsonl").write_text(json.dumps({
                "Loss/loss": {"mean": 1.0},
                # A realistic float32-style reporting discrepancy of 0.025 image.
                "Progress/kimg": {"mean": 16.000025},
            }) + "\n", encoding="utf-8")
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": None,
                "checker_version": "3",
            })()
            receipt = check_training_integrity.build_receipt(args)
            self.assertEqual(receipt["status"], "passed")

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

    def test_rejects_non_finite_ema(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir), checkpoint_finite=False)
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": None,
                "checker_version": "2",
            })()
            with self.assertRaisesRegex(SystemExit, "EMA has non-finite"):
                check_training_integrity.build_receipt(args)

    def test_rejects_declared_global_method_with_fixed_schedule(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir), method="fixed")
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "global_seed7_16k",
                "method": "global110", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "global-seed7-16k", "expected_training_commit": None,
                "checker_version": "2",
            })()
            with self.assertRaisesRegex(SystemExit, "requires schedule"):
                check_training_integrity.build_receipt(args)

    def test_rejects_unloadable_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir))
            (run / "network-snapshot-latest.pkl").write_bytes(b"not a pickle")
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": None,
                "checker_version": "2",
            })()
            with self.assertRaisesRegex(SystemExit, "cannot load checkpoint pickle"):
                check_training_integrity.build_receipt(args)

    def test_rejects_missing_ema(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir))
            checkpoint_path = run / "network-snapshot-latest.pkl"
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            del checkpoint["ema"]
            with checkpoint_path.open("wb") as handle:
                pickle.dump(checkpoint, handle)
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "fixed_seed7_16k",
                "method": "fixed", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "fixed-seed7-16k", "expected_training_commit": None,
                "checker_version": "2",
            })()
            with self.assertRaisesRegex(SystemExit, "EMA object"):
                check_training_integrity.build_receipt(args)

    def test_rejects_global_gap_scale_identity_mismatch(self):
        with TemporaryDirectory() as temp_dir:
            run = self.make_run(Path(temp_dir), method="global110")
            options_path = run / "training_options.json"
            options = json.loads(options_path.read_text(encoding="utf-8"))
            options["loss_kwargs"]["global_gap_scale"] = 1.2
            options_path.write_text(json.dumps(options), encoding="utf-8")
            args = type("Args", (), {
                "run_dir": run, "checkpoint": None, "checkpoint_id": "global_seed7_16k",
                "method": "global110", "training_seed": 7, "budget_kimg": 16,
                "training_run_id": "global-seed7-16k", "expected_training_commit": None,
                "checker_version": "2",
            })()
            with self.assertRaisesRegex(SystemExit, "global_gap_scale=1.1"):
                check_training_integrity.build_receipt(args)


if __name__ == "__main__":
    unittest.main()
