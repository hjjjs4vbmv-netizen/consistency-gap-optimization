import csv
import tempfile
import unittest
from pathlib import Path

from scripts import run_q256_cohort3 as cohort3


class Cohort3FrozenContractTest(unittest.TestCase):
    def test_frozen_mapping_and_arm_order(self):
        self.assertEqual(list(cohort3.SEED_GPU.items()), [(8, 0), (9, 1), (10, 2), (11, 3), (12, 4)])
        self.assertEqual(list(cohort3.ARMS), ["A", "B", "C", "D"])
        self.assertEqual(
            list(cohort3.ARMS.values()),
            [("1.0", "1.0"), ("1.1", "1.1"), ("1.1", "1.0"), ("1.0", "1.1")],
        )

    def test_formal_command_is_training_only_and_frozen(self):
        command = cohort3.training_command(
            run_dir=Path("/data/raw/ECT/ect_runs/test/formal/seed8/armA"),
            arm="A",
            seed=8,
            mode="formal",
        )
        self.assertIn("--metrics=none", command)
        self.assertIn("--duration=0.256", command)
        self.assertIn("--target-gap-scale=1.0", command)
        self.assertIn("--denominator-gap-scale=1.0", command)
        self.assertIn("--workers=1", command)
        self.assertIn("--bench=False", command)
        self.assertIn("--tf32=False", command)
        self.assertIn(f"--transfer={cohort3.TRANSFER}", command)
        self.assertFalse(any("fid" in item.lower() or "kid" in item.lower() for item in command))

    def test_resume_omits_transfer_and_planned_pause(self):
        state = Path("/data/raw/ECT/ect_runs/test/training-state-latest.pt")
        command = cohort3.training_command(
            run_dir=state.parent,
            arm="A",
            seed=cohort3.SMOKE_SEED,
            mode="smoke",
            resume=state,
        )
        self.assertIn(f"--resume={state}", command)
        self.assertFalse(any(item.startswith("--transfer=") for item in command))
        self.assertFalse(any(item.startswith("--stop-after-attempts=") for item in command))

    def test_wall_clock_fields_do_not_enter_computational_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            for path, elapsed in ((first, "1.0"), (second, "999.0")):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["attempted_iteration", "value", "elapsed_sec"])
                    writer.writeheader()
                    writer.writerow({"attempted_iteration": "1", "value": "same", "elapsed_sec": elapsed})
            self.assertEqual(cohort3.canonical_csv_digest(first), cohort3.canonical_csv_digest(second))

    def test_preregistration_digest_and_seed_binding(self):
        record = cohort3.preregistration_record()
        self.assertEqual(record["sha256"], "aef53e73f106bde476162006c51cc318e949c3374bf50d0992054722622ff24f")


if __name__ == "__main__":
    unittest.main()
