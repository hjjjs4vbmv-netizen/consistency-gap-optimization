import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/q128_matched_spacing_20260824"


class Q128MatchedSpacingResultsTest(unittest.TestCase):
    def test_audited_matrix_is_complete(self):
        audit = json.loads((RESULTS / "audit.json").read_text())
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["jobs"], 210)
        self.assertEqual(audit["metric_values"], 420)
        self.assertFalse(audit["invalidated_directories_included"])
        with (RESULTS / "evaluation_results.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 210)
        self.assertEqual(len({row["job_id"] for row in rows}), 210)
        self.assertEqual({row["status"] for row in rows}, {"SEALED_PASS"})

    def test_summary_regenerates_frozen_primary_contrast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_q128_matched_spacing_results.py"),
                    "--input",
                    str(RESULTS / "evaluation_results.csv"),
                    "--audit",
                    str(RESULTS / "audit.json"),
                    "--outdir",
                    str(outdir),
                ],
                check=True,
            )
            with (outdir / "contrast_summary.csv").open(newline="") as handle:
                contrasts = {
                    (row["contrast"], int(row["nfe"])): row
                    for row in csv.DictReader(handle)
                }
            self.assertAlmostEqual(
                float(contrasts[("Bmatch-Bsame", 1)]["mean"]),
                0.06220344639719825,
            )
            self.assertAlmostEqual(
                float(contrasts[("Bmatch-Bsame", 2)]["mean"]),
                0.07641163632687416,
            )
            with (outdir / "a_bsame_bmatch_trajectories.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 21)
            with (outdir / "ttq_exploratory.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 18)


if __name__ == "__main__":
    unittest.main()
