import csv
import tempfile
import unittest
from pathlib import Path

from scripts import collect_multibudget_results as collector


class MultiBudgetCollectorTest(unittest.TestCase):
    def make_rows(self):
        rows = []
        for metric, target in (("kid50k_full", 0.76), ("fid50k_full", 76.0)):
            for nfe in (1, 2):
                for budget in (512, 768, 1024):
                    for seed in (3, 4, 5):
                        for method in ("fixed", "global110"):
                            baseline = (
                                (1.0 if metric.startswith("kid") else 100.0)
                                - budget / (3000 if metric.startswith("kid") else 30)
                            )
                            value = baseline + seed * (0.002 if metric.startswith("kid") else 0.2)
                            if method == "global110":
                                value -= 0.03 if nfe == 1 else 0.06
                            rows.append({
                                "method": method,
                                "training_seed": seed,
                                "budget_kimg": budget,
                                "nfe": nfe,
                                "metric_name": metric,
                                "metric_value": value,
                                "training_time_hours": budget / 64 + seed * 0.01,
                                "quality_target": target,
                                "checkpoint_sha256": "{}-{}-{}-{}".format(method, seed, budget, metric),
                            })
        return rows

    def test_complete_matrix_writes_all_requested_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            rows = self.make_rows()
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            output = root / "collected"
            collector.main([
                "--input-csv", str(source), "--outdir", str(output),
                "--baseline-method", "fixed", "--candidate-method", "global110",
            ])
            for name in (
                "normalized_metrics.csv", "budget_curves.csv", "per_seed_trajectories.csv",
                "paired_deltas.csv", "paired_summary.csv", "time_to_quality.csv",
                "summary_table.md", "summary_table.tex", "figure_ready_budget_curves.csv",
                "figure_ready_per_seed_trajectories.csv", "figure_ready_paired_deltas.csv",
                "figure_ready_time_to_quality.csv",
                "collector_manifest.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            for stem in ("budget_curves", "per_seed_trajectories", "paired_deltas", "time_to_quality"):
                for extension in ("svg", "png"):
                    self.assertGreater((output / "figures" / "{}.{}".format(stem, extension)).stat().st_size, 0)
            with (output / "paired_deltas.csv").open(newline="", encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            self.assertEqual(len(pairs), 36)
            self.assertTrue(all(float(row["delta_candidate_minus_baseline"]) < 0 for row in pairs))
            with (output / "time_to_quality.csv").open(newline="", encoding="utf-8") as handle:
                self.assertIn("reached", {row["status"] for row in csv.DictReader(handle)})

    def test_incomplete_matrix_is_rejected(self):
        rows = self.make_rows()
        with self.assertRaisesRegex(SystemExit, "matrix incomplete"):
            collector.validate(rows[:-1], "fixed", "global110")


if __name__ == "__main__":
    unittest.main()
