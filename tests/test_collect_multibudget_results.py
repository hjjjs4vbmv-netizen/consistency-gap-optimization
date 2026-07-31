import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import collect_multibudget_results as collector
from scripts import summarize_budget_curve


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
                for extension in ("svg", "png", "pdf"):
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

    def test_frozen_q256_endpoint_sets_may_differ_by_budget(self):
        frozen = json.loads(Path("configs/q256_budget_matrix.frozen.json").read_text(encoding="utf-8"))
        endpoints = {
            int(contract["budget_kimg"]): contract["metric_names"]
            for contract in frozen["evaluation_contracts"]
        }
        self.assertEqual(endpoints, {
            512: ["kid5k_full", "fid5k_full"],
            768: ["kid5k_full", "fid5k_full"],
            1024: ["kid50k_full", "fid50k_full"],
        })
        rows = []
        for budget, metrics in endpoints.items():
            for metric in metrics:
                for nfe in (1, 2):
                    for seed in (3, 4, 5):
                        for method in ("fixed", "global110"):
                            baseline = 1.0 if metric.startswith("kid") else 100.0
                            value = baseline + seed * 0.001 - (0.02 if method == "global110" else 0.0)
                            rows.append({
                                "method": method, "training_seed": seed, "budget_kimg": budget,
                                "nfe": nfe, "metric_name": metric, "metric_value": value,
                                "training_time_hours": budget / 64, "quality_target": "",
                                "checkpoint_sha256": "{}-{}-{}-{}".format(method, seed, budget, metric),
                            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frozen_protocol.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            output = root / "collected"
            collector.main(["--input-csv", str(source), "--outdir", str(output)])
            with (output / "paired_deltas.csv").open(newline="", encoding="utf-8") as handle:
                paired = list(csv.DictReader(handle))
        self.assertEqual(len(paired), 36)
        self.assertEqual(
            {row["metric_name"] for row in paired if float(row["budget_kimg"]) in (512, 768)},
            {"kid5k_full", "fid5k_full"},
        )
        self.assertEqual(
            {row["metric_name"] for row in paired if float(row["budget_kimg"]) == 1024},
            {"kid50k_full", "fid50k_full"},
        )

    def test_protocol_tracks_split_5k_budget_curve_from_50k_formal_endpoints(self):
        rows = []
        track_specs = {
            "budget_curve": {
                "evaluation_contract": "q256-common-5k-v1", "sample_count": 5000,
                "generation_seed_range": "0-4999", "metric_seed": 20260730,
                "budgets": (256, 512, 768, 1024), "metrics": ("kid5k_full", "fid5k_full"),
            },
            "formal_endpoint": {
                "evaluation_contract": "q256-formal-50k-v1", "sample_count": 50000,
                "generation_seed_range": "0-49999", "metric_seed": 20260730,
                "budgets": (256, 1024), "metrics": ("kid50k_full", "fid50k_full"),
            },
        }
        for track, spec in track_specs.items():
            for budget in spec["budgets"]:
                for metric in spec["metrics"]:
                    for nfe in (1, 2):
                        for seed in (3, 4, 5):
                            for method in ("fixed", "global110"):
                                baseline = 1.0 if metric.startswith("kid") else 100.0
                                rows.append({
                                    "method": method, "training_seed": seed, "budget_kimg": budget,
                                    "nfe": nfe, "metric_name": metric,
                                    "metric_value": baseline + seed * 0.001 - (0.02 if method == "global110" else 0.0),
                                    "training_time_hours": budget / 64, "quality_target": "",
                                    "checkpoint_sha256": "{}-{}-{}-{}".format(method, seed, budget, metric),
                                    "sample_count": spec["sample_count"],
                                    "generation_seed_range": spec["generation_seed_range"],
                                    "metric_seed": spec["metric_seed"],
                                    "evidence_class": "formal" if track == "formal_endpoint" else (
                                        "auxiliary" if budget == 1024 else "quick"
                                    ),
                                    "evaluation_contract": spec["evaluation_contract"],
                                    "analysis_track": track,
                                })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "two_protocols.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            output = root / "collected"
            collector.main(["--input-csv", str(source), "--outdir", str(output)])
            with (output / "same_protocol_budget_curves.csv").open(newline="", encoding="utf-8") as handle:
                curves = list(csv.DictReader(handle))
            with (output / "formal_endpoint_comparison.csv").open(newline="", encoding="utf-8") as handle:
                formal = list(csv.DictReader(handle))
            self.assertGreater((output / "figures" / "same_protocol_budget_curves.pdf").stat().st_size, 0)
            self.assertGreater((output / "figures" / "formal_endpoint_comparison.pdf").stat().st_size, 0)
        self.assertEqual({float(row["budget_kimg"]) for row in curves}, {256, 512, 768, 1024})
        self.assertEqual({row["metric_name"] for row in curves}, {"kid5k_full", "fid5k_full"})
        self.assertEqual({row["sample_count"] for row in curves}, {"5000"})
        self.assertEqual({float(row["budget_kimg"]) for row in formal}, {256, 1024})
        self.assertEqual({row["metric_name"] for row in formal}, {"kid50k_full", "fid50k_full"})
        self.assertEqual({row["sample_count"] for row in formal}, {"50000"})

    def test_budget_curve_script_writes_paper_ready_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            rows = self.make_rows()
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            output = root / "figures"
            summarize_budget_curve.main([
                "--input-csv", str(source), "--outdir", str(output),
                "--baseline-method", "fixed", "--candidate-method", "global110",
            ])
            self.assertGreater((output / "budget_curves.pdf").stat().st_size, 0)
            self.assertTrue((output / "budget_curve_summary.csv").is_file())
            self.assertTrue((output / "paired_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
