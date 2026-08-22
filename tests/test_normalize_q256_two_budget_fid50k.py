import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import normalize_q256_two_budget_fid50k as normalizer
from scripts.collect_multibudget_results import read_rows


class NormalizeQ256TwoBudgetFid50kTest(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "schema_version": 1,
            "kind": "q256_two_budget_fid50k_source",
            "budgets_kimg": [256, 1024],
            "training_seeds": [3, 4, 5],
            "nfes": [1, 2],
            "arms": {
                "A": "arm_a_g100_weight100", "B": "arm_b_g110_weight110",
                "C": "arm_c_g110_weight100", "D": "arm_d_g100_weight110",
            },
            "metric_name": "fid50k_full",
            "sample_count": 50000,
            "generation_seed_range": "0-49999",
            "metric_seed": 20260730,
            "evidence_class": "formal_preregistered",
            "evaluation_contract": "q256-fid50k-50k-v1",
            "analysis_track": "two_budget_endpoint",
        }

    @staticmethod
    def raw_rows(budget):
        rows = []
        for seed in (3, 4, 5):
            for arm_index, arm in enumerate(("A", "B", "C", "D")):
                for nfe in (1, 2):
                    rows.append({
                        "seed": seed, "arm": arm, "nfe": nfe,
                        "fid50k_full": 340 - budget / 4 + seed + arm_index * 2 + nfe,
                        "status": "PASS", "receipt_sha256": "receipt-{}-{}-{}".format(budget, seed, arm),
                        "artifacts_tree_sha256": "tree-{}-{}-{}".format(budget, seed, arm),
                        "generated_features_sha256": "features-{}-{}-{}".format(budget, seed, arm),
                    })
        return rows

    def write_raw(self, root, budget, rows=None):
        path = root / "{}k.csv".format(budget)
        rows = self.raw_rows(budget) if rows is None else rows
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_builds_complete_fid50k_two_budget_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(self.config()) + "\n", encoding="utf-8")
            raw256, raw1024 = self.write_raw(root, 256), self.write_raw(root, 1024)
            output, manifest = root / "source.csv", root / "manifest.json"
            normalizer.main([
                "--config", str(config), "--budget-input", "256={}".format(raw256),
                "--budget-input", "1024={}".format(raw1024), "--out-csv", str(output),
                "--manifest", str(manifest),
            ])
            rows = read_rows(output)
            self.assertEqual(len(rows), 48)
            self.assertEqual({row["metric_name"] for row in rows}, {"fid50k_full"})
            self.assertEqual({row["analysis_track"] for row in rows}, {"two_budget_endpoint"})
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["record_count"], 48)
            self.assertIn("No FID-5k", payload["prohibition"])

    def test_rejects_a_missing_raw_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(self.config()) + "\n", encoding="utf-8")
            raw256 = self.write_raw(root, 256)
            rows1024 = [row for row in self.raw_rows(1024) if not (row["seed"] == 5 and row["arm"] == "D" and row["nfe"] == 2)]
            raw1024 = self.write_raw(root, 1024, rows1024)
            with self.assertRaisesRegex(SystemExit, "complete frozen 1024-kimg matrix"):
                normalizer.main([
                    "--config", str(config), "--budget-input", "256={}".format(raw256),
                    "--budget-input", "1024={}".format(raw1024), "--out-csv", str(root / "source.csv"),
                    "--manifest", str(root / "manifest.json"),
                ])


if __name__ == "__main__":
    unittest.main()
