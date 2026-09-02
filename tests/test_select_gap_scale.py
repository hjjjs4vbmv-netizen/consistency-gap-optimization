import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import select_gap_scale


class SelectGapScaleTest(unittest.TestCase):
    def make_complete_matrix(self, root: Path) -> tuple[Path, Path]:
        runs_root = root / "runs"
        eval_root = root / "eval"
        nfe1_kid = {
            "1": 0.300,
            "0.97": 0.310,
            "1.032": 0.290,
            "1.06": 0.290,
            "1.10": 0.400,
        }
        for spec in select_gap_scale.CELLS:
            label = spec["label"]
            run_dir = runs_root / label
            run_dir.mkdir(parents=True)
            validation = {
                "status": "passed",
                "expected_schedule": spec["expected_schedule"],
            }
            (run_dir / "validation.json").write_text(
                json.dumps(validation) + "\n", encoding="utf-8"
            )
            scale = spec["global_scale_text"]
            for nfe in select_gap_scale.NFES:
                cell = eval_root / label / f"nfe{nfe}"
                cell.mkdir(parents=True)
                (cell / "experiment_meta.env").write_text(
                    "exit_code=0\n", encoding="utf-8"
                )
                values = {
                    "kid5k_full": (
                        nfe1_kid[scale] if nfe == 1 else nfe1_kid[scale] / 2
                    ),
                    "fid5k_full": 300 + 10 * nfe + spec["global_scale"],
                }
                for metric, value in values.items():
                    payload = {"metric": metric, "results": {metric: value}}
                    (cell / f"metric-{metric}.jsonl").write_text(
                        json.dumps(payload) + "\n", encoding="utf-8"
                    )
        return runs_root, eval_root

    def test_selects_nfe1_kid_only_and_uses_tie_breaker(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root, eval_root = self.make_complete_matrix(root)
            outdir = root / "analysis"
            select_gap_scale.main(
                [
                    "--runs-root",
                    str(runs_root),
                    "--eval-root",
                    str(eval_root),
                    "--outdir",
                    str(outdir),
                ]
            )

            report = json.loads(
                (outdir / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["selected_global_scale_text"], "1.032")
            self.assertEqual(report["selected_label"], "global-g1p0320-seed0-256k")
            self.assertTrue(report["selected_beats_fixed_nfe1_kid"])
            self.assertAlmostEqual(report["nfe1_kid_delta_vs_fixed"], -0.01)
            self.assertEqual(
                (outdir / "selected_g.txt").read_text(encoding="utf-8"),
                "1.032\n",
            )
            with (outdir / "response_curve.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(
                [row["global_scale_text"] for row in rows if row["is_selected"] == "True"],
                ["1.032"],
            )
            markdown = (outdir / "response_curve.md").read_text(encoding="utf-8")
            self.assertIn("selection bias", markdown)
            self.assertIn("not standard FID-50k", markdown)

    def test_rejects_nonfinite_or_multiple_metric_results(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root, eval_root = self.make_complete_matrix(root)
            metric_path = (
                eval_root
                / "global-g0p9700-seed0-256k"
                / "nfe1"
                / "metric-kid5k_full.jsonl"
            )
            payload = {
                "metric": "kid5k_full",
                "results": {"kid5k_full": float("nan")},
            }
            metric_path.write_text(
                json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "exactly one result line"):
                select_gap_scale.main(
                    [
                        "--runs-root",
                        str(runs_root),
                        "--eval-root",
                        str(eval_root),
                        "--outdir",
                        str(root / "analysis"),
                    ]
                )
            self.assertFalse((root / "analysis").exists())


if __name__ == "__main__":
    unittest.main()
