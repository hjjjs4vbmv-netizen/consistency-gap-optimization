import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_gap_factorial.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_gap_factorial", MODULE_PATH
)
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


class SummarizeGapFactorialTests(unittest.TestCase):
    selected_scale = 1.032

    def make_matrix(self, root: Path):
        runs_root = root / "runs"
        eval_root = root / "eval"
        selection_path = root / "selection.json"
        runs_root.mkdir()
        eval_root.mkdir()

        selected_label = "global-g1p0320-seed0-256k"
        selection = {
            "schema_version": 1,
            "status": "passed",
            "selected_global_scale": self.selected_scale,
            "selected_global_scale_text": "1.032",
            "selected_label": selected_label,
            "selected": {
                "label": selected_label,
                "global_scale": self.selected_scale,
            },
        }
        selection_path.write_text(
            json.dumps(selection), encoding="utf-8"
        )

        arm_values = {
            "fixed": 10.0,
            "global": 9.0,
            "local-conservative": 8.0,
            "combined-conservative": 6.0,
            "local-aggressive": 7.5,
            "combined-aggressive": 5.5,
        }
        for spec in summarizer.arm_specs(self.selected_scale):
            for seed in summarizer.TRAINING_SEEDS:
                label = summarizer.run_label(spec, seed)
                run_dir = runs_root / label
                run_dir.mkdir()
                validation = {
                    "status": "passed",
                    "expected_schedule": spec["schedule"],
                    "final_processed_kimg": 256.0,
                }
                (run_dir / "validation.json").write_text(
                    json.dumps(validation), encoding="utf-8"
                )
                (run_dir / "experiment_meta.env").write_text(
                    "\n".join(
                        [
                            f"arm={spec['arm']}",
                            f"schedule={spec['schedule']}",
                            f"local_profile={spec['metadata_profile']}",
                            f"global_gap_scale={spec['scale']}",
                            f"seed={seed}",
                            "duration_mimg=0.256",
                            "exit_code=0",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                for nfe in summarizer.NFES:
                    eval_dir = eval_root / label / f"nfe{nfe}"
                    eval_dir.mkdir(parents=True)
                    (eval_dir / "experiment_meta.env").write_text(
                        f"label={label}\nnfe={nfe}\nexit_code=0\n",
                        encoding="utf-8",
                    )
                    for metric_index, metric in enumerate(
                        summarizer.METRICS
                    ):
                        value = (
                            arm_values[spec["arm"]]
                            + 0.1 * seed
                            + 0.01 * nfe
                            + 0.001 * metric_index
                        )
                        payload = {
                            "metric": metric,
                            "results": {metric: value},
                        }
                        (eval_dir / f"metric-{metric}.jsonl").write_text(
                            json.dumps(payload) + "\n", encoding="utf-8"
                        )
        return runs_root, eval_root, selection_path

    def invoke(self, runs_root, eval_root, selection_path, outdir):
        summarizer.main(
            [
                "--runs-root",
                str(runs_root),
                "--eval-root",
                str(eval_root),
                "--selection-json",
                str(selection_path),
                "--outdir",
                str(outdir),
            ]
        )

    def test_complete_matrix_outputs_and_contrasts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root, eval_root, selection_path = self.make_matrix(root)
            outdir = root / "summary"
            self.invoke(runs_root, eval_root, selection_path, outdir)

            for name in summarizer.OUTPUT_NAMES:
                self.assertTrue((outdir / name).is_file())
                self.assertGreater((outdir / name).stat().st_size, 0)

            with (outdir / "per_cell_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                cells = list(csv.DictReader(handle))
            self.assertEqual(len(cells), 36)

            with (outdir / "per_seed_effects.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                effects = list(csv.DictReader(handle))
            self.assertEqual(len(effects), 24)
            target = next(
                row
                for row in effects
                if row["profile"] == "conservative"
                and row["training_seed"] == "0"
                and row["nfe"] == "1"
                and row["metric"] == "kid5k_full"
            )
            self.assertAlmostEqual(
                float(target["global_at_local0_delta"]), -1.0
            )
            self.assertAlmostEqual(
                float(target["local_at_global0_delta"]), -2.0
            )
            self.assertAlmostEqual(
                float(target["combined_vs_fixed_delta"]), -4.0
            )
            self.assertAlmostEqual(
                float(target["additive_interaction_delta"]), -1.0
            )
            self.assertAlmostEqual(
                float(target["global_main_effect_delta"]), -1.5
            )
            self.assertAlmostEqual(
                float(target["local_main_effect_delta"]), -2.5
            )

            summary = json.loads(
                (outdir / "factorial_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["matrix"]["unique_training_cells"], 18)
            self.assertEqual(
                summary["matrix"]["unique_metric_files_read_once"], 72
            )
            self.assertTrue(summary["selection_overlap"]["present"])
            self.assertEqual(len(summary["summaries"]), 64)

    def test_rejects_failed_training_validation_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root, eval_root, selection_path = self.make_matrix(root)
            failed = (
                runs_root / "fixed-g1p0000-seed0-256k" / "validation.json"
            )
            failed.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "expected_schedule": "sigmoid",
                        "final_processed_kimg": 256.0,
                    }
                ),
                encoding="utf-8",
            )
            outdir = root / "summary"
            with self.assertRaises(SystemExit):
                self.invoke(runs_root, eval_root, selection_path, outdir)
            self.assertFalse(outdir.exists())

    def test_rejects_more_than_one_metric_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_root, eval_root, selection_path = self.make_matrix(root)
            metric_path = (
                eval_root
                / "fixed-g1p0000-seed0-256k"
                / "nfe1"
                / "metric-kid5k_full.jsonl"
            )
            metric_path.write_text(
                metric_path.read_text(encoding="utf-8") * 2,
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                self.invoke(
                    runs_root, eval_root, selection_path, root / "summary"
                )


if __name__ == "__main__":
    unittest.main()
