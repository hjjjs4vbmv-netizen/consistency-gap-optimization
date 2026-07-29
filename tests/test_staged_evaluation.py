import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from metrics import metric_main
from scripts import collect_staged_evaluation_results
from scripts import run_staged_evaluation


class StagedEvaluationTest(unittest.TestCase):
    def make_manifest(self, root: Path, include_receipt: bool = True):
        checkpoint = root / "checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        receipt = root / "checkpoint.integrity.json"
        checksum = run_staged_evaluation.sha256_file(checkpoint)
        if include_receipt:
            receipt.write_text(json.dumps({
                "schema_version": 1,
                "status": "passed",
                "checkpoint_id": "baseline_seed0",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checksum,
                "training_run_id": "baseline-seed0-run",
                "method": "baseline",
                "training_seed": 0,
                "budget_kimg": 16,
                "completion_passed": True,
                "logs_state_consistent": True,
                "finite_loss_state_passed": True,
                "checker_version": "1",
                "checker_git_commit": "0123456789abcdef",
                "checked_at_unix": 1_753_822_000,
            }), encoding="utf-8")
        cell = {
            "checkpoint_id": "baseline_seed0",
            "method": "baseline",
            "training_seed": 0,
            "budget_kimg": 16,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checksum,
        }
        if include_receipt:
            cell["integrity_receipt"] = str(receipt)
        manifest = root / "checkpoints.json"
        manifest.write_text(json.dumps({
            "protocol": run_staged_evaluation.PROTOCOL_ID,
            "cells": [cell],
        }), encoding="utf-8")
        return manifest, checkpoint, receipt

    def test_runner_builds_frozen_smoke_and_formal_commands(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _, _ = self.make_manifest(root)
            data = root / "cifar.zip"
            data.write_bytes(b"dataset")
            cells, comparison = run_staged_evaluation.load_cells(manifest, False)
            selected = run_staged_evaluation.select_cells(cells, "smoke", "baseline_seed0")
            smoke = run_staged_evaluation.build_jobs(
                selected, data, root / "smoke", "smoke", 29800, False
            )
            formal = run_staged_evaluation.build_jobs(
                cells, data, root / "formal", "formal", 29800, False
            )
            self.assertIsNone(comparison)
            self.assertEqual(len(smoke), 2)
            self.assertEqual(len(formal), 2)
            for job in smoke:
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-4999", command)
                self.assertIn("--metrics=kid5k_full,fid5k_full", command)
                self.assertIn("--metric-repeats=1", command)
                self.assertIn("--seed=20260730", command)
                self.assertEqual(job["evidence_class"], "quick")
            for job in formal:
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-49999", command)
                self.assertIn("--metrics=kid50k_full,fid50k_full", command)
                self.assertEqual(job["integrity_receipt"]["status"], "passed")
                self.assertEqual(job["mid_t"], [] if job["nfe"] == 1 else [0.821])

    def test_formal_runner_rejects_missing_integrity_receipt(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _, _ = self.make_manifest(root, include_receipt=False)
            cells, _ = run_staged_evaluation.load_cells(manifest, False)
            with self.assertRaisesRegex(SystemExit, "integrity_receipt"):
                run_staged_evaluation.build_jobs(
                    cells, root / "cifar.zip", root / "formal", "formal", 29800, False
                )

    def test_collector_writes_long_table_and_segregated_statistics(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _, _ = self.make_manifest(root)
            data = root / "cifar.zip"
            data.write_bytes(b"dataset")
            cells, comparison = run_staged_evaluation.load_cells(manifest, False)
            jobs = run_staged_evaluation.build_jobs(
                cells, data, root, "smoke", 29800, False
            )
            record = run_staged_evaluation.build_record(
                cells, comparison, data, root, "smoke", jobs,
                run_staged_evaluation.sha256_file(data),
            )
            record["status"] = "completed"
            for job in jobs:
                job["status"] = "completed"
                target = Path(job["output_directory"])
                target.mkdir(parents=True)
                for metric_index, metric_name in enumerate(job["metric_names"]):
                    (target / f"metric-{metric_name}.jsonl").write_text(
                        json.dumps({
                            "metric": metric_name,
                            "results": {metric_name: 1.0 + job["nfe"] + metric_index},
                        }) + "\n",
                        encoding="utf-8",
                    )
            (root / "run_manifest.json").write_text(json.dumps(record), encoding="utf-8")
            rows, summary = collect_staged_evaluation_results.collect(root)
            output = root / "summary"
            collect_staged_evaluation_results.write_outputs(output, rows, summary)
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["statistics_grouping"], [
                "evidence_class", "metric_name", "nfe", "method",
            ])
            self.assertTrue((output / "evaluation_results.csv").is_file())
            self.assertTrue((output / "evaluation_statistics.json").is_file())

    def test_kid50k_uses_the_frozen_metric_seed(self):
        opts = mock.Mock()
        opts.dataset_kwargs = {}
        opts.metric_seed = 20260730
        with mock.patch.object(
            metric_main.kernel_inception_distance,
            "compute_kid",
            return_value=0.125,
        ) as compute_kid:
            result = metric_main.kid50k_full(opts)
        self.assertEqual(result, {"kid50k_full": 0.125})
        self.assertEqual(compute_kid.call_args.kwargs["random_seed"], 20260730)


if __name__ == "__main__":
    unittest.main()
