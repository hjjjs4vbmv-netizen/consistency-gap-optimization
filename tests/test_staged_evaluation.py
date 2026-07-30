import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from metrics import frechet_inception_distance
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
                "checkpoint_load_passed": True,
                "ema_present": True,
                "ema_finite_passed": True,
                "schedule_identity_passed": True,
                "global_gap_scale_identity_passed": True,
                "method_identity_passed": True,
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
                if job["nfe"] == 1:
                    self.assertFalse(any(argument.startswith("--mid_t=") for argument in job["command"]))
                else:
                    self.assertIn("--mid_t=0.821", job["command"])
            for job in formal:
                command = " ".join(job["command"])
                self.assertIn("--sample-seeds=0-49999", command)
                self.assertIn("--metrics=kid50k_full,fid50k_full", command)
                self.assertEqual(job["integrity_receipt"]["status"], "passed")
                self.assertEqual(job["mid_t"], [] if job["nfe"] == 1 else [0.821])
                if job["nfe"] == 1:
                    self.assertFalse(any(argument.startswith("--mid_t=") for argument in job["command"]))
                else:
                    self.assertIn("--mid_t=0.821", job["command"])

    def test_formal_runner_rejects_missing_integrity_receipt(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _, _ = self.make_manifest(root, include_receipt=False)
            cells, _ = run_staged_evaluation.load_cells(manifest, False)
            with self.assertRaisesRegex(SystemExit, "integrity_receipt"):
                run_staged_evaluation.build_jobs(
                    cells, root / "cifar.zip", root / "formal", "formal", 29800, False
                )

    def test_formal_runner_rejects_missing_checkpoint_identity_gate(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, _, receipt = self.make_manifest(root)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            del payload["ema_finite_passed"]
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            cells, _ = run_staged_evaluation.load_cells(manifest, False)
            with self.assertRaisesRegex(SystemExit, "ema_finite_passed"):
                run_staged_evaluation.build_jobs(
                    cells, root / "cifar.zip", root / "formal", "formal", 29800, False
                )

    def test_formal_promotion_policy_requires_all_predeclared_checkpoints(self):
        cells = [
            {"checkpoint_id": "fixed_seed3"},
            {"checkpoint_id": "global_seed3"},
        ]
        policy = {
            "formal_promotion_policy": {
                "eligibility": "provenance_and_integrity_only",
                "quick_metric_performance": "not_an_eligibility_criterion",
                "required_checkpoint_ids": ["fixed_seed3", "global_seed3"],
            }
        }
        run_staged_evaluation.validate_formal_promotion_policy(policy, cells)
        policy["formal_promotion_policy"]["required_checkpoint_ids"] = ["fixed_seed3"]
        with self.assertRaisesRegex(SystemExit, "every predeclared checkpoint"):
            run_staged_evaluation.validate_formal_promotion_policy(policy, cells)

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
            environment = record["evaluation_environment"]
            self.assertEqual(environment["evaluation_git_commit"], record["evaluation_git_commit"])
            self.assertTrue(environment["python"]["version"])
            self.assertTrue(environment["scipy_version"])
            self.assertTrue(environment["pytorch_version"])
            self.assertIn("compiled_version", environment["cuda"])
            self.assertEqual(environment["inception_detector"]["identifier"], "inception-2015-12-05")
            self.assertEqual(environment["dataset_sha256"], record["dataset_sha256"])
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

    def test_collector_computes_deltas_from_explicit_pairing_contract(self):
        rows = [
            {"method": "fixed", "training_seed": 4, "budget_kimg": 256, "metric_name": "fid50k_full", "nfe": 1, "metric_value": 10.0, "checkpoint_id": "fixed4", "checkpoint_sha256": "a" * 64},
            {"method": "global110", "training_seed": 4, "budget_kimg": 256, "metric_name": "fid50k_full", "nfe": 1, "metric_value": 8.0, "checkpoint_id": "global4", "checkpoint_sha256": "b" * 64},
            {"method": "fixed", "training_seed": 5, "budget_kimg": 256, "metric_name": "fid50k_full", "nfe": 1, "metric_value": 12.0, "checkpoint_id": "fixed5", "checkpoint_sha256": "c" * 64},
            {"method": "global110", "training_seed": 5, "budget_kimg": 256, "metric_name": "fid50k_full", "nfe": 1, "metric_value": 9.0, "checkpoint_id": "global5", "checkpoint_sha256": "d" * 64},
        ]
        pairing = {
            "pairing_key": ["training_seed", "budget_kimg", "nfe", "metric_name"],
            "baseline_method": "fixed",
            "candidate_method": "global110",
            "candidate_label": "global_only",
            "delta_direction": "global_only - fixed",
        }
        result = collect_staged_evaluation_results.build_pairwise_statistics(rows, pairing)
        self.assertEqual(result["status"], "computed")
        statistic = result["statistics"][0]
        self.assertEqual(statistic["pair_count"], 2)
        self.assertEqual(statistic["mean_delta"], -2.5)
        self.assertEqual(statistic["global_wins"], 2)
        self.assertEqual(statistic["fixed_wins"], 0)
        self.assertEqual(statistic["ties"], 0)
        self.assertEqual([item["delta"] for item in result["paired_differences"]], [-2.0, -3.0])
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            collect_staged_evaluation_results.write_paired_outputs(output, result)
            self.assertTrue((output / "paired_differences.csv").is_file())
            self.assertTrue((output / "paired_statistics.json").is_file())
            self.assertTrue((output / "paired_statistics.md").is_file())

    def test_collector_rejects_missing_or_duplicated_fixed_global_pair(self):
        row = {
            "method": "fixed", "training_seed": 4, "budget_kimg": 256,
            "metric_name": "kid50k_full", "nfe": 2, "metric_value": 0.2,
            "checkpoint_id": "fixed4", "checkpoint_sha256": "a" * 64,
        }
        pairing = {
            "pairing_key": ["training_seed", "budget_kimg", "nfe", "metric_name"],
            "baseline_method": "fixed", "candidate_method": "global110",
            "candidate_label": "global_only", "delta_direction": "global_only - fixed",
        }
        with self.assertRaisesRegex(SystemExit, "unpaired fixed/global"):
            collect_staged_evaluation_results.build_pairwise_statistics([row], pairing)
        with self.assertRaisesRegex(SystemExit, "duplicate fixed"):
            collect_staged_evaluation_results.build_pairwise_statistics([row, row], pairing)

    def test_smoke_collection_does_not_attempt_the_full_paired_comparison(self):
        manifest = {
            "phase": "smoke",
            "evidence_class": "smoke_only",
            "comparison": {
                "pairing_key": ["training_seed", "budget_kimg", "nfe", "metric"],
                "baseline_method": "fixed",
                "candidate_method": "global110",
                "candidate_label": "global_only",
                "delta_direction": "global_only - fixed",
            },
        }
        rows = [{
            "evidence_class": "smoke_only", "metric_name": "fid5k_full",
            "nfe": 1, "method": "fixed", "metric_value": 10.0,
            "training_seed": 3, "budget_kimg": 256,
        }]
        result = collect_staged_evaluation_results.build_statistics(rows, manifest)
        self.assertEqual(result["pairwise_statistics"], {
            "status": "not_computed",
            "reason": "smoke phase does not form the predeclared comparison matrix",
        })

    def test_fid_uses_scipy_sqrtm_without_removed_disp_argument(self):
        class Stats:
            def get_mean_cov(self):
                return np.zeros(2), np.eye(2)

        opts = mock.Mock(rank=0)
        with mock.patch.object(
            frechet_inception_distance.metric_utils,
            "compute_feature_stats_for_dataset",
            return_value=Stats(),
        ), mock.patch.object(
            frechet_inception_distance.metric_utils,
            "compute_feature_stats_for_generator",
            return_value=Stats(),
        ), mock.patch.object(
            frechet_inception_distance.scipy.linalg,
            "sqrtm",
            return_value=np.eye(2),
        ) as sqrtm:
            result = frechet_inception_distance.compute_fid(
                opts, max_real=5000, num_gen=5000
            )

        self.assertEqual(result, 0.0)
        sqrtm.assert_called_once()
        np.testing.assert_array_equal(sqrtm.call_args.args[0], np.eye(2))
        self.assertEqual(sqrtm.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
