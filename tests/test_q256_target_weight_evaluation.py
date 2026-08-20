import json
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from scripts import collect_q256_target_weight_results as collector
from scripts import run_q256_target_weight_evaluation as evaluator


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def binding(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": evaluator.sha256_file(path)}


class Q256TargetWeightEvaluationTest(unittest.TestCase):
    def test_evaluator_preserves_process_cleanup_error_subtype(self):
        with TemporaryDirectory() as tmp, mock.patch.object(
            evaluator.training_launcher,
            "stream_process",
            side_effect=evaluator.training_launcher.ProcessCleanupError(
                "owned process group remains"
            ),
        ):
            with self.assertRaises(
                evaluator.training_launcher.ProcessCleanupError
            ):
                evaluator.stream_process(
                    ["python", "eval.py"],
                    env={},
                    log_path=Path(tmp) / "eval.log",
                    monitored_gpu_uuid="GPU-test",
                    gpu_monitor_record={},
                )

    def _minimal_durable_evaluation_plan(self, root: Path):
        output_root = root / "evaluation-lifecycle"
        for name in ("receipts", "manifests", "process_logs", "jobs"):
            (output_root / name).mkdir(parents=True, exist_ok=True)
        checkpoint = root / "checkpoint.pkl"
        checkpoint.write_bytes(b"checkpoint")
        dataset = root / "dataset.zip"
        dataset.write_bytes(b"dataset")
        job = {
            "job_id": "seed3-armA-nfe1",
            "output_directory": str(output_root / "jobs" / "seed3-armA-nfe1"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": evaluator.sha256_file(checkpoint),
        }
        plan = {"jobs": [job]}
        plan_path = output_root / "evaluation_plan.json"
        dump_json(plan_path, plan)
        return output_root, plan_path, plan, dataset

    def test_durable_plan_prejob_failure_writes_terminal_stop(self):
        with TemporaryDirectory() as tmp:
            output_root, plan_path, plan, dataset = self._minimal_durable_evaluation_plan(
                Path(tmp)
            )
            with mock.patch.object(
                evaluator,
                "source_snapshot",
                side_effect=evaluator.EvaluationError("prejob source audit failed"),
            ):
                with self.assertRaisesRegex(evaluator.EvaluationError, "prejob"):
                    evaluator.run_authorized_plan_jobs(
                        plan=plan,
                        plan_path=plan_path,
                        dataset={"path": str(dataset), "sha256": "d" * 64},
                        source={"content_sha256": "s" * 64},
                        gpu={"uuid": "GPU-test"},
                        process_env={},
                        output_root=output_root,
                        data_argument=dataset,
                        plan_sha256=evaluator.sha256_file(plan_path),
                    )
            completion = json.loads(
                (output_root / "evaluation_completion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(completion["status"], "STOPPED_FOR_AUDIT")
            self.assertEqual(completion["failed_job_id"], "seed3-armA-nfe1")
            self.assertIn("prejob source audit failed", completion["error"])

    def test_durable_plan_signal_writes_terminal_stop_and_restores_handler(self):
        with TemporaryDirectory() as tmp:
            output_root, plan_path, plan, dataset = self._minimal_durable_evaluation_plan(
                Path(tmp)
            )
            original_handler = signal.getsignal(signal.SIGTERM)

            def interrupt_source(*_args, **_kwargs):
                os.kill(os.getpid(), signal.SIGTERM)
                return {"content_sha256": "s" * 64}

            with mock.patch.object(
                evaluator, "source_snapshot", side_effect=interrupt_source
            ):
                with self.assertRaisesRegex(evaluator.EvaluationError, "SIGTERM"):
                    evaluator.run_authorized_plan_jobs(
                        plan=plan,
                        plan_path=plan_path,
                        dataset={"path": str(dataset), "sha256": "d" * 64},
                        source={"content_sha256": "s" * 64},
                        gpu={"uuid": "GPU-test"},
                        process_env={},
                        output_root=output_root,
                        data_argument=dataset,
                        plan_sha256=evaluator.sha256_file(plan_path),
                    )
            self.assertIs(signal.getsignal(signal.SIGTERM), original_handler)
            completion = json.loads(
                (output_root / "evaluation_completion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(completion["status"], "STOPPED_FOR_AUDIT")
            self.assertEqual(completion["received_signal"]["signal"], "SIGTERM")

    def test_direct_cli_help_loads_repo_dependencies(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(evaluator.REPO_ROOT / "scripts" / "run_q256_target_weight_evaluation.py"),
                "--help",
            ],
            cwd=Path("/tmp"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("--matrix-dir", completed.stdout)

    def test_runtime_contract_is_exact_and_requires_one_a100(self):
        runtime = {
            "python_version": evaluator.EXPECTED_PYTHON_VERSION,
            "torch_version": evaluator.EXPECTED_TORCH_VERSION,
            "torch_cuda_version": evaluator.EXPECTED_TORCH_CUDA_VERSION,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_names": ["NVIDIA A100 80GB PCIe"],
        }
        evaluator.validate_runtime(runtime)
        wrong = dict(runtime, torch_version="2.3.0")
        with self.assertRaisesRegex(evaluator.EvaluationError, "torch_version"):
            evaluator.validate_runtime(wrong)
        wrong = dict(runtime, cuda_device_names=["NVIDIA H100 80GB HBM3"])
        with self.assertRaisesRegex(evaluator.EvaluationError, "A100"):
            evaluator.validate_runtime(wrong)

    def test_evaluator_idle_record_matches_collector_contract(self):
        with mock.patch.object(
            evaluator.training_launcher,
            "query_gpu_compute_processes",
            return_value=[],
        ):
            record = evaluator.assert_gpu_idle({"uuid": "GPU-test"})
        evaluator.training_launcher.validate_gpu_idle_record(
            record,
            label="evaluation fixture",
            expected_gpu_uuid="GPU-test",
        )

    def test_source_snapshot_uses_git_1_8_compatible_status_and_branch_queries(self):
        calls = []

        def checked(args, **_kwargs):
            calls.append(list(args))
            if args[1] == "rev-parse" and "--is-inside-work-tree" in args:
                return "true"
            if args[1] == "status":
                return ""
            if args[1] == "symbolic-ref":
                return evaluator.EXPECTED_BRANCH
            if args[1] == "rev-parse":
                return "a" * 40
            raise AssertionError(args)

        tracked = b"\0".join(
            path.encode("utf-8") for path in sorted(evaluator._SOURCE_EXACT)
        ) + b"\0"
        with mock.patch.object(evaluator, "_checked_output", side_effect=checked), mock.patch.object(
            evaluator.subprocess, "check_output", return_value=tracked
        ):
            snapshot = evaluator.source_snapshot(require_clean=True)
        self.assertEqual(snapshot["git_branch"], evaluator.EXPECTED_BRANCH)
        self.assertIn(
            ["git", "status", "--porcelain", "--untracked-files=all"], calls
        )
        self.assertNotIn(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], calls
        )
        self.assertIn(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], calls
        )
        self.assertNotIn(["git", "branch", "--show-current"], calls)

    def make_training_run(self, root: Path, seed: int, arm: str) -> dict:
        run = root / f"seed{seed}" / f"arm{arm}"
        run.mkdir(parents=True)
        checkpoint = run / evaluator.CHECKPOINT_FILENAME
        checkpoint.write_bytes(f"checkpoint-{seed}-{arm}".encode())
        options = run / "training_options.json"
        dump_json(options, {"fixture": True})
        source_head = "1" * 40
        source_content = "2" * 64
        prereg_sha = evaluator.sha256_file(
            evaluator.REPO_ROOT
            / "analysis"
            / "q256_target_weight_factorial"
            / "preregistration.json"
        )
        launch = run / "launch_manifest.json"
        dump_json(
            launch,
            {
                "schema": "ect.q256.target-weight-factorial-launch/v2",
                "experiment_id": evaluator.EXPERIMENT_ID,
                "run_directory": str(run.resolve()),
                "training": {
                    "phase": "formal",
                    "arm": arm,
                    "seed": seed,
                    "factorial_protocol": evaluator.TRAINING_PROTOCOL,
                    "ct_train_total_kimg": 256,
                    "expected_processed_nimg": 256000,
                    "expected_optimizer_attempts": 2000,
                },
                "assets": {
                    "dataset": {"sha256": evaluator.DATASET_SHA256},
                    "transfer": {
                        "sha256": evaluator.training_launcher.EXPECTED_TRANSFER_SHA256
                    },
                },
                "source": {
                    "git_branch": evaluator.EXPECTED_BRANCH,
                    "git_clean": True,
                    "git_head": source_head,
                    "content_sha256": source_content,
                },
                "preregistration": {
                    "path": "analysis/q256_target_weight_factorial/preregistration.json",
                    "sha256": prereg_sha,
                },
                "gpu": {"uuid": "GPU-test"},
                "post_training_verifier": {"expected_skip_attempts": None},
            },
        )
        validation = run / evaluator.VALIDATION_FILENAME
        dump_json(
            validation,
            {
                "schema": evaluator.TRAINING_VALIDATION_SCHEMA,
                "status": "passed",
                "run_dir": str(run.resolve()),
                "mode": "formal",
                "arm": arm,
                "seed": seed,
                "source_git_head": source_head,
                "source_content_sha256": source_content,
                "initial_common_state_sha256": "3" * 64,
                "successful_optimizer_steps": 1998,
                "amp_skip_attempts": [1, 2],
                "amp_skip_signature_expected_value_enforced": False,
                "amp_skip_policy": evaluator.training_launcher.AMP_SKIP_POLICY,
            },
        )
        hashes = run / evaluator.HASH_RECEIPT_FILENAME
        dump_json(
            hashes,
            {
                "schema": evaluator.TRAINING_HASH_SCHEMA,
                "status": "passed",
                "run_dir": str(run.resolve()),
                "mode": "formal",
                "arm": arm,
                "seed": seed,
                "artifacts": {},
            },
        )
        for name in evaluator.training_launcher.CORE_ARM_ARTIFACTS:
            path = run / name
            if not path.exists():
                path.write_bytes(f"fixture-{seed}-{arm}-{name}".encode("utf-8"))
        hash_payload = json.loads(hashes.read_text(encoding="utf-8"))
        artifact_names = set(evaluator.training_launcher.CORE_ARM_ARTIFACTS) | {
            evaluator.VALIDATION_FILENAME
        }
        hash_payload["artifacts"] = {
            name: binding(run / name) for name in sorted(artifact_names)
        }
        dump_json(hashes, hash_payload)
        runner_log = run / "runner.log"
        verifier_log = run / "arm_verifier.log"
        runner_log.write_text("runner PASS\n", encoding="utf-8")
        verifier_log.write_text("verifier PASS\n", encoding="utf-8")
        monitor = {
            "schema": evaluator.training_launcher.GPU_MONITOR_SCHEMA,
            "status": "PASS",
            "gpu_uuid": "GPU-test",
            "root_process_pid": 123,
            "poll_interval_seconds": 1.0,
            "cadence_grace_seconds": 0.25,
            "probe_timeout_seconds": 0.4,
            "started_utc": "2026-08-19T00:00:00Z",
            "finished_utc": "2026-08-19T00:00:01Z",
            "first_check_started_utc": "2026-08-19T00:00:00Z",
            "last_check_started_utc": "2026-08-19T00:00:01Z",
            "checks_completed": 2,
            "first_check_offset_seconds": 0.0,
            "last_check_offset_seconds": 1.0,
            "monitor_duration_seconds": 1.0,
            "max_observed_poll_gap_seconds": 1.0,
            "max_observed_check_duration_seconds": 0.1,
            "max_observed_schedule_lateness_seconds": 0.01,
            "foreign_process_incident": None,
            "own_process_group_signals": [],
        }
        idle = {
            "checked_utc": "2026-08-19T00:00:01Z",
            "gpu_uuid": "GPU-test",
            "compute_process_count": 0,
            "query": "gpu_uuid,pid,process_name,used_gpu_memory",
        }
        dump_json(
            run / "runner_completion.json",
            {
                "schema": evaluator.training_launcher.RUNNER_COMPLETION_SCHEMA,
                "experiment_id": evaluator.EXPERIMENT_ID,
                "started_utc": "2026-08-19T00:00:00Z",
                "finished_utc": "2026-08-19T00:00:02Z",
                "status": "PASS",
                "returncode": 0,
                "verifier_returncode": 0,
                "launch_manifest": launch.name,
                "launch_manifest_sha256": evaluator.sha256_file(launch),
                "runner_log": runner_log.name,
                "runner_log_sha256": evaluator.sha256_file(runner_log),
                "verifier_log": verifier_log.name,
                "verifier_log_sha256": evaluator.sha256_file(verifier_log),
                "training_gpu_exclusivity_monitor": monitor,
                "verifier_gpu_exclusivity_monitor": monitor,
                "final_prelaunch_gpu_idle_check": idle,
                "post_training_gpu_idle_check": idle,
                "post_verifier_gpu_idle_check": idle,
                "verification": {
                    "validation_receipt_sha256": evaluator.sha256_file(
                        validation
                    ),
                    "artifact_hash_receipt_sha256": evaluator.sha256_file(
                        hashes
                    ),
                },
            },
        )
        return evaluator.validate_training_run(run, arm, seed)

    def make_training_matrix(self, root: Path) -> tuple[Path, list[dict]]:
        matrix = root / "formal" / "matrix-fixture"
        matrix.mkdir(parents=True)
        cells = []
        jobs = []
        completed = []
        for seed in evaluator.SEEDS:
            for arm in evaluator.ARMS:
                cell = self.make_training_run(root / "runs", seed, arm)
                cells.append(cell)
                jobs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "gpu": "0",
                        "master_port": 30000 + len(jobs),
                        "outdir": cell["run_dir"],
                        "resume": None,
                        "command_argv": ["fixture"],
                        "command_shell": "fixture",
                    }
                )
                completed.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "gpu": "0",
                        "returncode": 0,
                        "runner_completion": (
                            evaluator.training_launcher.validate_existing_runner_completion(
                                Path(cell["run_dir"])
                            )
                        ),
                    }
                )
        plan_path = matrix / "matrix_plan.json"
        dump_json(
            plan_path,
            {
                "schema": evaluator.TRAINING_MATRIX_SCHEMA,
                "experiment_id": evaluator.EXPERIMENT_ID,
                "phase": "formal",
                "mode": "fresh_exact_matrix",
                "expected_cell_count": 12,
                "expected_amp_skip_attempts": None,
                "amp_skip_policy": evaluator.training_launcher.AMP_SKIP_POLICY,
                "jobs": jobs,
            },
        )
        dump_json(
            matrix / "matrix_completion.json",
            {
                "schema": evaluator.TRAINING_COMPLETION_SCHEMA,
                "status": "PASS",
                "matrix_plan_sha256": evaluator.sha256_file(plan_path),
                "completed": completed,
                "skipped_existing_pass": [],
                "failures": [],
                "received_signal": None,
                "amp_skip_equivalence": {
                    str(seed): {
                        "arms": list(evaluator.ARMS),
                        "skip_attempts_by_arm": {
                            arm: [1, 2] for arm in evaluator.ARMS
                        },
                        "skip_count": 2,
                        "successful_optimizer_steps": 1998,
                        "initial_common_state_sha256": "3" * 64,
                    }
                    for seed in evaluator.SEEDS
                },
                "live_seed_identity": {
                    str(seed): {
                        "arms": list(evaluator.ARMS),
                        "amp_skip_attempts_by_arm": {
                            arm: [1, 2] for arm in evaluator.ARMS
                        },
                        "amp_skip_count": 2,
                        "successful_optimizer_steps": 1998,
                        "initial_common_state_sha256": "3" * 64,
                    }
                    for seed in evaluator.SEEDS
                },
            },
        )
        return matrix, cells

    def test_training_matrix_and_job_builder_are_exact(self):
        with TemporaryDirectory() as temp:
            matrix, _ = self.make_training_matrix(Path(temp))
            cells, matrix_record = evaluator.load_training_matrix(matrix)
            jobs = evaluator.build_jobs(cells, Path(temp) / "evaluation", 31800)
        self.assertEqual(len(cells), 12)
        self.assertEqual(matrix_record["selection_policy"], "all_exact_final_256kimg_cells_no_intermediate_selection")
        self.assertEqual(len(jobs), 24)
        self.assertEqual(
            {(job["seed"], job["arm"], job["nfe"]) for job in jobs},
            {
                (seed, arm, nfe)
                for seed in evaluator.SEEDS
                for arm in evaluator.ARMS
                for nfe in (1, 2)
            },
        )
        for job in jobs:
            command = job["command_argv_template"]
            self.assertIn("--sample-seeds=0-49999", command)
            self.assertIn("--metrics=kid50k_full,fid50k_full", command)
            self.assertIn("--metric-repeats=1", command)
            self.assertIn("--seed=20260730", command)
            self.assertIn("--fp16=False", command)
            self.assertIn("--retain-generated-artifacts", command)
            if job["nfe"] == 1:
                self.assertNotIn("--mid_t=0.821", command)
                self.assertEqual(job["mid_t"], [])
            else:
                self.assertIn("--mid_t=0.821", command)
                self.assertEqual(job["mid_t"], [0.821])

    def test_matrix_rejects_duplicate_cell_instead_of_selecting(self):
        with TemporaryDirectory() as temp:
            matrix, _ = self.make_training_matrix(Path(temp))
            path = matrix / "matrix_plan.json"
            plan = json.loads(path.read_text(encoding="utf-8"))
            plan["jobs"][-1] = dict(plan["jobs"][0])
            dump_json(path, plan)
            completion = json.loads((matrix / "matrix_completion.json").read_text(encoding="utf-8"))
            completion["matrix_plan_sha256"] = evaluator.sha256_file(path)
            dump_json(matrix / "matrix_completion.json", completion)
            with self.assertRaisesRegex(evaluator.EvaluationError, "duplicate"):
                evaluator.load_training_matrix(matrix)

    def test_matrix_rejects_cross_arm_initial_state_drift(self):
        with TemporaryDirectory() as temp:
            matrix, cells = self.make_training_matrix(Path(temp))
            target = next(
                cell for cell in cells if cell["seed"] == 3 and cell["arm"] == "D"
            )
            run_dir = Path(target["run_dir"])
            validation_path = run_dir / evaluator.VALIDATION_FILENAME
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["initial_common_state_sha256"] = "4" * 64
            dump_json(validation_path, validation)

            hashes_path = run_dir / evaluator.HASH_RECEIPT_FILENAME
            hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
            hashes["artifacts"][evaluator.VALIDATION_FILENAME] = binding(
                validation_path
            )
            dump_json(hashes_path, hashes)

            runner_path = run_dir / "runner_completion.json"
            runner = json.loads(runner_path.read_text(encoding="utf-8"))
            runner["verification"]["validation_receipt_sha256"] = (
                evaluator.sha256_file(validation_path)
            )
            runner["verification"]["artifact_hash_receipt_sha256"] = (
                evaluator.sha256_file(hashes_path)
            )
            dump_json(runner_path, runner)

            completion_path = matrix / "matrix_completion.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            for record in completion["completed"]:
                if record["seed"] == 3 and record["arm"] == "D":
                    record["runner_completion"] = (
                        evaluator.training_launcher.validate_existing_runner_completion(
                            run_dir
                        )
                    )
            dump_json(completion_path, completion)

            with self.assertRaisesRegex(
                evaluator.EvaluationError,
                "arm-specific initial common state",
            ):
                evaluator.load_training_matrix(matrix)

    def test_fixed_sampling_blocks_are_variation_only_and_immutable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            features_path = root / "features.npy"
            np.save(features_path, np.arange(60, dtype=np.float32).reshape(20, 3))
            output = root / "blocks.json"
            payload = evaluator.build_sampling_block_diagnostics(
                features_path, output, sample_count=20, block_size=5
            )
            self.assertEqual(payload["fixed_block_count"], 4)
            self.assertEqual(payload["independent_training_replicate_contribution"], 0)
            self.assertFalse(payload["quality_endpoint"])
            self.assertFalse(payload["selection_criterion"])
            self.assertEqual(
                [(row["sample_seed_start"], row["sample_seed_end"]) for row in payload["blocks"]],
                [(0, 4), (5, 9), (10, 14), (15, 19)],
            )
            with self.assertRaisesRegex(evaluator.EvaluationError, "overwrite"):
                evaluator.build_sampling_block_diagnostics(
                    features_path, output, sample_count=20, block_size=5
                )

    def test_seed_level_contrasts_and_cross_seed_summary(self):
        endpoints = []
        for seed in evaluator.SEEDS:
            for metric in evaluator.METRICS:
                for nfe in evaluator.NFE_SETTINGS:
                    base = float(seed + (100 if metric.startswith("kid") else 0) + nfe)
                    for arm, offset in {"A": 0, "B": -4, "C": -3, "D": -1}.items():
                        endpoints.append(
                            {
                                "training_seed": seed,
                                "metric": metric,
                                "nfe": nfe,
                                "arm": arm,
                                "value": base + offset,
                            }
                        )
        seed_rows, summaries = collector.build_factorial_tables(endpoints)
        self.assertEqual(len(seed_rows), 12)
        self.assertEqual(len(summaries), 36)
        row = next(
            item
            for item in seed_rows
            if item["metric"] == "fid50k_full" and item["nfe"] == 1 and item["training_seed"] == 3
        )
        self.assertEqual(row["target_at_baseline_weight"], -3)
        self.assertEqual(row["target_at_g_weight"], -3)
        self.assertEqual(row["weight_at_baseline_target"], -1)
        self.assertEqual(row["weight_at_g_target"], -1)
        self.assertEqual(row["target_x_weight"], 0)
        summary = next(
            item
            for item in summaries
            if item["metric"] == "fid50k_full" and item["nfe"] == 1 and item["quantity"] == "A"
        )
        self.assertEqual(summary["independent_n"], 3)
        self.assertEqual(summary["mean"], 5.0)
        self.assertEqual(summary["median"], 5.0)
        self.assertEqual(summary["minimum"], 4.0)
        self.assertEqual(summary["maximum"], 6.0)
        self.assertEqual(summary["range"], 2.0)

    def test_factorial_summary_fails_closed_on_missing_arm(self):
        endpoints = [
            {"training_seed": seed, "metric": metric, "nfe": nfe, "arm": arm, "value": 1.0}
            for seed in evaluator.SEEDS
            for metric in evaluator.METRICS
            for nfe in evaluator.NFE_SETTINGS
            for arm in evaluator.ARMS
        ]
        endpoints.pop()
        with self.assertRaisesRegex(collector.CollectionError, "incomplete"):
            collector.build_factorial_tables(endpoints)

    def test_block_summary_never_increases_independent_n(self):
        rows = []
        for seed in evaluator.SEEDS:
            for arm in evaluator.ARMS:
                for nfe in evaluator.NFE_SETTINGS:
                    for block in range(evaluator.BLOCK_COUNT):
                        rows.append(
                            {
                                "training_seed": seed,
                                "arm": arm,
                                "nfe": nfe,
                                "block_index": block,
                                "feature_mean_l2_distance_from_full": float(block),
                                "feature_variance_trace": float(block + 1),
                            }
                        )
        summaries = collector.summarize_block_variation(rows)
        self.assertEqual(len(summaries), 24)
        self.assertTrue(all(row["independent_n_contribution"] == 0 for row in summaries))
        self.assertTrue(all(row["quality_endpoint"] is False for row in summaries))

    def make_evaluation_root(
        self, root: Path, cells: list[dict]
    ) -> tuple[Path, Path]:
        eval_root = root / "evaluation"
        for name in ("receipts", "manifests", "process_logs", "jobs"):
            (eval_root / name).mkdir(parents=True, exist_ok=True)
        cache_root = eval_root / "evaluator_cache"
        cache_root.mkdir()
        cache_file = cache_root / "canonical-reference.pkl"
        cache_file.write_bytes(b"canonical-reference-cache")
        cache_artifacts = {cache_file.name: binding(cache_file)}
        cache_record = {
            "root": str(cache_root.resolve()),
            "artifact_count": 1,
            "artifacts": cache_artifacts,
            "tree_sha256": evaluator.canonical_sha256(cache_artifacts),
            "inception_detector_url": evaluator.INCEPTION_URL,
        }
        dataset = root / "canonical.zip"
        dataset.write_bytes(b"fixture-dataset")
        jobs = evaluator.build_jobs(cells, eval_root, 31800)
        revalidation_python = str(Path(sys.executable).resolve())
        training_arm_revalidation = []
        for cell in cells:
            run_dir = Path(cell["run_dir"]).resolve()
            validation = json.loads(
                (run_dir / evaluator.VALIDATION_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            training_arm_revalidation.append(
                {
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "status": "PASS",
                    "command_argv": [
                        revalidation_python,
                        str(
                            evaluator.REPO_ROOT
                            / "scripts"
                            / "verify_q256_target_weight_arm.py"
                        ),
                        "--run-dir",
                        str(run_dir),
                        "--arm",
                        cell["arm"],
                        "--seed",
                        str(cell["seed"]),
                        "--mode",
                        "formal",
                        "--check-only",
                    ],
                    "report_sha256": evaluator.canonical_sha256(validation),
                }
            )
        plan = {
            "schema": evaluator.PLAN_SCHEMA,
            "protocol": evaluator.PROTOCOL,
            "experiment_id": evaluator.EXPERIMENT_ID,
            "status": "authorized_exact_matrix",
            "selection_policy": "all_12_final_256kimg_checkpoints_no_intermediate_selection",
            "training_matrix": {
                "preregistration_sha256": evaluator.sha256_file(
                    evaluator.REPO_ROOT
                    / "analysis"
                    / "q256_target_weight_factorial"
                    / "preregistration.json"
                ),
                "expected_amp_skip_attempts": None,
            },
            "training_cells": cells,
            "training_arm_revalidation": training_arm_revalidation,
            "dataset": {
                "path": str(dataset.resolve()),
                "sha256": evaluator.DATASET_SHA256,
                "bytes": dataset.stat().st_size,
            },
            "evaluator_source": {"git_head": "4" * 40, "content_sha256": "5" * 64},
            "runtime": {"python_executable": revalidation_python},
            "gpu": {"uuid": "GPU-test"},
            "sample_count_per_job": evaluator.SAMPLE_COUNT,
            "sample_seed_range": evaluator.SAMPLE_SEEDS,
            "metric_seed": evaluator.METRIC_SEED,
            "metrics_per_job": list(evaluator.METRICS),
            "nfe_modes": {"1": [], "2": [0.821]},
            "job_count": 24,
            "jobs": jobs,
        }
        plan_path = eval_root / "evaluation_plan.json"
        dump_json(plan_path, plan)
        plan_sha = evaluator.sha256_file(plan_path)
        completed = []
        for job in jobs:
            job_id = job["job_id"]
            target = Path(job["output_directory"])
            target.mkdir(parents=True)
            dump_json(target / "training_options.json", {"synthetic": True})
            (target / "log.txt").write_text("Exiting...\n", encoding="utf-8")
            (target / "generated-features-kid50k_full-repeat00.npy").write_bytes(b"kid")
            (target / "generated-features-fid50k_full-repeat00.npy").write_bytes(b"fid")
            (target / "generated-samples.npy").write_bytes(b"samples")
            checkpoint = Path(job["checkpoint"])
            raw_metrics = []
            for metric in evaluator.METRICS:
                value = (
                    float(job["seed"] * 10 + evaluator.ARMS.index(job["arm"]) + job["nfe"])
                    / (1000 if metric.startswith("kid") else 1)
                )
                raw = target / f"metric-{metric}.jsonl"
                raw.write_text(
                    json.dumps(
                        {
                            "metric": metric,
                            "results": {metric: value},
                            "num_gpus": 1,
                            "snapshot_pkl": os.path.relpath(checkpoint, target),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raw_metrics.append(
                    {
                        "metric": metric,
                        "value": value,
                        "raw_path": str(raw),
                        "raw_sha256": evaluator.sha256_file(raw),
                    }
                )
            diagnostic_path = target / "sampling_block_diagnostics_v1.json"
            dump_json(
                diagnostic_path,
                {
                    "schema": evaluator.BLOCK_SCHEMA,
                    "status": "descriptive_variation_only",
                    "sample_seed_range": evaluator.SAMPLE_SEEDS,
                    "sample_count": evaluator.SAMPLE_COUNT,
                    "fixed_block_size": evaluator.BLOCK_SIZE,
                    "fixed_block_count": evaluator.BLOCK_COUNT,
                    "independent_training_replicate_contribution": 0,
                    "quality_endpoint": False,
                    "selection_criterion": False,
                    "blocks": [
                        {
                            "block_index": block,
                            "sample_seed_start": block * evaluator.BLOCK_SIZE,
                            "sample_seed_end": (block + 1) * evaluator.BLOCK_SIZE - 1,
                            "sample_count": evaluator.BLOCK_SIZE,
                            "feature_mean_l2_distance_from_full": float(block + 1),
                            "feature_variance_trace": float(block + 2),
                        }
                        for block in range(evaluator.BLOCK_COUNT)
                    ],
                },
            )
            artifacts = {
                path.name: binding(path)
                for path in target.iterdir()
                if path.is_file()
            }
            launch_path = eval_root / "manifests" / f"{job_id}.json"
            dump_json(
                launch_path,
                {
                    "schema": evaluator.JOB_LAUNCH_SCHEMA,
                    "evaluation_plan_sha256": plan_sha,
                    "job": job,
                    "gpu": plan["gpu"],
                    "gpu_exclusivity_monitor_contract": {
                        "schema": evaluator.training_launcher.GPU_MONITOR_SCHEMA,
                        "gpu_uuid": "GPU-test",
                        "poll_interval_seconds": 1.0,
                        "fail_closed": True,
                    },
                },
            )
            process_log = eval_root / "process_logs" / f"{job_id}.log"
            process_log.write_text("fixture\n", encoding="utf-8")
            monitor = {
                "schema": evaluator.training_launcher.GPU_MONITOR_SCHEMA,
                "status": "PASS",
                "gpu_uuid": "GPU-test",
                "root_process_pid": 123,
                "poll_interval_seconds": 1.0,
                "cadence_grace_seconds": 0.25,
                "probe_timeout_seconds": 0.4,
                "started_utc": "2026-08-19T00:00:00Z",
                "finished_utc": "2026-08-19T00:00:01Z",
                "first_check_started_utc": "2026-08-19T00:00:00Z",
                "last_check_started_utc": "2026-08-19T00:00:01Z",
                "checks_completed": 2,
                "first_check_offset_seconds": 0.0,
                "last_check_offset_seconds": 1.0,
                "monitor_duration_seconds": 1.0,
                "max_observed_poll_gap_seconds": 1.0,
                "max_observed_check_duration_seconds": 0.1,
                "max_observed_schedule_lateness_seconds": 0.01,
                "foreign_process_incident": None,
                "own_process_group_signals": [],
            }
            post_idle = {
                "checked_utc": "2026-08-19T00:00:01Z",
                "gpu_uuid": "GPU-test",
                "compute_process_count": 0,
                "query": "gpu_uuid,pid,process_name,used_gpu_memory",
            }
            dump_json(
                eval_root / "receipts" / f"{job_id}.json",
                {
                    "schema": evaluator.JOB_RECEIPT_SCHEMA,
                    "protocol": evaluator.PROTOCOL,
                    "status": "passed",
                    "job_id": job_id,
                    "seed": job["seed"],
                    "arm": job["arm"],
                    "nfe": job["nfe"],
                    "mid_t": job["mid_t"],
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "dataset_sha256": evaluator.DATASET_SHA256,
                    "evaluator_source_git_head": "4" * 40,
                    "evaluator_source_content_sha256": "5" * 64,
                    "sample_count": evaluator.SAMPLE_COUNT,
                    "sample_seed_range": evaluator.SAMPLE_SEEDS,
                    "metric_seed": evaluator.METRIC_SEED,
                    "precision": "fp32",
                    "returncode": 0,
                    "execution_error": None,
                    "gpu_exclusivity_monitor": monitor,
                    "post_job_gpu_idle_check": post_idle,
                    "launch_manifest": str(launch_path),
                    "launch_manifest_sha256": evaluator.sha256_file(launch_path),
                    "process_log": str(process_log),
                    "process_log_sha256": evaluator.sha256_file(process_log),
                    "artifacts": artifacts,
                    "artifacts_tree_sha256": evaluator.canonical_sha256(artifacts),
                    "metrics": raw_metrics,
                    "sampling_block_diagnostics": {
                        "path": str(diagnostic_path),
                        "sha256": evaluator.sha256_file(diagnostic_path),
                    },
                    "cache": cache_record,
                },
            )
            completed.append(job_id)
        dump_json(
            eval_root / "evaluation_completion.json",
            {
                "schema": evaluator.COMPLETION_SCHEMA,
                "protocol": evaluator.PROTOCOL,
                "status": "PASS",
                "job_count": 24,
                "evaluation_plan_sha256": plan_sha,
                "completed_job_ids": completed,
                "cache_tree_sha256": cache_record["tree_sha256"],
            },
        )
        return eval_root, dataset

    def test_full_collector_validates_12_runs_24_jobs_and_48_raw_values(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _matrix, cells = self.make_training_matrix(root)
            eval_root, dataset = self.make_evaluation_root(root, cells)
            with mock.patch.object(
                evaluator,
                "verify_dataset",
                return_value={
                    "path": str(dataset.resolve()),
                    "sha256": evaluator.DATASET_SHA256,
                    "bytes": dataset.stat().st_size,
                },
            ), mock.patch.object(
                evaluator,
                "source_snapshot",
                return_value={"git_head": "4" * 40, "content_sha256": "5" * 64},
            ), mock.patch.object(evaluator, "validate_evaluation_options"):
                endpoints, blocks, provenance = collector.validate_and_collect(eval_root)
            result = collector.build_result(endpoints, blocks, provenance)
            self.assertEqual(len(endpoints), 48)
            self.assertEqual(len(blocks), 240)
            self.assertEqual(len(result["seed_level_factorial"]), 12)
            self.assertEqual(len(result["cross_seed_summaries"]), 36)
            self.assertEqual(result["independent_unit"]["n"], 3)
            self.assertEqual(
                result["sampling_block_variation"]["independent_n_contribution"], 0
            )
            self.assertFalse(result["reporting_boundaries"]["automatic_interpretation_branch_selection"])

            outdir = root / "summary"
            collector.write_outputs(outdir, result, blocks)
            self.assertTrue((outdir / "endpoint_values.csv").is_file())
            self.assertTrue((outdir / "seed_level_factorial.csv").is_file())
            self.assertTrue((outdir / "cross_seed_summary.csv").is_file())
            self.assertTrue((outdir / "collection_receipt.json").is_file())
            with self.assertRaisesRegex(collector.CollectionError, "reuse"):
                collector.write_outputs(outdir, result, blocks)

    def test_collector_rejects_missing_raw_metric_even_with_pass_completion(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _matrix, cells = self.make_training_matrix(root)
            eval_root, dataset = self.make_evaluation_root(root, cells)
            missing = next((eval_root / "jobs").glob("*/metric-fid50k_full.jsonl"))
            missing.unlink()
            with mock.patch.object(
                evaluator,
                "verify_dataset",
                return_value={
                    "path": str(dataset.resolve()),
                    "sha256": evaluator.DATASET_SHA256,
                    "bytes": dataset.stat().st_size,
                },
            ), mock.patch.object(
                evaluator,
                "source_snapshot",
                return_value={"git_head": "4" * 40, "content_sha256": "5" * 64},
            ), mock.patch.object(evaluator, "validate_evaluation_options"):
                with self.assertRaisesRegex(collector.CollectionError, "missing"):
                    collector.validate_and_collect(eval_root)

    def test_collector_rejects_missing_arm_revalidation_record(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _matrix, cells = self.make_training_matrix(root)
            eval_root, dataset = self.make_evaluation_root(root, cells)
            plan_path = eval_root / "evaluation_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["training_arm_revalidation"].pop()
            dump_json(plan_path, plan)
            completion_path = eval_root / "evaluation_completion.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["evaluation_plan_sha256"] = evaluator.sha256_file(plan_path)
            dump_json(completion_path, completion)
            with mock.patch.object(
                evaluator,
                "verify_dataset",
                return_value={
                    "path": str(dataset.resolve()),
                    "sha256": evaluator.DATASET_SHA256,
                    "bytes": dataset.stat().st_size,
                },
            ), mock.patch.object(
                evaluator,
                "source_snapshot",
                return_value={"git_head": "4" * 40, "content_sha256": "5" * 64},
            ):
                with self.assertRaisesRegex(
                    collector.CollectionError, "12 fresh arm-revalidation"
                ):
                    collector.validate_and_collect(eval_root)

    def test_collector_rejects_missing_gpu_monitor_evidence(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _matrix, cells = self.make_training_matrix(root)
            eval_root, dataset = self.make_evaluation_root(root, cells)
            receipt_path = next((eval_root / "receipts").glob("*.json"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("gpu_exclusivity_monitor")
            dump_json(receipt_path, receipt)
            with mock.patch.object(
                evaluator,
                "verify_dataset",
                return_value={
                    "path": str(dataset.resolve()),
                    "sha256": evaluator.DATASET_SHA256,
                    "bytes": dataset.stat().st_size,
                },
            ), mock.patch.object(
                evaluator,
                "source_snapshot",
                return_value={"git_head": "4" * 40, "content_sha256": "5" * 64},
            ), mock.patch.object(evaluator, "validate_evaluation_options"):
                with self.assertRaisesRegex(
                    collector.CollectionError,
                    "GPU evidence failed",
                ):
                    collector.validate_and_collect(eval_root)


if __name__ == "__main__":
    unittest.main()
