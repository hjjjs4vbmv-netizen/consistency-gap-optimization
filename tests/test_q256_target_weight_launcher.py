import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_q256_target_weight_matrix as launcher


class TargetWeightLauncherTest(unittest.TestCase):
    def _write_planned_pause_fixture(self, run_dir: Path, attempts: int = 16):
        launch_manifest = {
            "gate_control": {
                "kind": "planned_exact_resume_pause",
                "stop_after_attempts": attempts,
                "scientific_training_contract_unchanged": True,
            },
            "training": launcher.training_contract("smoke", "A", 3),
            "gpu": {"uuid": "GPU-test"},
        }
        (run_dir / "launch_manifest.json").write_text(
            json.dumps(launch_manifest), encoding="utf-8"
        )
        (run_dir / "training_options.json").write_text(
            json.dumps(
                {
                    "stop_after_attempts": attempts,
                    "total_kimg": 4,
                    "resume_state_dump": None,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "initial_state_receipt_v1.json").write_text(
            json.dumps(
                {
                    "attempted_iteration": 0,
                    "processed_nimg": 0,
                    "seed": 3,
                    "factorial": {"arm": "A"},
                }
            ),
            encoding="utf-8",
        )
        digest = "a" * 64
        telemetry_rows = []
        summary_rows = []
        for attempt in range(1, attempts + 1):
            row = {field: "0" for field in launcher.FACTORIAL_TELEMETRY_FIELDS}
            row.update(
                {
                    "schema": "ect.q256.target-weight-training-telemetry/v1",
                    "protocol": launcher.FACTORIAL_PROTOCOL,
                    "arm": "A",
                    "target_gap_scale": "1.0",
                    "denominator_gap_scale": "1.0",
                    "attempted_iteration": str(attempt),
                    "successful_optimizer_steps": str(attempt),
                    "processed_nimg": str(attempt * 128),
                    "processed_kimg": str(attempt * 128 / 1000),
                    "stage": "0",
                    "loss": "1.0",
                    "raw_grad_norm": "1.0",
                    "raw_grad_finite_norm": "1.0",
                    "sanitized_grad_norm": "1.0",
                    "update_norm": "1.0",
                    "model_norm": "1.0",
                    "ema_norm": "1.0",
                    "sample_count": "128",
                    "target_delta_min": "0.1",
                    "target_delta_max": "0.2",
                    "target_delta_mean": "0.15",
                    "denominator_delta_min": "0.1",
                    "denominator_delta_max": "0.2",
                    "denominator_delta_mean": "0.15",
                    "learning_rate": "0.0001",
                    "grad_scale_before": "65536",
                    "grad_scale_after": "65536",
                    "step_skipped": "0",
                    "elapsed_sec": str(attempt),
                    "gpu_hours_cumulative": str(attempt / 3600),
                }
            )
            for field in launcher.FACTORIAL_DIGEST_FIELDS:
                row[field] = digest
            telemetry_rows.append(row)

            summary = {field: "0" for field in launcher.TRAIN_SUMMARY_FIELDS}
            summary.update(
                {
                    "attempted_iteration": str(attempt),
                    "successful_optimizer_steps": str(attempt),
                    "processed_nimg": str(attempt * 128),
                    "processed_kimg": str(attempt * 128 / 1000),
                    "loss": "1.0",
                    "grad_scale": "65536",
                    "step_skipped": "0",
                    "schedule": "sigmoid",
                    "stage": "0",
                    "next_loop_cur_tick": "1",
                    "elapsed_sec": str(attempt),
                    "peak_vram_gb": "1.0",
                }
            )
            summary_rows.append(summary)
        with (run_dir / "factorial_training_telemetry_v1.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=launcher.FACTORIAL_TELEMETRY_FIELDS
            )
            writer.writeheader()
            writer.writerows(telemetry_rows)
        with (run_dir / "train_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=launcher.TRAIN_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
        (run_dir / "network-snapshot-latest.pkl").write_bytes(b"snapshot")
        (run_dir / "training-state-latest.pt").write_bytes(b"state")
        (run_dir / "log.txt").write_text(
            f"Planned pause after {attempts} attempts; exiting.\n", encoding="utf-8"
        )
        (run_dir / "runner.log").write_text("clean launcher output\n", encoding="utf-8")
        return launch_manifest

    def _authorization_scope(self):
        file_hashes = {
            relative: hashlib.sha256(f"fixture:{relative}".encode("utf-8")).hexdigest()
            for relative in launcher.ROLE_E_AB_PARITY_SOURCE_FILES
        }
        source = {
            "git_head": "1" * 40,
            "git_tree": "7" * 40,
            "git_branch": launcher.EXPECTED_BRANCH,
            "git_clean": True,
            "content_sha256": "2" * 64,
            "files": [
                {"path": relative, "sha256": digest, "size_bytes": 1}
                for relative, digest in sorted(file_hashes.items())
            ],
        }
        runtime = {
            "python_version": "3.10.12 fixture",
            "platform": "fixture-linux",
            "torch_version": launcher.EXPECTED_TORCH_VERSION,
            "torch_cuda_version": launcher.EXPECTED_TORCH_CUDA_VERSION,
            "torch_cudnn_version": 8900,
            "cuda_device_count": 1,
            "visible_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "visible_gpu_name_nvidia_smi": "NVIDIA A100-SXM4-80GB",
            "visible_gpu_memory_mib_nvidia_smi": 81920,
            "visible_device_compute_capability": [8, 0],
            "software_sha256": "3" * 64,
            "critical_runtime_files": {
                "python": {"path": "/runtime/python", "sha256": "4" * 64}
            },
        }
        sandbox = {
            "sandbox_tree_metadata_sha256": "5" * 64,
            "critical_files_sha256": "6" * 64,
        }
        dataset = {"sha256": launcher.EXPECTED_DATASET_SHA256}
        transfer = {"sha256": launcher.EXPECTED_TRANSFER_SHA256}
        return source, runtime, sandbox, dataset, transfer, file_hashes

    def _role_e_gate_payload(self, source, runtime, file_hashes, evidence_root):
        manifest = "".join(
            f"{relative}\t{file_hashes[relative]}\n"
            for relative in sorted(file_hashes)
        )
        role_runtime = launcher.role_e_gate_runtime_scope(runtime)
        log = evidence_root / "pytest.log"
        junit = evidence_root / "pytest.xml"
        log.write_text("all frozen tests passed\n", encoding="utf-8")
        cases = "".join(
            f'<testcase classname="{classname}" name="{name}" />'
            for classname, name in launcher.ROLE_E_REQUIRED_TEST_CASES
        )
        junit.write_text(
            f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>',
            encoding="utf-8",
        )
        junit_summary = launcher._parse_role_e_junit(junit, "fixture")
        return {
            "schema": launcher.ROLE_E_AB_PARITY_SCHEMA,
            "status": "PASS",
            "source": {
                "commit": source["git_head"],
                "tree": source["git_tree"],
                "branch": source["git_branch"],
                "clean": True,
                "files": file_hashes,
                "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
                "launcher_content_sha256": source["content_sha256"],
            },
            "runtime": {
                **role_runtime,
                "cuda_visible_devices": role_runtime["gpu_uuid"],
            },
            "executed_git_archive_sha256": "8" * 64,
            "pytest_exit_code": 0,
            "junit": junit_summary,
            "evidence": {
                "pytest_log": str(log.resolve()),
                "pytest_log_sha256": launcher.sha256_file(log),
                "pytest_junit": str(junit.resolve()),
                "pytest_junit_sha256": launcher.sha256_file(junit),
            },
            "assertion_contract": {
                "required_test_cases": junit_summary["required_test_cases"],
                "all_collected_tests_passed_without_skip": True,
            },
        }

    def _authorization_payload(
        self, *, phase, source, runtime, sandbox, dataset, transfer, gates
    ):
        return {
            "schema": launcher.AUTHORIZATION_SCHEMA,
            "experiment_id": launcher.EXPERIMENT_ID,
            "phase": phase,
            "status": "authorized",
            "gates_status": "PASS",
            "authorization_id": f"{phase}-review-1",
            "authorized_by": "role-e-reviewer",
            "issued_at_utc": "2026-08-19T00:00:00Z",
            "allowed_arms": list(launcher.ARMS),
            "allowed_seeds": list(launcher.PHASES[phase]["seeds"]),
            "source_git_head": source["git_head"],
            "source_content_sha256": source["content_sha256"],
            "preregistration_sha256": launcher.preregistration_record()["sha256"],
            "dataset_sha256": dataset["sha256"],
            "transfer_sha256": transfer["sha256"],
            "expected_amp_skip_attempts": None,
            "amp_skip_policy": launcher.AMP_SKIP_POLICY,
            "role_e_gate_runtime": launcher.role_e_gate_runtime_scope(runtime),
            "runtime_sandbox_tree_metadata_sha256": sandbox[
                "sandbox_tree_metadata_sha256"
            ],
            "runtime_sandbox_critical_files_sha256": sandbox[
                "critical_files_sha256"
            ],
            "runtime_software_sha256": runtime["software_sha256"],
            "gate_receipts": gates,
        }

    def _write_smoke_arm_passes(self, root: Path, source):
        bindings = {}
        for arm in launcher.ARMS:
            run_dir = (root / f"smoke-{arm}").resolve()
            run_dir.mkdir()
            artifact = run_dir / "training-state-latest.pt"
            artifact.write_bytes(f"trusted-{arm}".encode("utf-8"))
            validation = {
                "schema": launcher.VALIDATION_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir),
                "mode": "smoke",
                "arm": arm,
                "seed": 3,
                "amp_skip_attempts": [2, 7],
                "amp_skip_signature_expected_value_enforced": False,
                "amp_skip_policy": launcher.AMP_SKIP_POLICY,
                "initial_common_state_sha256": "9" * 64,
                "source_git_head": source["git_head"],
                "source_content_sha256": source["content_sha256"],
            }
            validation_path = run_dir / launcher.VALIDATION_FILENAME
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            launch_manifest_path = run_dir / "launch_manifest.json"
            launch_manifest_path.write_text(
                json.dumps(
                    {
                        "run_directory": str(run_dir),
                        "gpu": {"uuid": "GPU-test"},
                        "training": {"phase": "smoke", "arm": arm, "seed": 3},
                    }
                ),
                encoding="utf-8",
            )
            for name in launcher.CORE_ARM_ARTIFACTS:
                path = run_dir / name
                if not path.exists():
                    path.write_bytes(f"trusted-{arm}-{name}".encode("utf-8"))
            artifact_paths = {
                name: run_dir / name for name in launcher.CORE_ARM_ARTIFACTS
            }
            artifact_paths[launcher.VALIDATION_FILENAME] = validation_path
            hashes = {
                "schema": launcher.HASH_RECEIPT_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir),
                "mode": "smoke",
                "arm": arm,
                "seed": 3,
                "artifacts": {
                    name: {
                        "bytes": path.stat().st_size,
                        "sha256": launcher.sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                },
            }
            hashes_path = run_dir / launcher.HASH_RECEIPT_FILENAME
            hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
            runner_log_path = run_dir / "runner.log"
            runner_log_path.write_text("runner PASS\n", encoding="utf-8")
            verifier_log_path = run_dir / "arm_verifier.log"
            verifier_log_path.write_text("verifier PASS\n", encoding="utf-8")
            monitor = {
                "schema": launcher.GPU_MONITOR_SCHEMA,
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
            runner_completion = {
                "schema": launcher.RUNNER_COMPLETION_SCHEMA,
                "experiment_id": launcher.EXPERIMENT_ID,
                "started_utc": "2026-08-19T00:00:00Z",
                "finished_utc": "2026-08-19T00:00:02Z",
                "status": "PASS",
                "returncode": 0,
                "verifier_returncode": 0,
                "launch_manifest": launch_manifest_path.name,
                "launch_manifest_sha256": launcher.sha256_file(
                    launch_manifest_path
                ),
                "runner_log": runner_log_path.name,
                "runner_log_sha256": launcher.sha256_file(runner_log_path),
                "verifier_log": verifier_log_path.name,
                "verifier_log_sha256": launcher.sha256_file(verifier_log_path),
                "training_gpu_exclusivity_monitor": monitor,
                "verifier_gpu_exclusivity_monitor": monitor,
                "final_prelaunch_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:00Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "post_training_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:01Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "post_verifier_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:02Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "verification": {
                    "validation_receipt_sha256": launcher.sha256_file(
                        validation_path
                    ),
                    "artifact_hash_receipt_sha256": launcher.sha256_file(
                        hashes_path
                    ),
                },
            }
            runner_completion_path = run_dir / "runner_completion.json"
            runner_completion_path.write_text(
                json.dumps(runner_completion), encoding="utf-8"
            )
            bindings[arm] = {
                "run_dir": str(run_dir),
                "validation_receipt_sha256": launcher.sha256_file(validation_path),
                "artifact_hash_receipt_sha256": launcher.sha256_file(hashes_path),
                "runner_completion_path": str(runner_completion_path),
                "runner_completion_sha256": launcher.sha256_file(
                    runner_completion_path
                ),
            }
        return bindings

    def test_frozen_arms_phases_and_seeds(self):
        self.assertEqual(
            launcher.ARMS,
            {
                "A": {"target_gap_scale": "1.0", "denominator_gap_scale": "1.0"},
                "B": {"target_gap_scale": "1.1", "denominator_gap_scale": "1.1"},
                "C": {"target_gap_scale": "1.1", "denominator_gap_scale": "1.0"},
                "D": {"target_gap_scale": "1.0", "denominator_gap_scale": "1.1"},
            },
        )
        self.assertEqual(launcher.PHASES["smoke"]["seeds"], (3,))
        self.assertEqual(launcher.PHASES["smoke"]["duration_mimg"], "0.004096")
        self.assertEqual(launcher.PHASES["smoke"]["expected_processed_nimg"], 4096)
        self.assertEqual(launcher.PHASES["smoke"]["expected_attempts"], 32)
        self.assertEqual(launcher.PHASES["formal"]["seeds"], (3, 4, 5))
        self.assertEqual(launcher.PHASES["formal"]["duration_mimg"], "0.256")
        self.assertEqual(launcher.PHASES["formal"]["expected_processed_nimg"], 256000)
        self.assertEqual(launcher.PHASES["formal"]["expected_attempts"], 2000)
        self.assertEqual(
            launcher.DEFAULT_TRANSFER,
            Path("/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl"),
        )
        self.assertEqual(
            launcher.DEFAULT_RUNS_ROOT,
            Path("/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819"),
        )
        self.assertEqual(
            launcher.DEFAULT_RUNTIME_SANDBOX,
            Path("/data/temp/ect001-pytorch2401-sandbox"),
        )
        with self.assertRaisesRegex(launcher.LaunchError, "permits only seeds"):
            launcher.validate_arm_seed_phase("smoke", "A", 4)

    def test_exact_frozen_command_and_resume_exclusivity(self):
        common = dict(
            python_bin=Path("/env/python"),
            data=Path("/assets/data.zip"),
            transfer=Path("/assets/transfer.pkl"),
            outdir=Path("/runs/cell"),
            phase="formal",
            arm="C",
            seed=4,
        )
        fresh = launcher.build_training_command(**common)
        self.assertIn("--factorial-protocol=q256_target_weight_v1", fresh)
        self.assertIn("--target-gap-scale=1.1", fresh)
        self.assertIn("--denominator-gap-scale=1.0", fresh)
        self.assertIn("--mapping=sigmoid", fresh)
        self.assertIn("--global-gap-scale=1.0", fresh)
        self.assertIn("--duration=0.256", fresh)
        self.assertIn("--batch=128", fresh)
        self.assertIn("--batch-gpu=16", fresh)
        self.assertIn("--optim=RAdam", fresh)
        self.assertIn("--fp16=True", fresh)
        self.assertIn("--tf32=False", fresh)
        self.assertIn("--enable_amp=True", fresh)
        self.assertIn("--metrics=none", fresh)
        self.assertEqual([arg for arg in fresh if arg.startswith("--transfer=")], ["--transfer=/assets/transfer.pkl"])
        self.assertFalse(any(arg.startswith("--resume=") for arg in fresh))

        state = Path("/runs/cell/training-state-latest.pt")
        resumed = launcher.build_training_command(**common, resume=state)
        self.assertEqual([arg for arg in resumed if arg.startswith("--resume=")], [f"--resume={state}"])
        self.assertFalse(any(arg.startswith("--transfer=") for arg in resumed))
        self.assertFalse(any(arg.startswith("--stop-after-attempts=") for arg in resumed))

        smoke_common = {
            **common,
            "phase": "smoke",
            "arm": "A",
            "seed": 3,
        }
        paused = launcher.build_training_command(
            **smoke_common, stop_after_attempts=16
        )
        self.assertEqual(
            [arg for arg in paused if arg.startswith("--stop-after-attempts=")],
            ["--stop-after-attempts=16"],
        )
        uninterrupted_smoke = launcher.build_training_command(**smoke_common)
        self.assertEqual(
            [arg for arg in paused if not arg.startswith("--stop-after-attempts=")],
            uninterrupted_smoke,
        )
        with self.assertRaisesRegex(launcher.LaunchError, "smoke-only"):
            launcher.build_training_command(**common, stop_after_attempts=16)
        with self.assertRaisesRegex(launcher.LaunchError, "fresh-only"):
            launcher.build_training_command(
                **smoke_common, resume=state, stop_after_attempts=16
            )
        for invalid in (1, 15, 31):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                launcher.LaunchError, "frozen gate target 16"
            ):
                launcher.build_training_command(
                    **smoke_common, stop_after_attempts=invalid
                )

        container_prefix = [
            "/usr/bin/apptainer",
            "exec",
            "--nv",
            "/data/temp/ect001-pytorch2401-sandbox",
            "python",
        ]
        containerized = launcher.build_training_command(
            **common, runtime_command=container_prefix
        )
        self.assertEqual(containerized[:5], container_prefix)
        self.assertEqual(containerized[5], str(launcher.REPO_ROOT / "ct_train.py"))

        verifier = launcher.build_verifier_command(
            python_bin="python",
            run_dir=Path("/runs/cell"),
            phase="formal",
            arm="C",
            seed=4,
            expected_skip_attempts="[]",
            runtime_command=container_prefix,
        )
        self.assertEqual(verifier[:5], container_prefix)
        self.assertIn(str(launcher.REPO_ROOT / "scripts" / "verify_q256_target_weight_arm.py"), verifier)
        self.assertEqual(verifier[-2:], ["--expected-skip-attempts", "[]"])

    def test_matrix_is_exact_and_seed_stays_on_one_gpu(self):
        jobs = launcher.make_matrix_jobs(
            phase="formal",
            seed_gpu={3: "0", 4: "1", 5: "0"},
            runs_root=Path("/runs"),
            matrix_id="matrix-1",
            authorization_receipt=Path("/auth.json"),
            data=Path("/data.zip"),
            transfer=Path("/transfer.pkl"),
            python_bin=Path("/env/python"),
            lock_root=Path("/locks"),
            base_port=31000,
        )
        self.assertEqual(len(jobs), 12)
        self.assertEqual([(job.seed, job.arm) for job in jobs], [(seed, arm) for seed in (3, 4, 5) for arm in "ABCD"])
        self.assertEqual({job.gpu for job in jobs if job.seed == 3}, {"0"})
        self.assertEqual({job.gpu for job in jobs if job.seed == 4}, {"1"})
        self.assertEqual({job.gpu for job in jobs if job.seed == 5}, {"0"})
        self.assertEqual(len({job.master_port for job in jobs}), 12)
        smoke = launcher.make_matrix_jobs(
            phase="smoke",
            seed_gpu={3: "0"},
            runs_root=Path("/runs"),
            matrix_id="smoke-1",
            authorization_receipt=Path("/auth.json"),
            data=Path("/data.zip"),
            transfer=Path("/transfer.pkl"),
            python_bin=Path("/env/python"),
            lock_root=Path("/locks"),
            base_port=32000,
        )
        self.assertEqual([(job.seed, job.arm) for job in smoke], [(3, arm) for arm in "ABCD"])
        self.assertTrue(
            all(
                not any(
                    argument.startswith("--stop-after-attempts")
                    for argument in job.command
                )
                for job in smoke
            )
        )

    def test_seed_gpu_map_is_required_and_exact(self):
        self.assertEqual(
            launcher.parse_seed_gpu(["3=0", "4=1", "5=0"], "formal"),
            {3: "0", 4: "1", 5: "0"},
        )
        with self.assertRaisesRegex(launcher.LaunchError, "requires exact seed GPU map"):
            launcher.parse_seed_gpu(["3=0", "4=1"], "formal")
        with self.assertRaisesRegex(launcher.LaunchError, "duplicate"):
            launcher.parse_seed_gpu(["3=0", "3=1"], "smoke")

    def test_gate_option_is_arm_only_and_source_contract_is_complete(self):
        arm_args = launcher.make_parser().parse_args(
            [
                "arm",
                "--phase",
                "smoke",
                "--arm",
                "A",
                "--seed",
                "3",
                "--gpu",
                "0",
                "--master-port",
                "29801",
                "--stop-after-attempts",
                "16",
            ]
        )
        self.assertEqual(arm_args.stop_after_attempts, 16)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.make_parser().parse_args(
                [
                    "matrix",
                    "--phase",
                    "smoke",
                    "--seed-gpu",
                    "3=0",
                    "--stop-after-attempts",
                    "16",
                ]
            )
        self.assertIn(
            "scripts/verify_q256_target_weight_smoke_matrix.py",
            launcher._SOURCE_EXACT,
        )

        verifier_tree = ast.parse(
            (
                launcher.REPO_ROOT
                / "scripts"
                / "verify_q256_target_weight_arm.py"
            ).read_text(encoding="utf-8")
        )
        verifier_fields = None
        for node in verifier_tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "TELEMETRY_FIELDS"
                    for target in node.targets
                )
            ):
                verifier_fields = ast.literal_eval(node.value)
                break
        self.assertEqual(len(launcher.FACTORIAL_TELEMETRY_FIELDS), 52)
        self.assertEqual(verifier_fields, launcher.FACTORIAL_TELEMETRY_FIELDS)

    def test_source_snapshot_uses_git_1_8_compatible_branch_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tracked = sorted(launcher._SOURCE_EXACT)
            for relative in tracked:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"source:{relative}\n", encoding="utf-8")
            calls = []

            def fake_checked_output(arguments, *, cwd=None, env=None):
                calls.append(tuple(arguments))
                responses = {
                    ("git", "rev-parse", "--is-inside-work-tree"): "true",
                    (
                        "git",
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                    ): "",
                    (
                        "git",
                        "symbolic-ref",
                        "--quiet",
                        "--short",
                        "HEAD",
                    ): launcher.EXPECTED_BRANCH,
                    ("git", "rev-parse", "HEAD"): "1" * 40,
                    ("git", "rev-parse", "HEAD^{tree}"): "2" * 40,
                }
                return responses[tuple(arguments)]

            tracked_bytes = ("\0".join(tracked) + "\0").encode("utf-8")
            with mock.patch.object(
                launcher, "checked_output", side_effect=fake_checked_output
            ), mock.patch.object(
                launcher.subprocess, "check_output", return_value=tracked_bytes
            ):
                snapshot = launcher.source_snapshot(repo, require_clean=True)
            self.assertEqual(snapshot["git_branch"], launcher.EXPECTED_BRANCH)
            self.assertIn(
                ("git", "symbolic-ref", "--quiet", "--short", "HEAD"), calls
            )
            self.assertNotIn(("git", "branch", "--show-current"), calls)
            self.assertNotIn(
                ("git", "status", "--porcelain=v1", "--untracked-files=all"),
                calls,
            )

    def test_runtime_prefix_does_not_nest_apptainer_when_already_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp).resolve()
            with mock.patch.dict(
                os.environ, {launcher.IN_SANDBOX_ENV: "1"}, clear=False
            ), mock.patch.object(launcher.sys, "version_info", (3, 10, 12)):
                command, record = launcher.runtime_prefix(sandbox, "python")
            self.assertEqual(command, [launcher.sys.executable])
            self.assertEqual(record["invocation_mode"], "already_inside_runtime_sandbox")
            self.assertTrue(record["already_inside_runtime_sandbox"])
            self.assertEqual(record["bind_specs"], list(launcher.RUNTIME_BIND_SPECS))
            with mock.patch.dict(
                os.environ, {launcher.IN_SANDBOX_ENV: "1"}, clear=False
            ), mock.patch.object(launcher.sys, "version_info", (3, 9, 18)):
                with self.assertRaisesRegex(launcher.LaunchError, "3.10 or newer"):
                    launcher.runtime_prefix(sandbox, "python")

    def test_arm_shell_bootstraps_old_host_python_through_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            sandbox = root / "sandbox"
            sandbox.mkdir()
            capture = root / "apptainer-argv.txt"
            old_python = root / "old-python"
            old_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            old_python.chmod(0o755)
            apptainer = root / "apptainer"
            apptainer.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
                encoding="utf-8",
            )
            apptainer.chmod(0o755)
            environment = dict(os.environ)
            environment.pop(launcher.IN_SANDBOX_ENV, None)
            environment.update(
                {
                    "PATH": f"{root}:{environment['PATH']}",
                    "ECT_BOOTSTRAP_PYTHON": str(old_python),
                    "ECT_RUNTIME_SANDBOX": str(sandbox),
                    "CAPTURE_PATH": str(capture),
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(launcher.ARM_SCRIPT),
                    "--phase",
                    "smoke",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                argv[:10],
                [
                    "exec",
                    "--nv",
                    "--bind",
                    "/data/raw:/data/raw",
                    "--bind",
                    "/data/temp:/data/temp",
                    str(sandbox),
                    "env",
                    f"{launcher.IN_SANDBOX_ENV}=1",
                    "python",
                ],
            )
            self.assertEqual(
                argv[10], str(launcher.REPO_ROOT / "scripts" / "run_q256_target_weight_matrix.py")
            )
            self.assertEqual(argv[11:], ["arm", "--phase", "smoke"])

    def test_arm_shell_uses_canonical_python_when_already_inside_sandbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            capture = root / "python-argv.txt"
            canonical_python = root / "python"
            canonical_python.write_text(
                "#!/bin/sh\n"
                "if [ \"${1-}\" = '-c' ]; then exit 0; fi\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
                encoding="utf-8",
            )
            canonical_python.chmod(0o755)
            wrong_python = root / "python3"
            wrong_python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
            wrong_python.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{root}:{environment['PATH']}",
                    launcher.IN_SANDBOX_ENV: "1",
                    "ECT_BOOTSTRAP_PYTHON": str(wrong_python),
                    "CAPTURE_PATH": str(capture),
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(launcher.ARM_SCRIPT),
                    "--phase",
                    "smoke",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                argv[0],
                str(launcher.REPO_ROOT / "scripts" / "run_q256_target_weight_matrix.py"),
            )
            self.assertEqual(argv[1:], ["arm", "--phase", "smoke"])

    def test_planned_pause_is_deeply_checked_and_immutably_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            launch_manifest = self._write_planned_pause_fixture(run_dir, attempts=16)
            probe = {
                "status": "PASS",
                "net_tensors_checked": 1,
                "ema_tensors_checked": 1,
                "snapshot_ema_tensors_checked": 1,
                "component_sha256": {
                    name: str(index) * 64
                    for index, name in enumerate(
                        (
                            "net",
                            "ema",
                            "optimizer",
                            "gradscaler",
                            "rng",
                            "sampler",
                            "loss",
                            "trajectory",
                            "control_state",
                        ),
                        start=1,
                    )
                },
            }
            with mock.patch.object(
                launcher, "checked_output", return_value=json.dumps(probe)
            ) as state_probe:
                report = launcher.verify_planned_pause_run(
                    run_dir,
                    arm="A",
                    seed=3,
                    stop_after_attempts=16,
                    runtime_command=("apptainer", "exec", "sandbox", "python"),
                    process_env={},
                )
            self.assertEqual(report["attempted_iterations"], 16)
            self.assertEqual(report["processed_nimg"], 2048)
            probe_argv = state_probe.call_args.args[0]
            probe_code = probe_argv[probe_argv.index("-c") + 1]
            self.assertIn("tick_start_nimg", probe_code)
            completion = {
                "schema": launcher.RUNNER_COMPLETION_SCHEMA,
                "experiment_id": launcher.EXPERIMENT_ID,
                "started_utc": "2026-08-19T00:00:00Z",
                "finished_utc": "2026-08-19T00:01:00Z",
                "status": launcher.PLANNED_PAUSE_STATUS,
                "returncode": 0,
                "launch_manifest": "launch_manifest.json",
                "launch_manifest_sha256": launcher.sha256_file(
                    run_dir / "launch_manifest.json"
                ),
                "runner_log": "runner.log",
                "runner_log_sha256": launcher.sha256_file(run_dir / "runner.log"),
                "final_prelaunch_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:00Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "training_gpu_exclusivity_monitor": {
                    "schema": launcher.GPU_MONITOR_SCHEMA,
                    "status": "PASS",
                    "gpu_uuid": "GPU-test",
                    "root_process_pid": 123,
                    "poll_interval_seconds": 1.0,
                    "cadence_grace_seconds": 0.25,
                    "probe_timeout_seconds": 0.4,
                    "started_utc": "2026-08-19T00:00:00Z",
                    "finished_utc": "2026-08-19T00:00:59Z",
                    "first_check_started_utc": "2026-08-19T00:00:00Z",
                    "last_check_started_utc": "2026-08-19T00:00:58Z",
                    "checks_completed": 59,
                    "first_check_offset_seconds": 0.0,
                    "last_check_offset_seconds": 58.0,
                    "monitor_duration_seconds": 59.0,
                    "max_observed_poll_gap_seconds": 1.0,
                    "max_observed_check_duration_seconds": 0.1,
                    "max_observed_schedule_lateness_seconds": 0.01,
                    "foreign_process_incident": None,
                    "own_process_group_signals": [],
                },
                "post_training_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:01:00Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "planned_pause_verification": report,
                "full_arm_verifier_invoked": False,
                "resume_required": True,
            }
            (run_dir / "runner_completion.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            bound = launcher.validate_planned_pause_completion(
                run_dir,
                launch_manifest=launch_manifest,
                stop_after_attempts=16,
            )
            self.assertEqual(bound["status"], launcher.PLANNED_PAUSE_STATUS)
            (run_dir / "training-state-latest.pt").write_bytes(b"changed-state")
            with self.assertRaisesRegex(launcher.LaunchError, "changed"):
                launcher.validate_planned_pause_completion(
                    run_dir,
                    launch_manifest=launch_manifest,
                    stop_after_attempts=16,
                )

    def test_run_arm_planned_pause_never_invokes_full_arm_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            args = argparse.Namespace(
                phase="smoke",
                arm="A",
                seed=3,
                gpu="0",
                master_port=29802,
                stop_after_attempts=16,
                expected_skip_attempts=None,
                runs_root=root,
                data=root / "data.zip",
                transfer=root / "transfer.pkl",
                python_bin="python",
                runtime_sandbox=root / "sandbox",
                lock_root=root / "locks",
                authorization_receipt=root / "authorization.json",
                resume=None,
                outdir=root / "run",
            )
            source = {
                "git_head": "1" * 40,
                "git_branch": launcher.EXPECTED_BRANCH,
                "git_clean": True,
                "content_sha256": "2" * 64,
            }
            dataset = {
                "resolved_path": str(args.data),
                "sha256": launcher.EXPECTED_DATASET_SHA256,
            }
            transfer = {
                "resolved_path": str(args.transfer),
                "sha256": launcher.EXPECTED_TRANSFER_SHA256,
            }
            sandbox = {
                "sandbox_tree_metadata_sha256": "3" * 64,
                "critical_files_sha256": "4" * 64,
            }
            runtime = {"software_sha256": "5" * 64}
            gpu = {"uuid": "GPU-test", "physical_index": 0, "name": "A100"}
            process_env = {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29802",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }

            def fake_stream(
                _command,
                *,
                cwd,
                env,
                log_path,
                gpu_monitor_record=None,
                **_kwargs,
            ):
                self.assertEqual(cwd, launcher.REPO_ROOT)
                self.assertEqual(env, process_env)
                if gpu_monitor_record is not None:
                    gpu_monitor_record["status"] = "PASS"
                log_path.write_text("planned process returned zero\n", encoding="utf-8")
                return 0

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(launcher, "validate_runs_root", return_value=root)
                )
                stack.enter_context(
                    mock.patch.object(launcher, "storage_preflight", return_value={})
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "runtime_prefix",
                        return_value=(["apptainer", "python"], {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "runtime_sandbox_fingerprint",
                        return_value=sandbox,
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "source_snapshot", return_value=source)
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "verify_asset",
                        side_effect=lambda _path, _sha, label: (
                            dataset if "dataset" in label else transfer
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "build_process_environment",
                        return_value=process_env,
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "query_gpu", return_value=gpu)
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "gpu_lock",
                        return_value=contextlib.nullcontext({"lock": "held"}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "assert_gpu_idle", return_value={})
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "runtime_environment", return_value=runtime
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "validate_authorization", return_value={})
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "copy_authorization_into_run",
                        return_value={
                            "receipt_path": "authorization/authorization_receipt.json",
                            "receipt_sha256": "6" * 64,
                            "gate_receipts": [],
                        },
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "preregistration_record",
                        return_value={"path": "preregistration.json", "sha256": "7" * 64},
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "host_environment", return_value={})
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "verify_internal_authorization", return_value={}
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "assert_master_port_available")
                )
                stack.enter_context(
                    mock.patch.object(launcher, "stream_process", side_effect=fake_stream)
                )
                planned_check = stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "verify_planned_pause_run",
                        return_value={
                            "status": "PASS",
                            "stop_after_attempts": 16,
                            "attempted_iterations": 16,
                            "processed_nimg": 2048,
                        },
                    )
                )
                full_postcheck = stack.enter_context(
                    mock.patch.object(launcher, "verify_completed_run")
                )
                full_verifier = stack.enter_context(
                    mock.patch.object(launcher, "build_verifier_command")
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(launcher.run_arm(args), 0)
            planned_check.assert_called_once()
            full_postcheck.assert_not_called()
            full_verifier.assert_not_called()
            completion = json.loads(
                (args.outdir / "runner_completion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completion["status"], launcher.PLANNED_PAUSE_STATUS)
            self.assertFalse(completion["full_arm_verifier_invoked"])
            self.assertTrue(completion["resume_required"])

    def test_amp_skip_policy_observes_by_default_and_allows_prospective_signature(self):
        self.assertIsNone(launcher.parse_expected_skip_attempts(None, "formal"))
        self.assertEqual(launcher.parse_expected_skip_attempts("[]", "formal"), [])
        self.assertEqual(launcher.parse_expected_skip_attempts("", "smoke"), [])
        self.assertEqual(
            launcher.parse_expected_skip_attempts("1,2,10", "formal"),
            [1, 2, 10],
        )
        with self.assertRaisesRegex(launcher.LaunchError, "strictly increasing"):
            launcher.parse_expected_skip_attempts("2,1", "formal")
        with self.assertRaisesRegex(launcher.LaunchError, "within"):
            launcher.parse_expected_skip_attempts("33", "smoke")
        with self.assertRaisesRegex(launcher.LaunchError, "tick-0 warm-up"):
            launcher.parse_expected_skip_attempts("79", "formal")
        with self.assertRaisesRegex(launcher.LaunchError, "tick-0 warm-up"):
            launcher.validate_amp_skip_signature(
                [79], phase="formal", label="immutable verifier receipt"
            )
        command = launcher.build_verifier_command(
            python_bin="python",
            run_dir=Path("/run"),
            phase="smoke",
            arm="A",
            seed=3,
            expected_skip_attempts="[1]",
        )
        self.assertEqual(command[-2:], ["--expected-skip-attempts", "[1]"])
        observed = launcher.build_verifier_command(
            python_bin="python",
            run_dir=Path("/run"),
            phase="smoke",
            arm="A",
            seed=3,
        )
        self.assertNotIn("--expected-skip-attempts", observed)

    def test_in_run_gpu_monitor_stops_only_our_process_group_on_foreign_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {}
            foreign = {
                "pid": 999999,
                "process_name": "/foreign/train.py",
                "used_gpu_memory_mib": "2048",
            }
            with mock.patch.object(
                launcher,
                "query_gpu_compute_processes",
                return_value=[foreign],
            ), mock.patch.object(
                launcher,
                "process_tree_pids",
                side_effect=lambda root_pid, **_kwargs: {root_pid},
            ):
                returncode = launcher.stream_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('started', flush=True); time.sleep(30)",
                    ],
                    cwd=launcher.REPO_ROOT,
                    env=os.environ,
                    log_path=Path(tmp) / "monitored.log",
                    monitored_gpu_uuid="GPU-test",
                    gpu_monitor_record=record,
                    gpu_monitor_interval_seconds=0.01,
                )
            self.assertNotEqual(returncode, 0)
            self.assertEqual(record["status"], "EXCLUSIVITY_LOST")
            self.assertEqual(
                record["foreign_process_incident"]["foreign_processes"],
                [foreign],
            )
            self.assertEqual(record["own_process_group_signals"], ["SIGTERM"])

    def test_monitor_thread_start_failure_reaps_launched_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            launched = []
            real_popen = subprocess.Popen

            def capture_popen(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                launched.append(process)
                return process

            with mock.patch.object(
                launcher.subprocess, "Popen", side_effect=capture_popen
            ), mock.patch.object(
                launcher.threading.Thread,
                "start",
                side_effect=RuntimeError("thread start fault"),
            ):
                with self.assertRaisesRegex(RuntimeError, "thread start fault"):
                    launcher.stream_process(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=launcher.REPO_ROOT,
                        env=os.environ,
                        log_path=Path(tmp) / "thread-start-fault.log",
                        monitored_gpu_uuid="GPU-test",
                        gpu_monitor_record={},
                        gpu_monitor_interval_seconds=1.0,
                    )
            self.assertEqual(len(launched), 1)
            self.assertIsNotNone(launched[0].poll())

    @unittest.skipUnless(Path("/proc").is_dir(), "requires Linux /proc process audit")
    def test_group_drain_kills_term_ignoring_stdout_descendant(self):
        code = (
            "import os, signal, sys, time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    while True: time.sleep(1)\n"
            "print('ready', flush=True)\n"
            "while True: time.sleep(1)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        self.assertEqual(process.stdout.readline().strip(), b"ready")
        signals = launcher.stop_and_reap_process_group(
            process, label="term-ignoring-descendant-test"
        )
        self.assertEqual(signals, ["SIGTERM", "SIGKILL"])
        self.assertIsNotNone(process.poll())
        with self.assertRaises(ProcessLookupError):
            os.killpg(process.pid, 0)
        process.stdout.close()

    def test_matrix_binds_passing_in_run_gpu_monitor_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runner_completion.json"
            monitor = {
                "schema": launcher.GPU_MONITOR_SCHEMA,
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
                "checks_completed": 3,
                "first_check_offset_seconds": 0.0,
                "last_check_offset_seconds": 1.0,
                "monitor_duration_seconds": 1.0,
                "max_observed_poll_gap_seconds": 1.0,
                "max_observed_check_duration_seconds": 0.1,
                "max_observed_schedule_lateness_seconds": 0.01,
                "foreign_process_incident": None,
                "own_process_group_signals": [],
            }
            payload = {
                "schema": launcher.RUNNER_COMPLETION_SCHEMA,
                "experiment_id": launcher.EXPERIMENT_ID,
                "started_utc": "2026-08-19T00:00:00Z",
                "finished_utc": "2026-08-19T00:00:02Z",
                "status": "PASS",
                "returncode": 0,
                "verifier_returncode": 0,
                "launch_manifest": "launch_manifest.json",
                "launch_manifest_sha256": "3" * 64,
                "runner_log": "runner.log",
                "runner_log_sha256": "4" * 64,
                "verifier_log": "arm_verifier.log",
                "verifier_log_sha256": "5" * 64,
                "training_gpu_exclusivity_monitor": monitor,
                "verifier_gpu_exclusivity_monitor": monitor,
                "final_prelaunch_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:00Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "post_training_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:01Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "post_verifier_gpu_idle_check": {
                    "checked_utc": "2026-08-19T00:00:02Z",
                    "gpu_uuid": "GPU-test",
                    "compute_process_count": 0,
                    "query": "gpu_uuid,pid,process_name,used_gpu_memory",
                },
                "verification": {
                    "validation_receipt_sha256": "1" * 64,
                    "artifact_hash_receipt_sha256": "2" * 64,
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = launcher.validate_runner_completion_receipt(path)
            self.assertEqual(record["sha256"], launcher.sha256_file(path))
            payload["training_gpu_exclusivity_monitor"] = {
                **monitor,
                "status": "EXCLUSIVITY_LOST",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "training GPU monitor"):
                launcher.validate_runner_completion_receipt(path)

    def test_in_run_gpu_monitor_fails_closed_on_unexpected_audit_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {}
            with mock.patch.object(
                launcher,
                "query_gpu_compute_processes",
                side_effect=RuntimeError("audit probe crashed"),
            ):
                returncode = launcher.stream_process(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=launcher.REPO_ROOT,
                    env=os.environ,
                    log_path=Path(tmp) / "audit-failure.log",
                    monitored_gpu_uuid="GPU-test",
                    gpu_monitor_record=record,
                    gpu_monitor_interval_seconds=0.01,
                )
            self.assertNotEqual(returncode, 0)
            self.assertEqual(record["status"], "STOPPED_FOR_AUDIT")
            self.assertEqual(
                record["foreign_process_incident"]["kind"],
                "GPU_AUDIT_FAILED",
            )
            self.assertIn(
                "RuntimeError: audit probe crashed",
                record["foreign_process_incident"]["error"],
            )

    def test_in_run_gpu_monitor_fails_closed_when_cadence_is_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {}

            def slow_probe(*_args, **_kwargs):
                time.sleep(0.08)
                return []

            with mock.patch.object(
                launcher,
                "query_gpu_compute_processes",
                side_effect=slow_probe,
            ), mock.patch.object(
                launcher,
                "process_tree_pids",
                side_effect=lambda root_pid, **_kwargs: {root_pid},
            ):
                returncode = launcher.stream_process(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=launcher.REPO_ROOT,
                    env=os.environ,
                    log_path=Path(tmp) / "cadence-failure.log",
                    monitored_gpu_uuid="GPU-test",
                    gpu_monitor_record=record,
                    gpu_monitor_interval_seconds=0.01,
                )
            self.assertNotEqual(returncode, 0)
            self.assertEqual(record["status"], "STOPPED_FOR_AUDIT")
            self.assertEqual(
                record["foreign_process_incident"]["kind"],
                "GPU_MONITOR_CADENCE_MISSED",
            )
            self.assertEqual(record["own_process_group_signals"], ["SIGTERM"])

    def test_authorization_fails_closed_and_binds_gate_hash(self):
        source, runtime, sandbox, dataset, transfer, file_hashes = (
            self._authorization_scope()
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "gate.json"
            gate.write_text(
                json.dumps(self._role_e_gate_payload(source, runtime, file_hashes, root)),
                encoding="utf-8",
            )
            receipt = root / "auth.json"
            payload = self._authorization_payload(
                phase="smoke",
                source=source,
                runtime=runtime,
                sandbox=sandbox,
                dataset=dataset,
                transfer=transfer,
                gates=[
                    {
                        "name": "role_e_ab_parity",
                        "schema": launcher.ROLE_E_AB_PARITY_SCHEMA,
                        "status": "PASS",
                        "path": gate.name,
                        "sha256": launcher.sha256_file(gate),
                    }
                ],
            )
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            validated = launcher.validate_authorization(
                receipt,
                phase="smoke",
                arm="D",
                seed=3,
                source=source,
                dataset=dataset,
                transfer=transfer,
                runtime_sandbox=sandbox,
                runtime=runtime,
            )
            self.assertEqual(validated["sha256"], launcher.sha256_file(receipt))
            run_dir = root / "run"
            run_dir.mkdir()
            internal = launcher.copy_authorization_into_run(run_dir, validated)
            launch_manifest = {
                "authorization": internal,
                "source": source,
                "runtime": runtime,
                "runtime_sandbox": sandbox,
                "assets": {"dataset": dataset, "transfer": transfer},
                "post_training_verifier": {"expected_skip_attempts": None},
            }
            launcher.verify_internal_authorization(
                run_dir,
                launch_manifest,
                expected_phase="smoke",
            )
            with self.assertRaisesRegex(launcher.LaunchError, "phase mismatch"):
                launcher.verify_internal_authorization(
                    run_dir,
                    launch_manifest,
                    expected_phase="formal",
                )
            copied_log = run_dir / internal["gate_evidence"][0]["path"]
            copied_log.chmod(0o600)
            copied_log.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "hash-mismatched"):
                launcher.verify_internal_authorization(
                    run_dir,
                    launch_manifest,
                    expected_phase="smoke",
                )

            failed_gate = self._role_e_gate_payload(source, runtime, file_hashes, root)
            failed_gate["status"] = "FAIL"
            gate.write_text(json.dumps(failed_gate), encoding="utf-8")
            payload["gate_receipts"][0]["sha256"] = launcher.sha256_file(gate)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "status mismatch"):
                launcher.validate_authorization(
                    receipt,
                    phase="smoke",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

            gate.write_text(
                json.dumps(self._role_e_gate_payload(source, runtime, file_hashes, root)),
                encoding="utf-8",
            )
            payload["gate_receipts"][0]["sha256"] = launcher.sha256_file(gate)
            payload["status"] = "NOT_AUTHORIZED"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "status"):
                launcher.validate_authorization(
                    receipt,
                    phase="smoke",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

            payload["status"] = "authorized"
            stale_gate = self._role_e_gate_payload(source, runtime, file_hashes, root)
            stale_gate["source"]["commit"] = "9" * 40
            gate.write_text(json.dumps(stale_gate), encoding="utf-8")
            payload["gate_receipts"][0]["sha256"] = launcher.sha256_file(gate)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "stale Git commit"):
                launcher.validate_authorization(
                    receipt,
                    phase="smoke",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

            wrong_gpu_gate = self._role_e_gate_payload(source, runtime, file_hashes, root)
            wrong_gpu_gate["runtime"]["gpu_uuid"] = (
                "GPU-ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
            )
            wrong_gpu_gate["runtime"]["cuda_visible_devices"] = (
                wrong_gpu_gate["runtime"]["gpu_uuid"]
            )
            gate.write_text(json.dumps(wrong_gpu_gate), encoding="utf-8")
            payload["gate_receipts"][0]["sha256"] = launcher.sha256_file(gate)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "runtime field 'gpu_uuid'"):
                launcher.validate_authorization(
                    receipt,
                    phase="smoke",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

            valid_gate = self._role_e_gate_payload(
                source, runtime, file_hashes, root
            )
            gate.write_text(json.dumps(valid_gate), encoding="utf-8")
            payload["gate_receipts"][0]["sha256"] = launcher.sha256_file(gate)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            Path(valid_gate["evidence"]["pytest_log"]).write_text(
                "tampered external evidence\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(launcher.LaunchError, "pytest_log hash mismatch"):
                launcher.validate_authorization(
                    receipt,
                    phase="smoke",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

    def test_formal_authorization_requires_exact_role_e_gate_set(self):
        source, runtime, sandbox, dataset, transfer, file_hashes = (
            self._authorization_scope()
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parity = root / "parity.json"
            parity.write_text(
                json.dumps(self._role_e_gate_payload(source, runtime, file_hashes, root)),
                encoding="utf-8",
            )
            matrix = root / "matrix.json"
            smoke_bindings = self._write_smoke_arm_passes(root, source)
            resume_root = root / "exact-resume-pair"
            resume_root.mkdir()
            resumed_bindings = self._write_smoke_arm_passes(resume_root, source)
            uninterrupted = smoke_bindings["A"]
            resumed = resumed_bindings["A"]
            fake_provenance = {
                "status": "PASS",
                "provenance_sha256": "7" * 64,
            }
            matrix_payload = {
                "schema": launcher.SMOKE_MATRIX_VALIDATION_SCHEMA,
                "status": "passed",
                "mode": "smoke",
                "seed": 3,
                "arms": smoke_bindings,
                "source_git_head": source["git_head"],
                "source_content_sha256": source["content_sha256"],
                "amp_skip_attempts": [2, 7],
                "amp_skip_signature_expected_value_enforced": False,
                "amp_skip_policy": launcher.AMP_SKIP_POLICY,
                "initial_common_state_sha256": "9" * 64,
                "exact_resume": {
                    "status": "passed",
                    "arm": "A",
                    "uninterrupted_run_dir": uninterrupted["run_dir"],
                    "resumed_run_dir": resumed["run_dir"],
                    "uninterrupted_validation_receipt_sha256": uninterrupted[
                        "validation_receipt_sha256"
                    ],
                    "resumed_validation_receipt_sha256": resumed[
                        "validation_receipt_sha256"
                    ],
                    "uninterrupted_artifact_hash_receipt_sha256": uninterrupted[
                        "artifact_hash_receipt_sha256"
                    ],
                    "resumed_artifact_hash_receipt_sha256": resumed[
                        "artifact_hash_receipt_sha256"
                    ],
                    "uninterrupted_runner_completion_sha256": uninterrupted[
                        "runner_completion_sha256"
                    ],
                    "resumed_runner_completion_sha256": resumed[
                        "runner_completion_sha256"
                    ],
                    "provenance": fake_provenance,
                    "provenance_sha256": fake_provenance["provenance_sha256"],
                },
            }
            matrix.write_text(json.dumps(matrix_payload), encoding="utf-8")
            matrix_sha = launcher.sha256_file(matrix)
            gates = [
                {
                    "name": "role_e_ab_parity",
                    "schema": launcher.ROLE_E_AB_PARITY_SCHEMA,
                    "status": "PASS",
                    "path": parity.name,
                    "sha256": launcher.sha256_file(parity),
                },
                {
                    "name": "four_arm_smoke_matrix",
                    "schema": launcher.SMOKE_MATRIX_VALIDATION_SCHEMA,
                    "status": "passed",
                    "path": matrix.name,
                    "sha256": matrix_sha,
                },
                {
                    "name": "exact_resume",
                    "schema": launcher.SMOKE_MATRIX_VALIDATION_SCHEMA,
                    "status": "passed",
                    "path": matrix.name,
                    "sha256": matrix_sha,
                },
            ]
            receipt = root / "formal-auth.json"
            payload = self._authorization_payload(
                phase="formal",
                source=source,
                runtime=runtime,
                sandbox=sandbox,
                dataset=dataset,
                transfer=transfer,
                gates=gates,
            )
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            deep_revalidation = mock.patch.object(
                launcher,
                "deep_revalidate_existing_arm",
                return_value={"status": "PASS", "report_sha256": "8" * 64},
            )
            deep_revalidation.start()
            self.addCleanup(deep_revalidation.stop)
            matrix_revalidation = mock.patch.object(
                launcher,
                "deep_revalidate_smoke_matrix",
                return_value={"status": "PASS", "report_sha256": "9" * 64},
            )
            matrix_revalidation.start()
            self.addCleanup(matrix_revalidation.stop)
            exact_provenance = mock.patch.object(
                launcher,
                "validate_exact_resume_provenance",
                return_value=fake_provenance,
            )
            exact_provenance.start()
            self.addCleanup(exact_provenance.stop)
            validated = launcher.validate_authorization(
                receipt,
                phase="formal",
                arm="C",
                seed=5,
                source=source,
                dataset=dataset,
                transfer=transfer,
                runtime_sandbox=sandbox,
                runtime=runtime,
            )
            self.assertEqual(len(validated["validated_gate_receipts"]), 3)
            self.assertEqual(
                validated["validated_gate_receipts"][1]["source_path"],
                validated["validated_gate_receipts"][2]["source_path"],
            )

            bound_artifact = root / "smoke-B" / "training-state-latest.pt"
            original_artifact = bound_artifact.read_bytes()
            bound_artifact.write_bytes(b"post-receipt mutation")
            with self.assertRaisesRegex(launcher.LaunchError, "changed"):
                launcher.validate_authorization(
                    receipt,
                    phase="formal",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )
            bound_artifact.write_bytes(original_artifact)

            matrix_payload["exact_resume"] = None
            matrix.write_text(json.dumps(matrix_payload), encoding="utf-8")
            for gate_item in payload["gate_receipts"][1:]:
                gate_item["sha256"] = launcher.sha256_file(matrix)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "exact_resume"):
                launcher.validate_authorization(
                    receipt,
                    phase="formal",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

            payload["gate_receipts"] = gates[:2]
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "exactly logical gates"):
                launcher.validate_authorization(
                    receipt,
                    phase="formal",
                    arm="A",
                    seed=3,
                    source=source,
                    dataset=dataset,
                    transfer=transfer,
                    runtime_sandbox=sandbox,
                    runtime=runtime,
                )

    def test_authorization_template_lists_phase_exact_gate_contract(self):
        source, runtime, sandbox, dataset, transfer, _ = self._authorization_scope()
        common = {
            "source": source,
            "dataset": dataset,
            "transfer": transfer,
            "runtime_sandbox": sandbox,
            "runtime": runtime,
        }
        smoke = launcher.authorization_template(phase="smoke", **common)
        formal = launcher.authorization_template(phase="formal", **common)
        self.assertEqual(
            [item["name"] for item in smoke["gate_receipts"]],
            ["role_e_ab_parity"],
        )
        self.assertEqual(
            [item["name"] for item in formal["gate_receipts"]],
            ["role_e_ab_parity", "four_arm_smoke_matrix", "exact_resume"],
        )
        for phase, template in (("smoke", smoke), ("formal", formal)):
            self.assertIsNone(template["expected_amp_skip_attempts"])
            self.assertEqual(template["amp_skip_policy"], launcher.AMP_SKIP_POLICY)
            self.assertEqual(
                template["role_e_gate_runtime"],
                launcher.role_e_gate_runtime_scope(runtime),
            )
            contract = launcher.AUTHORIZATION_GATE_CONTRACTS[phase]
            for item in template["gate_receipts"]:
                self.assertEqual(item["schema"], contract[item["name"]]["schema"])
                self.assertEqual(item["status"], contract[item["name"]]["status"])
                self.assertEqual(item["sha256"], "0" * 64)

    def test_resume_cells_never_create_fresh_replacements(self):
        resumes = launcher.parse_resume_cells(
            ["3:A=/runs/a/training-state-latest.pt", "3:D=/runs/d/training-state-000001.pt"],
            "smoke",
        )
        common = {
            "phase": "smoke",
            "seed_gpu": {3: "GPU-abc"},
            "runs_root": Path("/runs"),
            "matrix_id": "ignored-for-resume",
            "authorization_receipt": None,
            "data": Path("/data.zip"),
            "transfer": Path("/transfer.pkl"),
            "python_bin": Path("/env/python"),
            "lock_root": Path("/locks"),
            "base_port": 33000,
        }
        with self.assertRaisesRegex(launcher.LaunchError, "complete phase cell set"):
            launcher.make_matrix_jobs(**common, resume_cells=resumes)
        complete = {
            (3, arm): Path(f"/runs/{arm}/training-state-latest.pt")
            for arm in launcher.ARMS
        }
        jobs = launcher.make_matrix_jobs(**common, resume_cells=complete)
        self.assertEqual(
            [(job.seed, job.arm) for job in jobs],
            [(3, arm) for arm in launcher.ARMS],
        )
        self.assertTrue(all(job.resume is not None and job.outdir is None for job in jobs))
        self.assertTrue(all("--outdir" not in job.command for job in jobs))

    def test_existing_pass_receipts_are_hash_bound_for_matrix_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            validation = {
                "schema": launcher.VALIDATION_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir.resolve()),
                "mode": "smoke",
                "arm": "A",
                "seed": 3,
                "amp_skip_attempts": [2, 7],
                "amp_skip_signature_expected_value_enforced": False,
                "amp_skip_policy": launcher.AMP_SKIP_POLICY,
                "initial_common_state_sha256": "9" * 64,
            }
            (run_dir / launcher.VALIDATION_FILENAME).write_text(
                json.dumps(validation), encoding="utf-8"
            )
            artifact_paths = {}
            for name in launcher.CORE_ARM_ARTIFACTS:
                path = run_dir / name
                path.write_bytes(f"trusted-{name}".encode("utf-8"))
                artifact_paths[name] = path
            artifact_paths[launcher.VALIDATION_FILENAME] = (
                run_dir / launcher.VALIDATION_FILENAME
            )
            hashes = {
                "schema": launcher.HASH_RECEIPT_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir.resolve()),
                "mode": "smoke",
                "arm": "A",
                "seed": 3,
                "artifacts": {
                    name: {
                        "bytes": path.stat().st_size,
                        "sha256": launcher.sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                },
            }
            (run_dir / launcher.HASH_RECEIPT_FILENAME).write_text(
                json.dumps(hashes), encoding="utf-8"
            )
            record = launcher.validate_existing_verifier_receipts(
                run_dir, phase="smoke", arm="A", seed=3
            )
            self.assertEqual(record["amp_skip_attempts"], [2, 7])
            artifact = run_dir / "training-state-latest.pt"
            artifact.write_bytes(b"mutated-state")
            with self.assertRaisesRegex(launcher.LaunchError, "changed"):
                launcher.validate_existing_verifier_receipts(
                    run_dir, phase="smoke", arm="A", seed=3
                )

    def test_existing_pass_receipts_reject_incomplete_artifact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            artifact = run_dir / "training-state-latest.pt"
            artifact.write_bytes(b"trusted-state")
            validation = {
                "schema": launcher.VALIDATION_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir.resolve()),
                "mode": "smoke",
                "arm": "A",
                "seed": 3,
                "amp_skip_attempts": [],
                "amp_skip_signature_expected_value_enforced": False,
                "amp_skip_policy": launcher.AMP_SKIP_POLICY,
                "initial_common_state_sha256": "9" * 64,
            }
            validation_path = run_dir / launcher.VALIDATION_FILENAME
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            hashes = {
                "schema": launcher.HASH_RECEIPT_SCHEMA,
                "status": "passed",
                "run_dir": str(run_dir.resolve()),
                "mode": "smoke",
                "arm": "A",
                "seed": 3,
                "artifacts": {
                    artifact.name: {
                        "bytes": artifact.stat().st_size,
                        "sha256": launcher.sha256_file(artifact),
                    }
                },
            }
            (run_dir / launcher.HASH_RECEIPT_FILENAME).write_text(
                json.dumps(hashes), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                launcher.LaunchError, "missing required verifier artifacts"
            ):
                launcher.validate_existing_verifier_receipts(
                    run_dir, phase="smoke", arm="A", seed=3
                )

    def test_managed_deep_revalidation_is_registered_stopped_and_reaped(self):
        stop_event = threading.Event()
        registered = []
        unregistered = []

        def register(process):
            registered.append(process)
            stop_event.set()

        def unregister(process):
            unregistered.append(process)

        with self.assertRaisesRegex(
            launcher.LaunchError, "matrix audit stop was requested"
        ):
            launcher._run_deep_revalidation_command(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                process_env=os.environ,
                timeout_seconds=30.0,
                label="managed-test",
                stop_event=stop_event,
                register_process=register,
                unregister_process=unregister,
            )
        self.assertEqual(registered, unregistered)
        self.assertEqual(len(registered), 1)
        self.assertIsNotNone(registered[0].poll())

    def test_runtime_sandbox_fingerprint_detects_tree_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "etc").mkdir()
            os_release = root / "etc" / "os-release"
            os_release.write_text("NAME=test\n", encoding="utf-8")
            base = {
                "sandbox_path": str(root),
                "apptainer_executable": "/usr/bin/apptainer",
                "apptainer_executable_sha256": "0" * 64,
                "apptainer_version": "test",
                "python_command": "python",
            }
            before = launcher.runtime_sandbox_fingerprint(base)
            self.assertEqual(before["sandbox_regular_file_count"], 1)
            self.assertEqual(len(before["critical_files"]), 1)
            os_release.write_text("NAME=changed\n", encoding="utf-8")
            after = launcher.runtime_sandbox_fingerprint(base)
            self.assertNotEqual(
                before["sandbox_tree_metadata_sha256"],
                after["sandbox_tree_metadata_sha256"],
            )
            self.assertNotEqual(
                before["critical_files_sha256"], after["critical_files_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
