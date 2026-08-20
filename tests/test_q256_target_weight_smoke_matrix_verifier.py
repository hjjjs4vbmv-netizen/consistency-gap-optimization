import csv
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

from training import reproducibility
from scripts import run_q256_target_weight_matrix as launcher
from scripts import verify_q256_target_weight_arm as arm_verifier
from scripts import verify_q256_target_weight_smoke_matrix as matrix_verifier


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from test_q256_target_weight_verifier import RunFixture, sampler_state  # noqa: E402


class Q256TargetWeightSmokeMatrixVerifierTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(TESTS_DIR), previous_pythonpath)
            if part
        )

        def restore_pythonpath():
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath

        self.addCleanup(restore_pythonpath)

    def make_fixture(
        self,
        name,
        *,
        arm="A",
        skip_attempts=(2, 7),
        expected_skip_attempts=None,
    ):
        root = self.root / name
        root.mkdir()
        return RunFixture(
            root,
            arm=arm,
            skip_attempts=skip_attempts,
            expected_skip_attempts=expected_skip_attempts,
        )

    def finalize_arm(self, fixture):
        report = arm_verifier.verify_run(
            fixture.root,
            arm=fixture.arm,
            seed=3,
            mode="smoke",
            expected_skip_attempts=fixture.expected_skip_attempts,
        )
        runner_log = fixture.root / "runner.log"
        runner_log.write_text("runner PASS\n", encoding="utf-8")
        verifier_log = fixture.root / "arm_verifier.log"
        verifier_log.write_text("verifier PASS\n", encoding="utf-8")
        launch_manifest = fixture.root / "launch_manifest.json"
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
        completion = {
            "schema": launcher.RUNNER_COMPLETION_SCHEMA,
            "experiment_id": launcher.EXPERIMENT_ID,
            "started_utc": "2026-08-19T00:00:00Z",
            "finished_utc": "2026-08-19T00:00:02Z",
            "status": "PASS",
            "returncode": 0,
            "verifier_returncode": 0,
            "launch_manifest": launch_manifest.name,
            "launch_manifest_sha256": arm_verifier.sha256_file(
                launch_manifest
            ),
            "runner_log": runner_log.name,
            "runner_log_sha256": arm_verifier.sha256_file(runner_log),
            "verifier_log": verifier_log.name,
            "verifier_log_sha256": arm_verifier.sha256_file(verifier_log),
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
                "validation_receipt_sha256": arm_verifier.sha256_file(
                    fixture.root / arm_verifier.VALIDATION_FILENAME
                ),
                "artifact_hash_receipt_sha256": arm_verifier.sha256_file(
                    fixture.root / arm_verifier.HASH_RECEIPT_FILENAME
                ),
            },
        }
        (fixture.root / "runner_completion.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return report

    @staticmethod
    def _monitor():
        return {
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

    @staticmethod
    def _idle(timestamp):
        return {
            "checked_utc": timestamp,
            "gpu_uuid": "GPU-test",
            "compute_process_count": 0,
            "query": "gpu_uuid,pid,process_name,used_gpu_memory",
        }

    def finalize_resumed_arm(self, fixture):
        """Create a real preserved 16-attempt pause -> one resume provenance."""

        mutable_names = (
            "factorial_training_telemetry_v1.csv",
            "train_summary.csv",
            "network-snapshot-latest.pkl",
            "training-state-latest.pt",
            "log.txt",
        )
        final_bytes = {
            name: (fixture.root / name).read_bytes() for name in mutable_names
        }
        with (fixture.root / "factorial_training_telemetry_v1.csv").open(
            "rt", newline="", encoding="utf-8"
        ) as handle:
            final_rows = list(csv.DictReader(handle))

        original_manifest_path = fixture.root / "launch_manifest.json"
        original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        pause_gate = {
            "kind": "planned_exact_resume_pause",
            "stop_after_attempts": launcher.PLANNED_PAUSE_ATTEMPTS,
            "scientific_training_contract_unchanged": True,
        }
        original_manifest["gate_control"] = pause_gate
        original_manifest_path.write_text(
            json.dumps(original_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )

        options = json.loads(
            (fixture.root / "training_options.json").read_text(encoding="utf-8")
        )
        options["stop_after_attempts"] = launcher.PLANNED_PAUSE_ATTEMPTS
        options["resume_state_dump"] = None
        fixture.write_json("training_options.json", options)
        pause_rows = final_rows[: launcher.PLANNED_PAUSE_ATTEMPTS]
        fixture.write_telemetry(pause_rows)
        summary_rows = []
        for row in pause_rows:
            summary = {field: "0" for field in launcher.TRAIN_SUMMARY_FIELDS}
            summary.update(
                {
                    "attempted_iteration": row["attempted_iteration"],
                    "successful_optimizer_steps": row[
                        "successful_optimizer_steps"
                    ],
                    "processed_nimg": row["processed_nimg"],
                    "processed_kimg": row["processed_kimg"],
                    "loss": row["loss"],
                    "grad_scale": row["grad_scale_after"],
                    "step_skipped": row["step_skipped"],
                    "schedule": "sigmoid",
                    "stage": "0",
                    "next_loop_cur_tick": "1",
                    "elapsed_sec": row["elapsed_sec"],
                    "peak_vram_gb": "1.0",
                }
            )
            summary_rows.append(summary)
        with (fixture.root / "train_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=launcher.TRAIN_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
        pause_state = arm_verifier.torch_load_trusted(
            fixture.root / "training-state-latest.pt"
        )
        pause_state["attempted_iteration"] = launcher.PLANNED_PAUSE_ATTEMPTS
        pause_state["successful_optimizer_steps"] = int(
            pause_rows[-1]["successful_optimizer_steps"]
        )
        pause_state["cur_nimg"] = launcher.PLANNED_PAUSE_ATTEMPTS * 128
        pause_state["cur_tick"] = 1
        pause_state["tick_start_nimg"] = 128
        pause_state["elapsed_sec"] = float(pause_rows[-1]["elapsed_sec"])
        pause_state["rank_states"][0]["sampler_state"] = sampler_state(
            3, launcher.PLANNED_PAUSE_ATTEMPTS * 128
        )
        pause_state["gradscaler_state"]["scale"] = float(
            pause_rows[-1]["grad_scale_after"]
        )
        fixture.write_state(pause_state)
        (fixture.root / "log.txt").write_text(
            "Planned pause after 16 attempts; exiting.\n", encoding="utf-8"
        )
        runner_log = fixture.root / "runner.log"
        runner_log.write_text("planned pause PASS\n", encoding="utf-8")
        pause_report = launcher.verify_planned_pause_run(
            fixture.root,
            arm=fixture.arm,
            seed=3,
            stop_after_attempts=launcher.PLANNED_PAUSE_ATTEMPTS,
            runtime_command=[sys.executable],
            process_env=os.environ,
        )
        primary_completion = {
            "schema": launcher.RUNNER_COMPLETION_SCHEMA,
            "experiment_id": launcher.EXPERIMENT_ID,
            "started_utc": "2026-08-19T00:00:00Z",
            "finished_utc": "2026-08-19T00:00:02Z",
            "status": launcher.PLANNED_PAUSE_STATUS,
            "returncode": 0,
            "launch_manifest": "launch_manifest.json",
            "launch_manifest_sha256": launcher.sha256_file(
                original_manifest_path
            ),
            "runner_log": runner_log.name,
            "runner_log_sha256": launcher.sha256_file(runner_log),
            "final_prelaunch_gpu_idle_check": self._idle(
                "2026-08-19T00:00:00Z"
            ),
            "training_gpu_exclusivity_monitor": self._monitor(),
            "post_training_gpu_idle_check": self._idle(
                "2026-08-19T00:00:02Z"
            ),
            "planned_pause_verification": pause_report,
            "full_arm_verifier_invoked": False,
            "resume_required": True,
        }
        primary_completion_path = fixture.root / "runner_completion.json"
        primary_completion_path.write_text(
            json.dumps(primary_completion), encoding="utf-8"
        )
        planned_record = launcher.validate_planned_pause_completion(
            fixture.root,
            launch_manifest=original_manifest,
            stop_after_attempts=launcher.PLANNED_PAUSE_ATTEMPTS,
        )
        evidence = launcher.preserve_planned_pause_evidence(
            fixture.root, planned_record
        )
        planned_record = {**planned_record, "evidence": evidence}

        for name, content in final_bytes.items():
            (fixture.root / name).write_bytes(content)
        report = arm_verifier.verify_run(
            fixture.root,
            arm=fixture.arm,
            seed=3,
            mode="smoke",
            expected_skip_attempts=fixture.expected_skip_attempts,
        )
        resume_state = (
            fixture.root
            / evidence["directory"]
            / "training-state-latest.pt"
        )
        resume_manifest = {
            **original_manifest,
            "launch_kind": "resume",
            "status": "authorized_to_start",
            "gate_control": {
                "kind": "none",
                "stop_after_attempts": None,
                "scientific_training_contract_unchanged": True,
            },
            "original_gate_control": pause_gate,
            "original_launch_manifest_sha256": launcher.sha256_file(
                original_manifest_path
            ),
            "validated_planned_pause_completion": planned_record,
            "resume_state": str(resume_state),
            "resume_state_sha256": launcher.sha256_file(resume_state),
        }
        resume_command = launcher.build_training_command(
            python_bin=sys.executable,
            data=Path(
                original_manifest["assets"]["dataset"]["resolved_path"]
            ),
            transfer=Path(
                original_manifest["assets"]["transfer"]["resolved_path"]
            ),
            outdir=fixture.root,
            phase="smoke",
            arm=fixture.arm,
            seed=3,
            resume=resume_state,
            runtime_command=[sys.executable],
        )
        resume_manifest["exact_command_argv"] = resume_command
        resume_manifest["exact_command_shell"] = launcher.shlex.join(
            resume_command
        )
        token = "20260819T000003Z-deadbeef"
        resume_manifest_path = (
            fixture.root / f"resume_launch_manifest-{token}.json"
        )
        resume_manifest_path.write_text(
            json.dumps(resume_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        resume_log = fixture.root / f"runner-resume-{token}.log"
        resume_log.write_text("resume PASS\n", encoding="utf-8")
        verifier_log = fixture.root / f"arm-verifier-resume-{token}.log"
        verifier_log.write_text("verifier PASS\n", encoding="utf-8")
        completion = {
            "schema": launcher.RUNNER_COMPLETION_SCHEMA,
            "experiment_id": launcher.EXPERIMENT_ID,
            "started_utc": "2026-08-19T00:00:03Z",
            "finished_utc": "2026-08-19T00:00:05Z",
            "status": "PASS",
            "returncode": 0,
            "verifier_returncode": 0,
            "launch_manifest": resume_manifest_path.name,
            "launch_manifest_sha256": launcher.sha256_file(resume_manifest_path),
            "runner_log": resume_log.name,
            "runner_log_sha256": launcher.sha256_file(resume_log),
            "verifier_log": verifier_log.name,
            "verifier_log_sha256": launcher.sha256_file(verifier_log),
            "training_gpu_exclusivity_monitor": self._monitor(),
            "verifier_gpu_exclusivity_monitor": self._monitor(),
            "final_prelaunch_gpu_idle_check": self._idle(
                "2026-08-19T00:00:03Z"
            ),
            "post_training_gpu_idle_check": self._idle(
                "2026-08-19T00:00:04Z"
            ),
            "post_verifier_gpu_idle_check": self._idle(
                "2026-08-19T00:00:05Z"
            ),
            "verification": {
                "validation_receipt_sha256": launcher.sha256_file(
                    fixture.root / arm_verifier.VALIDATION_FILENAME
                ),
                "artifact_hash_receipt_sha256": launcher.sha256_file(
                    fixture.root / arm_verifier.HASH_RECEIPT_FILENAME
                ),
            },
        }
        (fixture.root / f"runner-completion-resume-{token}.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return report

    def make_matrix(self, *, skip_by_arm=None, mutators=None):
        skip_by_arm = skip_by_arm or {}
        mutators = mutators or {}
        fixtures = {}
        for arm in matrix_verifier.ARMS:
            fixture = self.make_fixture(
                f"arm{arm}",
                arm=arm,
                skip_attempts=skip_by_arm.get(arm, (2, 7)),
            )
            if arm in mutators:
                mutators[arm](fixture)
            self.finalize_arm(fixture)
            fixtures[arm] = fixture
        return fixtures, {arm: fixture.root for arm, fixture in fixtures.items()}

    def test_valid_matrix_emits_one_immutable_cross_arm_pass(self):
        _, run_dirs = self.make_matrix()
        receipt = self.root / "matrix-pass.json"
        report = matrix_verifier.verify_smoke_matrix(
            run_dirs, receipt_path=receipt
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["amp_skip_attempts_by_arm"],
            {arm: [2, 7] for arm in matrix_verifier.ARMS},
        )
        self.assertEqual(report["amp_skip_count"], 2)
        self.assertEqual(report["successful_optimizer_steps"], 30)
        self.assertEqual(
            report["amp_skip_policy"], matrix_verifier.AMP_SKIP_POLICY
        )
        self.assertTrue(
            report["trajectory_checks"]["native_A_target_equals_denominator"]
        )
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], matrix_verifier.VALIDATION_SCHEMA)
        self.assertNotIn("validation_receipt", payload)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "immutable matrix PASS receipt already exists",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs, receipt_path=receipt
            )

    def test_explicit_prospective_skip_signature_revalidates_without_mode_drift(self):
        fixtures = {}
        for arm in matrix_verifier.ARMS:
            fixture = self.make_fixture(
                f"explicit-arm{arm}",
                arm=arm,
                skip_attempts=(2, 7),
                expected_skip_attempts=[2, 7],
            )
            self.finalize_arm(fixture)
            fixtures[arm] = fixture.root
        report = matrix_verifier.verify_smoke_matrix(
            fixtures, write_receipt=False
        )
        self.assertEqual(report["amp_skip_count"], 2)

    def test_common_batch_and_factor_specific_trajectories_are_exact(self):
        def change_batch(fixture):
            rows = fixture.telemetry_rows()
            rows[4]["batch_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"D": change_batch})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "all-arms common trajectory.batch_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "second"
        self.root.mkdir()

        def change_target(fixture):
            rows = fixture.telemetry_rows()
            rows[10]["target_r_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"C": change_target})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "target scale 1.1.target_r_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "denominator"
        self.root.mkdir()

        def change_denominator(fixture):
            rows = fixture.telemetry_rows()
            rows[6]["denominator_r_sha256"] = "9" * 64
            fixture.write_telemetry(rows)

        _, run_dirs = self.make_matrix(mutators={"D": change_denominator})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "denominator scale 1.1.denominator_r_sha256 mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_native_arms_require_all_target_denominator_statistics_equal(self):
        def change_native_statistic(fixture):
            rows = fixture.telemetry_rows()
            rows[1]["denominator_delta_mean"] = "0.3"
            fixture.write_telemetry(rows)

        # Keep the denominator-scale A=C relation valid so this fixture reaches
        # the stricter within-native-arm target=denominator comparison.
        _, run_dirs = self.make_matrix(
            mutators={"A": change_native_statistic, "C": change_native_statistic}
        )
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "native arm A target/denominator delta_mean mismatch at attempt 2",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_initial_components_final_rank_state_and_skip_exposure_match(self):
        def change_initial(fixture):
            receipt = fixture.initial_receipt()
            receipt["hashes"]["optimizer"] = "9" * 64
            receipt["common_initial_state_sha256"] = reproducibility.state_sha256(
                receipt["hashes"]
            )
            fixture.write_json("initial_state_receipt_v1.json", receipt)

        _, run_dirs = self.make_matrix(mutators={"C": change_initial})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "common initial model/EMA/optimizer/GradScaler/RNG/sampler hashes mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "rng"
        self.root.mkdir()

        def change_rng(fixture):
            state = fixture.state()
            state["rank_states"][0]["rng_state"]["python"] = (
                random.Random(99).getstate()
            )
            fixture.write_state(state)

        _, run_dirs = self.make_matrix(mutators={"D": change_rng})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "final rank RNG state mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

        self.root = self.root / "skips"
        self.root.mkdir()
        _, run_dirs = self.make_matrix(skip_by_arm={"B": (2, 8)})
        report = matrix_verifier.verify_smoke_matrix(
            run_dirs, write_receipt=False
        )
        self.assertEqual(report["amp_skip_attempts_by_arm"]["B"], [2, 8])
        self.assertEqual(report["amp_skip_count"], 2)

        self.root = self.root / "unequal-count"
        self.root.mkdir()
        _, run_dirs = self.make_matrix(skip_by_arm={"B": (2,)})
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "AMP skip count mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_existing_single_arm_pass_receipt_must_still_bind_artifacts(self):
        _, run_dirs = self.make_matrix()
        telemetry = run_dirs["A"] / "factorial_training_telemetry_v1.csv"
        telemetry.write_bytes(telemetry.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "PASS-bound artifact changed",
        ):
            matrix_verifier.verify_smoke_matrix(run_dirs, write_receipt=False)

    def test_exact_resume_ignores_only_wall_clock_fields(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        rows = resumed.telemetry_rows()
        for index, row in enumerate(rows, start=1):
            elapsed = 100.0 + index
            row["elapsed_sec"] = f"{elapsed:.6f}"
            row["gpu_hours_cumulative"] = f"{elapsed / 3600:.9f}"
        resumed.write_telemetry(rows)
        state = resumed.state()
        state["elapsed_sec"] = 132.0
        resumed.write_state(state)
        self.finalize_resumed_arm(resumed)

        report = matrix_verifier.verify_smoke_matrix(
            run_dirs,
            resume_pair=(run_dirs["A"], resumed.root, "A"),
            write_receipt=False,
        )
        comparison = report["exact_resume"]
        self.assertEqual(comparison["status"], "passed")
        self.assertEqual(
            comparison["excluded_noncomputational_fields"],
            ["elapsed_sec", "gpu_hours_cumulative"],
        )

    def test_exact_resume_rejects_each_computational_difference(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        state = resumed.state()
        state["optimizer_state"]["param_groups"][0]["lr"] = 9e-5
        resumed.write_state(state)
        self.finalize_resumed_arm(resumed)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "exact-resume final optimizer mismatch",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs,
                resume_pair=(run_dirs["A"], resumed.root, "A"),
                write_receipt=False,
            )

        self.root = self.root / "telemetry"
        self.root.mkdir()
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        rows = resumed.telemetry_rows()
        rows[12]["loss"] = "1.5"
        resumed.write_telemetry(rows)
        self.finalize_resumed_arm(resumed)
        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
                r"computational telemetry mismatch at attempt 13, fields=\['loss'\]",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs,
                resume_pair=(run_dirs["A"], resumed.root, "A"),
                write_receipt=False,
            )

    def test_exact_resume_rejects_two_independent_fresh_runs(self):
        _, run_dirs = self.make_matrix()
        second_fresh = self.make_fixture("secondFreshA", arm="A")
        self.finalize_arm(second_fresh)

        with self.assertRaisesRegex(
            matrix_verifier.MatrixVerificationError,
            "exact-resume provenance failed",
        ):
            matrix_verifier.verify_smoke_matrix(
                run_dirs,
                resume_pair=(run_dirs["A"], second_fresh.root, "A"),
                write_receipt=False,
            )

    def test_exact_resume_rejects_changed_parent_options(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        self.finalize_resumed_arm(resumed)
        options_path = resumed.root / "training_options.json"
        options = json.loads(options_path.read_text(encoding="utf-8"))
        options["stop_after_attempts"] = None
        options_path.write_text(
            json.dumps(options, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(
            launcher.LaunchError,
            "pause evidence was spliced",
        ):
            launcher.validate_exact_resume_provenance(
                run_dirs["A"],
                resumed.root,
                arm="A",
                seed=3,
                runtime_command=[sys.executable],
                process_env=os.environ,
            )

    def test_exact_resume_rejects_noncanonical_resume_command(self):
        _, run_dirs = self.make_matrix()
        resumed = self.make_fixture("resumedA", arm="A")
        self.finalize_resumed_arm(resumed)
        manifest_path = next(resumed.root.glob("resume_launch_manifest-*.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["exact_command_argv"].append("--lr=0.0002")
        manifest["exact_command_shell"] = launcher.shlex.join(
            manifest["exact_command_argv"]
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(
            launcher.LaunchError,
            "differs from the frozen preserved-state command",
        ):
            launcher.validate_exact_resume_provenance(
                run_dirs["A"],
                resumed.root,
                arm="A",
                seed=3,
                runtime_command=[sys.executable],
                process_env=os.environ,
            )


if __name__ == "__main__":
    unittest.main()
