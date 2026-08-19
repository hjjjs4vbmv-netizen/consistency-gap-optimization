from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_q256_g110_moment_transport as runner


class Q256MomentTransportRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        (self.repo / "training").mkdir(parents=True)
        (self.repo / "ct_train.py").write_bytes(b"train-core\n")
        (self.repo / "training" / "ct_training_loop.py").write_bytes(b"loop-core\n")
        for name in ("loss.py", "schedules.py", "networks.py", "dataset.py"):
            (self.repo / "training" / name).write_bytes(f"{name}-core\n".encode())
        (self.repo / "scripts").mkdir()
        self.transformer = self.repo / "scripts" / "transport.py"
        self.transformer.write_bytes(b"transformer\n")
        self.runner_script = self.repo / "scripts" / "run_q256_g110_moment_transport.py"
        self.runner_script.write_bytes(b"runner\n")
        self.validator_script = self.repo / "validator.py"
        self.validator_script.write_bytes(b"validator\n")
        self.comparator_script = self.repo / "compare.py"
        self.comparator_script.write_bytes(b"comparator\n")
        self.dataset = self.root / "cifar.zip"
        self.dataset.write_bytes(b"canonical-cifar\n")
        self.container_identity = self.root / "container-identity.json"
        self.container_identity.write_text('{"sandbox":"frozen"}\n', encoding="utf-8")
        self.gate = self.root / "preflight.json"
        self.compatibility = self.root / "compatibility.json"
        self.run_root = self.root / "runs" / "q256-transport"
        self.manifest = self.make_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": runner.sha256_file(path)}

    def write_compatibility(self, *, fresh_f: bool) -> None:
        seeds = {}
        rows = []
        protocol_common = {
            "q": 256,
            "batch.batch_size": 128,
            "batch.batch_gpu": 16,
            "augmentation": {"augment_kwargs": None, "dataset_xflip": False},
            "precision.use_fp16": True,
            "precision.enable_amp": True,
            "precision.enable_tf32": False,
            "precision.loss_scaling": 1.0,
            "checkpoint_cadence.kimg_per_tick": 10,
            "checkpoint_cadence.snapshot_ticks": None,
            "checkpoint_cadence.state_dump_ticks": None,
            "checkpoint_cadence.ckpt_ticks": 10,
            "checkpoint_cadence.sample_ticks": 26,
            "checkpoint_cadence.eval_ticks": 50,
            "data.byte_sha256": runner.sha256_file(self.dataset),
            "start_kimg": 256,
            "endpoints_kimg": [512, 768, 1024],
        }
        for seed in (3, 4, 5):
            seeds[str(seed)] = {
                "controls": {
                    "F": {
                        "reuse_decision": "fresh_required" if fresh_f else "reusable",
                        "reusable": not fresh_f,
                    },
                    "G": {"reuse_decision": "fresh_required", "reusable": False},
                }
            }
            source = next(item for item in self.seed_specs if item["seed"] == seed)
            for arm in ("F", "G"):
                for field, expected in (
                    ("full_state_sha256", source["source_state"]["sha256"]),
                    ("checkpoint_sha256", source["source_snapshot"]["sha256"]),
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "scope": "source",
                            "field": field,
                            "expected": expected,
                            "observed": None,
                            "status": "missing",
                        }
                    )
                expected_protocol = {
                    **protocol_common,
                    "schedule": "sigmoid" if arm == "F" else "global_sigmoid",
                    "gap_scale": 1.0 if arm == "F" else 1.1,
                }
                for field, expected in expected_protocol.items():
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "scope": "run",
                            "field": field,
                            "expected": expected,
                            "observed": expected,
                            "status": "match",
                        }
                    )
        self.compatibility.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "reusable_controls": False,
                    "seeds": seeds,
                    "rows": rows,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_preflight(self) -> None:
        coefficients = {row["seed"]: row["coefficient"] for row in self.seed_specs}
        source_assets = {
            str(row["seed"]): {
                "source_state_sha256": row["source_state"]["sha256"],
                "checkpoint_sha256": row["source_snapshot"]["sha256"],
            }
            for row in self.seed_specs
        }
        self.gate.write_text(
            json.dumps(
                {
                    "status": "GO",
                    "formal_training_authorized": True,
                    "frozen_a_s": {
                        str(seed): value for seed, value in coefficients.items()
                    },
                    "source_assets": source_assets,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def make_manifest(self) -> dict:
        seeds = []
        coefficients = {
            3: 0.8370121196598016,
            4: 0.8073491626309143,
            5: 0.8233457134218897,
        }
        for seed, gpu in ((3, 0), (4, 1), (5, 0)):
            source_dir = self.root / "source" / f"seed{seed}"
            source_dir.mkdir(parents=True)
            state = source_dir / "training-state-latest.pt"
            snapshot = source_dir / "network-snapshot-latest.pkl"
            state.write_bytes(f"state-{seed}\n".encode())
            snapshot.write_bytes(f"snapshot-{seed}\n".encode())
            seeds.append(
                {
                    "seed": seed,
                    "gpu": gpu,
                    "coefficient": coefficients[seed],
                    "source_state": self.artifact(state),
                    "source_snapshot": self.artifact(snapshot),
                }
            )
        self.seed_specs = seeds
        self.write_preflight()
        self.write_compatibility(fresh_f=False)
        gate = {
            "name": "heldout-preflight",
            **self.artifact(self.gate),
            "verdict_key": "status",
            "expected": "GO",
            "binding": "preflight",
        }
        validator = [
            sys.executable,
            str(self.validator_script),
            "--state={state}",
            "--snapshot={snapshot}",
            "--expected-nimg={expected_nimg}",
            "--expected-mapping={expected_mapping}",
            "--expected-gap-scale={expected_gap_scale}",
            "--method={expected_method}",
            "--checkpoint-id={checkpoint_id}",
            "--training-run-id={training_run_id}",
            "--output={result_receipt}",
        ]
        comparator = [
            sys.executable,
            str(self.comparator_script),
            "--left-state={left_state}",
            "--right-state={right_state}",
            "--left-snapshot={left_snapshot}",
            "--right-snapshot={right_snapshot}",
            "--output={result_receipt}",
        ]
        return {
            "schema": runner.SCHEMA,
            "experiment_id": "q256-g110-moment-transport-20260819",
            "paths": {
                "repo_root": str(self.repo),
                "run_root": str(self.run_root),
                "transformer": str(self.transformer),
            },
            "runtime": {
                "python_command": [sys.executable],
                "worker_command": [sys.executable],
                "git_command": ["git"],
                "nvidia_smi_command": ["nvidia-smi"],
                "tmux_binary": "tmux",
                "master_port_base": 29630,
            },
            "provenance": {
                "execution_commit": "a" * 40,
                "dataset": self.artifact(self.dataset),
                "container_identity": self.artifact(self.container_identity),
                "core_code": [
                    self.artifact(self.repo / "ct_train.py"),
                    self.artifact(self.repo / "training" / "ct_training_loop.py"),
                    self.artifact(self.repo / "training" / "loss.py"),
                    self.artifact(self.repo / "training" / "schedules.py"),
                    self.artifact(self.repo / "training" / "networks.py"),
                    self.artifact(self.repo / "training" / "dataset.py"),
                    self.artifact(self.runner_script),
                    self.artifact(self.transformer),
                    self.artifact(self.validator_script),
                    self.artifact(self.comparator_script),
                ],
            },
            "resources": {
                "min_free_disk_bytes": 1,
                "min_free_gpu_mib": 1024,
                "max_gpu_utilization_pct": 10,
            },
            "training": {
                "source_kimg": 256,
                "endpoints_kimg": [512, 768, 1024],
                "smoke_steps": 32,
                "cond": False,
                "arch": "ddpmpp",
                "precond": "ect",
                "batch": 128,
                "batch_gpu": 16,
                "optim": "RAdam",
                "lr": 0.0001,
                "dropout": 0.2,
                "augment": 0,
                "xflip": False,
                "mapping": "global_sigmoid",
                "global_gap_scale": 1.10,
                "q": 256,
                "k": 8,
                "b": 1,
                "c": 0,
                "double": 10000,
                "ema_beta": 0.9993,
                "fp16": True,
                "enable_amp": True,
                "tf32": False,
                "ls": 1.0,
                "metrics": "none",
                "tick": 10,
                "snap": 0,
                "dump": 0,
                "ckpt": 10,
                "sample_every": 26,
                "eval_every": 50,
                "adaptive_update_kimg": 0.5,
            },
            "smoke": {
                "seed": 3,
                "gpu": 0,
                "validator_command": validator,
                "comparator_command": comparator,
                "validator_gate": {"verdict_key": "verdict", "expected": "GO"},
                "comparator_gate": {"verdict_key": "verdict", "expected": "GO"},
            },
            "formal": {
                "arms": ["G", "T"],
                "compatibility_report": self.artifact(self.compatibility),
                "validator_command": validator,
                "validator_gate": {"verdict_key": "verdict", "expected": "GO"},
            },
            "smoke_gates": [gate],
            "formal_gates": [gate],
            "seeds": seeds,
        }

    def normalized(self, value: dict | None = None) -> dict:
        return runner.validate_manifest(
            self.manifest if value is None else value,
            verify_artifacts=True,
        )

    def test_manifest_and_prepare_plan_freeze_source_hashes_and_coefficients(
        self,
    ) -> None:
        config = self.normalized()
        plan = runner.build_plan(config, "f" * 64, "prepare")
        self.assertEqual(plan["job_count"], 6)
        jobs = {job["job_id"]: job for job in plan["jobs"]}
        for seed in (3, 4, 5):
            noop = jobs[f"prepare-seed{seed}-noop"]
            transported = jobs[f"prepare-seed{seed}-transport"]
            self.assertEqual(
                noop["command"][noop["command"].index("--coefficient") + 1], "1"
            )
            expected = next(
                row["coefficient"] for row in config["seeds"] if row["seed"] == seed
            )
            self.assertEqual(
                transported["command"][
                    transported["command"].index("--coefficient") + 1
                ],
                format(expected, ".17g"),
            )
            self.assertIn("--expected-source-sha256", transported["command"])
            self.assertNotEqual(
                transported["output_artifacts"][0]["path"],
                next(
                    row["source_state"]["path"]
                    for row in config["seeds"]
                    if row["seed"] == seed
                ),
            )

    def test_smoke_plan_has_four_independent_32_step_runs_and_two_comparisons(
        self,
    ) -> None:
        config = self.normalized()
        jobs = runner.build_plan(config, "b" * 64, "smoke")["jobs"]
        trains = [job for job in jobs if job["kind"] == "train"]
        validators = [job for job in jobs if job["kind"] == "validate"]
        comparators = [job for job in jobs if job["kind"] == "compare"]
        self.assertEqual(len(trains), 4)
        self.assertEqual(len(validators), 4)
        self.assertEqual(len(comparators), 2)
        self.assertEqual({job["gpu"] for job in jobs}, {0})
        names = {job["job_id"] for job in trains}
        self.assertIn("smoke-seed3-noop-direct-train", names)
        self.assertIn("smoke-seed3-noop-rewrite-train", names)
        self.assertIn("smoke-seed3-transport-repeat1-train", names)
        self.assertIn("smoke-seed3-transport-repeat2-train", names)
        for job in trains:
            command = job["command"]
            self.assertEqual(
                job["environment"],
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29630",
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": "1",
                },
            )
            self.assertIn("--duration=0.26", command)
            self.assertIn("--seed=3", command)
            self.assertIn("--mapping=global_sigmoid", command)
            self.assertIn("--global-gap-scale=1.1", command)
            self.assertIn("--sample_every=26", command)
            self.assertIn("--eval_every=50", command)
            self.assertIn("--metrics=none", command)
            self.assertIn("--ls=1.0", command)
            self.assertEqual(len(job["input_artifacts"]), 2)
        direct = next(job for job in trains if "noop-direct" in job["job_id"])
        rewrite = next(job for job in trains if "noop-rewrite" in job["job_id"])
        self.assertIn(
            "/source/seed3/training-state-latest.pt", " ".join(direct["command"])
        )
        self.assertIn(
            "/staged/seed3/noop/training-state-latest.pt", " ".join(rewrite["command"])
        )
        for job in validators:
            self.assertIn("--expected-nimg=260096", job["command"])

    def test_formal_plan_is_paired_segmented_and_same_gpu_within_seed(self) -> None:
        config = self.normalized()
        jobs = runner.build_plan(config, "c" * 64, "formal")["jobs"]
        trains = [job for job in jobs if job["kind"] == "train"]
        validators = [job for job in jobs if job["kind"] == "validate"]
        self.assertEqual(len(trains), 18)
        self.assertEqual(len(validators), 18)
        expected_gpu = {3: 0, 4: 1, 5: 0}
        for seed in (3, 4, 5):
            seed_jobs = [job for job in trains if job["seed"] == seed]
            self.assertEqual({job["gpu"] for job in seed_jobs}, {expected_gpu[seed]})
            expected_order = [
                f"formal-seed{seed}-{arm}-{endpoint}k-train"
                for endpoint in (512, 768, 1024)
                for arm in ("G", "T")
            ]
            self.assertEqual([job["job_id"] for job in seed_jobs], expected_order)
            for endpoint in (512, 768, 1024):
                pair = [job for job in seed_jobs if f"-{endpoint}k-" in job["job_id"]]
                self.assertEqual({job["gpu"] for job in pair}, {expected_gpu[seed]})
                for job in pair:
                    self.assertIn(
                        f"--duration={runner._duration_arg(endpoint)}", job["command"]
                    )
                    self.assertIn("--tick=10", job["command"])
                    self.assertIn("--ckpt=10", job["command"])
            first_g = next(job for job in seed_jobs if "-G-512k-" in job["job_id"])
            first_t = next(job for job in seed_jobs if "-T-512k-" in job["job_id"])
            g768 = next(job for job in seed_jobs if "-G-768k-" in job["job_id"])
            self.assertIn(
                f"source/seed{seed}/training-state-latest.pt",
                " ".join(first_g["command"]),
            )
            self.assertIn(
                f"staged/seed{seed}/transport/training-state-latest.pt",
                " ".join(first_t["command"]),
            )
            self.assertIn(
                f"formal/seed{seed}/G/512k/training-state-latest.pt",
                " ".join(g768["command"]),
            )

    def test_fresh_required_f_expands_to_three_serial_arms_with_fixed_schedule(
        self,
    ) -> None:
        self.write_compatibility(fresh_f=True)
        manifest = copy.deepcopy(self.manifest)
        manifest["formal"]["arms"] = ["F", "G", "T"]
        manifest["formal"]["compatibility_report"] = self.artifact(self.compatibility)
        config = self.normalized(manifest)
        jobs = runner.build_plan(config, "7" * 64, "formal")["jobs"]
        trains = [job for job in jobs if job["kind"] == "train"]
        self.assertEqual(len(trains), 27)
        self.assertEqual(len([job for job in jobs if job["kind"] == "validate"]), 27)
        for seed in (3, 4, 5):
            seed_jobs = [job for job in trains if job["seed"] == seed]
            expected = [
                f"formal-seed{seed}-{arm}-{endpoint}k-train"
                for endpoint in (512, 768, 1024)
                for arm in ("F", "G", "T")
            ]
            self.assertEqual([job["job_id"] for job in seed_jobs], expected)
            self.assertEqual(len({job["gpu"] for job in seed_jobs}), 1)
            f512 = next(
                job for job in seed_jobs if f"seed{seed}-F-512k" in job["job_id"]
            )
            g512 = next(
                job for job in seed_jobs if f"seed{seed}-G-512k" in job["job_id"]
            )
            t512 = next(
                job for job in seed_jobs if f"seed{seed}-T-512k" in job["job_id"]
            )
            self.assertIn("--mapping=sigmoid", f512["command"])
            self.assertIn("--global-gap-scale=1.0", f512["command"])
            self.assertIn("--mapping=global_sigmoid", g512["command"])
            self.assertIn("--global-gap-scale=1.1", g512["command"])
            self.assertIn("--mapping=global_sigmoid", t512["command"])
            self.assertIn(
                f"source/seed{seed}/training-state-latest.pt", " ".join(f512["command"])
            )
            self.assertIn(
                f"source/seed{seed}/training-state-latest.pt", " ".join(g512["command"])
            )
            self.assertIn(
                f"staged/seed{seed}/transport/training-state-latest.pt",
                " ".join(t512["command"]),
            )
            f_validator = next(
                job
                for job in jobs
                if job["job_id"] == f"formal-seed{seed}-F-512k-validate"
            )
            t_validator = next(
                job
                for job in jobs
                if job["job_id"] == f"formal-seed{seed}-T-512k-validate"
            )
            self.assertIn("--expected-mapping=sigmoid", f_validator["command"])
            self.assertIn("--expected-gap-scale=1.0", f_validator["command"])
            self.assertIn("--expected-mapping=global_sigmoid", t_validator["command"])
            self.assertIn("--expected-gap-scale=1.1", t_validator["command"])
            self.assertIn("--method=fixed", f_validator["command"])
            self.assertIn("--method=global110", t_validator["command"])

    def test_compatibility_fresh_f_cannot_launch_only_g_t(self) -> None:
        self.write_compatibility(fresh_f=True)
        manifest = copy.deepcopy(self.manifest)
        manifest["formal"]["compatibility_report"] = self.artifact(self.compatibility)
        with self.assertRaisesRegex(
            runner.RunnerError, "inconsistent with compatibility"
        ):
            self.normalized(manifest)

    def test_reusable_f_cannot_be_silently_added_as_a_fresh_arm(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["formal"]["arms"] = ["F", "G", "T"]
        with self.assertRaisesRegex(
            runner.RunnerError, "inconsistent with compatibility"
        ):
            self.normalized(manifest)

    def test_protocol_changes_fail_closed(self) -> None:
        for key, bad_value in (
            ("q", 128),
            ("mapping", "sigmoid"),
            ("sample_every", 10000),
            ("eval_every", 49),
            ("tf32", True),
        ):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["training"][key] = bad_value
                with self.assertRaisesRegex(runner.RunnerError, f"training.{key}"):
                    self.normalized(manifest)

    def test_distributed_ports_are_frozen_per_physical_gpu(self) -> None:
        config = self.normalized()
        jobs = runner.build_plan(config, "6" * 64, "formal")["jobs"]
        ports_by_gpu = {}
        for job in jobs:
            ports_by_gpu.setdefault(job["gpu"], set()).add(
                job["environment"]["MASTER_PORT"]
            )
            self.assertEqual(job["environment"]["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(job["environment"]["RANK"], "0")
            self.assertEqual(job["environment"]["LOCAL_RANK"], "0")
            self.assertEqual(job["environment"]["WORLD_SIZE"], "1")
        self.assertEqual(ports_by_gpu, {0: {"29630"}, 1: {"29631"}})

    def test_invalid_master_port_base_fails_closed(self) -> None:
        for value in (1023, 65536, True, "29630"):
            with self.subTest(value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest["runtime"]["master_port_base"] = value
                with self.assertRaisesRegex(runner.RunnerError, "master_port_base"):
                    self.normalized(manifest)

    def test_missing_or_non_go_scientific_gate_fails_closed(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["formal_gates"] = []
        with self.assertRaisesRegex(runner.RunnerError, "formal_gates"):
            self.normalized(manifest)
        config = self.normalized()
        self.gate.write_text('{"status":"NO-GO"}\n', encoding="utf-8")
        # Update the expected digest so this exercises the verdict check, not hash binding.
        gate = copy.deepcopy(config["formal_gates"][0])
        gate["sha256"] = runner.sha256_file(self.gate)
        with self.assertRaisesRegex(runner.RunnerError, "not eligible"):
            runner.verify_external_gates([gate], config)

    def test_preflight_binding_rejects_wrong_coefficient_or_source(self) -> None:
        config = self.normalized()
        payload = json.loads(self.gate.read_text(encoding="utf-8"))
        wrong_coefficient = copy.deepcopy(payload)
        wrong_coefficient["frozen_a_s"]["3"] += 0.001
        with self.assertRaisesRegex(runner.RunnerError, "coefficient differs"):
            runner.verify_preflight_binding(wrong_coefficient, config)
        wrong_source = copy.deepcopy(payload)
        wrong_source["source_assets"]["4"]["source_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(runner.RunnerError, "source state differs"):
            runner.verify_preflight_binding(wrong_source, config)

    def test_compatibility_report_must_bind_the_execution_source_and_protocol(
        self,
    ) -> None:
        payload = json.loads(self.compatibility.read_text(encoding="utf-8"))
        target = next(
            row
            for row in payload["rows"]
            if row["seed"] == 3
            and row["arm"] == "F"
            and row["scope"] == "source"
            and row["field"] == "full_state_sha256"
        )
        target["expected"] = "0" * 64
        self.compatibility.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        manifest = copy.deepcopy(self.manifest)
        manifest["formal"]["compatibility_report"] = self.artifact(self.compatibility)
        with self.assertRaisesRegex(runner.RunnerError, "source identity differs"):
            self.normalized(manifest)

    def test_hashed_execution_tools_are_required(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["provenance"]["core_code"] = [
            item
            for item in manifest["provenance"]["core_code"]
            if item["path"] != str(self.transformer)
        ]
        with self.assertRaisesRegex(
            runner.RunnerError, "missing required training files"
        ):
            self.normalized(manifest)

    def test_truncated_prepare_and_smoke_gate_receipts_cannot_authorize(self) -> None:
        config = self.normalized()
        manifest_sha = "9" * 64
        prepare_plan = runner.build_plan(config, manifest_sha, "prepare")
        prepare_path = self.run_root / "_runner" / "prepare_receipt.json"
        prepare_path.parent.mkdir(parents=True)
        prepare_path.write_text(
            json.dumps(
                {
                    "schema": runner.PREPARE_SCHEMA,
                    "verdict": "GO",
                    "experiment_id": config["experiment_id"],
                    "manifest_sha256": manifest_sha,
                    "execution_commit": config["provenance"]["execution_commit"],
                    "jobs_sha256": prepare_plan["jobs_sha256"],
                    "artifacts": [],
                    "job_exit_receipts": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.RunnerError, "artifact set is incomplete"):
            runner._validate_prepare_receipt(config, manifest_sha)

        smoke_plan = runner.build_plan(config, manifest_sha, "smoke")
        smoke_gate = self.run_root / "_runner" / "smoke_gate.json"
        smoke_gate.write_text(
            json.dumps(
                {
                    "schema": runner.GATE_SCHEMA,
                    "phase": "smoke",
                    "verdict": "GO",
                    "experiment_id": config["experiment_id"],
                    "manifest_sha256": manifest_sha,
                    "execution_commit": config["provenance"]["execution_commit"],
                    "jobs_sha256": smoke_plan["jobs_sha256"],
                    "exit_receipts": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.RunnerError, "job set is incomplete"):
            runner._validate_internal_gate(config, "smoke", manifest_sha)

    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"][1]["source_state"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(runner.RunnerError, "SHA256 mismatch"):
            self.normalized(manifest)

    def test_source_resume_pair_must_be_adjacent_and_matching(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["seeds"][0]["source_snapshot"] = manifest["seeds"][1][
            "source_snapshot"
        ]
        with self.assertRaisesRegex(runner.RunnerError, "matching adjacent"):
            self.normalized(manifest)

    def test_safe_child_rejects_escape_components(self) -> None:
        with self.assertRaises(runner.RunnerError):
            runner._safe_child(self.run_root, "..", "escape")
        with self.assertRaises(runner.RunnerError):
            runner._safe_child(self.run_root, "/absolute")

    def test_result_gate_requires_exact_expected_value(self) -> None:
        receipt = self.root / "result.json"
        receipt.write_text('{"verdict":"NO-GO"}\n', encoding="utf-8")
        job = {
            "job_id": "comparison",
            "result_gate": {
                "path": str(receipt),
                "verdict_key": "verdict",
                "expected": "GO",
            },
        }
        with self.assertRaisesRegex(runner.RunnerError, "result gate failed"):
            runner._verify_result_gate(job)

    def test_execute_job_records_logs_gpu_environment_hashes_and_is_resumable(
        self,
    ) -> None:
        config = self.normalized()
        input_path = self.root / "input.bin"
        output_path = self.root / "output.bin"
        input_path.write_bytes(b"input")
        record_dir = self.run_root / "_runner" / "commands" / "unit" / "write-output"
        command = [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'output')",
            str(output_path),
        ]
        job = runner._make_job(
            job_id="unit-write-output",
            kind="validate",
            phase="smoke",
            seed=3,
            gpu=0,
            command=command,
            record_dir=record_dir,
            input_artifacts=[self.artifact(input_path)],
            output_artifacts=[{"path": str(output_path), "required": True}],
        )
        job["fingerprint"] = runner.sha256_bytes(runner.canonical_json(job).encode())
        gpu = {
            "index": 0,
            "name": "A100",
            "memory_free_mib": 80000,
            "memory_total_mib": 81920,
            "utilization_pct": 0,
            "raw": "0, A100, 80000, 81920, 0",
        }
        with mock.patch.object(runner, "_query_gpu", return_value=gpu):
            first = runner.execute_job(config, job, manifest_sha256="d" * 64)
            second = runner.execute_job(config, job, manifest_sha256="d" * 64)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["job_fingerprint"], first["job_fingerprint"])
        self.assertEqual(first["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(first["environment"]["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(first["environment"]["MASTER_PORT"], "29630")
        self.assertEqual(first["environment"]["WORLD_SIZE"], "1")
        self.assertEqual(first["outputs"][0]["sha256"], runner.sha256_file(output_path))
        for name in ("command.json", "stdout.log", "stderr.log", "exit.json"):
            self.assertTrue((record_dir / name).is_file())

    def test_train_never_reuses_an_existing_output_directory(self) -> None:
        config = self.normalized()
        run_dir = self.run_root / "formal" / "seed3" / "G" / "512k"
        run_dir.mkdir(parents=True)
        job = runner._make_job(
            job_id="unit-existing-train",
            kind="train",
            phase="formal",
            seed=3,
            gpu=0,
            command=[sys.executable, "-c", "raise SystemExit(99)"],
            record_dir=self.run_root / "_runner" / "commands" / "unit" / "existing",
            output_artifacts=runner._train_output_artifacts(run_dir),
        )
        job["fingerprint"] = runner.sha256_bytes(runner.canonical_json(job).encode())
        with self.assertRaisesRegex(runner.RunnerError, "existing output directory"):
            runner.execute_job(config, job, manifest_sha256="e" * 64)
        self.assertFalse(Path(job["record_dir"]).exists())

    def test_git_provenance_requires_exact_head_and_clean_worktree(self) -> None:
        config = self.normalized()
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=["a" * 40 + "\n", ""],
        ):
            runner.verify_git_provenance(config)
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=["a" * 40 + "\n", "?? untracked\n"],
        ):
            with self.assertRaisesRegex(runner.RunnerError, "not clean"):
                runner.verify_git_provenance(config)

    def test_lock_is_exclusive(self) -> None:
        config = self.normalized()
        with runner.phase_lock(config, "formal-gpu0"):
            with self.assertRaisesRegex(runner.RunnerError, "holds lock"):
                with runner.phase_lock(config, "formal-gpu0"):
                    pass

    def test_tmux_launch_uses_one_recoverable_worker_per_gpu(self) -> None:
        config = self.normalized()
        plan = runner.build_plan(config, "1" * 64, "formal")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/tmux"),
            mock.patch.object(runner, "_tmux_has_session", return_value=False),
            mock.patch.object(
                runner,
                "_tmux_pane_state",
                return_value={
                    "dead": False,
                    "start_command": "runner",
                    "pane_pid": "1",
                    "dead_status": "",
                },
            ),
            mock.patch.object(
                runner.subprocess, "run", return_value=completed
            ) as run_mock,
        ):
            launches = runner.launch_tmux_workers(
                config, self.root / "execution_manifest.json", plan, "formal"
            )
        self.assertEqual([item["gpu"] for item in launches], [0, 1])
        self.assertEqual(run_mock.call_count, 2)
        for call in run_mock.call_args_list:
            command = call.args[0]
            self.assertEqual(command[0], "tmux")
            self.assertIn("new-session", command)
            self.assertNotIn("-c", command)
            self.assertIn("--worker-gpu", command[-1])
            self.assertIn("--expected-manifest-sha256", command[-1])
        for launch in launches:
            self.assertIn("--worker-gpu", launch["command"])
            self.assertIn("--expected-manifest-sha256", launch["command"])
            self.assertIn("--worker-record-dir", launch["command"])
            self.assertIn("formal", launch["command"])

    def test_existing_tmux_session_must_match_frozen_live_pane(self) -> None:
        config = self.normalized()
        plan = runner.build_plan(config, "1" * 64, "formal")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/tmux"),
            mock.patch.object(runner, "_tmux_has_session", return_value=False),
            mock.patch.object(runner.subprocess, "run", return_value=completed),
        ):
            launched = runner.launch_tmux_workers(
                config, self.root / "execution_manifest.json", plan, "formal"
            )
        launch_path = Path(launched[0]["worker_record_dir"]) / "launch.json"
        identity = json.loads(launch_path.read_text(encoding="utf-8"))
        pane = {
            "dead": False,
            "start_command": identity["shell_command"],
            "pane_pid": "12345",
            "dead_status": "",
        }
        one_gpu_plan = copy.deepcopy(plan)
        one_gpu_plan["jobs"] = [job for job in plan["jobs"] if job["gpu"] == 0]
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/tmux"),
            mock.patch.object(runner, "_tmux_has_session", return_value=True),
            mock.patch.object(runner, "_tmux_pane_state", return_value=pane),
            mock.patch.object(runner.subprocess, "run") as run_mock,
        ):
            resumed = runner.launch_tmux_workers(
                config, self.root / "execution_manifest.json", one_gpu_plan, "formal"
            )
        self.assertEqual(resumed[0]["status"], "already_running")
        self.assertEqual(resumed[0]["pane_pid"], "12345")
        run_mock.assert_not_called()

        pane["start_command"] = "unrelated command"
        with (
            mock.patch.object(runner.shutil, "which", return_value="/usr/bin/tmux"),
            mock.patch.object(runner, "_tmux_has_session", return_value=True),
            mock.patch.object(runner, "_tmux_pane_state", return_value=pane),
        ):
            with self.assertRaisesRegex(
                runner.RunnerError, "frozen live worker command"
            ):
                runner.launch_tmux_workers(
                    config,
                    self.root / "execution_manifest.json",
                    one_gpu_plan,
                    "formal",
                )

    def test_external_tmux_can_run_one_gpu_foreground_worker(self) -> None:
        config = self.normalized()
        manifest_sha = "1" * 64
        plan = runner.build_plan(config, manifest_sha, "smoke")
        argv = [
            "--manifest",
            str(self.root / "execution_manifest.json"),
            "--phase",
            "smoke",
            "--foreground",
            "--worker-gpu",
            "0",
            "--expected-manifest-sha256",
            manifest_sha,
        ]
        with (
            mock.patch.object(
                runner, "load_manifest", return_value=(config, manifest_sha)
            ),
            mock.patch.object(runner, "build_plan", return_value=plan),
            mock.patch.object(runner, "verify_git_provenance"),
            mock.patch.object(runner, "_validate_prepare_receipt"),
            mock.patch.object(runner, "verify_external_gates"),
            mock.patch.object(runner, "_write_or_verify_json"),
            mock.patch.object(runner, "run_worker") as worker,
        ):
            self.assertEqual(runner.main(argv), 0)
        worker.assert_called_once_with(config, plan, manifest_sha, phase="smoke", gpu=0)
        attempts = sorted(
            (self.run_root / "_runner" / "workers" / "smoke-gpu0").glob("attempt-*")
        )
        self.assertEqual(len(attempts), 1)
        launch = json.loads((attempts[0] / "launch.json").read_text(encoding="utf-8"))
        self.assertEqual(launch["dispatch"], "external-host-tmux-foreground")
        self.assertEqual(launch["manifest_sha256"], manifest_sha)
        self.assertTrue((attempts[0] / "started.json").is_file())
        exit_receipt = json.loads(
            (attempts[0] / "exit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(exit_receipt["status"], "completed")

    def test_concurrent_identical_plan_publication_is_accepted(self) -> None:
        path = self.root / "plans" / "formal.json"
        payload = {"schema": "plan/v1", "jobs": [{"job_id": "one"}]}

        def publish_winner_then_report_collision(
            target: Path, candidate: dict[str, object]
        ) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise runner.RunnerError("refusing to overwrite receipt")

        with mock.patch.object(
            runner, "_write_new_json", side_effect=publish_winner_then_report_collision
        ):
            runner._write_or_verify_json(path, payload)

        path.unlink()
        different = {"schema": "plan/v1", "jobs": [{"job_id": "other"}]}

        def publish_different_then_report_collision(
            target: Path, _candidate: dict[str, object]
        ) -> None:
            target.write_text(
                json.dumps(different, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise runner.RunnerError("refusing to overwrite receipt")

        with mock.patch.object(
            runner,
            "_write_new_json",
            side_effect=publish_different_then_report_collision,
        ):
            with self.assertRaisesRegex(runner.RunnerError, "differs"):
                runner._write_or_verify_json(path, payload)


if __name__ == "__main__":
    unittest.main()
