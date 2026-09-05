import csv
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from scripts import build_m1_evaluation_slots as slots
from scripts import run_m1_evaluation_job as worker
from scripts import seal_m1_evaluation_slots as sealer
from scripts import summarize_m1_results as summary
from scripts import validate_m1_evaluation_job as validator
from tests.test_m1_evaluation_analysis import frozen_inventory, frozen_training_identity


class M1EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.training = frozen_training_identity()
        self.training["output_root"] = str(self.root / "training")
        self.training["roster"] = slots.normalize_roster(frozen_inventory())
        self.checkout_patch = mock.patch.object(
            validator, "verify_implementation_checkout",
            return_value={
                "head": self.training["implementation_commit"], "clean": True,
            },
        )
        self.checkout_patch.start()
        self.rows = slots.build_slots(
            slots.normalize_roster(frozen_inventory()), self.training
        )
        self.manifest = self.root / "slots.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=slots.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        self.snapshot = self.root / "snapshot.pkl"
        self.snapshot.write_bytes(b"M1 evaluator snapshot")
        self.terminal = self.root / "training-state.pt"
        self.terminal.write_bytes(b"M1 terminal state")

    def tearDown(self):
        self.checkout_patch.stop()
        self.temporary.cleanup()

    def slot(self, slot_id):
        return validator.load_slot(self.manifest, slot_id)

    def snapshot_receipt(self, slot, **updates):
        branch_manifest = self.root / f"{slot['roster_slot']}-{slot['branch']}-manifest.json"
        branch_manifest.write_text(
            json.dumps({
                "experiment_protocol": slots.PROTOCOL_ID,
                "seed": int(slot["seed"]),
                "branch": slot["branch"],
                "training_manifest_sha256": self.training["training_manifest_sha256"],
                "implementation_commit": self.training["implementation_commit"],
                "source_state": {"sha256": slot["frozen_source_state_sha256"]},
            }),
            encoding="utf-8",
        )
        training_slot_receipt = (
            Path(self.training["output_root"])
            / slot["roster_slot"] / "training_receipt.json"
        )
        training_slot_receipt.parent.mkdir(parents=True, exist_ok=True)
        training_slot_receipt.write_text(json.dumps({
            "schema": "ect.m1.training-slot/v1",
            "status": "PASS",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "roster_slot": slot["roster_slot"],
            "seed": int(slot["seed"]),
            "branches": {
                slot["branch"]: {
                    "status": "PASS",
                    "milestones": {"1024": {
                        "state_path": str(self.terminal.resolve()),
                        "state_sha256": validator.sha256_file(self.terminal),
                        "attempted_iteration": 8_000,
                        "cur_nimg": 1_024_000,
                    }},
                },
            },
        }), encoding="utf-8")
        training_slot = {
            "roster_slot": slot["roster_slot"],
            "training_slot_receipt_path": str(training_slot_receipt.resolve()),
            "training_slot_receipt_sha256": validator.sha256_file(
                training_slot_receipt
            ),
        }
        source_readout_sha256 = "c" * 64
        classifier = self.root / f"{slot['slot_id']}-classifier.json"
        classifier.write_text(json.dumps({
            "schema": validator.CLASSIFIER_SCHEMA,
            "status": "READOUT_VALID",
            "protocol_id": slots.PROTOCOL_ID,
            "classification": "FINITE_READOUT",
            "fixed_input": True, "fixed_input_executed": True,
            "fixed_input_spec": {
                "x": {"shape": [1, 3, 32, 32], "dtype": "float32", "fill_value": 0.0},
                "sigma": {"shape": [1], "dtype": "float32", "fill_value": 1.0},
                "class_labels": None, "force_fp32": True,
                "model_mode": "eval", "autograd": False, "device": "cuda:0",
            },
            "nonfinite_state_tensor_paths": [],
            "fixed_input_forward_error": None,
            "output_shape": [1, 3, 32, 32], "output_dtype": "float32",
            "output_nonfinite_count": 0, "invalid_fields": [],
            "source_attempted_iteration": 8_000,
            "source_cur_nimg": 1_024_000,
            "source_readout_sha256": source_readout_sha256,
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "implementation_commit": self.training["implementation_commit"],
            "implementation_checkout": {
                "head": self.training["implementation_commit"], "clean": True,
            },
            "seed": int(slot["seed"]), "branch": slot["branch"],
            "readout": slot["readout"],
            "frozen_source_state_sha256": slot["frozen_source_state_sha256"],
            "terminal_state_path": str(self.terminal.resolve()),
            "terminal_state_sha256": validator.sha256_file(self.terminal),
            "branch_manifest_path": str(branch_manifest.resolve()),
            "branch_manifest_sha256": validator.sha256_file(branch_manifest),
            **training_slot,
        }), encoding="utf-8")
        payload = {
            "schema": validator.EXPORT_SCHEMA,
            "status": "PASS",
            "protocol_id": slots.PROTOCOL_ID,
            "seed": int(slot["seed"]),
            "branch": slot["branch"],
            "readout": slot["readout"],
            "source_attempted_iteration": 8_000,
            "source_cur_nimg": 1_024_000,
            "quality_eligible": True,
            "gate_state": False,
            "source_state_path": str(self.terminal.resolve()),
            "terminal_state_sha256": validator.sha256_file(self.terminal),
            "branch_manifest_path": str(branch_manifest.resolve()),
            "branch_manifest_sha256": validator.sha256_file(branch_manifest),
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "implementation_commit": self.training["implementation_commit"],
            "implementation_checkout": {
                "head": self.training["implementation_commit"], "clean": True,
            },
            "frozen_source_state_sha256": slot["frozen_source_state_sha256"],
            "snapshot_path": str(self.snapshot),
            "snapshot_sha256": validator.sha256_file(self.snapshot),
            "source_readout_sha256": source_readout_sha256,
            "snapshot_readout_sha256": source_readout_sha256,
            "classifier_receipt_path": str(classifier),
            "classifier_receipt_sha256": validator.sha256_file(classifier),
            **training_slot,
        }
        payload.update(updates)
        path = self.root / f"{slot['slot_id']}-export.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def output(self, slot, include_kid=True, sample_range=None, complete_log=True):
        job = self.root / f"job-{slot['slot_id']}"
        job.mkdir()
        log = "done\nExiting...\n" if complete_log else "KID failed after FID artifact\n"
        (job / "log.txt").write_text(log, encoding="utf-8")
        (job / "generated-samples.npy").write_bytes(b"samples")
        start, end = sample_range or (
            int(slot["sample_seed_start"]), int(slot["sample_seed_end"])
        )
        options = {
            "sample_seeds": list(range(start, end + 1)),
            "seed": int(slot["metric_seed"]),
            "metrics": ["kid50k_full", "fid50k_full"],
            "metric_repeats": 1,
            "metric_generator_batch": 128,
            "retain_generated_artifacts": True,
            "mid_t": [],
            "network_kwargs": {"use_fp16": False},
            "resume_pkl": str(self.snapshot),
            "dataset_kwargs": {"path": str(self.root / "dataset.zip")},
        }
        (job / "training_options.json").write_text(json.dumps(options), encoding="utf-8")
        (job / "generated-features-fid50k_full-repeat00.npy").write_bytes(b"features")
        fid = {"metric": "fid50k_full", "num_gpus": 1, "results": {"fid50k_full": 3.25}}
        (job / "metric-fid50k_full.jsonl").write_text(json.dumps(fid) + "\n", encoding="utf-8")
        if include_kid:
            (job / "generated-features-kid50k_full-repeat00.npy").write_bytes(b"features")
            kid = {"metric": "kid50k_full", "num_gpus": 1, "results": {"kid50k_full": -0.0001}}
            (job / "metric-kid50k_full.jsonl").write_text(json.dumps(kid) + "\n", encoding="utf-8")
        (self.root / "dataset.zip").write_bytes(b"dataset")
        return job

    def test_worker_builds_exact_b0_b1_b2_ranges(self):
        for block, expected_range in slots.BLOCKS.items():
            slot = self.slot(f"S01-R_A-E_512-{block}")
            command = worker.build_command(
                slot, str(self.snapshot), self.root / "dataset.zip", self.root / "out",
                self.root / "evaluator", self.root / "runtime-python", 52000,
            )
            self.assertIn(f"--sample-seeds={expected_range[0]}-{expected_range[1]}", command)
            self.assertIn("--nfe=1", command)
            self.assertIn("--fp16=False", command)
            self.assertNotIn("--mid_t", command)

    def test_export_receipt_must_match_readout_identity(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(slot, readout="E_KEEP")
        with self.assertRaisesRegex(validator.ValidationError, "readout"):
            validator.load_snapshot_receipt(receipt, slot, self.training)

    def test_export_receipt_must_bind_training_and_terminal_state(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(
            slot, training_manifest_sha256="0" * 64
        )
        with self.assertRaisesRegex(validator.ValidationError, "training_manifest"):
            validator.load_snapshot_receipt(receipt, slot, self.training)
        receipt = self.snapshot_receipt(
            slot, terminal_state_sha256="0" * 64
        )
        with self.assertRaisesRegex(validator.ValidationError, "terminal state"):
            validator.load_snapshot_receipt(receipt, slot, self.training)

    def test_formal_export_requires_matching_valid_classifier_receipt(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(slot)
        export = json.loads(receipt.read_text(encoding="utf-8"))
        classifier_path = Path(export["classifier_receipt_path"])
        classifier = json.loads(classifier_path.read_text(encoding="utf-8"))
        classifier.update({
            "status": "SCIENTIFIC_READOUT_INVALID",
            "classification": "NONFINITE_FIXED_INPUT_OUTPUT",
            "output_nonfinite_count": 1,
            "invalid_fields": ["fixed_input_output"],
        })
        classifier_path.write_text(json.dumps(classifier), encoding="utf-8")
        export["classifier_receipt_sha256"] = validator.sha256_file(classifier_path)
        receipt.write_text(json.dumps(export), encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "classifier receipt"):
            validator.load_snapshot_receipt(receipt, slot, self.training)

    def test_formal_export_rejects_noncanonical_1024_state(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(slot)
        training_receipt = (
            Path(self.training["output_root"])
            / slot["roster_slot"] / "training_receipt.json"
        )
        payload = json.loads(training_receipt.read_text(encoding="utf-8"))
        payload["branches"][slot["branch"]]["milestones"]["1024"][
            "state_sha256"
        ] = "0" * 64
        training_receipt.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "canonical 1024"):
            validator.load_snapshot_receipt(receipt, slot, self.training)

        receipt = self.snapshot_receipt(slot)
        export = json.loads(receipt.read_text(encoding="utf-8"))
        del export["classifier_receipt_path"]
        del export["classifier_receipt_sha256"]
        receipt.write_text(json.dumps(export), encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "classifier receipt"):
            validator.load_snapshot_receipt(receipt, slot, self.training)

    def test_validator_rejects_b1_labeled_output_that_reused_b0(self):
        slot = self.slot("S01-R_A-E_512-B1")
        receipt = self.snapshot_receipt(slot)
        snapshot = validator.load_snapshot_receipt(receipt, slot, self.training)
        job = self.output(slot, sample_range=slots.BLOCKS["B0"])
        with self.assertRaisesRegex(validator.ValidationError, "sample_seeds"):
            validator.validate_output(slot, snapshot, job, self.root / "dataset.zip")

    def test_kid_failure_does_not_block_valid_fid(self):
        slot = self.slot("S01-R_A-E_512-B2")
        receipt = self.snapshot_receipt(slot)
        snapshot = validator.load_snapshot_receipt(receipt, slot, self.training)
        payload = validator.validate_output(
            slot, snapshot, self.output(slot, include_kid=False), self.root / "dataset.zip"
        )
        self.assertEqual(payload["status"], "SEALED_PARTIAL")
        self.assertEqual(payload["metrics"]["fid50k_full"]["status"], "SEALED_PASS")
        self.assertEqual(payload["metrics"]["kid50k_full"]["status"], "INCOMPLETE_TECHNICAL")
        self.assertEqual(payload["result_row"]["status"], "SEALED_PASS")

    def test_missing_completion_marker_prevents_sealed_fid(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(slot)
        snapshot = validator.load_snapshot_receipt(receipt, slot, self.training)
        payload = validator.validate_output(
            slot,
            snapshot,
            self.output(slot, include_kid=False, complete_log=False),
            self.root / "dataset.zip",
        )
        self.assertFalse(payload["log_completion_marker"])
        self.assertEqual(
            payload["metrics"]["fid50k_full"]["status"],
            "INCOMPLETE_TECHNICAL",
        )

    def test_nonzero_exit_or_timeout_prevents_all_metric_seals(self):
        slot = self.slot("S01-R_A-E_512-B0")
        receipt = self.snapshot_receipt(slot)
        snapshot = validator.load_snapshot_receipt(receipt, slot, self.training)
        job = self.output(slot)
        for exit_code, timed_out in ((7, False), (0, True)):
            payload = validator.validate_output(
                slot,
                snapshot,
                job,
                self.root / "dataset.zip",
                process_exit_code=exit_code,
                process_hard_timeout=timed_out,
            )
            self.assertEqual(payload["status"], "INCOMPLETE_TECHNICAL")
            self.assertTrue(all(
                metric["status"] == "INCOMPLETE_TECHNICAL"
                for metric in payload["metrics"].values()
            ))

    def test_complete_output_binds_range_count_nfe_precision_and_shared_features(self):
        slot = self.slot("S01-K_B-E_KEEP-B0")
        receipt = self.snapshot_receipt(slot)
        snapshot = validator.load_snapshot_receipt(receipt, slot, self.training)
        payload = validator.validate_output(
            slot, snapshot, self.output(slot), self.root / "dataset.zip"
        )
        self.assertEqual(payload["status"], "SEALED_PASS")
        self.assertEqual(payload["sample_seed_start"], 0)
        self.assertEqual(payload["sample_seed_end"], 49_999)
        self.assertEqual(payload["sample_count"], 50_000)
        self.assertEqual(payload["nfe"], 1)
        self.assertEqual(payload["precision"], "fp32")
        self.assertTrue(payload["kid_fid_shared_features"])
        self.assertEqual(payload["evaluator_commit"], slots.EVALUATOR_COMMIT)

    def test_runtime_environment_uses_py311_layout_not_py310(self):
        base = self.root / "runtime-base"
        environment = self.root / "runtime-env"
        cache = self.root / "cache"
        with mock.patch.dict(
            "os.environ",
            {"PYTHONPATH": "/shadow/modules", "PYTHONHOME": "/shadow/runtime"},
        ):
            env = worker.runtime_env(base, environment, cache, 0, 52000)
        self.assertIn("lib/python3.11/site-packages/torch/lib", env["LD_LIBRARY_PATH"])
        self.assertNotIn("python3.10", env["LD_LIBRARY_PATH"])
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertTrue(
            env["PATH"].startswith(
                f"{environment.resolve()}/bin:{base.resolve()}/bin:"
            )
        )

    def test_runtime_receipt_binds_py311_torch26_archive_and_executable(self):
        base = self.root / "runtime-base"
        environment = self.root / "runtime-env"
        (base / "lib/python3.11/site-packages/torch/lib").mkdir(parents=True)
        (environment / "bin").mkdir(parents=True)
        python = environment / "bin/python"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        archive = self.root / "runtime.tar.gz"
        archive.write_bytes(b"py311 torch260 runtime")
        freeze = self.root / "original-pip-freeze.txt"
        freeze.write_text("torch==2.6.0\n", encoding="utf-8")
        receipt = self.root / "runtime.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "ect.q256.training-compatible-evaluation-runtime/v1",
                    "status": "PASS",
                    "archive_path": str(archive),
                    "archive_sha256": validator.sha256_file(archive),
                    "runtime_probe": {
                        "python": "3.11.13",
                        "torch": "2.6.0+cu124",
                        "torch_cuda": "12.4",
                        "numpy": "2.1.2",
                        "scipy": "1.16.1",
                    },
                    "pip_freeze_path": str(freeze),
                    "pip_freeze_sha256": validator.sha256_file(freeze),
                }
            ),
            encoding="utf-8",
        )
        bound = validator.verify_runtime(base, environment, receipt)
        self.assertEqual(bound["runtime_python"], str(python.resolve()))
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        payload["runtime_probe"]["python"] = "3.10.12"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "Python 3.11"):
            validator.verify_runtime(base, environment, receipt)

    def test_rebuilt_single_prefix_runtime_binds_python311_and_pip_freeze(self):
        prefix = self.root / "rebuilt-prefix"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "lib/python3.11/site-packages/torch/lib").mkdir(parents=True)
        python = prefix / "bin/python3.11"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        freeze = self.root / "pip-freeze.txt"
        freeze.write_text("torch==2.6.0\n", encoding="utf-8")
        receipt = self.root / "rebuilt-runtime.json"
        receipt.write_text(json.dumps({
            "schema": "ect.m1.rebuilt-training-runtime/v1",
            "status": "PASS", "runtime_origin": "REBUILT_NOT_BYTE_IDENTICAL",
            "runtime_probe": {
                "python": "3.11.13", "torch": "2.6.0+cu124", "cuda": "12.4",
                "cudnn": 90100, "numpy": "2.1.2", "scipy": "1.16.1",
            },
            "pip_freeze": {
                "path": str(freeze), "sha256": validator.sha256_file(freeze),
            },
        }), encoding="utf-8")
        bound = validator.verify_runtime(prefix, prefix, receipt)
        self.assertEqual(bound["runtime_python"], str(python.resolve()))
        self.assertEqual(bound["runtime_origin"], "REBUILT_NOT_BYTE_IDENTICAL")

        other = self.root / "other-prefix"
        other.mkdir()
        with self.assertRaisesRegex(validator.ValidationError, "one explicitly"):
            validator.verify_runtime(prefix, other, receipt)
        freeze.write_text("torch==2.6.1\n", encoding="utf-8")
        with self.assertRaisesRegex(validator.ValidationError, "pip-freeze"):
            validator.verify_runtime(prefix, prefix, receipt)

    def test_live_runtime_probe_requires_exact_versions_and_cuda(self):
        observed = {**validator.EXPECTED_RUNTIME_PROBE, "cuda_available": True}
        with mock.patch.object(
            validator.subprocess, "check_output", return_value=json.dumps(observed)
        ):
            self.assertEqual(
                validator.probe_live_runtime(Path("/runtime/bin/python"), {}),
                observed,
            )

    def test_live_runtime_probe_rejects_pip_freeze_drift(self):
        observed = {**validator.EXPECTED_RUNTIME_PROBE, "cuda_available": True}
        frozen = b"torch==2.6.0\nnumpy==2.1.2\n"
        canonical = b"numpy==2.1.2\ntorch==2.6.0\n"
        with mock.patch.object(
            validator.subprocess, "check_output",
            side_effect=[json.dumps(observed), frozen],
        ):
            result = validator.probe_live_runtime(
                Path("/runtime/bin/python"), {},
                hashlib.sha256(canonical).hexdigest(),
            )
        self.assertEqual(
            result["pip_freeze_sha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        with mock.patch.object(
            validator.subprocess, "check_output",
            side_effect=[json.dumps(observed), frozen],
        ), self.assertRaisesRegex(validator.ValidationError, "pip-freeze"):
            validator.probe_live_runtime(
                Path("/runtime/bin/python"), {}, "0" * 64
            )

    def test_training_and_evaluation_zip_hashes_are_distinct(self):
        self.assertNotEqual(validator.TRAINING_DATASET_SHA256, validator.DATASET_SHA256)
        dataset = self.root / "dataset.zip"
        dataset.write_bytes(b"not used because hash is mocked")
        with mock.patch.object(
            validator, "sha256_file", return_value=validator.TRAINING_DATASET_SHA256
        ):
            with self.assertRaisesRegex(validator.ValidationError, "training ZIP"):
                validator.verify_evaluation_dataset(dataset)

    def test_frozen_evaluator_checks_commit_cleanliness_and_ct_eval_hash(self):
        evaluator = self.root / "evaluator"
        (evaluator / ".git").mkdir(parents=True)
        (evaluator / "ct_eval.py").write_bytes(b"frozen evaluator")
        with mock.patch.object(
            validator.subprocess,
            "check_output",
            side_effect=[slots.EVALUATOR_COMMIT + "\n", ""],
        ), mock.patch.object(
            validator,
            "sha256_file",
            return_value=validator.EVALUATOR_CT_EVAL_SHA256,
        ):
            validator.verify_evaluator(evaluator, slots.EVALUATOR_COMMIT)

    def test_adjacent_archive_cannot_vouch_for_an_unbound_evaluator_tree(self):
        evaluator = self.root / "unbound-evaluator"
        evaluator.mkdir()
        (evaluator / "ct_eval.py").write_bytes(b"frozen evaluator")
        archive = self.root / "evaluator.tar"
        archive.write_bytes(b"verified archive beside a different tree")
        with mock.patch.object(
            validator, "sha256_file",
            return_value=validator.EVALUATOR_CT_EVAL_SHA256,
        ), self.assertRaisesRegex(validator.ValidationError, "clean evaluator git"):
            validator.verify_evaluator(
                evaluator, slots.EVALUATOR_COMMIT, archive
            )

    def attempt_receipt(self, slot, attempt, fid_pass):
        payload = {
            "schema": validator.RECEIPT_SCHEMA,
            "status": "SEALED_PASS" if fid_pass else "INCOMPLETE_TECHNICAL",
            "slot_id": slot["slot_id"],
            "attempt": attempt,
        }
        if fid_pass:
            export = json.loads(
                self.snapshot_receipt(slot).read_text(encoding="utf-8")
            )
            payload.update(
                slot_index=int(slot["slot_index"]), seed=int(slot["seed"]),
                branch=slot["branch"], readout=slot["readout"], block=slot["block"],
                sample_seed_start=int(slot["sample_seed_start"]),
                sample_seed_end=int(slot["sample_seed_end"]),
                sample_count=int(slot["sample_count"]), nfe=int(slot["nfe"]),
                precision=slot["precision"], evaluator_commit=slot["evaluator_commit"],
                manifest_sha256=validator.sha256_file(self.manifest),
                terminal_state_path=export["source_state_path"],
                terminal_state_sha256=export["terminal_state_sha256"],
                branch_manifest_path=export["branch_manifest_path"],
                branch_manifest_sha256=export["branch_manifest_sha256"],
                training_manifest_sha256=export["training_manifest_sha256"],
                implementation_commit=export["implementation_commit"],
                implementation_checkout=export["implementation_checkout"],
                frozen_source_state_sha256=export["frozen_source_state_sha256"],
                training_runtime_receipt_sha256=self.training[
                    "training_runtime_receipt_sha256"
                ],
                live_runtime_probe={
                    **validator.EXPECTED_RUNTIME_PROBE, "cuda_available": True,
                    "pip_freeze_sha256": "5" * 64,
                },
                runtime_pip_freeze_sha256="5" * 64,
                process_exit_code=0,
                process_hard_timeout=False,
                log_completion_marker=True,
                result_row={
                    "slot_id": slot["slot_id"], "status": "SEALED_PASS",
                    "fid_status": "SEALED_PASS", "fid50k_full": 3.25,
                    "kid_status": "SEALED_PASS", "kid50k_full": 0.001,
                },
            )
        path = self.root / f"{slot['slot_id']}-attempt{attempt:02d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_retry_seal_selects_one_successful_attempt(self):
        slot = self.slot("S01-R_A-E_512-B0")
        self.attempt_receipt(slot, 0, False)
        self.attempt_receipt(slot, 1, True)
        attempts = sealer.load_attempts(
            self.root, slot, self.training, validator.sha256_file(self.manifest)
        )
        result = sealer.collapse_slot(slot, attempts, None)
        self.assertEqual(result["selected_attempt"], 1)
        self.assertEqual(result["fid_status"], "SEALED_PASS")

    def test_seal_rejects_cross_manifest_or_failed_process_success(self):
        slot = self.slot("S01-R_A-E_512-B0")
        path = self.attempt_receipt(slot, 0, True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(sealer.SealError, "provenance"):
            sealer.load_attempts(
                self.root, slot, self.training,
                validator.sha256_file(self.manifest),
            )

        payload["manifest_sha256"] = validator.sha256_file(self.manifest)
        payload["process_exit_code"] = 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(sealer.SealError, "provenance"):
            sealer.load_attempts(
                self.root, slot, self.training,
                validator.sha256_file(self.manifest),
            )

    def test_retry_after_success_is_rejected(self):
        slot = self.slot("S01-R_A-E_512-B0")
        self.attempt_receipt(slot, 0, True)
        self.attempt_receipt(slot, 1, False)
        attempts = sealer.load_attempts(
            self.root, slot, self.training, validator.sha256_file(self.manifest)
        )
        with self.assertRaisesRegex(sealer.SealError, "continued after"):
            sealer.collapse_slot(slot, attempts, None)

    def test_nontechnical_failure_cannot_be_retried(self):
        slot = self.slot("S01-R_A-E_512-B0")
        first = self.attempt_receipt(slot, 0, False)
        payload = json.loads(first.read_text(encoding="utf-8"))
        payload["status"] = "INVALID_IMPLEMENTATION"
        first.write_text(json.dumps(payload), encoding="utf-8")
        self.attempt_receipt(slot, 1, True)
        attempts = sealer.load_attempts(
            self.root, slot, self.training, validator.sha256_file(self.manifest)
        )
        with self.assertRaisesRegex(sealer.SealError, "technical failures"):
            sealer.collapse_slot(slot, attempts, None)

    def test_attempt_gap_is_rejected(self):
        slot = self.slot("S01-R_A-E_512-B0")
        self.attempt_receipt(slot, 1, True)
        with self.assertRaisesRegex(sealer.SealError, "attempt gap"):
            sealer.load_attempts(
                self.root, slot, self.training, validator.sha256_file(self.manifest)
            )

    def test_no_quality_canary_only_loads_snapshot_and_uses_ct_eval_dry_run(self):
        command = ["python", "ct_eval.py", "--sample-seeds=0-49999"]
        with mock.patch.object(
            validator,
            "probe_live_runtime",
            return_value={**validator.EXPECTED_RUNTIME_PROBE, "cuda_available": True},
        ), mock.patch.object(worker.subprocess, "run") as run:
            payload = worker.run_canary(
                command,
                str(self.snapshot),
                self.root / "runtime-env/bin/python",
                self.root,
                {},
            )
        self.assertEqual(payload["status"], "G4_NO_QUALITY_CANARY_PASS")
        self.assertEqual(run.call_count, 2)
        self.assertIn("--dry_run", run.call_args_list[1].args[0])

    @unittest.skip("obsolete G4 admission layer removed")
    def test_g4_gate_export_must_match_recorded_continuous_state(self):
        snapshot = {
            "source_state_path": "/gate/continuous.pt",
            "terminal_state_sha256": "a" * 64,
            "branch_manifest_sha256": "b" * 64,
        }
        gates = {"seeds": [{
            "seed": 50, "status": "PASS",
            "manifest_sha256_by_branch": {"R_A": "b" * 64},
            "artifacts": {"R_A_continuous_state": {
                "path": "/gate/continuous.pt", "sha256": "c" * 64,
            }},
        }]}
        with self.assertRaisesRegex(
            validator.ValidationError, "continuous state"
        ):
            g4.validate_gate_export(gates, 50, "R_A", snapshot)

    @unittest.skip("obsolete G4 admission layer removed")
    def test_short_gate_canaries_aggregate_for_formal_admission(self):
        training_runtime = self.root / "training-runtime.json"
        training_runtime.write_text("{}", encoding="utf-8")
        training_path = self.root / "training-manifest-g4.json"
        training_path.write_text(json.dumps({
            "schema": "ect.m1.training-run-manifest/v1",
            "experiment_protocol": slots.PROTOCOL_ID,
            "implementation_commit": "2" * 40,
            "output_root": str(self.root / "formal-training"),
            "runtime_receipt": {
                "path": str(training_runtime),
                "sha256": validator.sha256_file(training_runtime),
            },
            "roster": [
                {
                    "roster_slot": f"S{index + 1:02d}", "seed": seed,
                    "sources": {
                        arm: {"source_state_sha256": ("a" if arm == "A" else "b") * 64}
                        for arm in ("A", "B")
                    },
                }
                for index, seed in enumerate(range(50, 66))
            ],
        }), encoding="utf-8")
        training = slots.load_training_identity(training_path)
        rows = slots.build_slots(training["roster"], training)
        evaluation_manifest = self.root / "evaluation-g4.csv"
        with evaluation_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=slots.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        bad_evaluation_manifest = self.root / "evaluation-g4-wrong-source.csv"
        bad_rows = [dict(row) for row in rows]
        for row in bad_rows:
            if int(row["seed"]) == 50 and row["branch"].endswith("_A"):
                row["frozen_source_state_sha256"] = "9" * 64
        with bad_evaluation_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=slots.FIELDS)
            writer.writeheader()
            writer.writerows(bad_rows)
        with self.assertRaisesRegex(
            validator.ValidationError, "canonical training-bound"
        ):
            g4.validate_evaluation_manifest(bad_evaluation_manifest, training)
        self.training = training
        self.rows = rows
        self.manifest = evaluation_manifest
        formal_slot = self.slot("S01-R_A-E_512-B0")
        first_export = self.snapshot_receipt(
            formal_slot, source_attempted_iteration=4_032,
            source_cur_nimg=516_096, quality_eligible=False, gate_state=True,
        )
        branch_sha = json.loads(first_export.read_text(encoding="utf-8"))[
            "branch_manifest_sha256"
        ]
        gates_path = self.root / "g1-g3.json"
        gates_path.write_text(json.dumps({
            "schema": "ect.m1.training-gates/v1", "status": "PASS",
            "training_manifest_sha256": training["training_manifest_sha256"],
            "seeds": [{
                "seed": 50, "status": "PASS",
                "manifest_sha256_by_branch": {"R_A": branch_sha},
                "artifacts": {"R_A_continuous_state": {
                    "path": str(self.terminal.resolve()),
                    "sha256": validator.sha256_file(self.terminal),
                }},
            }],
        }), encoding="utf-8")
        dataset = self.root / "g4-dataset.zip"
        dataset.write_bytes(b"evaluation dataset")
        evaluator = self.root / "g4-evaluator"
        evaluator.mkdir()
        cache = self.root / "g4-cache"
        (cache / "downloads").mkdir(parents=True)
        receipts = self.root / "g4-receipts"
        receipts.mkdir()
        evaluation_runtime_receipt = self.root / "runtime-receipt.json"
        evaluation_runtime_receipt.write_text("{}", encoding="utf-8")
        runtime = {
            "runtime_base": str(self.root / "runtime-base"),
            "runtime_environment": str(self.root / "runtime-env"),
            "runtime_python": str(self.root / "runtime-env/bin/python"),
            "runtime_integrity_receipt": str(evaluation_runtime_receipt),
            "runtime_integrity_receipt_sha256": validator.sha256_file(
                evaluation_runtime_receipt
            ),
            "runtime_pip_freeze_sha256": "6" * 64,
            "runtime_origin": "REBUILT_NOT_BYTE_IDENTICAL",
        }
        classes = sorted(g4_sealer.EXPECTED_CLASSES)
        with mock.patch.object(
            validator, "verify_evaluator",
            return_value={"evaluator_commit": slots.EVALUATOR_COMMIT},
        ), mock.patch.object(
            validator, "verify_runtime", return_value=runtime,
        ), mock.patch.object(
            validator, "verify_evaluation_dataset",
            return_value=validator.DATASET_SHA256,
        ), mock.patch.object(
            worker, "gpu_resource_probe", return_value={
                "index": 0, "uuid": "GPU-test", "name": "A100",
                "free_mib": 40_000, "utilization_percent": 0,
            },
        ), mock.patch.object(
            worker, "run_canary", return_value={
                "status": "G4_NO_QUALITY_CANARY_PASS",
                "runtime_probe": {
                    **validator.EXPECTED_RUNTIME_PROBE, "cuda_available": True,
                    "pip_freeze_sha256": "6" * 64,
                },
            },
        ), mock.patch("builtins.print"):
            for readout, block in classes:
                gate_slot = self.slot(f"S01-R_A-{readout}-{block}")
                export = self.snapshot_receipt(
                    gate_slot, source_attempted_iteration=4_032,
                    source_cur_nimg=516_096, quality_eligible=False,
                    gate_state=True,
                )
                receipt = receipts / f"S01-R_A-{readout}-{block}-g4-canary.json"
                args = Namespace(
                    training_manifest=training_path,
                    evaluation_manifest=evaluation_manifest,
                    training_gates_receipt=gates_path,
                    gate_export_receipt=export, roster_slot="S01",
                    branch="R_A", readout=readout, block=block, gpu_index=0,
                    evaluator_repo=evaluator, evaluator_archive=None,
                    runtime_base=self.root / "runtime-base",
                    runtime_env=self.root / "runtime-env",
                    runtime_receipt=self.root / "runtime-receipt.json",
                    cache_root=cache, evaluation_dataset=dataset,
                    output_root=self.root / "g4-output", receipt=receipt,
                )
                self.assertEqual(g4.run(args), 0)
        with mock.patch.object(
            validator, "verify_evaluator",
            return_value={"evaluator_commit": slots.EVALUATOR_COMMIT},
        ), mock.patch.object(
            validator, "verify_runtime", return_value=runtime,
        ), mock.patch.object(
            validator, "verify_evaluation_dataset",
            return_value=validator.DATASET_SHA256,
        ):
            seal = g4_sealer.seal(
                training, gates_path,
                validator.sha256_file(evaluation_manifest), receipts,
            )
        self.assertEqual(seal["status"], "PASS")
        self.assertEqual(seal["canary_count"], 5)
        self.assertFalse(seal["quality_generation"])
        self.assertEqual(
            seal["evaluation_manifest_sha256"],
            validator.sha256_file(evaluation_manifest),
        )
        child = Path(seal["canary_receipts"][0]["path"])
        child_payload = json.loads(child.read_text(encoding="utf-8"))
        child_payload["snapshot"]["terminal_state_sha256"] = "0" * 64
        child.write_text(json.dumps(child_payload), encoding="utf-8")
        with self.assertRaisesRegex(g4_sealer.G4SealError, "snapshot fields"):
            g4_sealer.seal(
                training, gates_path,
                validator.sha256_file(evaluation_manifest), receipts,
            )

    def test_attempt_three_receipt_is_rejected_by_full_seal(self):
        unexpected = self.root / "S01-R_A-E_512-B0-attempt03.json"
        unexpected.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(sealer.SealError, "unexpected attempt"):
            sealer.seal_rows(
                self.rows, self.root, {}, self.training,
                validator.sha256_file(self.manifest),
            )

    def test_full_seal_emits_one_missing_row_per_unattempted_slot(self):
        rows = sealer.seal_rows(
            self.rows, self.root, {}, self.training,
            validator.sha256_file(self.manifest),
        )
        self.assertEqual(len(rows), 320)
        self.assertTrue(all(row["fid_status"] == "MISSING_RESULT" for row in rows))
        report = summary.summarize(self.rows, rows)
        self.assertEqual(report["matrix_status"], "INCOMPLETE_SLOT_LEDGER")

    def test_scientific_training_failure_mechanically_expands_whole_branch(self):
        log = self.root / "scientific-attempt.log"
        log.write_text(
            "FloatingPointError: non-finite online-EMA distance\n",
            encoding="utf-8",
        )
        branch_manifest = self.root / "scientific-branch-manifest.json"
        source_sha = self.training["sources"][(50, "A")]
        branch_manifest.write_text(json.dumps({
            "seed": 50, "branch": "R_A",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "implementation_commit": self.training["implementation_commit"],
            "source_state": {"sha256": source_sha},
        }), encoding="utf-8")
        attempt = self.root / "scientific-attempt.json"
        attempt.write_text(json.dumps({
            "schema": "ect.m1.training-attempt/v1",
            "status": "SCIENTIFIC_FAILURE", "reason": "NUMERIC_FLOATING_POINT",
            "seed": 50, "branch": "R_A",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "frozen_source_state_sha256": source_sha,
            "branch_manifest_path": str(branch_manifest),
            "branch_manifest_sha256": validator.sha256_file(branch_manifest),
            "log_path": str(log), "log_sha256": validator.sha256_file(log),
        }), encoding="utf-8")
        training_slot = self.root / "training-slot.json"
        training_slot.write_text(json.dumps({
            "schema": "ect.m1.training-slot/v1",
            "status": "COMPLETE_WITH_SCIENTIFIC_FAILURES",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "roster_slot": "S01", "seed": 50,
            "branches": {
                "R_A": {
                    "status": "SCIENTIFIC_FAILURE",
                    "attempt_receipt": str(attempt),
                },
                "K_A": {"status": "PASS"}, "K_B": {"status": "PASS"},
                "R_B": {"status": "PASS"},
            },
        }), encoding="utf-8")
        evidence = self.root / "scientific-evidence.json"
        evidence.write_text(json.dumps({
            "schema": "ect.m1.scientific-missing-evidence/v1",
            "receipts": [str(training_slot)],
        }), encoding="utf-8")
        with mock.patch.object(
            validator, "verify_implementation_checkout",
            return_value={
                "head": self.training["implementation_commit"], "clean": True,
            },
        ):
            terminal = sealer.load_scientific_terminal_rows(
                evidence, self.rows, self.training
            )
        self.assertEqual(len(terminal), 5)
        self.assertEqual(
            set(terminal),
            {row["slot_id"] for row in self.rows if row["roster_slot"] == "S01" and row["branch"] == "R_A"},
        )
        collapsed = sealer.collapse_slot(
            self.slot("S01-R_A-E_512-B0"), [],
            terminal["S01-R_A-E_512-B0"],
        )
        self.assertEqual(collapsed["evidence_path"], str(attempt.resolve()))
        self.assertEqual(
            collapsed["evidence_sha256"], validator.sha256_file(attempt)
        )
        attempt_payload = json.loads(attempt.read_text(encoding="utf-8"))
        attempt_payload["status"] = "INCOMPLETE_TECHNICAL"
        attempt.write_text(json.dumps(attempt_payload), encoding="utf-8")
        with self.assertRaisesRegex(sealer.SealError, "attempt receipt"):
            sealer.load_scientific_terminal_rows(
                evidence, self.rows, self.training
            )

    def test_readout_invalid_evidence_expands_all_planned_blocks(self):
        terminal_state = self.root / "invalid-terminal.pt"
        terminal_state.write_bytes(b"terminal")
        branch_manifest = self.root / "invalid-branch.json"
        branch_manifest.write_text(json.dumps({
            "seed": 50, "branch": "R_A",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "implementation_commit": self.training["implementation_commit"],
            "source_state": {"sha256": "a" * 64},
        }), encoding="utf-8")
        training_slot_receipt = (
            Path(self.training["output_root"]) / "S01" / "training_receipt.json"
        )
        training_slot_receipt.parent.mkdir(parents=True, exist_ok=True)
        training_slot_receipt.write_text(json.dumps({
            "schema": "ect.m1.training-slot/v1", "status": "PASS",
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "roster_slot": "S01", "seed": 50,
            "branches": {"R_A": {"status": "PASS", "milestones": {
                "1024": {
                    "state_path": str(terminal_state.resolve()),
                    "state_sha256": validator.sha256_file(terminal_state),
                    "attempted_iteration": 8_000, "cur_nimg": 1_024_000,
                },
            }}},
        }), encoding="utf-8")
        invalid = self.root / "readout-invalid.json"
        invalid.write_text(json.dumps({
            "schema": validator.CLASSIFIER_SCHEMA,
            "status": "SCIENTIFIC_READOUT_INVALID",
            "protocol_id": slots.PROTOCOL_ID,
            "training_manifest_sha256": self.training["training_manifest_sha256"],
            "implementation_commit": self.training["implementation_commit"],
            "implementation_checkout": {
                "head": self.training["implementation_commit"], "clean": True,
            },
            "classification": "NONFINITE_FIXED_INPUT_OUTPUT",
            "fixed_input": True, "fixed_input_executed": True,
            "fixed_input_spec": {
                "x": {"shape": [1, 3, 32, 32], "dtype": "float32", "fill_value": 0.0},
                "sigma": {"shape": [1], "dtype": "float32", "fill_value": 1.0},
                "class_labels": None, "force_fp32": True,
                "model_mode": "eval", "autograd": False, "device": "cuda:0",
            },
            "nonfinite_state_tensor_paths": [],
            "output_shape": [1, 3, 32, 32], "output_dtype": "float32",
            "output_nonfinite_count": 1,
            "invalid_fields": ["fixed_input_output"],
            "seed": 50, "branch": "R_A", "readout": "E_512",
            "frozen_source_state_sha256": "a" * 64,
            "source_attempted_iteration": 8_000,
            "source_cur_nimg": 1_024_000,
            "source_readout_sha256": "c" * 64,
            "terminal_state_path": str(terminal_state),
            "terminal_state_sha256": validator.sha256_file(terminal_state),
            "branch_manifest_path": str(branch_manifest),
            "branch_manifest_sha256": validator.sha256_file(branch_manifest),
            "roster_slot": "S01",
            "training_slot_receipt_path": str(training_slot_receipt.resolve()),
            "training_slot_receipt_sha256": validator.sha256_file(
                training_slot_receipt
            ),
        }), encoding="utf-8")
        evidence = self.root / "readout-evidence.json"
        evidence.write_text(json.dumps({
            "schema": "ect.m1.scientific-missing-evidence/v1",
            "receipts": [str(invalid)],
        }), encoding="utf-8")
        terminal = sealer.load_scientific_terminal_rows(
            evidence, self.rows, self.training
        )
        self.assertEqual(
            set(terminal),
            {f"S01-R_A-E_512-{block}" for block in slots.BLOCKS},
        )
        payload = json.loads(invalid.read_text(encoding="utf-8"))
        payload["fixed_input_executed"] = False
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(
            validator, "verify_implementation_checkout",
            return_value={
                "head": self.training["implementation_commit"], "clean": True,
            },
        ), self.assertRaisesRegex(sealer.SealError, "fixed-input observation"):
            sealer.load_scientific_terminal_rows(evidence, self.rows, self.training)

        payload.update({
            "classification": "NONFINITE_READOUT_STATE",
            "fixed_input": False, "fixed_input_executed": False,
            "fixed_input_forward_error": {
                "type": "RuntimeError", "message": "nonfinite state forward failed",
            },
            "nonfinite_state_tensor_paths": ["model.weight"],
            "output_shape": None, "output_dtype": None,
            "output_nonfinite_count": None,
            "invalid_fields": ["state_dict:model.weight"],
        })
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(
            validator, "verify_implementation_checkout",
            return_value={
                "head": self.training["implementation_commit"], "clean": True,
            },
        ):
            direct = sealer.load_scientific_terminal_rows(
                evidence, self.rows, self.training
            )
        self.assertEqual(len(direct), 3)

    def test_scientific_terminal_accepts_only_prior_technical_attempts(self):
        slot = self.slot("S01-R_A-E_512-B0")
        technical_path = self.attempt_receipt(slot, 0, False)
        terminal = {
            "status": "SCIENTIFIC_READOUT_INVALID",
            "reason": "NONFINITE_FIXED_INPUT_OUTPUT",
            "evidence_path": str(self.root / "fixed-input.json"),
            "evidence_sha256": "a" * 64,
        }
        row = sealer.collapse_slot(
            slot, [(technical_path, json.loads(technical_path.read_text()))],
            terminal,
        )
        self.assertEqual(row["selected_attempt"], 0)
        self.assertEqual(row["receipt_path"], str(technical_path.resolve()))
        self.assertEqual(row["evidence_path"], terminal["evidence_path"])

        payload = json.loads(technical_path.read_text())
        payload["result_row"] = {"fid_status": "SEALED_PASS"}
        with self.assertRaisesRegex(sealer.SealError, "nontechnical attempt"):
            sealer.collapse_slot(slot, [(technical_path, payload)], terminal)
        payload["status"] = "SEALED_PASS"
        with self.assertRaisesRegex(sealer.SealError, "nontechnical attempt"):
            sealer.collapse_slot(slot, [(technical_path, payload)], terminal)


if __name__ == "__main__":
    unittest.main()
