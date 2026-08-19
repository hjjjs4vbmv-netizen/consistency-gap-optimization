import copy
import csv
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import verify_q256_target_weight_arm as verifier
from training import reproducibility


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0]]))
        self.register_buffer("counter", torch.tensor([3.0]))


class TinyLoss:
    def __init__(self, factorial):
        self.factorial = copy.deepcopy(factorial)
        self.schedule = SimpleNamespace(name="sigmoid")
        self.q = 256
        self.c = 0.0


def sampler_state(seed, consumed_samples):
    return {
        "schema": "ect.infinite-sampler/v1",
        "dataset_size": 50000,
        "rank": 0,
        "num_replicas": 1,
        "shuffle": True,
        "seed": seed,
        "window_size": 0.5,
        "consumed_samples": consumed_samples,
    }


def rng_state():
    return {
        "schema": reproducibility.RNG_STATE_SCHEMA,
        "python": random.Random(7).getstate(),
        "numpy": np.random.RandomState(8).get_state(),
        "torch_cpu": torch.Generator().manual_seed(9).get_state(),
        "torch_cuda_all": [],
        "torch_cuda_device_count": 0,
    }


class RunFixture:
    def __init__(self, root: Path, *, arm="A", skip_attempts=(2, 7)):
        self.root = root
        self.arm = arm
        self.seed = 3
        self.skip_attempts = set(skip_attempts)
        self.factorial = verifier.expected_factorial(arm)
        self.net = TinyNetwork()
        self.ema = copy.deepcopy(self.net)
        self.write_all()

    def options(self):
        target, denominator = verifier.ARMS[self.arm]
        return {
            "seed": self.seed,
            "total_kimg": 4,
            "batch_size": 128,
            "batch_gpu": 16,
            "enable_amp": True,
            "enable_tf32": False,
            "loss_scaling": 1.0,
            "ema_beta": 0.9993,
            "kimg_per_tick": 10,
            "snapshot_ticks": None,
            "state_dump_ticks": None,
            "ckpt_ticks": 10,
            "sample_ticks": 26,
            "double_ticks": 10000,
            "metrics": [],
            "resume_pkl": "/immutable/edm-cifar10.pkl",
            "loss_kwargs": {
                "class_name": "training.loss.ECMLoss",
                "factorial_protocol": verifier.PROTOCOL,
                "target_gap_scale": target,
                "denominator_gap_scale": denominator,
                "adj": "sigmoid",
                "global_gap_scale": 1.0,
                "q": 256,
                "c": 0.0,
                "k": 8.0,
                "b": 1.0,
            },
            "optimizer_kwargs": {
                "class_name": "torch.optim.RAdam",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "network_kwargs": {
                "class_name": "training.networks.ECMPrecond",
                "use_fp16": True,
                "dropout": 0.2,
            },
            "augment_kwargs": None,
            "dataset_kwargs": {
                "path": "/immutable/cifar10.zip",
                "use_labels": False,
                "xflip": False,
                "resolution": 32,
                "max_size": 50000,
            },
        }

    def initial_receipt(self):
        sampler = sampler_state(self.seed, 0)
        rng_sha = "1" * 64
        sampler_sha = reproducibility.state_sha256(sampler)
        module_sha = reproducibility.module_state_sha256(self.net)
        hashes = {
            "model": module_sha,
            "ema": module_sha,
            "optimizer": "2" * 64,
            "gradscaler": "3" * 64,
            "rank_rng": [rng_sha],
            "rank_sampler": [sampler_sha],
        }
        trajectory = self.trajectory_config()
        return {
            "schema": reproducibility.INITIAL_RECEIPT_SCHEMA,
            "seed": self.seed,
            "attempted_iteration": 0,
            "processed_nimg": 0,
            "factorial": copy.deepcopy(self.factorial),
            "dataset_path": "/immutable/cifar10.zip",
            "transfer_path": "/immutable/edm-cifar10.pkl",
            "world_size": 1,
            "batch_size": 128,
            "batch_gpu": 16,
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(
                trajectory
            ),
            "hashes": hashes,
            "common_initial_state_sha256": reproducibility.state_sha256(hashes),
            "rank_states": [{
                "rank": 0,
                "world_size": 1,
                "rng_sha256": rng_sha,
                "sampler_sha256": sampler_sha,
                "sampler_state": sampler,
            }],
        }

    def trajectory_config(self):
        return {
            "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
            "seed": self.seed,
            "batch_size": 128,
            "batch_gpu": 16,
            "loss_kwargs": copy.deepcopy(self.options()["loss_kwargs"]),
        }

    def telemetry_rows(self):
        rows = []
        skipped = 0
        scale = 65536.0
        for attempt in range(1, 33):
            is_skip = attempt in self.skip_attempts
            before = scale
            after = before / 2 if is_skip else before
            scale = after
            skipped += int(is_skip)
            elapsed = float(attempt)
            target, denominator = verifier.ARMS[self.arm]
            row = {
                "schema": verifier.TELEMETRY_SCHEMA,
                "protocol": verifier.PROTOCOL,
                "arm": self.arm,
                "target_gap_scale": str(target),
                "denominator_gap_scale": str(denominator),
                "attempted_iteration": str(attempt),
                "successful_optimizer_steps": str(attempt - skipped),
                "processed_nimg": str(attempt * 128),
                "processed_kimg": f"{attempt * 128 / 1000:.6f}",
                "stage": "0",
                "loss": "1.25",
                "loss_nonfinite_count": "0",
                "raw_grad_norm": "inf" if is_skip else "2.5",
                "raw_grad_finite_norm": "2.5",
                "raw_grad_nonfinite_count": "1" if is_skip else "0",
                "sanitized_grad_norm": "2.5",
                "sanitized_grad_nonfinite_count": "0",
                "update_norm": "0" if is_skip else "0.01",
                "update_nonfinite_count": "0",
                "model_norm": "4.0",
                "model_nonfinite_count": "0",
                "ema_norm": "4.0",
                "ema_nonfinite_count": "0",
                "sample_count": "128",
                "batch_sha256": "e" * 64,
                "t_sha256": "f" * 64,
                "base_r_sha256": "0" * 64,
                "target_r_sha256": ("a" if target == 1.0 else "b") * 64,
                "denominator_r_sha256": (
                    "a" if denominator == 1.0 else "b"
                ) * 64,
                "target_delta_sha256": (
                    "c" if target == 1.0 else "d"
                ) * 64,
                "denominator_delta_sha256": (
                    "c" if denominator == 1.0 else "d"
                ) * 64,
                "base_r_zero_count": "2",
                "target_r_zero_count": "3",
                "target_r_equal_t_count": "0",
                "target_scaled_to_zero_count": "1",
                "denominator_r_zero_count": "3",
                "denominator_r_equal_t_count": "0",
                "denominator_scaled_to_zero_count": "1",
                "target_delta_min": "0.001",
                "target_delta_max": "2.0",
                "target_delta_mean": "0.2",
                "denominator_delta_min": "0.001",
                "denominator_delta_max": "2.0",
                "denominator_delta_mean": "0.2",
                "factor_nonfinite_count": "0",
                "nonpositive_denominator_count": "0",
                "learning_rate": "0.0001",
                "grad_scale_before": str(before),
                "grad_scale_after": str(after),
                "step_skipped": "1" if is_skip else "0",
                "elapsed_sec": f"{elapsed:.6f}",
                "gpu_hours_cumulative": f"{elapsed / 3600:.9f}",
            }
            rows.append(row)
        return rows

    def state(self):
        trajectory = self.trajectory_config()
        return {
            "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
            "net": self.net,
            "ema": self.ema,
            "optimizer_state": {
                "state": {0: {"step": torch.tensor(30.0)}},
                "param_groups": [{"params": [0], "lr": 1e-4}],
            },
            "gradscaler_state": {
                "scale": 16384.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 30,
            },
            "rank_states": [{
                "rank": 0,
                "world_size": 1,
                "rng_state": rng_state(),
                "sampler_state": sampler_state(self.seed, 4096),
            }],
            "factorial": copy.deepcopy(self.factorial),
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(
                trajectory
            ),
            "attempted_iteration": 32,
            "successful_optimizer_steps": 32 - len(self.skip_attempts),
            "cur_nimg": 4096,
            "cur_tick": 2,
            "tick_start_nimg": 4096,
            "elapsed_sec": 32.0,
            "loss_fn_state": {
                "schedule_name": "sigmoid",
                "stage": 0,
                "ratio": 255 / 256,
                "schedule": {},
            },
            "snapshot_grid_z": [torch.zeros(2, 3)],
            "snapshot_grid_c": [torch.zeros(2, 0)],
            "snapshot_grid_size": (2, 1),
        }

    def snapshot(self):
        return {
            "ema": copy.deepcopy(self.ema),
            "loss_fn": TinyLoss(self.factorial),
            "augment_pipe": None,
            "dataset_kwargs": {
                "path": "/immutable/cifar10.zip",
                "use_labels": False,
                "xflip": False,
            },
        }

    def launch_manifest(self):
        auth_dir = self.root / "authorization"
        auth_dir.mkdir(exist_ok=True)
        receipt_path = auth_dir / "authorization_receipt.json"
        gate_path = auth_dir / "gate-01-correctness.json"
        receipt_path.write_text(
            json.dumps({"status": "authorized"}) + "\n", encoding="utf-8"
        )
        gate_path.write_text(
            json.dumps({"status": "PASS"}) + "\n", encoding="utf-8"
        )
        return {
            "schema": verifier.LAUNCH_SCHEMA,
            "experiment_id": verifier.EXPERIMENT_ID,
            "launch_kind": "fresh_transfer",
            "status": "authorized_to_start",
            "run_directory": str(self.root),
            "original_launch_manifest_sha256": None,
            "training": verifier.expected_launch_training_contract(
                "smoke", self.arm, self.seed
            ),
            "source": {
                "git_clean": True,
                "git_branch": "experiment/q256-target-weight-factorial",
                "git_head": "4" * 40,
                "content_sha256": "5" * 64,
            },
            "preregistration": {
                "path": "analysis/q256_target_weight_factorial/preregistration.json",
                "sha256": "6" * 64,
            },
            "assets": {
                "dataset": {
                    "resolved_path": "/immutable/cifar10.zip",
                    "sha256": verifier.DATASET_SHA256,
                    "size_bytes": 166000134,
                },
                "transfer": {
                    "resolved_path": "/immutable/edm-cifar10.pkl",
                    "sha256": verifier.TRANSFER_SHA256,
                    "size_bytes": 223173327,
                },
            },
            "runtime": {"cuda_available": True, "cuda_device_count": 1},
            "gpu": {"name": "NVIDIA A100 80GB PCIe", "memory_total_mib": 81920},
            "process_environment": {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"},
            "authorization": {
                "receipt_path": "authorization/authorization_receipt.json",
                "receipt_sha256": verifier.sha256_file(receipt_path),
                "gate_receipts": [{
                    "name": "correctness",
                    "path": "authorization/gate-01-correctness.json",
                    "sha256": verifier.sha256_file(gate_path),
                }],
            },
        }

    def write_json(self, name, value):
        (self.root / name).write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_telemetry(self, rows=None, fields=None):
        fields = verifier.TELEMETRY_FIELDS if fields is None else fields
        rows = self.telemetry_rows() if rows is None else rows
        with (self.root / "factorial_training_telemetry_v1.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_state(self, state=None):
        torch.save(self.state() if state is None else state,
                   self.root / "training-state-latest.pt")

    def write_snapshot(self, snapshot=None):
        with (self.root / "network-snapshot-latest.pkl").open("wb") as handle:
            pickle.dump(self.snapshot() if snapshot is None else snapshot, handle)

    def write_all(self):
        self.write_json("launch_manifest.json", self.launch_manifest())
        self.write_json("training_options.json", self.options())
        self.write_json("initial_state_receipt_v1.json", self.initial_receipt())
        self.write_telemetry()
        self.write_state()
        self.write_snapshot()
        (self.root / "train_summary.csv").write_text(
            "attempted_iteration,processed_nimg\n32,4096\n", encoding="utf-8"
        )
        (self.root / "log.txt").write_text("Training complete\nExiting...\n", encoding="utf-8")


class Q256TargetWeightVerifierTest(unittest.TestCase):
    def make_fixture(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        return RunFixture(root, **kwargs)

    def test_valid_smoke_emits_immutable_validation_and_hash_receipts(self):
        fixture = self.make_fixture(arm="C", skip_attempts=(2, 7))
        report = verifier.verify_run(
            fixture.root,
            arm="C",
            seed=3,
            mode="smoke",
            expected_skip_attempts=[2, 7],
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["amp_skip_attempts"], [2, 7])
        validation_path = fixture.root / verifier.VALIDATION_FILENAME
        hash_path = fixture.root / verifier.HASH_RECEIPT_FILENAME
        self.assertTrue(validation_path.is_file())
        receipt = json.loads(hash_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(
            receipt["artifacts"][verifier.VALIDATION_FILENAME]["sha256"],
            verifier.sha256_file(validation_path),
        )
        self.assertIn("launch_manifest.json", receipt["artifacts"])
        self.assertIn(
            "authorization/gate-01-correctness.json", receipt["artifacts"]
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, "immutable verifier output already exists"
        ):
            verifier.verify_run(
                fixture.root, arm="C", seed=3, mode="smoke"
            )

    def test_four_arm_mapping_is_derived_from_exact_factors(self):
        for arm in verifier.ARMS:
            with self.subTest(arm=arm):
                fixture = self.make_fixture(arm=arm, skip_attempts=())
                report = verifier.verify_run(
                    fixture.root, arm=arm, seed=3, mode="smoke",
                    write_receipts=False,
                )
                self.assertEqual(report["factorial"]["arm"], arm)

        fixture = self.make_fixture(arm="D", skip_attempts=())
        options = fixture.options()
        options["loss_kwargs"]["target_gap_scale"] = 1.1
        fixture.write_json("training_options.json", options)
        with self.assertRaisesRegex(
            verifier.VerificationError, "loss.target_gap_scale mismatch"
        ):
            verifier.verify_run(
                fixture.root, arm="D", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_launch_contract_and_bound_gate_hashes_are_fail_closed(self):
        fixture = self.make_fixture(skip_attempts=())
        manifest = fixture.launch_manifest()
        manifest["training"]["q"] = 128
        fixture.write_json("launch_manifest.json", manifest)
        with self.assertRaisesRegex(
            verifier.VerificationError, "launch manifest training contract mismatch"
        ):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

        fixture = self.make_fixture(skip_attempts=())
        (fixture.root / "authorization/gate-01-correctness.json").write_text(
            '{"status":"tampered"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(verifier.VerificationError, "hash mismatch"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_telemetry_header_and_attempt_continuity_are_fail_closed(self):
        fixture = self.make_fixture(skip_attempts=())
        fields = (*verifier.TELEMETRY_FIELDS, "unexpected")
        rows = fixture.telemetry_rows()
        for row in rows:
            row["unexpected"] = ""
        fixture.write_telemetry(rows, fields)
        with self.assertRaisesRegex(verifier.VerificationError, "header is not exact v1"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

        fixture = self.make_fixture(skip_attempts=())
        rows = fixture.telemetry_rows()
        rows[4]["attempted_iteration"] = "6"
        fixture.write_telemetry(rows)
        with self.assertRaisesRegex(verifier.VerificationError, "attempted_iteration mismatch"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

        fixture = self.make_fixture(skip_attempts=())
        rows = fixture.telemetry_rows()
        rows[0]["batch_sha256"] = "not-a-sha256"
        fixture.write_telemetry(rows)
        with self.assertRaisesRegex(verifier.VerificationError, "batch_sha256 must be"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_amp_skip_relationship_and_optional_signature_are_strict(self):
        fixture = self.make_fixture(skip_attempts=(2, 7))
        with self.assertRaisesRegex(verifier.VerificationError, "signature mismatch"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                expected_skip_attempts=[2, 8], write_receipts=False,
            )

        rows = fixture.telemetry_rows()
        rows[1]["raw_grad_nonfinite_count"] = "0"
        fixture.write_telemetry(rows)
        with self.assertRaisesRegex(verifier.VerificationError, "does not match AMP skip"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_nonfinite_telemetry_and_incomplete_state_are_rejected(self):
        fixture = self.make_fixture(skip_attempts=())
        rows = fixture.telemetry_rows()
        rows[10]["ema_norm"] = "nan"
        fixture.write_telemetry(rows)
        with self.assertRaisesRegex(verifier.VerificationError, "ema_norm must be finite"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

        fixture = self.make_fixture(skip_attempts=())
        state = fixture.state()
        del state["ema"]
        fixture.write_state(state)
        with self.assertRaisesRegex(verifier.VerificationError, "missing.*ema"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_state_snapshot_must_be_loadable_finite_and_exactly_bound(self):
        fixture = self.make_fixture(skip_attempts=())
        state = fixture.state()
        state["rank_states"][0]["sampler_state"]["consumed_samples"] = 4000
        fixture.write_state(state)
        with self.assertRaisesRegex(verifier.VerificationError, "consumed_samples mismatch"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

        fixture = self.make_fixture(skip_attempts=())
        snapshot = fixture.snapshot()
        with torch.no_grad():
            snapshot["ema"].weight[0, 0] = float("nan")
        fixture.write_snapshot(snapshot)
        with self.assertRaisesRegex(verifier.VerificationError, "non-finite tensor"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_initial_receipt_hashes_are_recomputed(self):
        fixture = self.make_fixture(skip_attempts=())
        receipt = fixture.initial_receipt()
        receipt["rank_states"][0]["sampler_sha256"] = "0" * 64
        fixture.write_json("initial_state_receipt_v1.json", receipt)
        with self.assertRaisesRegex(verifier.VerificationError, "sampler hash mismatch"):
            verifier.verify_run(
                fixture.root, arm="A", seed=3, mode="smoke",
                write_receipts=False,
            )

    def test_skip_parser_accepts_csv_and_json_without_guessing(self):
        self.assertEqual(verifier.parse_expected_skip_attempts("2,7"), [2, 7])
        self.assertEqual(verifier.parse_expected_skip_attempts("[2, 7]"), [2, 7])
        self.assertIsNone(verifier.parse_expected_skip_attempts(None))
        with self.assertRaises(Exception):
            verifier.parse_expected_skip_attempts("7,2")


if __name__ == "__main__":
    unittest.main()
