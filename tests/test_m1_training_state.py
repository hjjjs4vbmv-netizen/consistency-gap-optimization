import copy
import csv
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from training import m1, reproducibility, schedule_switch
from training import ct_training_loop
from scripts import build_m1_source_inventory as source_inventory
from scripts import build_m1_training_manifest as training_manifest
from scripts import run_m1_training_gates as training_gates
from scripts import run_m1_training_slot as training_slot
from analysis.q256_optimizer_restart_ema_rebuild_v1 import export_readout


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("running", torch.tensor([3.0]))


class M1TrainingStateTests(unittest.TestCase):
    def make_manifest(self, directory, branch="R_A"):
        source = Path(directory) / "source.pt"
        source.write_bytes(b"source")
        origin = "B" if branch.endswith("_B") else "A"
        return {
            "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
            "experiment_protocol": m1.PROTOCOL_ID,
            "run_kind": "formal",
            "branch": branch,
            "seed": 50,
            "origin_arm": origin,
            "continuation_arm": "A",
            "switch_kimg": 512,
            "final_kimg": 1024,
            "protocol_sha256": "1" * 64,
            "implementation_commit": "2" * 40,
            "source_checkpoint_manifest_sha256": "3" * 64,
            "source_state": {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "internal_state_sha256": {
                    "net": "4" * 64,
                    "ema": "5" * 64,
                    "optimizer": "6" * 64,
                    "gradscaler": "7" * 64,
                    "rank_rng": ["8" * 64],
                    "rank_sampler": ["9" * 64],
                },
            },
        }

    def test_manifest_uses_origin_only_and_always_continues_with_a(self):
        with tempfile.TemporaryDirectory() as directory:
            for branch in ("K_A", "K_B", "R_A", "R_B"):
                manifest = self.make_manifest(directory, branch)
                path = Path(directory) / f"{branch}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                loaded = schedule_switch.load_run_manifest(path)
                self.assertEqual(loaded["continuation_arm"], "A")
                self.assertEqual(
                    schedule_switch.continuation_factorial(loaded)["arm"], "A"
                )
            manifest["continuation_arm"] = "B"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "continuation arm"):
                schedule_switch.load_run_manifest(path)

    def test_absolute_gate_targets_and_resume_boundaries(self):
        values = {
            "planned_pause_protocol": m1.PROTOCOL_ID,
            "strict_reproducibility": True,
            "seed": 50,
            "total_kimg": 1024,
            "resume_state_dump": "/run/training-state-kimg000512.pt",
            "schedule_switch_manifest": "/run/formal_run_manifest.json",
            "schedule_switch_experiment_protocol": m1.PROTOCOL_ID,
        }
        for target in (4016, 4032):
            self.assertEqual(
                ct_training_loop.validate_planned_pause(
                    stop_after_attempts=target, **values
                ),
                target,
            )
        for target in (4000, 4015, 4033):
            with self.assertRaisesRegex(ValueError, "absolute 4016 or 4032"):
                ct_training_loop.validate_planned_pause(
                    stop_after_attempts=target, **values
                )
        m1.validate_gate_resume_boundary(4016, 4000)
        m1.validate_gate_resume_boundary(4032, 4000)
        m1.validate_gate_resume_boundary(4032, 4016)
        with self.assertRaisesRegex(RuntimeError, "requires restored attempt"):
            m1.validate_gate_resume_boundary(4016, 4016)
        values["schedule_switch_experiment_protocol"] = (
            schedule_switch.TERMINAL_HISTORY_N30_PROTOCOL
        )
        with self.assertRaisesRegex(ValueError, "requires an M1 full-state"):
            ct_training_loop.validate_planned_pause(
                stop_after_attempts=4032, **values
            )

    def test_m1_branch_init_can_resume_header_only_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_summary.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=ct_training_loop._TRAIN_SUMMARY_FIELDS
                )
                writer.writeheader()
            rows, backup = ct_training_loop.load_and_migrate_train_summary(
                path, allow_empty_current=True
            )
            self.assertEqual(rows, [])
            self.assertIsNone(backup)
            with self.assertRaisesRegex(RuntimeError, "no data rows"):
                ct_training_loop.load_and_migrate_train_summary(path)

    def test_optimizer_keep_and_reset_have_the_frozen_meaning(self):
        model = TinyModule()
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())

        self.assertEqual(m1.apply_optimizer_intervention(optimizer, "K_A"), 0)
        self.assertEqual(
            reproducibility.state_sha256(optimizer.state_dict()),
            reproducibility.state_sha256(before),
        )
        groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
        self.assertEqual(m1.apply_optimizer_intervention(optimizer, "R_A"), 1)
        self.assertFalse(optimizer.state)
        self.assertEqual(optimizer.state_dict()["param_groups"], groups)

    def test_e512_is_an_independent_full_copy_and_parameter_only_shadow(self):
        online = TinyModule()
        rng_before = torch.get_rng_state()
        ema_512 = m1.initialize_ema_512(online)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIsNot(ema_512, online)
        self.assertTrue(torch.equal(ema_512.running, online.running))
        with torch.no_grad():
            online.weight.add_(2)
            online.running.add_(5)
        m1.update_ema_512(ema_512, online, 0.5)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(ema_512.weight, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(ema_512.running, torch.tensor([3.0])))
        self.assertTrue(torch.equal(online.weight, torch.tensor([3.0, 4.0])))

    def test_branch_init_is_complete_fixed_boundary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory)
            metadata = m1.initial_metadata(
                manifest, reset_count=1, successful_steps_at_init=3999
            )
            state = {
                "net": TinyModule(),
                "ema": TinyModule(),
                "ema_512": TinyModule(),
                "optimizer_state": {"state": {}, "param_groups": []},
                "gradscaler_state": {"scale": 1.0},
                "rank_states": [{"rng_state": {}, "sampler_state": {}}],
                "loss_fn_state": {},
                "cur_nimg": 512000,
                "attempted_iteration": 4000,
                "successful_optimizer_steps": 3999,
                "cur_tick": 128,
                "tick_start_nimg": 512000,
                "trajectory_config": {},
                "trajectory_config_sha256": reproducibility.state_sha256({}),
                "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
                "factorial": {},
                "schedule_switch": {},
                "snapshot_grid_z": [],
                "snapshot_grid_c": [],
                "snapshot_grid_size": (0, 0),
                "m1": metadata,
            }
            path = m1.save_branch_init_state(state, directory)
            self.assertEqual(Path(path).name, "training-state-kimg000512.pt")
            restored = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("ema_512", restored)
            self.assertEqual(restored["m1"]["reset_count"], 1)
            with self.assertRaises(FileExistsError):
                m1.save_branch_init_state(state, directory)

    def test_branch_init_directly_proves_keep_and_reset_from_source(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = {
                "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
                "seed": 50, "world_size": 1,
                "batch_size": 128, "batch_gpu": 16,
            }
            source = {
                "net": TinyModule(), "ema": TinyModule(),
                "optimizer_state": {
                    "state": {0: {"step": 3}},
                    "param_groups": [{"lr": 1e-4, "params": [0]}],
                },
                "gradscaler_state": {"scale": 65536.0},
                "rank_states": [{
                    "rng_state": {"torch_cpu": torch.get_rng_state()},
                    "sampler_state": {"consumed_samples": 512000},
                }],
                "loss_fn_state": {"stage": 0},
                "attempted_iteration": 4000,
                "successful_optimizer_steps": 3999,
                "cur_nimg": 512000, "cur_tick": 1,
                "tick_start_nimg": 510000,
                "snapshot_grid_z": [], "snapshot_grid_c": [],
                "snapshot_grid_size": (0, 0),
                "factorial": {
                    "protocol": "q256_target_weight_v1", "arm": "A",
                    "target_gap_scale": 1.0,
                    "denominator_gap_scale": 1.0,
                },
                "trajectory_config": trajectory,
                "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
                "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
            }
            for branch, reset in (("K_A", False), ("R_A", True)):
                manifest = self.make_manifest(directory, branch)
                manifest["source_state"]["internal_state_sha256"] = (
                    schedule_switch.internal_state_hashes(source)
                )
                branch_state = copy.deepcopy(source)
                if reset:
                    branch_state["optimizer_state"]["state"] = {}
                branch_state["ema_512"] = copy.deepcopy(source["net"])
                branch_state["m1"] = m1.initial_metadata(
                    manifest, int(reset), source["successful_optimizer_steps"]
                )
                self.assertEqual(
                    len(m1.validate_branch_init_against_source(
                        branch_state, source, manifest
                    )), 64
                )

    def test_resume_validation_and_explicit_readouts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory, "K_B")
            online = TinyModule()
            keep = copy.deepcopy(online)
            rebuilt = copy.deepcopy(online)
            with torch.no_grad():
                online.weight.fill_(1)
                keep.weight.fill_(2)
                rebuilt.weight.fill_(3)
            state = {
                "net": online,
                "ema": keep,
                "ema_512": rebuilt,
                "m1": m1.initial_metadata(manifest, reset_count=0),
                "trajectory_config": {"dataset_kwargs": {"path": "/data.zip"}},
            }
            m1.validate_resumed_state(state, manifest)
            for readout, value in (("ONLINE", 1), ("E_KEEP", 2), ("E_512", 3)):
                snapshot = m1.evaluator_snapshot(state, readout)
                self.assertTrue(torch.equal(
                    snapshot["ema"].weight, torch.full((2,), float(value))
                ))
                self.assertIsNot(snapshot["ema"], m1.readout_module(state, readout))
                self.assertFalse(snapshot["ema"].training)
                self.assertFalse(snapshot["ema"].weight.requires_grad)
            broken = copy.deepcopy(state)
            del broken["ema_512"]
            with self.assertRaisesRegex(RuntimeError, "E_512"):
                m1.validate_resumed_state(broken, manifest)

    def test_terminal_validation_reconciles_progress_and_m1_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory, "K_A")
            trajectory = {"seed": 50, "total_kimg": 1024}
            metadata = m1.initial_metadata(
                manifest, reset_count=0, successful_steps_at_init=3990
            )
            metadata = m1.checkpoint_metadata(metadata, 3998)
            state = {
                "net": TinyModule(), "ema": TinyModule(), "ema_512": TinyModule(),
                "optimizer_state": {}, "gradscaler_state": {},
                "rank_states": [{"sampler_state": {"consumed_samples": 1024000}}],
                "loss_fn_state": {}, "attempted_iteration": 8000,
                "successful_optimizer_steps": 7988,
                "cur_nimg": 1024000, "cur_tick": 1, "tick_start_nimg": 1024000,
                "trajectory_config": trajectory,
                "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
                "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
                "factorial": {}, "schedule_switch": {}, "m1": metadata,
            }
            self.assertEqual(
                m1.validate_terminal_state(state, manifest)[
                    "successful_steps_since_init"
                ],
                3998,
            )
            state["successful_optimizer_steps"] += 1
            with self.assertRaisesRegex(RuntimeError, "do not reconcile"):
                m1.validate_terminal_state(state, manifest)

    def test_inventory_builds_512_identity_from_state_and_binds_initial_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = {
                "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
                "seed": 50, "world_size": 1,
                "batch_size": 128, "batch_gpu": 16,
            }
            state = {
                "net": TinyModule(),
                "ema": TinyModule(),
                "optimizer_state": {},
                "gradscaler_state": {},
                "rank_states": [{"rng_state": {}, "sampler_state": {}}],
                "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
            }
            state_path = root / "training-state-kimg000512.pt"
            torch.save(state, state_path)
            sampler = {
                "schema": "ect.infinite-sampler/v1",
                "dataset_size": 50000, "rank": 0, "num_replicas": 1,
                "shuffle": True, "seed": 50, "window_size": 0.5,
                "consumed_samples": 0,
            }
            rng_sha = "5" * 64
            sampler_sha = reproducibility.state_sha256(sampler)
            hashes = {
                "model": "1" * 64, "ema": "1" * 64,
                "optimizer": "2" * 64, "gradscaler": "3" * 64,
                "rank_rng": [rng_sha], "rank_sampler": [sampler_sha],
            }
            receipt = {
                "schema": reproducibility.INITIAL_RECEIPT_SCHEMA,
                "seed": 50,
                "attempted_iteration": 0,
                "processed_nimg": 0,
                "factorial": {
                    "protocol": "q256_target_weight_v1", "arm": "A",
                    "target_gap_scale": 1.0, "denominator_gap_scale": 1.0,
                },
                "world_size": 1, "batch_size": 128, "batch_gpu": 16,
                "dataset_path": "/data/cifar.zip",
                "transfer_path": "/data/transfer.pkl",
                "trajectory_config": trajectory,
                "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
                "hashes": hashes,
                "common_initial_state_sha256": reproducibility.state_sha256(hashes),
                "rank_states": [{
                    "rank": 0, "world_size": 1, "rng_sha256": rng_sha,
                    "sampler_sha256": sampler_sha, "sampler_state": sampler,
                }],
            }
            receipt_path = root / "initial_state_receipt_v1.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            record, loaded, common_initial = source_inventory._load_source(
                state_path, receipt_path, 50, "A"
            )
            self.assertEqual(record["source_state_bytes"], state_path.stat().st_size)
            self.assertEqual(
                record["source_state_sha256"],
                schedule_switch.sha256_file(str(state_path)),
            )
            self.assertEqual(record["provenance_receipt_path"], str(receipt_path))
            self.assertEqual(loaded["trajectory_config_sha256"], receipt[
                "trajectory_config_sha256"
            ])
            self.assertEqual(common_initial, receipt["common_initial_state_sha256"])

            receipt["rank_states"][0]["sampler_state"].pop("schema")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "rank/sampler"):
                source_inventory._load_source(state_path, receipt_path, 50, "A")

    def test_source_state_fallback_is_uniform_and_primary_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            latest = prefix / "training-state-latest.pt"
            latest.write_bytes(b"latest")
            self.assertEqual(source_inventory.select_source_state(prefix), latest)
            primary = prefix / "training-state-kimg000512.pt"
            primary.write_bytes(b"primary")
            self.assertEqual(source_inventory.select_source_state(prefix), primary)

    def test_parameterized_manifest_and_commands_are_frozen_without_64_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "cifar.zip"
            dataset.write_bytes(b"dataset")
            protocol = root / "M1_PROTOCOL.md"
            protocol.write_text("protocol", encoding="utf-8")
            inventory_path = root / "inventory.json"
            inventory_path.write_text("{}", encoding="utf-8")
            source_file = root / "source.pt"
            source_file.write_bytes(b"source")
            receipt_file = root / "initial_state_receipt_v1.json"
            receipt_file.write_text("{}", encoding="utf-8")
            runtime_receipt = root / "runtime.json"
            pip_freeze = root / "pip-freeze.txt"
            pip_freeze.write_text("torch==2.6.0\n", encoding="utf-8")
            runtime_receipt.write_text(json.dumps({
                "schema": "ect.m1.rebuilt-training-runtime/v1",
                "status": "PASS",
                "runtime_origin": "REBUILT_NOT_BYTE_IDENTICAL",
                "runtime_probe": {
                    "python": "3.11.13", "torch": "2.6.0+cu124",
                    "cuda": "12.4", "cudnn": 90100,
                    "numpy": "2.1.2", "scipy": "1.16.1",
                },
                "pip_freeze": {
                    "path": str(pip_freeze),
                    "sha256": schedule_switch.sha256_file(str(pip_freeze)),
                },
                "reference_archive_sha256": None,
            }), encoding="utf-8")
            internal = {
                "net": "1" * 64, "ema": "2" * 64,
                "optimizer": "3" * 64, "gradscaler": "4" * 64,
                "rank_rng": ["5" * 64], "rank_sampler": ["6" * 64],
            }
            candidates = []
            for seed in range(50, 66):
                sources = {}
                for arm in ("A", "B"):
                    sources[arm] = {
                        "source_state_path": str(source_file),
                        "source_state_bytes": source_file.stat().st_size,
                        "source_state_sha256": schedule_switch.sha256_file(str(source_file)),
                        "provenance_receipt_path": str(receipt_file),
                        "provenance_receipt_sha256": schedule_switch.sha256_file(str(receipt_file)),
                        "internal_state_sha256": internal,
                        "common_initial_state_sha256": "7" * 64,
                        "support_files": {
                            name: {
                                "path": str(receipt_file),
                                "bytes": receipt_file.stat().st_size,
                                "sha256": schedule_switch.sha256_file(str(receipt_file)),
                            }
                            for name in (
                                "train_summary.csv",
                                "factorial_training_telemetry_v1.csv",
                                "training_options.json",
                            )
                        },
                    }
                candidates.append({
                    "seed": seed, "checked": True, "qualified": True,
                    "reason": "QUALIFIED", "sources": sources,
                })
            candidates.extend({
                "seed": seed, "checked": False, "qualified": False,
                "reason": "NOT_CHECKED_AFTER_ROSTER_FILLED",
            } for seed in range(66, 80))
            inventory = {
                "schema": "ect.m1.source-inventory/v1", "status": "PASS",
                "candidates": candidates,
            }
            dataset_sha = schedule_switch.sha256_file(str(dataset))
            with mock.patch.object(
                training_manifest, "TRAINING_DATASET_SHA256", dataset_sha
            ):
                built = training_manifest.build_manifest(
                    inventory, inventory_path, protocol,
                    implementation_commit="a" * 40,
                    dataset_path=dataset,
                    dataset_sha256=dataset_sha,
                    runtime_python=Path("/usr/bin/python3"),
                    runtime_receipt=runtime_receipt,
                    output_root=root / "runs",
                )
            built["_training_manifest_sha256"] = "f" * 64
            self.assertEqual(len(built["roster"]), 16)
            self.assertNotIn("runs", built)
            self.assertEqual(built["roster"][0]["order"], list(training_manifest.ORDERS[0]))
            manifest_dir = root / "gate"
            manifest_dir.mkdir()
            branch_manifest = training_gates.write_branch_manifest(
                built, built["roster"][0], "K_B", manifest_dir, shadow=True
            )
            loaded = json.loads(branch_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["source_checkpoint_manifest_sha256"],
                schedule_switch.sha256_file(str(inventory_path)),
            )
            formal_command = training_gates.command(
                built, built["roster"][0], branch_manifest, source_file
            )
            self.assertTrue(any(value.startswith("--immutable-checkpoint-kimg=") for value in formal_command))
            self.assertFalse(any(value.startswith("--stop-after-attempts=") for value in formal_command))

            tampered = copy.deepcopy(inventory)
            tampered["candidates"][5]["checked"] = False
            with self.assertRaisesRegex(RuntimeError, "cannot qualify without being checked"):
                training_manifest.selected_roster(tampered)

    def test_training_environment_uses_only_rebuilt_prefix_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            python = prefix / "bin/python3.11"
            python.parent.mkdir()
            python.write_bytes(b"python")
            (prefix / "lib/python3.11/site-packages/torch/lib").mkdir(parents=True)
            (prefix / "lib/python3.11/site-packages/nvidia/cublas/lib").mkdir(
                parents=True
            )
            with mock.patch.dict("os.environ", {
                "LD_LIBRARY_PATH": "/foreign/conda/lib",
                "PYTHONPATH": "/foreign/python",
                "PYTHONHOME": "/foreign/home",
            }):
                env = training_gates.environment(0, runtime_python=python)
            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(":"),
                [
                    str(prefix / "lib"),
                    str(prefix / "lib/python3.11/site-packages/torch/lib"),
                    str(prefix / "lib/python3.11/site-packages/nvidia/cublas/lib"),
                ],
            )
            self.assertNotIn("foreign", env["LD_LIBRARY_PATH"])
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertEqual(
                training_gates.canonical_pip_freeze(b"torch==2.6\nnumpy==2.1\n"),
                b"numpy==2.1\ntorch==2.6\n",
            )

    def test_recorded_gate_runtime_binds_canonical_freeze_and_library_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "runtime/bin/python3.11"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            (root / "runtime/lib/python3.11/site-packages/torch/lib").mkdir(
                parents=True
            )
            freeze = root / "pip-freeze.txt"
            freeze.write_bytes(b"numpy==2.1.2\nscipy==1.16.1\ntorch==2.6.0\n")
            receipt = root / "runtime.json"
            receipt.write_text(json.dumps({
                "pip_freeze": {
                    "path": str(freeze),
                    "sha256": schedule_switch.sha256_file(str(freeze)),
                }
            }), encoding="utf-8")
            contract = {
                "python": "3.11.13", "torch": "2.6.0+cu124",
                "cuda": "12.4", "numpy": "2.1.2", "scipy": "1.16.1",
                "cudnn": 90100,
            }
            training = {
                "runtime_python": str(python),
                "runtime_receipt": {"path": str(receipt)},
                "runtime_contract": contract,
            }
            hardware = {
                "runtime": contract,
                "runtime_python": str(python),
                "runtime_prefix": str(python.parent.parent),
                "runtime_library_paths": [
                    str(root / "runtime/lib"),
                    str(root / "runtime/lib/python3.11/site-packages/torch/lib"),
                ],
                "runtime_receipt_sha256": schedule_switch.sha256_file(str(receipt)),
                "canonical_pip_freeze_path": str(freeze),
                "canonical_pip_freeze_sha256": schedule_switch.sha256_file(str(freeze)),
            }
            training_gates.validate_recorded_runtime_probe(training, hardware)
            hardware["canonical_pip_freeze_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "recorded gate runtime"):
                training_gates.validate_recorded_runtime_probe(training, hardware)

    def test_exporter_requires_clean_matching_implementation_checkout(self):
        manifest = {"implementation_commit": "a" * 40}
        with mock.patch.object(
            export_readout.subprocess,
            "check_output",
            side_effect=["a" * 40 + "\n", ""],
        ):
            self.assertEqual(
                export_readout.verify_implementation_checkout(manifest),
                {"head": "a" * 40, "clean": True},
            )
        with mock.patch.object(
            export_readout.subprocess,
            "check_output",
            side_effect=["a" * 40 + "\n", "?? untracked\n"],
        ):
            with self.assertRaisesRegex(RuntimeError, "clean frozen"):
                export_readout.verify_implementation_checkout(manifest)

    def test_gate_artifact_validator_rehashes_bound_files(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.write_bytes(b"changed")
            labels = set()
            for branch in ("K_A", "K_B", "R_A", "R_B"):
                labels.update({
                    f"{branch}_continuous_state", f"{branch}_continuous_manifest",
                    f"{branch}_continuous_telemetry", f"{branch}_continuous_log",
                    f"{branch}_staged_state", f"{branch}_staged_manifest",
                    f"{branch}_staged_log_4016", f"{branch}_staged_log_4032",
                })
            for branch in ("K_A", "K_B"):
                labels.update({
                    f"{branch}_shadow_state", f"{branch}_shadow_manifest",
                    f"{branch}_shadow_log", f"{branch}_legacy_state",
                    f"{branch}_legacy_manifest", f"{branch}_legacy_log",
                })
            check = {
                "artifacts": {
                    label: {"path": str(artifact), "sha256": "0" * 64}
                    for label in labels
                }
            }
            with self.assertRaisesRegex(RuntimeError, "artifact SHA256"):
                training_gates.validate_gate_seed_artifacts(check, {}, {})

    def test_normalized_state_ignores_provenance_but_detects_numeric_changes(self):
        state = {
            "net": TinyModule(), "ema": TinyModule(), "ema_512": TinyModule(),
            "optimizer_state": {"state": {}, "param_groups": [{"lr": 1e-4}]},
            "gradscaler_state": {"scale": 65536.0},
            "rank_states": [{"rng_state": {"cpu": torch.tensor([1])}}],
            "loss_fn_state": {"window": torch.tensor([2.0])},
            "attempted_iteration": 4032,
            "successful_optimizer_steps": 4032,
            "cur_nimg": 516096, "cur_tick": 51, "tick_start_nimg": 510000,
            "m1": {"reset_count": 0, "successful_steps_since_init": 32},
            "factorial": {"arm": "A"},
            "trajectory_config": {"dataset_kwargs": {"path": "/first/data.zip"}},
            "trajectory_config_sha256": "1" * 64,
            "reproducibility_schema": "first-schema",
            "schedule_switch": {
                "implementation_commit": "1" * 40,
                "source_state_path": "/first/source.pt",
            },
        }
        metadata_only = copy.deepcopy(state)
        metadata_only["trajectory_config"] = {
            "dataset_kwargs": {"path": "/second/data.zip"}
        }
        metadata_only["trajectory_config_sha256"] = "2" * 64
        metadata_only["reproducibility_schema"] = "second-schema"
        metadata_only["schedule_switch"] = {
            "implementation_commit": "2" * 40,
            "source_state_path": "/second/source.pt",
        }
        with mock.patch.object(
            training_gates.torch, "load", side_effect=[state, metadata_only]
        ):
            left = training_gates.normalized_online_state(Path("left"), include_m1=True)
            right = training_gates.normalized_online_state(Path("right"), include_m1=True)
        self.assertEqual(left, right)

        tensor_change = copy.deepcopy(state)
        tensor_change["net"].weight.data[0] += 1
        with mock.patch.object(
            training_gates.torch, "load", side_effect=[state, tensor_change]
        ):
            self.assertNotEqual(
                training_gates.normalized_online_state(Path("left")),
                training_gates.normalized_online_state(Path("right")),
            )
        optimizer_change = copy.deepcopy(state)
        optimizer_change["optimizer_state"]["param_groups"][0]["lr"] = 2e-4
        with mock.patch.object(
            training_gates.torch, "load", side_effect=[state, optimizer_change]
        ):
            self.assertNotEqual(
                training_gates.normalized_online_state(Path("left")),
                training_gates.normalized_online_state(Path("right")),
            )

    def test_training_attempt_receipts_seal_exact_scientific_failure(self):
        self.assertTrue(training_slot.scientific_floating_point([
            "[rank0]: FloatingPointError: strict factorial training invariant failure: loss"
        ]))
        self.assertTrue(training_slot.scientific_floating_point([
            "FloatingPointError: non-finite RAdam moment state"
        ]))
        self.assertTrue(training_slot.scientific_floating_point([
            "FloatingPointError: denominator realized gaps must be strictly positive"
        ]))
        self.assertFalse(training_slot.scientific_floating_point([
            "RuntimeError: worker exited"
        ]))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume = run_dir / "training-state-kimg000512.pt"
            resume.write_bytes(b"state")
            (run_dir / "formal_run_manifest.json").write_text(json.dumps({
                "training_manifest_sha256": "1" * 64,
                "source_state": {"sha256": "2" * 64},
            }), encoding="utf-8")
            log = run_dir / "formal-attempt-1.log"
            with log.open("xb") as handle:
                training_slot.write_attempt_launch_header(
                    handle, "K_A", 50, 1, resume,
                    ["python", "ct_train.py", f"--resume={resume}"],
                )
                handle.write(b"technical failure\n")
            training_slot.write_attempt_receipt(
                run_dir, "K_A", 50, 1, log, resume,
                "INCOMPLETE_TECHNICAL", 1, "NONZERO_EXIT",
            )
            receipts = training_slot.load_attempt_receipts(run_dir, "K_A", 50)
            self.assertEqual(receipts[0]["status"], "INCOMPLETE_TECHNICAL")
            self.assertEqual(receipts[0]["resume_path"], str(resume))
            self.assertEqual(
                receipts[0]["resume_sha256"],
                schedule_switch.sha256_file(str(resume)),
            )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            resume = run_dir / "source.pt"
            resume.write_bytes(b"state")
            stale = run_dir / "stale-source.pt"
            stale.write_bytes(b"stale")
            (run_dir / "training_options.json").write_text(json.dumps({
                "resume_state_dump": str(stale),
            }), encoding="utf-8")
            (run_dir / "formal_run_manifest.json").write_text(json.dumps({
                "training_manifest_sha256": "1" * 64,
                "source_state": {"sha256": "2" * 64},
            }), encoding="utf-8")
            log = run_dir / "formal-attempt-1.log"
            with log.open("xb") as handle:
                training_slot.write_attempt_launch_header(
                    handle, "R_A", 50, 1, resume,
                    ["python", "ct_train.py", f"--resume={resume}"],
                )
                handle.write(
                    b"FloatingPointError: non-finite online-EMA distance\n"
                )
            training_slot.recover_interrupted_attempt(
                run_dir, "R_A", 50, authorized=True
            )
            receipt = training_slot.load_attempt_receipts(run_dir, "R_A", 50)[0]
            self.assertEqual(receipt["status"], "SCIENTIFIC_FAILURE")
            self.assertEqual(receipt["reason"], "NUMERIC_FLOATING_POINT")
            self.assertEqual(receipt["resume_path"], str(resume))
            self.assertNotEqual(receipt["resume_path"], str(stale))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "formal-attempt-1.log").write_text(
                "launcher interrupted\n", encoding="utf-8"
            )
            (run_dir / "formal_run_manifest.json").write_text(json.dumps({
                "training_manifest_sha256": "1" * 64,
                "source_state": {"sha256": "2" * 64},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "launch provenance"):
                training_slot.recover_interrupted_attempt(
                    run_dir, "R_A", 50, authorized=True
                )

    def test_resume_selection_falls_back_from_corrupt_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "training-state-latest.pt"
            milestone = root / "training-state-kimg000896.pt"
            latest.write_bytes(b"broken")
            milestone.write_bytes(b"valid")
            state = {"attempted_iteration": 7000, "cur_nimg": 896000}
            with (
                mock.patch.object(
                    training_slot.torch, "load",
                    side_effect=[RuntimeError("corrupt archive"), state],
                ),
                mock.patch.object(
                    training_slot.schedule_switch, "verify_switched_state"
                ),
                mock.patch.object(training_slot.m1, "validate_resumed_state"),
            ):
                selected, loaded, rejected = training_slot.select_resume_state(
                    ((latest, None), (milestone, 7000)), {}
                )
            self.assertEqual(selected, milestone)
            self.assertIs(loaded, state)
            self.assertIn("corrupt archive", rejected[0]["reason"])

    def test_resume_selection_prefers_newer_immutable_over_valid_stale_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "training-state-latest.pt"
            milestone = root / "training-state-kimg000896.pt"
            latest.write_bytes(b"old")
            milestone.write_bytes(b"new")
            old = {"attempted_iteration": 6000, "cur_nimg": 768000}
            new = {"attempted_iteration": 7000, "cur_nimg": 896000}
            with (
                mock.patch.object(training_slot.torch, "load", side_effect=[old, new]),
                mock.patch.object(training_slot.schedule_switch, "verify_switched_state"),
                mock.patch.object(training_slot.m1, "validate_resumed_state"),
            ):
                selected, loaded, rejected = training_slot.select_resume_state(
                    ((latest, None), (milestone, 7000)), {}
                )
            self.assertEqual(selected, milestone)
            self.assertIs(loaded, new)
            self.assertEqual(rejected, [])

    def test_formal_admission_accepts_only_bound_g4_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.csv"
            evaluation.write_text("slot_id\n", encoding="utf-8")
            gates = root / "gates.json"
            gates.write_text("{}", encoding="utf-8")
            training = {
                "_training_manifest_sha256": "1" * 64,
                "implementation_commit": "2" * 40,
                "_training_manifest_path": str(root / "training.json"),
            }
            (root / "training.json").write_text("{}", encoding="utf-8")
            seal = {
                "schema": "ect.m1.g4-canary-seal/v1", "status": "PASS",
                "protocol_id": m1.PROTOCOL_ID,
                "quality_eligible": False, "quality_generation": False,
                "quality_metrics_executed": False,
                "training_manifest_sha256": "1" * 64,
                "implementation_commit": "2" * 40,
                "evaluation_manifest_sha256": schedule_switch.sha256_file(
                    str(evaluation)
                ),
                "training_gates_receipt_sha256": schedule_switch.sha256_file(
                    str(gates)
                ),
                "evaluator_commit": "d6aba02fb88e9db0993623895eb2228ed717d810",
                "canary_count": 5,
            }
            canary_paths = []
            for index in range(5):
                canary = root / f"c{index}-g4-canary.json"
                canary.write_text("{}", encoding="utf-8")
                canary_paths.append({"path": str(canary)})
            seal["canary_receipts"] = canary_paths
            path = root / "g4.json"
            path.write_text(json.dumps(seal), encoding="utf-8")
            with (
                mock.patch.object(
                    training_slot.evaluation_slots, "load_training_identity",
                    return_value={},
                ),
                mock.patch.object(
                    training_slot.g4_sealer, "seal", return_value=seal
                ),
            ):
                self.assertEqual(
                    training_slot.validate_g4_receipt(
                        path, training, evaluation, gates
                    )["status"],
                    "PASS",
                )
            seal["quality_metrics_executed"] = True
            path.write_text(json.dumps(seal), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "G4 receipt"):
                training_slot.validate_g4_receipt(
                    path, training, evaluation, gates
                )


if __name__ == "__main__":
    unittest.main()
