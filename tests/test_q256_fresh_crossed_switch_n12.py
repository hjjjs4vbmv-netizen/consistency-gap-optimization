import copy
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import experiment
from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import evaluation
from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import monitor
from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import parity
from analysis.q256_fresh_crossed_switch_n12_matpool_v1 import statistics as frozen_statistics
from training import reproducibility, schedule_switch
from training import ct_training_loop


class AssignmentTests(unittest.TestCase):
    def test_six_gpu_two_wave_assignment(self):
        rows = [experiment.assignment(seed) for seed in experiment.SEEDS]
        self.assertEqual([row["gpu_index"] for row in rows[:6]], list(range(6)))
        self.assertEqual([row["gpu_index"] for row in rows[6:]], list(range(6)))
        self.assertEqual([row["wave"] for row in rows], [1] * 6 + [2] * 6)
        for gpu in range(6):
            self.assertEqual([row["seed"] for row in rows if row["gpu_index"] == gpu],
                             [31 + gpu, 37 + gpu])

    def test_prefix_order_reverses_by_wave(self):
        expected = {
            31: ["A", "B"], 32: ["B", "A"], 33: ["A", "B"],
            34: ["B", "A"], 35: ["A", "B"], 36: ["B", "A"],
            37: ["B", "A"], 38: ["A", "B"], 39: ["B", "A"],
            40: ["A", "B"], 41: ["B", "A"], 42: ["A", "B"],
        }
        self.assertEqual(
            {seed: experiment.assignment(seed)["prefix_order"] for seed in experiment.SEEDS},
            expected,
        )

    def test_suffix_latin_square_cycles_three_times(self):
        observed = [tuple(experiment.assignment(seed)["suffix_order"])
                    for seed in experiment.SEEDS]
        self.assertEqual(observed, list(experiment.SUFFIX_ORDERS) * 3)
        for position in range(4):
            self.assertEqual(
                sorted(order[position] for order in observed),
                ["AA"] * 3 + ["AB"] * 3 + ["BA"] * 3 + ["BB"] * 3,
            )


class ProtocolTests(unittest.TestCase):
    def minimal_protocol(self):
        return {
            "schema": experiment.PROTOCOL_SCHEMA,
            "experiment_id": experiment.EXPERIMENT_ID,
            "seeds": list(experiment.SEEDS),
            "evaluation": {"expected_jobs": 264,
                           "task_definitions": experiment.planned_evaluation_jobs()},
            "training": {"world_size": 1},
            "gpu_assignment": [experiment.assignment(seed) for seed in experiment.SEEDS],
            "storage_plan": {"full_state_count": 228, "ema_snapshot_count": 228,
                             "headroom_fraction": 0.30, "estimated_bytes": 100,
                             "minimum_free_bytes": 130,
                             "kid_fid_generated_features_are_hardlinked_after_byte_identity_validation": True},
        }

    def test_protocol_matrix_invariants(self):
        experiment.validate_protocol(self.minimal_protocol())
        jobs = 24 + 12 * 4 * 4 + 12 * 4
        self.assertEqual(jobs, 264)
        planned = experiment.planned_evaluation_jobs()
        self.assertEqual(len({(job["seed"], job["kind"], job["cell"],
                              job["budget_kimg"], job["nfe"]) for job in planned}), 264)
        self.assertEqual(sum(job["kind"] == "prefix" for job in planned), 24)
        self.assertEqual(sum(job["kind"] == "suffix" and job["nfe"] == 1 for job in planned), 192)
        self.assertEqual(sum(job["kind"] == "suffix" and job["nfe"] == 2 for job in planned), 48)

    def test_protocol_rejects_seed_or_assignment_change(self):
        for mutation in ("seeds", "assignment"):
            protocol = self.minimal_protocol()
            if mutation == "seeds":
                protocol["seeds"][-1] = 43
            else:
                protocol["gpu_assignment"][0]["gpu_index"] = 5
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                experiment.validate_protocol(protocol)


class PlannedPauseAuthorizationTests(unittest.TestCase):
    def call(self, **changes):
        values = {"stop_after_attempts": 4000,
                  "planned_pause_protocol": schedule_switch.FRESH_N12_PROTOCOL,
                  "strict_reproducibility": True, "seed": 31, "total_kimg": 1024,
                  "resume_state_dump": None, "schedule_switch_manifest": None}
        values.update(changes)
        return ct_training_loop.validate_planned_pause(**values)

    def test_fresh_formal_and_engineering_pauses_are_exactly_authorized(self):
        self.assertEqual(self.call(), 4000)
        self.assertEqual(self.call(
            planned_pause_protocol=schedule_switch.FRESH_N12_ENGINEERING_PROTOCOL,
            seed=20260831), 4000)
        self.assertEqual(self.call(stop_after_attempts=16,
                                   planned_pause_protocol=None), 16)

    def test_long_pause_rejects_any_contract_drift(self):
        mutations = (
            {"stop_after_attempts": 3999}, {"planned_pause_protocol": None},
            {"seed": 30}, {"total_kimg": 640}, {"resume_state_dump": "/state.pt"},
            {"schedule_switch_manifest": "/manifest.json"},
            {"strict_reproducibility": False},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.call(**mutation)

    def test_protocol_without_pause_is_rejected(self):
        with self.assertRaises(ValueError):
            self.call(stop_after_attempts=None)


class MonitorPidNamespaceTests(unittest.TestCase):
    def test_one_unmapped_host_pid_is_owned_only_with_cuda_evidence(self):
        row = {"pid": 900001, "gpu_uuid": "GPU-x", "process_name": "[Not Found]"}
        owned, foreign = monitor.classify_gpu_apps(
            [row], "GPU-x", {123}, alive=True, owned_cuda_context=True)
        self.assertEqual(owned, [row]); self.assertEqual(foreign, [])
        owned, foreign = monitor.classify_gpu_apps(
            [row], "GPU-x", {123}, alive=True, owned_cuda_context=False)
        self.assertEqual(owned, []); self.assertEqual(foreign, [row])

    def test_second_unmapped_process_remains_foreign(self):
        rows = [{"pid": pid, "gpu_uuid": "GPU-x", "process_name": "[Not Found]"}
                for pid in (900001, 900002)]
        owned, foreign = monitor.classify_gpu_apps(
            rows, "GPU-x", {123}, alive=True, owned_cuda_context=True)
        self.assertEqual(owned, []); self.assertEqual(foreign, rows)


class EngineeringParityLauncherTests(unittest.TestCase):
    def test_four_gpu_indices_can_be_shifted_without_changing_science(self):
        args = parity.parser().parse_args([
            "launch", "--runtime-manifest", "/runtime.json",
            "--dataset", "/data.zip", "--transfer", "/transfer.pkl",
            "--implementation-commit", "1" * 40, "--output-root", "/out",
            "--gpu-indices", "2", "3", "4", "5",
        ])
        self.assertEqual(args.gpu_indices, [2, 3, 4, 5])


class FreshManifestTests(unittest.TestCase):
    def make_state(self, arm):
        net = torch.nn.Linear(2, 2)
        ema = copy.deepcopy(net)
        trajectory = {
            "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
            "seed": 31,
            "total_kimg": 1024,
            "dataset_kwargs": {"path": "/assets/cifar.zip"},
            "loss_kwargs": {"arm": arm, "target_gap_scale": experiment.ARMS[arm][0],
                            "denominator_gap_scale": experiment.ARMS[arm][1]},
        }
        return {
            "net": net, "ema": ema, "optimizer_state": {"state": {}, "param_groups": []},
            "gradscaler_state": {"scale": 1.0}, "attempted_iteration": 4000,
            "successful_optimizer_steps": 3999, "cur_nimg": 512000,
            "rank_states": [{"rng_state": {"schema": "test"},
                             "sampler_state": {"consumed_samples": 512000}}],
            "factorial": {"protocol": "q256_target_weight_v1", "arm": arm,
                          "target_gap_scale": experiment.ARMS[arm][0],
                          "denominator_gap_scale": experiment.ARMS[arm][1]},
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
        }

    def test_all_four_fresh_cells_validate_and_share_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_records = {}
            for arm in ("A", "B"):
                state = self.make_state(arm)
                path = root / f"{arm}.pt"
                torch.save(state, path)
                source_records[arm] = {
                    "path": str(path.resolve()), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "internal_state_sha256": schedule_switch.internal_state_hashes(state),
                }
            loaded = {}
            for cell, (origin, continuation) in experiment.CELLS.items():
                manifest = {
                    "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
                    "experiment_protocol": schedule_switch.FRESH_N12_PROTOCOL,
                    "run_kind": "formal", "branch": cell, "seed": 31,
                    "origin_arm": origin, "continuation_arm": continuation,
                    "switch_kimg": 512, "final_kimg": 1024,
                    "protocol_sha256": "1" * 64, "implementation_commit": "2" * 40,
                    "source_checkpoint_manifest_sha256": "3" * 64,
                    "source_state": source_records[origin],
                    "immutable_output_root": str(root / cell),
                }
                path = root / f"{cell}.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                loaded[cell] = schedule_switch.load_run_manifest(path)
            self.assertEqual(loaded["AA"]["source_state"]["sha256"],
                             loaded["AB"]["source_state"]["sha256"])
            self.assertEqual(loaded["BA"]["source_state"]["sha256"],
                             loaded["BB"]["source_state"]["sha256"])
            self.assertNotEqual(loaded["AA"]["source_state"]["sha256"],
                                loaded["BA"]["source_state"]["sha256"])


class FrozenStatisticsTests(unittest.TestCase):
    @staticmethod
    def summary(ci95, ci90, negatives=12, loso=-0.04):
        return {"ci95_two_sided": list(ci95), "ci90_two_sided": list(ci90),
                "negative_count": negatives, "leave_one_seed_out_means": [loso] * 12}

    def test_primary_categories_and_equivalence_precedence(self):
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((-0.05, -0.04), (-0.049, -0.041)))
        self.assertEqual(verdict, "STRONG_SUCCESS")
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((-0.01, 0.01), (-0.008, 0.008), negatives=6, loso=0.0))
        self.assertEqual(verdict, "INFORMATIVE_PRACTICAL_NULL")
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((-0.05, -0.001), (-0.04, -0.004), negatives=9, loso=-0.01))
        self.assertEqual(verdict, "WEAK_DIRECTIONAL_REPLICATION")
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((-0.05, 0.05), (-0.04, 0.04), negatives=6, loso=0.0))
        self.assertEqual(verdict, "INCONCLUSIVE")
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((0.001, 0.02), (0.003, 0.018), negatives=0, loso=0.01))
        self.assertEqual(verdict, "INFORMATIVE_PRACTICAL_NULL")
        verdict, _ = frozen_statistics.primary_verdict(
            self.summary((0.04, 0.08), (0.045, 0.075), negatives=0, loso=0.06))
        self.assertEqual(verdict, "OPPOSITE_DIRECTION_FALSIFICATION")


class BlindEvaluationManifestTests(unittest.TestCase):
    def test_public_manifest_is_opaque_complete_and_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal"
            training = formal / "training"
            training.mkdir(parents=True)
            protocol = {
                "schema": experiment.PROTOCOL_SCHEMA,
                "experiment_id": experiment.EXPERIMENT_ID,
                "seeds": list(experiment.SEEDS),
                "evaluation": {"expected_jobs": 264, "shuffle_seed": 20260831,
                               "task_definitions": experiment.planned_evaluation_jobs()},
                "training": {"world_size": 1},
                "gpu_assignment": [experiment.assignment(seed) for seed in experiment.SEEDS],
                "storage_plan": {"full_state_count": 228, "ema_snapshot_count": 228,
                                 "headroom_fraction": 0.30, "estimated_bytes": 100,
                                 "minimum_free_bytes": 130,
                                 "kid_fid_generated_features_are_hardlinked_after_byte_identity_validation": True},
                "paths": {"formal_output_root": str(formal)},
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
            protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            (root / "protocol.sha256").write_text(f"{protocol_sha}  protocol.json\n", encoding="ascii")
            (formal / "training_matrix_completion_receipt.json").write_text(
                json.dumps({"status": "PASS", "protocol_sha256": protocol_sha}), encoding="utf-8")
            (formal / "training_integrity_report.json").write_text(
                json.dumps({"status": "PASS", "protocol_sha256": protocol_sha}), encoding="utf-8")
            for seed in experiment.SEEDS:
                seed_root = training / f"seed{seed}"
                seed_root.mkdir()
                (seed_root / "seed_completion_receipt.json").write_text(
                    json.dumps({"status": "PASS"}), encoding="utf-8")
                for arm in experiment.ARMS:
                    path = seed_root / f"prefix_{arm}" / "kimg0512"
                    path.mkdir(parents=True)
                    (path / "network-snapshot.pkl").write_bytes(f"{seed}-{arm}".encode())
                for cell in experiment.CELLS:
                    for budget in (640, 768, 896, 1024):
                        path = seed_root / cell / f"kimg{budget:04d}"
                        path.mkdir(parents=True)
                        (path / "network-snapshot.pkl").write_bytes(f"{seed}-{cell}-{budget}".encode())
            public_path = root / "control" / "public.json"
            private_path = root / "control" / "private.json"
            evaluation.prepare(SimpleNamespace(protocol=protocol_path,
                                               public_manifest=public_path,
                                               private_map=private_path))
            public = json.loads(public_path.read_text(encoding="utf-8"))
            private = json.loads(private_path.read_text(encoding="utf-8"))
            self.assertEqual(len(public["jobs"]), 264)
            self.assertEqual(len({job["opaque_id"] for job in public["jobs"]}), 264)
            self.assertEqual([sum(job["gpu_index"] == gpu for job in public["jobs"])
                              for gpu in range(6)], [44] * 6)
            allowed = {"queue_index", "opaque_id", "gpu_index", "checkpoint_alias",
                       "checkpoint_sha256", "status"}
            self.assertTrue(all(set(job) == allowed for job in public["jobs"]))
            self.assertNotIn("seed", json.dumps(public["jobs"]))
            self.assertEqual(public["private_map_sha256"],
                             hashlib.sha256(private_path.read_bytes()).hexdigest())
            self.assertEqual(len(private["jobs"]), 264)


if __name__ == "__main__":
    unittest.main()
