import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.q256_switchpoint_sweep import evaluate, train


def write_training_terminals(training: Path) -> None:
    terminal = training / "terminal"; terminal.mkdir(parents=True)
    identities = {(job["seed"], job["trajectory"]) for job in evaluate.formal_jobs(training)}
    for seed, trajectory in identities:
        (terminal / f"continuation-seed{seed:03d}-{trajectory}.json").write_text(
            json.dumps({"status": "PASS"})
        )


def write_training_matrix(training: Path, protocol: Path) -> None:
    (training / "training_matrix_receipt.json").write_text(json.dumps({
        "status": "PASS", "jobs": 84, "expected_jobs": 84,
        "protocol_sha256": evaluate.experiment.sha256_file(protocol),
    }))


class TrainingPlanTests(unittest.TestCase):
    def test_static_schedule_is_balanced(self):
        jobs = train.training_jobs()
        self.assertEqual(sum(job["phase"] == "prefix" for job in jobs), 24)
        self.assertEqual(sum(job["phase"] == "continuation" for job in jobs), 60)
        loads = {gpu: 0 for gpu in range(8)}
        for job in jobs:
            if job["phase"] == "prefix":
                loads[job["gpu"]] += 512
            else:
                loads[job["gpu"]] += 1024 - job["switch_kimg"]
        self.assertEqual(loads, {gpu: 6528 for gpu in range(8)})

    def test_snapshot_only_command_is_trajectory_consistent(self):
        base = ["python", "--tick=10", "--snap=0", "--dump=7",
                "--immutable-checkpoint-kimg=640,768,896,1024",
                "--planned-pause-protocol=old"]
        prefix_a = train.normalize_command(base, {"phase": "prefix", "arm": "A"})
        prefix_b = train.normalize_command(base, {"phase": "prefix", "arm": "B"})
        suffix = train.normalize_command(
            base, {"phase": "continuation", "arm": "A", "name": "CTRL"}
        )
        for command in (prefix_a, prefix_b, suffix):
            self.assertTrue({"--tick=128", "--snap=0", "--dump=0"}.issubset(command))
        self.assertIn("--immutable-checkpoint-kimg=512", prefix_a)
        self.assertIn("--immutable-checkpoint-kimg=128,256,384,512", prefix_b)
        self.assertIn("--immutable-checkpoint-kimg=640,768,896,1024", suffix)

    def test_each_continuation_saves_only_exact_required_milestones(self):
        base = ["python", "--immutable-checkpoint-kimg=640,768,896,1024"]
        expected = {
            "CTRL": "640,768,896,1024",
            "BA128": "640,1024",
            "BA256": "768,1024",
            "BA384": "896,1024",
            "BA512": "1024",
        }
        for name, milestones in expected.items():
            command = train.normalize_command(
                base, {"phase": "continuation", "arm": "A", "name": name}
            )
            self.assertIn(f"--immutable-checkpoint-kimg={milestones}", command)

    def test_matrix_summary_allows_nine_through_eleven_complete_seeds(self):
        for expected in (9, 10, 11):
            receipts = [{"status": "PASS" if seed < 81 + expected else "EXHAUSTED_FAILURE",
                         "job": {"seed": seed}} for seed in range(81, 93) for _ in range(7)]
            with self.subTest(expected=expected):
                summary = train.matrix_summary(receipts)
                self.assertEqual(summary["status"], "COMPLETE_WITH_FAILURES")
                self.assertEqual(summary["n_complete_seeds"], expected)


class EvaluationPlanTests(unittest.TestCase):
    def test_formal_and_companion_counts_are_exact(self):
        formal = evaluate.formal_jobs(Path("/training"))
        identities = {(j["seed"], j["trajectory"], j["kimg"]) for j in formal}
        self.assertEqual((len(formal), len(identities)), (132, 132))
        self.assertEqual(sum(j["role"] == "primary" for j in formal), 96)
        self.assertEqual(sum(j["role"] == "secondary" for j in formal), 36)
        companions = evaluate.companion_jobs(Path("/training"))
        self.assertEqual(len(companions), 8)
        self.assertEqual([j["gpu"] for j in companions], list(range(8)))
        self.assertEqual({(j["sample_start"], j["sample_end"]) for j in companions}, set(evaluate.BLOCKS))

    def test_prepare_keeps_identity_out_of_public_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); training = root / "training"; evaluation_root = root / "evaluation"
            for job in evaluate.formal_jobs(training):
                checkpoint = Path(job["checkpoint"]); checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"checkpoint")
            write_training_terminals(training)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"paths": {"training": str(training),
                                                       "evaluation": str(evaluation_root)}}))
            write_training_matrix(training, protocol)
            opaque = [f"opaque{i:03d}" for i in range(132)]
            with mock.patch.object(evaluate.secrets, "token_hex", side_effect=opaque):
                evaluate.prepare(protocol)
            public = json.loads((evaluation_root / "formal" / "public_manifest.json").read_text())
            private = json.loads((evaluation_root / "private_map.json").read_text())
            forbidden = {"seed", "trajectory", "kimg", "role"}
            self.assertEqual(len(public["jobs"]), 132)
            self.assertTrue(all(forbidden.isdisjoint(job) for job in public["jobs"]))
            self.assertTrue(all(forbidden.issubset(job) for job in private["jobs"]))

    def test_missing_checkpoint_reaches_documented_terminal_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); training = root / "training"; evaluation_root = root / "evaluation"
            jobs = evaluate.formal_jobs(training)
            for job in jobs[1:]:
                checkpoint = Path(job["checkpoint"]); checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"checkpoint")
            write_training_terminals(training)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"paths": {"training": str(training),
                "evaluation": str(evaluation_root), "assets": str(root / "assets")}}))
            write_training_matrix(training, protocol)
            with mock.patch.object(evaluate.secrets, "token_hex", side_effect=[f"opaque{i:03d}" for i in range(132)]):
                evaluate.prepare(protocol)
            formal = evaluation_root / "formal"; (formal / "receipts").mkdir()
            def pass_attempt(_protocol, job, _repo, _cache, target):
                evaluate.experiment.atomic_json(target / "receipts" / f"{job['opaque_id']}.json", {"status": "PASS"})
            with mock.patch.object(evaluate.evaluation, "evaluator_ok"), mock.patch.object(evaluate, "_formal_attempt", side_effect=pass_attempt):
                for gpu in range(8):
                    evaluate.formal_worker(protocol, root, gpu)
            evaluate.seal(protocol)
            matrix = json.loads((formal / "matrix_seal.json").read_text())
            self.assertEqual(matrix["status"], "SEALED_WITH_DOCUMENTED_FAILURES")
            self.assertEqual(matrix["failures"], 1)
            private = json.loads((evaluation_root / "private_map.json").read_text())
            missing = next(job for job in private["jobs"] if job["training_status"] == "TRAINING_UNAVAILABLE")
            terminal = json.loads((formal / "terminal" / f"{missing['opaque_id']}.json").read_text())
            self.assertEqual((terminal["status"], terminal["attempts"], terminal["root_cause"]),
                             ("EXHAUSTED_FAILURE", 0, "CHECKPOINT_ARTIFACT_MISSING"))

    def test_decode_refuses_an_unsealed_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"paths": {"evaluation": str(root / "evaluation"),
                                                       "analysis": str(root / "analysis")}}))
            with self.assertRaises(FileNotFoundError):
                evaluate.decode(protocol)


if __name__ == "__main__":
    unittest.main()
