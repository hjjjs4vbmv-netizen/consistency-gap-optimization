import json
import math
import unittest
from pathlib import Path

from analysis.noise_floor import analyze, run_matrix, validation


MANIFEST = Path(__file__).parents[1] / "analysis" / "noise_floor" / "manifest.json"
N30_MANIFEST = MANIFEST.with_name("n30_companion.json")


class NoiseFloorTests(unittest.TestCase):
    def test_full_matrix_and_blocks_are_exact(self):
        manifest = run_matrix.load_manifest(MANIFEST)
        jobs = run_matrix.build_jobs(manifest)

        self.assertEqual(len(manifest["checkpoints"]), 16)
        self.assertEqual(len(jobs), 160)
        self.assertEqual(len({job["job_id"] for job in jobs}), 160)
        self.assertEqual(
            [(block["start"], block["end"]) for block in manifest["blocks"]],
            [
                (50000, 99999),
                (100000, 149999),
                (150000, 199999),
                (200000, 249999),
                (250000, 299999),
            ],
        )
        self.assertLessEqual(
            set(manifest["canary_jobs"]), {job["job_id"] for job in jobs})
        self.assertEqual(manifest["not_evaluated"], [])
        self.assertEqual(
            {item["id"] for item in manifest["contrasts"]},
            {"Q256_Q_seed5", "Q256_HA_seed3", "Q256_HA_seed4",
             "Q256_HA_seed5", "TW_BD_256", "TW_BD_1024",
             "Q128_Bsame_A", "Q128_Cmatch_Bmatch"},
        )
        self.assertEqual(
            [item["id"] for item in manifest["checkpoints"][:13]],
            ["NF-01", "NF-03", "NF-05", "NF-06", "NF-07", "NF-08",
             "NF-09", "NF-10", "NF-11", "NF-13", "NF-14", "NF-15",
             "NF-16"],
        )
        supplement = run_matrix.select_jobs(jobs, "NF-02,NF-04,NF-12")
        self.assertEqual(len(supplement), 30)
        self.assertEqual([job["job_index"] for job in supplement], list(range(130, 160)))

    def test_n30_companion_selection_is_mechanical(self):
        manifest = run_matrix.load_manifest(N30_MANIFEST)
        jobs = run_matrix.build_jobs(manifest)

        self.assertEqual(len(manifest["checkpoints"]), 6)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(
            [(item["seed"], item["arm"]) for item in manifest["checkpoints"]],
            [(50, "AA"), (50, "BA"), (51, "AA"), (51, "BA"),
             (52, "AA"), (52, "BA")],
        )
        self.assertEqual(manifest["rotations"], [])
        self.assertEqual(manifest["expected_rotation_rows"], 0)

    def test_summary_uses_only_five_analysis_blocks(self):
        summary = analyze.summarize([1, 2, 3, 4, 5])

        self.assertEqual(summary["mean"], 3)
        self.assertTrue(math.isclose(summary["sd"], math.sqrt(2.5)))
        self.assertTrue(math.isclose(summary["two_sd"], 2 * math.sqrt(2.5)))
        self.assertEqual(
            analyze.summarize([1, 2, 3, 4, None])["status"], "INCOMPLETE")

    def test_missing_b0_anchor_does_not_block_generation_summary(self):
        checkpoints = [
            {"id": "BA", "identity": "BA", "cohort": "n30",
             "b0": {"1": {"fid": 2.0, "kid": 0.2},
                    "2": {"fid": None, "kid": None}}},
            {"id": "AA", "identity": "AA", "cohort": "n30",
             "b0": {"1": {"fid": 3.0, "kid": 0.3},
                    "2": {"fid": None, "kid": None}}},
        ]
        manifest = {
            "checkpoints": checkpoints,
            "expected_checkpoints": 2,
            "contrasts": [{"id": "BA_MINUS_AA", "lhs": "BA", "rhs": "AA"}],
            "expected_contrasts": 1,
        }
        values = {}
        for checkpoint in checkpoints:
            for nfe in (1, 2):
                for metric in ("fid", "kid"):
                    for index in range(1, 6):
                        values[(checkpoint["id"], nfe, metric, f"B{index}")] = float(index)

        checkpoint_rows = analyze.checkpoint_rows(manifest, values)
        contrast_rows, _ = analyze.contrast_rows(manifest, values)

        self.assertIsNone(next(row for row in checkpoint_rows
                               if row["nfe"] == 2)["anchor_B0"])
        nfe2 = next(row for row in contrast_rows
                    if row["nfe"] == 2 and row["metric"] == "fid")
        self.assertEqual(nfe2["status"], "COMPLETE")
        self.assertIsNone(nfe2["anchor_B0"])
        self.assertIsNone(nfe2["same_anchor_sign_k"])

    def test_gpu_inputs_and_nfe_load_are_balanced(self):
        with self.assertRaises(RuntimeError):
            run_matrix.parse_gpus("0,0")
        self.assertEqual(run_matrix.parse_gpus("0,1"), [0, 1])
        manifest = run_matrix.load_manifest(MANIFEST)
        jobs = run_matrix.build_jobs(manifest)
        counts = {(slot, nfe): 0 for slot in (0, 1) for nfe in (1, 2)}
        for job in jobs:
            counts[(run_matrix.gpu_slot(job["job_index"]), job["nfe"])] += 1
        for nfe in (1, 2):
            self.assertLessEqual(abs(counts[(0, nfe)] - counts[(1, nfe)]), 1)

    def test_stale_receipt_binding_is_rejected(self):
        manifest = run_matrix.load_manifest(MANIFEST)
        job = run_matrix.build_jobs(manifest)[0]
        receipt = {
            "status": "PASS",
            "job_id": job["job_id"],
            "job_index": job["job_index"],
            "checkpoint_id": job["checkpoint"]["id"],
            "checkpoint": job["checkpoint"]["path"],
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "block": job["block"],
            "nfe": job["nfe"],
            "metric_seed": manifest["metric_seed"],
            "evaluator_commit": "stale",
            "dataset_sha256": manifest["dataset_sha256"],
            "runtime_sha256": manifest["runtime_sha256"],
        }
        with self.assertRaisesRegex(RuntimeError, "evaluator_commit"):
            validation.validate_receipt(manifest, job, receipt)

    def test_frozen_contrast_orientation_and_b0_rotation(self):
        manifest = json.loads(MANIFEST.read_text())
        checkpoints = {item["id"]: item for item in manifest["checkpoints"]}

        q = math.log(checkpoints["NF-06"]["b0"]["1"]["fid"]) - math.log(
            checkpoints["NF-05"]["b0"]["1"]["fid"])
        early_kid = (
            checkpoints["NF-07"]["b0"]["1"]["kid"]
            - checkpoints["NF-13"]["b0"]["1"]["kid"]
        )
        late_kid = (
            checkpoints["NF-14"]["b0"]["1"]["kid"]
            - checkpoints["NF-08"]["b0"]["1"]["kid"]
        )

        self.assertGreater(q, 0)
        self.assertLess(early_kid, 0)
        self.assertGreater(late_kid, 0)
        self.assertLess(early_kid * late_kid, 0)

        history = math.log(checkpoints["NF-12"]["b0"]["1"]["fid"]) - math.log(
            checkpoints["NF-11"]["b0"]["1"]["fid"])
        self.assertLess(history, 0)
        delayed = next(item for item in manifest["rotations"]
                       if item["id"] == "DR_Q256_SEED5")
        self.assertEqual((delayed["nfe"], delayed["metric"]), (1, "fid"))

    def test_empty_not_evaluated_csv_has_header(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not_evaluated.csv"
            analyze.write_csv(path, [], fieldnames=("id", "reason"))
            self.assertEqual(path.read_text(), "id,reason\n")

    def test_empty_rotation_csv_has_header(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotation_variation.csv"
            analyze.write_csv(path, [], fieldnames=analyze.ROTATION_FIELDS)
            self.assertEqual(path.read_text(), ",".join(analyze.ROTATION_FIELDS) + "\n")


if __name__ == "__main__":
    unittest.main()
