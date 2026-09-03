import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.q256_switchpoint_sweep import analyze, companion_summary, result_conversion


def rows_for(pattern, missing=None, root_cause=""):
    missing = missing or set()
    rows = []
    for seed in analyze.SEEDS:
        for switch, g_value in zip(analyze.SWITCH_POINTS, pattern):
            invalid = (seed, switch) in missing
            ctrl_fid = 10.0
            rows.append(
                {
                    "seed": seed,
                    "s_kimg": switch,
                    "ba_valid": not invalid,
                    "ctrl_valid": True,
                    "ba_fid50k": None if invalid else ctrl_fid * math.exp(g_value),
                    "ctrl_fid50k": ctrl_fid,
                    "root_cause": root_cause if invalid else "",
                }
            )
    return rows


def decoded_matrix(failed=None):
    failed = failed or set()
    results = []
    for seed in analyze.SEEDS:
        for kimg in (640, 768, 896, 1024):
            results.append(decoded_cell(seed, "CTRL", kimg, "primary", 10.0, failed))
        for switch, g_value in zip(analyze.SWITCH_POINTS, (-0.1, -0.2, -0.3, -0.4)):
            results.append(decoded_cell(
                seed, f"BA{switch}", analyze.ENDPOINTS[switch], "primary",
                10.0 * math.exp(g_value), failed,
            ))
        for switch, h_value in zip((128, 256, 384), (-0.15, -0.25, -0.35)):
            results.append(decoded_cell(
                seed, f"BA{switch}", 1024, "secondary", 10.0 * math.exp(h_value), failed,
            ))
    return {"status": "PASS", "results": results}


def decoded_cell(seed, trajectory, kimg, role, fid, failed):
    identity = (seed, trajectory, kimg)
    row = {
        "seed": seed, "trajectory": trajectory, "kimg": kimg, "role": role,
        "opaque_id": f"job-{seed}-{trajectory}-{kimg}",
        "status": "EXHAUSTED_FAILURE" if identity in failed else "PASS",
    }
    if row["status"] == "PASS":
        row["fid50k_full"] = fid
        row["kid50k_full"] = 0.001
    return row


class PageTestTest(unittest.TestCase):
    def test_strong_frozen_direction_is_ordered(self):
        result = analyze.analyze(rows_for([-0.1, -0.2, -0.3, -0.4]))
        self.assertEqual(result["page_test"]["verdict"], analyze.ORDERED_VERDICT)
        self.assertEqual(result["page_test"]["p_exact"], 1 / (24 ** 12))

    def test_reverse_direction_has_unit_p_value(self):
        result = analyze.analyze(rows_for([-0.4, -0.3, -0.2, -0.1]))
        self.assertEqual(result["page_test"]["p_exact"], 1.0)
        self.assertEqual(result["page_test"]["verdict"], "ORDERING_NOT_RESOLVED")

    def test_exact_ties_use_midrank(self):
        test = analyze.exact_page_test([[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(test["L_observed"], 25.0)
        self.assertEqual(test["p_exact"], 1.0)

    def test_nontrivial_exact_tail_matches_enumerated_reference(self):
        test = analyze.exact_page_test([[4.0, 1.0, 3.0, 2.0]])
        self.assertEqual(test["L_observed"], 27.0)
        self.assertEqual(test["p_exact"], 9 / 24)


class CompletenessAndSummariesTest(unittest.TestCase):
    def test_nine_complete_seeds_run_unchanged_four_point_primary(self):
        missing = {(seed, 128) for seed in (81, 82, 83)}
        result = analyze.analyze(rows_for([-0.1, -0.2, -0.3, -0.4], missing, "BA_128"))
        self.assertEqual(result["n_complete"], 9)
        self.assertEqual(result["primary_status"], "ANALYZED")
        self.assertEqual(result["missingness"]["arm_concentration_flag"], ["BA_128"])
        self.assertEqual(result["page_test"]["verdict"], analyze.ORDERED_VERDICT)

    def test_below_nine_aborts_primary(self):
        missing = {(seed, 128) for seed in (81, 82, 83, 84)}
        result = analyze.analyze(rows_for([-0.1, -0.2, -0.3, -0.4], missing, "BA_128"))
        self.assertEqual(result["primary_status"], "ABORTED_INCOMPLETE")
        self.assertIsNone(result["page_test"])
        self.assertEqual(result["allowed_wording_key"], "DESCRIPTIVE_ONLY")

    def test_descriptive_and_adjacent_summaries(self):
        result = analyze.analyze(rows_for([-0.1, -0.2, -0.3, -0.4]))
        point = result["point_summaries"]["G_128"]
        self.assertAlmostEqual(point["median"], -0.1)
        self.assertAlmostEqual(point["sample_sd"], 0.0)
        self.assertEqual(point["sign_counts"], {"negative": 12, "zero": 0, "positive": 0})
        adjacent = result["adjacent_paired_differences"]["G_256_minus_G_128"]
        self.assertAlmostEqual(adjacent["mean"], -0.1)
        self.assertEqual(adjacent["sign_counts"]["negative"], 12)

    def test_odd_and_even_medians(self):
        self.assertEqual(analyze.descriptive_summary([3.0, 1.0, 2.0])["median"], 2.0)
        self.assertEqual(analyze.descriptive_summary([1.0, 2.0, 100.0, 200.0])["median"], 51.0)


class DecodedResultsConversionTest(unittest.TestCase):
    def test_decode_generates_primary_csv_and_descriptive_h(self):
        decoded = decoded_matrix()
        rows, h_values = result_conversion.convert_decoded(decoded)
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(row["seed_complete"] for row in rows))
        common = analyze.summarize_common_endpoint(h_values)
        self.assertEqual(common["inferential_role"], "NONE")
        self.assertEqual(common["points"]["H_128"]["summary"]["n"], 12)
        self.assertNotIn("verdict", common)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decoded_path = root / "decoded_results.json"
            decoded_path.write_text(json.dumps(decoded), encoding="utf-8")
            with mock.patch("sys.argv", ["analyze.py", "--decoded-results", str(decoded_path),
                                         "--output-dir", str(root / "out")]):
                analyze.main()
            path = root / "out" / "fixed_chase_seed_results.csv"
            reread = result_conversion.read_rows(path)
            self.assertEqual(analyze.analyze(reread)["n_complete"], 12)
            self.assertTrue((root / "out" / "common_endpoint_descriptive.json").is_file())
            self.assertTrue((root / "out" / "FINAL_ANALYSIS.md").is_file())

    def test_secondary_failure_does_not_change_primary_completeness(self):
        failed = {(81, "BA128", 1024)}
        rows, h_values = result_conversion.convert_decoded(decoded_matrix(failed))
        primary = analyze.analyze(rows)
        common = analyze.summarize_common_endpoint(h_values)
        self.assertEqual(primary["n_complete"], 12)
        self.assertEqual(common["points"]["H_128"]["available_seeds"], list(range(82, 93)))
        self.assertEqual(common["points"]["H_256"]["summary"]["n"], 12)

    def test_decoded_root_cause_is_preserved(self):
        decoded = decoded_matrix({(81, "BA128", 640)})
        failed = next(row for row in decoded["results"] if row["status"] == "EXHAUSTED_FAILURE")
        failed["root_cause"] = "B_TRUNK_FAILURE"
        rows, _ = result_conversion.convert_decoded(decoded)
        seed_rows = [row for row in rows if row["seed"] == 81]
        self.assertEqual({row["root_cause"] for row in seed_rows}, {"B_TRUNK_FAILURE"})


class CompanionSummaryTest(unittest.TestCase):
    def test_five_generation_blocks_are_descriptive_only(self):
        decoded = decoded_matrix()
        receipts = []
        for block in range(1, 5):
            for trajectory, fid in (("CTRL", 10.0), ("BA512", 9.0 - block / 10)):
                receipts.append({"status": "PASS", "job": {"block": block, "trajectory": trajectory},
                                 "values": {"fid50k_full": fid}})
        result = companion_summary.summarize(decoded, receipts)
        self.assertEqual((result["status"], result["n_blocks"]), ("PASS", 5))
        self.assertEqual(result["inferential_role"], "NONE")
        self.assertNotIn("verdict", result)
        self.assertEqual(len(result["blocks"]), 5)


if __name__ == "__main__":
    unittest.main()
