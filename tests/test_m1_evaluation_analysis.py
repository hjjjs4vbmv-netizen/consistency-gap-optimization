import math
import unittest

from scripts import build_m1_evaluation_slots as slots
from scripts import summarize_m1_results as summary


def frozen_inventory():
    candidates = []
    for seed in range(50, 80):
        checked = seed < 66
        row = {
            "seed": seed,
            "checked": checked,
            "qualified": checked,
            "reason": "QUALIFIED" if checked else "NOT_CHECKED_AFTER_ROSTER_FILLED",
        }
        if checked:
            row["sources"] = {
                arm: {
                    "source_state_path": f"/sources/seed{seed}/{arm}.pt",
                    "source_state_bytes": 1000 + seed,
                    "source_state_sha256": ("a" if arm == "A" else "b") * 64,
                    "provenance_receipt_path": f"/sources/seed{seed}/{arm}.json",
                    "provenance_receipt_sha256": "c" * 64,
                    "internal_state_sha256": {"net": "d" * 64, "rng": ["e" * 64]},
                }
                for arm in ("A", "B")
            }
        candidates.append(row)
    return {
        "schema": "ect.m1.source-inventory/v1",
        "status": "PASS",
        "candidates": candidates,
    }


def frozen_training_identity():
    return {
        "training_manifest_sha256": "f" * 64,
        "implementation_commit": "2" * 40,
        "training_runtime_receipt_sha256": "4" * 64,
        "sources": {
            (seed, arm): ("a" if arm == "A" else "b") * 64
            for seed in range(50, 66)
            for arm in ("A", "B")
        },
    }


class M1EvaluationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.roster = slots.normalize_roster(frozen_inventory())
        self.manifest = slots.build_slots(self.roster, frozen_training_identity())
        self.results = [self.result_for(row) for row in self.manifest]

    @staticmethod
    def result_for(row):
        seed_offset = -0.001 * (row["seed"] - 50)
        block_effect = {"B0": -0.06, "B1": -0.09, "B2": -0.12}
        log_fid = math.log(10.0)
        if row["readout"] == "E_512":
            if row["branch"] == "K_B":
                log_fid -= 0.02
            elif row["branch"] == "R_B":
                log_fid += block_effect[row["block"]] + seed_offset
        return {
            "slot_id": row["slot_id"],
            "status": "SEALED_PASS",
            "fid50k_full": math.exp(log_fid),
            "kid50k_full": 0.001,
        }

    def set_status(self, slot_id, status):
        for row in self.results:
            if row["slot_id"] == slot_id:
                row["status"] = status
                row["fid50k_full"] = ""
                return
        self.fail(f"missing fixture slot {slot_id}")

    def test_exact_320_slot_matrix_and_three_generation_blocks(self):
        self.assertEqual(len(self.manifest), 320)
        self.assertEqual(len({row["slot_id"] for row in self.manifest}), 320)
        counts = {}
        for row in self.manifest:
            counts[row["readout"]] = counts.get(row["readout"], 0) + 1
        self.assertEqual(counts, {"ONLINE": 64, "E_KEEP": 64, "E_512": 192})
        for block, expected in slots.BLOCKS.items():
            rows = [row for row in self.manifest if row["block"] == block]
            expected_count = 192 if block == "B0" else 64
            self.assertEqual(len(rows), expected_count)
            self.assertTrue(all(
                (row["sample_seed_start"], row["sample_seed_end"]) == expected
                for row in rows
            ))
        self.assertTrue(all(row["budget_kimg"] == 1024 for row in self.manifest))
        self.assertTrue(all(row["nfe"] == 1 for row in self.manifest))
        self.assertTrue(all(
            row["evaluator_commit"] == slots.EVALUATOR_COMMIT
            for row in self.manifest
        ))

    def test_roster_requires_ordered_checked_inventory_and_source_identity(self):
        with self.assertRaises(slots.SlotError):
            slots.normalize_roster(list(range(50, 66)))
        inventory = frozen_inventory()
        inventory["candidates"][0], inventory["candidates"][1] = (
            inventory["candidates"][1], inventory["candidates"][0]
        )
        with self.assertRaisesRegex(slots.SlotError, "strictly ordered"):
            slots.normalize_roster(inventory)
        inventory = frozen_inventory()
        inventory["candidates"][16]["checked"] = True
        inventory["candidates"][16]["sources"] = inventory["candidates"][15]["sources"]
        with self.assertRaisesRegex(slots.SlotError, "after the 16th"):
            slots.normalize_roster(inventory)
        inventory = frozen_inventory()
        del inventory["candidates"][0]["sources"]["A"]["source_state_sha256"]
        with self.assertRaisesRegex(slots.SlotError, "SHA256"):
            slots.normalize_roster(inventory)

    def test_checked_unqualified_candidate_is_recorded_then_skipped(self):
        inventory = frozen_inventory()
        missing = inventory["candidates"][0]
        missing["qualified"] = False
        missing["reason"] = "MISSING_SOURCE_STATE"
        missing["sources"] = {
            arm: {
                "expected": {
                    "source_state_path": f"/expected/seed50/{arm}.pt",
                    "source_state_bytes": 1050,
                    "source_state_sha256": ("a" if arm == "A" else "b") * 64,
                    "provenance_receipt_path": f"/expected/seed50/{arm}.json",
                    "provenance_receipt_sha256": "c" * 64,
                },
                "actual": {
                    "source_state_path": None,
                    "source_state_bytes": None,
                    "source_state_sha256": None,
                    "provenance_receipt_path": None,
                    "provenance_receipt_sha256": None,
                },
            }
            for arm in ("A", "B")
        }
        replacement = inventory["candidates"][16]
        replacement.update(
            checked=True,
            qualified=True,
            reason="QUALIFIED",
            sources={
                arm: {
                    "source_state_path": f"/sources/seed66/{arm}.pt",
                    "source_state_bytes": 1066,
                    "source_state_sha256": ("a" if arm == "A" else "b") * 64,
                    "provenance_receipt_path": f"/sources/seed66/{arm}.json",
                    "provenance_receipt_sha256": "c" * 64,
                    "internal_state_sha256": {"net": "d" * 64, "rng": ["e" * 64]},
                }
                for arm in ("A", "B")
            },
        )
        roster = slots.normalize_roster(inventory)
        self.assertEqual([row["seed"] for row in roster], list(range(51, 67)))

    def test_complete_analysis_averages_log_differences_within_seed(self):
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["matrix_status"], "RESOLVED")
        self.assertEqual(report["primary"]["n_R"], 16)
        self.assertEqual(report["secondary"]["n_4"], 16)
        self.assertEqual(
            report["primary"]["status"],
            "B_ADVANTAGE_SUPPORTED_CONDITIONAL",
        )
        self.assertEqual(report["secondary"]["status"], "ESTIMATE_ONLY")
        first = report["primary"]["per_seed"][0]
        interaction = report["secondary"]["per_seed"][0]
        self.assertAlmostEqual(first["d"], (-0.06 - 0.09 - 0.12) / 3)
        self.assertAlmostEqual(interaction["i"], first["d"] - (-0.02))

    def test_missing_primary_block_keeps_primary_and_secondary_incomplete(self):
        self.results = [
            row for row in self.results
            if row["slot_id"] != "S01-R_B-E_512-B1"
        ]
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["matrix_status"], "INCOMPLETE_SLOT_LEDGER")
        self.assertEqual(report["primary"]["status"], "INCOMPLETE_TECHNICAL")
        self.assertEqual(
            report["secondary"]["status"],
            "INCOMPLETE_TECHNICAL_SECONDARY",
        )
        self.assertNotIn(50, report["primary"]["S_R"])
        self.assertNotIn("mean", report["primary"])

    def test_k_scientific_failure_does_not_remove_valid_r_pair(self):
        for block in slots.BLOCKS:
            self.set_status(f"S09-K_A-E_512-{block}", "NOT_RUN_NO_ENDPOINT")
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["matrix_status"], "RESOLVED")
        self.assertEqual(report["primary"]["n_R"], 16)
        self.assertIn(58, report["primary"]["S_R"])
        self.assertEqual(report["secondary"]["n_4"], 15)
        self.assertNotIn(58, report["secondary"]["S_4"])
        retained_offsets = [offset for offset in range(16) if offset != 8]
        expected_interaction = -0.07 - sum(retained_offsets) / len(retained_offsets) * 0.001
        self.assertAlmostEqual(report["secondary"]["mean"], expected_interaction)

    def test_r_scientific_failure_changes_only_conditional_set(self):
        for block in slots.BLOCKS:
            self.set_status(f"S16-R_A-E_512-{block}", "NOT_RUN_NO_ENDPOINT")
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["primary"]["n_R"], 15)
        self.assertNotIn(65, report["primary"]["S_R"])
        self.assertEqual(
            report["primary"]["status"],
            "B_ADVANTAGE_SUPPORTED_CONDITIONAL",
        )

    def test_k_technical_missingness_blocks_secondary_but_not_primary(self):
        self.set_status("S09-K_A-E_512-B2", "INCOMPLETE_TECHNICAL")
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["primary"]["n_R"], 16)
        self.assertEqual(
            report["primary"]["status"],
            "B_ADVANTAGE_SUPPORTED_CONDITIONAL",
        )
        self.assertEqual(
            report["secondary"]["status"],
            "INCOMPLETE_TECHNICAL_SECONDARY",
        )
        self.assertNotIn("mean", report["secondary"])

    def test_kid_only_technical_status_does_not_remove_fid(self):
        for row in self.results:
            row["fid_status"] = "SEALED_PASS"
            row["kid_status"] = "INCOMPLETE_TECHNICAL"
            row["status"] = "SEALED_PARTIAL"
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(
            report["matrix_status"], "RESOLVED_WITH_KID_MISSINGNESS"
        )
        self.assertEqual(report["primary"]["n_R"], 16)
        self.assertEqual(
            report["primary"]["status"],
            "B_ADVANTAGE_SUPPORTED_CONDITIONAL",
        )

    def test_summary_rejects_noncanonical_non_e512_slot(self):
        self.manifest[0]["readout"] = "E_512"
        with self.assertRaisesRegex(summary.SummaryError, "canonical"):
            summary.summarize(self.manifest, self.results)

    def test_matrix_invalid_precedes_missing_result(self):
        self.results = self.results[1:]
        self.set_status("S01-K_A-E_KEEP-B0", "INVALID_IMPLEMENTATION")
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["matrix_status"], "INVALID_IMPLEMENTATION")

    def test_kid_invalid_also_precedes_missing_but_not_primary(self):
        self.results = self.results[1:]
        self.results[0]["kid_status"] = "INVALID_IMPLEMENTATION"
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["matrix_status"], "INVALID_IMPLEMENTATION")
        self.assertEqual(report["primary"]["n_R"], 16)

    def test_zero_sd_reports_effect_scale_without_verdict(self):
        for row in self.results:
            manifest = next(item for item in self.manifest if item["slot_id"] == row["slot_id"])
            if manifest["readout"] == "E_512" and manifest["branch"] == "R_B":
                row["fid50k_full"] = 9.0
        report = summary.summarize(self.manifest, self.results)
        primary = report["primary"]
        self.assertEqual(primary["status"], "DEGENERATE_ZERO_SD_DESCRIPTIVE_ONLY")
        self.assertAlmostEqual(primary["geometric_fid_ratio"], 0.9)
        self.assertAlmostEqual(primary["improvement_percent"], 10.0)
        self.assertIsNone(primary["geometric_fid_ratio_ci95"])

    def test_complete_report_closes_descriptive_and_denominator_outputs(self):
        report = summary.summarize(self.manifest, self.results)
        self.assertEqual(report["primary"]["n_R_denominator"], 16)
        self.assertEqual(report["secondary"]["n_4_denominator"], 16)
        self.assertEqual(len(report["full_seed_arm_status"]), 16)
        self.assertEqual(report["description_table"]["rows"], 320)
        self.assertEqual(report["description_table"]["b0_three_readout_rows"], 192)
        self.assertEqual(sum(report["primary"]["block_direction_counts"].values()), 48)
        first_block = report["primary"]["per_seed"][0]["blocks"][0]
        self.assertIn("R_A_fid50k_full", first_block)
        self.assertIn("R_B_fid50k_full", first_block)
        manifest = summary._manifest_index(self.manifest)
        results = summary._result_index(self.results, set(manifest))
        descriptions = summary.build_description_rows(manifest, results)
        self.assertEqual(len(descriptions), 320)
        self.assertEqual(
            {row["readout"] for row in descriptions if row["block"] == "B0"},
            {"ONLINE", "E_KEEP", "E_512"},
        )
        self.assertTrue(all("kid_status" in row for row in descriptions))


if __name__ == "__main__":
    unittest.main()
