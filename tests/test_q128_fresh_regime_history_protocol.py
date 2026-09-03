import csv, hashlib, json, math, tempfile, unittest
from pathlib import Path
from scripts.run_q128_fresh_regime_history_n8_v1 import telemetry_gate
from training import schedule_switch

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"analysis/q128_fresh_regime_history_n8_v1"

class ProtocolTest(unittest.TestCase):
    def test_frozen_hashes_and_identity(self):
        for line in (A/"protocol.sha256").read_text().splitlines():
            digest, rel=line.split("  ")
            path=(A/rel).resolve()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
        p=json.loads((A/"protocol.json").read_text())
        self.assertEqual(p["cohort"]["formal_seeds"],list(range(201,209)))
        self.assertEqual(p["cohort"]["replacement_pool"],list(range(209,213)))
        self.assertEqual(set(p["cohort"]["formal_seeds"]) & {3,4,5},set())
    def test_analytic_identity_and_counterbalance(self):
        self.assertEqual(0.55/128,1.10/256)
        p=json.loads((A/"protocol.json").read_text())
        orders=list(p["arm_order"].values())
        self.assertTrue(all(set(x)=={"A","Bsame","Bmatch","Cmatch","Dmatch"} for x in orders))
        for pos in range(5):
            counts={arm:sum(row[pos]==arm for row in orders) for arm in orders[0]}
            self.assertLessEqual(max(counts.values())-min(counts.values()),1)
    def test_q128_switch_protocol_is_separate(self):
        self.assertEqual(schedule_switch.SUPPORTED_PROTOCOL_SEEDS[schedule_switch.Q128_FRESH_PROTOCOL],tuple(range(201,213))+(999,))
        self.assertIn(schedule_switch.PROTOCOL,schedule_switch.SUPPORTED_PROTOCOL_SEEDS)

    def telemetry_row(self, **updates):
        row={
            "attempted_iteration":"1", "successful_optimizer_steps":"0",
            "processed_nimg":"128", "sample_count":"128", "step_skipped":"1",
            "loss":"1.0", "loss_nonfinite_count":"0", "raw_grad_norm":"inf",
            "raw_grad_finite_norm":"2.0", "raw_grad_nonfinite_count":"4",
            "sanitized_grad_norm":"3.0", "sanitized_grad_nonfinite_count":"0",
            "update_norm":"0", "update_nonfinite_count":"0", "model_norm":"4.0",
            "model_nonfinite_count":"0", "ema_norm":"4.0", "ema_nonfinite_count":"0",
            "factor_nonfinite_count":"0", "nonpositive_denominator_count":"0",
            "target_r_equal_t_count":"0", "target_scaled_to_zero_count":"0",
            "denominator_r_equal_t_count":"0", "denominator_scaled_to_zero_count":"0",
            "target_delta_min":"0.1", "target_delta_mean":"0.2", "target_delta_max":"0.3",
            "denominator_delta_min":"0.1", "denominator_delta_mean":"0.2",
            "denominator_delta_max":"0.3", "learning_rate":"0.0001",
            "grad_scale_before":"65536", "grad_scale_after":"32768",
        }
        row.update(updates)
        return row

    def telemetry_report(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"telemetry.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            return telemetry_gate(path)

    def test_amp_warmup_skip_preserves_finite_state_gate(self):
        success=self.telemetry_row(
            attempted_iteration="2", successful_optimizer_steps="1", processed_nimg="256",
            step_skipped="0", raw_grad_norm="5.0", raw_grad_nonfinite_count="0",
            update_norm="0.01", grad_scale_before="32768", grad_scale_after="32768")
        report=self.telemetry_report([self.telemetry_row(), success])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["amp_skip_attempts"], [1])

    def test_raw_nonfinite_without_skip_fails_closed(self):
        row=self.telemetry_row(
            step_skipped="0", successful_optimizer_steps="1", raw_grad_norm="inf",
            update_norm="0.01", grad_scale_after="65536")
        report=self.telemetry_report([row])
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("must match AMP skip" in item for item in report["failures"]))

    def test_late_amp_skip_fails_closed(self):
        report=self.telemetry_report([self.telemetry_row(processed_nimg="10000")])
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("outside frozen warm-up" in item for item in report["failures"]))

if __name__=="__main__": unittest.main()
