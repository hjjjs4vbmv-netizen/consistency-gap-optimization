import hashlib, json, math, unittest
from pathlib import Path
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

if __name__=="__main__": unittest.main()
