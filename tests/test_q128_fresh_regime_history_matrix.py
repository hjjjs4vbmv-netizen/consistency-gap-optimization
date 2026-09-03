import json, subprocess, sys, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"analysis/q128_fresh_regime_history_n8_v1"
class MatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable,str(ROOT/"scripts/generate_q128_fresh_evaluation_manifest.py")],check=True)
        cls.m=json.loads((A/"evaluation_manifest.json").read_text())
    def test_complete_unique_opaque_matrix(self):
        self.assertEqual(self.m["job_count"],272)
        self.assertEqual(len({x["opaque_id"] for x in self.m["jobs"]}),272)
        self.assertTrue(all(len(x["opaque_id"])==20 for x in self.m["jobs"]))
    def test_primary_and_key_secondary_counts(self):
        self.assertEqual(sum(x["category"]=="PRIMARY" for x in self.m["jobs"]),48)
        self.assertEqual(sum(x["category"]=="KEY_SECONDARY" for x in self.m["jobs"]),32)
        self.assertTrue(all(x["checkpoint_sha256"] is None for x in self.m["jobs"]))
    def test_required_cells(self):
        cells={(x["seed"],x["trajectory"],x["budget_kimg"],x["nfe"]) for x in self.m["jobs"]}
        for seed in range(201,209):
            for arm in ("A","Bsame","Bmatch","Cmatch","Dmatch"):
                for budget in (512,768,1024): self.assertIn((seed,arm,budget,1),cells)
                self.assertIn((seed,arm,1024,2),cells)
            for branch in ("AB","BA"):
                for budget in (640,768,896,1024): self.assertIn((seed,branch,budget,1),cells)

if __name__=="__main__": unittest.main()
