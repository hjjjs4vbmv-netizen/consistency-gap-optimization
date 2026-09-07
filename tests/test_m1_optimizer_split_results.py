import copy
import gzip
import json
import unittest
from pathlib import Path

from scripts.analyze_m1_optimizer_split import analyze


class DiagnosticResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        with gzip.open(root/'analysis/q256_optimizer_restart_ema_rebuild_v1/optimizer_split/evidence.json.gz','rt') as f:
            cls.evidence=json.load(f)

    def test_reported_counts_and_updates(self):
        result=analyze(self.evidence)
        self.assertEqual(result['total_observed_attempts'],1938)
        self.assertEqual([result['groups'][o]['completed'] for o in ('K','R','clear_moments','reset_step')],[8,7,8,7])
        self.assertEqual(result['groups']['R']['skipped_attempts'],12)
        self.assertEqual(result['groups']['reset_step']['skipped_attempts'],14)
        self.assertAlmostEqual(result['groups']['clear_moments']['first_update_ratio_median'],13.608685003438325)

    def test_missing_cell_rejected(self):
        data=copy.deepcopy(self.evidence)
        data['cells'].pop()
        with self.assertRaisesRegex(ValueError,'complete and unique'):
            analyze(data)

    def test_changed_input_stream_rejected(self):
        data=copy.deepcopy(self.evidence)
        data['cells'][1]['telemetry'][0]['batch_sha256']='different'
        with self.assertRaisesRegex(ValueError,'random-input fields differ'):
            analyze(data)

    def test_wrong_intervention_rejected(self):
        data=copy.deepcopy(self.evidence)
        cell=next(c for c in data['cells'] if c['summary']['operation']=='clear_moments')
        cell['intervention']['after']['step_values']=[0.]
        with self.assertRaisesRegex(ValueError,'clear_moments changed step'):
            analyze(data)


if __name__=='__main__':
    unittest.main()
