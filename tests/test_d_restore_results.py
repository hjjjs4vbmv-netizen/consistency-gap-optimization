"""Independent checks of the published fixed cohort, inference and archive retention."""
import json
import math
import statistics
import unittest
from pathlib import Path
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]/'analysis/q256_d_restore_vs_hold_v1'
RESULTS=ROOT/'results'


class PublishedDDResultsTests(unittest.TestCase):
    def test_complete_fixed_matrix_and_unchanged_DA(self):
        rows=json.loads((RESULTS/'DA_DD_160.json').read_text())
        self.assertEqual(len(rows),160)
        self.assertEqual([r for r in rows if r['path']=='DA'],json.loads((ROOT/'frozen/old_DA_80.json').read_text()))
        new=[r for r in rows if r['path']=='DD']
        self.assertEqual(len(new),80);self.assertTrue(all(r['status']=='PASS' for r in new))
        self.assertEqual({r['seed'] for r in new},set(range(50,66)))
        self.assertEqual(len({(r['seed'],r['readout'],r['block']) for r in new}),80)
        training=json.loads((RESULTS/'training_summary.json').read_text())
        self.assertEqual(sum(r['new_attempts'] for r in training),64000)
        self.assertEqual(sum(r['new_successful_updates'] for r in training),63970)
        self.assertEqual(sum(r['new_amp_skips'] for r in training),30)
        self.assertTrue(all(r['end_attempt']==8000 and r['new_successful_updates']+r['new_amp_skips']==4000 for r in training))

    def test_independent_paired_formula_and_prespecified_TOST(self):
        rows=json.loads((RESULTS/'DA_DD_160.json').read_text());s=json.loads((RESULTS/'statistics.json').read_text())
        data={(r['seed'],r['path'],r['block']):r['FID'] for r in rows if r['readout']=='E_512'}
        values=[sum(math.log(data[seed,'DA',b]/data[seed,'DD',b]) for b in ('B0','B1','B2'))/3 for seed in range(50,66)]
        mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/4
        self.assertEqual(s['primary']['n'],16)
        self.assertAlmostEqual(mean,s['primary']['mean_log_difference'],places=12)
        self.assertAlmostEqual(sd,s['primary']['seed_sd'],places=12)
        for probability,key in ((.975,'ci95'),(.95,'ci90')):
            half=t.ppf(probability,15)*se
            for actual,expected in zip(s['primary'][key],(mean-half,mean+half)):self.assertAlmostEqual(actual,expected,places=12)
        margin=math.log(1.03)
        equivalent=mean-t.ppf(.95,15)*se>-margin and mean+t.ppf(.95,15)*se<margin
        self.assertFalse(equivalent);self.assertFalse(s['primary']['practically_equivalent'])
        self.assertEqual(s['auxiliary']['common_seeds'],[i for i in range(50,66) if i not in (58,65)])
        self.assertIn(58,s['complete_pair_seeds']);self.assertIn(65,s['complete_pair_seeds'])

    def test_complete_archive_and_process_cap(self):
        a=json.loads((RESULTS/'archive_index.json').read_text());c=json.loads((RESULTS/'cost_summary.json').read_text())
        self.assertEqual(a['status'],'COMPLETE_ARCHIVED')
        self.assertEqual((a['required_full_states'],a['generated_sample_arrays'],a['FID_KID_feature_arrays']),(80,80,160))
        self.assertEqual(len(a['files']),len({f['path'] for f in a['files']}))
        self.assertTrue(all(not f['path'].startswith('/') and len(f['sha256'])==64 for f in a['files']))
        self.assertTrue(all(r['status']=='COMPLETE_VERIFIED' and not r['sha256_differences'] for r in a['node_copy_receipts']))
        self.assertAlmostEqual(c['total_gpuh'],sum(c[k] for k in ('training_gpuh','evaluation_gpuh','export_gpuh','engineering_gpuh')),places=12)
        self.assertLessEqual(c['total_gpuh'],80)


if __name__=='__main__':unittest.main()
