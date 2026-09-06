import math
from pathlib import Path
import tempfile
import unittest

from analysis.q256_terminal_history_512_source_backfill_v1.common import (
    check_hash, join_future, quadrants, sha256, source_pairs)

def metric(seed,arm,fid,status='PASS'):
    return dict(seed=seed,arm=arm,fid50k=fid,kid50k=.01,evaluation_status=status)

def endpoint(seed,aa,ba):
    return dict(seed=seed,aa_fid50k=aa,ba_fid50k=ba,log_fid_contrast_ba_minus_aa=math.log(ba)-math.log(aa))

class SourceBackfillContracts(unittest.TestCase):
    def test_source_inclusion_independent_of_future_endpoint(self):
        source=source_pairs([metric(50,'A',10),metric(50,'B',12),metric(68,'A',8),metric(68,'B',9)])
        self.assertEqual([r['seed'] for r in source],[50,68])
        joint=join_future(source,[endpoint(50,6,5)])
        self.assertEqual([r['seed'] for r in joint],[50])
        self.assertTrue(joint[0]['delayed_reversal'])
        self.assertEqual(len(source),2)

    def test_partial_or_failed_source_is_not_a_pair(self):
        source=source_pairs([metric(67,'B',4),metric(70,'A',5),metric(70,'B','',status='FAILED')])
        self.assertEqual(source,[])

    def test_four_quadrants_and_exact_axis_ties(self):
        values=[(1,-1),(1,1),(-1,-1),(-1,1),(0,-1),(1,0)]
        counts=quadrants([dict(Q=q,H_A=h) for q,h in values])
        self.assertEqual(counts,dict(reversal=1,bad_to_bad=1,good_to_good=1,reverse_loss=1,on_axis=2))

    def test_no_epsilon_relabeling(self):
        counts=quadrants([dict(Q=1e-15,H_A=-1e-15)])
        self.assertEqual(counts['reversal'],1)

    def test_nonfinite_pass_metric_rejected(self):
        for bad in (0,-1,float('inf'),float('nan')):
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):
                source_pairs([metric(50,'A',5),metric(50,'B',bad)])

    def test_frozen_endpoint_contrast_is_checked(self):
        source=source_pairs([metric(50,'A',10),metric(50,'B',12)])
        corrupted=endpoint(50,6,5); corrupted['log_fid_contrast_ba_minus_aa']=0
        with self.assertRaises(RuntimeError): join_future(source,[corrupted])

    def test_duplicate_seed_arm_rejected(self):
        with self.assertRaises(RuntimeError): source_pairs([metric(50,'A',10),metric(50,'A',12)])

    def test_duplicate_frozen_endpoint_rejected(self):
        with self.assertRaises(RuntimeError): join_future([],[endpoint(50,6,5),endpoint(50,6,5)])

    def test_mutated_artifact_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact'; p.write_bytes(b'original'); h=sha256(p)
            self.assertEqual(check_hash(p,h),h)
            p.write_bytes(b'mutated')
            with self.assertRaises(RuntimeError): check_hash(p,h)

if __name__=='__main__': unittest.main()
