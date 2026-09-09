import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import torch
from training import d_restore as dd, history_component as hc, schedule_switch as sw, m1
from training.ct_training_loop import validate_planned_pause
from analysis.q256_d_restore_vs_hold_v1 import protocol as p, analyze
from tests import test_history_component as fixtures


class DDRestoreTests(unittest.TestCase):
    def fixture(self,d):
        old,source,da=fixtures.HistoryComponentTests().fixture(d,'D')
        binding={'pr108_head':p.PR108_HEAD,**{k:'a'*64 for k in ('prefix_sha256','da_branch_init_sha256','da_terminal_sha256')}}
        manifest=p.manifest(50,Path(d)/'source.pt',Path(d)/'DD',binding)
        state=copy.deepcopy(da); del state['history_component']
        state['schedule_switch']=sw.state_metadata(manifest)
        state['d_restore']=dd.initial_metadata(manifest,0,3999)
        return manifest,source,da,state

    def test_independent_protocol_and_dd_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,source,da,state=self.fixture(d)
            self.assertTrue(dd.validate_against_da_init(state,da,source,manifest))
            self.assertEqual(state['d_restore']['current_arm'],'D')
            self.assertEqual(state['d_restore']['current_denominator_gap_scale'],1.1)
            self.assertFalse(hc.is_manifest(manifest))
            path=Path(d)/'m.json'; path.write_text(json.dumps(manifest))
            self.assertEqual(sw.load_run_manifest(path),manifest)
            self.assertEqual(sw.continuation_factorial(manifest)['arm'],'D')
            for key,val in [('branch','DA'),('continuation_arm','A'),('seed',66)]:
                bad={**manifest,key:val}; path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError): sw.load_run_manifest(path)
            source['ema_512']=source['net']
            with self.assertRaises(RuntimeError): sw.verify_source_state(source,manifest)

    def test_once_only_resume_and_keep(self):
        with tempfile.TemporaryDirectory() as d:
            manifest,source,da,state=self.fixture(d)
            net=copy.deepcopy(source['net']); opt=torch.optim.RAdam(net.parameters(),lr=1e-4)
            for v in net.parameters(): v.grad=torch.ones_like(v)
            opt.step(); before=copy.deepcopy(opt.state_dict()); rng=torch.get_rng_state().clone()
            self.assertEqual(dd.apply_optimizer_intervention(opt,'DD'),0)
            self.assertTrue(m1._equal_state(before,opt.state_dict())); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            dd.save_branch_init_state(state,d)
            with self.assertRaises(FileExistsError): dd.save_branch_init_state(state,d)
            saved=torch.load(Path(d)/'training-state-kimg000512.pt',weights_only=False)
            dd.validate_resumed_state(saved,manifest)
            saved['d_restore']['ema_512_init_count']=2
            with self.assertRaises(RuntimeError): dd.validate_resumed_state(saved,manifest)
            state['optimizer_state']['param_groups'][0]['lr']=.2
            with self.assertRaises(RuntimeError): dd.validate_against_da_init(state,da,source,manifest)

    def test_bound_command_and_short_pause(self):
        cmd=p.command(python='python',dataset='/data.zip',output='/run',seed=58,resume='/source.pt',switch_manifest='/m.json')
        for flag in ('--duration=1.024','--target-gap-scale=1.0','--denominator-gap-scale=1.1','--nproc_per_node=1','--metrics=none','--batch=128','--batch-gpu=16'):
            self.assertIn(flag,cmd)
        self.assertEqual(len(p.training_queue()),16); self.assertEqual(len(p.evaluation_slots()),80)
        self.assertEqual(len(p.import_controls()),80)
        kw=dict(planned_pause_protocol=p.ENGINEERING_ID,strict_reproducibility=True,seed=50,total_kimg=1024,
                resume_state_dump='/s.pt',schedule_switch_manifest='/m.json',schedule_switch_experiment_protocol=p.ENGINEERING_ID)
        self.assertEqual(validate_planned_pause(stop_after_attempts=4016,**kw),4016)
        with self.assertRaises(ValueError): validate_planned_pause(stop_after_attempts=8000,**kw)

    def rows(self):
        old={(r['seed'],r['readout'],r['block']):r for r in p.import_controls()}
        rows=p.evaluation_slots()
        for r in rows:
            da=old[r['seed'],r['readout'],r['block']]
            r.update(status='PASS',FID=da['FID']*math.exp(.01+.0001*(r['seed']-50)),KID=da['KID'])
        return rows

    def test_seed_not_block_is_unit_and_AA_common_subset(self):
        result=analyze.summarize(self.rows())
        self.assertEqual(result['primary']['n'],16)
        self.assertAlmostEqual(result['primary']['mean_log_difference'],-.01075)
        self.assertEqual(result['auxiliary']['n'],14)
        self.assertTrue(result['primary']['practically_equivalent'])
        self.assertEqual(result['primary']['direction_inference'],'A_favored')
        self.assertIn(58,result['complete_pair_seeds']); self.assertIn(65,result['complete_pair_seeds'])

    def test_missing_failure_extreme_and_no_imputation(self):
        rows=self.rows()
        for r in rows:
            if r['seed']==50: r.update(status='NO_ENDPOINT',FID=None,KID=None)
        result=analyze.summarize(rows)
        self.assertEqual(result['primary']['n'],15); self.assertFalse(result['full_cohort_equivalence'])
        self.assertEqual(len(result['outcomes']),16)
        rows[0]['FID']=100
        with self.assertRaises(ValueError): analyze.summarize(rows)
        rows=self.rows(); rows[0]['FID']=1e20
        self.assertEqual(analyze.summarize(rows)['primary']['n'],16)
        rows[0].update(status='TECHNICAL_FAILURE',FID=None,KID=None)
        self.assertEqual(analyze.summarize(rows)['primary'],{})


if __name__=='__main__': unittest.main()

class DDBudgetTests(unittest.TestCase):
    def test_global_partitions_include_engineering_and_no_duplicate_seed(self):
        from analysis.q256_d_restore_vs_hold_v1 import budget
        with tempfile.TemporaryDirectory() as d:
            allocation={'id':'fixed','remaining_quota_gpuh':200,'engineering_cap_gpuh':3,
                        'nodes':[{'node':'one','cap_gpuh':77,'seeds':list(p.SEEDS)}]}
            path=Path(d)/'budget.json';budget.initialize(path,allocation,'one')
            self.assertEqual(budget.reserve(path,'test',3.5,50),3.5)
            with self.assertRaises(RuntimeError):budget.reserve(path,'test',3.5,50)
            allocation['nodes'][0]['cap_gpuh']=78
            with self.assertRaises(RuntimeError):budget.initialize(Path(d)/'other.json',allocation,'one')

    def test_evaluation_identity_and_shared_feature_validator_reused(self):
        from analysis.q256_d_restore_vs_hold_v1 import evaluate
        manifest={'experiment_protocol':p.EXPERIMENT_ID,'seed':58,'branch':'DD'}
        evaluate.validate_export_identity(manifest,58,'DD')
        with self.assertRaises(RuntimeError):evaluate.validate_export_identity(manifest,58,'DA')
        with self.assertRaises(RuntimeError):evaluate.validate_export_identity({**manifest,'experiment_protocol':p.old.EXPERIMENT_ID},58,'DD')
