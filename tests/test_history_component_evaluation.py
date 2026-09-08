import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from analysis.q256_history_component_chase_v1 import evaluate,protocol as p


class EvaluationTests(unittest.TestCase):
    def test_export_rejects_wrong_seed_path_or_engineering_population(self):
        good={'experiment_protocol':p.EXPERIMENT_ID,'seed':50,'branch':'CA'}
        evaluate.validate_export_identity(good,50,'CA')
        for changes in ({'seed':51},{'branch':'DA'},{'experiment_protocol':p.ENGINEERING_ID}):
            with self.assertRaisesRegex(RuntimeError,'formal training identity'):
                evaluate.validate_export_identity({**good,**changes},50,'CA')

    def config(self,d):
        root=Path(d)
        cfg={'budget_ledger':str(root/'budget.json'),'preflight_receipt':str(root/'preflight.json'),
             'evaluation_output':str(root/'evaluation')}
        stages={}
        for row in p.training_queue():
            if row['seed']!=50:continue
            for phase in ('prefix','suffix'):
                stages[f"{row['slot']}/{row['path']}/{phase}"]={'status':'PASS','attempts':[{'process_gpuh':2.}]}
        (root/'budget.json').write_text(json.dumps({'cap_gpuh':12.5,'scoped_seeds':[50],'status':'RUNNING','stages':stages}))
        (root/'preflight.json').write_text('{"short_checks":{"process_gpuh":0.1}}')
        return cfg

    def test_budget_scope_waits_for_all_node_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d); path=Path(cfg['budget_ledger']);budget=json.loads(path.read_text())
            budget['stages']['S01/CA/suffix']['status']='RUNNING';path.write_text(json.dumps(budget))
            with self.assertRaisesRegex(RuntimeError,'incomplete training'):
                evaluate.admit_evaluation(cfg,p.evaluation_slots()[0])

    def test_cost_ledger_contains_no_quality(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d);slot=p.evaluation_slots()[0]
            self.assertTrue(evaluate.admit_evaluation(cfg,slot))
            evaluate.account_evaluation(cfg,{**slot,'status':'PASS','process_gpuh':.1,'FID':500.,'KID':.4})
            raw=Path(cfg['budget_ledger']).read_text()
            self.assertNotIn('FID',raw);self.assertNotIn('KID',raw)
            self.assertEqual(json.loads(raw)['evaluation_jobs'][slot['job_id']]['attempts'],[{'status':'PASS','process_gpuh':.1}])

    def test_dedicated_card_accepts_a_ready_path_without_waiting_for_other_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d);cfg['evaluation_only']=True
            ledger=Path(cfg['budget_ledger'])
            ledger.write_text(json.dumps({'cap_gpuh':1.,'scoped_seeds':[50],'status':'RUNNING',
                'stages':{},'evaluation_only':True,'allocation_id':'test-global-allocation'}))
            self.assertTrue(evaluate.admit_evaluation(cfg,p.evaluation_slots()[0]))

    def test_job_partition_counts_only_owned_slots_and_rejects_other_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d);cfg['evaluation_only']=True
            slots=p.evaluation_slots();ledger=Path(cfg['budget_ledger'])
            ledger.write_text(json.dumps({'cap_gpuh':.06,'scoped_seeds':[50],'status':'RUNNING',
                'scoped_job_ids':[slots[0]['job_id']], 'stages':{},'evaluation_only':True,
                'allocation_id':'test-disjoint-global-jobs'}))
            self.assertTrue(evaluate.admit_evaluation(cfg,slots[0]))
            self.assertAlmostEqual(json.loads(ledger.read_text())['evaluation_forecast_gpuh'],.05)
            with self.assertRaisesRegex(RuntimeError,'outside allocated job scope'):
                evaluate.admit_evaluation(cfg,slots[1])

    def test_dedicated_card_requires_explicit_separate_budget_binding(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d);ledger=Path(cfg['budget_ledger'])
            ledger.write_text(json.dumps({'cap_gpuh':1.,'scoped_seeds':[50],'status':'RUNNING',
                'stages':{},'evaluation_only':True,'allocation_id':'test-global-allocation'}))
            with self.assertRaisesRegex(RuntimeError,'separate allocated ledger'):
                evaluate.admit_evaluation(cfg,p.evaluation_slots()[0])

    def test_successful_job_cannot_be_resampled(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.config(d);slot=p.evaluation_slots()[0]
            result={**slot,'status':'PASS','FID':500.,'KID':.4}
            receipt=Path(cfg['evaluation_output'])/'receipts'/f"{slot['job_id']}.json"
            receipt.parent.mkdir(parents=True);receipt.write_text(json.dumps(result))
            with patch.object(evaluate.subprocess,'Popen',side_effect=AssertionError('resampled')):
                self.assertEqual(evaluate.run_job(cfg,slot,None,0),result)

    def test_all_generation_blocks_keep_frozen_evaluator_options(self):
        for row in p.evaluation_slots()[:5]:
            wire={**{k:str(v) for k,v in row.items()},'slot_id':row['job_id'],
                  'metrics':'kid50k_full,fid50k_full','nfe':'1','precision':'fp32'}
            cmd=evaluate.frozen_command.build_command(wire,'/snapshot.pkl',Path('/dataset.zip'),Path('/output'),Path('/evaluator'),Path('/python'),46000)
            self.assertIn(f"--sample-seeds={row['sample_seed_start']}-{row['sample_seed_end']}",cmd)
            self.assertIn('--fp16=False',cmd);self.assertIn('--nfe=1',cmd)
            self.assertIn('--metric-generator-batch=128',cmd)
            self.assertIn('--retain-generated-artifacts',cmd)


if __name__=='__main__':unittest.main()
