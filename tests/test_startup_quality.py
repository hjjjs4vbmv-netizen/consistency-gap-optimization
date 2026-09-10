import copy
import unittest
from unittest.mock import patch
import torch
from training.startup_quality import Controller, ARMS, use_native_A
from training.startup_update import clocks

class StartupQuality(unittest.TestCase):
    def make(self,arm):
        p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
        o=torch.optim.RAdam([p],lr=1e-4)
        return p,o,Controller(o,arm)
    def test_exact_native_intervention_and_boundaries(self):
        for arm,m in [(ARMS[0],1.1),(ARMS[1],1/1.1)]:
            p,o,c=self.make(arm);q=torch.nn.Parameter(p.detach().clone());ref=torch.optim.RAdam([q],lr=1e-4)
            for step in range(1,10):
                c.begin();p.grad=torch.tensor([.3,-.7],dtype=p.dtype);q.grad=p.grad.clone()
                o.step();ref.param_groups[0]['lr']=1e-4*(m if step<=5 else 1);ref.step()
                self.assertTrue(torch.equal(p,q));self.assertEqual(o.param_groups[0]['lr'],1e-4)
                self.assertEqual(c.inactive,step>=6);self.assertEqual(c.successful_steps,step)
    def test_actual_gradscaler_skip(self):
        p,o,c=self.make(ARMS[0]);scaler=torch.amp.GradScaler('cpu',init_scale=8.)
        for attempt in range(9):
            c.begin();o.zero_grad(set_to_none=True)
            value=p.sum() * (float('inf') if attempt in (0,3,6) else 1.)
            scaler.scale(value).backward();scaler.unscale_(o)
            before=c.successful_steps;scale=scaler.get_scale();scaler.step(o);scaler.update()
            skip=scaler.get_scale()<scale
            self.assertEqual(c.successful_steps,before+int(not skip))
            self.assertEqual(c.called,not skip)
        self.assertEqual(c.successful_steps,6)
    def test_skip_does_not_advance(self):
        p,o,c=self.make(ARMS[0]);c.begin()
        self.assertEqual(c.state_dict(),{'successful_steps':0,'inactive':False});self.assertFalse(o.state)
    def test_exception_restores_lr(self):
        p=torch.nn.Parameter(torch.ones(2));o=torch.optim.RAdam([p],lr=1e-4)
        def fail():raise RuntimeError('injected optimizer exception')
        o.step=fail;c=Controller(o,ARMS[0]);p.grad=torch.ones_like(p)
        with self.assertRaisesRegex(RuntimeError,'injected'):o.step()
        self.assertEqual(o.param_groups[0]['lr'],1e-4);self.assertEqual(c.successful_steps,0)
    def test_disagreeing_clocks_fail_closed(self):
        ps=[torch.nn.Parameter(torch.ones(1)) for _ in range(2)];o=torch.optim.RAdam(ps,lr=1e-4);c=Controller(o,ARMS[0])
        for p in ps:p.grad=torch.ones_like(p)
        o.step();o.state[ps[1]]['step']+=1
        with self.assertRaisesRegex(RuntimeError,'clocks'):o.step()
        self.assertEqual(o.param_groups[0]['lr'],1e-4)
    def test_inactive_does_not_scan(self):
        p,o,c=self.make(ARMS[0]);p.grad=torch.ones_like(p)
        for _ in range(6):c.begin();o.step()
        with patch('training.startup_update.clocks',side_effect=AssertionError('expensive scan')):
            c.begin();o.step()
        self.assertEqual(c.successful_steps,7)
    def test_pause_resume_steps_3_5_6(self):
        for boundary in (3,5,6):
            for arm in ARMS:
                p,o,c=self.make(arm)
                for _ in range(boundary):p.grad=torch.ones_like(p);c.begin();o.step()
                q=torch.nn.Parameter(p.detach().clone());qo=torch.optim.RAdam([q],lr=1e-4)
                qo.load_state_dict(copy.deepcopy(o.state_dict()));qc=Controller(qo,arm,c.state_dict())
                for _ in range(8-boundary):
                    p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);c.begin();qc.begin();o.step();qo.step()
                self.assertTrue(torch.equal(p,q));self.assertEqual(c.state_dict(),qc.state_dict())
    def test_native_A_configuration(self):
        x=type('Loss',(),{})();use_native_A(x)
        self.assertEqual(x.factorial['arm'],'A');self.assertEqual(x.factorial['denominator_gap_scale'],1)

if __name__=='__main__':unittest.main()

class QualityGates(unittest.TestCase):
    def test_evaluation_requires_full_terminal_matrix(self):
        import json,tempfile
        from pathlib import Path
        from analysis.q256_startup_step_responder_pilot_v1.evaluate import training_gate
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'matrix.json'
            config={'training_matrix_frozen':str(path)}
            with self.assertRaises(FileNotFoundError):training_gate(config)
            path.write_text(json.dumps({'outcomes':[]}))
            with self.assertRaisesRegex(RuntimeError,'complete'):training_gate(config)
    def test_global_budget_includes_remote_and_live_cost(self):
        from analysis.q256_startup_step_responder_pilot_v1.worker import projection
        import time
        ledger=dict(allocation_cap_gpuh=67.5,engineering_used_gpuh=.5,engineering_remaining_gpuh=1.375,evaluation_remaining_gpuh=6.375,
                    jobs={'a':dict(status='RUNNING',started_wall=time.time()-3600,estimate_gpuh=2),
                          'b':dict(status='PENDING',estimate_gpuh=3),'c':dict(status='PASS',actual_gpuh=2)})
        x=projection(ledger)
        self.assertAlmostEqual(x['global_conservative_projected_gpuh'],37.75,places=5)
        self.assertEqual(x['other_host_reserved_envelope_gpuh'],22.5)
    def test_exact_sign_flip_and_equivalence(self):
        from analysis.q256_startup_step_responder_pilot_v1.analyze import summarize
        x=summarize([.01]*8)
        self.assertEqual(x['exact_sign_flip_p'],2/256);self.assertTrue(x['TOST']['equivalent'])
        self.assertEqual(x['n'],8)
        self.assertFalse(summarize([.1]*8)['TOST']['equivalent'])
    def test_planned_queue_and_blocks(self):
        from analysis.q256_startup_step_responder_pilot_v1 import protocol as p
        self.assertEqual(len(p.queue()),16);self.assertEqual(len(p.slots()),48)
        for s in p.SEEDS:
            jobs=[x for x in p.queue() if x['seed']==s]
            self.assertEqual(jobs[0]['arm'],p.ARMS[0 if s%2==0 else 1])
            self.assertEqual(jobs[0]['logical_gpu'],s-50)

class IdentityGates(unittest.TestCase):
    def manifest(self):
        from training import startup_quality as q,startup_update as e
        return dict(q.FIXED,seed=50,arm=q.ARMS[0],native_prefix_arm='D',lr_multiplier=1.1,
                    dataset_sha256=e.DATA_SHA256,transfer_sha256=e.TRANSFER_SHA256,
                    reference_initial_receipt={'path':'reference','sha256':'a'*64},immutable_output_root='/test/seed50/D_startup_up5',
                    old_control_bindings=[dict(seed=50,arm=a,block=b) for a in ('AA','DA') for b in ('B0','B1','B2')])
    def test_engineering_and_cross_identity_rejected(self):
        from training import startup_quality as q
        m=self.manifest()
        with self.assertRaisesRegex(ValueError,'engineering'):q.validate_state({'startup_engineering':{}},m)
        changed={**m,'seed':51}
        with self.assertRaisesRegex(ValueError,'identity'):q.validate_state({'startup_quality':{'manifest':changed}},m)
        changed={**m,'preflight_only':True}
        with self.assertRaisesRegex(ValueError,'identity'):q.validate_state({'startup_quality':{'manifest':changed}},m)
    def test_controls_cannot_substitute_seed_or_block(self):
        from training import startup_quality as q
        m=self.manifest();m['old_control_bindings'][0]['seed']=51
        with self.assertRaisesRegex(ValueError,'old controls'):q.validate_manifest(m)

class OwnedGpuBudget(unittest.TestCase):
    def test_owned_ect_has_no_paid_budget_gate(self):
        from analysis.q256_startup_step_responder_pilot_v1.worker import projection
        ledger=dict(allocation_cap_gpuh=90,budget_exempt=True,cap_scope='owned_ect_unmetered',engineering_used_gpuh=0,engineering_remaining_gpuh=0,evaluation_remaining_gpuh=0,
                    jobs={'a':dict(status='PASS',actual_gpuh=500)})
        x=projection(ledger)
        self.assertEqual(x['global_conservative_projected_gpuh'],0)
        self.assertEqual(x['owned_unmetered_projected_gpuh'],500)
    def test_paid_matpool_does_not_reserve_owned_ect(self):
        from analysis.q256_startup_step_responder_pilot_v1.worker import projection
        ledger=dict(allocation_cap_gpuh=90,budget_exempt=False,cap_scope='paid_matpool_only',engineering_used_gpuh=1,engineering_remaining_gpuh=1.5,evaluation_remaining_gpuh=8.5,
                    jobs={'a':dict(status='PENDING',estimate_gpuh=60)})
        x=projection(ledger)
        self.assertEqual(x['global_conservative_projected_gpuh'],71)
        self.assertEqual(x['other_host_reserved_envelope_gpuh'],0)
