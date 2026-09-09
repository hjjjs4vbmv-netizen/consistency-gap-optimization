import copy,json,tempfile,unittest
from pathlib import Path
import torch
from training import startup_update as s
from training.ct_training_loop import validate_planned_pause

class StartupTests(unittest.TestCase):
    def manifest(self):
        return dict(protocol=s.PROTOCOL,arm='D_compensate',seed=50,attempts=64,
                    initialization='fresh_transfer',engineering_only=True)
    def optimizer(self):
        p=torch.nn.Parameter(torch.tensor([1.,-2.,3.],dtype=torch.float64))
        return p,torch.optim.RAdam([p],lr=1e-4,betas=(.9,.999),eps=1e-8)
    def test_native_formula_and_moments(self):
        for arm,factor in [('D_compensate',1.1),('A_mimic',1/1.1),('A',1.),('D',1.)]:
            p,o=self.optimizer();q,ref=self.optimizer()
            for step in range(1,9):
                grad=torch.tensor([step/7.,-.4,.1],dtype=p.dtype)
                p.grad=grad.clone();q.grad=grad.clone()
                before=copy.deepcopy(o.state_dict());grad_before=p.grad.clone()
                with s.temporary_lr(o,arm) as plan:
                    self.assertEqual(plan[0]['branch'],'non_adaptive' if step<=5 else 'rectified')
                    self.assertEqual(plan[0]['multiplier'],factor if step<=5 else 1.)
                    # Entering the intervention has no direct gradient or state writes.
                    self.assertTrue(torch.equal(p.grad,grad_before))
                    for key,value in before['state'].get(0,{}).items():
                        self.assertTrue(torch.equal(value,o.state[p][key]))
                    o.step()
                ref.param_groups[0]['lr']=1e-4*(factor if step<=5 else 1.)
                ref.step()
                self.assertEqual(o.param_groups[0]['lr'],1e-4)
                self.assertTrue(torch.equal(p,q))
                for key in ('exp_avg','exp_avg_sq','step'):
                    self.assertTrue(torch.equal(o.state[p][key],ref.state[q][key]))
                self.assertTrue(torch.equal(p.grad,grad_before))
    def test_exception_restores_lr_inside_native_step(self):
        p,o=self.optimizer();p.grad=torch.ones_like(p)
        def fail(*args):raise RuntimeError('injected native step pre-hook failure')
        hook=o.register_step_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with s.temporary_lr(o,'D_compensate'):o.step()
        hook.remove()
        self.assertEqual(o.param_groups[0]['lr'],1e-4)
        self.assertEqual(len(o.state),0)
    def test_real_cpu_amp_skip_and_observer_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);net=torch.nn.Linear(3,1,bias=False,dtype=torch.float32)
            o=torch.optim.RAdam(net.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
            receipt=dict(seed=50,attempted_iteration=0,hashes={},trajectory_config=dict(total_kimg=1024,loss_kwargs={}))
            (root/'initial_state_receipt_v1.json').write_text(json.dumps(receipt))
            m={**self.manifest(),'reference_receipt':str(root/'initial_state_receipt_v1.json')}
            observer=s.Observer(m,net,o,root)
            scaler=torch.amp.GradScaler('cpu',init_scale=8.,growth_interval=2000)
            success=0
            for attempt in range(9):
                o.zero_grad(set_to_none=True);rng=torch.get_rng_state().clone();observer.begin(attempt,success)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                scale=scaler.get_scale()
                loss=net(torch.ones(1,3)).sum()
                if attempt in (0,3,6):loss=loss*float('inf')
                scaler.scale(loss).backward();scaler.unscale_(o)
                nf=sum(int((~torch.isfinite(p.grad)).sum()) for p in net.parameters())
                observer.before_step(nf)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                before=[p.detach().clone() for p in net.parameters()]
                scaler.step(o);scaler.update()
                skipped=scaler.get_scale()<scale
                observer.after_step(before,scale,scaler.get_scale(),skipped)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                if not skipped:
                    success+=1
                    self.assertEqual(observer.event['actual_step_plan'][0]['multiplier'],1.1 if success<=5 else 1.)
                self.assertEqual(s.clocks(o),[success])
            self.assertEqual(success,6)
            self.assertEqual(observer.calls,6)
            self.assertEqual(o.param_groups[0]['lr'],1e-4)
            self.assertEqual(len(list(root.glob('update-success-*.pt'))),3)
    def test_inconsistent_active_clocks_rejected(self):
        p,o=self.optimizer();q=torch.nn.Parameter(p.detach().clone());o.add_param_group({'params':[q]})
        p.grad=torch.ones_like(p);o.step();q.grad=torch.ones_like(q)
        with self.assertRaisesRegex(RuntimeError,'clocks disagree'):s.step_plan(o,'D_compensate')
        self.assertEqual(s.clocks(o),[1,0])
    def test_strict_pause_and_old_guards(self):
        good=dict(stop_after_attempts=64,planned_pause_protocol=s.PROTOCOL,strict_reproducibility=True,
                  seed=50,total_kimg=1024,resume_state_dump=None,schedule_switch_manifest=None,startup_check=self.manifest())
        self.assertEqual(validate_planned_pause(**good),64)
        for changes in [dict(seed=52),dict(stop_after_attempts=63),dict(stop_after_attempts=65),
                        dict(resume_state_dump='D@512.pt'),dict(total_kimg=8.192),dict(startup_check=None),
                        dict(planned_pause_protocol=None),dict(stop_after_attempts=None),dict(strict_reproducibility=False)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_planned_pause(**{**good,**changes})
        legacy={**good,'planned_pause_protocol':None,'startup_check':None,'stop_after_attempts':16}
        self.assertEqual(validate_planned_pause(**legacy),16)
        with self.assertRaises(ValueError):validate_planned_pause(**{**legacy,'stop_after_attempts':64})
        formal={**legacy,'planned_pause_protocol':'q256_terminal_history_n30_matpool_v1','stop_after_attempts':4000}
        self.assertEqual(validate_planned_pause(**formal),4000)
        with self.assertRaises(ValueError):validate_planned_pause(**{**formal,'stop_after_attempts':64})
if __name__=='__main__':unittest.main(verbosity=2)
