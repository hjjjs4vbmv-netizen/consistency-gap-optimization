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
