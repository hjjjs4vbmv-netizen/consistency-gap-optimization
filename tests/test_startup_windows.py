import copy
import unittest
from types import SimpleNamespace

import torch

from training import startup_windows as w, reproducibility as repro
from training.loss import resolve_target_weight_factorial


def window(arm):
    start, end, multiplier = w.WINDOWS[arm]
    return dict(start_success_step=start, end_success_step=end, lr_multiplier=multiplier)


class WindowControllerTests(unittest.TestCase):
    def make(self, arm, initial=None, optimizer_state=None, controller_state=None):
        p = torch.nn.Parameter(torch.tensor([.2, -.7], dtype=torch.float64) if initial is None else initial.clone())
        opt = torch.optim.RAdam([p], lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
        if optimizer_state is not None:
            opt.load_state_dict(copy.deepcopy(optimizer_state))
        control = w.Controller(opt, window(arm), controller_state)
        return p, opt, control

    def step(self, p, opt, control, index, skipped=False):
        control.begin()
        before = p.detach().clone()
        p.grad = torch.tensor([.31 + index * .017, -.27 + index * .023], dtype=p.dtype)
        if not skipped:
            opt.step()
        norm = (p.detach() - before).norm().item()
        control.observe_update([p], [before], norm)
        return p.detach() - before

    def test_noop_is_bitwise_native_with_skips(self):
        p, opt, control = self.make('AA')
        vanilla = torch.nn.Parameter(p.detach().clone())
        reference = torch.optim.RAdam([vanilla], lr=1e-4)
        for attempt in range(24):
            skipped = attempt in (0, 3, 8, 14)
            self.step(p, opt, control, attempt, skipped)
            vanilla.grad = p.grad.clone()
            if not skipped:
                reference.step()
            self.assertTrue(torch.equal(p, vanilla))
            self.assertEqual(repro.state_sha256(opt.state_dict()), repro.state_sha256(reference.state_dict()))
        self.assertEqual(control.successful_steps, 20)

    def test_delayed_uses_successes_and_recovers_at_eleven(self):
        p, opt, control = self.make('A_delayed_down6_10')
        for attempt in range(20):
            skipped = attempt in (0, 5, 8)
            prior = control.successful_steps
            self.step(p, opt, control, attempt, skipped)
            self.assertEqual(control.successful_steps, prior + (not skipped))
            expected = 1 / 1.1 if not skipped and 6 <= prior + 1 <= 10 else 1.0
            self.assertEqual(control.multiplier, expected)
            self.assertEqual(opt.param_groups[0]['lr'], 1e-4)
            w.Controller.validate_controller_state(control.state_dict(), window('A_delayed_down6_10'), control.successful_steps)
        self.assertTrue(control.state_dict()['inactive'])

    def test_pause_resume_state_and_exposure_match_all_boundaries(self):
        for arm in ('D_startup_up5', 'A_delayed_down6_10'):
            for boundary in (4, 5, 6, 9, 10, 11):
                with self.subTest(arm=arm, boundary=boundary):
                    p, opt, control = self.make(arm)
                    for i in range(boundary):
                        self.step(p, opt, control, i)
                    r, resumed_opt, resumed_control = self.make(arm, p.detach(), opt.state_dict(), control.state_dict())
                    for i in range(boundary, 18):
                        self.step(p, opt, control, i)
                        self.step(r, resumed_opt, resumed_control, i)
                    self.assertTrue(torch.equal(p, r))
                    self.assertEqual(repro.state_sha256(opt.state_dict()), repro.state_sha256(resumed_opt.state_dict()))
                    self.assertEqual(repro.state_sha256(control.state_dict()), repro.state_sha256(resumed_control.state_dict()))

    def test_same_state_local_scaling_and_unchanged_moments(self):
        p, opt, base = self.make('AA')
        for i in range(5):
            self.step(p, opt, base, i)
        r = torch.nn.Parameter(p.detach().clone())
        alternative = torch.optim.RAdam([r], lr=1e-4)
        alternative.load_state_dict(copy.deepcopy(opt.state_dict()))
        state = base.state_dict()
        state.update(window=window('A_delayed_down6_10'), inactive=False)
        delayed = w.Controller(alternative, window('A_delayed_down6_10'), state)
        delta = self.step(p, opt, base, 5)
        changed = self.step(r, alternative, delayed, 5)
        torch.testing.assert_close(changed, delta / 1.1, rtol=1e-8, atol=1e-16)
        self.assertEqual(repro.state_sha256(opt.state_dict()['state']), repro.state_sha256(alternative.state_dict()['state']))

    def test_finally_restores_lr_on_exception(self):
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.RAdam([p], lr=1e-4)
        def broken():
            self.assertEqual(opt.param_groups[0]['lr'], 1e-4 * (1 / 1.1))
            raise RuntimeError('test exception')
        opt.step = broken
        control = w.Controller(opt, window('A_startup_down5'))
        p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            opt.step()
        self.assertEqual(opt.param_groups[0]['lr'], 1e-4)
        self.assertEqual(control.successful_steps, 0)

    def test_q128_native_identity_and_once_only_boundary(self):
        loss = SimpleNamespace(q=128, factorial=resolve_target_weight_factorial(w.native_protocol(128), 1., 1.1, q=128))
        net = torch.nn.Linear(2, 2)
        opt = torch.optim.RAdam(net.parameters(), lr=1e-4)
        optimizer_before = repro.state_sha256(opt.state_dict())
        rng_before = torch.get_rng_state().clone()
        ema, count = w.transition_to_suffix(loss, net, None, 0, 128, 512000)
        self.assertEqual(loss.q, 128)
        self.assertEqual(loss.factorial['protocol'], w.native_protocol(128))
        self.assertEqual(loss.factorial['arm'], 'A')
        self.assertEqual(repro.module_state_sha256(ema), repro.module_state_sha256(net))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(optimizer_before, repro.state_sha256(opt.state_dict()))
        with torch.no_grad():
            next(net.parameters()).add_(1)
        same, count2 = w.transition_to_suffix(loss, net, ema, count, 128, 512000)
        self.assertIs(same, ema)
        self.assertEqual(count2, 1)
        self.assertNotEqual(repro.module_state_sha256(same), repro.module_state_sha256(net))
        with self.assertRaises(ValueError):
            w.use_native_A(loss, 256)
        with self.assertRaises(ValueError):
            resolve_target_weight_factorial(w.native_protocol(128), 1., 1.1, q=256)

    def test_complete_queue_and_balanced_four_arm_order(self):
        from analysis.startup_window_experiments_v1 import protocol as p
        config = dict(cohort128=list(range(301, 309)), allocation=[
            dict(logical_gpu=i, host='test', local_gpu=i, uuid=f'gpu-{i}') for i in range(8)])
        rows = p.queue(config)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(p.evaluation_slots(rows)), 120)
        for i in range(8):
            lane = [r for r in rows if r['logical_gpu'] == i]
            self.assertEqual(lane[0]['arm'], 'A_delayed_down6_10')
            self.assertEqual(lane[0]['seed'], 50 + i)
            self.assertEqual([r['arm'] for r in lane[1:]], list(w.ARMS128[i % 4:] + w.ARMS128[:i % 4]))
            self.assertEqual({r['seed'] for r in lane[1:]}, {301 + i})

    def test_historical_comparators_are_exactly_AA_and_early(self):
        from analysis.startup_window_experiments_v1 import protocol as p
        rows = p.old_controls()
        self.assertEqual(len(rows), 48)
        self.assertEqual({r['arm'] for r in rows}, {'AA', 'A_startup_down5'})
        self.assertEqual({(r['seed'], r['arm'], r['block']) for r in rows},
            {(s, a, b) for s in range(50, 58) for a in ('AA', 'A_startup_down5') for b in ('B0', 'B1', 'B2')})


if __name__ == '__main__':
    unittest.main()
