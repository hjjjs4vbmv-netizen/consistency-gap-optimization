import copy
import unittest

import torch

from training.m1_optimizer_split import intervene, observe_forward


class OptimizerSplitTest(unittest.TestCase):
    def test_observation_preserves_loss_gradient_rng_and_removes_hooks(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=torch.nn.Conv2d(1,1,1,bias=False)
            def forward(self, images, sigma):
                return self.model(images)/sigma
        def objective(_self, net, images, labels=None, augment_pipe=None):
            online=net(images+torch.randn_like(images)*.01,torch.tensor([.8]))
            target=net(images,torch.tensor([.4])).detach()
            return (online-target).square().flatten(1).sum(1).sqrt()/.4
        net=Model()
        images=torch.ones(2,1,2,2)
        torch.manual_seed(42)
        direct=objective(None,net,images)
        direct.sum().backward()
        gradient=net.model.weight.grad.clone()
        rng=torch.get_rng_state()
        net.zero_grad()
        torch.manual_seed(42)
        observed,_,report=observe_forward(objective,None,net,images,None,None)
        observed.sum().backward()
        self.assertTrue(torch.equal(direct,observed))
        self.assertTrue(torch.equal(gradient,net.model.weight.grad))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(report['model_input_dtypes'],['torch.float32']*2)
        self.assertEqual(len(net._forward_hooks),0)
        self.assertEqual(len(net.model._forward_pre_hooks),0)

    def test_four_operations_preserve_parameters_groups_and_untargeted_state(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.RAdam([p], lr=.0001)
        for _ in range(8):
            p.grad = torch.tensor([.2, -.1])
            optimizer.step()
        source = copy.deepcopy(optimizer.state_dict())
        parameters = p.detach().clone()
        for operation in ('K', 'R', 'clear_moments', 'reset_step'):
            optimizer.load_state_dict(copy.deepcopy(source))
            intervene(optimizer, operation)
            actual = optimizer.state_dict()
            self.assertEqual(actual['param_groups'], source['param_groups'])
            self.assertTrue(torch.equal(p, parameters))
            if operation == 'R':
                self.assertEqual(actual['state'], {})
                continue
            for key, value in actual['state'][0].items():
                zeroed = ((operation == 'clear_moments' and key != 'step') or
                          (operation == 'reset_step' and key == 'step'))
                expected = torch.zeros_like(value) if zeroed else source['state'][0][key]
                self.assertTrue(torch.equal(value, expected), (operation, key))


if __name__ == '__main__':
    unittest.main()
