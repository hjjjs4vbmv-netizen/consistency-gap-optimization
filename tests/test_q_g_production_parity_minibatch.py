from __future__ import annotations

import torch

from analysis.q_g_production_parity_audit import minibatch_parity
from training.loss import ECMLoss


class TinyConsistencyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.time_scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        x,
        sigma,
        labels=None,
        augment_labels=None,
        force_fp32=False,
    ):
        del labels, augment_labels, force_fp32
        return self.scale * x + self.time_scale * sigma


def test_minibatch_parity_compares_loss_and_one_sided_gradient():
    net = TinyConsistencyNet().train()
    loss = ECMLoss(
        P_mean=-1.1,
        P_std=2.0,
        sigma_data=0.5,
        q=256,
        c=0.0,
        k=8.0,
        b=1.0,
        adj="sigmoid",
    )
    loss.update_schedule(0)
    images = torch.arange(4 * 3 * 4 * 4, dtype=torch.uint8).reshape(4, 3, 4, 4)
    labels = torch.zeros(4, 0)
    result = minibatch_parity(
        net,
        loss,
        iter([(images, labels)]),
        batches=1,
        device=torch.device("cpu"),
        audit_seed=20260825,
        force_fp32=True,
        field_tolerance=1e-6,
        coordinate_eps_multiplier=32.0,
    )
    assert result["coordinate_gate_passed"]
    observed_max = max(
        result["max_weight_relative_l2"],
        result["max_target_output_relative_l2"],
        result["max_loss_relative_l2"],
        result["max_gradient_relative_l2"],
    )
    assert result["field_gate_passed"] == (observed_max <= 1e-6)
    assert result["max_gradient_relative_l2"] < 1e-4
