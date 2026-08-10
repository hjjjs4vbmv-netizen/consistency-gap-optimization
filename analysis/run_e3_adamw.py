"""E3 — universality: does the non-scalar residual reproduce with AdamW?

The gradient residual R_grad is a property of the loss/gradient (optimizer-
independent), so it should reproduce with any optimizer. The update distortion
R_opt depends on the optimizer's update rule. This script runs a paired
g=1.0 vs g=1.3 step with AdamW on an arm state and reports R_grad, R_opt, and
the gauge, to test whether the residual is RAdam-specific.

Usage (on server):
  .venv/bin/python analysis/run_e3_adamw.py --tick 8
"""
from __future__ import annotations
import argparse, copy, pickle, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from analysis.radam_stateful_update_audit import (
    load_training_state, _grad_by_name, _update_scale_and_residual,
    support_aware_gauge_summary,
)
from analysis.radam_update_gauge import fixed_ect_loss
from training.schedules import get_schedule
from training.dataset import ImageFolderDataset
from torch.utils.data import DataLoader

BASE = Path("/data/raw/ECT/ect_runs/gap_lr_matched_q128_s3_v1")
ARM = "arm_a_g1_0_lr_fixed_s3"
DATA = "/data/raw/ECT/datasets/cifar10-32x32.zip"

def load_loss_from_ckpt(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)["loss_fn"]

def paired_adamw_step(net, loss, images, labels, gain, seed):
    """One paired step with AdamW: returns (grads, actual_update, moments)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net2 = copy.deepcopy(net).train().requires_grad_(True)
    opt = torch.optim.AdamW(net2.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    sched = get_schedule("global_sigmoid", q=float(loss.q), k=float(loss.k),
                         b=float(loss.b), global_gap_scale=gain)
    t = (torch.randn(images.shape[0], 1, 1, 1, device=images.device)
         * loss.P_std + loss.P_mean).exp()
    eps = torch.randn_like(images)
    dropout = torch.cuda.get_rng_state(device=images.device) if images.device.type == "cuda" else torch.get_rng_state()
    opt.zero_grad(set_to_none=True)
    l = fixed_ect_loss(net2, loss, sched, images, labels, t, eps, dropout)
    l.mean().backward()
    grads = _grad_by_name(net2)
    before = {n: p.detach().double().cpu().clone() for n, p in net2.named_parameters()}
    opt.step()
    actual = {n: p.detach().double().cpu() - before[n] for n, p in net2.named_parameters()}
    moments = {n: (opt.state[p]["exp_avg"].detach().double().cpu().clone(),
                   opt.state[p]["exp_avg_sq"].detach().double().cpu().clone())
               for n, p in net2.named_parameters() if p in opt.state}
    return grads, actual, moments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)
    ts = BASE / ARM / f"training-state-{a.tick:06d}.pt"
    ckpt = BASE / ARM / "network-snapshot-latest.pkl"
    net, _, _, _, _ = load_training_state(ts, device, lr=1e-4, betas=(0.9, 0.999), eps_opt=1e-8)
    loss = load_loss_from_ckpt(ckpt)

    torch.manual_seed(20260808)
    ds = ImageFolderDataset(path=DATA, use_labels=False, xflip=False, cache=True,
                            resolution=net.img_resolution)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(20260808))
    images, labels = next(iter(loader))
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)

    g1, u1, m1 = paired_adamw_step(net, loss, images, labels, 1.0, 20260808)
    g13, u13, m13 = paired_adamw_step(net, loss, images, labels, 1.3, 20260808)

    a_star, c_grad, r_grad, gc, _ = _update_scale_and_residual(g1, g13)
    s_star, c_star, r_opt, oc, _ = _update_scale_and_residual(u1, u13)
    print(f"AdamW tick={a.tick}: a_K*={a_star:.4f} R_grad={r_grad:.4f} "
          f"s_K*={s_star:.4f} R_opt={r_opt:.4f} R_opt-R_grad={r_opt-r_grad:.4f}")

if __name__ == "__main__":
    main()
