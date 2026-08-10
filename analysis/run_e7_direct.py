"""E7 — optimizer-state dependence, direct method (bypasses the audit's sigmoid gate).

The stateful audit's main() requires a 'sigmoid' checkpoint, which blocks the
global_sigmoid arm states. Here we call run_stateful_pair DIRECTLY on each arm_a
training-state tick (different n_K / optimizer maturity), using the arm's own
global_sigmoid checkpoint for loss_fn. This measures how the update distortion
(R_opt, R_grad, h_actual, a_K*) evolves with optimizer state.

Usage (on server):
  .venv/bin/python analysis/run_e7_direct.py --tick 6
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from analysis.radam_stateful_update_audit import (
    load_training_state, run_stateful_pair,
)
from training.dataset import ImageFolderDataset
from torch.utils.data import DataLoader

DATA = "/data/raw/ECT/datasets/cifar10-32x32.zip"

def load_loss_from_ckpt(path: Path):
    """Load loss_fn directly, bypassing the audit's 'sigmoid' gate."""
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["loss_fn"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-state", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device)
    ts = a.training_state
    ckpt = a.checkpoint
    print(f"training-state={ts}")

    net, optimizer, scaler_state, loss_fn_state, meta = load_training_state(
        ts, device, lr=1e-4, betas=(0.9, 0.999), eps_opt=1e-8)
    loss = load_loss_from_ckpt(ckpt)
    print(f"  loss schedule={loss.schedule.name} q={loss.q} k={loss.k} b={loss.b}")

    torch.manual_seed(20260808)
    ds = ImageFolderDataset(path=DATA, use_labels=False, xflip=False, cache=True,
                            resolution=net.img_resolution)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(20260808))
    images, labels = next(iter(loader))
    images = images.to(device).to(torch.float32) / 127.5 - 1
    labels = labels.to(device)

    audit, layers = run_stateful_pair(
        net, optimizer, loss, images, labels,
        gains=(1.0, 1.3), amp=True, scaler_state=scaler_state,
        random_seed=20260808, microbatch_size=None, support_atol=0.0,
    )
    w = audit["whole_model"]
    print(f"  n_K={audit['stateful_radam']['n_K']}")
    print(f"  a_K*={w['a_K_star']:.4f}  R_grad={w['R_grad']:.4f}  "
          f"s_K*={w['s_K_star']:.4f}  R_opt={w['R_opt']:.4f}  "
          f"R_opt-R_grad={w['R_opt_minus_R_grad']:.4f}  H_K={w['H_K']:.4f}")
    print(f"  h_update_mean={w.get('h_update_weighted_mean')}  "
          f"h_update_std={w.get('h_update_weighted_std')}")

if __name__ == "__main__":
    main()
