# Training Protocol — Confirmatory Fixed vs Global (g=1.10) Study

**Role C — Training & Scaling Lead.** Branch `role-c/confirmatory-gap-g110`,
frozen commit `3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43` (PR #23 merged into
`main`). Today only **seed 3** is launched; seeds 4 & 5 are staged, not started.

## 1. Experiment purpose

Validate that a single global gap multiplier `g=1.10` on the official ECT
sigmoid mapping changes the trained model relative to the unmodified official
sigmoid, under otherwise identical settings, for a 256 kimg paired comparison.
This isolates the *global-only* intervention from any local controller.

## 2. Method definitions

Only two methods are compared. Both reuse the official ECT sigmoid mapping
`r_sigmoid(t;m)`; the factorized gap is `d = g * (t - r_sigmoid)`.

| Method | `--mapping` | `--global-gap-scale` | Local controller | Notes |
| --- | --- | --- | --- | --- |
| **Fixed** | `sigmoid` | `1.0` | disabled | `global_gap_scale==1.0` short-circuits to bitwise parity with the official formula (`_apply_global_gap_scale` returns `base_r` unchanged). |
| **Global-only** | `global_sigmoid` | `1.10` | disabled | Same official sigmoid gap scaled by one fixed `g=1.10`. No local t-bin controller. |

No local controller is enabled in either arm. `g=1.10` is fixed and not
searched. The resolved config diff confirms the two arms differ ONLY by
`adj` (mapping name), `global_gap_scale`, and `run_dir`.

## 3. Common training parameters (identical for both arms)

```
--data       /mnt/ect_project/datasets/cifar10-32x32.zip   (sha256 08c9ed1b…f372)
--transfer   /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl (sha256 4d5dcc1f…b4da)
--cond=False --arch=ddpmpp --precond=ect
--batch=128 --batch-gpu=16 --optim=RAdam --lr=0.0001 --dropout=0.2 --augment=0
-q 256 -k 8 -b 1 -c 0 --double=10000 --ema_beta=0.9993
--fp16=True --enable_amp=True --metrics=none
--duration=0.256  (256 kimg)   --tick=10 --snap=0 --dump=0 --ckpt=10 --sample_every=26
--nosubdir
```

Environment: conda env `myconda` (Python 3.13.5, PyTorch 2.8.0+cu128, CUDA, 1x
NVIDIA A100-PCIE-40GB). `--double=10000` keeps the whole run at stage 0
(`decay=1/256`), matching the 1-hour protocol.

## 4. Smoke test standard (passed)

Each smoke = seed 3, 2 kimg (`--duration=0.002 --tick=1 --ckpt=1`). A smoke
passes only when ALL of the following hold:

- **Config**: resolved `training_options.json` shows the expected
  `adj` / `global_gap_scale` / `local controller disabled`.
- **Numerics**: no NaN/Inf/overflow/traceback beyond the expected AMP warm-up
  `step_skipped=1` at tick 0 (GradScaler halves from 65536 until finite).
  `step_skipped` must settle to 0 by tick 1; loss finite thereafter.
- **Checkpoint**: `network-snapshot-latest.pkl` (ema+loss_fn+augment_pipe+
  dataset_kwargs) and `training-state-latest.pt` (net+optimizer_state+
  gradscaler_state+loss_fn_state+cur_nimg/cur_tick) both present and loadable.
- **Resume**: resuming from the checkpoint continues `kimg` continuously,
  keeps `global_gap_scale` (verified `gap_over_sigmoid_gap_mean≈1.10` held),
  keeps `schedule_name`, does not re-initialize optimizer/scaler, and does not
  overwrite the source checkpoint.

Both smokes passed (see `results/` smoke summaries). A real bug was found and
fixed during resume verification: under PyTorch 2.8, `torch.load` defaults to
`weights_only=True` and rejects the repo's own training-state (contains
`torch_utils.persistence` objects). Patched `training/ct_training_loop.py:627`
to `weights_only=False`. Resume then verified end-to-end.

## 5. Resume procedure

Resume replaces `--transfer` with `--resume=<run>/training-state-latest.pt`,
re-states `--mapping`, `--global-gap-scale`, and `--seed` for safety.
The schedule/gap state is also carried in the training-state; after resume
confirm `gap_over_sigmoid_gap_mean` still equals `1.10` for the global arm.

Two resume modes are distinguished:

### 5a. Verification resume (new outdir, source untouched)

Used to test that resume works correctly. Writes to a **new** `--outdir` so
the authoritative run handed off to Role D is never modified:

```
--resume=/root/ect_runs/confirmatory_256k/seed3_global110/training-state-latest.pt --outdir=/root/ect_runs/resume_checks/seed3_global110
```

### 5b. Actual interruption resume (same outdir, continues the run)

Used only when the original run was interrupted and must continue in-place.
Writes to the **same** `--outdir`:

```
--resume=/root/ect_runs/confirmatory_256k/seed3_global110/training-state-latest.pt --outdir=/root/ect_runs/confirmatory_256k/seed3_global110
```

Do **not** use mode 5b for testing — it overwrites checkpoints in the
authoritative run directory.

## 6. Formal run launch conditions (all met)

1. fixed smoke normal ✓  2. global smoke normal ✓  3. both resume ✓
4. configs differ only by gap scale ✓  5. commit frozen ✓  6. B's code frozen ✓

## 7. Checkpoint naming convention

Output root `/root/ect_runs/confirmatory_256k/{seedN_fixed,seedN_global110}/`.
Each run contains `training_options.json`, `train_summary.csv`, `stats.jsonl`,
`train.log`, `commit_sha.txt`, `start_utc.txt`, `pid.txt`,
`network-snapshot-latest.pkl`, `training-state-latest.pt`, and sample PNGs.
Checkpoints saved every `--ckpt=10` ticks (latest is authoritative).

## 8. Handoff to Role D (evaluation)

Only checkpoints that passed the integrity check (loadable snapshot + training
state, expected `total_kimg`, expected schedule, finite loss history) are
handed off. See the handoff table in the run-status file.

## 9. What is NOT done today

No local controller; no other gap scale; seeds 4 & 5 staged but not started;
no `git pull` after formal launch; smoke checkpoints are NOT formal results.
