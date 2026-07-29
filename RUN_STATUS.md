# Confirmatory 256k — Run Status (seed 3)

Training commit: `3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43` (training_code_sha; recorded in each run dir `commit_sha.txt`)
PR head: `79143c685e5588948972c17457b1c51c7a77bb49` (pr_head_sha; docs + resume fix only, not a training baseline)
Output root: `/root/ect_runs/confirmatory_256k/`
GPU: 1x NVIDIA A100-PCIE-40GB (two runs share the GPU, ~7GB total)

| Method | Seed | Outdir | PID | Port | Start (UTC) | Status | Latest kimg |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: |
| Fixed | 3 | /root/ect_runs/confirmatory_256k/seed3_fixed | (see pid.txt) | 29501 | (see start_utc.txt) | COMPLETED | 256 |
| Global 1.10 | 3 | /root/ect_runs/confirmatory_256k/seed3_global110 | (see pid.txt) | 29502 | (see start_utc.txt) | COMPLETED | 256 |

Notes:
- Both launched from identical COMMON args; differ ONLY by mapping + global-gap-scale.
- Checkpoints saved every 10 ticks (`--ckpt=10`); `network-snapshot-latest.pkl` +
  `training-state-latest.pt` are the authoritative final artifacts.
- Resume command (if interrupted) is in CONFIRMATORY_COMMANDS.sh section 3.
- Patch applied: `training/ct_training_loop.py:627` `weights_only=False` (PyTorch 2.8 resume compat).
