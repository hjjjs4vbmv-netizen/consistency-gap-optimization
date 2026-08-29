# q256 512-kimg crossed schedule switch — seeds 3–7 v3

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-30
- Verification Status: UNVERIFIED UNTIL SOURCE INVENTORY AND PARITY PASS
- Version Label: q256_ab_crossed_switch_seed3_7_v3

## Effective amendments

This is the executable five-GPU protocol. It supersedes v2 before parity or switch results exist.

- Five exclusive A100 40GB GPUs: seed3→GPU0, seed4→GPU1, seed5→GPU2, seed6→GPU3, seed7→GPU4.
- Same-seed branches run sequentially on the same physical GPU.
- The MatPool training dataset path is `/mnt/ect_project/datasets/cifar10-32x32.zip` and is byte-identical to canonical CIFAR-10, SHA256 `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- Global batch 128, batch-gpu 16, all training/evaluation/analysis settings remain unchanged.

## Frozen matrix

- 10 exact A/B source states at 512 kimg.
- 10 no-op parity cells; required 10/10 computational-state match.
- 10 crossed trajectories and 40 immutable milestones.
- 80 switched FP32 evaluation jobs.
- Five training seeds are the statistical units.

The complete machine-readable protocol is `protocol.json`; its digest is recorded in `protocol.sha256`.

Frozen protocol SHA256: `195ca2843791c0ea28ac5a87f3c9e0fb24a4fb8c9214b665331dbbd92648b32d`.
