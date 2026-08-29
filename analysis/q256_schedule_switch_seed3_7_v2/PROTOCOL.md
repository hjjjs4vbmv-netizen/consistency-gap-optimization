# q256 512-kimg crossed schedule switch — seeds 3–7 v2

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-29
- Verification Status: UNVERIFIED UNTIL SOURCE INVENTORY AND PARITY PASS
- Version Label: q256_ab_crossed_switch_seed3_7_v2

## Amendment

This protocol incorporates the currently provisioned five-GPU MatPool host. It supersedes the seed3–7 v1 execution allocation before parity or switch results exist.

The scientific design is unchanged. The only operational change is five exclusive NVIDIA A100-PCIE-40GB GPUs:

| GPU | Seed | Execution order |
|---:|---:|---|
| 0 | 3 | parity A→A, B→B; then formal A→B, B→A |
| 1 | 4 | parity A→A, B→B; then formal A→B, B→A |
| 2 | 5 | parity A→A, B→B; then formal A→B, B→A |
| 3 | 6 | parity A→A, B→B; then formal A→B, B→A |
| 4 | 7 | parity A→A, B→B; then formal A→B, B→A |

Both branches for one seed run sequentially on the same physical GPU. Different seeds run concurrently. Global batch remains 128 and batch-gpu remains 16.

## Effective frozen design

- Seeds: 3–7.
- Source: exact A/B full states at 512 kimg.
- Parity: 10 no-op cells to 640 kimg; required verdict 10/10 computational-state match.
- Formal: 10 crossed trajectories and 40 immutable milestones at 640/768/896/1024 kimg.
- Evaluation: 80 FP32 jobs, 50,000 samples, NFE1/2, NFE2 mid_t 0.821, KID then FID from shared generated features.
- Analysis and claim boundaries are identical to v1.

The full machine-readable effective protocol is `protocol.json`; its digest is recorded in `protocol.sha256`.

Frozen protocol SHA256: `59298bcb34a72e18ad52b9bb9d0ecc337402b2a81e4dd72e1b3d5df3fda82f24`.
