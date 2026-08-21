# Role E disjoint 5k evaluation handoff

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: gap_lr_seed45_role_e_handoff_v1

## Outcome

The six completed seed-4/5 A/B/C endpoints are frozen for Role E's disjoint
5k repeat evaluation. The machine-readable source of truth is
`results/gap_lr_seed_replication_role_e_handoff/role_e_disjoint_5k_handoff.json`.
Every cell uses the numbered 256-kimg EMA file
`network-snapshot-000008.pkl`; the mutable `network-snapshot-latest.pkl`
alias is not permitted.

This handoff authorizes evaluation only. It does not authorize training,
seed-4/5 FID-50k, or a new method/control. It also does not claim that the
D1--D5 deviations received an independent quality-blind review. The execution
basis is the Leader's explicit 2026-08-13 directive after PR #50 merged.

## Frozen endpoints

| Seed | Arm | Run ID | Gap | RAdam LR | Final numbered EMA SHA256 |
| ---: | :---: | --- | ---: | ---: | --- |
| 4 | A | `arm_a_g1_0_lr_fixed_s4` | 1.0 | `1.0e-4` | `ec724a4705cab6a789f05404a2fc82b362d5e3ef3aa5ed24735b82583059b684` |
| 4 | B | `arm_b_g1_3_lr_fixed_s4` | 1.3 | `1.0e-4` | `e6adb0548babb1de2aaa4a55e22ae4adfbe4d7daae2f2547e11e46628b726595` |
| 4 | C | `arm_c_g1_3_lr_matched_s4` | 1.3 | `1.2963523762588692e-4` | `b5d19259a9089ba2bc8b8cb90e7dcd669b065a364efbb4f99736aae5bdded31e` |
| 5 | A | `arm_a_g1_0_lr_fixed_s5` | 1.0 | `1.0e-4` | `97837ecba0f11d5b7d25c1eada17adf8ce5d5671ceae6553291f1405c5c16455` |
| 5 | B | `arm_b_g1_3_lr_fixed_s5` | 1.3 | `1.0e-4` | `fce3c1f2c14357b617f51e7220dd3dfe0e02c3e9894318678d7e167bff6af36a` |
| 5 | C | `arm_c_g1_3_lr_matched_s5` | 1.3 | `1.2963523762588692e-4` | `48e7fa22cef49b158b9b99da71f20c472149ebced9028b6f5c165653a2762852` |

All six files are 223,172,916 bytes. The corresponding public receipt paths
and receipt SHA256 values are frozen in the JSON handoff. Before each
evaluation, Role E must rehash the checkpoint and fail closed on any mismatch.

Seed 3 remains bound to the existing formal handoff and the already evaluated
`network-snapshot-000008.pkl` endpoints under
`gap_lr_matched_q128_s3_v1`; this delivery does not replace those artifacts.

## Evaluation contract

- NFE: 1; precision: FP32.
- Metrics per block: `fid5k_full` and `kid5k_full`, one metric repeat.
- New sampling-seed blocks: `5000-9999`, `10000-14999`, and
  `15000-19999`.
- The earlier `0-4999` block must not be reused.
- Within each training seed and block, A/B/C must receive the identical
  ordered sampling-seed list.
- Report `delta_gap = B-A` and `delta_ctrl = C-B` separately for FID and KID.
- The primary diagnostic is whether `delta_ctrl(seed 4) > 0` and
  `delta_ctrl(seed 5) < 0` persist across the three disjoint blocks.

The handoff deliberately does not use the absorption ratio as a primary
quantity and does not authorize seed-4/5 FID-50k.

## Preflight

From the repository root, validate the Git-tracked handoff and all receipt
bindings:

```bash
python scripts/validate_gap_lr_seed_replication_eval_handoff.py
```

On the evaluation host, additionally compute SHA256 and byte size for each
resolved checkpoint path and compare them with the six frozen endpoint rows
before generation begins. Evaluation output should record the handoff JSON
SHA256, checkpoint SHA256, exact sample block, evaluator seed, source commit,
command, start/end timestamps, and exit status.

## Status boundary

This artifact is `ANALYZED`, not `VERIFIED`, because the raw external
checkpoints could not be re-read from the present host and no evaluation was
run as part of the Collaborator task. The public receipts themselves passed
their strengthened artifact checks; the local validator confirms that the
handoff copies those exact identities without drift.
