# Source and protocol compatibility audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_compatibility_v1

## Repository refs

| Ref | Commit | State | Training-core relation |
| --- | --- | --- | --- |
| `origin/main` | `c01a61d767793ac52b427e4064f1f71583a17e1c` | canonical base | selected |
| PR #65 `analysis/g110-gradient-state-factorial` | `befeb683afa4e9dcf5b2896be37f48edefae71ba` | open, not merged | production training-core blobs identical to main |
| PR #66 `experiment/q256-g110-moment-transport-continuation` | `4b61945e8dbb5843089fe067aab906b3ed233fdf` | open draft, non-mergeable | production training-core blobs identical to main; shared audit modules diverged/old |

The isolated worktree and branch
`experiment/q256-target-weight-factorial` were created from
`origin/main@c01a61d`. PR #66 was not merged. No original PR receipts,
manifests, or results were modified.

## Old A/B reuse matrix

| Field | Old A | Old B | Reuse consequence |
| --- | --- | --- | --- |
| authoritative initial transfer identity | incomplete binding | incomplete/mismatched binding | fresh |
| model/EMA/optimizer/GradScaler hashes | missing in compatibility contract | missing or mismatched | fresh |
| Python/NumPy/CPU/CUDA RNG | not serialized | not serialized | fresh |
| sampler cursor/state | not serialized | not serialized | fresh |
| protocol/config hashes | mismatch | mismatch | fresh |
| checkpoint cadence fields | missing | missing | fresh |
| same source shared by all four arms | not established | failed | fresh |

The archived compatibility receipt is FAIL with
`reusable_controls=false`; therefore all A/B/C/D x seeds 3/4/5 are fresh.

## Frozen known artifact identities

- Canonical training data archive:
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- Authoritative initial EDM transfer checkpoint:
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.
- Historical formal evaluator: commit
  `8375d46ca4c65e85ab399fcf1effe22ebb766790`, FP32, seeds 0-49999,
  metric seed 20260730, NFE=1 and NFE=2 (`mid_t=0.821`).

## Pending live-server fields

The previous server endpoint `region-9.autodl.pro:34360` currently refuses the
TCP connection. Until access returns, the following are blockers rather than
assumptions: server checkout/commit, content hashes, dataset and transfer file
hashes, Python/PyTorch/CUDA/container identity, exact GPU inventory and UUIDs,
free disk, active jobs/tmux locks, and immutable output/source paths.

