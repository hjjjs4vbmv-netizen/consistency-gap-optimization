# q256 512-kimg crossed schedule switch — seeds 3–7

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-29
- Verification Status: UNVERIFIED UNTIL SOURCE INVENTORY AND PARITY PASS
- Version Label: q256_ab_crossed_switch_seed3_7_v1

## Frozen change

This is a new recovery-cohort protocol. The only scientific change from `q256_ab_crossed_switch_v1` is:

`seeds = [14, 15, 16, 17, 18]` → `seeds = [3, 4, 5, 6, 7]`.

The replacement was selected solely from current full-state availability, before observing any new crossed-switch result. It is not represented as the unavailable seed14–18 experiment.

All model, optimizer, data, schedule, switch point, budget, evaluation, contrast, and no-result-dependent-selection rules remain unchanged.

## Frozen matrix

- Sources: 5 seeds × A/B at exactly 512 kimg = 10 full states.
- No-op parity: 10 cells through 640 kimg; required result is `10/10 COMPUTATIONAL_STATE_MATCH`.
- Formal training: 5 seeds × `A_to_B`/`B_to_A` = 10 trajectories.
- Formal milestones: 10 × 4 budgets = 40.
- New evaluation: 10 × 4 budgets × NFE1/2 = 80 jobs.
- Statistical unit: training seed (`n=5`).

## Canonical sources

Seeds 3–5:

`/data/raw/ECT/ect_runs/q256-target-weight-replay-curve-v1-20260822/runs/q256-target-weight-replay-curve-v1/seed{seed}/arm{arm}/training-state-kimg000512.pt`

Seeds 6–7:

`/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/secondary-precision-extension/seed6-7-ab-128k-learning-curve-v1/seed{seed}/arm{arm}/checkpoints/512k/training-state.pt`

Before parity, all ten source files must pass full internal-state inventory and the two source cohorts must pass normalized trajectory-config compatibility. A snapshot is never an acceptable substitute.

## Frozen intervention and evaluation

- Switch point: 512 kimg; final budget: 1024 kimg.
- `A_to_B`: future target and denominator gap scales both become 1.1.
- `B_to_A`: future target and denominator gap scales both become 1.0.
- Everything else, including model/EMA/RAdam/GradScaler/RNG/sampler/counters/data order, is preserved.
- Evaluation is FP32 KID50k then FID50k from shared generated features, 50,000 seeds `0..49999`, metric seed `20260730`, NFE 1/2, and NFE2 `mid_t=0.821`.

Archived A/B controls may be imported only after checkpoint, dataset, and evaluator-semantic identity checks. An incompatible control receipt must be reported or regenerated with the frozen evaluator; it cannot be silently mixed.

## Claim boundary

The result is descriptive conditional post-switch evidence for the availability-selected seeds3–7 cohort. It does not establish a universal arm ranking, a causal percentage decomposition, equivalence to seeds14–18, or validity beyond these five training seeds.

The canonical machine-readable protocol is `protocol.json`; its SHA256 is recorded in `protocol.sha256`.

Frozen protocol SHA256: `829634281b4b68044c7a2e7d9e164b7401942a6814043dcab20b0a21a6427235`.
