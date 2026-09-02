# q256 fresh confirmatory v2 — protocol DRAFT

**Status: DRAFT — pending the 2026-09-22 G5 decision and the L1 gates. No
experiment may start against this document. It carries no SHA freeze yet.**

This draft operationalizes route 3 ("delay + new preregistered cohort") as
revised after a three-seat adversarial design review (hostile reviewer /
statistical methodologist / execution realist; unanimous verdict: major
revision of the v1 plan). Companion runbook: `docs/RUNBOOK_ICLR2027_EXECUTION.md`
v1.2.

## What changed from the v1 route-3 sketch, and why

| v1 sketch | v2 draft | Reason |
| --- | --- | --- |
| Four-arm, n=16, primary H with preregistered Q covariate (ANCOVA/stratification) | **Two-arm (AA/BA), n=24, primary unadjusted H_A** | corr(C1,C2)=0.965 in the fresh n=11 data — the second continuation is nearly redundant; equal budget buys power 0.96 vs 0.82, assurance ~0.83 vs ~0.72. Q is post-treatment: conditioning on it has no interventional reading, its extreme segments may be empty, and mean-reversion alone can generate the observed cross-cohort sign flip. |
| Moderation (H~Q) as a headline question | **Moderation demoted**: dose trend in g (a design variable) is the only confirmatory moderation channel, gatekept behind the primary; H~Q retained only as an errors-in-variables persistence regression, estimation-only | Post-treatment conditioning + regression artifact + unblinded-derived thresholds + interaction power infeasible at any achievable n. |
| One-shot cohort | **Binding futility-only interim at n=12** (by seed number, sealed single-token output, zero alpha spend) + blinded SD re-estimation (cap n=28) | The previous replication was INCONCLUSIVE; a second one-shot with ~28% miss probability and no stopping rule is not acceptable risk management. |
| Complete-case primary (as in n=11) | **Ordered replacement pool primary**; complete-case demoted to sensitivity; B-arm termination count preregistered as a secondary instability endpoint | seed38/AB showed missingness is treatment-relevant; complete-case censors exactly the unstable tail and biases toward null. |
| g in {1.05, 1.1, 1.2} dose arm | **g in {1.0, 1.1, 1.2}** (1.05 dropped) | 1.05's expected effect is unpowered at any affordable n; a flat segment there would read as causal refutation while being pure noise. |
| Regime crossed switch, q128 crossed, ImageNet crossed | **Archived as next-cycle preregistered stage-2 / rebuttal ammunition** | Order-of-operations: generalize after the main effect is confirmed; and the project must not repeat the spent-stage-2 mistake. |

## Freeze checklist (all must hold before `protocol.sha256` exists)

1. G5 selects route 3.
2. Z8 (cross-cohort H~Q, per-cohort fits, bootstrap sign stability) done; if
   the discovery-side correlation sign is unstable, strike all moderation
   narrative from this protocol before freezing.
3. Z2-upgraded noise floor done (3 checkpoints x 5 generation seeds + 1–2
   same-seed retraining segments); write sigma_e into
   `primary_endpoint.planning_assumptions`.
4. Replace the planning SD if the noise floor materially revises it; recompute
   MDE/assurance lines.
5. Commit + record timestamp; the freeze commit must precede the first line of
   any stage-1 training log.

## Key numbers (from `protocol_draft.json`)

- Primary: one-sided paired t on H_A, n=24, alpha 0.05; MDE(80%) ≈ 0.058
  log-FID; assurance at the fresh posterior ≈ 0.83.
- Gatekept chain: H_A → dose trend b1 (two-sided) → optional H_A@2048.
- Budget, double ledger: 127–171 RE training + 250–320 evaluation jobs ≈
  400–500 A100·h all-in (fresh n=11 actual: ~253).
- Target venues: ICML 2027 or TMLR. ICLR 2028 rejected (calendar cost).
