# q256 fresh confirmatory v2 — protocol DRAFT

**Status: DRAFT — pending the 2026-09-22 G5 decision and the L1 gates. No
experiment may start against this document. It carries no SHA freeze yet.**

This draft operationalizes route 3 ("delay + new preregistered cohort") as
revised after a three-seat adversarial design review (unanimous verdict:
major revision of the v1 sketch) and a subsequent 2026-09-02 external
request-changes review, whose eight blocking findings are all addressed in
this revision. Companion runbook: `docs/RUNBOOK_ICLR2027_EXECUTION.md` v1.3.
Committed planning/validation scripts in this directory:
`planning_calculations.py` (+ committed output `planning_calculations.json`)
and `type_I_error_simulation.py` (+ committed output
`type_I_error_simulation.json`).

## What changed from the v1 sketch, and why

| v1 sketch | v2 draft | Reason |
| --- | --- | --- |
| Four-arm, n=16, primary H with preregistered Q covariate (ANCOVA/stratification) | **Two arms per seed (AA/BA), n=24, primary unadjusted H_A** | corr(C1,C2)=0.965 in the fresh n=11 data — the second continuation is nearly redundant; equal budget buys power 0.947 vs 0.783, assurance 0.823 vs 0.697 (exact noncentral-t, `planning_calculations.py`). Q is post-treatment: conditioning on it has no interventional reading, its extreme segments may be empty, and mean-reversion alone can generate the observed cross-cohort sign flip. |
| Called "two-arm crossed design" | **Renamed: matched two-history continuation under a fixed current-A policy**, with `identification_limits` written into the protocol | With only AA/BA the design is not crossed: it identifies H_A under one fixed continuation policy, and cannot estimate the current-policy main effect, the interaction, or cross-continuation transport. PR #95/#97 remain the crossed evidence. |
| PR #96 not mentioned | **PR #96 integrated as the temporal-regime boundary** | PR #96's pulse-chase (10 seeds, 384→512 B exposure, shared A continuation to 640) found INFORMATIVE_NULL at the frozen 3% margin. Route 3 therefore tests a *different temporal regime* (0–512 kimg history followed to 1024): not every short spacing difference leaves detectable carryover — the claim is specific to the full-history scale. |
| Moderation (H~Q) as a headline question | **Moderation demoted**: dose trend in g is the only confirmatory moderation channel, gatekept behind the primary; H~Q retained only as an errors-in-variables persistence regression, estimation-only | Post-treatment conditioning + regression artifact + unblinded-derived thresholds + interaction power infeasible at any achievable n. |
| One-shot cohort | **Binding futility-only interim at n=12** + blinded SD re-estimation (cap 28) | The previous replication was INCONCLUSIVE; a second one-shot with ~28% miss probability and no stopping rule is not acceptable risk management. |
| "SD re-estimation does not affect alpha" asserted in one line | **Formulas written out + Monte Carlo type-I validation committed** | s12 and the conditional-power formula are now pinned in the protocol; `type_I_error_simulation.py` (200k reps, committed output) shows unconditional type-I 0.0495 at the planning SD and 0.0490 in the SD=0.20 stress run (extension trigger firing 86%), both within the ≤0.055 freeze bound. |
| Verdict rules = "same five categories as n=11" | **Full decision table in the protocol**: inputs, five conditions in precedence order, edge cases | Preregistration cannot inherit core adjudication by reference. The table pins: one-sided p<0.05 with hi95≥0 → INCONCLUSIVE (exactly the fresh n=11 configuration); equivalence takes precedence over WEAK; gatekeeping step 1 passes only on STRONG or WEAK. |
| Complete-case primary (as in n=11) | **Ordered replacement pool primary, with an explicit completion-conditioned estimand** + B-arm termination count coprimary + **hard rule: >4 replacements → cohort EXECUTION_FAILED, no verdict rendered** | seed38/AB showed missingness is treatment-relevant; complete-case censors exactly the unstable tail and biases toward null. The replacement rule does not eliminate informative missingness — it moves the estimand to the completer population, which must be stated next to the estimate, with the all-started tipping-point analysis preserving the unconditional reading. |
| Dose model H_{s,g} = b0 + b1(g−1.0) + u_s + eps, Wald test | **Within-seed linear contrast** L_s = −0.5·Y_{1.0} + 0·Y_{1.1} + 0.5·Y_{1.2} on terminal log-FID, paired t over 8 seeds (two-sided 0.05) + co-reported exact 2^8 sign-flip p | The v1 model degenerated at g=1.0 and leaned on a random-effect Wald test at n=8. The contrast is well-defined per seed, has computed power 0.899 at the linear projection (MDE 0.078), and its missing-data rule (replacement pool → complete-contrast + tipping point) is written out. |
| g in {1.05, 1.1, 1.2} dose arm | **g in {1.0, 1.1, 1.2}** (1.05 dropped) | 1.05's expected effect is unpowered at any affordable n; a flat segment there would read as causal refutation while being pure noise. |
| Budget: three inconsistent ranges (127–149/127–171, 250–300/250–320, 400–460/400–500) | **Three named packages**: MINIMAL (127 RE, 250 jobs, ~400 h), WITH_HORIZON_SUBSET (149 RE, 300 jobs, ~460 h), WITH_HORIZON_FULL (171 RE, 320 jobs, ~500 h) | Single source of truth in `budget_packages`; every document must cite package names, never raw ranges. |
| Regime crossed switch, q128 crossed, ImageNet crossed | **Archived as next-cycle preregistered stage-2 / rebuttal ammunition** | Order-of-operations: generalize after the main effect is confirmed; the project must not repeat the spent-stage-2 mistake. |

## Freeze checklist (all must hold before `protocol.sha256` exists)

1. G5 selects route 3.
2. Z8 (cross-cohort H~Q, per-cohort fits, bootstrap sign stability) done; if
   the discovery-side correlation sign is unstable, strike all moderation
   narrative from this protocol before freezing.
3. Z2-upgraded noise floor done (3 checkpoints × 5 generation seeds + 1–2
   same-seed retraining segments); write sigma_e into
   `primary_endpoint.planning_assumptions`.
4. `planning_calculations.py` rerun with any revised assumptions;
   `planning_calculations.json` re-committed; MDE/assurance lines updated.
5. `type_I_error_simulation.py` rerun under the frozen parameters;
   unconditional empirical type-I error ≤ 0.055 in both null scenarios.
6. `protocol_lint.py` passes on the final protocol.json.
7. Commit + record timestamp; the freeze commit (SHA256 over the four named
   files: protocol.json, interim_futility.py, planning_calculations.py,
   planning_calculations.json) must precede the first line of any stage-1
   training log.

## Key numbers (all from the committed scripts, not hand calculation)

- Primary: one-sided paired t on H_A, n=24, alpha 0.05. Power **0.947** at
  the fresh H_A point estimate (−0.0776); **0.938** at the pooled-H point
  estimate (review-concordance variant, recorded). MDE(80%) **0.0591**.
  Assurance **0.823** (flat prior; posterior N(−0.0776, 0.1129/√11);
  order-200 Gauss-Hermite over the exact noncentral-t power).
- Rejected alternative: four-arm n=16 → power 0.783, assurance 0.697.
- Dose within-seed contrast (n=8, two-sided 0.05): power **0.899** at the
  linear projection; MDE(80%) **0.078**; SD planning value 0.0798
  (= sd(H_A)/√2, to be replaced by the Z2 noise floor).
- Interim type-I validation (200k reps): unconditional **0.0495** (planning
  SD) / **0.0490** (stress SD 0.20, extension firing 86%); futility stop
  probability under the design effect **0.9%**.
- Budget: MINIMAL 127 RE / 250 jobs / ~400 h; WITH_HORIZON_SUBSET 149 / 300 /
  ~460 h; WITH_HORIZON_FULL 171 / 320 / ~500 h (fresh n=11 actual: ~253 h).
- Target venues: ICML 2027 or TMLR. ICLR 2028 rejected (calendar cost).