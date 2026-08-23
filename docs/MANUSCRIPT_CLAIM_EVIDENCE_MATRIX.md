# Manuscript Claim–Evidence Matrix

Frozen on 2026-08-23 for manuscript v1. Training seed is the independent empirical unit. Repeated budgets, NFE readouts, FID/KID pairs, and within-seed audit minibatches are not independent replicates.

| ID | Manuscript claim | Status/ceiling | Direct evidence | Main-text location | Required boundary |
|---|---|---|---|---|---|
| C1 | In the studied ECT objective, changing pair spacing changes both the detached target endpoint and the explicit inverse-gap denominator. | **EXACT** | `theory/gap_scaling_theorem.md` §7; PR #81 equations | Sections 1 and 3; Figure 1 | One-sided stop-gradient derivative; matched online branch. |
| C2 | At a matched state, the two roles admit an exact finite-gap loss/gradient decomposition and exact four-cell identities. | **EXACT** | `theory/gap_scaling_theorem.md` Prop. 3/Cor. 3; manipulation check in formal report | Section 3 | Does not imply an additive outcome decomposition. |
| C3 | Under the local assumptions, the one-sided consistency gradient scales as `delta^kappa`, `kappa = nu(p-1)-alpha`; ECT with `p=1, alpha=1` gives `kappa=-1`. | **CONDITIONAL THEORY** | `theory/gap_scaling_theorem.md` Theorem 1/Cor. 2; PR #68 commit `debe1d5` | Section 3, compact corollary; proof details in appendix | ECT specialization leaves `nu` untested and is not a trajectory theorem. |
| C4 | At 256 kimg, the complete `g=1.10` arm B favored A in all seedwise FID/KID directions at NFE1 and NFE2. | **FORMAL DESCRIPTIVE** | `analysis/q256_target_weight_factorial/formal_evaluation_seed3_5_results_dcca41b.csv` (3 seeds × 2 metrics × 2 NFE = 12 directions) | Section 5.1; Table 1 | Absolute NFE1 FID is 306.8–331.9; no significance claim at `n=3`. |
| C5 | The original endpoint factorial did not support a universal target/denominator percentage decomposition. | **FORMAL DESCRIPTIVE** | Formal contrast table; target contrasts directionally stable at NFE1, denominator/interaction mixed | Sections 4 and 5.1 | Objective identities are exact; quality contrasts are trajectory outcomes. |
| C6 | Deterministic replay reproduces the 1024-kimg terminal states and supplies 12 four-arm trajectories over seven budgets. | **FORMAL REPLAY** | PR #79 `final_summary.json`; `replay_1024_parity.csv`; 168/168 evaluations | Section 4 and 5.2 | Replay is the same formal seed cohort, not 12 independent seeds. |
| C7 | Early arm separations contract substantially by 1024 kimg, while rankings remain seed- and NFE-dependent. | **FORMAL + SECONDARY DESCRIPTIVE** | PR #79 and PR #80 learning-curve CSVs; Figures 2 and 4 | Sections 5.2 and 5.4 | “Contract” describes magnitude, not monotonicity at every checkpoint. |
| C8 | For the observed-grid threshold FID-50k@NFE1 ≤ 10, B is one checkpoint earlier for 5/10 seeds, tied for 2, later for 2, and censored for 1. | **POST-HOC DESCRIPTIVE** | `docs/figure_source_data/figure3_time_to_quality_source.csv`; PR #78/#79/#80 source CSVs | Section 5.3; Figure 3 | No interpolation; threshold and cohorts are disclosed; no universal compute-saving percentage. |
| C9 | At 1024 kimg, B–A NFE1 directions remain mixed: 2/3 favorable in the formal cohort, 2/2 in the secondary A/B cohort, and 3/5 in the secondary factorial cohort. | **EVIDENCE-STATUS-SEPARATED** | Unified source CSV at budget 1024 | Section 5.4 | Do not pool cohorts into a confirmatory statistic. |
| C10 | The complete intervention does not define a universal static best arm: four-arm winners and rankings vary across seed, budget, and NFE. | **SUPPORTED BOUNDARY CLAIM** | PR #79/#80 four-arm curves; Figure 4 | Sections 5.4 and 7 | Applies to this fixed-q CIFAR-10 setting; not a theorem over all schedules. |
| C11 | Frozen-state RAdam updates can depart from scalar equivalence, and reset state is not a memory-neutral control. | **SUPPORTING DIAGNOSTIC** | PR #65 `seed_summary.csv` and report | Section 6 | Virtual next updates only; eight minibatches per seed are repeated measures. |
| C12 | The preregistered moment-transport manipulation failed its held-out gate, so no continuation/FID test was run. | **NEGATIVE RESULT / NO-GO** | PR #66 `outcome.md`; preflight verdict | Section 6 and limitations | Supports stopping the causal mechanism claim, not its reversal. |
| C13 | Pair spacing is best interpreted here as a locally structured finite-budget training control with trajectory- and horizon-dependent consequences. | **SYNTHESIS** | C1–C12 | Abstract, Introduction, Discussion, Conclusion | Not a universal schedule recommendation or causal optimizer mechanism. |

## Explicitly prohibited manuscript claims

- A universally best gap, target-dominant quality mechanism, or denominator irrelevance.
- A universal 13% compute saving, cross-q/cross-dataset generalization, or statistical significance inferred from repeated checkpoints.
- Optimizer-memory causation of FID/KID changes.
- The local scaling law as a validation of residual order `nu` in the ECT `p=1` configuration.
- Rebranding PR #78 or PR #80 as prospective confirmation.

