# Runbook Amendment: Fresh q128 Regime and History Generalization Study v1

Protocol: `q128_fresh_regime_history_n8_v1`

The team decided to execute the new q128 generalization study ahead of the previously anticipated schedule. This amendment is frozen before any formal training log is created.

The completed q128 five-arm study at `results/q128_matched_spacing_20260824/` (seeds 3–5) is a discovery reference only. It is not part of the new cohort's primary inference, is never pooled with the fresh data, and its frozen configuration is immutable. After the fresh cohort is sealed and decoded, its discovery effect size may be shown beside the fresh effect in a table explicitly labelled `DESCRIPTIVE EVIDENCE SYNTHESIS`.

The sole primary inference cohort contains eight fresh seeds: 201–208. Each seed is permanently assigned to one of eight GPUs, and all native arms and continuations for that seed execute sequentially on that GPU. No seed may migrate between GPUs.

The primary question is whether Bsame history from 0–512 kimg improves future NFE1 quality after switching to current A from 512–1024 kimg: `H_A = logFID(BA@1024) - logFID(AA@1024)`, with the preregistered directional alternative `E[H_A] < 0`. Primary directional success is `mean(H_A) < 0` and a one-sided paired t-test p-value below 0.05. This threshold intentionally differs from PR97: sign count, LOSO, permutation p-value, the 3% margin, two-sided CI exclusion of zero, and TOST are reported independently and are not additional primary gates.

Key secondary questions are: (1) the legal A/Bsame phase contrast `P = [logFID(Bsame)-logFID(A)]_1024 - [logFID(Bsame)-logFID(A)]_512`, tested one-sided at 0.05; and (2) whether the Bmatch/Cmatch diagnostic trajectory ranking is NFE-dependent at 1024 kimg. No secondary result can rescue or veto the primary result, and Cmatch/Dmatch never enter the history primary.

Practical magnitude is a separate axis. TOST uses the frozen equivalence band `±log(1.03)` at alpha 0.05. `MATERIAL_NEGATIVE_SUPPORTED` is reported only when the one-sided 95% upper bound is below `-log(1.03)` and is not required for primary success.

There is no post-hoc seed addition. Seeds 209, 210, 211, and 212 form a presorted replacement pool. A seed missing any primary-required A@512, Bsame@512, AA@1024, or BA@1024 artifact, or suffering terminal training failure, is excluded as a whole primary unit and replaced by the next unused pool seed. Exact technical reruns are permitted only when trajectory identity is unchanged and the failure and rerun are recorded. More than four replacements yields `EXECUTION_FAILED` with no scientific verdict. Replacement does not erase informative missingness: all-started, arm-specific, and B-history-specific failures and the completion-conditioned estimand must be reported.

No old n=3 observation may be combined with the fresh n=8 to create an n=11 primary analysis, pooled paired test, or pooled p-value. No success threshold, metric, NFE, budget, margin, arm order, or cohort membership may change after outcome observation.
