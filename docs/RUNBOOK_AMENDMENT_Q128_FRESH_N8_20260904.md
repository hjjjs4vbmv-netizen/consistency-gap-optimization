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

## Pre-formal runtime repair 001

The first seed999 smoke invocation, after the initial protocol freeze but before any optimizer step or formal training log, stopped at process import with `ModuleNotFoundError: No module named 'click'`. The failed invocation is retained as immutable engineering evidence. It produced no formal run, quality metric, checkpoint, or scientific observation.

The repair is limited to installing Click 8.1.8 from the locally retained wheel `click-8.1.8-py3-none-any.whl`, whose SHA256 is `63c132bbbed01578a06712a2d1f497bb62d9c1c0d329b7903a866228027263b2`. The repaired environment must pass the training and evaluation import gate before smoke attempt 002. The exact machine-readable receipt is `analysis/q128_fresh_regime_history_n8_v1/runtime_repair_001.json`.

This repair does not change the frozen scientific configuration, protocol JSON, analysis plan, protocol SHA256, source commit, cohort, replacements, GPU assignment, arm order, checkpoints, evaluation matrix, metrics, hypotheses, or verdict thresholds. The protocol SHA256 therefore remains unchanged. This amendment and its receipt require a new pushed freeze commit, and that commit timestamp must precede smoke attempt 002 and every formal training log. Smoke attempt 002 uses a fresh run directory; attempt 001 is never overwritten. Any further smoke or parity failure remains fail-closed and prevents formal launch.

## Pre-formal verifier repair 002

Smoke attempt 002 confirmed that the repaired runtime enters training, but review of its engineering telemetry exposed an error in this experiment's new wrapper: it rejected every `raw_grad_nonfinite_count`, including the overflow-driven GradScaler warm-up skips already permitted and strictly validated by the repository's frozen q128/q256 verifiers. The attempt was terminated before the long resume/switch phase because the wrapper's old aggregate rule made its final verdict deterministically fail. Its partial seed999 artifacts and log remain immutable engineering evidence; no formal seed or quality metric was produced.

The verifier repair adopts the existing strict AMP rule without changing training: raw-gradient non-finites are permitted only when the same attempt is marked skipped, only below 10,000 processed images, only when GradScaler decreases, and only when the parameter update norm is zero. Loss, finite-gradient component, sanitized gradient, update, model, EMA, target/denominator factors, and all non-skipped raw gradients must remain finite. Zero clipping and positive target/denominator gaps remain mandatory. A late skip, an unmatched raw non-finite, a non-finite scientific state, or a skipped attempt that changes parameters fails closed.

This is a verifier correctness repair, not a relaxation of the scientific gate or a change to the trajectory. It matches the pre-existing q128/q256 AMP contract and adds regression tests for an allowed warm-up skip, an unmatched raw non-finite, and a late skip. The source manifest binds the corrected wrapper. The protocol SHA256 and all scientific choices remain unchanged. A new freeze commit must precede smoke attempt 003 and every formal log; attempt 003 must use another fresh directory.
