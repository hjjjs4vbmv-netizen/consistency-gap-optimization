# TASK 2 — q256 switch-point sweep

Experiment ID: `q256_switchpoint_sweep_v1`
Status: `READY_TO_FREEZE`
Execution root: `/data/ECT002/q256_switchpoint_sweep`

This protocol has one primary question: after the same 512-kimg A chase, does the BA-versus-A NFE1 log-FID contrast show the prespecified ordering as the B prefix is lengthened? The training seed is the independent unit. Generation seeds, checkpoints, metrics, and NFEs are not independent units.

## Claim ceiling

The experiment can identify ordered dependence on B-prefix duration along the four nested `B-prefix -> A-chase` schedules tested here. It does not separately identify an early writing window, dose accumulation, memory decay, optimizer or EMA mediation, or behavior outside this q256 CIFAR-10 setting. A positive result does not imply that every adjacent pair differs significantly.

## Frozen assets

The following identities are inherited from the PR97 fresh q256 protocol. They must be staged below the execution root at the paths in `protocol.json` and verified before formal training or evaluation:

- canonical CIFAR-10 SHA256 `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`;
- transfer checkpoint SHA256 `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`;
- evaluator source commit `d6aba02fb88e9db0993623895eb2228ed717d810`;
- Inception detector SHA256 `f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4`;
- real-feature files SHA256 `ba44251acffc10807a2d74f78ab8e51cea174a375232e0170f650df53d2f73ac` and `314834241c02ff0ff60c2aa72bc603e06e3c6f6b26896af3abfc005810b501bd`.

No formal artifact may be written under `/root`. Formal run manifests bind the committed `protocol.json`, implementation commit, input checkpoint, and evaluator assets; no separate SHA sidecar is used.

## Training design

Seeds are `81..92`. Arms are `A=(target, denominator)=(1.0,1.0)` and `B=(1.1,1.1)`. The training semantics and hyperparameters match PR97: q256, official sigmoid mapping, ECT preconditioning, DDPM++, RAdam with learning rate `1e-4`, global batch 128, batch-GPU 16, dropout 0.2, EMA beta 0.9993, FP16/AMP, TF32 off, deterministic algorithms on, cuDNN benchmark off, no augmentation, and no x-flip.

For each seed:

1. `B-trunk`: train B from 0 to 512 kimg and retain full states at 128, 256, 384, and 512 kimg.
2. `CTRL`: train A from 0 to 512, perform the established A-to-A no-op switch-resume at 512, and continue to 1024.
3. `BA128`, `BA256`, `BA384`, and `BA512`: resume from the matching B-trunk full state, switch to A, and continue to 1024.

All continuation arms use the same established switch-resume implementation. Exact source-state,
sampler-cursor, optimizer, EMA, RNG, and trajectory-configuration validation is an
implementation-level prerequisite; it is not a per-seed outcome or exclusion rule.

Required evaluation snapshots are exported from exact immutable full states at the
listed kimg budget. Natural maintenance snapshots are not used because the first
maintenance event offsets their tick grid by one training batch.

Required evaluation snapshots are:

- CTRL at 640, 768, 896, and 1024;
- BA128 at 640 and 1024;
- BA256 at 768 and 1024;
- BA384 at 896 and 1024;
- BA512 at 1024.

The B-trunk switch states and each trajectory's 1024 full state are retained. No replacement seed is allowed.

## GPU schedule

Formal work uses GPU indices `0..7`. Each training seed remains on its assigned GPU:

The frozen two-phase queues are recorded in `protocol.json`. Prefix A and B jobs
are split across all eight GPUs; after a phase barrier, continuations are balanced
so every GPU receives exactly 6,528 kimg in total. Each invocation stays on one
frozen GPU, and no ordering decision uses telemetry or decoded quality.

Within each queue seeds run in ascending order. Evaluation jobs use deterministic round-robin assignment over indices `0..7`; scheduling cannot depend on telemetry or decoded quality.

## Estimands

### Primary: fixed 512-kimg A chase

For seed `i` and switch point `s` in `{128,256,384,512}`:

\[
G_s(i)=\log FID^{NFE1}_{50k}(BA(s),t=s+512)
-\log FID^{NFE1}_{50k}(CTRL,t=s+512).
\]

The four cells are `BA128@640 - CTRL@640`, `BA256@768 - CTRL@768`, `BA384@896 - CTRL@896`, and `BA512@1024 - CTRL@1024`. The frozen direction is

\[
G_{128}\geq G_{256}\geq G_{384}\geq G_{512}.
\]

### Secondary descriptive: common 1024 endpoint

\[
H_s(i)=\log FID^{NFE1}_{50k}(BA(s)@1024)
-\log FID^{NFE1}_{50k}(CTRL@1024).
\]

`H_s` combines differences in B-prefix and subsequent A-chase duration. It receives no hypothesis test or verdict. No ratio normalized by `H_512` is defined or reported.

## Evaluation and blinding

Every formal job uses the frozen evaluator, FP32, NFE1, 50,000 generated samples with generation seeds `0..49999`, and metric seed `20260730`. FID50k and KID50k share the generated features. Only log FID50k is inferential; KID is retained but cannot alter a result.

The primary matrix contains eight jobs per seed: four CTRL cells and the four fixed-chase BA cells, for `12 x 8 = 96` jobs. The common-endpoint secondary adds BA128, BA256, and BA384 at 1024, for `12 x 3 = 36` jobs. CTRL@1024 and BA512@1024 are reused from the primary matrix. The pre-seal total is exactly **132 unique jobs**. NFE2 and other milestones are not part of this task.

The 132 jobs use an opaque manifest and private decode map. No FID or KID value may be read, summarized, or communicated until training integrity and analysis-code checks have completed and every planned job has reached one terminal state:

- `PASS`: valid evaluation receipt; or
- `EXHAUSTED_FAILURE`: either the required training checkpoint is unavailable with a documented terminal training receipt, so the evaluation has zero attempts, or the one allowed identical evaluation rerun also failed and both attempts are documented.

All PASS gives `SEALED_PASS`; any documented exhausted failure gives `SEALED_WITH_DOCUMENTED_FAILURES`. Either terminal seal permits one decode. Primary eligibility is then determined only by the complete-seed rule; a secondary-only failure does not block the primary.

## Analysis and verdict

The primary is a one-sided exact Page ordered-alternatives test with training seed as block and ordered treatments `s=128,256,384,512`. The implementation ranks `Z_s=-G_s`, with larger `Z` receiving larger rank, so a large Page statistic represents the frozen direction. Ties receive average ranks. The exact null distribution corresponds to the actual complete-seed count and four treatments.

The sole positive verdict is `ORDERED_PREFIX_DEPENDENCE`, issued only when:

1. exact one-sided Page `p <= 0.05`; and
2. `mean(G_512) < mean(G_128)`.

Every other analyzed outcome is `ORDERING_NOT_RESOLVED`. This is neither an equivalence result nor evidence of no effect.

The report contains all seed-level G values; per-point mean, median, sample SD, two-sided 95% Student-t CI, and sign counts; the Page statistic and exact p value; and descriptive summaries of the three adjacent paired differences. The H summaries contain seed-level values, available `n`, mean, median, sample SD, 95% CI, and sign counts only. No per-point test, H trend test, subgroup test, or data-driven shape label is allowed.

The implementation must be committed before decode and directly test Page direction, reverse direction, ties, an independently checked exact p value, odd/even medians, and separation of primary from secondary completeness.

## Post-seal generation-noise companion

After decode, seed 81 CTRL@1024 and BA512@1024 are evaluated on five paired, non-overlapping 50k generation blocks: `0..49999`, `50000..99999`, `100000..149999`, `150000..199999`, and `200000..249999`. Block 0 reuses the matching sealed formal jobs. Report the five paired log-FID contrasts, their sample SD, and range. These are repeated measurements, do not increase `n`, and cannot change the verdict. Missing companion results are not replaced and do not affect the primary.

## Failure rules

A primary-complete seed has valid A/B trunks and branch training plus PASS receipts for all eight primary cells.

| trigger | action |
|---|---|
| protocol/commit gate fails, or any formal metric is decoded before terminal seal | entire cohort becomes `EXPLORATORY_ONLY`; no primary verdict and no repair by rerun |
| 12 primary-complete seeds | run the frozen primary |
| 9, 10, or 11 primary-complete seeds | run the same four-point primary and report actual `n` and every exclusion |
| fewer than 9 primary-complete seeds | `PRIMARY_ABORTED_INSUFFICIENT_COMPLETE_SEEDS`; descriptive archive only |
| B-trunk fails | all four BA branches and the seed are unavailable for primary |
| a primary cell reaches `EXHAUSTED_FAILURE` | exclude the seed from primary |
| a secondary-only cell reaches `EXHAUSTED_FAILURE` | retain the seed for primary; report the H point's actual available `n` |
| receipt mismatch | rerun the identical cell once; a second mismatch is `EXHAUSTED_FAILURE` |
| numerical failure | allow only the existing one bounded same-configuration recovery; a second failure terminates the trajectory |
| at least three failures concentrate in one switch-point arm | report `DIFFERENTIAL_ARM_FAILURE` and that primary estimates condition on four-point complete seeds; no three-point primary is permitted |
| companion incomplete | report `COMPANION_INCOMPLETE`; no replacement and no effect on primary |

Every exclusion has one concrete root-cause category backed by a training, parity, evaluation, or infrastructure receipt. Arm-concentrated failure may be called differential or potentially informative missingness, but not proven informative missingness.

## Frozen reporting language

For `ORDERED_PREFIX_DEPENDENCE`:

> After a fixed 512-kimg A chase, the BA-versus-A log-FID contrast showed the prespecified ordered dependence on B-prefix duration across the four schedules tested, with longer prefixes associated with a more negative contrast.

For `ORDERING_NOT_RESOLVED`:

> The fresh cohort did not resolve the prespecified ordered dependence on B-prefix duration; the four point estimates and intervals are reported descriptively.

For the H curve:

> At the common 1024-kimg endpoint, the prespecified BA-versus-A contrasts were [values and intervals]; these descriptive contrasts combine differences in B-prefix and subsequent A-chase duration.

## Downstream boundaries

TASK2 `H_512` is estimand-aligned only with TASK1's descriptive single-cell contrast `H_A = logFID_BA - logFID_AA`, not with TASK1's primary factorial history contrast H. Comparison is side-by-side and descriptive only, with no pooling or cross-cohort test. TASK2 cannot modify or rescue TASK1's frozen `INCONCLUSIVE` verdict. TASK1 `SD(H) approximately 0.119` is at most a rough planning reference and is not an estimand-aligned power guarantee for G or the Page test.

For PR #96, define the descriptive common-endpoint contrast `Delta_384:512^T2 = H_512 - H_384`. PR #96 reported equivalence for the analogous 384-512 pulse contrast after a 128-kimg common-A chase; TASK2 observes it after a 512-kimg common-A chase in independent seeds. Concordance or discordance may reflect horizon or cohort variation. Neither result rescues, overturns, or pools with the other. The primary G curve is not a direct replication of PR #96.

This cohort is never pooled with seeds 3-7, 19-28, 31-42, or 51-80. The companion is local to one checkpoint pair and is not a regime-wide noise floor.
