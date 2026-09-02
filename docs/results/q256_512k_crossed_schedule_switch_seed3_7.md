# q256 512-kimg crossed schedule switch — seeds 3–7

Status: **EXECUTION AND ANALYSIS PIPELINE PASS**

This status records protocol and artifact validation. It is not a verdict that a
scientific hypothesis passed.

Protocol: `q256_ab_crossed_switch_seed3_7_v3`

Protocol SHA256: `195ca2843791c0ea28ac5a87f3c9e0fb24a4fb8c9214b665331dbbd92648b32d`

## Scope

This is a pre-result-frozen recovery protocol over the availability-selected
seeds 3–7, with compatibility-audited archived controls and pre-parity
operational amendments. It is not the unavailable seed14–18 confirmatory
experiment and does not support a universal schedule ranking or a global causal
percentage.

Each seed supplied two exact 512-kimg full states, arm A `(target, denominator)=(1.0, 1.0)` and arm B `(1.1, 1.1)`. The formal intervention continued A states under B (`A→B`) and B states under A (`B→A`) to 1024 kimg. Archived `A→A` and `B→B` trajectories are the controls.

The frozen contrasts are:

- `S_A = AB - AA`: effect of using current B rather than A after an A history.
- `S_B = BB - BA`: effect of using current B rather than A after a B history.
- `H_A = BA - AA`: B-history versus A-history under current A.
- `H_B = BB - AB`: B-history versus A-history under current B.
- `I_switch = BB - BA - AB + AA`: history-by-current-schedule interaction.

KID and FID are lower-is-better, so a positive `S_A` or `S_B` favors current A, while a negative `H_A` or `H_B` favors a B history.

## Execution evidence

- Source inventory and cross-cohort compatibility: 10/10 PASS.
- No-op parity: 10/10 `COMPUTATIONAL_STATE_MATCH`.
- Formal training: 10/10 trajectories PASS; 40/40 immutable milestones at 640/768/896/1024 kimg.
- Evaluation: 80/80 FP32 jobs PASS; every receipt confirms identical KID/FID generated-feature hashes.
- Analysis: 100 trajectory rows, 100 contrast rows, 20 seed-level AULC rows, and 12 plots.
- Statistical unit: five training seeds. Budget × NFE rows are repeated measurements, not independent samples.

The compact execution evidence manifest records the SHA256 of every training manifest, telemetry file, milestone state/snapshot, evaluation receipt, and result artifact. The complete result files and receipts are retained in the repository.

## Main finite-horizon finding

### NFE1 at 1024 kimg

The crossed continuations separate the role of the spacing used before the
512-kimg switch from the spacing used afterward. The strongest result is the
prior-spacing contrast: B history yielded lower NFE1 FID under both current
policies in 5/5 seeds. Current A was also favored over current B after A history
in 5/5 seeds and after B history in 4/5 seeds.

| Contrast | FID mean | FID sign | KID mean | KID sign |
|---|---:|---:|---:|---:|
| `S_A = AB-AA` | +0.1945 | 5/5 positive | +1.608e-4 | 5/5 positive |
| `S_B = BB-BA` | +0.1416 | 4/5 positive | +1.415e-4 | 4/5 positive |
| `H_A = BA-AA` | -0.6192 | 5/5 negative | -3.907e-4 | 5/5 negative |
| `H_B = BB-AB` | -0.6721 | 5/5 negative | -4.099e-4 | 4/5 negative |

The endpoint interaction was smaller in its signed mean and not sign-consistent:
FID mean `-0.0529` (3 negative, 2 positive) and KID mean `-1.927e-5`
(3 negative, 2 positive). The log-FID sensitivity analysis gives the same
three-part picture: `H_A=-0.0703` and `H_B=-0.0754`, `S_A=+0.0216` and
`S_B=+0.0164`, and `I_switch=-0.0051` with mixed signs. Within this audited
domain, the late current-policy contrast is therefore approximately
transportable across the two histories; the experiment does not establish a
strong history-by-current interaction.

### Delayed source-to-future ranking reversal

At the 512-kimg switch, B had worse NFE1 FID than A in seeds 5, 6, and 7. By
1024 kimg, B history yielded lower NFE1 FID under both current policies in each
of those seeds, as it did in all five seeds overall. Therefore, for CIFAR-10 at
q256, a 512-kimg switch, continuations through 1024 kimg, and NFE1 FID, the
metric at the switch did not always rank training states by their future
generation quality. This is a descriptive source-to-future reversal, not a
claim that FID generally lacks predictive value.

### Sequence ranking

At NFE1 and 1024 kimg, BA had the lowest FID among AA/AB/BA/BB in 4/5 seeds;
BB was lowest in seed 6. This is consistent with coarse-then-fine temporal role
separation in this q256 setting. It does not establish a universal curriculum or
a new state-of-the-art schedule.

### NFE2 at 1024 kimg

Current-schedule effects were much smaller and mixed across seeds. FID means were `S_A=+0.0030` (3 positive, 2 negative) and `S_B=+0.0266` (4 positive, 1 negative); KID means were `S_A=-7.458e-6` (2 positive, 3 negative) and `S_B=+2.354e-5` (3 positive, 2 negative).

History contrasts still mostly favored a B history, but less strongly than at NFE1: FID `H_A=-0.0786` and `H_B=-0.0550`, each negative in 4/5 seeds; KID `H_A=-5.939e-5` and `H_B=-2.839e-5`, also negative in 4/5 seeds.

The NFE2 interaction remained small/mixed: FID mean `+0.0236` (3 positive, 2 negative) and KID mean `+3.100e-5` (4 positive, 1 negative).

### AULC

The normalized 512–1024 kimg AULC supports the NFE1 current-schedule pattern,
with mean FID `S_A=+0.1527`, `S_B=+0.1173` and mean KID
`S_A=+1.256e-4`, `S_B=+1.157e-4`. NFE2 AULC current-schedule effects were near
zero and mixed. Endpoint history effects were more sign-consistent than AULC
history effects.

## Analysis provenance and claim boundary

The raw-FID contrasts follow the frozen analysis plan. The log-FID table and
delayed-reversal table were added after results were known and are marked
`post-unblind descriptive` in their first line; they are sensitivity and
interpretive analyses, not retroactively confirmatory endpoints.

The experiment supports finite-horizon prior-spacing quality carryover in this
five-seed q256/CIFAR-10 cohort. It does not establish a confirmed interaction
null, global causality beyond these paired seeds, mediation by the parameter,
RAdam-moment, or EMA differences measured in the separate same-state audit, or
a unique optimizer mechanism.

## Training integrity note

Every formal trajectory contains exactly 4,000 attempted iterations (`4001…8000`) and ends at 1024.000 kimg. Across all ten continuations, AMP recorded 20 protected step skips associated with 25 raw-gradient nonfinite elements. Post-sanitization gradient, update, model, EMA, factor, denominator, and per-step loss nonfinite counts were all zero. The resumed logger's first display window printed `loss nan` because its display accumulator was empty; the step-level telemetry was finite from the first continuation step.

## Artifacts

- [Result report](../../results/q256_schedule_switch_seed3_7/REPORT.md)
- [Per-seed trajectories](../../results/q256_schedule_switch_seed3_7/per_seed_trajectories.csv)
- [Per-seed contrasts](../../results/q256_schedule_switch_seed3_7/per_seed_contrasts.csv)
- [Per-seed AULC](../../results/q256_schedule_switch_seed3_7/per_seed_aulc.csv)
- [Contrast summaries](../../results/q256_schedule_switch_seed3_7/contrast_summaries.csv)
- [Post-unblind log-FID sensitivity](../../results/q256_schedule_switch_seed3_7/contrast_summaries_logfid.csv)
- [Post-unblind delayed reversals](../../results/q256_schedule_switch_seed3_7/per_seed_delayed_reversal.csv)
- [Analysis audit](../../results/q256_schedule_switch_seed3_7/analysis_audit.json)
- [Execution evidence manifest](../../analysis/q256_schedule_switch_seed3_7_v3/evidence/execution_evidence_manifest.json)
- [Source inventory report](../../analysis/q256_schedule_switch_seed3_7_v3/evidence/SOURCE_INVENTORY_REPORT.md)
- [Parity report](../../analysis/q256_schedule_switch_seed3_7_v3/evidence/PARITY_REPORT_V4.md)
- [Control-import report](../../analysis/q256_schedule_switch_seed3_7_v3/evidence/CONTROL_IMPORT_REPORT.md)

Representative plots:

- [Four trajectories, FID NFE1](../../results/q256_schedule_switch_seed3_7/per_seed_four_trajectories_fid50k_full_nfe1.png)
- [Current-schedule effects, FID NFE1](../../results/q256_schedule_switch_seed3_7/current_schedule_effects_fid50k_full_nfe1.png)
- [History and interaction, FID NFE1](../../results/q256_schedule_switch_seed3_7/history_and_interaction_fid50k_full_nfe1.png)
