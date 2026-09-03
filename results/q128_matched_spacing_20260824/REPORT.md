# q128 matched-spacing five-arm results

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-24
- Verification Status: ANALYZED
- Version Label: q128_matched_spacing_results_v1

## Audit status

- 210/210 unique `SEALED_PASS` jobs and 420/420 metric values.
- FP32, 50,000 samples, sample seeds 0-49999, metric seed 20260730.
- NFE2 uses `mid_t=0.821`; invalidated/pre-reuse directories are excluded.
- Preassigned server/data partitions override redundant attempts without quality selection.
- The deployed evaluator is bound by full source SHA256 in `audit.json`.

## Evidence status and blindness chronology

Historical canonical q128 `A/Bsame` quality was known before the matched-spacing calibration and five-arm freeze. The matched-arm outcomes were unknown, and the fresh 210-job matrix was not unblinded until all jobs were sealed. The exact first-unblind wall-clock time was not independently logged; `provenance/blindness_chronology.json` records a bounded interval rather than inventing a timestamp.

AULC was first defined in the post-unblind analysis commit. It is therefore deterministic descriptive evidence, not a frozen, preregistered, or primary outcome.

## Calibration identity

At stage 0 with `c=0` and no clipping, the sigmoid gap is proportional to `g/q`. Therefore `0.55/128 = 1.10/256`, making the selected q128 scale an analytic exact match to the q256 `g=1.10` reference, not merely a numerical distribution fit. The quality-blind million-sample calibration independently reports objective 0, zero clipping, and zero residual quantiles. All 15 formal trajectories remained at stage 0.

## Training integrity

- 15/15 fresh trajectories and 105/105 immutable state/snapshot pairs pass.
- 120,000 attempts produced 119,825 accepted optimizer steps and 175 AMP skips.
- All state and snapshot SHA256 values were recomputed against their receipts.
- Within each seed, initialization, optimizer, GradScaler, RNG, sampler/minibatch order, and normalized non-arm configuration hashes match across all five arms.
- Every preflight records the same A100-40GB model, runtime, dataset, and transfer hashes. The v1 schema did not record GPU UUID; this is disclosed and not reconstructed.
- All three `A` and all three `Bsame` runs are fresh five-arm cells, not historical-output reuse.

## Arm summaries

AULC is normalized trapezoidal area under the natural-log FID curve; lower is better.
Values below are three-seed means from the historical B0 generation block.
They are retained as measurements, not winner assignments. Bmatch-related NFE2
ordering is **not interpreted—pending Task 3**, because generation-block repeats
are not complete across all seeds and arms.

| Arm | NFE1 AULC | NFE2 AULC | 1024 NFE1 FID / KID | 1024 NFE2 FID / KID |
| --- | ---: | ---: | ---: | ---: |
| A | 3.1797 | 1.8980 | 8.798 / 0.005566 | 2.982 / 0.001105 |
| Bsame | 3.1997 | 1.8727 | 8.518 / 0.005373 | 2.956 / 0.001085 |
| Bmatch | 3.2619 | 1.9492 | 8.084 / 0.004928 | 2.749 / 0.000958 |
| Cmatch | 3.1291 | 1.8466 | 7.447 / 0.004426 | 2.816 / 0.000965 |
| Dmatch | 3.4220 | 2.2326 | 9.904 / 0.006313 | 2.975 / 0.001135 |

The only available paired generation-block audit of the Bmatch NFE2 ordering is
the seed-3 Cmatch--Bmatch comparison at 1024 kimg. The table reports each cell's
generation SD rather than converting the smaller point estimate into a winner.
SD is the sample SD across B1--B5; B0 is an anchor and is excluded from the mean
and SD.

| Cell | Metric scale | B0 value | B1--B5 mean | Generation SD | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| seed 3, Cmatch, NFE2 | log FID | 1.02447 | 1.00188 | 0.00525 | not interpreted |
| seed 3, Bmatch, NFE2 | log FID | 1.01758 | 1.02528 | 0.00387 | not interpreted |
| seed 3, Cmatch, NFE2 | raw KID | 0.00092153 | 0.00088364 | 0.00003143 | not interpreted |
| seed 3, Bmatch, NFE2 | raw KID | 0.00093287 | 0.00094160 | 0.00003637 | not interpreted |

For the paired contrast, Cmatch--Bmatch log FID changes from B0 `+0.00690` to
a B1--B5 mean of `-0.02340` (paired generation SD `0.00458`; B0 sign retained
in 0/5 blocks). Raw KID changes from `-0.00001135` to `-0.00005796` (paired
generation SD `0.00001129`; B0 sign retained in 5/5 blocks). The metric-specific
pattern is therefore not interpreted as an arm winner.

No Task 7 `2SD` value is transferred to another cell or used as a TIE or
equivalence rule.

## Post-unblind deterministic AULC contrasts

AULC was not present in the immutable pre-unblind protocol. It is reported as a deterministic, full-curve descriptive summary, not a frozen or preregistered primary outcome. Negative values favor the first named arm because lower AULC is better.

| Contrast | NFE | Mean | Median | Range | Negative seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bmatch-Bsame | 1 | 0.0622 | 0.0004 | [-0.0254, 0.2115] | 1/3 |
| Bmatch-Bsame | 2 | 0.0764 | -0.0383 | [-0.0510, 0.3185] | 2/3 |
| Bmatch-A | 1 | 0.0822 | 0.0243 | [-0.0270, 0.2494] | 1/3 |
| Bmatch-A | 2 | 0.0512 | -0.0642 | [-0.0657, 0.2835] | 2/3 |
| Bsame-A | 1 | 0.0200 | 0.0239 | [-0.0016, 0.0378] | 1/3 |
| Bsame-A | 2 | -0.0252 | -0.0275 | [-0.0350, -0.0132] | 3/3 |
| Cmatch-A | 1 | -0.0506 | -0.0558 | [-0.0618, -0.0342] | 3/3 |
| Cmatch-A | 2 | -0.0513 | -0.0430 | [-0.1161, 0.0051] | 2/3 |
| Dmatch-A | 1 | 0.2424 | 0.2536 | [0.0982, 0.3753] | 0/3 |
| Dmatch-A | 2 | 0.3347 | 0.3958 | [0.0344, 0.5739] | 0/3 |
| interaction | 1 | -0.1095 | -0.0634 | [-0.3168, 0.0516] | 2/3 |
| interaction | 2 | -0.2321 | -0.1742 | [-0.4651, -0.0571] | 3/3 |

## Interpretation

- The descriptive `Bmatch-Bsame` AULC contrast is not directionally stable: NFE1 is negative for 1/3 seeds and NFE2 for 2/3 seeds. Seed 3 has a large early-curve penalty.
- Under the historical B0 block, `Bmatch` has lower FID and KID than `Bsame`
  for all three seeds at NFE1. The corresponding NFE2 ordering is not
  interpreted pending Task 3; generation-block sensitivity was not evaluated
  for the Bmatch--Bsame contrast.
- `Cmatch-A` improves NFE1 AULC for 3/3 seeds and NFE2 for 2/3 seeds.
- `Dmatch-A` is worse for 3/3 seeds at both NFEs.
- The outcome-level interaction is negative for 2/3 seeds at NFE1 and 3/3 at NFE2; it is not an objective-level causal decomposition.

## Direction consistency

Under B0, FID and KID agree in direction for 36/42 `Bmatch-Bsame` descriptive
cells. At the terminal checkpoint they agree in the three NFE1 cells; the NFE2
ordering is not interpreted pending Task 3. These cells are repeated readouts,
not independent evidence units.

## Exploratory TTQ

q128 TTQ was not preregistered. `ttq_exploratory.csv` therefore reports it only as a descriptive auxiliary analysis.

## Limitations and fallacy scan

- Three seeds are insufficient for population-level significance claims; no p-value is used as the primary narrative.
- AULC and its contrasts were defined after unblinding, so they do not mitigate researcher-degrees-of-freedom or look-elsewhere risk. Their value is deterministic complete-curve summarization, not confirmatory status.
- Structural and causal fallacies are not indicated by this paired controlled design, but the 11/11 statistical fallacy checklist was reviewed.
- Verification status is `ANALYZED`, not `VERIFIED`, because metrics were not independently rerun in this report step.

## Files

- `evaluation_results.csv`: authoritative 210-job raw matrix.
- `audit.json` and `duplicate_attempts.json`: matrix/protocol audit and redundant-attempt record.
- `per_seed_aulc.csv`, `per_seed_aulc_contrasts.csv`, `contrast_summary.csv`: post-unblind deterministic descriptive AULC analysis.
- `arm_summary.csv` and `direction_consistency.csv`: arm-level and FID/KID summaries.
- `a_bsame_bmatch_trajectories.csv`: requested per-seed, per-checkpoint comparison.
- `ttq_exploratory.csv`: non-preregistered descriptive TTQ.
- `validation_summary.json`: machine-readable validation status and limitations.
- `provenance/calibration_manifest.json` and `calibration_exact_match.json`: compact quality-blind calibration and analytic equality audit.
- `provenance/blindness_chronology.json` / `.md`: historical-vs-fresh blindness chronology.
- `provenance/training/`: 15-trajectory integrity, hardware assignment, and 210-file hash audit.
- `provenance/INTEGRITY_REVIEW.md`: data checks, seven-mode failure audit, and residual limitations.
