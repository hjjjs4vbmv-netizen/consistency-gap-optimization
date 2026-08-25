# Second-q q128 A/B learning-curve results

## Material Passport

- Origin: collaborator-owned second-q execution
- Protocol: `second-q-q128-ab-pair-spacing-v2-canonical-dataset`
- Training commit: `c1e2a197010727cc463eca01a5e14ad6b6ba50db`
- Evaluation status: **48/48 PASS**
- Verification status: **175/175 essential archive files SHA256 verified**
- Evidence class: prospective paired cross-q confirmation

## Scope

This experiment tests whether the finite-budget behavior of the q256 A/B
pair-spacing intervention recurs at q=128. It contains only:

- A: target/denominator gap scales 1.0/1.0;
- B: target/denominator gap scales 1.1/1.1;
- paired training seeds 3, 4, and 5;
- immutable budgets 256, 384, 512, 640, 768, 896, and 1024 kimg.

No new q256 seeds, C/D arms, optimizer ablation, or RAdam mechanism study is
part of this result.

## Provenance closure

The original q128 evidence was excluded because its dataset archive was not
proven byte-identical to the q256 training archive. The V2 experiment instead
uses the exact q256 canonical CIFAR-10 ZIP:

`08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`

The execution-node loader smoke passed over all 50,000 RGB 32x32 images, ten
balanced classes, the frozen label mapping, and the exact preprocessing path.
The transfer checkpoint, deterministic runtime, detector, and FID/KID
references are frozen in `provenance/GO_CANONICAL_DATASET.json`.

The first launch was rejected before any optimizer step because the inherited
strict protocol validator admitted only q=256. The validation-only amendment
admits q in {128, 256}; q128 and q256 A/B native bitwise-parity tests passed.
No schedule, loss, optimizer, RNG, or checkpoint computation changed.

All six trajectories reached 1024 kimg with 8,000 attempts, 7,988-7,989
successful optimizer steps, and zero nonfinite loss/update/model/EMA or
nonpositive-denominator events. The CPU-only exporter produced 42/42 EMA
snapshots and receipts from the immutable full states with RNG unchanged.

## Frozen evaluation semantics

- Primary: NFE=1 FID-50k at 512, 640, 768, 896, and 1024 kimg.
- Secondary: KID-50k from the same generated features and NFE=2 at 768, 896,
  and 1024 kimg with `mid_t=[0.821]`.
- Samples: exactly 50,000, generation seeds 0-49999.
- Metric seed: 20260730.
- Precision: FP32.
- Pairing unit: training seed.
- Delta: B minus A; negative values favor B.

Execution priority was scheduling only. Every one of the 30 primary and 18
secondary jobs completed; no early result selected or removed a budget.

## NFE=1 primary results

### FID-50k

| Budget (kimg) | A mean | B mean | Mean B-A | SD(B-A) | B wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 30.336691 | 35.384795 | +5.048105 | 7.283490 | 0/3 |
| 640 | 12.217271 | 12.204178 | -0.013093 | 0.141234 | 2/3 |
| 768 | 9.800465 | 9.579517 | -0.220949 | 0.375585 | 2/3 |
| 896 | 8.943881 | 8.776823 | -0.167058 | 0.398077 | 2/3 |
| 1024 | 8.798337 | 8.517851 | -0.280485 | 0.481815 | 2/3 |

### KID-50k

| Budget (kimg) | A mean | B mean | Mean B-A | B wins |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.023983 | 0.027940 | +0.003956 | 0/3 |
| 640 | 0.007754 | 0.007757 | +0.000003 | 1/3 |
| 768 | 0.006053 | 0.005877 | -0.000175 | 3/3 |
| 896 | 0.005539 | 0.005440 | -0.000099 | 2/3 |
| 1024 | 0.005566 | 0.005373 | -0.000194 | 2/3 |

## NFE=2 secondary results

### FID-50k

| Budget (kimg) | A mean | B mean | Mean B-A | SD(B-A) | B wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | 3.300761 | 3.268549 | -0.032212 | 0.054436 | 2/3 |
| 896 | 3.035497 | 3.053322 | +0.017825 | 0.025939 | 1/3 |
| 1024 | 2.981832 | 2.956493 | -0.025339 | 0.018606 | 3/3 |

### KID-50k

| Budget (kimg) | A mean | B mean | Mean B-A | B wins |
| ---: | ---: | ---: | ---: | ---: |
| 768 | 0.001241 | 0.001150 | -0.000091 | 3/3 |
| 896 | 0.001112 | 0.001098 | -0.000014 | 2/3 |
| 1024 | 0.001105 | 0.001085 | -0.000020 | 2/3 |

## Interpretation

The q128 study confirms a finite-budget phase pattern rather than a uniform
benefit. B is harmful at 512 kimg, approximately neutral at 640 kimg, and
improves the mean NFE=1 FID/KID in the 768-1024 kimg high-quality region.
Direction is heterogeneous across training seeds: high-quality NFE=1 FID has
2/3 B wins, while NFE=2 FID reaches 3/3 wins only at 1024 kimg.

This is descriptive paired evidence with three independent training seeds. It
does not support a significance claim, but it does support the narrower cross-q
claim that the direction of the 1.10 intervention depends on finite-budget and
quality regime.

## Files

- `final/evaluation_results.*`: all 48 absolute metric records.
- `final/paired_results.*`: all 24 seed-level paired records.
- `final/paired_summary.*`: eight budget/NFE summaries.
- `final/completion_receipt.json`: 48-job completion receipt.
- `provenance/frozen_manifest.*`: immutable evaluation matrix.
- `provenance/ema_export_summary.json`: 42 snapshot export receipts.
- `provenance/archive_manifest.json`: essential archive manifest.
- `provenance/local_verification.json`: 175/175 local SHA256 verification.
