# q256 factorial seed6/7 secondary precision extension: final results

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_factorial_seed6_7_extension_results_v1

## Scope and status

Seeds 6 and 7 are an independent **secondary precision extension** outside the
original seeds3/4/5 preregistration. They neither replace seeds3/4/5 nor become
part of the original preregistration retrospectively.

- Training: **PASS**, 8/8 seed × arm endpoints complete at 256.000 kimg.
- Integrity: **PASS**, seed6 and seed7 full four-arm audits both passed.
- Frozen evaluation: **PASS**, 16/16 jobs and 32/32 metric values complete.
- Evaluation completion time: `2026-08-21T05:15:25Z`.
- No evaluation job was selected, stopped, or changed in response to previews,
  loss values, seeds3/4/5 results, or intermediate extension results.

The detailed training endpoint, telemetry, loss, denominator, non-finite, and
training-objective diagnostic record is in
`q256_factorial_seed6_7_extension_report.md`. In both seeds, the requested
training-objective pattern was reproduced: target geometry raised the raw
objective, denominator scaling lowered it, B versus A was approximately
cancelling, and the interaction was small relative to the component contrasts.
These are training diagnostics only.

## Frozen evaluation protocol

FP32; 50,000 samples per job; sample seeds `0..49999`; metric seed `20260730`;
KID and FID computed from byte-identical retained generated Inception features;
all NFE1 jobs before all NFE2 jobs; `mid_t=0.821` for NFE2; one fixed CIFAR-10
reference. Training source was
`dcca41b19e7c45512b5fbe98776520396a1bf9ac`. Evaluation ran on physical GPU1,
`GPU-ef9edaf6-d661-e143-efd1-154c1ad29f10`, using the same numerical semantics
as the frozen GPU0 evaluation.

## Per-seed frozen results

Lower FID and KID are better. Values are copied from the 16 immutable PASS
receipts and rounded only for display.

| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---:|:---:|---:|---:|---:|---:|
| 6 | A | 305.224734915839 | 0.313867150185185 | 62.855686342485 | 0.047106663518518 |
| 6 | B | 304.565170691106 | 0.322960391546547 | 42.075807219950 | 0.030632105858358 |
| 6 | C | 308.770000659359 | 0.325635416811812 | 44.241865132447 | 0.029789228598599 |
| 6 | D | 304.081007487814 | 0.321920921779279 | 40.730548665983 | 0.028477421694194 |
| 7 | A | 336.940712074021 | 0.360819374171672 | 264.200122635229 | 0.280926646594094 |
| 7 | B | 326.070011432932 | 0.340131077249750 | 200.181972196292 | 0.210480356003504 |
| 7 | C | 333.612653519318 | 0.356550195070070 | 263.213962462358 | 0.280429330260260 |
| 7 | D | 306.208600077407 | 0.309107013721221 | 123.422270258965 | 0.115155929399399 |

## Across-seed descriptive summary

Values are mean ± sample SD over the two independent extension training seeds
(`n=2`). The SDs estimate only seed6/7 dispersion and are not confirmatory
uncertainty estimates.

| Arm | NFE | FID mean ± SD | KID mean ± SD |
|:---:|---:|---:|---:|
| A | 1 | 321.082723494930 ± 22.426582520508 | 0.337343262178428 ± 0.033200235972634 |
| B | 1 | 315.317591062019 ± 15.206218716882 | 0.331545734398148 ± 0.012141508298358 |
| C | 1 | 321.191327089339 ± 17.566408299940 | 0.341092805940941 ± 0.021860049345293 |
| D | 1 | 305.144803782610 ± 1.504435147704 | 0.315513967750250 ± 0.009060801281354 |
| A | 2 | 163.527904488857 ± 142.372016256783 | 0.164016655056306 ± 0.165335695609663 |
| B | 2 | 121.128889708121 ± 111.797941402170 | 0.120556230930931 ± 0.127171917262167 |
| C | 2 | 153.727913797403 ± 154.836654912621 | 0.155109279429429 ± 0.177229315522247 |
| D | 2 | 82.076409462474 ± 58.471877086387 | 0.071816675546797 ± 0.061290960581481 |

## Descriptive four-arm contrasts

The contrast order is `[C−A, B−D, D−A, B−C]`; interaction is
`I = B−C−D+A`. These are frozen-endpoint descriptions, not new preregistered
tests. Negative metric contrasts favor the first term because lower is better.

| Metric | NFE | Seed | C−A | B−D | D−A | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|---:|
| FID | 1 | 6 | 3.545265743520 | 0.484163203292 | -1.143727428025 | -4.204829968254 | -3.061102540228 |
| FID | 1 | 7 | -3.328058554703 | 19.861411355525 | -30.732111996614 | -7.542642086386 | 23.189469910229 |
| KID | 1 | 6 | 0.011768266626627 | 0.001039469767267 | 0.008053771594094 | -0.002675025265265 | -0.010728796859359 |
| KID | 1 | 7 | -0.004269179101601 | 0.031024063528529 | -0.051712360450450 | -0.016419117820320 | 0.035293242630130 |
| FID | 2 | 6 | -18.613821210038 | 1.345258553967 | -22.125137676502 | -2.166057912497 | 19.959079764005 |
| FID | 2 | 7 | -0.986160172871 | 76.759701937327 | -140.777852376265 | -63.031990266066 | 77.745862110198 |
| KID | 2 | 6 | -0.017317434919920 | 0.002154684164164 | -0.018629241824324 | 0.000842877259760 | 0.019472119084084 |
| KID | 2 | 7 | -0.000497316333834 | 0.095324426604104 | -0.165770717194695 | -0.069948974256757 | 0.095821742937938 |

Within this extension, NFE1 contrast directions are not uniformly stable across
seed6/7, while NFE2 magnitudes show pronounced between-seed heterogeneity. The
evaluation readouts therefore should not be inferred from the stable raw-loss
diagnostic structure. With only two extension seeds, these results are reported
descriptively and do not revise the original seeds3/4/5 preregistered analysis.

## Integrity and provenance

- All 16 receipts have `status=passed`, return code 0, 50,000 samples, FP32,
  sample seeds `0..49999`, metric seed `20260730`, and a PASS in-run GPU
  exclusivity monitor with no foreign-process incident.
- Every job contains both `kid50k_full` and `fid50k_full`; within each job their
  retained generated-feature SHA-256 values are byte-identical.
- NFE1 has 8/8 PASS receipts with no `mid_t`; NFE2 has 8/8 PASS receipts with
  `mid_t=[0.821]`.
- Evaluation completion SHA-256:
  `c93bd08a97669e605ef839406aa4e852756d552dcdb4cb65a7f95c49456a74e2`.
- Evaluation plan SHA-256:
  `c54f1cc269e09ada23b70159be385579d588b52db9bb616e56790a421a5093e3`.
- Canonical digest of the 16 sorted `filename:receipt_sha256` records:
  `21e6fe99ce6972fa55fc5fd98f0f920522be862689e6e952bc9f98d95c931d1b`.
- The seed6 audit-adapter compatibility failure remains preserved as
  `seed6_integrity_audit.failed-adapter-v1.{json,md}` and is not represented as
  a scientific failure or a PASS result. The compatible final audits for seed6
  and seed7 both passed without changing the scientific gate.

The companion CSV contains all full-precision per-seed metrics and receipt,
checkpoint, feature, and artifact-tree hashes. The companion JSON contains the
machine-readable protocol, training diagnostics, aggregates, contrasts, audit
assertions, and provenance.
