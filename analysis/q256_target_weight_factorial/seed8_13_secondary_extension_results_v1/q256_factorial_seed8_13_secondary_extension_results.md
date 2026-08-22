# q256 seed8–13 secondary extension: frozen FID/KID results

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-22
- Verification Status: VERIFIED
- Version Label: q256_factorial_seed8_13_secondary_extension_results_v1

## Scope, status, and chronology

This is the complete **secondary precision extension** for seeds8–13. It is not part of the original seeds3–5 preregistration and is not a valid execution of PR #72 Cohort III.

- Training chain start: `2026-08-21T06:54:19Z`.
- PR #72 Cohort III freeze commit: `0672283a3a325b352c8c8009763f1f3222a3b2f1` at `2026-08-21T07:51:45Z`.
- The secondary chain therefore started 57 minutes 26 seconds before that freeze, and it used two A100 queues rather than PR #72's fixed five-GPU seed mapping.
- Seeds8–12 overlap the planned Cohort III seed labels, but these observed trajectories cannot be relabeled as prospective held-out confirmation. A future clean confirmation must use a newly frozen, genuinely unseen seed set or explicitly amend/reclassify #72.
- Training: **PASS**, 24/24 arms at 256.000 kimg; integrity: **PASS**, 6/6 seed audits; evaluation: **PASS**, 48/48 jobs.
- Evaluation completion: `2026-08-21T22:50:35Z`.

Frozen evaluation used FP32, 50,000 samples per job, sample seeds `0..49999`, metric seed `20260730`, KID then FID from byte-identical retained generated features, all 24 NFE1 jobs before all 24 NFE2 jobs, and `mid_t=0.821` for NFE2. Lower is better.

## Per-seed results

| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---:|:---:|---:|---:|---:|---:|
| 8 | A | 344.957551066 | 0.390897620553 | 333.159686101 | 0.378543409249 |
| 8 | B | 173.440514781 | 0.165858124730 | 318.260073380 | 0.349803773791 |
| 8 | C | 349.337689036 | 0.397376368621 | 335.115389897 | 0.383124227172 |
| 8 | D | 156.981287081 | 0.147021282285 | 318.968447093 | 0.352134116652 |
| 9 | A | 332.514349993 | 0.363081061279 | 302.893346403 | 0.325601980776 |
| 9 | B | 348.860988121 | 0.381166424680 | 289.183946238 | 0.309721469089 |
| 9 | C | 298.825110410 | 0.321411035918 | 318.929236908 | 0.347684358599 |
| 9 | D | 351.449360345 | 0.383596954952 | 278.015619260 | 0.295926650300 |
| 10 | A | 315.097211261 | 0.324865379444 | 76.987949504 | 0.064204502943 |
| 10 | B | 253.535174269 | 0.262330770428 | 45.446374258 | 0.035971660493 |
| 10 | C | 312.069778992 | 0.322797210906 | 74.950791517 | 0.061863406502 |
| 10 | D | 271.296828985 | 0.282318110621 | 44.992416548 | 0.033620670050 |
| 11 | A | 324.092760423 | 0.337702474414 | 181.773534184 | 0.187970816774 |
| 11 | B | 309.353032216 | 0.321655874515 | 63.294702242 | 0.048414545445 |
| 11 | C | 312.582957631 | 0.318875162310 | 127.700992647 | 0.120758888946 |
| 11 | D | 316.504431309 | 0.324926611489 | 145.298278769 | 0.141233310030 |
| 12 | A | 301.430096214 | 0.318811535423 | 43.519582311 | 0.031672274452 |
| 12 | B | 300.051824612 | 0.301733096867 | 83.754952770 | 0.071208145566 |
| 12 | C | 305.266316578 | 0.315909976942 | 61.463852059 | 0.046424986849 |
| 12 | D | 301.794824474 | 0.307318075008 | 68.747624668 | 0.054308337598 |
| 13 | A | 242.112350764 | 0.259896117778 | 292.739981806 | 0.316209260598 |
| 13 | B | 322.907291093 | 0.337670693701 | 227.953379495 | 0.244428475453 |
| 13 | C | 304.644019555 | 0.326871943666 | 290.271502313 | 0.312217088756 |
| 13 | D | 322.508572831 | 0.342341913774 | 280.066946456 | 0.300202772828 |

## Primary descriptive readout

The factorial primary-style readout is FID-50k@NFE1 `B−A`, but it is descriptive here because this run predates and does not follow #72.

| Group | n | Mean B−A | Median B−A | Range | Direction (−/+/0) |
|:---|---:|---:|---:|:---:|:---:|
| Complete secondary run, seeds8–13 | 6 | -25.342582 | -8.059000 | [-171.517036, 80.794940] | 4/2/0 |
| PR #72-overlap labels only, seeds8–12 (nonconfirmatory) | 5 | -46.570087 | -14.739728 | [-171.517036, 16.346638] | 4/1/0 |
| Extra sensitivity seed13 | 1 | 80.794940 | — | — | 0/1/0 |

Across all six seeds, the mean favors B over A, but the direction is 4/6 rather than uniform. Seed13 reverses strongly (`B−A=+80.794940`), while seed8 is extremely favorable (`−171.517036`). The mean is therefore heterogeneous and must not be described as universal B dominance.

## NFE1 factorial contrasts

A negative value favors the first term; interaction is `I=B−C−D+A`. Each seed is the independent unit.

| Metric | Contrast | Mean | Median | Range | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|:---:|
| FID-50k | B-A | -25.342582 | -8.059000 | [-171.517036, 80.794940] | 85.286216 | 4/2/0 |
| FID-50k | C-A | 3.753592 | 0.404394 | [-33.689240, 62.531669] | 32.050525 | 3/3/0 |
| FID-50k | B-D | -2.064413 | -2.165686 | [-17.761655, 16.459228] | 11.152138 | 4/2/0 |
| FID-50k | D-A | -23.278169 | -3.611800 | [-187.976264, 80.396222] | 90.423574 | 3/3/0 |
| FID-50k | B-C | -29.096175 | -4.222209 | [-175.897174, 50.035878] | 80.201435 | 4/2/0 |
| FID-50k | I=B-C-D+A | -5.818005 | -0.610408 | [-62.132951, 31.100867] | 31.742078 | 3/3/0 |
| KID-50k | B-A | -0.037473201 | -0.016562519 | [-0.225039496, 0.077774576] | 0.103007804 | 4/2/0 |
| KID-50k | C-A | 0.001331252 | -0.002484864 | [-0.041670025, 0.066975826] | 0.036388617 | 4/2/0 |
| KID-50k | B-D | -0.002851327 | -0.003970979 | [-0.019987340, 0.018836842] | 0.012451164 | 5/1/0 |
| KID-50k | D-A | -0.034621873 | -0.012134662 | [-0.243876338, 0.082445796] | 0.111002187 | 4/2/0 |
| KID-50k | B-C | -0.038804452 | -0.005698084 | [-0.231518244, 0.059755389] | 0.102096239 | 3/3/0 |
| KID-50k | I=B-C-D+A | -0.004182579 | 0.004837337 | [-0.071647046, 0.039239495] | 0.038186648 | 3/3/0 |

## NFE2 secondary contrasts

A negative value favors the first term; interaction is `I=B−C−D+A`. Each seed is the independent unit.

| Metric | Contrast | Mean | Median | Range | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|:---:|
| FID-50k | B-A | -33.863442 | -23.220594 | [-118.478832, 40.235370] | 53.624329 | 5/1/0 |
| FID-50k | C-A | -3.773719 | -0.040727 | [-54.072542, 17.944270] | 26.197743 | 3/3/0 |
| FID-50k | B-D | -18.032651 | -0.127208 | [-82.003577, 15.007328] | 39.598007 | 3/3/0 |
| FID-50k | D-A | -15.830791 | -19.534483 | [-36.475255, 25.228042] | 22.216333 | 5/1/0 |
| FID-50k | B-C | -30.089723 | -29.624854 | [-64.406290, 22.291101] | 32.048894 | 5/1/0 |
| FID-50k | I=B-C-D+A | -14.258932 | -3.902253 | [-49.645087, 2.491116] | 20.349430 | 5/1/0 |
| KID-50k | B-A | -0.040775696 | -0.028486239 | [-0.139556271, 0.039535871] | 0.060151185 | 5/1/0 |
| KID-50k | C-A | -0.005354881 | 0.001119861 | [-0.067211928, 0.022082378] | 0.031915735 | 3/3/0 |
| KID-50k | B-D | -0.019646298 | 0.000010324 | [-0.092818765, 0.016899808] | 0.044490654 | 3/3/0 |
| KID-50k | D-A | -0.021129398 | -0.028042312 | [-0.046737507, 0.022636063] | 0.023612419 | 5/1/0 |
| KID-50k | B-C | -0.035420814 | -0.035641671 | [-0.072344344, 0.024783159] | 0.035034887 | 5/1/0 |
| KID-50k | I=B-C-D+A | -0.014291417 | -0.007599360 | [-0.051782126, 0.004692087] | 0.021235369 | 4/2/0 |

## Integrity and interpretation guardrails

- All 24 training cells and six four-arm integrity audits passed; semantic non-finite, raw-gradient/skip mismatch, and nonpositive-denominator counts are zero.
- All 48 evaluation receipts passed with one fixed GPU exclusivity monitor and no foreign-process incident; raw metrics and sampling diagnostics re-hash to their receipts.
- Within every job, FID and KID use byte-identical retained generated-feature hashes.
- The exact job order is all NFE1 first, then all NFE2; no seed, arm, checkpoint, or metric was omitted after observation.
- No p-value or confirmatory interval is reported. With `n=6`, multiple descriptive contrasts, large seed heterogeneity, and an execution that predates #72, inferential or causal wording is not supported.
- Statistical fallacy scan (11/11): Simpson aggregation is guarded by per-seed rows; ecological, Berkson, collider, base-rate, regression-to-mean, and reverse-causality patterns are not applicable to this matrix; survivorship and look-elsewhere are guarded by complete exact matrices; garden-of-forking-paths remains a caution because this is a post hoc secondary analysis; causal language is explicitly excluded.

The CSV contains all 48 full-precision job rows and artifact hashes. The JSON contains protocol bindings, full descriptive summaries for seeds8–13 and the separately labeled seeds8–12 overlap subset, integrity hashes, and the chronology boundary.
