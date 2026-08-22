# q256 target geometry × denominator weighting: seed14–18 at 1024 kimg

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: VERIFIED
- Version Label: q256_seed14_18_1024k_validation_v2

## Validation status

**VERIFIED / CAUTION.** Training status is PASS (20/20 cells) and evaluation status is PASS (40/40 jobs). This is the complete post-preregistration seed14–18 secondary sensitivity matrix at 1024 kimg. CAUTION reflects five descriptive training seeds and multiple endpoint contrasts, not an artifact-integrity defect.

Frozen protocol: FP32; 50,000 generated samples per job; sample seeds `0..49999`; metric seed `20260730`; `kid50k_full` then `fid50k_full` from byte-identical retained generated Inception features; `mid_t=0.821` for NFE2; one canonical CIFAR-10 reference. Lower is better.

Training source: `458205192722883df393a8d017c26e6fa46f48f7`. Evaluator source: `d6aba02fb88e9db0993623895eb2228ed717d810`. Dataset SHA-256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`. Accepted evaluation root: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-recovery-v2/frozen-evaluation-1024k-v1`.

## Training endpoints

| Seed | Arm | Attempts | Successful steps | Processed kimg | Final loss | Final skip | Provenance |
|---:|:---:|---:|---:|---:|---:|---:|:---|
| 14 | A | 8000 | 7988 | 1024.0 | 17.17559505 | 0 | hash-adopted completed v1 armA |
| 14 | B | 8000 | 7987 | 1024.0 | 17.22553325 | 0 | strict 256→1024 budget-only resume |
| 14 | C | 8000 | 7987 | 1024.0 | 18.84788704 | 0 | strict 256→1024 budget-only resume |
| 14 | D | 8000 | 7988 | 1024.0 | 15.53938818 | 0 | strict 256→1024 budget-only resume |
| 15 | A | 8000 | 7987 | 1024.0 | 16.49700952 | 0 | hash-adopted completed v1 armA |
| 15 | B | 8000 | 7987 | 1024.0 | 16.33091593 | 0 | strict 256→1024 budget-only resume |
| 15 | C | 8000 | 7988 | 1024.0 | 17.86255550 | 0 | strict 256→1024 budget-only resume |
| 15 | D | 8000 | 7987 | 1024.0 | 15.15824211 | 0 | strict 256→1024 budget-only resume |
| 16 | A | 8000 | 7987 | 1024.0 | 17.98277557 | 0 | hash-adopted completed v1 armA |
| 16 | B | 8000 | 7987 | 1024.0 | 18.01697528 | 0 | strict 256→1024 budget-only resume |
| 16 | C | 8000 | 7987 | 1024.0 | 19.85969722 | 0 | strict 256→1024 budget-only resume |
| 16 | D | 8000 | 7986 | 1024.0 | 16.47175884 | 0 | strict 256→1024 budget-only resume |
| 17 | A | 8000 | 7987 | 1024.0 | 16.96695387 | 0 | hash-adopted completed v1 armA |
| 17 | B | 8000 | 7987 | 1024.0 | 16.70711529 | 0 | strict 256→1024 budget-only resume |
| 17 | C | 8000 | 7987 | 1024.0 | 18.41051722 | 0 | strict 256→1024 budget-only resume |
| 17 | D | 8000 | 7987 | 1024.0 | 15.31538796 | 0 | strict 256→1024 budget-only resume |
| 18 | A | 8000 | 7987 | 1024.0 | 17.49603200 | 0 | hash-adopted completed v1 armA |
| 18 | B | 8000 | 7988 | 1024.0 | 17.38211548 | 0 | strict 256→1024 budget-only resume |
| 18 | C | 8000 | 7987 | 1024.0 | 19.17701721 | 0 | strict 256→1024 budget-only resume |
| 18 | D | 8000 | 7987 | 1024.0 | 16.07064474 | 0 | strict 256→1024 budget-only resume |

Recovery boundary: v1 failure evidence remains at `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-1024k-from-v2-4582051-v1`; arm A was not retrained.

## Per-seed FID/KID results

| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---:|:---:|---:|---:|---:|---:|
| 14 | A | 8.815100439 | 0.005384948934 | 2.705757434 | 0.000912241154 |
| 14 | B | 7.548732339 | 0.004591032002 | 2.730005278 | 0.000940025100 |
| 14 | C | 8.053548226 | 0.004867319427 | 2.726511629 | 0.000870777988 |
| 14 | D | 7.361975882 | 0.004501718861 | 2.683404689 | 0.000886344339 |
| 15 | A | 8.332940207 | 0.005210136542 | 2.690475807 | 0.000924879052 |
| 15 | B | 8.260723989 | 0.005338630906 | 2.761589025 | 0.000977598001 |
| 15 | C | 8.358430358 | 0.005363982873 | 2.738119808 | 0.000971699052 |
| 15 | D | 8.400500920 | 0.005311211679 | 2.698200971 | 0.000945603526 |
| 16 | A | 13.128034915 | 0.008191012457 | 3.580937408 | 0.001471662190 |
| 16 | B | 10.915391734 | 0.006604127905 | 3.284333509 | 0.001658054842 |
| 16 | C | 12.952890184 | 0.008287585230 | 3.505843640 | 0.001692091236 |
| 16 | D | 11.635468716 | 0.007163878889 | 3.239102119 | 0.001338086939 |
| 17 | A | 7.938108896 | 0.004565353486 | 2.722471175 | 0.000830331542 |
| 17 | B | 9.185863205 | 0.005657259602 | 2.804404138 | 0.000947462132 |
| 17 | C | 8.687729216 | 0.005199380646 | 2.814952599 | 0.000947130343 |
| 17 | D | 7.991888361 | 0.004981008331 | 2.872520278 | 0.000940425668 |
| 18 | A | 7.666068394 | 0.004465182658 | 2.684382270 | 0.000892476314 |
| 18 | B | 8.600273744 | 0.005336337380 | 3.020252569 | 0.001047120373 |
| 18 | C | 7.888370708 | 0.004762479657 | 2.950713498 | 0.001013920926 |
| 18 | D | 7.911978106 | 0.004850318416 | 2.955817868 | 0.001015974990 |

## Across-seed descriptive summary

Values are mean ± sample standard deviation over seed14–18 (`n=5`).

| Arm | NFE | FID mean ± SD | KID mean ± SD |
|:---:|---:|---:|---:|
| A | 1 | 9.176050570 ± 2.251213345 | 0.005563326815 ± 0.001521772423 |
| B | 1 | 8.902197002 ± 1.271606407 | 0.005505477559 ± 0.000728479110 |
| C | 1 | 9.188193739 ± 2.126550710 | 0.005696149567 ± 0.001468958115 |
| D | 1 | 8.660362397 ± 1.703784737 | 0.005361627235 ± 0.001048380015 |
| A | 2 | 2.876804819 ± 0.393898376 | 0.001006318050 ± 0.000262659063 |
| B | 2 | 2.920116904 ± 0.233100756 | 0.001114052090 ± 0.000307032643 |
| C | 2 | 2.947228235 ± 0.324808527 | 0.001099123909 ± 0.000335540906 |
| D | 2 | 2.889809185 ± 0.226924068 | 0.001025287093 ± 0.000180822067 |

## Result synopsis

- Primary NFE1 B−A FID mean is -0.273854, favorable in 3/5 seeds; paired KID mean is -0.000057849, favorable in 2/5.
- D has the lowest cohort-mean NFE1 FID (8.660362) and KID (0.005361627); B−D FID is favorable in only 2/5 seeds.
- At NFE2, B−A reverses on average: FID +0.043312 (1/5 favorable) and KID +0.000107734 (0/5 favorable).
- The 1024 kimg extension therefore does not support a uniformly best B arm or seed-stable endpoint factorization. It remains descriptive secondary sensitivity evidence.

## Primary-readout factorial contrasts

These reuse the originally frozen factorial definitions; seed14–18 and the 1024 kimg extension are secondary sensitivity evidence. A negative contrast favors its first term. The interaction is `I=B−C−D+A`.

| Metric | Seed | B−A | C−A | D−A | B−D | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE1 (primary) | 14 | -1.266368 | -0.761552 | -1.453125 | 0.186756 | -0.504816 | 0.948309 |
| FID-50k@NFE1 (primary) | 15 | -0.072216 | 0.025490 | 0.067561 | -0.139777 | -0.097706 | -0.165267 |
| FID-50k@NFE1 (primary) | 16 | -2.212643 | -0.175145 | -1.492566 | -0.720077 | -2.037498 | -0.544932 |
| FID-50k@NFE1 (primary) | 17 | 1.247754 | 0.749620 | 0.053779 | 1.193975 | 0.498134 | 0.444355 |
| FID-50k@NFE1 (primary) | 18 | 0.934205 | 0.222302 | 0.245910 | 0.688296 | 0.711903 | 0.465993 |
| KID-50k@NFE1 | 14 | -0.000793917 | -0.000517630 | -0.000883230 | 0.000089313 | -0.000276287 | 0.000606943 |
| KID-50k@NFE1 | 15 | 0.000128494 | 0.000153846 | 0.000101075 | 0.000027419 | -0.000025352 | -0.000126427 |
| KID-50k@NFE1 | 16 | -0.001586885 | 0.000096573 | -0.001027134 | -0.000559751 | -0.001683457 | -0.000656324 |
| KID-50k@NFE1 | 17 | 0.001091906 | 0.000634027 | 0.000415655 | 0.000676251 | 0.000457879 | 0.000042224 |
| KID-50k@NFE1 | 18 | 0.000871155 | 0.000297297 | 0.000385136 | 0.000486019 | 0.000573858 | 0.000188722 |

Cross-seed summaries use the five training seeds as independent units. `Range` is [minimum, maximum], `Span` is maximum minus minimum, and SD is sample SD.

| Metric | Contrast | Mean | Median | Range | Span | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|---:|:---:|
| FID-50k@NFE1 (primary) | B-A | -0.273854 | -0.072216 | [-2.212643, 1.247754] | 3.460397 | 1.462818 | 3/2/0 |
| FID-50k@NFE1 (primary) | C-A | 0.012143 | 0.025490 | [-0.761552, 0.749620] | 1.511173 | 0.552698 | 2/3/0 |
| FID-50k@NFE1 (primary) | D-A | -0.515688 | 0.053779 | [-1.492566, 0.245910] | 1.738476 | 0.877152 | 2/3/0 |
| FID-50k@NFE1 (primary) | B-D | 0.241835 | 0.186756 | [-0.720077, 1.193975] | 1.914052 | 0.738213 | 2/3/0 |
| FID-50k@NFE1 (primary) | B-C | -0.285997 | -0.097706 | [-2.037498, 0.711903] | 2.749401 | 1.091073 | 3/2/0 |
| FID-50k@NFE1 (primary) | I=B-C-D+A | 0.229691 | 0.444355 | [-0.544932, 0.948309] | 1.493241 | 0.586166 | 2/3/0 |
| KID-50k@NFE1 | B-A | -0.000057849 | 0.000128494 | [-0.001586885, 0.001091906] | 0.002678791 | 0.001129099 | 2/3/0 |
| KID-50k@NFE1 | C-A | 0.000132823 | 0.000153846 | [-0.000517630, 0.000634027] | 0.001151657 | 0.000419256 | 1/4/0 |
| KID-50k@NFE1 | D-A | -0.000201700 | 0.000101075 | [-0.001027134, 0.000415655] | 0.001442788 | 0.000700535 | 2/3/0 |
| KID-50k@NFE1 | B-D | 0.000143850 | 0.000089313 | [-0.000559751, 0.000676251] | 0.001236002 | 0.000477524 | 1/4/0 |
| KID-50k@NFE1 | B-C | -0.000190672 | -0.000025352 | [-0.001683457, 0.000573858] | 0.002257315 | 0.000903907 | 3/2/0 |
| KID-50k@NFE1 | I=B-C-D+A | 0.000011028 | 0.000042224 | [-0.000656324, 0.000606943] | 0.001263266 | 0.000461496 | 2/3/0 |

## Secondary NFE2 contrasts

These reuse the originally frozen factorial definitions; seed14–18 and the 1024 kimg extension are secondary sensitivity evidence. A negative contrast favors its first term. The interaction is `I=B−C−D+A`.

| Metric | Seed | B−A | C−A | D−A | B−D | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE2 | 14 | 0.024248 | 0.020754 | -0.022353 | 0.046601 | 0.003494 | 0.025846 |
| FID-50k@NFE2 | 15 | 0.071113 | 0.047644 | 0.007725 | 0.063388 | 0.023469 | 0.015744 |
| FID-50k@NFE2 | 16 | -0.296604 | -0.075094 | -0.341835 | 0.045231 | -0.221510 | 0.120325 |
| FID-50k@NFE2 | 17 | 0.081933 | 0.092481 | 0.150049 | -0.068116 | -0.010548 | -0.160598 |
| FID-50k@NFE2 | 18 | 0.335870 | 0.266331 | 0.271436 | 0.064435 | 0.069539 | -0.201897 |
| KID-50k@NFE2 | 14 | 0.000027784 | -0.000041463 | -0.000025897 | 0.000053681 | 0.000069247 | 0.000095144 |
| KID-50k@NFE2 | 15 | 0.000052719 | 0.000046820 | 0.000020724 | 0.000031994 | 0.000005899 | -0.000014826 |
| KID-50k@NFE2 | 16 | 0.000186393 | 0.000220429 | -0.000133575 | 0.000319968 | -0.000034036 | 0.000099539 |
| KID-50k@NFE2 | 17 | 0.000117131 | 0.000116799 | 0.000110094 | 0.000007036 | 0.000000332 | -0.000109762 |
| KID-50k@NFE2 | 18 | 0.000154644 | 0.000121445 | 0.000123499 | 0.000031145 | 0.000033199 | -0.000090299 |

Cross-seed summaries use the five training seeds as independent units. `Range` is [minimum, maximum], `Span` is maximum minus minimum, and SD is sample SD.

| Metric | Contrast | Mean | Median | Range | Span | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|---:|:---:|
| FID-50k@NFE2 | B-A | 0.043312 | 0.071113 | [-0.296604, 0.335870] | 0.632474 | 0.225700 | 1/4/0 |
| FID-50k@NFE2 | C-A | 0.070423 | 0.047644 | [-0.075094, 0.266331] | 0.341425 | 0.125527 | 1/4/0 |
| FID-50k@NFE2 | D-A | 0.013004 | 0.007725 | [-0.341835, 0.271436] | 0.613271 | 0.230628 | 2/3/0 |
| FID-50k@NFE2 | B-D | 0.030308 | 0.046601 | [-0.068116, 0.064435] | 0.132551 | 0.055755 | 1/4/0 |
| FID-50k@NFE2 | B-C | -0.027111 | 0.003494 | [-0.221510, 0.069539] | 0.291049 | 0.112807 | 2/3/0 |
| FID-50k@NFE2 | I=B-C-D+A | -0.040116 | 0.015744 | [-0.201897, 0.120325] | 0.322222 | 0.135924 | 2/3/0 |
| KID-50k@NFE2 | B-A | 0.000107734 | 0.000117131 | [0.000027784, 0.000186393] | 0.000158609 | 0.000066886 | 0/5/0 |
| KID-50k@NFE2 | C-A | 0.000092806 | 0.000116799 | [-0.000041463, 0.000220429] | 0.000261892 | 0.000097245 | 1/4/0 |
| KID-50k@NFE2 | D-A | 0.000018969 | 0.000020724 | [-0.000133575, 0.000123499] | 0.000257074 | 0.000105494 | 2/3/0 |
| KID-50k@NFE2 | B-D | 0.000088765 | 0.000031994 | [0.000007036, 0.000319968] | 0.000312931 | 0.000130296 | 0/5/0 |
| KID-50k@NFE2 | B-C | 0.000014928 | 0.000005899 | [-0.000034036, 0.000069247] | 0.000103284 | 0.000038654 | 1/4/0 |
| KID-50k@NFE2 | I=B-C-D+A | -0.000004041 | -0.000014826 | [-0.000109762, 0.000099539] | 0.000209301 | 0.000099121 | 3/2/0 |

## Directional interpretation boundary

The primary endpoint is FID-50k@NFE1. Direction consistency is descriptive over five seeds. The report does not reinterpret the frozen arms, claim a causal percentage decomposition, turn endpoint differences into an optimizer-mechanism claim, or use NFE2 to overwrite the primary endpoint.

Primary direction counts (negative/positive/zero): FID B-A=3/2/0; FID C-A=2/3/0; FID D-A=2/3/0; FID B-D=2/3/0; FID B-C=3/2/0; FID I=B-C-D+A=2/3/0; KID B-A=2/3/0; KID C-A=1/4/0; KID D-A=2/3/0; KID B-D=1/4/0; KID B-C=3/2/0; KID I=B-C-D+A=2/3/0.

## Integrity and provenance

- All 40 job receipts have `status=PASS`; all five workers have exact 8-job `WORKER_PASS` completions.
- Every receipt is SHA-bound by its worker completion; raw metric records were re-hashed during collection.
- Within every job, FID and KID use byte-identical retained generated-feature SHA-256 values.
- Seed-to-GPU mapping remained seed14→GPU0 through seed18→GPU4.
- The initial 1024 kimg v1 attempt completed all five arm A cells, then failed only in the post-training state verifier because `PYTHONPATH` omitted `torch_utils`; its failure receipts remain preserved.
- Recovery-v2 did not retrain arm A. It copied each completed A cell into a new root only after full-file SHA-256 manifests matched, then trained B→C→D and ran evaluation.
- All 20 final cells have 8,000 attempted iterations, exactly 1,024.0 processed kimg, finite final loss, and `step_skipped=0` in the final row.

## Statistical fallacy scan

Coverage: **11/11 checked**. Simpson/ecological/Berkson/collider/base-rate/regression-to-mean/reverse-causality concerns are not implicated by this paired factorial summary. There was no attrition: every authorized seed, arm, and job is retained. Look-elsewhere and multiple-comparison risk remains because several endpoints and contrasts are shown without confirmatory multiplicity correction. Garden-of-forking-paths risk is reduced by frozen arms and evaluation settings, but this 1024 kimg extension is post-preregistration and remains descriptive. The controlled interventions support within-protocol endpoint contrasts, not universal optimizer-mechanism or causal-percentage claims.

The CSV contains all 40 full-precision job rows and artifact hashes. The JSON contains the corresponding machine-readable protocol, descriptive summaries, contrasts, and completion bindings.
