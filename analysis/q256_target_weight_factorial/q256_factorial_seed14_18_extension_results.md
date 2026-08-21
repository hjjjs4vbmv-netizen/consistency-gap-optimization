# q256 target geometry × denominator weighting: seed14–18 formal FID/KID evaluation

Verification status: **VERIFIED**. Evaluation status: **PASS (40/40)**. This is the complete seed14–18 × arm A/B/C/D × NFE1/NFE2 matrix.

Frozen protocol: FP32; 50,000 generated samples per job; sample seeds `0..49999`; metric seed `20260730`; `kid50k_full` then `fid50k_full` from byte-identical retained generated Inception features; `mid_t=0.821` for NFE2; one canonical CIFAR-10 reference. Lower is better.

Training source: `dcca41b19e7c45512b5fbe98776520396a1bf9ac`. Evaluator source: `d6aba02fb88e9db0993623895eb2228ed717d810`. Dataset SHA-256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`. Accepted evaluation root: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260821/secondary-precision-extension/seed14-18-dcca41b-v2/frozen-evaluation-seed14-18-v4-native`.

## Per-seed results

| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---:|:---:|---:|---:|---:|---:|
| 14 | A | 327.211722434 | 0.342735161454 | 194.929230016 | 0.202059017042 |
| 14 | B | 301.942410211 | 0.318241725393 | 71.933390561 | 0.057306999492 |
| 14 | C | 313.976463583 | 0.320762307965 | 131.679092414 | 0.123666033819 |
| 14 | D | 299.180569710 | 0.314766826907 | 69.851849556 | 0.055435303288 |
| 15 | A | 325.420901561 | 0.336770706276 | 75.227439252 | 0.061300780703 |
| 15 | B | 322.952536005 | 0.338615603624 | 73.751399902 | 0.059115530556 |
| 15 | C | 315.983067819 | 0.330859890836 | 66.689306473 | 0.051828274319 |
| 15 | D | 327.220002025 | 0.340762149364 | 76.643809894 | 0.062589969817 |
| 16 | A | 358.907031896 | 0.417230531461 | 340.104578534 | 0.392917612230 |
| 16 | B | 288.565018312 | 0.314872327047 | 325.684344209 | 0.360718308301 |
| 16 | C | 350.195710245 | 0.404354477357 | 337.007341744 | 0.386065754957 |
| 16 | D | 329.199492976 | 0.369890510586 | 332.351497437 | 0.375533395255 |
| 17 | A | 312.440682599 | 0.319743362800 | 139.360734432 | 0.134244111659 |
| 17 | B | 333.534467361 | 0.354548958346 | 227.122150957 | 0.240705763378 |
| 17 | C | 329.049459192 | 0.346376561582 | 195.920965482 | 0.205064243816 |
| 17 | D | 264.836480481 | 0.275782028311 | 45.613408602 | 0.033165924267 |
| 18 | A | 303.569637201 | 0.312757012102 | 62.828253415 | 0.049003935373 |
| 18 | B | 288.004041386 | 0.306007129377 | 64.187735601 | 0.050909348549 |
| 18 | C | 301.411651014 | 0.319474488306 | 47.080525641 | 0.032909418716 |
| 18 | D | 276.915390388 | 0.291967796992 | 44.777437516 | 0.033795683599 |

## Across-seed descriptive summary

Values are mean ± sample standard deviation over seed14–18 (`n=5`).

| Arm | NFE | FID mean ± SD | KID mean ± SD |
|:---:|---:|---:|---:|
| A | 1 | 325.509995138 ± 21.038370388 | 0.345847354819 ± 0.041725347165 |
| B | 1 | 306.999694655 ± 20.523374300 | 0.326457148757 ± 0.019728674300 |
| C | 1 | 322.123270371 ± 18.500524970 | 0.344365545209 ± 0.035220009900 |
| D | 1 | 299.470387116 ± 28.992155073 | 0.318633862432 ± 0.037680292461 |
| A | 2 | 162.490047130 ± 112.562675117 | 0.167905091401 ± 0.140028703029 |
| B | 2 | 152.535804246 ± 118.377207160 | 0.153751190055 ± 0.140739404720 |
| C | 2 | 155.675446351 ± 117.034367809 | 0.159906745126 ± 0.143448390841 |
| D | 2 | 113.847600601 ± 122.973872704 | 0.112104055245 ± 0.147835583398 |

## Primary-readout factorial contrasts

These use the factorial definitions originally frozen for the preregistered study. Seed14–18 themselves are post-preregistration secondary sensitivity seeds. A negative contrast favors its first term. The interaction is `I=B−C−D+A`.

| Metric | Seed | C−A | B−D | D−A | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE1 (primary) | 14 | -13.235259 | 2.761841 | -28.031153 | -12.034053 | 15.997099 |
| FID-50k@NFE1 (primary) | 15 | -9.437834 | -4.267466 | 1.799100 | 6.969468 | 5.170368 |
| FID-50k@NFE1 (primary) | 16 | -8.711322 | -40.634475 | -29.707539 | -61.630692 | -31.923153 |
| FID-50k@NFE1 (primary) | 17 | 16.608777 | 68.697987 | -47.604202 | 4.485008 | 52.089210 |
| FID-50k@NFE1 (primary) | 18 | -2.157986 | 11.088651 | -26.654247 | -13.407610 | 13.246637 |
| KID-50k@NFE1 | 14 | -0.021972853 | 0.003474898 | -0.027968335 | -0.002520583 | 0.025447752 |
| KID-50k@NFE1 | 15 | -0.005910815 | -0.002146546 | 0.003991443 | 0.007755713 | 0.003764270 |
| KID-50k@NFE1 | 16 | -0.012876054 | -0.055018184 | -0.047340021 | -0.089482150 | -0.042142129 |
| KID-50k@NFE1 | 17 | 0.026633199 | 0.078766930 | -0.043961334 | 0.008172397 | 0.052133731 |
| KID-50k@NFE1 | 18 | 0.006717476 | 0.014039332 | -0.020789215 | -0.013467359 | 0.007321856 |

Cross-seed summaries use the five training seeds as independent units. `Range` is [minimum, maximum], `Span` is maximum minus minimum, and SD is sample SD.

| Metric | Contrast | Mean | Median | Range | Span | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|---:|:---:|
| FID-50k@NFE1 (primary) | C-A | -3.386725 | -8.711322 | [-13.235259, 16.608777] | 29.844035 | 11.866874 | 4/1/0 |
| FID-50k@NFE1 (primary) | B-D | 7.529308 | 2.761841 | [-40.634475, 68.697987] | 109.332462 | 39.483917 | 2/3/0 |
| FID-50k@NFE1 (primary) | D-A | -26.039608 | -28.031153 | [-47.604202, 1.799100] | 49.403303 | 17.732901 | 4/1/0 |
| FID-50k@NFE1 (primary) | B-C | -15.123576 | -12.034053 | [-61.630692, 6.969468] | 68.600160 | 27.604351 | 3/2/0 |
| FID-50k@NFE1 (primary) | I=B-C-D+A | 10.916032 | 13.246637 | [-31.923153, 52.089210] | 84.012363 | 29.977859 | 1/4/0 |
| KID-50k@NFE1 | C-A | -0.001481810 | -0.005910815 | [-0.021972853, 0.026633199] | 0.048606052 | 0.018887891 | 3/2/0 |
| KID-50k@NFE1 | B-D | 0.007823286 | 0.003474898 | [-0.055018184, 0.078766930] | 0.133785114 | 0.047799049 | 2/3/0 |
| KID-50k@NFE1 | D-A | -0.027213492 | -0.027968335 | [-0.047340021, 0.003991443] | 0.051331464 | 0.020622539 | 4/1/0 |
| KID-50k@NFE1 | B-C | -0.017908396 | -0.002520583 | [-0.089482150, 0.008172397] | 0.097654547 | 0.040982204 | 3/2/0 |
| KID-50k@NFE1 | I=B-C-D+A | 0.009305096 | 0.007321856 | [-0.042142129, 0.052133731] | 0.094275861 | 0.034555493 | 1/4/0 |

## Secondary NFE2 contrasts

A negative contrast favors its first term. The interaction is `I=B−C−D+A`.

| Metric | Seed | C−A | B−D | D−A | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE2 | 14 | -63.250138 | 2.081541 | -125.077380 | -59.745702 | 65.331679 |
| FID-50k@NFE2 | 15 | -8.538133 | -2.892410 | 1.416371 | 7.062093 | 5.645723 |
| FID-50k@NFE2 | 16 | -3.097237 | -6.667153 | -7.753081 | -11.322998 | -3.569916 |
| FID-50k@NFE2 | 17 | 56.560231 | 181.508742 | -93.747326 | 31.201185 | 124.948511 |
| FID-50k@NFE2 | 18 | -15.747728 | 19.410298 | -18.050816 | 17.107210 | 35.158026 |
| KID-50k@NFE2 | 14 | -0.078392983 | 0.001871696 | -0.146623714 | -0.066359034 | 0.080264679 |
| KID-50k@NFE2 | 15 | -0.009472506 | -0.003474439 | 0.001289189 | 0.007287256 | 0.005998067 |
| KID-50k@NFE2 | 16 | -0.006851857 | -0.014815087 | -0.017384217 | -0.025347447 | -0.007963230 |
| KID-50k@NFE2 | 17 | 0.070820132 | 0.207539839 | -0.101078187 | 0.035641520 | 0.136719707 |
| KID-50k@NFE2 | 18 | -0.016094517 | 0.017113665 | -0.015208252 | 0.017999930 | 0.033208182 |

Cross-seed summaries use the five training seeds as independent units. `Range` is [minimum, maximum], `Span` is maximum minus minimum, and SD is sample SD.

| Metric | Contrast | Mean | Median | Range | Span | SD | Direction (−/+/0) |
|:---|:---|---:|---:|:---:|---:|---:|:---:|
| FID-50k@NFE2 | C-A | -6.814601 | -8.538133 | [-63.250138, 56.560231] | 119.810369 | 42.713959 | 4/1/0 |
| FID-50k@NFE2 | B-D | 38.688204 | 2.081541 | [-6.667153, 181.508742] | 188.175896 | 80.460312 | 2/3/0 |
| FID-50k@NFE2 | D-A | -48.642447 | -18.050816 | [-125.077380, 1.416371] | 126.493751 | 56.987767 | 4/1/0 |
| FID-50k@NFE2 | B-C | -3.139642 | 7.062093 | [-59.745702, 31.201185] | 90.946887 | 35.229661 | 2/3/0 |
| FID-50k@NFE2 | I=B-C-D+A | 45.502804 | 35.158026 | [-3.569916, 124.948511] | 128.518428 | 51.982059 | 1/4/0 |
| KID-50k@NFE2 | C-A | -0.007998346 | -0.009472506 | [-0.078392983, 0.070820132] | 0.149213115 | 0.053001870 | 4/1/0 |
| KID-50k@NFE2 | B-D | 0.041647135 | 0.001871696 | [-0.014815087, 0.207539839] | 0.222354926 | 0.093445579 | 2/3/0 |
| KID-50k@NFE2 | D-A | -0.055801036 | -0.017384217 | [-0.146623714, 0.001289189] | 0.147912903 | 0.064578826 | 4/1/0 |
| KID-50k@NFE2 | B-C | -0.006155555 | 0.007287256 | [-0.066359034, 0.035641520] | 0.102000554 | 0.040323627 | 2/3/0 |
| KID-50k@NFE2 | I=B-C-D+A | 0.049645481 | 0.033208182 | [-0.007963230, 0.136719707] | 0.144682937 | 0.059189318 | 1/4/0 |

## Directional interpretation boundary

The primary endpoint is FID-50k@NFE1. Direction consistency is reported descriptively over five seeds; the report does not reinterpret the frozen factorial arms, claim a causal percentage decomposition, or use the secondary NFE2 readout to overwrite the primary endpoint.

Primary direction counts (negative/positive/zero): FID C-A=4/1/0; FID B-D=2/3/0; FID D-A=4/1/0; FID B-C=3/2/0; FID I=B-C-D+A=1/4/0; KID C-A=3/2/0; KID B-D=2/3/0; KID D-A=4/1/0; KID B-C=3/2/0; KID I=B-C-D+A=1/4/0.

## Integrity and provenance

- All 40 job receipts have `status=PASS`; all five workers have exact 8-job `WORKER_PASS` completions.
- Every receipt is SHA-bound by its worker completion; raw metric records were re-hashed during collection.
- Within every job, FID and KID use byte-identical retained generated-feature SHA-256 values.
- Seed-to-GPU mapping remained seed14→GPU0 through seed18→GPU4.
- Earlier v1/v2/v3 orchestration attempts and their failure evidence remain preserved and are not counted as results; this report accepts only the fresh v4 native-runtime PASS chain.

The CSV contains all 40 full-precision job rows and artifact hashes. The JSON contains the corresponding machine-readable protocol, descriptive summaries, contrasts, and completion bindings.
