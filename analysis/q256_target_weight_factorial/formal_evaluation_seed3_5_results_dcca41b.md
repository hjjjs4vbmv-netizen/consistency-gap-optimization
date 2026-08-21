# q256 target geometry × denominator weighting: seed3–5 formal evaluation

Verification status: **VERIFIED**. Evaluation status: **PASS (24/24)**. This is the complete formal seed3/4/5 × arm A/B/C/D × NFE1/NFE2 result matrix. All values below come from immutable PASS receipts in the v3/v5/v6 continuation chain; v4/v5 failure evidence remains preserved and is not counted as a result.

Frozen protocol: FP32; 50,000 samples per job; sample seeds `0..49999`; metric seed `20260730`; KID then FID from byte-identical retained generated Inception features; all NFE1 jobs before all NFE2 jobs; `mid_t=0.821` for NFE2; one CIFAR-10 reference. Training source: `dcca41b19e7c45512b5fbe98776520396a1bf9ac`. Formal GPU0: `GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859`.

## Per-seed results

| Seed | Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---:|:---:|---:|---:|---:|---:|
| 3 | A | 331.884554744 | 0.353531374637 | 238.774319609 | 0.252761695946 |
| 3 | B | 310.507547139 | 0.313636837573 | 100.497569065 | 0.088206853664 |
| 3 | C | 318.174614952 | 0.343721830453 | 294.103416689 | 0.312204178188 |
| 3 | D | 329.234318292 | 0.345016475223 | 194.670179467 | 0.202755819062 |
| 4 | A | 310.234854753 | 0.315592510165 | 113.816732009 | 0.103910684284 |
| 4 | B | 306.773381693 | 0.310122785053 | 112.892926805 | 0.102270812115 |
| 4 | C | 304.566281428 | 0.307217715180 | 103.089535592 | 0.091746502623 |
| 4 | D | 324.722492834 | 0.339143093048 | 176.960718669 | 0.180121738521 |
| 5 | A | 318.638529753 | 0.329434699442 | 75.266429038 | 0.062150206999 |
| 5 | B | 308.854892082 | 0.326125451789 | 70.236021799 | 0.056118489655 |
| 5 | C | 308.414344315 | 0.325736081314 | 70.485902851 | 0.056507776707 |
| 5 | D | 315.200350185 | 0.329756152860 | 75.407868983 | 0.062109354377 |

## Across-seed descriptive summary

Values are mean ± sample standard deviation over seed3–5 (`n=3`). Lower is better. These are descriptive summaries only; no protocol or selection decision was changed in response to them.

| Arm | NFE | FID mean ± SD | KID mean ± SD |
|:---:|---:|---:|---:|
| A | 1 | 320.252646417 ± 10.914733477 | 0.332852861415 ± 0.019199016454 |
| B | 1 | 308.711940305 ± 1.871182595 | 0.316628358138 ± 0.008410305116 |
| C | 1 | 310.385080231 ± 7.014950845 | 0.325558542316 ± 0.018252705225 |
| D | 1 | 323.052387103 ± 7.164496149 | 0.337971907044 ± 0.007697279854 |
| A | 2 | 142.619160219 ± 85.474513587 | 0.139607529077 ± 0.100194228754 |
| B | 2 | 94.542172556 ± 21.943175405 | 0.082198718478 ± 0.023655495840 |
| C | 2 | 155.892951711 ± 120.798794237 | 0.153486152506 ± 0.138578500167 |
| D | 2 | 149.012922373 ± 64.355917862 | 0.148328970653 ± 0.075521136602 |

## Preregistered NFE1 contrasts

These are the contrasts frozen before formal training. Lower FID/KID is better, so a negative contrast favors its first term. The interaction is `I = B−C−D+A`.

| Metric | Seed | C−A | B−D | D−A | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE1 (primary) | 3 | -13.709940 | -18.726771 | -2.650236 | -7.667068 | -5.016831 |
| FID-50k@NFE1 (primary) | 4 | -5.668573 | -17.949111 | 14.487638 | 2.207100 | -12.280538 |
| FID-50k@NFE1 (primary) | 5 | -10.224185 | -6.345458 | -3.438180 | 0.440548 | 3.878727 |
| KID-50k@NFE1 | 3 | -0.009809544 | -0.031379638 | -0.008514899 | -0.030084993 | -0.021570093 |
| KID-50k@NFE1 | 4 | -0.008374795 | -0.029020308 | 0.023550583 | 0.002905070 | -0.020645513 |
| KID-50k@NFE1 | 5 | -0.003698618 | -0.003630701 | 0.000321453 | 0.000389370 | 0.000067917 |

Cross-seed summaries use the three training seeds as the independent units. `Range` is `[minimum, maximum]`, `Span` is maximum minus minimum, and SD is the sample SD. All quantities were calculated from the full-precision raw cells and are rounded only for display.

| Metric | Contrast | Mean | Median | Range | Span | SD |
|:---|:---|---:|---:|:---:|---:|---:|
| FID-50k@NFE1 (primary) | C−A | -9.867566 | -10.224185 | [-13.709940, -5.668573] | 8.041366 | 4.032527 |
| FID-50k@NFE1 (primary) | B−D | -14.340447 | -17.949111 | [-18.726771, -6.345458] | 12.381313 | 6.934773 |
| FID-50k@NFE1 (primary) | D−A | 2.799741 | -2.650236 | [-3.438180, 14.487638] | 17.925818 | 10.129680 |
| FID-50k@NFE1 (primary) | B−C | -1.673140 | 0.440548 | [-7.667068, 2.207100] | 9.874168 | 5.265506 |
| FID-50k@NFE1 (primary) | I | -4.472881 | -5.016831 | [-12.280538, 3.878727] | 16.159265 | 8.093354 |
| KID-50k@NFE1 | C−A | -0.007294319 | -0.008374795 | [-0.009809544, -0.003698618] | 0.006110926 | 0.003195532 |
| KID-50k@NFE1 | B−D | -0.021343549 | -0.029020308 | [-0.031379638, -0.003630701] | 0.027748937 | 0.015385069 |
| KID-50k@NFE1 | D−A | 0.005119046 | 0.000321453 | [-0.008514899, 0.023550583] | 0.032065482 | 0.016562351 |
| KID-50k@NFE1 | B−C | -0.008930184 | 0.000389370 | [-0.030084993, 0.002905070] | 0.032990063 | 0.018363731 |
| KID-50k@NFE1 | I | -0.014049230 | -0.020645513 | [-0.021570093, 0.000067917] | 0.021638011 | 0.012234545 |

### Preregistered primary interpretation

At NFE1, fresh native B is better than A for both FID and KID in all three seeds. More specifically, both target contrasts (`C−A` and `B−D`) are negative for FID and KID in every seed, whereas both denominator contrasts (`D−A` and `B−C`) have mixed signs and the interaction sign is not stable across seeds. This pattern is consistent with the preregistered target-geometry-dominant branch. It is descriptive endpoint evidence at `n=3`, not a causal percentage decomposition: it does not establish that any percentage of the B−A benefit is explained by target geometry, and it does not establish that denominator weighting is irrelevant.

## Secondary NFE2 contrasts: strongly heterogeneous

NFE2 is a preregistered secondary readout and is reported separately. Its seed-level contrast directions and magnitudes are strongly heterogeneous and do not overwrite the NFE1 primary interpretation.

| Metric | Seed | C−A | B−D | D−A | B−C | I |
|:---|---:|---:|---:|---:|---:|---:|
| FID-50k@NFE2 | 3 | 55.329097 | -94.172610 | -44.104140 | -193.605848 | -149.501707 |
| FID-50k@NFE2 | 4 | -10.727196 | -64.067792 | 63.143987 | 9.803391 | -53.340595 |
| FID-50k@NFE2 | 5 | -4.780526 | -5.171847 | 0.141440 | -0.249881 | -0.391321 |
| KID-50k@NFE2 | 3 | 0.059442482 | -0.114548965 | -0.050005877 | -0.223997325 | -0.173991448 |
| KID-50k@NFE2 | 4 | -0.012164182 | -0.077850926 | 0.076211054 | 0.010524309 | -0.065686745 |
| KID-50k@NFE2 | 5 | -0.005642430 | -0.005990865 | -0.000040853 | -0.000389287 | -0.000348434 |

The seed3 NFE2 trajectory is the clearest warning against dismissing denominator weighting or interaction: relative to C, B improves FID by 193.606 and KID by 0.223997, while the target-only contrast `C−A` reverses direction. NFE2 therefore shows that denominator weighting or interaction can matter strongly for some trajectories or readout modes.

## Same-state/same-batch manipulation check

The frozen `q=256`, `g=1.10`, `c=0` objective is clip-free in the focused deterministic multi-parameter fixture. With identical initial parameters, batch, and RNG, the added required correctness test verifies `G_D=(1/1.10)G_A` and `G_B=(1/1.10)G_C`, as implied by the shared A/D and B/C stop-gradient targets and their denominator-only change. This is an objective manipulation check, not a mechanism experiment.

The recorded PyTorch 2.3.0 CPU check used `rtol=2×10⁻⁶` and `atol=10⁻⁶`. For `G_D−G_A/1.10`, the maximum absolute gradient residual was `7.32421875×10⁻⁴` and the maximum elementwise relative residual was `3.50060333×10⁻⁷`; for `G_B−G_C/1.10`, they were `7.32421875×10⁻⁴` and `3.81879659×10⁻⁷`. The maximum absolute realized-denominator scaling residual was `6.05359674×10⁻⁹` for both pairs, with no denominator scale-to-zero event. The test is now a mandatory identity in the frozen correctness gate; the complete machine-readable record is `gradient_scaling_manipulation_check.json`.

## Integrity and provenance

- All 24 job receipts have `status=passed`, return code 0, and a PASS in-run GPU exclusivity monitor with no foreign-process incident.
- Within every job, FID and KID use byte-identical generated-feature SHA-256 values.
- v3 supplies seed3/4 NFE1 (8 PASS receipts); v5 supplies seed5 NFE1 and seed3/4 NFE2 (12 PASS receipts); v6 supplies seed5 NFE2 (4 PASS receipts).
- v3 completion SHA-256: `f585a88d172cfa452ce1d149a0e7d3e15a8a02df77fa1dcd87a847cc3e35455d` (`STOPPED_FOR_AUDIT`).
- v5 completion SHA-256: `01d55d1f3882471e59bf88b0dba8e97e359ecd6afb06a0451171f9110dd74d88` (`STOPPED_FOR_AUDIT`; real foreign `cudaCheck` incident preserved fail closed after its 12 PASS receipts).
- v6 completion SHA-256: `d827916918c4310d4b4075dd41259a78871c48882a601ab76a7d0123b9429103` (`PASS`).

## Comparability deviations and limitations

- Seed3/C and seed4/C were resumed after the `/dev/shm` infrastructure incident through the prevalidated deterministic exact-resume path. The incident evidence and pre-resume artifacts remain preserved; neither run was replaced or selected around.
- Seed4/B completed 1991 accepted optimizer updates, whereas seed4 A/C/D completed 1990 each. The common endpoint remains 2000 attempts and 256.000 kimg, but the accepted-update difference is a formal within-seed comparability deviation. In particular, the small seed4 B−C differences at NFE1 should not be overinterpreted.
- These deviations are disclosed rather than rerun away. All 12 fresh runs and all 24 formal evaluation receipts are retained, and no additional q256 full-training expansion is introduced by this report revision.

The CSV contains all 24 full-precision metrics and per-job receipt/artifact/feature hashes. The JSON is the corresponding machine-readable aggregate with protocol, summaries, audit assertions, and source-root mapping.
