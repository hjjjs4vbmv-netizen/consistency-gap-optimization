# q256 formal evaluation: seed3–5 aggregate

Status: **PASS (24/24)**. This is the complete formal seed3/4/5 × arm A/B/C/D × NFE1/NFE2 result matrix. All values below come from immutable PASS receipts in the v3/v5/v6 continuation chain; v4/v5 failure evidence remains preserved and is not counted as a result.

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

Arm B has the lowest descriptive mean FID and KID at both NFE1 and NFE2. With only three seeds and substantial NFE2 dispersion, this statement is not an inferential significance claim.

## Integrity and provenance

- All 24 job receipts have `status=passed`, return code 0, and a PASS in-run GPU exclusivity monitor with no foreign-process incident.
- Within every job, FID and KID use byte-identical generated-feature SHA-256 values.
- v3 supplies seed3/4 NFE1 (8 PASS receipts); v5 supplies seed5 NFE1 and seed3/4 NFE2 (12 PASS receipts); v6 supplies seed5 NFE2 (4 PASS receipts).
- v3 completion SHA-256: `f585a88d172cfa452ce1d149a0e7d3e15a8a02df77fa1dcd87a847cc3e35455d` (`STOPPED_FOR_AUDIT`).
- v5 completion SHA-256: `01d55d1f3882471e59bf88b0dba8e97e359ecd6afb06a0451171f9110dd74d88` (`STOPPED_FOR_AUDIT`; real foreign `cudaCheck` incident preserved fail closed after its 12 PASS receipts).
- v6 completion SHA-256: `d827916918c4310d4b4075dd41259a78871c48882a601ab76a7d0123b9429103` (`PASS`).

The CSV contains all 24 full-precision metrics and per-job receipt/artifact/feature hashes. The JSON is the corresponding machine-readable aggregate with protocol, summaries, audit assertions, and source-root mapping.
