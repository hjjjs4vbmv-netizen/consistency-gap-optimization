# q256 replay learning-curve FID/KID final summary

- Status: **PASS**
- Evaluation jobs: **168/168**
- Training checkpoints joined: **84/84**
- Seeds: 3, 4, 5
- Arms: A, B, C, D
- Budgets: 256, 384, 512, 640, 768, 896, 1024 kimg
- NFE: 1 and 2 (`mid_t=0.821` for NFE=2)
- Metrics: FID-50k and KID-50k
- Sampling: FP32, 50,000 samples, seeds 0-49999, metric seed 20260730
- Training audit: 3/3 PASS; 84 inventory rows; 12/12 endpoint parity checks BITWISE_EQUIVALENT
- KID/FID generated-feature identity: 168/168 PASS

Files:

- `evaluation_results.csv`: one row per seed × arm × budget × NFE.
- `factorial_contrasts.csv`: preregistered seed-level factorial contrasts.
- `paired_contrast_summary.csv`: paired mean, sample SD, range, and direction counts.
- `final_summary.json`: machine-readable completion and provenance receipt.

The server archive preserves the original package bytes and
`SERVER_PACKAGE_SHA256SUMS.txt`. CSV files committed to Git are normalized from
CRLF to LF for repository review; `SHA256SUMS.txt` identifies these normalized
PR copies.
