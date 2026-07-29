# Staged evaluation smoke record

**Protocol:** `staged-checkpoint-evaluation-v1`

**Execution date:** 2026-07-30
**Evidence class:** quick — 5k-sample screening proxy, not a formal 50k benchmark

## Evaluated inputs

| Field | Value |
| --- | --- |
| Evaluation Git commit | `220e2adfc53a5ec123498a61fabcd9995745006c` |
| Checkpoint ID | `official_edm_vp` |
| Checkpoint SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Dataset SHA-256 | `9fd64620e37bfc0c995535fa52701c9641bcd07635008bfda0c9fbddde1a4ed6` |
| Precision / topology | FP32 / one NVIDIA A100-PCIE-40GB |
| Generation seeds | ascending `0-4999` for every cell |
| Metric seed | `20260730` |

## Completed quick cells

| NFE | `mid_t` | KID-5k | FID-5k |
| ---: | --- | ---: | ---: |
| 1 | `[]` | 0.4308025538921356 | 388.1013354336475 |
| 2 | `[0.821]` | 0.16969034075737 | 163.72094360599598 |

All four metric records and the staged runner manifest completed successfully.
The long-form result table and statistics are retained as external server-side
evaluation artifacts; generated samples and metric outputs are intentionally
not versioned in Git.

## Fixed-seed determinism acceptance

The same checkpoint passed the image-level acceptance check:

- FP32, seeds `0-63`, 64 images for each NFE (128 total);
- NFE=1 uses `mid_t=[]`; NFE=2 uses `mid_t=[0.821]`;
- work-group sizes 8 and 16 produced pixel-identical output;
- a repeated generation pass was pixel-identical; and
- the verifier accepted all 131 SHA-256 manifest entries.

This record validates evaluator plumbing and fixed-seed sampling only. The
checkpoint is an external smoke input with no training-integrity receipt, so
none of these values is formal evidence and it is not eligible for a formal
50k run.
