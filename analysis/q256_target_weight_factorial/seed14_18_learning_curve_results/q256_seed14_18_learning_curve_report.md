# q256 seed14–18 learning-curve final audit

**PASS.** Training completed 20/20 trajectories and 120/120 immutable milestones. External evaluation completed 240/240 unique FP32 KID/FID-50k jobs, with 240/240 durable receipts and zero feature-SHA or integrity mismatches.

This is a post-preregistration secondary sensitivity learning-curve analysis over five descriptive seeds. It does not replace the preregistered primary analysis and does not establish a universal arm ranking, optimizer mechanism, or causal effect size.

## 1024 kimg endpoint means

| Arm | NFE | FID mean | FID SD | KID mean | KID SD |
|---|---:|---:|---:|---:|---:|
| A | 1 | 9.176055 | 2.251217 | 0.00556332 | 0.00152177 |
| A | 2 | 2.876808 | 0.393900 | 0.00100634 | 0.00026268 |
| B | 1 | 8.902204 | 1.271609 | 0.00550551 | 0.00072844 |
| B | 2 | 2.920119 | 0.233103 | 0.00111406 | 0.00030702 |
| C | 1 | 9.188192 | 2.126556 | 0.00569611 | 0.00146898 |
| C | 2 | 2.947230 | 0.324811 | 0.00109913 | 0.00033555 |
| D | 1 | 8.660364 | 1.703783 | 0.00536164 | 0.00104837 |
| D | 2 | 2.889808 | 0.226922 | 0.00102527 | 0.00018078 |

## Integrity

- Frozen matrix: seeds 14–18 × arms A–D × 384/512/640/768/896/1024 kimg × NFE1/2.
- Every job used FP32, 50,000 samples, sample seeds 0–49999, and metric seed 20260730.
- Every NFE2 job used mid_t=0.821.
- KID/FID generated-feature SHA-256 was identical within every job.
- Two interrupted repartition outputs were quarantined and never used; both jobs were regenerated to PASS receipts.
- The durable progress-state race affected only a monitoring JSON file; formal receipts were not affected, and shard state files now independently report PASS.

## Provenance

- Formal training source: `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.
- Learning-curve replay source: `f4115a89c764081e01be4290f0868cb8f625825e`.
- Frozen evaluator source: `d6aba02fb88e9db0993623895eb2228ed717d810`.
- Full receipt-level evidence and checkpoints are retained on the experiment servers; this repository package is the compact, hash-bound result surface.
