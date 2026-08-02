# q128 new-checkpoint smoke report

## Final status

The formal evaluation of the newly delivered 1024-kimg checkpoint set is complete: **12/12 jobs completed**.

Formal source manifest: `/root/staged_eval/formal-q128-new-1024k/run_manifest.json`.
Configuration: FP32; 50,000 generated samples per job (seeds `0-49999`); metrics `kid50k_full` and `fid50k_full`.

## Matrix coverage and pre-flight checks

| Checkpoint | Method | Seed | Load | EMA finite | Schedule | Gap scale | NFE=1 | NFE=2 |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- |
| q128-1024k-seed3-fixed | fixed | 3 | passed | passed | sigmoid | 1.0 | completed | completed |
| q128-1024k-seed3-global110 | global110 | 3 | passed | passed | global_sigmoid | 1.1 | completed | completed |
| q128-1024k-seed4-fixed | fixed | 4 | passed | passed | sigmoid | 1.0 | completed | completed |
| q128-1024k-seed4-global110 | global110 | 4 | passed | passed | global_sigmoid | 1.1 | completed | completed |
| q128-1024k-seed5-fixed | fixed | 5 | passed | passed | sigmoid | 1.0 | completed | completed |
| q128-1024k-seed5-global110 | global110 | 5 | passed | passed | global_sigmoid | 1.1 | completed | completed |

## Selection policy

All predeclared cells (3 seeds × 2 methods × 2 NFE) entered the full formal 50k evaluation. No smoke or early metric result was used as an inclusion or exclusion criterion.

## Final artifacts

- `evaluation/q128_formal_job_status.csv`: 12 completed jobs.
- `results/q128_256k_formal/evaluation_results.csv`: 24 validated metric rows.
- `results/q128_256k_formal/paired_differences.csv`: 12 fixed/global paired differences.
