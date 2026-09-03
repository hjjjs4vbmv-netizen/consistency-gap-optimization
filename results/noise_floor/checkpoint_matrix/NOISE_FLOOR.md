# Generation-block sensitivity

Checkpoint summaries complete: 52/52.

B0 is a historical anchor and is excluded from all means and SDs. FID is on the natural-log scale; KID is on the raw scale. The 2SD value is descriptive and is not a TIE or confidence rule.

## Exact contrasts

| Contrast | NFE | Metric | Status | Same B0 sign | 2SD |
|---|---:|---|---|---:|---:|
| Q256_Q_seed5 | 1 | fid | COMPLETE | 5/5 | 0.00715534226 |
| Q256_Q_seed5 | 1 | kid | COMPLETE | 5/5 | 0.000227295048 |
| Q256_Q_seed5 | 2 | fid | COMPLETE | 5/5 | 0.00653634899 |
| Q256_Q_seed5 | 2 | kid | COMPLETE | 5/5 | 7.76258765e-05 |
| TW_BD_256 | 1 | fid | COMPLETE | 5/5 | 0.000791628248 |
| TW_BD_256 | 1 | kid | COMPLETE | 5/5 | 0.000518620603 |
| TW_BD_256 | 2 | fid | COMPLETE | 5/5 | 0.00430607842 |
| TW_BD_256 | 2 | kid | COMPLETE | 5/5 | 0.000240706056 |
| TW_BD_1024 | 1 | fid | COMPLETE | 5/5 | 0.0065591167 |
| TW_BD_1024 | 1 | kid | COMPLETE | 5/5 | 6.3302746e-05 |
| TW_BD_1024 | 2 | fid | COMPLETE | 5/5 | 0.0074094441 |
| TW_BD_1024 | 2 | kid | COMPLETE | 2/5 | 5.10498188e-05 |
| Q128_Bsame_A | 1 | fid | COMPLETE | 5/5 | 0.01033436 |
| Q128_Bsame_A | 1 | kid | COMPLETE | 5/5 | 7.91038662e-05 |
| Q128_Bsame_A | 2 | fid | COMPLETE | 5/5 | 0.00742337305 |
| Q128_Bsame_A | 2 | kid | COMPLETE | 5/5 | 2.03517201e-05 |
| Q128_Cmatch_Bmatch | 1 | fid | COMPLETE | 5/5 | 0.0131583389 |
| Q128_Cmatch_Bmatch | 1 | kid | COMPLETE | 5/5 | 0.000121775812 |
| Q128_Cmatch_Bmatch | 2 | fid | COMPLETE | 0/5 | 0.00915911196 |
| Q128_Cmatch_Bmatch | 2 | kid | COMPLETE | 5/5 | 2.25788648e-05 |

## Rotation

| Rotation | NFE | Metric | B0 | New blocks | Status |
|---|---:|---|---:|---:|---|
| TW_BD_256_TO_1024 | 1 | fid | 0 | 0/5 | COMPLETE |
| TW_BD_256_TO_1024 | 1 | kid | 1 | 5/5 | COMPLETE |
| TW_BD_256_TO_1024 | 2 | fid | 0 | 0/5 | COMPLETE |
| TW_BD_256_TO_1024 | 2 | kid | 1 | 2/5 | COMPLETE |

## Not evaluated

- `Q256_HA_seed3`: NF-02 BA@1024 inaccessible
- `Q256_HA_seed4`: NF-04 BA@1024 inaccessible
- `Q256_HA_seed5`: NF-12 BA@1024 inaccessible
- `DR_Q256_SEED5`: Q256_HA_seed5 not evaluated

Because BA@1024 was not re-evaluated, no q256 history contrast (H_A) or delayed reversal is evaluated here. This post-seal analysis is limited to checkpoint-level variation, seed5 Q@512, the paired target-weight B-D contrasts and rotation, and the two listed single-seed q128 contrasts; all labels are contrast-specific and do not modify frozen inference.
