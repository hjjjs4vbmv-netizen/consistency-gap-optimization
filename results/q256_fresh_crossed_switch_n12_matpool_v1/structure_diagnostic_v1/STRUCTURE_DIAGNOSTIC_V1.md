# H structure diagnostic (descriptive only)

Bound to `H_C_I_Q_G_per_seed.csv` SHA256 `4d8bc83f7e9254878294a38fc3ad2ac40c84445d497dd166cec4f37b2e197461`. Nothing below can alter the frozen primary verdict (**INCONCLUSIVE**).

## Association between H and the switch-point gap Q

- Pearson corr(H, Q) = +0.598; Spearman = +0.345; OLS slope of H on Q = +0.120.

## Subgroups by switch-point gap

| Subgroup | Seeds | n | H mean | 95% CI | H < 0 |
| --- | --- | ---: | ---: | --- | ---: |
| Q < 0 (B better at switch) | 31, 36, 39, 40, 42 | 5 | -0.1337 | [-0.3256, +0.0582] | 4/5 |
| Q > 0 (B worse at switch) | 32, 33, 34, 35, 37, 41 | 6 | -0.0271 | [-0.0835, +0.0292] | 4/6 |
| All except Q < 0 | 32, 33, 34, 35, 37, 41 | 6 | -0.0271 | [-0.0835, +0.0292] | 4/6 |
| Q < -0.5 (B better at switch) | 31, 40 | 2 | -0.2956 | [-0.7639, +0.1726] | 2/2 |
| Q > 0.5 (B worse at switch) | 34, 35, 41 | 3 | -0.0473 | [-0.2239, +0.1293] | 2/3 |
| All except Q < -0.5 | 32, 33, 34, 35, 36, 37, 39, 41, 42 | 9 | -0.0267 | [-0.0650, +0.0116] | 6/9 |

## Delayed-rank-reversal replication

Eligible (Q > 0): 4/6 reversed at 1024 kimg.
- seed32: Q = +0.0083, H = +0.0281 -> not reversed
- seed33: Q = +0.0890, H = -0.0218 -> reversed
- seed34: Q = +0.9746, H = -0.0868 -> reversed
- seed35: Q = +0.5683, H = +0.0348 -> not reversed
- seed37: Q = +0.0467, H = -0.0272 -> reversed
- seed41: Q = +0.5348, H = -0.0899 -> reversed

Eligible (Q > 0.5): 2/3 reversed at 1024 kimg.
- seed34: Q = +0.9746, H = -0.0868 -> reversed
- seed35: Q = +0.5683, H = +0.0348 -> not reversed
- seed41: Q = +0.5348, H = -0.0899 -> reversed

## Reading

A negative pooled H mixes two distinct seed-level histories: persistence of an advantage B already held at the switch (Q < 0) and genuine delayed reversal (Q > 0 with H < 0). The subgroup rows separate them; only the reversal-eligible rows bear on the mid-training-misranking reading. All quantities are descriptive and post-unblind.
