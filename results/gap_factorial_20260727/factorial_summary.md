# Global/local gap factorial summary

> KID-5k and FID-5k are 5,000-sample proxy metrics, not standard 50,000-sample benchmarks.

Selected global gap scale: **g\* = 1.1**. Lower is better for both metrics.

This is a paired three-training-seed (`n=3`) descriptive analysis. The 95% intervals use the two-sided Student-t critical value `t(df=2)=4.3026527`; they are not evidence for broad population-level significance.

Seed 0 was used to select g\* in the response-curve stage and its fixed/global observations are reused in this formal matrix. Therefore seed 0 has selection/evaluation overlap; interpret selected-g\* effects as selection-aware descriptive estimates.

## Held-out seed 1/2 headline calculation

For arm `A` and metric `M`, the reported headline is

`100 × (mean(M_A,seed1, M_A,seed2) / mean(M_fixed,seed1, M_fixed,seed2) - 1)`.

It is the percentage difference between the two arithmetic metric means. It is **not** the mean of the two per-seed percentage changes.

| Arm | NFE | Metric | Fixed mean | Arm mean | Headline % | Seed 1 % | Seed 2 % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| global | 1 | kid5k_full | 0.373479 | 0.36385 | -2.57818% | -2.1827% | -2.86679% |
| global | 1 | fid5k_full | 342.953 | 328.453 | -4.22805% | -6.60986% | -2.22065% |
| global | 2 | kid5k_full | 0.270937 | 0.219221 | -19.0877% | -70.5018% | -5.25576% |
| global | 2 | fid5k_full | 246.29 | 200.437 | -18.6174% | -60.8296% | -4.1158% |
| local-conservative | 1 | kid5k_full | 0.373479 | 0.373456 | -0.00616428% | -0.229682% | 0.156954% |
| local-conservative | 1 | fid5k_full | 342.953 | 342.824 | -0.0376718% | -0.0915416% | 0.00772972% |
| local-conservative | 2 | kid5k_full | 0.270937 | 0.270454 | -0.178322% | 0.719665% | -0.419907% |
| local-conservative | 2 | fid5k_full | 246.29 | 246.1 | -0.0770239% | 0.669709% | -0.333557% |
| combined-conservative | 1 | kid5k_full | 0.373479 | 0.372011 | -0.393185% | 2.62254% | -2.594% |
| combined-conservative | 1 | fid5k_full | 342.953 | 336.17 | -1.97794% | -2.10947% | -1.86709% |
| combined-conservative | 2 | kid5k_full | 0.270937 | 0.218337 | -19.4141% | -69.6979% | -5.88623% |
| combined-conservative | 2 | fid5k_full | 246.29 | 202.311 | -17.8564% | -57.2435% | -4.32532% |
| local-aggressive | 1 | kid5k_full | 0.373479 | 0.376354 | 0.769801% | 3.90926% | -1.52131% |
| local-aggressive | 1 | fid5k_full | 342.953 | 344.336 | 0.403133% | 2.3771% | -1.26053% |
| local-aggressive | 2 | kid5k_full | 0.270937 | 0.288663 | 6.54272% | 36.9647% | -1.6417% |
| local-aggressive | 2 | fid5k_full | 246.29 | 261.443 | 6.15255% | 27.5746% | -1.20678% |
| combined-aggressive | 1 | kid5k_full | 0.373479 | 0.362757 | -2.87073% | -1.84154% | -3.62181% |
| combined-aggressive | 1 | fid5k_full | 342.953 | 327.906 | -4.38771% | -6.26854% | -2.80254% |
| combined-aggressive | 2 | kid5k_full | 0.270937 | 0.216254 | -20.1828% | -69.8566% | -6.81906% |
| combined-aggressive | 2 | fid5k_full | 246.29 | 198.465 | -19.418% | -61.171% | -5.07421% |

At NFE=2, both held-out seeds improve directionally for `global` and `combined-aggressive`, but seed 1 has a much larger effect. Seed 1 accounts for the following share of the total absolute two-seed metric decrease:

| Arm | Metric | Seed 1 share |
| --- | --- | ---: |
| global | kid5k_full | 78.3025% |
| global | fid5k_full | 83.5455% |
| combined-aggressive | kid5k_full | 73.3761% |
| combined-aggressive | fid5k_full | 80.5503% |

A “win” means a negative paired contrast because lower is better. For interaction rows, negative means the combination is better than the corresponding additive prediction on the raw scale. The geometric relative percentage for that row is the multiplicative interaction `combined × fixed / (global × local) - 1`.

## Validated matrix

- Unique training cells: 18 (fixed/global are shared across profiles)
- Evaluated training-seed × NFE cells: 36
- Scalar metric files read exactly once: 72
- Selection artifact status: `passed`

## Effect definitions

| Effect | Raw-scale paired contrast | Log-scale contrast |
| --- | --- | --- |
| `global_at_local0` | `global - fixed` | `log(global / fixed)` |
| `global_at_local1` | `combined - local` | `log(combined / local)` |
| `local_at_global0` | `local - fixed` | `log(local / fixed)` |
| `local_at_global1` | `combined - global` | `log(combined / global)` |
| `combined_vs_fixed` | `combined - fixed` | `log(combined / fixed)` |
| `additive_interaction` | `combined - global - local + fixed` | `log(combined * fixed / (global * local)) (multiplicative interaction)` |
| `global_main_effect` | `0.5 * [(global - fixed) + (combined - local)]` | `0.5 * [log(global / fixed) + log(combined / local)]` |
| `local_main_effect` | `0.5 * [(local - fixed) + (combined - global)]` | `0.5 * [log(local / fixed) + log(combined / global)]` |

## Three-seed summaries

| Profile | NFE | Metric | Effect | Mean Δ | Sample SD | 95% t CI | Wins/3 | Geometric relative % | Relative 95% t CI |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| conservative | 1 | kid5k_full | `global_at_local0` | -0.0651906 | 0.0962748 | [-0.304351, 0.173969] | 3/3 | -20.1464% | [-66.1408%, 88.3268%] |
| conservative | 1 | kid5k_full | `global_at_local1` | -0.0698903 | 0.119008 | [-0.365523, 0.225743] | 2/3 | -22.8353% | [-74.7788%, 136.087%] |
| conservative | 1 | kid5k_full | `local_at_global0` | 0.0008247 | 0.00162696 | [-0.00321689, 0.00486629] | 1/3 | 0.196202% | [-0.909826%, 1.31458%] |
| conservative | 1 | kid5k_full | `local_at_global1` | -0.00387505 | 0.0219845 | [-0.0584876, 0.0507375] | 1/3 | -3.17762% | [-24.9259%, 24.8709%] |
| conservative | 1 | kid5k_full | `combined_vs_fixed` | -0.0690656 | 0.117486 | [-0.360916, 0.222785] | 2/3 | -22.6839% | [-74.4683%, 134.132%] |
| conservative | 1 | kid5k_full | `additive_interaction` | -0.00469975 | 0.0236002 | [-0.0633259, 0.0539264] | 1/3 | -3.36722% | [-25.8801%, 25.9836%] |
| conservative | 1 | kid5k_full | `global_main_effect` | -0.0675405 | 0.107595 | [-0.334821, 0.19974] | 2/3 | -21.5024% | [-70.7676%, 110.79%] |
| conservative | 1 | kid5k_full | `local_main_effect` | -0.00152518 | 0.0101853 | [-0.0268269, 0.0237766] | 1/3 | -1.50515% | [-12.7998%, 11.2525%] |
| conservative | 1 | fid5k_full | `global_at_local0` | -55.0242 | 70.4659 | [-230.071, 120.023] | 3/3 | -18.2838% | [-58.4254%, 60.6161%] |
| conservative | 1 | fid5k_full | `global_at_local1` | -56.8444 | 86.9325 | [-272.797, 159.108] | 3/3 | -19.794% | [-66.2101%, 90.3825%] |
| conservative | 1 | fid5k_full | `local_at_global0` | 0.320006 | 0.793916 | [-1.65219, 2.2922] | 1/3 | 0.0919939% | [-0.495908%, 0.683369%] |
| conservative | 1 | fid5k_full | `local_at_global1` | -1.50023 | 17.1998 | [-44.2268, 41.2264] | 1/3 | -1.75786% | [-19.0029%, 19.1588%] |
| conservative | 1 | fid5k_full | `combined_vs_fixed` | -56.5244 | 86.1541 | [-270.543, 157.494] | 3/3 | -19.7202% | [-65.9838%, 89.4638%] |
| conservative | 1 | fid5k_full | `additive_interaction` | -1.82023 | 17.9813 | [-46.4882, 42.8477] | 1/3 | -1.84816% | [-19.5515%, 19.7509%] |
| conservative | 1 | fid5k_full | `global_main_effect` | -55.9343 | 78.6163 | [-251.228, 139.359] | 3/3 | -19.0424% | [-62.4952%, 74.7545%] |
| conservative | 1 | fid5k_full | `local_main_effect` | -0.590111 | 8.20972 | [-20.9842, 19.804] | 1/3 | -0.837248% | [-9.69536%, 8.88977%] |
| conservative | 2 | kid5k_full | `global_at_local0` | -0.0413928 | 0.0343019 | [-0.126603, 0.043818] | 3/3 | -35.8822% | [-87.9335%, 240.702%] |
| conservative | 2 | kid5k_full | `global_at_local1` | -0.0425915 | 0.0331696 | [-0.124989, 0.0398063] | 3/3 | -35.6759% | [-87.4575%, 229.884%] |
| conservative | 2 | kid5k_full | `local_at_global0` | 0.000628193 | 0.00232829 | [-0.00515559, 0.00641197] | 1/3 | 0.358716% | [-1.31056%, 2.05623%] |
| conservative | 2 | kid5k_full | `local_at_global1` | -0.000570569 | 0.0018877 | [-0.00525987, 0.00411873] | 1/3 | 0.681595% | [-3.65842%, 5.21712%] |
| conservative | 2 | kid5k_full | `combined_vs_fixed` | -0.0419633 | 0.0330722 | [-0.124119, 0.0401926] | 3/3 | -35.4452% | [-87.3144%, 228.507%] |
| conservative | 2 | kid5k_full | `additive_interaction` | -0.00119876 | 0.00146857 | [-0.00484689, 0.00244936] | 2/3 | 0.321725% | [-3.23364%, 4.00772%] |
| conservative | 2 | kid5k_full | `global_main_effect` | -0.0419922 | 0.0337325 | [-0.125788, 0.0418041] | 3/3 | -35.7791% | [-87.6977%, 235.249%] |
| conservative | 2 | kid5k_full | `local_main_effect` | 2.8812e-05 | 0.00198821 | [-0.00491018, 0.0049678] | 1/3 | 0.520026% | [-2.2589%, 3.37796%] |
| conservative | 2 | fid5k_full | `global_at_local0` | -34.6092 | 36.4091 | [-125.054, 55.8361] | 3/3 | -28.7509% | [-80.3296%, 158.075%] |
| conservative | 2 | fid5k_full | `global_at_local1` | -33.5731 | 34.1031 | [-118.29, 51.1438] | 3/3 | -26.8498% | [-77.2877%, 135.596%] |
| conservative | 2 | fid5k_full | `local_at_global0` | 0.272988 | 1.30758 | [-2.97523, 3.5212] | 1/3 | 0.233043% | [-1.03794%, 1.52034%] |
| conservative | 2 | fid5k_full | `local_at_global1` | 1.30906 | 2.81802 | [-5.69129, 8.30942] | 1/3 | 2.90743% | [-9.35219%, 16.8251%] |
| conservative | 2 | fid5k_full | `combined_vs_fixed` | -33.3001 | 33.6578 | [-116.911, 50.3106] | 3/3 | -26.6794% | [-77.0216%, 133.955%] |
| conservative | 2 | fid5k_full | `additive_interaction` | 1.03608 | 2.39995 | [-4.92574, 6.99789] | 1/3 | 2.66817% | [-8.71977%, 15.4768%] |
| conservative | 2 | fid5k_full | `global_main_effect` | -34.0911 | 35.2545 | [-121.668, 53.486] | 3/3 | -27.8066% | [-78.8632%, 146.579%] |
| conservative | 2 | fid5k_full | `local_main_effect` | 0.791027 | 1.83999 | [-3.77976, 5.36182] | 1/3 | 1.56143% | [-5.14793%, 8.74538%] |
| aggressive | 1 | kid5k_full | `global_at_local0` | -0.0651906 | 0.0962748 | [-0.304351, 0.173969] | 3/3 | -20.1464% | [-66.1408%, 88.3268%] |
| aggressive | 1 | kid5k_full | `global_at_local1` | -0.0711347 | 0.0997616 | [-0.318956, 0.176687] | 3/3 | -22.9739% | [-70.3667%, 100.215%] |
| aggressive | 1 | kid5k_full | `local_at_global0` | -0.00197855 | 0.0126439 | [-0.0333878, 0.0294307] | 2/3 | -0.273741% | [-8.90974%, 9.18101%] |
| aggressive | 1 | kid5k_full | `local_at_global1` | -0.00792264 | 0.0120269 | [-0.0377992, 0.0219539] | 2/3 | -3.80484% | [-17.8826%, 12.6864%] |
| aggressive | 1 | kid5k_full | `combined_vs_fixed` | -0.0731132 | 0.108177 | [-0.341841, 0.195614] | 3/3 | -23.1847% | [-72.1843%, 112.131%] |
| aggressive | 1 | kid5k_full | `additive_interaction` | -0.0059441 | 0.0080417 | [-0.0259208, 0.0140326] | 2/3 | -3.54079% | [-13.567%, 7.6484%] |
| aggressive | 1 | kid5k_full | `global_main_effect` | -0.0681626 | 0.0979512 | [-0.311487, 0.175162] | 3/3 | -21.5729% | [-68.3129%, 94.1106%] |
| aggressive | 1 | kid5k_full | `local_main_effect` | -0.0049506 | 0.0116658 | [-0.03393, 0.0240288] | 2/3 | -2.0552% | [-12.84%, 10.0641%] |
| aggressive | 1 | fid5k_full | `global_at_local0` | -55.0242 | 70.4659 | [-230.071, 120.023] | 3/3 | -18.2838% | [-58.4254%, 60.6161%] |
| aggressive | 1 | fid5k_full | `global_at_local1` | -59.3316 | 75.0723 | [-245.822, 127.158] | 3/3 | -20.2739% | [-62.5866%, 69.8926%] |
| aggressive | 1 | fid5k_full | `local_at_global0` | -1.25864 | 7.60431 | [-20.1488, 17.6315] | 2/3 | -0.290506% | [-5.85581%, 5.60379%] |
| aggressive | 1 | fid5k_full | `local_at_global1` | -5.56609 | 8.84171 | [-27.5301, 16.3979] | 2/3 | -2.71881% | [-13.2204%, 9.05367%] |
| aggressive | 1 | fid5k_full | `combined_vs_fixed` | -60.5903 | 79.0169 | [-256.879, 135.698] | 3/3 | -20.5055% | [-63.8587%, 74.852%] |
| aggressive | 1 | fid5k_full | `additive_interaction` | -4.30745 | 6.06727 | [-19.3794, 10.7645] | 2/3 | -2.43538% | [-10.3447%, 6.17173%] |
| aggressive | 1 | fid5k_full | `global_main_effect` | -57.1779 | 72.7423 | [-237.88, 123.524] | 3/3 | -19.285% | [-60.5566%, 65.1714%] |
| aggressive | 1 | fid5k_full | `local_main_effect` | -3.41236 | 7.66797 | [-22.4607, 15.6359] | 2/3 | -1.51214% | [-9.07696%, 6.68207%] |
| aggressive | 2 | kid5k_full | `global_at_local0` | -0.0413928 | 0.0343019 | [-0.126603, 0.043818] | 3/3 | -35.8822% | [-87.9335%, 240.702%] |
| aggressive | 2 | kid5k_full | `global_at_local1` | -0.0542653 | 0.0593119 | [-0.201604, 0.0930736] | 3/3 | -41.6982% | [-92.831%, 374.139%] |
| aggressive | 2 | kid5k_full | `local_at_global0` | 0.0111605 | 0.0272258 | [-0.0564721, 0.0787932] | 2/3 | 10.2442% | [-30.8968%, 75.8788%] |
| aggressive | 2 | kid5k_full | `local_at_global1` | -0.00171198 | 0.0042984 | [-0.0123898, 0.00896584] | 1/3 | 0.244147% | [-4.41014%, 5.12505%] |
| aggressive | 2 | kid5k_full | `combined_vs_fixed` | -0.0431048 | 0.0324919 | [-0.123819, 0.0376097] | 3/3 | -35.7257% | [-87.3951%, 227.744%] |
| aggressive | 2 | kid5k_full | `additive_interaction` | -0.0128725 | 0.0250141 | [-0.075011, 0.049266] | 1/3 | -9.07085% | [-40.5948%, 39.1815%] |
| aggressive | 2 | kid5k_full | `global_main_effect` | -0.047829 | 0.0468064 | [-0.164102, 0.0684444] | 3/3 | -38.8593% | [-90.6991%, 301.918%] |
| aggressive | 2 | kid5k_full | `local_main_effect` | 0.00472428 | 0.0149477 | [-0.0324079, 0.0418564] | 2/3 | 5.12535% | [-18.5191%, 35.631%] |
| aggressive | 2 | fid5k_full | `global_at_local0` | -34.6092 | 36.4091 | [-125.054, 55.8361] | 3/3 | -28.7509% | [-80.3296%, 158.075%] |
| aggressive | 2 | fid5k_full | `global_at_local1` | -45.7843 | 57.1678 | [-187.797, 96.2283] | 3/3 | -34.4014% | [-87.4276%, 242.272%] |
| aggressive | 2 | fid5k_full | `local_at_global0` | 10.1362 | 21.4193 | [-43.0722, 63.3446] | 1/3 | 8.02949% | [-24.4865%, 54.5468%] |
| aggressive | 2 | fid5k_full | `local_at_global1` | -1.03893 | 2.23356 | [-6.5874, 4.50954] | 2/3 | -0.53799% | [-2.24477%, 1.19859%] |
| aggressive | 2 | fid5k_full | `combined_vs_fixed` | -35.6481 | 36.0373 | [-125.17, 53.8735] | 3/3 | -29.1342% | [-80.5794%, 158.59%] |
| aggressive | 2 | fid5k_full | `additive_interaction` | -11.1751 | 20.7722 | [-62.7762, 40.4259] | 1/3 | -7.93069% | [-36.0849%, 32.6253%] |
| aggressive | 2 | fid5k_full | `global_main_effect` | -40.1967 | 46.7869 | [-156.422, 76.0284] | 3/3 | -31.6345% | [-84.2741%, 197.207%] |
| aggressive | 2 | fid5k_full | `local_main_effect` | 4.54865 | 11.1362 | [-23.1153, 32.2126] | 1/3 | 3.65727% | [-13.0675%, 23.5997%] |

The CSV files contain every per-cell value and every per-seed raw/log contrast used above.
