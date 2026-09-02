# q256 seed3 formal result snapshot

Captured from the completed seed3 A/B/C/D cells under `formal-direct-dcca41b-deterministic-v1` on 2026-08-20.

All four cells completed 2000 attempted iterations, 1990 accepted optimizer updates, and exactly 256.000 kimg, with no semantic non-finite or raw-gradient/AMP-skip consistency failure.

| Arm | Target scale | Denominator scale | Final-row loss | Last-200 loss mean | AMP skips | Paired-grid visual read |
|---|---:|---:|---:|---:|---:|---|
| A | 1.0 | 1.0 | 16.88090193 | 16.24412828 | 10 | Strong high-frequency noise |
| B | 1.1 | 1.1 | 16.52927291 | 16.21319667 | 10 | More recognizable structure than A |
| C | 1.1 | 1.0 | 18.29579282 | 17.75489447 | 10 | Strongest noise / weakest structure |
| D | 1.0 | 1.1 | 15.36517978 | 14.80949258 | 10 | Clearest structure among the four |

The fixed-grid visual ordering is D, then B, then A/C. The loss values are supporting diagnostics only because the arms change target/denominator geometry and therefore do not necessarily share an identical scalar objective. These runs used `--metrics=none`; no FID or other common external quality metric is available yet, so this snapshot is not a final statistical arm ranking.

## Final sample grids

### A — control

![seed3 arm A final](armA_final.png)

### B — target and denominator scaled

![seed3 arm B final](armB_final.png)

### C — target scaled only

![seed3 arm C final](armC_final.png)

### D — denominator scaled only

![seed3 arm D final](armD_final.png)

## SHA-256

- `armA_final.png`: `780133af6eaad6b48f4a12d818d85f029ba7c5bb7345a73d87a3c3e0fc43e5c5`
- `armB_final.png`: `828fd46f7e52dc48fcab650a5d71e9f8beddef0e1f3aefa0a249c84bb46a5be9`
- `armC_final.png`: `ab347e55e526f42206676b4bcb093b26bf5ec5652af1c1ae6620fd61fbf61741`
- `armD_final.png`: `996673937462c190875329416e91db4c74b3d2f038ceddcec7f703fc5c35872f`
