# q256 terminal history replication (n=30)

This is a fresh, two-arm replication over seeds 50--79.  Each seed first
produces matched A/B histories through 512 kimg, then continues both histories
under current arm A through 1024 kimg.  The only endpoint trajectories are
therefore `AA` and `BA`; `AB` and `BB` are neither scheduled nor launched.

The primary estimand is the paired seed-level contrast
`log(FID50k_BA) - log(FID50k_AA)` at 1024 kimg and NFE1.  The frozen decision
rule uses a two-sided 95% Student-t interval for directional evidence and TOST
with a two-sided 90% interval against `+/-log(1.03)` for practical equivalence.

Training failures are outcomes.  Every planned trajectory receives a terminal
status, no failed cell is retried automatically, and a source failure is
recorded as `NOT_RUN_SOURCE_FAILURE` for its dependent endpoint trajectory.
Other predeclared seeds continue so arm-specific failure rates and informative
missingness remain estimable.

Deployment is split over two MatPool nodes:

- `node8`: seeds 50--65 on eight A100 GPUs;
- `node7`: seeds 66--79 on seven A100 GPUs.

Each GPU processes exactly two seeds sequentially.
