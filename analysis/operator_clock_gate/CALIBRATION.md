# Pre-formal finite-difference calibration

No formal matrix cell or downstream scientific endpoint was inspected during
this calibration. The only calibration cell was fixed in advance as arm A,
audit minibatch `2026082601`, projection direction index 0, on the verified
seed-3 q256/256-kimg source.

Retained field attempts:

- `[0.01, 0.003, 0.001, 0.0003]`: finest adjacent change 2.81245;
- `[0.3, 0.1, 0.03, 0.01]`: finest adjacent change 0.21216;
- `[0.03, 0.02, 0.015, 0.01]`: changes 0.107997, 0.115119,
  0.156148.

The last grid is frozen for the formal field matrix because it brackets the
least unstable observed FP16 operational scale without choosing a successful
threshold. The original 5% gate remains, so this calibration cell is still a
failure and must not be called a converged classical Jacobian.

Retained algorithmic attempts showed non-monotone AMP discontinuities:

- `[0.03, 0.02, 0.015, 0.01]`: every plus branch skipped while every minus
  branch stepped;
- `[0.003, 0.001, 0.0003, 0.0001]`: paired skip failed at the endpoints;
- `[0.001, 0.0007, 0.0005, 0.0003]`: paired skip failed at `0.0007`, while
  the common skip regime also changed across epsilon.

The last grid is frozen for the formal algorithmic matrix as a local audit of
the observed AMP boundary. A valid receipt requires identical plus/minus skip
behavior at every epsilon, a single skip regime across the sweep, source-state
preservation, and the unchanged 5% convergence gate. Failure is an algorithmic
finding, not a license to drop an epsilon or replace a direction.
