# Squared-GN JVP harness calibration

**PASS_CALIBRATED: an oracle-accurate finite-difference plateau was found.**

The reverse-over-reverse autograd oracle was finite, and both source state and
input assets were preserved. The admitted consecutive scales are:

```text
0.00390625, 0.001953125, 0.0009765625, 0.00048828125
```

| Quantity | Best epsilon | Relative error |
|---|---:|---:|
| Residual tangent | 0.001953125 | 0.0104295 |
| Squared-GN action | 0.0009765625 | 0.00929992 |

At the original scale neighborhood, errors remain large: epsilon 0.015625 has
residual-tangent error 0.339 and action error
0.210. At epsilon 0.00390625 these fall to
0.034 and 0.017.
Almost every FP32 parameter coordinate remains distinguishable between the
positive and negative branches throughout the sweep, so the observed plateau is
not explained by widespread coordinate collapse.

## Interpretation boundary

This calibration identifies a numerical-scale problem in the original
squared-GN correctness cell. It does not retroactively reopen the old factorial
and does not localize the production Jacobian failure. A new protocol may use an
interior plateau scale, but it must freeze that choice before evaluating any
full-field or production regime.
