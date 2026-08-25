# q--g production parity audit status

The frozen audit is complete. It compares the reference
`(q=256, g=1.10)` with the stage-0 effective-spacing-matched candidate
`(q=232.72727272727272, g=1)` on an 8,192-point dense grid and on two fixed
real minibatches from the seed-3, arm-A, 512-kimg checkpoint.

The strict numerical-parity verdict is **FAIL at the preregistered `1e-6`
field tolerance**. Pair coordinates pass the `32 * eps(dtype)` coordinate gate
in FP32 and FP64, with no clipping, but the FP32 inverse-gap weight has relative
L2 error `2.3792e-6`. FP64 weight error is `4.4013e-15` and passes.

On the real checkpoint, the maximum relative L2 discrepancies were:

| precision path | target output | per-sample loss | one-sided gradient |
| --- | ---: | ---: | ---: |
| native network precision | `1.0515e-4` | `4.6375e-4` | `1.6294e-2` |
| forced FP32 reference | `1.7780e-7` | `2.9184e-6` | `6.1987e-5` |

The full machine-readable receipt is
`production_parity_seed3_A_k512.json`. Asset receipts and hashes passed, the
network state was preserved, and no optimizer was constructed or stepped.

This result rejects *strict implementation-level equivalence at the frozen
tolerance*. It does not establish that nominal `q` is an independent scientific
mechanism, and it does not show that these small numerical discrepancies cause
cross-run quality differences. The bounded interpretation remains that `q` is
a schedule parameter whose stage-0 relation to effective pair spacing can be
matched analytically, while the two production arithmetic paths are not
numerically identical in finite precision.
