# q--g production parity audit status

The frozen dense-grid phase is complete. Pair coordinates pass the
`32 * eps(dtype)` gate in both FP32 and FP64, with no clipping. The FP32
inverse-gap weight has relative L2 error `2.3792e-6`, exceeding the frozen
`1e-6` field tolerance. FP64 weight error is `4.4013e-15` and passes.

Accordingly, the strict production-equivalence verdict is already **FAIL at
the frozen tolerance**. This is a small numerical implementation mismatch
caused by the two production arithmetic paths; it is not evidence that nominal
`q` defines an independent scientific mechanism.

The real-checkpoint phase remains required to quantify whether this mismatch
propagates to target outputs, per-sample losses, and one-sided gradients. Its
runner and thresholds are frozen in
`analysis/Q_G_PRODUCTION_PARITY_AUDIT_PROTOCOL.md`. At the time of this partial
receipt, the authorized server was closing SSH connections before
authentication, so no checkpoint measurement was labeled complete.

