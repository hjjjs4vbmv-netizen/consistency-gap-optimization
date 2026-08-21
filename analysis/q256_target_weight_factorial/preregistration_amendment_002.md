# Preregistration amendment 002: AMP smoke acceptance rule

Recorded on 2026-08-20 after engineering smoke and before any formal
256-kimg training or quality evaluation. No FID or KID was generated or viewed.

The seed-3 A/B/C/D smoke arms each completed 32 attempts with nine tick-0 AMP
warm-up skips and 23 successful optimizer updates. Arm C moved one skip from
attempt 12 to attempt 11; the other three arms skipped attempt 12. The loss was
finite throughout, and raw nonfinite gradients occurred exactly on skipped
attempts.

The smoke gate now requires every skip to remain inside tick-0 warm-up, finite
loss, raw nonfinite gradients only on skipped attempts, equal skip counts, and
equal successful-update counts across arms. It records every arm's skip
locations but does not require those locations to be identical. Dynamic AMP,
the optimizer, objective, four-arm definitions, 256-kimg budget, endpoints,
contrasts, and reporting rules are unchanged.

This amendment closes the engineering discrepancy; it does not authorize a
separate AMP mechanism audit. After the compact exact-resume gate passes, the
next action is the 12-run formal matrix.
