# Execution

The supplied protocol is unchanged. This deployment uses the four already authorized
GPUs; no additional instance is rented automatically.

| Node role | GPU 0 seeds | GPU 1 seeds |
|---|---|---|
| Cloud ingress / relay | 55; then 62 L_A / X_A_from_B | 56; then 62 L_B / X_B_from_A |
| Private ECT | 59; then 61 L_A / X_A_from_B | 60; then 61 L_B / X_B_from_A |

The first seed of each lane runs `L_A`, `L_B`, `X_A_from_B`, `X_B_from_A` in that
fixed order. The final seed on each node is shared by its two GPUs as shown above,
with L before X on each GPU. Every branch appears once; each lane has three L and
three X branches, avoiding idle cards during the final two seeds.
Each branch starts once its own complete sources and the original runtime are present;
evaluation-only source files and other seeds' files do not block it. Original sources
remain read-only. New outputs use separate M2 and SWAP namespaces.

`python -m scripts.run_state_interventions --help` describes the single-GPU entry.
It requires passing bounded engineering evidence and records the execution commit.
The engineering entry is separate and is never used for formal training.
The runner records scientific failures and continues the fixed queue. Technical
failures are reported as unresolved, without automatic repairs or restarts. Any
later exact technical recovery must preserve its original evidence and remain within
the protocol's maximum of two recoveries.

Existing per-attempt telemetry is retained. `first64.jsonl` adds actual RAdam step
values, the EMA attempt clock and the pre-attempt RNG fingerprint without drawing
random numbers. Short own-state replay, intervention input alignment and exact
continuous-versus-resumed state checks cover this observer. The optional per-source
counterfactual update-vector comparison is not added to the formal training path.

Evaluation adapters are not authorized to launch until export and evaluator checks
pass. This does not block training: readouts are saved in full states. No interim
FID is used for scheduling or implementation acceptance.
