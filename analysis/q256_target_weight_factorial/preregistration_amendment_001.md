# Preregistration amendment 001: evaluation compute forecast

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_prereg_amendment_001

This amendment was recorded before any formal four-arm training or evaluation.
It corrects one operational forecast in the frozen v1 preregistration. It does
not change the scientific question, arms, training budget, endpoints, sampling
seeds, NFE definitions, contrasts, interaction, stopping rules, or reporting
rules.

The v1 estimate of 12--16 evaluation GPU-hours treated endpoints as if they
were separate sampling jobs. The production staged evaluator computes FID and
KID from the same 50,000-sample job at a given checkpoint and NFE. The formal
design therefore contains 12 final checkpoints x 2 NFE modes = 24 sampling
jobs, not 48 independent metric jobs.

The closest archived same-evaluator timing evidence is the q128 formal run:

- NFE=1 mean: 285.115 seconds per checkpoint;
- NFE=2 mean: 426.918 seconds per checkpoint;
- 12 checkpoints x both modes: 8,544.396 seconds, or 2.373 GPU-hours.

The corrected planning envelope is 2.4--4.0 evaluation GPU-hours, with the
lower edge representing the direct q128 timing proxy and the upper edge
allowing cache validation, I/O, and operational variation. This is a resource
forecast only. Live A100 timing remains a launch-time measurement, and any
failed or invalid job is handled by the preregistered fail-closed rules rather
than by changing an endpoint.

With 6--8 training GPU-hours and 2.4--4.0 evaluation GPU-hours, the corrected
end-to-end planning envelope is 8.4--12 GPU-hours. Two exclusive A100s should
complete the compute after all gates clear in approximately 8--12 wall-clock
hours including smoke, validation, collection, and archival work; two calendar
days remain reserved for one invalid-run replacement.
