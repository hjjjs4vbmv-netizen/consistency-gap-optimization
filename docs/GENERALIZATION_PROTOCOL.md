# Generalization Protocol

## Research question

How do global gap calibration and localized feedback separately affect consistency training?

The primary confirmatory comparison is fixed sigmoid versus global-only gap calibration. Localized feedback is a secondary mechanism analysis and must not replace the primary comparison.

## Status of q=128 evidence

The existing q=128 results were not formally frozen before they were observed.
They remain legacy retrospective exploratory screening evidence produced from
a pre-merge implementation, rather than confirmatory generalization evidence.
The source archive differs materially from the reference merged implementation
in training-related files, and canonical dataset-content equivalence could not
be established on this node.

A separate, fresh prospective q=128 matrix is now frozen in
[`FROZEN_EVALUATION_MATRICES.md`](FROZEN_EVALUATION_MATRICES.md): fixed versus
global110, seeds 3/4/5, 256 kimg, NFE=1/2, and KID/FID-50k. It contains no
results yet and does not promote or reuse the legacy evidence.

## Primary setting (frozen elsewhere)

- Dataset: CIFAR-10 32x32 EDM ZIP
- Architecture / preconditioning: ddpmpp / ECT
- Transfer initialization: official EDM CIFAR-10 unconditional VP checkpoint
- Training seeds: 3, 4, 5
- Primary endpoint: NFE=1 KID/FID-50k
- Secondary endpoint: NFE=2 with `mid_t=[0.821]`
- Long budget: at least 1024 kimg
- Global-only scale: `g=1.10`
- Baseline schedule parameters include `q=256`, `k=8`, `b=1`, `c=0`

The value `g=1.10` was selected before the confirmatory seeds and must not be re-selected after observing seeds 3, 4, or 5.

## Retrospective exploratory setting

The q=128 schedule was evaluated retrospectively as an exploratory axis. Because the dataset archive differed bytewise and canonical content equivalence was not available for verification, this evidence does not support a claim that q was the only changed experimental input.

Exploratory setting that was run:

- Name: `schedule-q128`
- Only intended change from the primary setting: `q=128` instead of `q=256`
- Compared methods: fixed sigmoid and global-only with `g=1.10`
- Training seeds: 3, 4, 5
- Sampling modes: NFE=1 and NFE=2 with `mid_t=[0.821]`
- Screening metrics: KID/FID-5k, clearly labeled as proxy results
- Confirmatory metrics: none; this retrospective result is not promoted to 50k confirmation

Changing only random seed does not count as a second setting.

## Freeze gate

The intended freeze gate required Paper Lead and Gap collaborator approval before any result was inspected. That approval was not completed. The unmet gate consisted of:

1. the single changed parameter (`q=128`),
2. the unchanged global scale (`g=1.10`),
3. the training seeds and budget,
4. the checkpoint and data identities,
5. the metric code commit and sampling seed list,
6. the promotion rule from 5k screening to 50k confirmation.

The table is retained to document the historical ordering. Its pending entries show that results preceded formal freeze approval.

| Role | Name or handle | Date (UTC) | Approved commit | Decision |
| --- | --- | --- | --- | --- |
| Paper Lead |  |  |  | pending |
| Gap collaborator |  |  |  | pending |
| Role E |  |  |  | proposed |

## No retrospective promotion

The following gate would have been required for prospective promotion, but it was not completed before results were observed:

- both methods complete under identical conditions,
- checkpoint and dataset SHA256 values match the manifest,
- no NaN or Inf is observed,
- the metric pipeline passes its reproducibility check,
- no protocol field changed after results became visible.

No retrospective promotion is permitted. The fresh prospective matrix requests
q=128 FID/KID-50k only for newly trained, predeclared checkpoints.

## Required metadata

Every training and evaluation cell must record:

- repository commit and dirty state,
- method and global scale,
- schedule parameter set,
- training seed and sampling seed range,
- dataset and transfer-checkpoint SHA256,
- checkpoint SHA256,
- budget and precision,
- GPU, Python, PyTorch, CUDA, and cuDNN versions,
- NFE and `mid_t`,
- sample count and metric name,
- metric implementation commit,
- elapsed time and output directory.

## Interpretation

The q=128 result is a legacy retrospective exploratory diagnostic. It may describe NFE- and seed-dependent behavior, but it cannot establish confirmatory generalization, justify retuning `g`, or redefine the primary endpoint.
