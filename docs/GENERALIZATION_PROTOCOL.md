# Generalization Protocol

## Research question

How do global gap calibration and localized feedback separately affect consistency training?

The primary confirmatory comparison is fixed sigmoid versus global-only gap calibration. Localized feedback is a secondary mechanism analysis and must not replace the primary comparison.

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

## Secondary setting selection

The lowest-cost valid generalization axis is a second ECT schedule parameter set. It preserves the dataset, model, initialization, optimizer, batch size, precision, training seeds, budgets, sampling seeds, and metric implementation while changing one schedule parameter.

Proposed frozen secondary setting:

- Name: `schedule-q128`
- Only intended change from the primary setting: `q=128` instead of `q=256`
- Compared methods: fixed sigmoid and global-only with `g=1.10`
- Training seeds: 3, 4, 5
- Sampling modes: NFE=1 and NFE=2 with `mid_t=[0.821]`
- Screening metrics: KID/FID-5k, clearly labeled as proxy results
- Confirmatory metrics: KID/FID-50k only if the secondary setting reaches the predeclared promotion gate

Changing only random seed does not count as a second setting.

## Freeze gate

Before any secondary-setting result is inspected, the Paper Lead and Gap collaborator must record approval of:

1. the single changed parameter (`q=128`),
2. the unchanged global scale (`g=1.10`),
3. the training seeds and budget,
4. the checkpoint and data identities,
5. the metric code commit and sampling seed list,
6. the promotion rule from 5k screening to 50k confirmation.

Record the approval in the table below. A blank row means the setting is proposed but not yet formally frozen.

| Role | Name or handle | Date (UTC) | Approved commit | Decision |
| --- | --- | --- | --- | --- |
| Paper Lead |  |  |  | pending |
| Gap collaborator |  |  |  | pending |
| Role E |  |  |  | proposed |

## Promotion rule

The secondary setting is intentionally minimal. Run the paired short-budget smoke first. Promote it to formal 50k evaluation only when:

- both methods complete under identical conditions,
- checkpoint and dataset SHA256 values match the manifest,
- no NaN or Inf is observed,
- the metric pipeline passes its reproducibility check,
- no protocol field changed after results became visible.

Promotion is an engineering-validity gate, not a requirement that global-only performs better.

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

The secondary setting tests whether the direction of the fixed-versus-global-only effect survives a predeclared schedule change. It is not a second opportunity to tune `g`, choose favorable seeds, or redefine the primary endpoint.
