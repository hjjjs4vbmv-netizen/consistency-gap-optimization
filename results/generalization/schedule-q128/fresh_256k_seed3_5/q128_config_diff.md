# Fresh q=128 configuration comparison

## Experiment matrix

The experiment contains six fresh training cells:

| Method | Seed | q | Target |
|---|---:|---:|---:|
| Fixed sigmoid | 3, 4, 5 | 128 | 256 kimg |
| Global-only 1.10 | 3, 4, 5 | 128 | 256 kimg |

## Shared training configuration

- Global batch size: 128
- Per-GPU batch size: 16
- Optimizer: RAdam
- Learning rate: 0.0001
- Dropout: 0.2
- Augmentation probability: 0
- FP16 and AMP GradScaler: enabled
- TF32: disabled
- k=8, b=1, c=0
- Scheduler doubling interval: 10000 ticks
- Training starts from the same official transfer checkpoint.

## Controlled differences

Fixed uses `schedule=sigmoid` and `global_gap_scale=1.0`.

Global-only uses `schedule=global_sigmoid` and
`global_gap_scale=1.10`.

Across paired cells, only the method configuration, training seed,
run identifier, and output directory vary.

## Relationship to q=256

The uploaded q=256 1024-kimg configurations were used only to audit
shared hyperparameters. Their resume checkpoint paths and output
directories were not reused. All q=128 cells are fresh runs from the
common transfer checkpoint and terminate at 256 kimg.

## Dataset identity

The experiment used the locally available CIFAR-10 ZIP. Its exact
SHA256 is recorded in `metadata.json`. It is not claimed to be
byte-identical to the ZIP recorded by the q=256 formal evaluation.
