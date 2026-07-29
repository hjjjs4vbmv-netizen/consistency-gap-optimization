# q=128 Seed 3 Engineering Preflight

This directory contains lightweight evidence from a paired 4.096 kimg
engineering smoke:

- fixed sigmoid, q=128;
- global-only, q=128 and g=1.10;
- training seed 3;
- identical assets, optimizer, batch, precision, and budget.

This is preliminary engineering evidence produced from an unmerged PR #23
source archive. It validates command parsing, paired execution, telemetry,
checkpoint writing, and the expected gap direction.

It is not a generation-quality result:

- no KID or FID was run;
- the q=128 secondary setting was not yet formally approved;
- the source implementation was not yet merged into main;
- it must not be used as paper evidence or compared with formal benchmarks.

Checkpoints, training states, generated images, and complete logs are stored
outside Git.
