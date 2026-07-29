# q=128 Screening Conclusion

## Scope

This experiment compares fixed sigmoid against global-only gap scaling
with `global_gap_scale=1.10`.

- Training seeds: 3, 4, 5
- Training budget: 256 kimg
- Schedule parameter: q=128
- Evaluation: KID/FID-5k proxy
- NFE modes: 1 and 2
- Metric recomputations: 3 per cell on one generated sample set

This is legacy retrospective exploratory evidence produced from a pre-merge implementation, not confirmatory generalization evidence or a formal FID-50k result. The statistical unit is the training seed (`n=3`); metric recomputations are not independent samples.

## Result

The result is mixed and NFE-dependent.

For NFE=1, global-only won on two of three individual training seeds,
but its across-seed mean was slightly worse for both FID-5k and KID-5k.

For NFE=2, global-only won on two of three training seeds and improved
the across-seed mean for both FID-5k and KID-5k.

The paired differences have substantial dispersion across training
seeds. The experiment therefore does not establish a robust general
quality advantage for global-only calibration.

## Interpretation

q=128 may be retained as a retrospective mechanism diagnostic. It must not be described as a predeclared secondary setting or used as headline evidence that global-only calibration improves the primary NFE=1 endpoint.

## Limitations

- 5k-sample proxy metrics, not formal FID/KID-50k.
- Only three training seeds.
- 256 kimg training budget.
- Experiment used a pre-merge source archive with material training-code differences from the reference merged implementation.
- q=128 had not been formally frozen before results were observed.
- Dataset archives differ bytewise; canonical content equivalence was unavailable, so an only-q-changes claim is unsupported.
