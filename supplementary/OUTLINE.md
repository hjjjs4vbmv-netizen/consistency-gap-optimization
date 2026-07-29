# Supplementary Material Outline

## A. Reproducibility statement

- software and hardware environment
- dataset preparation and identity
- transfer initialization and identity
- training and evaluation commands
- checkpoint, manifest, and result naming conventions

## B. Method details

- baseline sigmoid gap schedule
- global-only calibration with fixed `g=1.10`
- global/local factorization and geometric-mean constraint
- localized-feedback implementation and clipping
- realized gap ratio and clipping-rate definitions

## C. Experimental protocol

- primary and secondary endpoints
- confirmatory training seeds 3, 4, and 5
- fixed sampling seeds and NFE definitions
- 5k screening versus 50k confirmatory evaluation
- paired-delta and percentage-change definitions
- failure, missing-cell, and rerun policies

## D. Primary-setting results

- per-seed fixed versus global-only table
- aggregate KID/FID-50k table
- NFE=1 primary analysis
- NFE=2 secondary analysis
- long-budget quality-versus-budget curves

## E. Mechanism analysis

- global calibration contribution
- localized feedback contribution
- global/local interaction
- controller correction, clipping, and realized gap plots
- failure and negative-result cases

## F. Retrospective exploratory setting

- motivation for examining q=128 after the primary protocol
- explicit statement that q=128 was not frozen before results were observed
- pre-merge source provenance and material implementation differences
- dataset archive mismatch and unavailable canonical content comparison
- per-seed 5k proxy screening results only
- NFE- and seed-dependent effect direction without confirmatory claims

## G. Training stability

- successful and skipped optimizer steps
- NaN and Inf counts
- trailing loss mean and dispersion
- runtime and memory
- resume validation

## H. Additional qualitative results

- deterministic same-seed samples
- NFE=1 and NFE=2 grids
- examples of improvement, ties, and regressions
- sample-selection rule

## I. Limitations and responsible interpretation

- number of independent training seeds
- CIFAR-10 and architecture scope
- proxy versus formal metric limits
- hyperparameter-selection boundaries
- computational budget and untested settings

## J. Artifact manifest

- source commit
- anonymous release commit
- configuration files
- dataset and transfer SHA256
- checkpoint SHA256 list
- table and figure generation inputs
