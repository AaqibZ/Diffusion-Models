# Target-Oriented Diffusion Models

This directory contains the experimental code and statistical analysis
associated with the Reviewers uncertainty analysis for
target-oriented diffusion models.

## Purpose

The purpose of this experiment is to quantify the variability of
diffusion outcomes for stochastic target-oriented diffusion models
using repeated Monte Carlo simulations.

The analysis reports the mean final activation together with measures
of statistical uncertainty, including standard deviation (SD), standard
error of the mean (SEM), and 95% confidence intervals (CI).

## Models

The following stochastic target-oriented diffusion models are
considered:

- VMID
- MAT
- FSC-SB
- FSC-N
- IC-u
- LT-u
- UAD
- ISR
- ACT

## Datasets

The experiments are conducted on three temporal network datasets:

- CollegeMsg
- Email-Eu-Core
- Bitcoin-OTC

## Experimental Configuration

The statistical uncertainty analysis uses the following configuration:

- Number of temporal snapshots: 10
- Seed-set sizes: 8, 12, 16, and 20
- Confidence level: 95%
- Diffusion outcome: final activation count

For each dataset and seed-set size, the diffusion process is repeated
across Monte Carlo runs. The resulting final activation counts are used
to estimate the mean and variability of the diffusion outcome.

## Statistical Measures

For each model, dataset, and seed-set size, the following statistics
are reported:

- Mean final activation
- Standard deviation (SD)
- Standard error of the mean (SEM)
- 95% confidence interval (CI)

The standard error is calculated as:

SEM = SD / sqrt(N)

where `N` is the number of Monte Carlo runs.

The 95% confidence interval is calculated as:

CI95% = Mean +/- 1.96 * SEM

These statistics provide an estimate of the uncertainty associated
with the stochastic diffusion outcomes.

## Files

### `rerun_experiments.py`

Contains the experimental procedure used to perform the repeated
Monte Carlo simulations for the stochastic target-oriented diffusion
models.

### `results/`

Contains the statistical results and corresponding visualization
files.

The results include:

- Mean final activation
- Standard deviation
- Standard error of the mean
- 95% confidence interval
- Plots illustrating the mean activation and associated uncertainty


This experiment addresses Reviewer's comment regarding the absence
of statistical uncertainty measures:

> "No statistical significance testing. Error bars or confidence
> intervals should be included."

The additional experiments provide uncertainty estimates for the
target-oriented diffusion models. The resulting confidence intervals
and error bars are used to supplement the comparative diffusion
analysis presented in the manuscript.
