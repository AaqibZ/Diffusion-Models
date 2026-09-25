# Reviewer #1: Statistical Uncertainty Analysis

This directory contains the additional experiments conducted in response
to Reviewer's comment regarding statistical uncertainty and the inclusion
of error bars or confidence intervals.

## Reviewer comment

> No statistical significance testing. Error bars or confidence intervals
> should be included.

## Purpose

Additional stochastic experiments were conducted to quantify the variability
of diffusion spread across repeated realizations.

For the stochastic diffusion models, the following statistics are reported:

- Mean final activation
- Standard deviation (SD)
- Standard error of the mean (SEM)
- 95% confidence interval (CI)

The 95% confidence interval is computed as:

CI = mean ± 1.96 × SD / sqrt(N)

where N is the number of Monte Carlo realizations.

## Experimental categories

The additional analysis is organized according to the model categories
used in the paper:

1. Process-oriented diffusion models
2. Interaction-oriented diffusion models
3. Competition-oriented diffusion models
4. Structure-oriented diffusion models
5. Target-oriented diffusion models

Each directory contains the corresponding experimental scripts,
statistical results, and uncertainty visualizations.

These experiments are provided as additional analyses for Reviewer
and do not replace the original experimental implementation.
