# Process-Oriented Results

This directory contains the statistical results associated with the
Reviewer uncertainty analysis for the stochastic process-oriented
diffusion models.

## Experiment

The analysis considers:

- 7 stochastic diffusion models: IC, SI, SIR, SEIR, SCIR, FSIR, and irSIR
- 3 temporal network datasets: CollegeMsg, Email-Eu-Core, and Bitcoin-OTC
- Seed-set sizes: 8, 12, 16, and 20
- 1,000 Monte Carlo runs per model configuration
- 95% confidence intervals

## Result File

`process_oriented_final_activation_summary.csv`

The file reports the final activation statistics for every
dataset-model-seed-size combination.

Columns include:

- Dataset
- Model
- Seed_Size
- N_MC
- Final_Mean_Activation
- Final_SD
- Final_SEM
- CI95_Lower
- CI95_Upper
- CI95_Margin
