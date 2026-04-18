# Surrogate-Assisted Optimisation of an Aircraft-Wing Maintenance Panel

Multi-objective shape optimisation of an aluminium maintenance panel with an
oval-shaped port under biaxial loading. Two regression surrogates — a Gaussian
Process (GPR) and a neural network (NN) — are trained on FEA stress data, then
driven by NSGA-II to approximate the mass / max-von-Mises-stress Pareto front.

## Problem

| Variable | Description | Bounds (m) |
|----------|-------------|------------|
| `W1`     | Plate half-width | 0.30 – 0.70 |
| `W2`     | Port half-width  | 0.10 – 0.15 |
| `R`      | Port fillet radius | 0.03 – 0.07 |
| `t`      | Plate thickness  | 0.01 – 0.02 |

**Objectives** (both minimised):
- Mass `m = ρ · t · (4·W1² − 4·W2² + (4−π)·R²)`, with `ρ = 2700 kg/m³`.
- Maximum von Mises stress `σ_max`, predicted by the trained surrogate.

Training data comes from 256 quarter-symmetric Abaqus FEA simulations
(`data/TrainingData_256.csv`), split 80/20 into train/test.

## Repository layout

```
plane-wing/
├── data/        Raw FEA training data (CSV) and Abaqus stress-extract scripts
├── notebooks/   Model-development notebooks
│   ├── sampling_test.ipynb   LHS / Sobol / uniform sampling comparison
│   ├── test_gpr.ipynb        GPR kernel selection & training
│   └── test_nn.ipynb         NN architecture search, training, Pareto comparison
├── src/         Runnable Python entry points
│   ├── MaintenancePlate_Optimisation.py      NSGA-II driven by the GPR surrogate
│   ├── MaintenancePlate_Optimisation_NN.py   NSGA-II driven by the NN surrogate
│   └── MaintenancePlate_StressExtract.py     Abaqus-side FEA automation
├── models/      Trained surrogates (gpr_best_pipeline.pkl, nn_surrogate.pt, nn_scalers.joblib)
├── results/     All CSV outputs (experiment sweeps, Pareto fronts, metrics)
└── figures/     All PNG figures generated for the report
```

## Approach

### Surrogates
- **GPR** (`test_gpr.ipynb`) — scikit-learn pipeline with `StandardScaler` and
  an ARD-Matern kernel, selected by 5-fold CV on RMSE.
- **NN** (`test_nn.ipynb`) — a 4-layer × 128-unit MLP with SiLU activations
  (architecture chosen from an 8-experiment CV sweep over activation, depth,
  width, regularisation, optimiser, learning rate, batch size and training-set
  size). Trained with Adam + early stopping.

### Optimisation
NSGA-II (`pymoo`) with population 100, 100 generations, LHS initialisation,
SBX crossover, polynomial mutation, seed 1. The surrogate replaces the FEA
call inside `Problem._evaluate`, giving millisecond-scale evaluations.

## Key results (held-out test set, N = 52)

| Metric           | GPR    | NN     |
|------------------|--------|--------|
| RMSE (MPa)       | see `results/final_metrics.csv` and `comparison-code` cell | **0.307** |
| MAE (MPa)        | -      | **0.203** |
| MAPE (%)         | -      | **0.418** |
| R²               | -      | **0.9994** |

(Run the final cell in `test_nn.ipynb` to reproduce the full side-by-side table.)

Final Pareto fronts are in `results/pareto_front_{gpr,nn}.csv` and
`figures/fig_pareto_comparison.png`.

## Reproducing

```bash
pip install numpy pandas scikit-learn torch pymoo joblib matplotlib
```

```bash
# 1. (Re)train the GPR surrogate
jupyter notebook notebooks/test_gpr.ipynb

# 2. (Re)train the NN surrogate and run all hyperparameter experiments
jupyter notebook notebooks/test_nn.ipynb

# 3. Run NSGA-II with each surrogate
python src/MaintenancePlate_Optimisation.py      # GPR-driven
python src/MaintenancePlate_Optimisation_NN.py   # NN-driven
```

All random seeds are fixed (`SEED = 42` for training, `seed = 1` for NSGA-II),
so numbers in the tables should reproduce exactly.

## Notes

- The original Abaqus `.cae` file and the UCL coursework brief are excluded
  from version control (see `.gitignore`) — they are not mine to redistribute.
- FEA in `src/MaintenancePlate_StressExtract.py` requires Abaqus on the UCL
  Winion cluster; the surrogate pipeline here is fully standalone.
- The scripts were developed on macOS / Python 3.11. No GPU is required —
  the NN has ~50 k parameters and trains in ~1.5 s on CPU.

## Context

Originally submitted as Coursework 2 for **MECH0107 — Data-Driven Methods for
Engineers** (UCL).
