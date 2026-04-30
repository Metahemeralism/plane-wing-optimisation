# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surrogate-assisted multi-objective optimization for an aluminum aircraft maintenance panel design. The project trains two regression surrogates — a Gaussian Process (GPR) and a neural network (NN) — on 256 FEA (Finite Element Analysis) samples, then uses them in NSGA-II to jointly minimize panel mass and maximum von Mises stress.

**Course context**: MECH0107 — Data-Driven Methods for Engineers (UCL, Coursework 2).

## Environment & Dependencies

Uses the `data-driven` conda environment:

```bash
conda activate data-driven
pip install numpy pandas scikit-learn torch pymoo joblib matplotlib
```

Python scripts should be run via:
```bash
conda run -n data-driven python <script>
# or
/opt/anaconda3/envs/data-driven/bin/python3 <script>
```

## Commands

**Train GPR surrogate** (outputs `models/gpr_best_pipeline.pkl`):
```bash
jupyter notebook notebooks/test_gpr.ipynb
```

**Train NN surrogate** (outputs `models/nn_surrogate.pt`, `models/nn_scalers.joblib`):
```bash
jupyter notebook notebooks/test_nn.ipynb
```

**Run NSGA-II optimization**:
```bash
conda run -n data-driven python src/MaintenancePlate_Optimisation.py      # GPR-driven
conda run -n data-driven python src/MaintenancePlate_Optimisation_NN.py   # NN-driven
```

**Sampling comparison**:
```bash
jupyter notebook notebooks/sampling_test.ipynb
```

**FEA data generation** (Abaqus only, requires UCL cluster with Abaqus installed):
```bash
python src/MaintenancePlate_StressExtract.py
```

## Architecture

### Design Variables (4D)
- `W1`: plate half-width (0.30–0.70 m)
- `W2`: port half-width (0.10–0.15 m)
- `R`: port fillet radius (0.03–0.07 m)
- `t`: plate thickness (0.01–0.02 m)

### Objectives (both minimized)
- **Mass**: computed analytically — `m = ρ·t·(4·W1² − 4·W2² + (4−π)·R²)`, ρ = 2700 kg/m³
- **Max von Mises stress**: predicted by a trained surrogate (not computed via FEA)

### Pipeline
```
data/TrainingData_256.csv (256 FEA samples)
  ↓ 80/20 train/test split, SEED=42
  ↓
Surrogate training
  ├─ GPR: sklearn Pipeline(StandardScaler → GaussianProcessRegressor, Matérn ν=2.5, ARD)
  └─ NN:  PyTorch MLP (4 → 128×4 → 1, SiLU activations, Adam lr=1e-3, batch=16, no dropout)
  ↓
NSGA-II (pymoo): pop=100, 100 generations, LHS init, SBX crossover, PM mutation, seed=1
  ↓
results/pareto_front_{gpr,nn}.csv + figures/
```

### Saved Models
- `models/gpr_best_pipeline.pkl` — sklearn pipeline (scaler + GPR), load with `joblib.load()`
- `models/nn_surrogate.pt` — PyTorch state dict, architecture reconstructed from metadata
- `models/nn_scalers.joblib` — dict with `feature_scaler`, `target_scaler`, and `arch` keys

### Key Results (52-sample held-out test)
| Metric | GPR | NN |
|--------|-----|----|
| RMSE (MPa) | **0.162** | 0.307 |
| MAPE (%) | **0.258** | 0.418 |
| R² | — | 0.9994 |
| Train time (s) | 5.31 | **1.50** |

GPR outperforms NN at this dataset size (204 training samples, 4D inputs). Both models produce ~100 Pareto-optimal solutions from NSGA-II.

## Reproducibility

All random seeds are fixed — train/test split uses `SEED = 42`, NSGA-II uses `seed = 1`. The NN hyperparameter CV re-seeds per fold as `seed + fold_idx`. Results should be exactly reproducible on the same Python/sklearn/torch versions.

## Notebooks Structure

Each notebook is self-contained and runs sequentially:
- **`sampling_test.ipynb`**: Compares LHS vs Sobol vs uniform sampling; confirms LHS was the right choice for generating training data.
- **`test_gpr.ipynb`**: Kernel selection (RBF, Matérn 1.5/2.5, RQ) via 5-fold CV; ARD length scale analysis; trains final GPR; generates all GPR figures.
- **`test_nn.ipynb`**: 8 sequential hyperparameter experiments (activation → depth → width → regularization → optimizer → lr → batch → n_samples); trains final NN; generates all NN figures and the GPR vs NN Pareto comparison.

## Optimization Scripts

`MaintenancePlate_Optimisation.py` and `MaintenancePlate_Optimisation_NN.py` are parallel in structure — both define a `PlateOptimization(Problem)` subclass with `_evaluate()` that computes mass analytically and stress via the respective surrogate. The NN version reconstructs the MLP from `nn_scalers.joblib` metadata before running inference.
