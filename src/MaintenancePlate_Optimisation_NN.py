"""
NSGA-II optimisation of the maintenance-plate design using the trained
neural-network surrogate for von Mises stress.

Mirrors MaintenancePlate_Optimisation.py (GPR version) so results are
directly comparable - same bounds, same NSGA-II config, same seed.

Outputs:
    - results/pareto_front_nn.csv    (Pareto-optimal mass/stress pairs)
    - results/pareto_all_nn.csv      (every evaluated candidate)
    - results/pareto_designs_nn.csv  (the corresponding design variables)
    - figures/Fig_CW2_ParetoFront_NN.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from torch import nn
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Constants (same as GPR script)
rho = 2700  # Density of aluminum alloy (kg/m^3)
lower_bounds = np.array([0.3, 0.10, 0.03, 0.01])
upper_bounds = np.array([0.7, 0.15, 0.07, 0.02])

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
_models_dir = os.path.join(_root, 'models')
_results_dir = os.path.join(_root, 'results')
_figures_dir = os.path.join(_root, 'figures')
os.makedirs(_results_dir, exist_ok=True)
os.makedirs(_figures_dir, exist_ok=True)


# ---- Rebuild the MLP so we can load the saved state dict ----
# This must match build_mlp() in test_nn.ipynb exactly.
def build_mlp(n_inputs=4, hidden_layers=(32, 32), activation='relu', dropout=0.0):
    act = {
        'relu': nn.ReLU(),
        'ELU': nn.ELU(),
        'tanh': nn.Tanh(),
        'sine': nn.SiLU(),
        'sigmoid': nn.Sigmoid(),
        'soft_plus': nn.Softplus(),
    }
    if activation not in act:
        raise ValueError(f'Unsupported activation: {activation}')
    layers, prev = [], n_inputs
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act[activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# Load scalers + architecture, then reconstruct and load weights
_nn_meta = joblib.load(os.path.join(_models_dir, 'nn_scalers.joblib'))
feature_scaler = _nn_meta['feature_scaler']
target_scaler = _nn_meta['target_scaler']
arch = _nn_meta['arch']

nn_model = build_mlp(
    n_inputs=4,
    hidden_layers=tuple(arch['hidden_layers']),
    activation=arch['activation'],
    dropout=arch['dropout'],
)
nn_model.load_state_dict(torch.load(os.path.join(_models_dir, 'nn_surrogate.pt'),
                                    map_location='cpu'))
nn_model.eval()


def predict_stress_nn(X):
    """Vectorised NN stress prediction in MPa. X shape: (N, 4)."""
    X_scaled = feature_scaler.transform(X)
    with torch.no_grad():
        y_scaled = nn_model(torch.tensor(X_scaled, dtype=torch.float32))
        y_scaled = y_scaled.numpy().ravel()
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


class PlateOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_constr=0,
                         xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, X, out, *args, **kwargs):
        W1, W2, R, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        mass_values = rho * t * (4 * W1**2 - 4 * W2**2 + (4 - np.pi) * R**2)
        stress_values = predict_stress_nn(X)
        out['F'] = np.column_stack([mass_values, stress_values])


algorithm = NSGA2(
    pop_size=100,
    sampling=LHS(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.2, eta=20),
    eliminate_duplicates=True,
)

problem = PlateOptimization()
res = minimize(problem, algorithm, ('n_gen', 100),
               seed=1, verbose=True, save_history=True)

# Pareto-optimal set
pareto_front = res.F
masses_pareto, stresses_pareto = pareto_front[:, 0], pareto_front[:, 1]

# Full history (all candidates across generations)
all_solutions = np.vstack([gen.pop.get('F') for gen in res.history])
masses_all, stresses_all = all_solutions[:, 0], all_solutions[:, 1]

# Save CSVs for downstream plotting / comparison
pd.DataFrame({'mass_kg': masses_pareto,
              'stress_mpa': stresses_pareto}).to_csv(
    os.path.join(_results_dir, 'pareto_front_nn.csv'), index=False)
pd.DataFrame({'mass_kg': masses_all,
              'stress_mpa': stresses_all}).to_csv(
    os.path.join(_results_dir, 'pareto_all_nn.csv'), index=False)

# Also save the Pareto-optimal DESIGNS (X) so we can cross-check with FEA later
pareto_X = res.X
pd.DataFrame(pareto_X, columns=['W1', 'W2', 'R', 't']).assign(
    mass_kg=masses_pareto, stress_mpa=stresses_pareto).to_csv(
    os.path.join(_results_dir, 'pareto_designs_nn.csv'), index=False)

# Figure
plt.figure(figsize=(8, 6))
plt.scatter(masses_all, stresses_all, c='gray', alpha=0.25,
            label='Sub-Optimal Solutions')
plt.scatter(masses_pareto, stresses_pareto, c='C1',
            label='Pareto Front (Optimal Solutions)')
plt.xlabel('Mass (kg)')
plt.ylabel('Maximum von Mises Stress (MPa)')
plt.title('NN Surrogate \u2014 NSGA-II Pareto Front')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_figures_dir, 'Fig_CW2_ParetoFront_NN.png'), dpi=500)
print('\nSaved:')
print('  results/pareto_front_nn.csv')
print('  results/pareto_all_nn.csv')
print('  results/pareto_designs_nn.csv')
print('  figures/Fig_CW2_ParetoFront_NN.png')
