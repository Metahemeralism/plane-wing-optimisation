"""
This script performs multi-objective optimization of a stiffened plate using NSGA-II.
The two objectives to minimize are:
1. The mass of the plate.
2. The maximum von Mises stress.

The optimization is performed using the pymoo framework, and results are visualized
in a Pareto front plot. Von Mises stress is predicted using a trained neural network
surrogate model loaded from models/nn_surrogate.pt.

Mirrors MaintenancePlate_Optimisation.py (GPR version) so results are
directly comparable — same bounds, same NSGA-II config, same seed.
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

# Constants
rho = 2700  # Density of aluminum alloy (kg/m^3)

# Variable bounds from Table 1 of the brief
# W1: Plate half-width (m), W2: Maintenance port half-width (m), R: Fillet radius (m), t: Plate thickness (m)
lower_bounds = np.array([0.3, 0.10, 0.03, 0.01])  # Minimum values
upper_bounds = np.array([0.7, 0.15, 0.07, 0.02])  # Maximum values

# Resolve paths relative to this script so it runs from any working directory
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
_models_dir = os.path.join(_root, 'models')
_results_dir = os.path.join(_root, 'results')
_figures_dir = os.path.join(_root, 'figures')
os.makedirs(_results_dir, exist_ok=True)
os.makedirs(_figures_dir, exist_ok=True)

# Rebuild the MLP architecture so we can load the saved state dict
# Must match build_mlp() in test_nn.ipynb exactly
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
    layers.append(nn.Linear(prev, 1))  # linear output for regression
    return nn.Sequential(*layers)


# Load the saved scalers and architecture metadata from training
_nn_meta = joblib.load(os.path.join(_models_dir, 'nn_scalers.joblib'))
feature_scaler = _nn_meta['feature_scaler']
target_scaler  = _nn_meta['target_scaler']
arch           = _nn_meta['arch']

# Reconstruct the model and load the trained weights
nn_model = build_mlp(
    n_inputs=4,
    hidden_layers=tuple(arch['hidden_layers']),
    activation=arch['activation'],
    dropout=arch['dropout'],
)
nn_model.load_state_dict(torch.load(os.path.join(_models_dir, 'nn_surrogate.pt'),
                                    map_location='cpu'))
nn_model.eval()  # Set to evaluation mode (disables dropout etc.)


def predict_stress_nn(X):
    """Vectorised NN stress prediction in MPa. X shape: (N, 4)."""
    X_scaled = feature_scaler.transform(X)  # Apply the same scaling used during training
    with torch.no_grad():
        y_scaled = nn_model(torch.tensor(X_scaled, dtype=torch.float32))
        y_scaled = y_scaled.numpy().ravel()
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()  # Inverse-transform back to MPa


# Define the optimization problem class
class PlateOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=4,     # Number of design variables
                         n_obj=2,     # Number of objectives (mass and stress)
                         n_constr=0,  # No constraints in this problem
                         xl=lower_bounds,  # Lower bounds for variables
                         xu=upper_bounds)  # Upper bounds for variables

    def _evaluate(self, X, out, *args, **kwargs):
        # Extract design variables from the input matrix
        W1, W2, R, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Compute the mass of the plate
        mass_values = rho * t * (4 * W1**2 - 4 * W2**2 + (4 - np.pi) * R**2)

        # Predict von Mises stress using the NN surrogate
        stress_values = predict_stress_nn(X)

        # Store the two objectives (mass and stress) for optimization
        out['F'] = np.column_stack([mass_values, stress_values])

# Define the optimization algorithm (NSGA-II)
algorithm = NSGA2(
    pop_size=100,                     # Population size
    sampling=LHS(),                   # Use Latin Hypercube Sampling for better diversity in initial population
    crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover (SBX)
    mutation=PM(prob=0.2, eta=20),    # Polynomial Mutation (PM)
    eliminate_duplicates=True,        # Remove duplicate solutions
)

# Instantiate the optimization problem
problem = PlateOptimization()

# Run the optimization process
res = minimize(problem,
               algorithm,
               ('n_gen', 100),    # Number of generations for evolution
               seed=1,
               verbose=True,
               save_history=True)  # Save full optimization history for analysis

# Extract the final Pareto-optimal solutions
pareto_front = res.F  # Objective values of Pareto-optimal solutions
masses_pareto, stresses_pareto = pareto_front[:, 0], pareto_front[:, 1]

# Extract all sub-optimal solutions from optimization history
all_solutions = np.vstack([gen.pop.get('F') for gen in res.history])
masses_all, stresses_all = all_solutions[:, 0], all_solutions[:, 1]

# Save Pareto data (front, full history, designs) for later comparison
pd.DataFrame({'mass_kg': masses_pareto,
              'stress_mpa': stresses_pareto}).to_csv(
    os.path.join(_results_dir, 'pareto_front_nn.csv'), index=False)
pd.DataFrame({'mass_kg': masses_all,
              'stress_mpa': stresses_all}).to_csv(
    os.path.join(_results_dir, 'pareto_all_nn.csv'), index=False)
pd.DataFrame(res.X, columns=['W1', 'W2', 'R', 't']).assign(
    mass_kg=masses_pareto, stress_mpa=stresses_pareto).to_csv(
    os.path.join(_results_dir, 'pareto_designs_nn.csv'), index=False)

# Plot the Pareto Front and Sub-Optimal Solutions
plt.figure(figsize=(8, 6))
plt.scatter(masses_all, stresses_all, c='gray', alpha=0.25, label='Sub-Optimal Solutions')
plt.scatter(masses_pareto, stresses_pareto, c='C1', label='Pareto Front (Optimal Solutions)')
plt.xlabel('Mass (kg)')
plt.ylabel('Maximum von Mises Stress (MPa)')
plt.title('NN Surrogate — NSGA-II Pareto Front')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_figures_dir, 'Fig_CW2_ParetoFront_NN.png'), dpi=500)  # Save the figure
plt.show()
