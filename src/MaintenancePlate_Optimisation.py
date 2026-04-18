"""
This script performs multi-objective optimization of a stiffened plate using NSGA-II.
The two objectives to minimize are:
1. The mass of the plate.
2. The maximum von Mises stress.

The optimization is performed using the pymoo framework, and results are visualized
in a Pareto front plot. Von Mises stress is predicted using a trained GPR surrogate
model loaded from models/gpr_best_pipeline.pkl.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
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
lower_bounds = np.array([0.3,  0.10, 0.03, 0.01])
upper_bounds = np.array([0.7,  0.15, 0.07, 0.02])

# Load the trained GPR pipeline (includes scaler + fitted GPR)
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
_models_dir = os.path.join(_root, 'models')
_results_dir = os.path.join(_root, 'results')
_figures_dir = os.path.join(_root, 'figures')
os.makedirs(_results_dir, exist_ok=True)
os.makedirs(_figures_dir, exist_ok=True)
gpr_pipeline = joblib.load(os.path.join(_models_dir, 'gpr_best_pipeline.pkl'))

# Define the optimization problem class
class PlateOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=4,    # Number of design variables
                         n_obj=2,    # Number of objectives (mass and stress)
                         n_constr=0, # No constraints in this problem
                         xl=lower_bounds,
                         xu=upper_bounds)

    def _evaluate(self, X, out, *args, **kwargs):
        # Extract design variables from the input matrix
        W1, W2, R, t = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        # Compute the mass of the plate
        mass_values = rho * t * (4 * W1**2 - 4 * W2**2 + (4 - np.pi) * R**2)

        # Predict von Mises stress using the GPR surrogate (pipeline handles scaling internally)
        stress_values = gpr_pipeline.predict(X)

        # Store the two objectives (mass and stress) for optimization
        out["F"] = np.column_stack([mass_values, stress_values])

# Define the optimization algorithm (NSGA-II)
algorithm = NSGA2(
    pop_size=100,  # Population size
    sampling=LHS(),  # Use Latin Hypercube Sampling for better diversity in initial population
    crossover=SBX(prob=0.9, eta=15),  # Simulated Binary Crossover (SBX)
    mutation=PM(prob=0.2, eta=20),    # Polynomial Mutation (PM)
    eliminate_duplicates=True         # Remove duplicate solutions
)

# Instantiate the optimization problem
problem = PlateOptimization()

# Run the optimization process
res = minimize(problem,
               algorithm,
               ("n_gen", 100),  # Number of generations for evolution
               seed=1,
               verbose=True,
               save_history=True)  # Save full optimization history for analysis

# Extract the final Pareto-optimal solutions
pareto_front = res.F
masses_pareto, stresses_pareto = pareto_front[:, 0], pareto_front[:, 1]

# Extract all sub-optimal solutions from optimization history
all_solutions = np.vstack([gen.pop.get("F") for gen in res.history])
masses_all, stresses_all = all_solutions[:, 0], all_solutions[:, 1]

# Save Pareto data (front, full history, designs) for later comparison
pd.DataFrame({'mass_kg': masses_pareto,
              'stress_mpa': stresses_pareto}).to_csv(
    os.path.join(_results_dir, 'pareto_front_gpr.csv'), index=False)
pd.DataFrame({'mass_kg': masses_all,
              'stress_mpa': stresses_all}).to_csv(
    os.path.join(_results_dir, 'pareto_all_gpr.csv'), index=False)
pd.DataFrame(res.X, columns=['W1', 'W2', 'R', 't']).assign(
    mass_kg=masses_pareto, stress_mpa=stresses_pareto).to_csv(
    os.path.join(_results_dir, 'pareto_designs_gpr.csv'), index=False)

# Plot the Pareto Front and Sub-Optimal Solutions
plt.figure(figsize=(8, 6))
plt.scatter(masses_all, stresses_all, c="gray", alpha=0.25, label="Sub-Optimal Solutions")
plt.scatter(masses_pareto, stresses_pareto, c="blue", label="Pareto Front (Optimal Solutions)")
plt.xlabel("Mass (kg)")
plt.ylabel("Maximum von Mises Stress (MPa)")
plt.title("GPR Surrogate — NSGA-II Pareto Front")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(_figures_dir, "Fig_CW2_ParetoFront_GPR.png"), dpi=500)
plt.show()
