# MECH0107 — Coursework 2: Project Reference

## Overview

Surrogate models (GPR and NN) for the shape optimisation of an aircraft wing maintenance panel. Multi-objective optimisation using NSGA-II to minimise mass and maximum von Mises stress.

---

## Key Details

| Item                  | Detail                                              |
| --------------------- | --------------------------------------------------- |
| Module                | MECH0107 — Data-Driven Methods for Engineers        |
| Assessment            | Individual, 60% of module                           |
| Page limit            | 12 pages (references excluded)                      |
| Penalty               | >10% over page count → 10 percentage point deduction |
| Submission format     | Single `.zip` containing report PDF + all code + datasets |
| Anonymity             | No name or student number anywhere                  |
| AI policy             | Category 2 — assistive only, no code writing or idea generation |

---

## Problem Description

- Thin square aluminium plate with an oval-shaped maintenance port
- Subjected to biaxial loading (σ₁/σ₂ = 2)
- Quarter-symmetry exploited in FEA
- Material: aluminium alloy, E = 70 GPa, ν = 0.33, ρ = 2700 kg/m³

---

## Design Variables

| Variable | Description                      | Lower Bound | Upper Bound | Units |
| -------- | -------------------------------- | ----------- | ----------- | ----- |
| W1       | Plate half-width                 | 0.3         | 0.7         | m     |
| W2       | Maintenance port half-width      | 0.1         | 0.15        | m     |
| R        | Maintenance port fillet radius   | 0.03        | 0.07        | m     |
| t        | Plate thickness                  | 0.01        | 0.02        | m     |

> **WARNING:** The bounds in `MaintenancePlate_Optimisation.py/.m` are DIFFERENT
> from Table 1 above. You MUST update them to match Table 1 before running optimisation.

---

## Objective Functions

### 1. Mass (minimise)

```
m(W1, W2, R, t) = ρ * t * [4*W1² - 4*W2² + (4 - π)*R²]
```

where ρ = 2700 kg/m³

### 2. Maximum von Mises Stress (minimise → maximises safety)

```
σ_max(W1, W2, R, t) = f(W1, W2, R, t)
```

where `f` is either: FEA, GPR surrogate, or NN surrogate.

---

## Average Design (for Discussion 3e comparison)

| W1    | W2    | R    | t     |
| ----- | ----- | ---- | ----- |
| 0.5 m | 0.125 m | 0.05 m | 0.015 m |

---

## Files from Moodle

| # | File                                          | Purpose                                    | Modify? | Run on    |
| - | --------------------------------------------- | ------------------------------------------ | ------- | --------- |
| 1 | `MaintenancePlate.cae`                        | Abaqus FEA model                           | NO      | Winion    |
| 2 | `MaintenancePlate_StressExtract_Input.txt`    | Input params for Abaqus script             | YES     | Winion    |
| 3 | `MaintenancePlate_StressExtract.py`           | Abaqus automation script                   | NO      | Winion    |
| 4 | `MaintenancePlate_StressExtract_Output.txt`   | Output from Abaqus (stress + runtime)      | YES     | Winion    |
| 5 | `MaintenancePlate_StressExtract_Function.m`   | MATLAB wrapper to call Abaqus              | NO      | Winion    |
| 6 | `MaintenancePlate_Optimisation.py` / `.m`     | NSGA-II optimisation (has dummy stress eq.) | YES     | Own machine |

---

## Winion Setup

1. Connect to UCL VPN
2. Open File Explorer via RDP from: `http://daisy.meng.ucl.ac.uk/StaffIntranet/ITandComms/ComputerSystems/GuidanceDocuments/1089-MENG.html`
3. Use node 0 or 1
4. Login: `mecheng2012\MechengUsername` with Mech Eng password
5. Working directory: `D:/Temp/YourUserName/MECH0107_CW2/`
6. **Back up files regularly — nodes reset periodically and data loss is not grounds for EC**

---

## FEA Function Usage

```matlab
[maxVMStress, scriptRuntime] = MaintenancePlate_StressExtract_Function(W1, W2, R, t)
```

- Example: `MaintenancePlate_StressExtract_Function(0.6, 0.2, 0.06, 0.02)`
- Returns: `Max Von Mises Stress: 41.60 MPa`, `Script Runtime: 23.31 seconds`
- **Abaqus must not be open** when running this function
- Each call takes ~23 seconds

---

## Report Structure

### 1. Description of the Surrogate Model Setup (30%)

- **GPR**: data sampling, preprocessing, model design choices + quantitative justifications
- **NN**: data sampling, preprocessing, model design choices + quantitative justifications
- Choices must be backed by systematic experiments (e.g. kernel comparisons, architecture comparisons)

### 2. Results (30%)

- **GPR**: error evaluation (RMSE, MAE, etc.) + Pareto front plot
- **NN**: error evaluation (RMSE, MAE, etc.) + Pareto front plot

### 3. Discussion (20%)

**(a) Was using surrogate models justified?**
  - (i) Time to generate data samples
  - (ii) Time to train models
  - (iii) Speed-up per prediction vs FEA
  - (iv) Total optimisation time with surrogates
  - (v) Estimated time for optimisation with FEA directly

**(b) Which model is most/least appropriate?**
  - Consider: computational time + predictive accuracy

**(c) When would the least appropriate model become most appropriate?**
  - Consider: number of design variables, dataset size, training time, generalisation

**(d) Revised constraints — are surrogates still valid?**
  - Tightened bounds (subset of training domain)
  - Relaxed bounds (extrapolation beyond training domain)
  - Strategies to improve accuracy under new constraints

**(e) Optimal design vs average design comparison**
  - Pick one Pareto front, one optimal design
  - Percentage difference in σ_max and m vs average design
  - Additional practical engineering considerations
  - How to incorporate them into surrogate models

### Presentation & Structure (10%)

- Figures: titles, axis labels, ticks, legends, colours
- No spelling/coding/mathematical errors
- No code snippets in report

### Code Quality (10%)

- Runs correctly
- Well-structured, non-redundant
- Every line commented clearly

---

## Timing Data to Record

Track these throughout — you need them for Discussion 3a:

- [ ] Total time for FEA data generation (Phase 2)
- [ ] GPR training time
- [ ] NN training time
- [ ] GPR average single-prediction time
- [ ] NN average single-prediction time
- [ ] FEA single-run time (~23s from example)
- [ ] GPR optimisation wall-clock time
- [ ] NN optimisation wall-clock time

---

## Submission Checklist

- [ ] Report as single `.pdf` (≤12 pages, no identifying info)
- [ ] All code files (well-commented, simple naming)
- [ ] All generated datasets
- [ ] Modified files included (optimisation script, etc.)
- [ ] Generative AI declaration (if any tools used)
- [ ] Everything in a single `.zip` (not `.rar`)
- [ ] Bounds in optimisation script match Table 1
- [ ] Figures have titles, axis labels, legends, colours
- [ ] Discussion answers reference your own quantitative results