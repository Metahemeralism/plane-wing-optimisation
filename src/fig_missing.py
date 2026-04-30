"""
Generates the three figures missing from figures/ that the report brief requires:
  1. fig_sampling_comparison.png  — justify LHS over random/Sobol (Section 1)
  2. fig_gpr_kernel_comparison.png — justify Matérn 2.5 + ARD length scales (Section 1)
  3. fig_gpr_error_eval.png       — GPR parity + 95% CI prediction (Section 2)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline

# ── shared style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
C_BLUE   = '#0072B2'
C_ORANGE = '#D55E00'
C_GREEN  = '#009E73'
C_LIGHT  = '#56B4E9'
SEED     = 42

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Sampling comparison
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES  = 64
N_DIMS     = 4
N_REPEATS  = 50

def _amd(s):
    d = cdist(s, s); np.fill_diagonal(d, np.inf)
    return float(np.mean(np.min(d, axis=1)))

def _disc(s):
    return float(qmc.discrepancy(s, method='L2-star'))

# Random
rng = np.random.default_rng(SEED)
rand_amd, rand_disc = [], []
for i in range(N_REPEATS):
    s = np.random.default_rng(SEED + i).random((N_SAMPLES, N_DIMS))
    rand_amd.append(_amd(s)); rand_disc.append(_disc(s))
rand_pts = np.random.default_rng(SEED).random((N_SAMPLES, N_DIMS))

# Sobol
sobol_pts  = qmc.Sobol(d=N_DIMS).random(N_SAMPLES)
sobol_amd  = _amd(sobol_pts)
sobol_disc = _disc(sobol_pts)

# LHS
lhs_amd, lhs_disc = [], []
for i in range(N_REPEATS):
    s = qmc.LatinHypercube(d=N_DIMS, seed=SEED + i).random(N_SAMPLES)
    lhs_amd.append(_amd(s)); lhs_disc.append(_disc(s))
lhs_pts = qmc.LatinHypercube(d=N_DIMS, seed=SEED).random(N_SAMPLES)

# Scale to [0,1] for scatter — show W1 vs W2 dimensions
BOUNDS_LO = np.array([0.30, 0.10, 0.03, 0.01])
BOUNDS_HI = np.array([0.70, 0.15, 0.07, 0.02])

def scale(pts):
    return pts * (BOUNDS_HI - BOUNDS_LO) + BOUNDS_LO

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6),
                         gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})

scatter_kw = dict(s=28, edgecolor='white', linewidth=0.5, alpha=0.85)
for ax, pts, title, col in [
    (axes[0], scale(rand_pts),  'Random',  C_ORANGE),
    (axes[1], scale(sobol_pts), 'Sobol',   C_LIGHT),
    (axes[2], scale(lhs_pts),   'LHS',     C_BLUE),
]:
    ax.scatter(pts[:, 0], pts[:, 1], color=col, **scatter_kw)
    ax.set_xlabel(r'$W_1$ (m)'); ax.set_ylabel(r'$W_2$ (m)')
    ax.set_title(title)

# Metric summary bar chart
methods = ['Random', 'Sobol', 'LHS']
amd_means  = [np.mean(rand_amd),  sobol_amd,  np.mean(lhs_amd)]
amd_stds   = [np.std(rand_amd),   0.0,        np.std(lhs_amd)]
disc_means = [np.mean(rand_disc), sobol_disc, np.mean(lhs_disc)]
disc_stds  = [np.std(rand_disc),  0.0,        np.std(lhs_disc)]

x = np.array([0, 1, 2])
w = 0.35
ax = axes[3]
ax.bar(x - w/2, amd_means,  w, yerr=amd_stds,  capsize=4,
       color=[C_ORANGE, C_LIGHT, C_BLUE], label='AMD ↑', alpha=0.85)
ax2 = ax.twinx(); ax2.grid(False)
ax2.bar(x + w/2, disc_means, w, yerr=disc_stds, capsize=4,
        color=[C_ORANGE, C_LIGHT, C_BLUE], hatch='///', alpha=0.55, label='Discrepancy ↓')
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel('AMD (higher = better)')
ax2.set_ylabel('Discrepancy (lower = better)', labelpad=8)
ax.set_title('Space-filling metrics')
# Combined legend
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

fig.suptitle('Sampling method comparison (64 points, 4D design space)', y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig('figures/fig_sampling_comparison.png')
print("Saved → figures/fig_sampling_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — GPR kernel comparison + ARD length scales
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/TrainingData_256.csv')
X  = df[['W1', 'W2', 'R', 't']].values
y  = df['MaxVonMisesStress_MPa'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

kernels = {
    'RBF':         C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*4, length_scale_bounds=(1e-3, 1e3)),
    'Matérn 1.5':  C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*4, length_scale_bounds=(1e-3, 1e3), nu=1.5),
    'Matérn 2.5':  C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*4, length_scale_bounds=(1e-3, 1e3), nu=2.5),
    'Rat. Quad.':  C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
}

cv_means, cv_stds, length_scales = {}, {}, {}
for name, kernel in kernels.items():
    pipe = Pipeline([('sc', StandardScaler()),
                     ('gpr', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                                      normalize_y=True, random_state=SEED))])
    scores = cross_val_score(pipe, X_train, y_train, cv=kf,
                             scoring='neg_mean_absolute_percentage_error')
    cv_means[name] = -100 * scores.mean()
    cv_stds[name]  =  100 * scores.std()
    pipe.fit(X_train, y_train)
    ls = pipe.named_steps['gpr'].kernel_.k2.length_scale
    length_scales[name] = np.atleast_1d(ls)
    print(f"  {name}: CV MAPE = {cv_means[name]:.3f} ± {cv_stds[name]:.3f} %")

names = list(kernels.keys())
colors = [C_ORANGE if n != 'Matérn 2.5' else C_GREEN for n in names]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4))

# Panel (a): kernel CV MAPE
bars = ax_l.bar(names, [cv_means[n] for n in names],
                yerr=[cv_stds[n] for n in names],
                color=colors, capsize=5, edgecolor='white')
ax_l.set_ylabel('5-fold CV MAPE (%)')
ax_l.set_title('(a) Kernel comparison')
ax_l.tick_params(axis='x', rotation=15)
ax_l.text(bars[2].get_x() + bars[2].get_width()/2,
          cv_means['Matérn 2.5'] + cv_stds['Matérn 2.5'] + 0.03,
          '★ chosen', ha='center', fontsize=9, color=C_GREEN)

# Panel (b): ARD length scales for each kernel (only ARD kernels have 4 values)
feat_labels = [r'$W_1$', r'$W_2$', r'$R$', r'$t$']
x_pos = np.arange(4)
bar_w = 0.2
ard_names = [n for n in names if len(length_scales[n]) == 4]
palette = [C_ORANGE, C_LIGHT, C_GREEN, C_BLUE]
for k, (name, col) in enumerate(zip(ard_names, palette)):
    offset = (k - len(ard_names)/2 + 0.5) * bar_w
    ax_r.bar(x_pos + offset, length_scales[name], bar_w,
             label=name, color=col, edgecolor='white', alpha=0.85)
ax_r.set_xticks(x_pos); ax_r.set_xticklabels(feat_labels)
ax_r.set_ylabel('ARD length scale (larger = less influential)')
ax_r.set_title('(b) Learned ARD length scales')
ax_r.legend(fontsize=9)

fig.suptitle('GPR kernel selection', y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig('figures/fig_gpr_kernel_comparison.png')
print("Saved → figures/fig_gpr_kernel_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — GPR error evaluation (parity + 95% CI sorted prediction)
# ─────────────────────────────────────────────────────────────────────────────
best_pipe = joblib.load('models/gpr_best_pipeline.pkl')
scaler    = best_pipe.named_steps['scaler']
gpr       = best_pipe.named_steps['gpr']
X_test_s  = scaler.transform(X_test)
y_pred, y_std = gpr.predict(X_test_s, return_std=True)

err    = y_pred - y_test
rmse   = float(np.sqrt(np.mean(err**2)))
mape   = float(np.mean(np.abs(err / y_test)) * 100)
r2     = 1.0 - float(np.sum(err**2) / np.sum((y_test - y_test.mean())**2))

fig, (ax_p, ax_ci) = plt.subplots(1, 2, figsize=(11, 5))

# Panel (a): parity
lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
ax_p.errorbar(y_test, y_pred, yerr=1.96 * y_std, fmt='none',
              ecolor=C_BLUE, elinewidth=0.8, capsize=2, alpha=0.5)
sc = ax_p.scatter(y_test, y_pred, c=np.abs(err), cmap='viridis',
                  s=50, edgecolor='white', linewidth=0.5, zorder=3)
ax_p.plot(lims, lims, '--', color='#555', lw=1.3, label='Ideal (y = x)', zorder=2)
ax_p.set_xlim(lims); ax_p.set_ylim(lims); ax_p.set_aspect('equal')
ax_p.set_xlabel(r'True $\sigma_{\max}$ (MPa)')
ax_p.set_ylabel(r'Predicted $\sigma_{\max}$ (MPa)')
ax_p.set_title('(a) GPR parity plot')
ax_p.text(0.03, 0.97,
          f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f} MPa\nMAPE = {mape:.2f} %',
          transform=ax_p.transAxes, va='top', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#555', alpha=0.9))
ax_p.legend(loc='lower right', fontsize=9)
plt.colorbar(sc, ax=ax_p, label=r'$|$residual$|$ (MPa)', shrink=0.8)

# Panel (b): sorted 95% CI
sort_idx = np.argsort(y_test)
x_idx    = np.arange(len(y_test))
ax_ci.fill_between(x_idx,
                   y_pred[sort_idx] - 1.96 * y_std[sort_idx],
                   y_pred[sort_idx] + 1.96 * y_std[sort_idx],
                   alpha=0.25, color=C_BLUE, label='95% CI')
ax_ci.plot(x_idx, y_pred[sort_idx], color=C_BLUE, lw=1.6, label='GPR prediction')
ax_ci.scatter(x_idx, y_test[sort_idx], color=C_ORANGE, s=22,
              edgecolor='white', linewidth=0.4, zorder=3, label='True (FEA)')
ax_ci.set_xlabel('Sample index (sorted by true stress)')
ax_ci.set_ylabel(r'$\sigma_{\max}$ (MPa)')
ax_ci.set_title('(b) Predictions with 95% confidence interval')
ax_ci.legend(fontsize=9)

fig.suptitle('GPR surrogate — error evaluation on held-out test set (n = 52)',
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig('figures/fig_gpr_error_eval.png')
print("Saved → figures/fig_gpr_error_eval.png")
