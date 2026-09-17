#!/usr/bin/env python3
"""
Script 58 (VTE revision): publication-quality Figures 2 & 3 (primary cohort).

Redesign of arm_ipe/57 figures:
  - Fig 2: reliability curves with decile bins formed on PRE-recalibration
    scores (equal row counts preserved even though isotonic output is tied),
    percent axes, journal styling.
  - Fig 3: two-panel DCA on out-of-fold predictions. Left: net benefit scaled
    per 1,000 admissions over thresholds 0.1-2.0% with the 0.5-2% clinical
    operating range shaded. Right: incremental net benefit vs Padua.

Outputs (ajm/new/output_era/): Figure_2_calibration_v3.png/.svg,
Figure_3_dca_v3.png/.svg, revision_threshold_calib_v3.json
Run from the repository root that contains results_vte/.
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEED = 42

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

C_PADUA, C_LR, C_XGB = '#636363', '#1a6fb5', '#c22e38'

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
padua_raw = pool['padua_score'].values.astype(float)
N = len(y)
print(f'pool N={N} events={y.sum()}', flush=True)

platt = LogisticRegression(C=1.0, max_iter=2000).fit(padua_raw.reshape(-1, 1), y)
padua_prob = platt.predict_proba(padua_raw.reshape(-1, 1))[:, 1]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {'lr': np.zeros(N), 'xgb': np.zeros(N)}
for tr, te in sgkf.split(X, y, groups):
    imp = SimpleImputer(strategy='median')
    m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    m.fit(imp.fit_transform(X[tr]), y[tr])
    oof['lr'][te] = m.predict_proba(imp.transform(X[te]))[:, 1]
    mx = XGBClassifier(**XGB_PARAMS)
    mx.fit(X[tr], y[tr])
    oof['xgb'][te] = mx.predict_proba(X[te])[:, 1]
    print(f'fold done: lr auc={roc_auc_score(y[te], oof["lr"][te]):.4f} '
          f'xgb auc={roc_auc_score(y[te], oof["xgb"][te]):.4f}', flush=True)

auc_lr = roc_auc_score(y, oof['lr'])
auc_xgb = roc_auc_score(y, oof['xgb'])
print(f'OOF AUC lr={auc_lr:.4f} xgb={auc_xgb:.4f}', flush=True)

iso = {k: IsotonicRegression(out_of_bounds='clip').fit(oof[k], y) for k in oof}
probs_cal = {'padua': padua_prob,
             'lr': iso['lr'].predict(oof['lr']),
             'xgb': iso['xgb'].predict(oof['xgb'])}

# ---- calibration data: decile bins on the continuous (pre-isotonic) score --
def calib_curve(p_plot, p_order, bins=10):
    edges = np.quantile(p_order, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p_order, edges[1:-1], right=False), 0, bins - 1)
    mp, of, nn = [], [], []
    for b in range(bins):
        m_ = idx == b
        if m_.sum() >= 50:
            mp.append(float(np.mean(p_plot[m_])))
            of.append(float(np.mean(y[m_])))
            nn.append(int(m_.sum()))
    return {'bin_mean_predicted': mp, 'bin_observed_fraction': of, 'bin_n': nn}

calib = {
    'padua_platt': calib_curve(probs_cal['padua'], padua_prob),
    'lr_pre':      calib_curve(oof['lr'], oof['lr']),
    'lr_post':     calib_curve(probs_cal['lr'], oof['lr']),
    'xgb_pre':     calib_curve(oof['xgb'], oof['xgb']),
    'xgb_post':    calib_curve(probs_cal['xgb'], oof['xgb']),
}
for k, v in calib.items():
    print(f"{k}: {len(v['bin_mean_predicted'])} pts "
          f"(rows/bin {min(v['bin_n'])}-{max(v['bin_n'])})", flush=True)

# ---- uniform-width bins of the RAW (pre-recalibration) predictions --------
def uniform_bin_curve(p):
    edges = np.linspace(0, p.max(), 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    out = []
    for b in range(10):
        m_ = idx == b
        if m_.sum():
            out.append({'pred': float(np.mean(p[m_])),
                        'obs': float(np.mean(y[m_])), 'n': int(m_.sum())})
    return out

calib['lr_pre_uniform'] = {'bins': uniform_bin_curve(oof['lr'])}
calib['xgb_pre_uniform'] = {'bins': uniform_bin_curve(oof['xgb'])}
for k in ['lr_pre_uniform', 'xgb_pre_uniform']:
    print(k, [(round(b['pred'], 4), round(b['obs'], 5), b['n'])
              for b in calib[k]['bins']], flush=True)

# ---- DCA on OOF ------------------------------------------------------------
def net_benefit(p, pt):
    flag = p >= pt
    tp = (flag & (y == 1)).sum()
    fp = (flag & (y == 0)).sum()
    return tp / N - fp / N * pt / (1 - pt)

prev = y.mean()
thr = np.linspace(0.0005, 0.03, 120)
dca = {m: np.array([net_benefit(p, t) for t in thr]) * 1e3
       for m, p in probs_cal.items()}
treat_all = (prev - (1 - prev) * thr / (1 - thr)) * 1e3
band = (thr >= 0.005) & (thr <= 0.02)
nb_band = {m: float(dca[m][band].mean()) for m in dca}
for m, v in nb_band.items():
    print(f'mean NB 0.5-2% {m}: {v:+.4f} per 1,000', flush=True)

# ---- shared style ----------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.linewidth': 0.9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dcdcdc', 'grid.linewidth': 0.55,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42})

OUT = f'{RES}/ajm/new/output_era'

# ---- Figure 2: small multiples, one panel per score ------------------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharex=True, sharey=True)
xmax = 0.90

panel_specs = [
    ('Padua (Platt-mapped)', [('padua_platt', 'Padua (Platt)', C_PADUA, 's',
                               'solid', 'white')]),
    ('Logistic regression', [('lr_pre', 'Uncalibrated', C_LR, 'o',
                              (0, (3, 2)), 'white'),
                             ('lr_post', 'Isotonic-recalibrated', C_LR, 'o',
                              'solid', C_LR)]),
    ('XGBoost', [('xgb_pre', 'Uncalibrated', C_XGB, '^',
                  (0, (3, 2)), 'white'),
                 ('xgb_post', 'Isotonic-recalibrated', C_XGB, '^',
                  'solid', C_XGB)]),
]
for j, (ttl, series) in enumerate(panel_specs):
    ax = axes[j]
    ax.plot([0, xmax], [0, xmax], color='#999999', ls='--', lw=1.0,
            label='Ideal', zorder=1)
    for key, lab, c, mkr, ls_, mfc in series:
        dd = calib[key]
        ax.plot(np.array(dd['bin_mean_predicted']) * 100,
                np.array(dd['bin_observed_fraction']) * 100,
                mkr + '-', color=c, ms=4.8, lw=1.6, ls=ls_, mec=c, mfc=mfc,
                mew=1.2, label=lab, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.04, 0.84)
    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel('Mean predicted VTE risk per bin (%)')
    if j == 0:
        ax.set_ylabel('Observed VTE fraction (%)')
    ax.legend(frameon=False, fontsize=7.8, loc='upper left',
              handlelength=2.2, borderaxespad=0.3)
    ax.set_title(f"({'abc'[j]}) {ttl}", fontsize=9.5, loc='left', pad=5)

fig.suptitle('')
fig.tight_layout(w_pad=1.6)
fig.savefig(f'{OUT}/Figure_2_calibration_v3.png', dpi=300)
fig.savefig(f'{OUT}/Figure_2_calibration_v3.svg')
plt.close(fig)
print('saved Figure_2_calibration_v3.png/svg', flush=True)

# ---- Figure 3: DCA, net benefit per 1,000 + incremental panel ---------------
lo, hi = thr[0] * 100, 2.0
xr = (thr * 100 <= hi)
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9),
                         gridspec_kw={'width_ratios': [1.12, 1]})
ax = axes[0]
ax.axvspan(0.5, 2.0, color='#f0f0ee', zorder=0)
ax.axhline(0, color='#555555', ls=':', lw=1.0, label='Treat none', zorder=1)
ax.plot(thr[xr] * 100, treat_all[xr], color='#222222', ls='-.',
        lw=1.3, label='Treat all', zorder=2)
for m, lab, c in [('padua', 'Padua (Platt)', C_PADUA),
                  ('lr', 'LR (isotonic)', C_LR),
                  ('xgb', 'XGBoost (isotonic)', C_XGB)]:
    ax.plot(thr[xr] * 100, dca[m][xr], color=c, lw=1.8, label=lab, zorder=3)
ax.set_xlim(lo, hi)
ax.set_ylim(-21, 3.5)
ax.set_xlabel('Threshold probability $p_t$ (%)')
ax.set_ylabel('Net benefit (per 1,000 admissions)')
ax.legend(frameon=False, fontsize=8.5, loc='lower left',
          handlelength=2.2, borderaxespad=0.4)
ax.text(1.24, 2.45, 'clinical operating\nrange 0.5–2%', ha='center',
        va='top', fontsize=7.6, color='#777777', style='italic')

ax = axes[1]
ax.axvspan(0.5, 2.0, color='#f0f0ee', zorder=0)
ax.axhline(0, color='#555555', ls=':', lw=1.1, zorder=1)
ax.plot([0.5, 2.0], [0, 0], lw=0, zorder=1)
delta_lr = dca['lr'] - dca['padua']
delta_xgb = dca['xgb'] - dca['padua']
l1, = ax.plot(thr[xr] * 100, delta_xgb[xr], color=C_XGB, lw=1.9,
              label='ΔNB vs Padua: XGBoost')
l2, = ax.plot(thr[xr] * 100, delta_lr[xr], color=C_LR, lw=1.9,
              label='ΔNB vs Padua: LR')
ymax = 1.25 * max(delta_xgb[xr].max(), delta_lr[xr].max())
ax.set_xlim(lo, hi)
ax.set_ylim(-0.08, ymax)
ax.set_xlabel('Threshold probability $p_t$ (%)')
ax.set_ylabel('Incremental net benefit\nvs Padua (per 1,000)')
ax.annotate('XGBoost', xy=(hi, delta_xgb[xr][-1]), xytext=(-2, 5),
            textcoords='offset points', ha='right', fontsize=8.5,
            color=C_XGB, fontweight='bold')
ax.annotate('Logistic regression', xy=(hi, delta_lr[xr][-1]),
            xytext=(-2, -11), textcoords='offset points', ha='right',
            fontsize=8.5, color=C_LR, fontweight='bold')
fig.tight_layout(w_pad=2.4)
fig.savefig(f'{OUT}/Figure_3_dca_v3.png', dpi=300)
fig.savefig(f'{OUT}/Figure_3_dca_v3.svg')
plt.close(fig)
print('saved Figure_3_dca_v3.png/svg', flush=True)

out = {'oof_auc': {'lr': auc_lr, 'xgb': auc_xgb},
       'calibration': calib,
       'dca': {'thresholds_pct': (thr * 100).tolist(),
               'treat_all_per1000': treat_all.tolist(),
               'models_per1000': {m: v.tolist() for m, v in dca.items()},
               'mean_nb_band_per1000': nb_band},
       'meta': {'protocol': 'seed-42 5-fold StratifiedGroupKFold OOF '
                            '(group=subject_id), primary cohort incl prior VTE, '
                            'excl 24-h-window events; bins are deciles of the '
                            'pre-recalibration score; DCA on OOF probabilities '
                            '(isotonic-recalibrated; Padua Platt)',
                'n_pool': N, 'n_events': int(y.sum())}}
with open(f'{OUT}/revision_threshold_calib_v3.json', 'w') as f:
    json.dump(out, f, indent=2)
print('saved revision_threshold_calib_v3.json', flush=True)
