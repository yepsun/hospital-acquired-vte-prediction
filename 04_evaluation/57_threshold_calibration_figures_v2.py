#!/usr/bin/env python3
"""
Script 57 (VTE revision): reproduce seed-42 static OOF predictions, then
  1. threshold operating characteristics (0.5% / 1% / 2%) for Padua, LR, XGB
  2. calibration curve data (10 equal-width bins, pre/post isotonic)
  3. decision curve data on OOF
  4. Figures: calibration plot + DCA plot (300 dpi PNG)

Outputs:
  results_vte/revision_threshold_calib_inclprior_excl24h.json
  results_vte/ajm/new/output/Figure_2_calibration_inclprior_excl24h.png, results_vte/ajm/new/output/Figure_3_dca_inclprior_excl24h.png
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data/primary_inclprior_excl24h_model_dataset.parquet')
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
print(f'OOF AUC lr={auc_lr:.4f} xgb={auc_xgb:.4f} '
      f'(stored 0.8102/0.8277)', flush=True)

iso = {k: IsotonicRegression(out_of_bounds='clip').fit(oof[k], y) for k in oof}
oof_cal = {k: iso[k].predict(oof[k]) for k in oof}

PROBS = {'padua': padua_prob, 'lr': oof_cal['lr'], 'xgb': oof_cal['xgb']}

# ── 1. threshold operating characteristics ──
def op_char(p, thr):
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {'threshold': thr, 'n_flagged': int(flag.sum()),
            'flag_rate_pct': 100.0 * flag.mean(),
            'alerts_per_1000': 1000.0 * flag.mean(),
            'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'ppv': tp / (tp + fp) if tp + fp else None,
            'npv': tn / (tn + fn) if tn + fn else None}

thresholds = [0.005, 0.01, 0.02]
op = {m: [op_char(p, t) for t in thresholds] for m, p in PROBS.items()}
for m in op:
    for r in op[m]:
        print(f"{m} thr={r['threshold']}: flag={r['flag_rate_pct']:.1f}% "
              f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} "
              f"ppv={r['ppv']:.4f} npv={r['npv']:.5f}", flush=True)

# ── 2. calibration curves (10 quantile bins: equal rows per bin, so the
#       high-risk tail is adequately powered under 0.16% prevalence) ──
def calib_curve(p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = 0.0, p.max() + 1e-12
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    mp, of, nn = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 50:
            mp.append(float(p[m].mean()))
            of.append(float(y[m].mean()))
            nn.append(int(m.sum()))
    return {'bin_mean_predicted': mp, 'bin_observed_fraction': of, 'bin_n': nn}

calib = {'lr_pre': calib_curve(oof['lr']), 'lr_post': calib_curve(oof_cal['lr']),
         'xgb_pre': calib_curve(oof['xgb']), 'xgb_post': calib_curve(oof_cal['xgb']),
         'padua_platt': calib_curve(padua_prob)}

# ── 3. DCA on OOF ──
def net_benefit(p, pt):
    flag = p >= pt
    tp = (flag & (y == 1)).sum()
    fp = (flag & (y == 0)).sum()
    return tp / N - fp / N * pt / (1 - pt)

dca_pts = np.linspace(0.001, 0.05, 100)
prev = y.mean()
dca = {'thresholds': dca_pts.tolist(),
       'treat_all': [prev - (1 - prev) * pt / (1 - pt) for pt in dca_pts],
       'models': {m: [float(net_benefit(p, pt)) for pt in dca_pts]
                  for m, p in PROBS.items()}}
band = (dca_pts >= 0.005) & (dca_pts <= 0.02)
for m in PROBS:
    arr = np.array(dca['models'][m])
    print(f'mean NB 0.5-2% {m}: {arr[band].mean():.6f}', flush=True)

# ── 4. figures ──
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})

xmax = max(max(calib[k]['bin_mean_predicted'])
           for k in calib if calib[k]['bin_mean_predicted']) * 1.15
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, xmax], [0, xmax], 'k--', lw=0.8, label='Ideal')
for key, lab, c, mkr in [('padua_platt', 'Padua (Platt)', '#7f7f7f', 's'),
                         ('lr_post', 'LR (isotonic)', '#1f77b4', 'o'),
                         ('xgb_post', 'XGBoost (isotonic)', '#d62728', '^')]:
    d = calib[key]
    ax.plot(d['bin_mean_predicted'], d['bin_observed_fraction'], mkr + '-',
            color=c, ms=4, lw=1.2, label=lab)
for key, lab, c in [('lr_pre', 'LR (uncalibrated)', '#1f77b4'),
                    ('xgb_pre', 'XGBoost (uncalibrated)', '#d62728')]:
    d = calib[key]
    ax.plot(d['bin_mean_predicted'], d['bin_observed_fraction'], ':', color=c,
            lw=1.0, alpha=0.6, label=lab)
ax.set_xlabel('Mean predicted risk (quantile bin)')
ax.set_ylabel('Observed VTE fraction')
ax.set_xlim(0, xmax)
ax.set_ylim(0, xmax)
ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig('results_vte/ajm/new/output/Figure_2_calibration_inclprior_excl24h.png', dpi=300)
print('saved results_vte/ajm/new/output/Figure_2_calibration_inclprior_excl24h.png', flush=True)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(dca_pts * 100, dca['treat_all'], 'k--', lw=0.8, label='Treat all')
ax.axhline(0, color='k', lw=0.8, ls=':', label='Treat none')
for m, lab, c in [('padua', 'Padua (Platt)', '#7f7f7f'),
                  ('lr', 'LR (isotonic)', '#1f77b4'),
                  ('xgb', 'XGBoost (isotonic)', '#d62728')]:
    ax.plot(dca_pts * 100, dca['models'][m], lw=1.4, color=c, label=lab)
ax.set_xlabel('Threshold risk (%)')
ax.set_ylabel('Net benefit')
ax.set_xlim(0, 5)
ax.set_ylim(-0.001, 0.005)
ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout()
fig.savefig('results_vte/ajm/new/output/Figure_3_dca_inclprior_excl24h.png', dpi=300)
print('saved results_vte/ajm/new/output/Figure_3_dca_inclprior_excl24h.png', flush=True)

out = {'oof_auc': {'lr': auc_lr, 'xgb': auc_xgb},
       'threshold_operating_characteristics': op,
       'calibration': calib, 'dca': dca,
       'meta': {'protocol': 'seed-42 5-fold StratifiedGroupKFold OOF (group=subject_id); '
                            'LR/XGB probabilities isotonic-recalibrated on OOF (apparent); '
                            'Padua via univariate Platt on pool',
                'n_pool': N, 'n_events': int(y.sum())}}
with open(f'{RES}/ajm/new/output/revision_threshold_calib_inclprior_excl24h.json', 'w') as f:
    json.dump(out, f, indent=2)
print('saved results_vte/revision_threshold_calib_inclprior_excl24h.json', flush=True)
