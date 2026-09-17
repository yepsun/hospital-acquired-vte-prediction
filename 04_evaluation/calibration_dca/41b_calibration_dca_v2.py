#!/usr/bin/env python3
"""
Script 41 (VTE M4 / Task 4): probability calibration, AUPRC/PR curves, DCA.

  - seed42 single 5-fold OOF predictions (train+val pool, same protocol as
    scripts/40_static_models.py); isotonic regression fit on OOF per model
    (LR / XGB); pre/post Brier, ECE (10 uniform bins) + calibration curve data
    (bin mean predicted vs observed fraction) saved for plotting.
  - test set: models retrained on full pool, isotonic (from OOF) applied to
    test predictions; PR curve data (50 threshold points) + AUPRC for
    LR / XGB / Padua.
  - DCA on test: net benefit over thresholds 0.1%-5% (100 points),
    LR / XGB / Padua vs treat-all vs treat-none. Padua probability via
    univariate logistic (Platt) calibration fit on the pool (script 40).

Checks: post-calibration ECE < 0.05; model net benefit > Padua over 0.5-2%.

Outputs: results_vte/calibration_dca.json
"""
import json, time, warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEED = 42

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[df['hadm_id'].isin(set(splits['test']))].reset_index(drop=True)
X, y = pool[FEATS].values.astype(np.float32), pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
Xt, yt = test[FEATS].values.astype(np.float32), test['vte_event'].values.astype(int)
print(f'pool N={len(y)} events={y.sum()} | test N={len(yt)} events={yt.sum()}',
      flush=True)


def fit_predict(kind, X_tr, y_tr, X_te):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def cal_stats(yb, p):
    pt, pp = calibration_curve(yb, p, n_bins=10)
    return {'brier': float(brier_score_loss(yb, p)),
            'ece': float(np.mean(np.abs(pt - pp))),
            'curve': {'bin_mean_predicted': [float(v) for v in pp],
                      'bin_observed_fraction': [float(v) for v in pt]}}


# ── Step 1: seed42 OOF + isotonic calibration ──
print('OOF (seed42, 5-fold) ...', flush=True)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
oofs = {k: np.zeros(len(y)) for k in ['lr', 'xgb']}
for tr, te in sgkf.split(X, y, groups):
    for k in ['lr', 'xgb']:
        oofs[k][te] = fit_predict(k, X[tr], y[tr], X[te])
    print(f'  fold done ({time.strftime("%H:%M:%S")})', flush=True)

iso = {k: IsotonicRegression(out_of_bounds='clip').fit(oofs[k], y)
       for k in ['lr', 'xgb']}

results = {'calibration': {}}
for k in ['lr', 'xgb']:
    pre, post = cal_stats(y, oofs[k]), cal_stats(y, iso[k].predict(oofs[k]))
    results['calibration'][k] = {
        'pre': pre, 'post': post,
        'oof_auc': float(roc_auc_score(y, oofs[k])),
        'note': 'isotonic regression fit on seed42 5-fold OOF predictions; '
                'post-calibration metrics are apparent (in-sample on OOF)'}
    print(f"{k.upper()}: Brier {pre['brier']:.5f}->{post['brier']:.5f}  "
          f"ECE {pre['ece']:.4f}->{post['ece']:.4f}", flush=True)

# ── Step 2: test-set PR curves + AUPRC ──
print('test-set predictions ...', flush=True)
test_prob = {}
for k in ['lr', 'xgb']:
    p_raw = fit_predict(k, X, y, Xt)
    test_prob[k] = iso[k].predict(p_raw)  # isotonic-calibrated (monotonic: AUC/PR unchanged)

platt = LogisticRegression(C=1.0, max_iter=2000).fit(
    pool['padua_score'].values.reshape(-1, 1), y)
padua_prob_t = platt.predict_proba(test['padua_score'].values.reshape(-1, 1))[:, 1]
test_prob['padua'] = padua_prob_t

PR_THR = np.geomspace(1e-4, 0.5, 50)
results['pr_curves'] = {'thresholds': [float(t) for t in PR_THR], 'models': {}}
for k, p in test_prob.items():
    prec, rec = [], []
    for t in PR_THR:
        pos = p >= t
        tp = int((pos & (yt == 1)).sum())
        prec.append(float(tp / pos.sum()) if pos.sum() else 1.0)
        rec.append(float(tp / yt.sum()))
    results['pr_curves']['models'][k] = {
        'auprc': float(average_precision_score(yt, p)),
        'auc': float(roc_auc_score(yt, p)),
        'precision': prec, 'recall': rec,
        'note': ('isotonic-calibrated probabilities (LR/XGB); Padua = raw score '
                 'Platt-calibrated on pool' if k == 'padua' else
                 'isotonic-calibrated probabilities')}
    print(f"  {k.upper()}: test AUC={roc_auc_score(yt, p):.4f} "
          f"AUPRC={average_precision_score(yt, p):.4f}", flush=True)

# ── Step 3: DCA ──
dca_thr = np.linspace(0.001, 0.05, 100)
n_t, ev_t = len(yt), float(yt.sum())


def net_benefit(p, thr):
    pos = p >= thr
    tp = (pos & (yt == 1)).sum()
    fp = (pos & (yt == 0)).sum()
    return tp / n_t - fp / n_t * thr / (1 - thr)


prev = ev_t / n_t
nb_all = prev - (1 - prev) * dca_thr / (1 - dca_thr)
results['dca'] = {
    'thresholds': [float(t) for t in dca_thr],
    'treat_all': [float(v) for v in nb_all],
    'treat_none': [0.0] * len(dca_thr),
    'models': {k: [float(net_benefit(p, t)) for t in dca_thr]
               for k, p in test_prob.items()},
    'note': 'test set net benefit = TP/n - FP/n * pt/(1-pt); LR/XGB use '
            'isotonic-calibrated probabilities; Padua Platt-calibrated'}

# ── Step 4: checks ──
band = (dca_thr >= 0.005) & (dca_thr <= 0.02)
checks = {'post_ece_lt_0.05': {k: results['calibration'][k]['post']['ece'] < 0.05
                               for k in ['lr', 'xgb']},
          'dca_band_0.5_2pct_mean_nb': {
              k: float(np.mean(np.array(v)[band]))
              for k, v in results['dca']['models'].items()},
          'model_beats_padua_in_band': {
              k: bool(np.all(np.array(results['dca']['models'][k])[band]
                             > np.array(results['dca']['models']['padua'])[band]))
              for k in ['lr', 'xgb']}}
results['checks'] = checks
print('checks:', json.dumps(checks, indent=1), flush=True)

results['meta'] = {'features': FEATS, 'n_pool': len(y), 'n_test': len(yt),
                   'n_test_events': int(yt.sum()),
                   'protocol': 'OOF StratifiedGroupKFold 5-fold seed42 '
                               '(group=subject_id); isotonic on OOF; models '
                               'retrained on full pool for test predictions'}
with open(f'{RES}/ajm/new/output_era/calibration_dca_inclprior_excl24h.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved -> results_vte/calibration_dca.json', flush=True)
