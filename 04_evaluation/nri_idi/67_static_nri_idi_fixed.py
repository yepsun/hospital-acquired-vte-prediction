#!/usr/bin/env python3
"""
Script 67 (era / correctness fix): admission-level STATIC NRI / IDI on the
primary cohort, re-run under the same (correct) category-based NRI definition
used by the dynamic analysis, so static and dynamic numbers sit on one
definition.

Protocol mirrors scripts/61_static_oof_multiseed.py (seed-42 5-fold
StratifiedGroupKFold OOF on the era development pool, group=subject_id, LR /
XGBoost, 57-feature main set) and scripts/98_ajm_sensitivity_cohorts.py
(patient-level cluster bootstrap, 2,000 replicates, negatives capped at 50k,
Padua mapped to probability by univariate logistic (Platt) calibration on the
pool).

NRI (Pencina category-based):
  [P(up|event) - P(down|event)] + [P(down|non-event) - P(up|non-event)]
The old buggy variant (non-event sign reversed) is emitted alongside.

Output (new filename only):
  results_vte/ajm/new/output_era/static_oof_nri_idi_fixed_inclprior_excl24h.json
Run from the repository root that contains results_vte/.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEED = 42
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]
OUT = f'{RES}/ajm/new/output_era/static_oof_nri_idi_fixed_inclprior_excl24h.json'

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
padua_raw = np.nan_to_num(pool['padua_score'].values.astype(float),
                          nan=float(np.nanmedian(pool['padua_score'].values.astype(float))))
print(f'pool N={len(y)} events={y.sum()} ({y.mean()*100:.3f}%)', flush=True)


def fit_predict(kind, X_tr, y_tr, X_te):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nri(p_old, p_new, yb, thr):
    ev, nev = yb == 1, yb == 0
    up_e = ((p_new >= thr) & (p_old < thr) & ev).sum() / ev.sum()
    dn_e = ((p_new < thr) & (p_old >= thr) & ev).sum() / ev.sum()
    dn_n = ((p_new < thr) & (p_old >= thr) & nev).sum() / nev.sum()
    up_n = ((p_new >= thr) & (p_old < thr) & nev).sum() / nev.sum()
    return float((up_e - dn_e) + (dn_n - up_n))


def nri_buggy(p_old, p_new, yb, thr):
    ev, nev = yb == 1, yb == 0
    up_e = ((p_new >= thr) & (p_old < thr) & ev).sum() / ev.sum()
    dn_e = ((p_new < thr) & (p_old >= thr) & ev).sum() / ev.sum()
    up_n = ((p_new >= thr) & (p_old < thr) & nev).sum() / nev.sum()
    dn_n = ((p_new < thr) & (p_old >= thr) & nev).sum() / nev.sum()
    return float((up_e - dn_e) + (up_n - dn_n))


def idi(p_old, p_new, yb):
    ev, nev = yb == 1, yb == 0
    return float((p_new[ev].mean() - p_old[ev].mean())
                 - (p_new[nev].mean() - p_old[nev].mean()))


def ds(vals):
    vals = np.asarray(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


t0 = time.time()
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in ['lr', 'xgb']}
for tr, te in sgkf.split(X, y, groups):
    for k in ['lr', 'xgb']:
        oof[k][te] = fit_predict(k, X[tr], y[tr], X[te])
print(f'OOF done ({time.time()-t0:.0f}s): '
      f'LR AUC={roc_auc_score(y, oof["lr"]):.4f} '
      f'XGB AUC={roc_auc_score(y, oof["xgb"]):.4f}', flush=True)

platt = LogisticRegression(C=1.0, max_iter=2000).fit(padua_raw.reshape(-1, 1), y)
padua_prob = platt.predict_proba(padua_raw.reshape(-1, 1))[:, 1]
P = {**oof, 'padua_prob': padua_prob, 'padua_raw': padua_raw,
     'padua': padua_raw}

ev_rows = np.where(y == 1)[0]
neg_rows = np.where(y == 0)[0]
subj_codes, subj_uniq = pd.factorize(groups)
n_subj = len(subj_uniq)
ev_subj, neg_subj = subj_codes[ev_rows], subj_codes[neg_rows]

PAIRS = {'lr_vs_padua': ('padua_prob', 'lr'),
         'xgb_vs_padua': ('padua_prob', 'xgb'),
         'xgb_vs_lr': ('lr', 'xgb')}
nri_boot = {(a, t): [] for a in PAIRS for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in PAIRS}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua']}

rng = np.random.RandomState(SEED)
for b in range(N_BOOT):
    counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
    ev_mult = counts[ev_subj]
    take = ev_mult > 0
    if take.sum() < 2:
        continue
    be_rows = np.repeat(ev_rows[take], ev_mult[take])
    bn_rows = neg_rows[counts[neg_subj] > 0]
    if len(bn_rows) > NEG_CAP:
        bn_rows = rng.choice(bn_rows, NEG_CAP, replace=False)
    rows = np.concatenate([be_rows, bn_rows])
    yb = y[rows]
    a = {k: fast_auc(yb, P[k][rows]) for k in auc_boot}
    for k in a:
        auc_boot[k].append(a[k])
    for name, (old, new) in PAIRS.items():
        for t in NRI_THRESHOLDS:
            nri_boot[(name, t)].append(nri(P[old][rows], P[new][rows], yb, t))
        idi_boot[name].append(idi(P[old][rows], P[new][rows], yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)

out = {
    'method': 'seed-42 5-fold StratifiedGroupKFold OOF on the era development '
              'pool (group=subject_id); patient-level cluster bootstrap 2000 '
              'replicates, negatives downsampled to 50k; Padua probability from '
              'univariate logistic (Platt) calibration on the pool; NRI uses the '
              'standard Pencina category-based definition (up_e-dn_e)+(dn_n-up_n)',
    'n_pool': int(len(y)), 'n_events': int(y.sum()),
    'oof_auc': {k: float(roc_auc_score(y, P[k] if k != 'padua' else padua_raw))
                for k in ['lr', 'xgb', 'padua']},
    'oof_auc_ci': {k: {'auc': float(roc_auc_score(y, P[k] if k != 'padua' else padua_raw)),
                       'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
                   for k, v2 in auc_boot.items()},
    'nri': {f'{name}_thr{t}': {'point': nri(P[old], P[new], y, t),
                               'point_old_buggy': nri_buggy(P[old], P[new], y, t),
                               'ci': [float(v) for v in np.percentile(
                                   nri_boot[(name, t)], [2.5, 97.5])]}
            for name, (old, new) in PAIRS.items() for t in NRI_THRESHOLDS},
    'idi': {name: {'point': idi(P[old], P[new], y),
                   'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
            for name, (old, new), v2 in
            [(n, p, idi_boot[n]) for n, p in PAIRS.items()]},
    'n_replicates_used': len(auc_boot['lr']),
}
json.dump(out, open(OUT, 'w'), indent=2)
print('saved ->', OUT, f'({time.time()-t0:.0f}s)', flush=True)
for k, d in out['oof_auc_ci'].items():
    print(f"  OOF AUC {k}: {d['auc']:.4f} [{d['ci'][0]:.4f},{d['ci'][1]:.4f}]", flush=True)
for k, d in out['nri'].items():
    print(f"  NRI {k}: correct={d['point']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] (old buggy {d['point_old_buggy']:+.4f})",
          flush=True)
for k, d in out['idi'].items():
    print(f"  IDI {k}: {d['point']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]", flush=True)
