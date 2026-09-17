#!/usr/bin/env python3
"""
Script 61b: multi-seed static-model OOF inference for the *corrected*
"include prior VTE, retain 24-h-window events" arm.

Byte-identical protocol to 61_static_oof_multiseed.py; only the dataset, the
splits file and the output path are repointed, because 61 hard-codes them.
Input cohort is data_era/sens_inclprior_fixed_model_dataset.parquet
(381,191 rows / 2,145 events), rebuilt by
build_inclprior_cohort_era_fixed.py.

For seeds 42/43/44: 5-fold StratifiedGroupKFold (by subject) OOF predictions
for LR / XGBoost on the 57-feature main set (same protocol as scripts 40/41),
plus raw Padua and IMPROVE scores on the same rows. Patient-level cluster
bootstrap (2000 reps, negatives capped at 50k) gives delta-AUC vs Padua and
vs IMPROVE per seed.

Output: results_vte/ajm/new/output_era/static_oof_multiseed_inclprior_fixed.json
"""
import json, warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/sens_inclprior_fixed_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data_era/sens_inclprior_fixed_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values


def score_col(col):
    s = pool[col].values.astype(float)
    return np.nan_to_num(s, nan=float(np.nanmedian(s)))


padua = score_col('padua_score')
improve = score_col('improve_score')


def fit_predict(kind, X_tr, y_tr, X_te, seed):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def ds(vals):
    vals = np.asarray(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(np.mean(vals)), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def fast_auc(yb, pb):
    """AUC via Mann-Whitney rank statistic (ties handled by rankdata)."""
    from scipy.stats import rankdata
    r = rankdata(pb)
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


results = {'method': 'per-seed 5-fold StratifiedGroupKFold (group subject_id) '
                     'OOF; patient-level cluster bootstrap 2000 reps, '
                     'negatives downsampled to 50k per replicate',
           'n_pool': int(len(y)), 'n_events': int(y.sum()),
           'seeds': {}}

for seed in SEEDS:
    print(f'== seed {seed} ==', flush=True)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=seed)
    oof = {k: np.zeros(len(y)) for k in ['lr', 'xgb']}
    for tr, te in sgkf.split(X, y, groups):
        oof['lr'][te] = fit_predict('lr', X[tr], y[tr], X[te], seed)
        oof['xgb'][te] = fit_predict('xgb', X[tr], y[tr], X[te], seed)
    aucs = {'lr': float(roc_auc_score(y, oof['lr'])),
            'xgb': float(roc_auc_score(y, oof['xgb'])),
            'padua': float(roc_auc_score(y, padua)),
            'improve': float(roc_auc_score(y, improve))}
    print('  OOF AUC:', {k: round(v, 4) for k, v in aucs.items()}, flush=True)

    subj_codes, subj_uniq = pd.factorize(groups)
    n_subj = len(subj_uniq)
    ev_rows = np.where(y == 1)[0]
    neg_rows = np.where(y == 0)[0]
    ev_subj, neg_subj = subj_codes[ev_rows], subj_codes[neg_rows]
    rng = np.random.RandomState(seed)
    deltas = {'lr_vs_padua': [], 'xgb_vs_padua': [],
              'lr_vs_improve': [], 'xgb_vs_improve': [],
              'xgb_vs_lr': []}
    P = {**oof, 'padua': padua, 'improve': improve}
    for b in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        ev_mult = counts[ev_subj]
        take = ev_mult > 0
        if take.sum() < 2:
            continue
        be = np.repeat(ev_rows[take], ev_mult[take])
        bn = neg_rows[counts[neg_subj] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        rows = np.concatenate([be, bn])
        yb = y[rows]
        a = {k: fast_auc(yb, P[k][rows]) for k in P}
        deltas['lr_vs_padua'].append(a['lr'] - a['padua'])
        deltas['xgb_vs_padua'].append(a['xgb'] - a['padua'])
        deltas['lr_vs_improve'].append(a['lr'] - a['improve'])
        deltas['xgb_vs_improve'].append(a['xgb'] - a['improve'])
        deltas['xgb_vs_lr'].append(a['xgb'] - a['lr'])
        if (b + 1) % 500 == 0:
            print(f'  boot {b + 1}/{N_BOOT}', flush=True)
    results['seeds'][str(seed)] = {
        'oof_auc': aucs,
        'delta_auc': {k: ds(v) for k, v in deltas.items()},
    }
    for k, d in results['seeds'][str(seed)]['delta_auc'].items():
        print(f"  dAUC {k}: {d['estimate']:+.4f} "
              f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}",
              flush=True)

with open(f'{RES}/ajm/new/output_era/static_oof_multiseed_inclprior_fixed.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved -> results_vte/ajm/new/output_era/static_oof_multiseed_inclprior_fixed.json',
      flush=True)
