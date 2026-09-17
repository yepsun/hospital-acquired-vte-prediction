#!/usr/bin/env python3
"""Strict-vs-main control comparison, like-for-like and leakage-fixed.

The published table (output_era/strict_control_apples2apples_inclprior_excl24h.json,
produced by 50c_strict_control_apples2apples_v2.py) has two defects:

  1. estimand -- the "main control pool" column is evaluated on the WHOLE
     379,720-row cohort, i.e. including the 66,656-row / 120-event test
     partition that the primary analysis holds out, whereas the primary
     estimand (script 98:170-179) is the development pool (313,064 / 554).
  2. leakage -- eval_models() calls `imp.fit_transform(Xf)` on every row of
     the subset (:88), so each fold's training features carry the imputation
     statistics of its own validation rows.

This script recomputes the same four models (Padua, IMPROVE, LR, XGBoost)
over the same fold scheme (StratifiedGroupKFold 3x5, seeds 42/43/44, XGBoost
random_state fixed at 42 as in script 98) on four row subsets:

  dev_main     development pool of the primary arm    (313,064 / 554)
  dev_strict   its events + strict imaging-clean controls drawn from it
  whole_main   published whole-cohort main pool       (379,720 / 674)
  whole_strict published whole-cohort strict pool     ( 21,143 / 674)

and under two imputation policies:

  leaky   imputer fitted on all rows of the subset (reproduces 50c)
  fixed   imputer fitted on the training rows of each fold only

Strict control identities are read from the published
output_era/strict_control_ids_inclprior_excl24h.parquet; the radiology-note
scan that produced them is not repeated. XGBoost is trained on the imputed
matrix in both policies, so leaky-vs-fixed isolates the imputer's fitting
scope. Writes NEW files only.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte/ajm/new'
SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000
MODELS = ['padua', 'improve', 'lr', 'xgb']

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=42)

FEATS = json.load(open('results_vte/feature_sets_v2.json'))['main']
df = pd.read_parquet(f'{RES}/data_era/primary_inclprior_excl24h_model_dataset.parquet')
splits = json.load(open(f'{RES}/data_era/primary_inclprior_excl24h_splits.json'))
pool_ids = set(splits['train']) | set(splits['val'])
strict_ids = set(pd.read_parquet(
    f'{RES}/output_era/strict_control_ids_inclprior_excl24h.parquet')['hadm_id'])

non = df[df['vte_event'] == 0]
assert strict_ids <= set(non['hadm_id']), 'strict ids are not all main-pool non-events'
ev_ids = set(df.loc[df.vte_event == 1, 'hadm_id'])
dev_ev_ids = ev_ids & pool_ids
dev_strict_ids = strict_ids & pool_ids
assert dev_strict_ids <= set(non['hadm_id'])
print(f'primary cohort {len(df):,} / {int(df.vte_event.sum()):,} events | '
      f'strict controls {len(strict_ids):,} (dev pool {len(dev_strict_ids):,})',
      flush=True)

SUBSETS = {
    'dev_main': df[df['hadm_id'].isin(pool_ids)],
    'dev_strict': df[df['hadm_id'].isin(dev_ev_ids | dev_strict_ids)],
    'whole_main': df,
    'whole_strict': df[df['hadm_id'].isin(ev_ids | strict_ids)],
}
assert len(SUBSETS['dev_main']) == 313_064, len(SUBSETS['dev_main'])
assert int(SUBSETS['dev_main'].vte_event.sum()) == 554
assert len(SUBSETS['dev_strict']) == 554 + 16_931, len(SUBSETS['dev_strict'])
assert int(SUBSETS['dev_strict'].vte_event.sum()) == 554
assert len(SUBSETS['whole_main']) == 379_720
assert len(SUBSETS['whole_strict']) == 674 + 20_469, len(SUBSETS['whole_strict'])
for tag, sub in SUBSETS.items():
    print(f'   {tag}: {len(sub):,} rows / {int(sub.vte_event.sum()):,} events',
          flush=True)

only = [s for s in (sys.argv[sys.argv.index('--only') + 1].split(',')
                    if '--only' in sys.argv else []) if s]
if only:
    unknown = set(only) - set(SUBSETS)
    assert not unknown, f'unknown subset(s): {unknown}'
    SUBSETS = {k: v for k, v in SUBSETS.items() if k in only}
    print(f'--only {only}: computing {list(SUBSETS)}', flush=True)


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ds(vals):
    vals = np.asarray(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def run_subset(sub, tag):
    """Per-fold AUC for the 4 models under both imputation policies."""
    X = sub[FEATS].values.astype(np.float32)
    y = sub['vte_event'].values.astype(int)
    groups = sub['subject_id'].values
    padua = sub['padua_score'].values.astype(np.float64)
    improve = sub['improve_score'].values.astype(np.float64)
    fold = {p: {m: [] for m in MODELS} for p in ('leaky', 'fixed')}
    oof = {p: {m: np.zeros(len(y)) for m in ('lr', 'xgb')} for p in ('leaky', 'fixed')}
    imp = SimpleImputer(strategy='median')
    X_leaky = imp.fit_transform(X).astype(np.float32)
    n_fold = 0
    t0 = time.time()
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X, y, groups):
            n_fold += 1
            fold['leaky']['padua'].append(roc_auc_score(y[te], padua[te]))
            fold['leaky']['improve'].append(roc_auc_score(y[te], improve[te]))
            fold['fixed']['padua'].append(fold['leaky']['padua'][-1])
            fold['fixed']['improve'].append(fold['leaky']['improve'][-1])

            fix = SimpleImputer(strategy='median').fit(X[tr])
            Xtr_fixed, Xte_fixed = fix.transform(X[tr]), fix.transform(X[te])

            lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
            lr.fit(X_leaky[tr], y[tr])
            p = lr.predict_proba(X_leaky[te])[:, 1]
            fold['leaky']['lr'].append(roc_auc_score(y[te], p))
            if seed == 42:
                oof['leaky']['lr'][te] = p

            lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
            lr.fit(Xtr_fixed, y[tr])
            p = lr.predict_proba(Xte_fixed)[:, 1]
            fold['fixed']['lr'].append(roc_auc_score(y[te], p))
            if seed == 42:
                oof['fixed']['lr'][te] = p

            xgb = XGBClassifier(**XGB_PARAMS)
            xgb.fit(X_leaky[tr], y[tr])
            p = xgb.predict_proba(X_leaky[te])[:, 1]
            fold['leaky']['xgb'].append(roc_auc_score(y[te], p))
            if seed == 42:
                oof['leaky']['xgb'][te] = p

            xgb = XGBClassifier(**XGB_PARAMS)
            xgb.fit(Xtr_fixed, y[tr])
            p = xgb.predict_proba(Xte_fixed)[:, 1]
            fold['fixed']['xgb'].append(roc_auc_score(y[te], p))
            if seed == 42:
                oof['fixed']['xgb'][te] = p
        print(f'  [{tag}] seed {seed} done ({time.time() - t0:.0f}s)', flush=True)

    res = {'rows': int(len(y)), 'events': int(y.sum()),
           'controls': int(len(y) - y.sum()), 'n_folds': n_fold,
           'protocols': {}}
    for pol in ('leaky', 'fixed'):
        cell = {}
        for m in MODELS:
            v = np.array(fold[pol][m])
            mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
            entry = {'fold_mean_auc': mean, 'se': se,
                     'ci95': [mean - 1.96 * se, mean + 1.96 * se]}
            if m in ('lr', 'xgb'):
                entry['oof_auc_seed42'] = float(roc_auc_score(y, oof[pol][m]))
            cell[m] = entry
        res['protocols'][pol] = cell
    res['delta_fold_mean'] = {
        pol: {'lr_vs_padua': res['protocols'][pol]['lr']['fold_mean_auc']
                            - res['protocols'][pol]['padua']['fold_mean_auc'],
              'xgb_vs_padua': res['protocols'][pol]['xgb']['fold_mean_auc']
                             - res['protocols'][pol]['padua']['fold_mean_auc'],
              'lr_vs_improve': res['protocols'][pol]['lr']['fold_mean_auc']
                              - res['protocols'][pol]['improve']['fold_mean_auc'],
              'xgb_vs_improve': res['protocols'][pol]['xgb']['fold_mean_auc']
                               - res['protocols'][pol]['improve']['fold_mean_auc'],
              'xgb_vs_lr': res['protocols'][pol]['xgb']['fold_mean_auc']
                          - res['protocols'][pol]['lr']['fold_mean_auc']}
        for pol in ('leaky', 'fixed')}

    # patient-level cluster bootstrap on the seed-42 OOF predictions
    subj_codes, _ = pd.factorize(groups)
    n_subj = len(np.unique(subj_codes))
    ev_rows, neg_rows = np.where(y == 1)[0], np.where(y == 0)[0]
    ev_subj, neg_subj = subj_codes[ev_rows], subj_codes[neg_rows]
    P = {'padua': padua, 'improve': improve,
         'lr_leaky': oof['leaky']['lr'], 'xgb_leaky': oof['leaky']['xgb'],
         'lr_fixed': oof['fixed']['lr'], 'xgb_fixed': oof['fixed']['xgb']}
    pairs = {'lr_vs_padua': ('padua', 'lr'), 'xgb_vs_padua': ('padua', 'xgb'),
             'lr_vs_improve': ('improve', 'lr'),
             'xgb_vs_improve': ('improve', 'xgb'), 'xgb_vs_lr': ('lr', 'xgb')}
    boot = {f'{n}|{pol}': [] for n in pairs for pol in ('leaky', 'fixed')}
    rng = np.random.RandomState(42)
    for _ in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        mult = counts[ev_subj]
        take = mult > 0
        if take.sum() < 2:
            continue
        bw = np.repeat(ev_rows[take], mult[take])
        bn = neg_rows[counts[neg_subj] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        rows = np.concatenate([bw, bn])
        yb = y[rows]
        a = {k: fast_auc(yb, v[rows]) for k, v in P.items()}
        for n, (old, new) in pairs.items():
            for pol in ('leaky', 'fixed'):
                old_v = a[old] if old in ('padua', 'improve') else a[f'{old}_{pol}']
                boot[f'{n}|{pol}'].append(a[f'{new}_{pol}'] - old_v)
    res['delta_auc_bootstrap'] = {}
    for n in pairs:
        for pol in ('leaky', 'fixed'):
            res['delta_auc_bootstrap'][f'{n}|{pol}'] = ds(boot[f'{n}|{pol}'])
    res['oof_auc_seed42'] = {k: float(roc_auc_score(y, v)) for k, v in P.items()}
    return res


META = {
    'dataset': 'data_era/primary_inclprior_excl24h_model_dataset.parquet',
    'splits': 'data_era/primary_inclprior_excl24h_splits.json',
    'strict_ids': 'output_era/strict_control_ids_inclprior_excl24h.parquet',
    'models': MODELS, 'features': len(FEATS), 'feature_set': 'main',
    'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44), group subject_id',
    'xgb_params': {k: v for k, v in XGB_PARAMS.items() if k != 'n_jobs'},
    'pool': f'{len(pool_ids):,} rows (train+val); '
            f'{int(df[df.hadm_id.isin(pool_ids)].vte_event.sum())} events',
    'imputation': "leaky = SimpleImputer fitted on all rows of the subset "
                  "(50c:88); fixed = fitted on training rows of each fold",
    'bootstrap': 'patient-level cluster bootstrap, 2000 replicates, seed-42 OOF '
                 'predictions, bootstrap done on the leaky OOF for the leaky '
                 'policy and the fixed OOF for the fixed policy',
}
PATH = f'{RES}/output_era/strict_control_apples2apples_era_fixed.json'
# Re-running with --only <tags> keeps previously computed subsets and refreshes
# the requested ones, so a long run can be resumed without recomputing.
out = json.load(open(PATH)) if (only and os.path.exists(PATH)) else {}
out['meta'] = META
for tag in SUBSETS:
    out.pop(tag, None)
for tag, sub in SUBSETS.items():
    print(f'== {tag}: {len(sub):,} rows / {int(sub.vte_event.sum()):,} events',
          flush=True)
    out[tag] = run_subset(sub, tag)
    for pol in ('leaky', 'fixed'):
        c = out[tag]['protocols'][pol]
        print(f'   [{tag}/{pol}] ' + '  '.join(
            f'{m}={c[m]["fold_mean_auc"]:.4f}' for m in MODELS), flush=True)
    with open(PATH, 'w') as f:
        json.dump(out, f, indent=2)
print(f'saved -> {PATH}')
