#!/usr/bin/env python3
"""Script 60b: medical / surgical service subgroup discrimination incl. Caprini.

Same subgroup definition and protocol as scripts_era/60_service_subgroups_v2.py
(first curr_service in the MIMIC `services` table; surgical = SURG/NSURG/PSURG/
VSURG/CSURG/TSURG/ORTHO/TRAUM), extended with the Caprini score and with
patient-level cluster-bootstrap CIs + pairwise AUC differences inside the
surgical subgroup (the comparison a reviewer asks for).

Reads:  results_vte/ajm/new/data_era/primary_inclprior_excl24h_model_dataset_caprini.parquet
        results_vte/ajm/new/data_era/primary_inclprior_excl24h_splits.json
        mimic4.db  (table `services` only, cached to a new parquet)
Writes: results_vte/ajm/new/data_era/services_primary_caprini.parquet
        results_vte/ajm/new/output_era/service_subgroups_caprini_inclprior_excl24h.json
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

RES = 'results_vte'
D = f'{RES}/ajm/new/data_era'
O = f'{RES}/ajm/new/output_era'
SVC_CACHE = f'{D}/services_primary_caprini.parquet'
OUT = f'{O}/service_subgroups_caprini_inclprior_excl24h.json'
SEED = 42
N_BOOT = 2000
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

MED_SERVICES = {'MED', 'CMED', 'OMED', 'NMED', 'GU', 'GYN', 'PSYCH', 'OBS',
                'EYE', 'DENT', 'ENT', 'NB'}
SURG_SERVICES = {'SURG', 'NSURG', 'PSURG', 'VSURG', 'CSURG', 'TSURG', 'ORTHO',
                 'TRAUM'}


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def main():
    fs = json.load(open(f'{RES}/feature_sets_v2.json'))
    splits = json.load(open(f'{D}/primary_inclprior_excl24h_splits.json'))
    FEATS = fs['main']
    mdf = pd.read_parquet(
        f'{D}/primary_inclprior_excl24h_model_dataset_caprini.parquet')
    pool_ids = set(splits['train']) | set(splits['val'])
    pool = mdf[mdf['hadm_id'].isin(pool_ids)].reset_index(drop=True)

    if os.path.exists(SVC_CACHE):
        svc = pd.read_parquet(SVC_CACHE)
        print(f'services loaded from cache {SVC_CACHE}', flush=True)
    else:
        con = sqlite3.connect('mimic4.db')
        svc = pd.read_sql('SELECT hadm_id, transfertime, curr_service '
                          'FROM services', con)
        con.close()
        svc = svc.sort_values('transfertime').drop_duplicates('hadm_id',
                                                             keep='first')
        svc['service_group'] = np.where(
            svc.curr_service.isin(SURG_SERVICES), 'surgical',
            np.where(svc.curr_service.isin(MED_SERVICES), 'medical', 'other'))
        svc = svc[['hadm_id', 'curr_service', 'service_group']]
        svc.to_parquet(SVC_CACHE, index=False)
        print(f'services cached -> {SVC_CACHE}', flush=True)

    pool = pool.merge(svc, on='hadm_id', how='left')
    print(f'pool N={len(pool)} events={int(pool["vte_event"].sum())} '
          f'service coverage={pool["service_group"].notna().mean():.3f}',
          flush=True)

    out = {}
    for grp, label in [('medical', 'Medical service'),
                       ('surgical', 'Surgical service')]:
        sub = pool[pool.service_group == grp].reset_index(drop=True)
        if sub.empty:
            continue
        X = sub[FEATS].values.astype(np.float32)
        y = sub['vte_event'].values.astype(int)
        g = sub['subject_id'].values
        if len(np.unique(y)) < 2:
            continue
        oof = {'lr': np.zeros(len(y)), 'xgb': np.zeros(len(y))}
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr, te in sgkf.split(X, y, g):
            imp = SimpleImputer(strategy='median')
            m = LogisticRegression(C=1.0, max_iter=2000)
            m.fit(imp.fit_transform(X[tr]), y[tr])
            oof['lr'][te] = m.predict_proba(imp.transform(X[te]))[:, 1]
            mx = XGBClassifier(**XGB_PARAMS)
            mx.fit(X[tr], y[tr])
            oof['xgb'][te] = mx.predict_proba(X[te])[:, 1]
        scores = {n: sub[c].values.astype(float) for n, c in
                  [('padua', 'padua_score'), ('improve', 'improve_score'),
                   ('caprini', 'caprini_score')]}
        preds = dict(oof, **scores)
        aucs = {k: float(roc_auc_score(y, v)) for k, v in preds.items()}
        auprcs = {k: float(average_precision_score(y, v))
                  for k, v in preds.items()}

        # patient-level cluster bootstrap within the subgroup
        codes, uniq = pd.factorize(sub['subject_id'])
        n_subj = len(uniq)
        rng = np.random.RandomState(SEED)
        boot = {k: [] for k in preds}
        deltas = {f'{a}_vs_{b}': [] for a, b in
                  [('xgb', 'caprini'), ('xgb', 'padua'), ('caprini', 'padua'),
                   ('caprini', 'improve'), ('xgb', 'lr')]}
        for _ in range(N_BOOT):
            counts = np.bincount(rng.randint(0, n_subj, n_subj),
                                 minlength=n_subj)
            rows = np.repeat(np.arange(len(y)), counts[codes])
            yb = y[rows]
            if yb.sum() < 2 or (1 - yb).sum() < 2:
                continue
            a = {k: fast_auc(yb, v[rows]) for k, v in preds.items()}
            for k in a:
                boot[k].append(a[k])
            for p1, p2 in [('xgb', 'caprini'), ('xgb', 'padua'),
                           ('caprini', 'padua'), ('caprini', 'improve'),
                           ('xgb', 'lr')]:
                deltas[f'{p1}_vs_{p2}'].append(a[p1] - a[p2])

        out[grp] = {
            'n': int(len(y)), 'events': int(y.sum()),
            'event_rate': float(y.mean()),
            'lr_auc': aucs['lr'], 'xgb_auc': aucs['xgb'],
            'padua_auc': aucs['padua'], 'improve_auc': aucs['improve'],
            'caprini_auc': aucs['caprini'],
            'lr_auprc': auprcs['lr'], 'xgb_auprc': auprcs['xgb'],
            'padua_auprc': auprcs['padua'], 'improve_auprc': auprcs['improve'],
            'caprini_auprc': auprcs['caprini'],
            'auc_ci': {k: [float(v) for v in np.percentile(v2, [2.5, 97.5])]
                       for k, v2 in boot.items()},
            'delta_auc': {k: ds(v) for k, v in deltas.items()},
            'caprini_high_prop': float((sub['caprini_high'] == 1).mean()),
            'padua_high_prop': float((sub['padua_high'] == 1).mean()),
            'caprini_event_rate_high': float(
                y[(sub['caprini_high'] == 1).values].mean()),
            'caprini_event_rate_low': float(
                y[(sub['caprini_high'] == 0).values].mean()),
            'service_counts': {k: int(v) for k, v in
                               sub['curr_service'].value_counts().items()},
        }
        a = out[grp]
        print(f"{label}: n={len(y)} events={int(y.sum())} "
              f"rate={y.mean():.4f} | LR={a['lr_auc']:.4f} XGB={a['xgb_auc']:.4f} "
              f"Caprini={a['caprini_auc']:.4f} Padua={a['padua_auc']:.4f} "
              f"IMPROVE={a['improve_auc']:.4f}", flush=True)
        for k, d in a['delta_auc'].items():
            print(f"  dAUC {k}: {d['estimate']:+.4f} "
                  f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] "
                  f"p={d['p_two_sided']:.4f}", flush=True)

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
