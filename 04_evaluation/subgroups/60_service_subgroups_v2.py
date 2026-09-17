#!/usr/bin/env python3
"""Script 60: medical/surgical service subgroup discrimination (Table S2).

Assigns each admission a service = first curr_service in the MIMIC services
table. Groups MED/CMED/OMED/NMED/GU/GYN/PSYCH/OBS etc. as 'medical'; surgical
and trauma services as 'surgical'. Within each subgroup, computes seed-42
5-fold OOF AUC for LR and XGBoost (static 57-feature set) and AUCs for the
Padua and IMPROVE scores.

Outputs: results_vte/service_subgroups_inclprior_excl24h.json
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

RES = 'results_vte'
SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

MED_SERVICES = {'MED', 'CMED', 'OMED', 'NMED', 'GU', 'GYN', 'PSYCH', 'OBS',
                'EYE', 'DENT', 'ENT', 'NB'}
SURG_SERVICES = {'SURG', 'NSURG', 'PSURG', 'VSURG', 'CSURG', 'TSURG', 'ORTHO',
                 'TRAUM'}


def main():
    fs = json.load(open(f'{RES}/feature_sets_v2.json'))
    splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
    FEATS = fs['main']
    mdf = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
    pool_ids = set(splits['train']) | set(splits['val'])
    pool = mdf[mdf['hadm_id'].isin(pool_ids)].reset_index(drop=True)

    con = sqlite3.connect('mimic4.db')
    svc = pd.read_sql('SELECT hadm_id, transfertime, curr_service FROM services',
                      con)
    con.close()
    svc = svc.sort_values('transfertime').drop_duplicates('hadm_id', keep='first')
    svc['service_group'] = np.where(
        svc.curr_service.isin(SURG_SERVICES), 'surgical',
        np.where(svc.curr_service.isin(MED_SERVICES), 'medical', 'other'))
    pool = pool.merge(svc[['hadm_id', 'curr_service', 'service_group']],
                      on='hadm_id', how='left')

    out = {}
    for grp, label in [('medical', 'Medical service'), ('surgical', 'Surgical service')]:
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
        padua_auc = roc_auc_score(y, sub['padua_score'].values)
        improve_auc = roc_auc_score(y, sub['improve_score'].values)
        out[grp] = {
            'n': int(len(y)), 'events': int(y.sum()),
            'event_rate': float(y.mean()),
            'lr_auc': float(roc_auc_score(y, oof['lr'])),
            'xgb_auc': float(roc_auc_score(y, oof['xgb'])),
            'padua_auc': float(padua_auc),
            'improve_auc': float(improve_auc),
        }
        print(f"{label}: n={len(y)} events={int(y.sum())} rate={y.mean():.4f} | "
              f"LR={out[grp]['lr_auc']:.4f} XGB={out[grp]['xgb_auc']:.4f} "
              f"Padua={padua_auc:.4f} IMPROVE={improve_auc:.4f}", flush=True)

    with open(f'{RES}/ajm/new/output_era/service_subgroups_inclprior_excl24h.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('saved ->', f' {RES}/ajm/new/output_era/service_subgroups_inclprior_excl24h.json')


if __name__ == '__main__':
    main()
