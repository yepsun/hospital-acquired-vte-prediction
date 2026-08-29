#!/usr/bin/env python3
"""
Stratified AUC by prior-VTE status for the inclprior_excl24h primary arm.

Refits the script-98 static protocol (seed-42 5-fold StratifiedGroupKFold OOF)
and reports AUC for Padua, LR, XGBoost within prior_vte_any==1 and ==0
subgroups, to show the ML advantage is not carried by the history flag alone.

Output: results_vte/ajm/new/output/prior_vte_stratified_auc.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RES = 'results_vte'
SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
FEATS = fs['main']
splits = json.load(open(f'{RES}/ajm/new/data/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{RES}/ajm/new/data/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)

X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
padua = pool['padua_score'].values.astype(float)
prior = pool['prior_vte_any'].fillna(0).values.astype(int)

oof = {'lr': np.zeros(len(y)), 'xgb': np.zeros(len(y))}
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr, te in sgkf.split(X, y, groups):
    imp = SimpleImputer(strategy='median')
    m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    m.fit(imp.fit_transform(X[tr]), y[tr])
    oof['lr'][te] = m.predict_proba(imp.transform(X[te]))[:, 1]
    m = XGBClassifier(**XGB_PARAMS).fit(X[tr], y[tr])
    oof['xgb'][te] = m.predict_proba(X[te])[:, 1]
    print(f'fold done, events={y[te].sum()}', flush=True)

out = {}
for pv in [0, 1]:
    msk = prior == pv
    ys = y[msk]
    row = {'n': int(msk.sum()), 'events': int(ys.sum())}
    for name, p in [('padua', padua), ('lr', oof['lr']), ('xgb', oof['xgb'])]:
        row[f'auc_{name}'] = float(roc_auc_score(ys, p[msk]))
    row['delta_xgb_vs_padua'] = row['auc_xgb'] - row['auc_padua']
    row['delta_lr_vs_padua'] = row['auc_lr'] - row['auc_padua']
    out[f'prior_vte_{pv}'] = row
    print(pv, json.dumps(row), flush=True)

out['meta'] = {'cohort': 'inclprior_excl24h pool', 'seed': SEED,
               'note': 'seed-42 5-fold OOF; AUC within prior_vte_any strata'}
with open(f'{RES}/ajm/new/output/prior_vte_stratified_auc.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved -> results_vte/ajm/new/output/prior_vte_stratified_auc.json')
