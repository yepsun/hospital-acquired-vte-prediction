#!/usr/bin/env python3
"""Freeze MIMIC static models (LR + XGB) on the AJM primary pool.

Mirrors scripts/65_eicu_frozen_models.py but on the AJM primary cohort:
  - dataset: results_vte/ajm_primary_model_dataset.parquet (379,265, no prior VTE)
  - splits:  results_vte/ajm_primary_splits.json
  - features: results_vte/feature_sets_v2.json 'main' (57 features)
Fits on train+val pool (no CV), saves joblib models + feature order so the
external validation script can apply them to the local cohort.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RES = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte'
MOD = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval/frozen_models'
SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)


def main():
    os.makedirs(MOD, exist_ok=True)
    fs = json.load(open(f'{RES}/feature_sets_v2.json'))
    splits = json.load(open(f'{RES}/ajm_primary_splits.json'))
    FEATS = fs['main']

    df = pd.read_parquet(f'{RES}/ajm_primary_model_dataset.parquet')
    pool_ids = set(splits['train']) | set(splits['val'])
    pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
    X = pool[FEATS].astype(np.float32)
    y = pool['vte_event'].astype(int)
    print(f'pool N={len(pool):,} events={int(y.sum()):,} '
          f'({y.mean() * 100:.3f}%)', flush=True)

    # LR with training-median imputer (as 40b)
    imp = SimpleImputer(strategy='median')
    Ximp = imp.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    lr.fit(Ximp, y)
    joblib.dump(lr, f'{MOD}/final_lr.joblib')
    joblib.dump(imp, f'{MOD}/final_lr_imputer.joblib')
    print(f'LR fitted, saved', flush=True)

    # XGB native NaN
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X, y)
    joblib.dump(xgb, f'{MOD}/final_xgb.joblib')
    imp_s = pd.Series(xgb.feature_importances_, index=FEATS) \
        .sort_values(ascending=False).head(5)
    print(f'XGB fitted, saved (top5: {imp_s.round(3).to_dict()})', flush=True)

    lrp = lr.predict_proba(Ximp)[:, 1]
    xp = xgb.predict_proba(X)[:, 1]
    print(f'train sanity AUC LR={roc_auc_score(y, lrp):.4f} '
          f'XGB={roc_auc_score(y, xp):.4f}', flush=True)

    json.dump({'features': FEATS, 'seed': SEED,
               'n_pool': int(len(pool)), 'n_events': int(y.sum()),
               'xgb_params': XGB_PARAMS,
               'note': 'trained on MIMIC AJM-primary train+val pool'},
              open(f'{MOD}/feature_order.json', 'w'), indent=2)
    print('feature_order.json saved', flush=True)


if __name__ == '__main__':
    main()
