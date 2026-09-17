#!/usr/bin/env python3
"""Freeze the admission-safe (deployment) MIMIC static models for external
validation.

Same protocol and SAME DEVELOPMENT POOL as freeze_mimic_models_ipe.py (the
script that produced extval/frozen_models_ipe/), only the feature list differs:

  LR       : L2, C=1.0, lbfgs, max_iter=2000, training-median imputation
             (SimpleImputer fitted on the training pool, applied to the
             external cohort)
  XGBoost  : 300 trees, max_depth 3, learning_rate 0.05, subsample 0.8,
             colsample_bytree 0.8, min_child_weight 5, reg_lambda 1,
             logloss, scale_pos_weight 1, tree_method hist, random_state 42

Arms (feature lists are the verbatim `main` order of feature_sets_v2.json with
entries removed, matching scripts_era/98c_deploysafe_full_suite.py:199-208):
  strict_42       : main minus STRICT_DROP (15 discharge-coded ICD/procedure
                    columns; scripts_era/62_review_gaps.py:42-46)
  conservative_40 : strict_42 minus prior_vte / prior_vte_any

Two pools are frozen:
  data (default, PRIMARY) : results_vte/ajm/new/data/primary_inclprior_excl24h_*
        -- the exact dataset/splits used by the existing frozen models
           (frozen_models_ipe/feature_order.json: the pre-era development pool)
        -> extval/frozen_models_ipe_deploysafe/<arm>/
  era (secondary)         : results_vte/ajm/new/data_era/primary_inclprior_excl24h_*
        -- the cohort behind the manuscript's internal 42-vs-57 numbers
           (XGBoost OOF 0.7864 vs 0.8090)
        -> extval/frozen_models_era_deploysafe/<arm>/

Never overwrites an existing file or directory: the output directories are new.

Run:  /Users/Yepsun/myenv/bin/python3 freeze_mimic_models_deploysafe_ipe.py
      /Users/Yepsun/myenv/bin/python3 freeze_mimic_models_deploysafe_ipe.py --pool era
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RES = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte'
BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'
SEED = 42
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

# verbatim from scripts_era/62_review_gaps.py:42-46
STRICT_DROP = ['cancer_active', 'heart_failure', 'copd', 'chronic_liver',
               'ckd', 'diabetes', 'stroke', 'mi', 'obesity_icd', 'varicose',
               'thrombophilia', 'infection_severe', 'rheumatologic',
               'surgery_flag', 'trauma_flag']
CONSERVATIVE_EXTRA = ['prior_vte', 'prior_vte_any']

POOLS = {
    'data': {
        'dataset': f'{RES}/ajm/new/data/primary_inclprior_excl24h_model_dataset.parquet',
        'splits': f'{RES}/ajm/new/data/primary_inclprior_excl24h_splits.json',
        'outdir': f'{BASE}/frozen_models_ipe_deploysafe',
        'label': 'PRIMARY (identical dataset/splits to frozen_models_ipe)',
    },
    'era': {
        'dataset': f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet',
        'splits': f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json',
        'outdir': f'{BASE}/frozen_models_era_deploysafe',
        'label': 'SECONDARY (era cohort behind the manuscript internal numbers)',
    },
}


def build_arms(MAIN, include_main57=False):
    arms = {
        'strict_42': [f for f in MAIN if f not in STRICT_DROP],
        'conservative_40': [f for f in MAIN
                            if f not in STRICT_DROP + CONSERVATIVE_EXTRA],
    }
    if include_main57:
        # needed only for the secondary era-pool comparison, so that the
        # internal 0.8090 (57 feats) vs 0.7864 (42 feats) contrast can be
        # followed through to the external cohort on one pool
        arms = {'main_57': list(MAIN), **arms}
    return arms


def freeze(arm, feats, pool, pool_meta, outdir):
    os.makedirs(outdir, exist_ok=True)
    X = pool[feats].astype(np.float32)
    y = pool['vte_event'].astype(int)
    t0 = time.time()

    imp = SimpleImputer(strategy='median')
    Ximp = imp.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs').fit(Ximp, y)
    joblib.dump(lr, f'{outdir}/final_lr.joblib')
    joblib.dump(imp, f'{outdir}/final_lr_imputer.joblib')
    lr_sec = time.time() - t0

    t1 = time.time()
    xgb = XGBClassifier(**XGB_PARAMS).fit(X, y)
    joblib.dump(xgb, f'{outdir}/final_xgb.joblib')
    xgb_sec = time.time() - t1

    lrp = lr.predict_proba(Ximp)[:, 1]
    xp = xgb.predict_proba(X)[:, 1]
    meta = {
        'features': feats, 'seed': SEED, 'arm': arm,
        'n_features': len(feats),
        'n_pool': int(len(pool)), 'n_events': int(y.sum()),
        'event_rate': float(y.mean()),
        'xgb_params': XGB_PARAMS,
        'lr_params': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs',
                      'max_iter': 2000, 'imputation': 'training-median'},
        'dropped_from_main_57': [f for f in pool_meta['main']
                                 if f not in feats],
        'dataset': pool_meta['dataset'], 'splits': pool_meta['splits'],
        'pool_definition': 'hadm_id in splits.train | splits.val',
        'feature_order_source': f'{RES}/feature_sets_v2.json ["main"] order',
        'note': ('admission-safe feature set re-frozen under the exact primary '
                 'protocol of freeze_mimic_models_ipe.py; '
                 + pool_meta['label']),
        'train_pool_apparent_auc': {'lr': float(roc_auc_score(y, lrp)),
                                    'xgb': float(roc_auc_score(y, xp))},
        'top5_xgb_importance': pd.Series(
            xgb.feature_importances_, index=feats)
            .sort_values(ascending=False).head(5).round(5).to_dict(),
        'runtime_s': {'lr': round(lr_sec, 1), 'xgb': round(xgb_sec, 1)},
    }
    json.dump(meta, open(f'{outdir}/feature_order.json', 'w'), indent=2)
    print(f'  [{arm}] n_feat={len(feats)} pool={len(pool):,} '
          f'events={int(y.sum()):,} ({y.mean()*100:.3f}%) '
          f'apparent AUC LR={meta["train_pool_apparent_auc"]["lr"]:.4f} '
          f'XGB={meta["train_pool_apparent_auc"]["xgb"]:.4f} '
          f'-> {outdir} ({time.time()-t0:.0f}s)', flush=True)
    return meta


def main():
    pool_key = 'era' if '--pool' in sys.argv and 'era' in sys.argv else 'data'
    P = POOLS[pool_key]
    fs = json.load(open(f'{RES}/feature_sets_v2.json'))
    MAIN = fs['main']
    arms = build_arms(MAIN, include_main57=(pool_key == 'era'))

    df = pd.read_parquet(P['dataset'])
    splits = json.load(open(P['splits']))
    pool_ids = set(splits['train']) | set(splits['val'])
    pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
    print(f'pool={pool_key} [{P["label"]}]\n  dataset={P["dataset"]}\n'
          f'  splits={P["splits"]}\n  pool N={len(pool):,} '
          f'events={int(pool["vte_event"].sum()):,} '
          f'rate={pool["vte_event"].mean()*100:.4f}%', flush=True)

    missing = [f for f in MAIN if f not in pool.columns]
    if missing:
        raise SystemExit(f'missing features in {P["dataset"]}: {missing}')

    P_out = dict(P, main=MAIN)
    out = {}
    for arm, feats in arms.items():
        print(f'=== arm {arm} ({len(feats)} features, '
              f'dropped {len(MAIN)-len(feats)})', flush=True)
        out[arm] = freeze(arm, feats, pool, P_out, f'{P["outdir"]}/{arm}')

    json.dump({'pool': pool_key, 'dataset': P['dataset'],
               'splits': P['splits'], 'outdir': P['outdir'],
               'arms': out,
               'strict_drop_15': STRICT_DROP,
               'conservative_extra_2': CONSERVATIVE_EXTRA},
              open(f'{P["outdir"]}/freeze_manifest.json', 'w'), indent=2)
    print(f'manifest -> {P["outdir"]}/freeze_manifest.json', flush=True)


if __name__ == '__main__':
    main()
