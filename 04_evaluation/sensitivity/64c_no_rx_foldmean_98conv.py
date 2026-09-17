#!/usr/bin/env python3
"""
Script 64c: XGBoost fold-mean AUC under the script-98 seed convention.

Script 98 (the primary static protocol) builds XGBClassifier with
random_state=SEED (42) baked into XGB_PARAMS, so all three CV repeats use the
same model seed and only the StratifiedGroupKFold split seed varies; scripts
61/62 instead pass random_state=seed, so repeats 43/44 also change the model
seed. The two conventions agree at seed 42 (the primary estimate) but give
slightly different 15-fold means. Script 64 reports the per-seed convention
(matching the published multi-seed OOF table in 61); this addendum reports the
script-98 convention so the published fold-mean AUC (XGB 0.8088 on the
57-feature set) has a like-for-like 52-feature counterpart.

LR is deterministic and unaffected.

Output: results_vte/ajm/new/output_era/no_rx_foldmean_98convention_inclprior_excl24h.json
"""
import json, time, warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
D = f'{RES}/ajm/new/data_era'
OUT = f'{RES}/ajm/new/output_era/no_rx_foldmean_98convention_inclprior_excl24h.json'
SEED = 42
SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000

# exactly script 98's XGB_PARAMS, including the fixed random_state
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

T0 = time.time()
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
MAIN = fs['main']
RX_DROP = ['rx_heparin', 'rx_warfarin', 'rx_doac', 'rx_antiplatelet',
           'rx_hormone']
NORX = [f for f in MAIN if f not in RX_DROP]

splits = json.load(open(f'{D}/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{D}/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
print(f'pool N={len(y)} events={y.sum()}', flush=True)


def fold_boot_ci(fr):
    rng = np.random.RandomState(SEED)
    vals = [np.mean([fr[i] for i in rng.randint(0, len(fr), len(fr))])
            for _ in range(N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


out = {'feature_sets': {'main_57': MAIN, 'no_rx_52': NORX}, 'results': {}}
for name, feats in [('main_57', MAIN), ('no_rx_52', NORX)]:
    X = pool[feats].values.astype(np.float32)
    aucs, per_seed = [], {}
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                    random_state=seed)
        s_aucs = []
        for tr, te in sgkf.split(X, y, groups):
            m = XGBClassifier(**XGB_PARAMS)
            m.fit(X[tr], y[tr])
            s_aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        per_seed[str(seed)] = float(np.mean(s_aucs))
        aucs.extend(s_aucs)
    ci = fold_boot_ci(aucs)
    out['results'][name] = {
        'fold_mean_auc': float(np.mean(aucs)),
        'fold_mean_auc_ci': ci,
        'per_seed_fold_mean': per_seed,
        'n_folds': len(aucs),
    }
    print(f'{name}: XGB fold-mean AUC={np.mean(aucs):.4f} '
          f'[{ci[0]:.4f}-{ci[1]:.4f}] per-seed {per_seed}', flush=True)

out['delta_fold_mean_no_rx_minus_main'] = (
    out['results']['no_rx_52']['fold_mean_auc']
    - out['results']['main_57']['fold_mean_auc'])
out['meta'] = {
    'convention': 'script 98: XGBClassifier random_state fixed at 42 for all '
                  'three repeats; only the CV split seed varies',
    'published_57_feature_reference': {
        'fold_mean_auc': 0.8088045237589251,
        'fold_mean_auc_ci': [0.795800331600418, 0.8210219006314209],
        'source': 'output_era/sensitivity_inclprior_excl24h_result.json'},
    'runtime_s': round(time.time() - T0, 1),
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f'Saved -> {OUT} ({time.time() - T0:.0f}s)', flush=True)
