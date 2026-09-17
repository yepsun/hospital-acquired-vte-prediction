#!/usr/bin/env python3
"""
Script 69 (era / correctness fix): reconcile the decision-curve net-benefit
numbers reported in the manuscript's Table S9.

Discrepancy being resolved
--------------------------
The manuscript (ATVB `supplemental_material.md`, Table S9) reports mean net
benefit over the 0.5-2% band, per 1,000 admissions, as Padua +0.0241,
LR +0.1903, XGBoost +0.2304. 65_nested_calibration_dca_fixed.py reported the
"old/apparent" value as Padua +0.0011, LR +0.1428, XGBoost +0.1665. This script
identifies the source of the manuscript numbers and re-derives both estimands
under one code path.

Finding (see the `provenance` block of the JSON):
  the manuscript numbers are exactly `output_era/calibration_dca_inclprior_excl24h.json`
  (script 41b) - i.e. they are computed on the HELD-OUT TEST PARTITION
  (n = 66,656; 120 events), not on the development pool (n = 313,064; 554 events)
  that the Table S9 caption names. Consequently they are NOT apparent: 41b fits
  both the isotonic map and the Padua Platt map on development rows and applies
  them to untouched test rows.

Estimand components (identical for every block below):
  net benefit = TP/n - FP/n * pt/(1-pt), scaled by 1,000
  grid = np.linspace(0.001, 0.05, 100)            (100 points, step 0.00049495)
  band = grid points with 0.005 <= pt <= 0.02     (indices 9..38, 30 points)
  band value = unweighted mean of NB over those grid points
  "at 0.5%/1%/2%" = nearest grid points, indices 8/18/38
  (thresholds 0.0049596 / 0.0099091 / 0.0198081, NOT exactly 0.005/0.01/0.02)

Output (new filename only):
  results_vte/ajm/new/output_era/dca_reconciliation_inclprior_excl24h.json
  results_vte/ajm/new/output_era/nested_calib_test_probs_inclprior_excl24h.npz
Run from the repository root that contains results_vte/.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEEDS = [42, 43, 44]
GRID = np.linspace(0.001, 0.05, 100)
BAND = (GRID >= 0.005) & (GRID <= 0.02)
THR_IDX = {'0.5%': 8, '1%': 18, '2%': 38}
SRC_41B = f'{RES}/ajm/new/output_era/calibration_dca_inclprior_excl24h.json'
NPZ_POOL = f'{RES}/ajm/new/output_era/nested_calib_oof_inputs_inclprior_excl24h.npz'
OUT = f'{RES}/ajm/new/output_era/dca_reconciliation_inclprior_excl24h.json'
OUT_NPZ = f'{RES}/ajm/new/output_era/nested_calib_test_probs_inclprior_excl24h.npz'

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1)


def net_benefit(yb, p, pt):
    flag = p >= pt
    tp = (flag & (yb == 1)).sum()
    fp = (flag & (yb == 0)).sum()
    return float(tp / len(yb) - fp / len(yb) * pt / (1 - pt))


def nb_block(yb, probs, grid=GRID, band=BAND, scale=1e3):
    """Reproduces 41b's DCA aggregation exactly (unweighted mean over band grid)."""
    prev = yb.mean()
    nb = {m: np.array([net_benefit(yb, p, t) for t in grid]) for m, p in probs.items()}
    return {
        'mean_nb_band': {m: float(v[band].mean() * scale) for m, v in nb.items()},
        'nb_at_nearest_grid_point': {
            k: {m: float(v[i] * scale) for m, v in nb.items()} for k, i in THR_IDX.items()},
        'threshold_at_nearest_grid_point': {k: float(grid[i]) for k, i in THR_IDX.items()},
        'nb_at_exact_thresholds': {
            f'{t * 100:g}%': {m: float(net_benefit(yb, p, t) * scale) for m, p in probs.items()}
            for t in [0.005, 0.01, 0.02]},
        'treat_all_at_nearest_grid_point': {
            k: float((prev - (1 - prev) * grid[i] / (1 - grid[i])) * scale)
            for k, i in THR_IDX.items()},
        'band_grid_points': int(band.sum()),
        'band_range': [float(grid[band].min()), float(grid[band].max())],
        'n_rows': int(len(yb)), 'n_events': int(yb.sum()),
    }


results = {'estimand': {
    'net_benefit': 'TP/n - FP/n * pt/(1-pt)',
    'grid': 'np.linspace(0.001, 0.05, 100)',
    'band': 'grid points with 0.005 <= pt <= 0.02 (indices 9..38, 30 points)',
    'aggregation': 'unweighted mean of NB over the band grid points, x1000',
    'at_thresholds': 'nearest grid points: 0.5%->0.0049596, 1%->0.0099091, 2%->0.0198081',
    'scale': 'per 1,000 admissions (divide by 1000 for per-admission)'}}

# ── 0. provenance: the manuscript numbers live in 41b's JSON ────────────────
src = json.load(open(SRC_41B))
man_band = src['checks']['dca_band_0.5_2pct_mean_nb']
man_at = {k: {m: src['dca']['models'][m][i] * 1e3
              for m in src['dca']['models']} for k, i in THR_IDX.items()}
results['provenance'] = {
    'source_file': SRC_41B,
    'source_script': 'scripts_era/41b_calibration_dca_v2.py',
    'row_set': 'held-out TEST partition',
    'n_test': src['meta']['n_test'], 'n_test_events': src['meta']['n_test_events'],
    'band_mean_per1000': {m: float(v * 1e3) for m, v in man_band.items()},
    'at_nearest_grid_point_per1000': man_at,
    'treat_all_at_nearest_grid_point_per1000':
        {k: float(src['dca']['treat_all'][i] * 1e3) for k, i in THR_IDX.items()},
    'caption_in_manuscript': 'Table S9 caption says "out-of-fold development-pool '
                             'predictions, n = 313,064; 554 events" - the caption '
                             'does not match the row set the NB columns were '
                             'computed on'}
print('provenance (from 41b JSON): band/1000',
      {m: round(v * 1e3, 4) for m, v in man_band.items()}, flush=True)

# ── 1. rebuild the test-partition probabilities ─────────────────────────────
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']
df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[df['hadm_id'].isin(set(splits['test']))].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
padua_pool = np.nan_to_num(pool['padua_score'].values.astype(float),
                           nan=float(np.nanmedian(pool['padua_score'].values.astype(float))))
Xt = test[FEATS].values.astype(np.float32)
yt = test['vte_event'].values.astype(int)
padua_test = np.nan_to_num(test['padua_score'].values.astype(float),
                           nan=float(np.nanmedian(pool['padua_score'].values.astype(float))))
print(f'pool {len(y)}/{int(y.sum())} | test {len(yt)}/{int(yt.sum())}', flush=True)


def fit_test(kind, seed):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X), y)
        return m.predict_proba(imp.transform(Xt))[:, 1]
    m = XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(X, y)
    return m.predict_proba(Xt)[:, 1]


def dca_pool(yb, p, grid=GRID, band=BAND):
    """Development-pool counterpart of nb_block, same grid so the two estimands
    differ only by row set."""
    return nb_block(yb, p, grid=grid, band=band)


t0 = time.time()
z = np.load(NPZ_POOL)
platt_pool = LogisticRegression(C=1.0, max_iter=2000).fit(
    padua_pool.reshape(-1, 1), y)
padua_test_pool = platt_pool.predict_proba(padua_test.reshape(-1, 1))[:, 1]

seed_blocks = {}
test_probs_store = {'y': yt.astype(np.int8), 'padua_score': padua_test.astype(np.float32),
                    'padua_prob_pool_platt': padua_test_pool.astype(np.float32)}
for seed in SEEDS:
    raw = {k: fit_test(k, seed) for k in ['lr', 'xgb']}
    # calibrator fitted on the whole development pool's seed-s OOF (development
    # rows only) -> test rows untouched: this is the manuscript's (41b) recipe
    iso = {k: IsotonicRegression(out_of_bounds='clip').fit(z[f'{k}_seed{seed}'], y)
           for k in ['lr', 'xgb']}
    cal_pool = {k: iso[k].predict(raw[k]) for k in raw}
    # cross-fitted sensitivity: average over the 5 per-fold calibrators
    cal_cf = {}
    subj = pool['subject_id'].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_id = np.full(len(y), -1)
    for f, (_, te) in enumerate(sgkf.split(X, y, subj)):
        fold_id[te] = f
    for k in raw:
        acc = np.zeros(len(yt))
        for f in range(5):
            held = fold_id == f
            ir = IsotonicRegression(out_of_bounds='clip').fit(
                z[f'{k}_seed{seed}'][~held], y[~held])
            acc += ir.predict(raw[k])
        cal_cf[k] = acc / 5

    blk = {'manuscript_estimand_nested_test': nb_block(
               yt, {'padua': padua_test_pool, 'lr': cal_pool['lr'], 'xgb': cal_pool['xgb']}),
           'manuscript_estimand_nested_test_crossfitted': nb_block(
               yt, {'padua': padua_test_pool, 'lr': cal_cf['lr'], 'xgb': cal_cf['xgb']}),
           'same_estimand_on_development_pool_oof': dca_pool(
               y, {'padua': z[f'padua_apparent_seed{seed}'],
                   'lr': z[f'lr_apparent_seed{seed}'], 'xgb': z[f'xgb_apparent_seed{seed}']}),
           'binning_control_test_same_probabilities_58grid': nb_block(
               yt, {'padua': padua_test_pool, 'lr': cal_pool['lr'], 'xgb': cal_pool['xgb']},
               grid=np.linspace(0.0005, 0.03, 120),
               band=(np.linspace(0.0005, 0.03, 120) >= 0.005) & (np.linspace(0.0005, 0.03, 120) <= 0.02))}
    seed_blocks[str(seed)] = blk
    test_probs_store[f'lr_raw_seed{seed}'] = raw['lr'].astype(np.float32)
    test_probs_store[f'xgb_raw_seed{seed}'] = raw['xgb'].astype(np.float32)
    test_probs_store[f'lr_cal_seed{seed}'] = cal_pool['lr'].astype(np.float32)
    test_probs_store[f'xgb_cal_seed{seed}'] = cal_pool['xgb'].astype(np.float32)
    print(f'seed {seed} done ({time.time()-t0:.0f}s): nested test band/1000 '
          f"{ {m: round(v,4) for m, v in blk['manuscript_estimand_nested_test']['mean_nb_band'].items()} }",
          flush=True)

results['seeds'] = seed_blocks
results['primary_seed_42'] = seed_blocks['42']
results['seed_range_nested_test_band_per1000'] = {
    m: [min(seed_blocks[str(s)]['manuscript_estimand_nested_test']['mean_nb_band'][m]
            for s in SEEDS),
        max(seed_blocks[str(s)]['manuscript_estimand_nested_test']['mean_nb_band'][m]
            for s in SEEDS)]
    for m in ['padua', 'lr', 'xgb']}
np.savez_compressed(OUT_NPZ, **test_probs_store)
json.dump(results, open(OUT, 'w'), indent=2)
print('saved ->', OUT, flush=True)
print('saved ->', OUT_NPZ, flush=True)

b = results['primary_seed_42']
print('\n--- table (per 1,000) ---')
hdr = ['block', 'padua', 'lr', 'xgb']
print('%-46s %10s %10s %10s' % tuple(hdr))
for tag in ['manuscript_estimand_nested_test', 'manuscript_estimand_nested_test_crossfitted',
            'same_estimand_on_development_pool_oof', 'binning_control_test_same_probabilities_58grid']:
    d = b[tag]['mean_nb_band']
    print('%-46s %10.4f %10.4f %10.4f' % (tag, d['padua'], d['lr'], d['xgb']))
print('\n41b (manuscript source)     %10.4f %10.4f %10.4f' % (
    results['provenance']['band_mean_per1000']['padua'],
    results['provenance']['band_mean_per1000']['lr'],
    results['provenance']['band_mean_per1000']['xgb']))
