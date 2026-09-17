#!/usr/bin/env python3
"""
Script 65 (era / correctness fix): NESTED (out-of-sample) recalibration,
calibration metrics and decision-curve re-emission for the primary cohort.

Motivation
----------
41b_calibration_dca_v2.py, 57_threshold_calibration_figures_v2.py and
58_publication_figures_v3.py fit the isotonic recalibration map on the seed-42
5-fold OOF predictions and then evaluate the calibrated metrics on those same
OOF predictions ("apparent", in-sample on OOF). 39b_baseline_scores_eval_v2.py
Platt-fits the Padua probability map on the test set. Both inflate the
reported calibration quality.

This script re-emits the same analyses with a nested (out-of-sample) protocol:

  1. 3x5 StratifiedGroupKFold (seeds 42/43/44, group=subject_id) OOF
     predictions for LR / XGBoost on the development pool (train+val).
  2. Nested isotonic map: for each held-out fold k the calibrator is fitted on
     the OOF predictions of the OTHER four folds only and applied to fold k
     ("cross-fitted" recalibration). Fitting on the whole pool and evaluating
     on the pool is kept as the apparent comparator.
  3. Held-out test partition: models retrained on the full development pool;
     calibrator fitted on the whole pool's OOF predictions (development rows
     only) and applied to the untouched test rows. Padua's univariate logistic
     (Platt) map is also fitted on development rows only, never on test.
  4. ECE (10 equal-width bins), Brier, reliability deciles (Figure 2 layout:
     deciles of the pre-recalibration score), DCA net benefit per 1,000 over
     the 0.5-2% band and at 0.5%/1%/2%, and Table S5 threshold operating
     characteristics - each reported for uncalibrated vs nested-calibrated and
     alongside the old apparent numbers.

No existing file in data_era/ output_era/ scripts_era/ is modified.

Outputs (new filenames only):
  results_vte/ajm/new/output_era/nested_calib_oof_inputs_inclprior_excl24h.npz
  results_vte/ajm/new/output_era/nested_calibration_dca_inclprior_excl24h.json
  results_vte/ajm/new/output_era/nested_calibration_dca_partial_seed<N>.json
Run from the repository root that contains results_vte/.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEEDS = [42, 43, 44]
N_SPLITS = 5
THRESHOLDS_OPS = [0.005, 0.01, 0.02]
DCA_GRID = np.linspace(0.0005, 0.03, 120)
OUT_JSON = f'{RES}/ajm/new/output_era/nested_calibration_dca_inclprior_excl24h.json'
OUT_NPZ = f'{RES}/ajm/new/output_era/nested_calib_oof_inputs_inclprior_excl24h.npz'

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
test = df[df['hadm_id'].isin(set(splits['test']))].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
padua_pool = np.nan_to_num(pool['padua_score'].values.astype(float),
                           nan=float(np.nanmedian(pool['padua_score'].values.astype(float))))
Xt = test[FEATS].values.astype(np.float32)
yt = test['vte_event'].values.astype(int)
padua_test = np.nan_to_num(test['padua_score'].values.astype(float),
                           nan=float(np.nanmedian(pool['padua_score'].values.astype(float))))
N, NE = len(y), len(yt)
print(f'pool N={N} events={y.sum()} ({y.mean()*100:.3f}%) | '
      f'test N={NE} events={yt.sum()} ({yt.mean()*100:.3f}%)', flush=True)


def fit_predict(kind, X_tr, y_tr, X_te, seed):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def uniform_bin_curve(yb, p, bins=10):
    """10 equal-width bins (ECE definition; matches calibration_curve)."""
    pt, pp = calibration_curve(yb, p, n_bins=bins, strategy='uniform')
    return {'bin_mean_predicted': [float(v) for v in pp],
            'bin_observed_fraction': [float(v) for v in pt]}


def ece_uniform(yb, p, bins=10):
    pt, pp = calibration_curve(yb, p, n_bins=bins, strategy='uniform')
    return float(np.mean(np.abs(pt - pp)))


def decile_curve(yb, p_plot, p_order, bins=10):
    """Figure 2 layout: deciles formed on the pre-recalibration score."""
    edges = np.quantile(p_order, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p_order, edges[1:-1], right=False), 0, bins - 1)
    mp, of, nn = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 50:
            mp.append(float(np.mean(p_plot[m])))
            of.append(float(np.mean(yb[m])))
            nn.append(int(m.sum()))
    return {'bin_mean_predicted': mp, 'bin_observed_fraction': of, 'bin_n': nn}


def cal_stats(yb, p, p_order):
    return {'brier': float(brier_score_loss(yb, p)),
            'ece': ece_uniform(yb, p),
            'uniform10_curve': uniform_bin_curve(yb, p),
            'decile_curve': decile_curve(yb, p, p_order)}


def net_benefit(yb, p, pt):
    flag = p >= pt
    tp = (flag & (yb == 1)).sum()
    fp = (flag & (yb == 0)).sum()
    return float(tp / len(yb) - fp / len(yb) * pt / (1 - pt))


def dca_block(yb, probs, grid):
    prev = yb.mean()
    treat_all = (prev - (1 - prev) * grid / (1 - grid)) * 1e3
    models = {m: np.array([net_benefit(yb, p, t) for t in grid]) * 1e3
              for m, p in probs.items()}
    band = (grid >= 0.005) & (grid <= 0.02)
    return {'thresholds_pct': (grid * 100).tolist(),
            'treat_all_per1000': treat_all.tolist(),
            'models_per1000': {m: v.tolist() for m, v in models.items()},
            'mean_nb_band_per1000': {m: float(v[band].mean())
                                     for m, v in models.items()},
            'nb_at_thresholds_per1000': {
                f'{t * 100:g}%': {m: float(net_benefit(yb, p, t) * 1e3)
                                  for m, p in probs.items()}
                for t in THRESHOLDS_OPS},
            'treat_all_at_thresholds_per1000': {
                f'{t * 100:g}%': float((prev - (1 - prev) * t / (1 - t)) * 1e3)
                for t in THRESHOLDS_OPS}}


def op_char(yb, p, thr):
    flag = p >= thr
    tp = int((flag & (yb == 1)).sum())
    fp = int((flag & (yb == 0)).sum())
    fn = int((~flag & (yb == 1)).sum())
    tn = int((~flag & (yb == 0)).sum())
    return {'threshold': thr, 'n_flagged': int(flag.sum()),
            'flag_rate_pct': 100.0 * float(flag.mean()),
            'alerts_per_1000': 1000.0 * float(flag.mean()),
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'ppv': tp / (tp + fp) if tp + fp else None,
            'npv': tn / (tn + fn) if tn + fn else None}


# ── Step 1: 3x5 OOF + nested (cross-fitted) calibrators ─────────────────────
results = {'meta': {
    'n_pool': N, 'n_pool_events': int(y.sum()),
    'n_test': NE, 'n_test_events': int(yt.sum()),
    'protocol': '3x5 StratifiedGroupKFold (seeds 42/43/44, group=subject_id); '
                'NESTED isotonic map fitted on the OOF predictions of the other '
                'four folds and applied to the held-out fold; test partition '
                'uses models retrained on the full pool and a calibrator fitted '
                'on the whole development-pool OOF (development rows only); '
                'Padua Platt map fitted on development rows only',
    'ece_definition': '10 equal-width bins, unweighted mean |observed - predicted|',
    'dca_definition': 'net benefit = TP/n - FP/n * pt/(1-pt), per 1,000 admissions, '
                      'grid 0.05-3.0% (120 points), band 0.5-2%',
    'seed_primary': 42}}
json.dump(results, open(OUT_JSON, 'w'), indent=2)

oof_store, fold_store = {}, {}
for seed in SEEDS:
    t0 = time.time()
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = {k: np.zeros(N) for k in ['lr', 'xgb']}
    fold_id = np.full(N, -1, dtype=np.int16)
    for f, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        fold_id[te] = f
        for k in ['lr', 'xgb']:
            oof[k][te] = fit_predict(k, X[tr], y[tr], X[te], seed)
        print(f'  seed {seed} fold {f} done ({time.time()-t0:.0f}s)',
              flush=True)

    # nested isotonic: fit on the other folds' OOF, apply to this fold
    nested = {k: np.zeros(N) for k in oof}
    apparent = {k: np.zeros(N) for k in oof}
    for k in oof:
        iso_full = IsotonicRegression(out_of_bounds='clip').fit(oof[k], y)
        apparent[k] = iso_full.predict(oof[k])
        for f in range(N_SPLITS):
            held = fold_id == f
            iso_k = IsotonicRegression(out_of_bounds='clip').fit(oof[k][~held], y[~held])
            nested[k][held] = iso_k.predict(oof[k][held])
    # nested Platt for Padua (fit on other folds only)
    padua_nested = np.zeros(N)
    for f in range(N_SPLITS):
        held = fold_id == f
        pl = LogisticRegression(C=1.0, max_iter=2000).fit(
            padua_pool[~held].reshape(-1, 1), y[~held])
        padua_nested[held] = pl.predict_proba(padua_pool[held].reshape(-1, 1))[:, 1]
    padua_apparent = LogisticRegression(C=1.0, max_iter=2000).fit(
        padua_pool.reshape(-1, 1), y).predict_proba(padua_pool.reshape(-1, 1))[:, 1]

    oof_store[seed] = {'lr': oof['lr'], 'xgb': oof['xgb'],
                       'lr_nested': nested['lr'], 'xgb_nested': nested['xgb'],
                       'lr_apparent': apparent['lr'], 'xgb_apparent': apparent['xgb'],
                       'padua_nested': padua_nested,
                       'padua_apparent': padua_apparent,
                       'fold_id': fold_id.astype(np.int16), 'y': y.astype(np.int8)}
    fold_store[seed] = fold_id

    # ── calibration metrics (pool OOF): uncalibrated / apparent / nested ──
    seed_block = {'oof_auc': {k: float(roc_auc_score(y, oof[k])) for k in oof},
                  'calibration': {}, 'padua': {}}
    for k in ['lr', 'xgb']:
        seed_block['calibration'][k] = {
            'uncalibrated': cal_stats(y, oof[k], oof[k]),
            'apparent_isotonic_on_oof': cal_stats(y, apparent[k], oof[k]),
            'nested_isotonic_crossfitted': cal_stats(y, nested[k], oof[k])}
        for tag, d in seed_block['calibration'][k].items():
            print(f"  seed {seed} {k.upper()} {tag}: Brier={d['brier']:.5f} "
                  f"ECE={d['ece']:.4f}", flush=True)
    seed_block['padua'] = {
        'apparent_platt_pool': cal_stats(y, padua_apparent, padua_apparent),
        'nested_platt_crossfitted': cal_stats(y, padua_nested, padua_apparent)}

    # ── DCA on pool OOF (apparent vs nested) ──────────────────────────────
    seed_block['dca_pool'] = {
        'apparent': dca_block(y, {'padua': padua_apparent,
                                  'lr': apparent['lr'], 'xgb': apparent['xgb']},
                              DCA_GRID),
        'nested': dca_block(y, {'padua': padua_nested,
                                'lr': nested['lr'], 'xgb': nested['xgb']},
                            DCA_GRID)}
    # ── Table S5 threshold operating characteristics ─────────────────────
    seed_block['threshold_ops'] = {
        'apparent': {m: [op_char(y, p, t) for t in THRESHOLDS_OPS]
                     for m, p in {'padua': padua_apparent, 'lr': apparent['lr'],
                                  'xgb': apparent['xgb']}.items()},
        'nested': {m: [op_char(y, p, t) for t in THRESHOLDS_OPS]
                   for m, p in {'padua': padua_nested, 'lr': nested['lr'],
                                'xgb': nested['xgb']}.items()}}
    results['seed_' + str(seed)] = seed_block
    results['_partial_through_seed'] = seed
    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    json.dump(seed_block, open(f'{RES}/ajm/new/output_era/'
                               f'nested_calibration_dca_partial_seed{seed}.json', 'w'),
              indent=2)
    print(f'seed {seed} complete in {time.time()-t0:.0f}s', flush=True)

np.savez_compressed(OUT_NPZ, y=y.astype(np.int8),
                    **{f'{k}_seed{s}': oof_store[s][k]
                       for s in SEEDS for k in
                       ['lr', 'xgb', 'lr_nested', 'xgb_nested', 'lr_apparent',
                        'xgb_apparent', 'padua_nested', 'padua_apparent', 'fold_id']})
print('saved', OUT_NPZ, flush=True)

# ── Step 2: untouched test partition ────────────────────────────────────────
print('test partition: retrain on full pool ...', flush=True)
t0 = time.time()
test_prob_raw = {k: fit_predict(k, X, y, Xt, SEED_ := 42) for k in ['lr', 'xgb']}
print(f'  test fits done ({time.time()-t0:.0f}s)', flush=True)
iso_test = {k: IsotonicRegression(out_of_bounds='clip').fit(oof_store[42][k], y)
            for k in ['lr', 'xgb']}
platt_test = LogisticRegression(C=1.0, max_iter=2000).fit(
    padua_pool.reshape(-1, 1), y)
padua_prob_test = platt_test.predict_proba(padua_test.reshape(-1, 1))[:, 1]

test_block = {'auc': {k: float(roc_auc_score(yt, test_prob_raw[k])) for k in ['lr', 'xgb']},
              'calibration': {},
              'platt_params': {'padua': [float(v) for v in
                                         platt_test.coef_.ravel().tolist() + platt_test.intercept_.tolist()],
                               'fit_on': 'development pool rows only (train+val), never test'}}
for k in ['lr', 'xgb']:
    cal = iso_test[k].predict(test_prob_raw[k])
    test_block['calibration'][k] = {
        'uncalibrated': cal_stats(yt, test_prob_raw[k], test_prob_raw[k]),
        'nested_isotonic_from_pool_oof': cal_stats(yt, cal, test_prob_raw[k])}
    print(f"  test {k.upper()}: AUC={test_block['auc'][k]:.4f} "
          f"Brier {test_block['calibration'][k]['uncalibrated']['brier']:.5f}"
          f"->{test_block['calibration'][k]['nested_isotonic_from_pool_oof']['brier']:.5f} "
          f"ECE {test_block['calibration'][k]['uncalibrated']['ece']:.4f}"
          f"->{test_block['calibration'][k]['nested_isotonic_from_pool_oof']['ece']:.4f}",
          flush=True)
    test_block['calibration'][k]['note'] = ('calibrator fitted on the development '
                                            'pool OOF only; test rows untouched')
test_block['padua'] = {'auc': float(roc_auc_score(yt, padua_test)),
                       'platt_from_pool': cal_stats(yt, padua_prob_test, padua_test)}
test_block['dca'] = {
    'nested': dca_block(yt, {'padua': padua_prob_test,
                             'lr': iso_test['lr'].predict(test_prob_raw['lr']),
                             'xgb': iso_test['xgb'].predict(test_prob_raw['xgb'])},
                        DCA_GRID)}
test_block['threshold_ops'] = {
    'nested': {m: [op_char(yt, p, t) for t in THRESHOLDS_OPS]
               for m, p in {'padua': padua_prob_test,
                            'lr': iso_test['lr'].predict(test_prob_raw['lr']),
                            'xgb': iso_test['xgb'].predict(test_prob_raw['xgb'])}.items()}}
results['test_partition'] = test_block
results['_partial_through_seed'] = 'complete'
json.dump(results, open(OUT_JSON, 'w'), indent=2)
print('saved ->', OUT_JSON, flush=True)
for m in ['padua', 'lr', 'xgb']:
    print(f"  test nested mean NB 0.5-2% {m}: "
          f"{test_block['dca']['nested']['mean_nb_band_per1000'][m]:+.4f} per 1,000",
          flush=True)
