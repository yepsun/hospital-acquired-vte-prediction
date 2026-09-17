#!/usr/bin/env python3
"""External validation of the ADMISSION-SAFE (deployment) MIMIC models.

The existing external validation (run_external_validation_ipe.py) covers only
the 57-feature static models. The manuscript now recommends the 42-feature
admission-safe variant (15 discharge-coded ICD/procedure flags removed), so
this script applies the re-frozen 42- and 40-feature models to the SAME
external cohort(s), with the SAME outcome definition, feature harmonisation and
evaluation as run_external_validation_ipe.py:

  * cohort      : extval/cohort_ipe.parquet (+ features.parquet, scores.parquet)
  * two analysis populations, exactly as the existing runs
      - primary  : fully enumerated high-risk stratum (no weighting)
                   -> external_validation_result_highrisk_deploysafe.json
      - secondary: weighted full population (inverse-probability sampling
                   weight on the non-high-risk stratum)
                   -> external_validation_result_deploysafe.json
  * AUC         : weighted Mann-Whitney estimator, Efron 2,000-replicate
                  bootstrap CI, negatives capped at 50,000 (identical
                  resampling design and RNG stream to the main run, so the
                  57-feature rows reproduce the published numbers exactly)
  * ΔAUC        : versus Padua and IMPROVE on the identical rows
  * AUPRC       : weighted average precision
  * calibration : Brier + expected calibration error, raw and isotonic
                  recalibrated, both in-sample (as published) and
                  cross-fitted (held-out), under equal-count decile and
                  equal-width binning

Feature-coverage audit: for every arm the script reports which of the frozen
training features exist in the external feature table, which are entirely
missing, and how the missing ones are handled. It never silently imputes a
feature that is absent from the external data.

Never overwrites an existing file: all outputs end in `_deploysafe`.

Run:  /Users/Yepsun/myenv/bin/python3 run_external_validation_deploysafe_ipe.py
      EXTVAL_HIGH_ONLY=1 /Users/Yepsun/myenv/bin/python3 run_external_validation_deploysafe_ipe.py
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'
SEED = 42
N_BOOT = 2000
NEG_CAP = 50_000
N_BINS = 10

NH_FULL, NH_SAMP = 76_188, 7_072
SW_HIGH, SW_NONHIGH = 1.0, NH_FULL / NH_SAMP

HIGH_RISK_ONLY = os.environ.get('EXTVAL_HIGH_ONLY', '0') == '1'

# legacy 57-feature frozen models (unchanged, already published)
FROZEN = {
    'main57': f'{BASE}/frozen_models_ipe',
    'strict42': f'{BASE}/frozen_models_ipe_deploysafe/strict_42',
    'cons40': f'{BASE}/frozen_models_ipe_deploysafe/conservative_40',
}
# secondary era-pool refits (same external rows, different MIMIC pool);
# present only if freeze_mimic_models_deploysafe_ipe.py --pool era was run
ERA_DIRS = {'main57_era': f'{BASE}/frozen_models_era_deploysafe/main_57',
            'strict42_era': f'{BASE}/frozen_models_era_deploysafe/strict_42',
            'cons40_era': f'{BASE}/frozen_models_era_deploysafe/conservative_40'}


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wfast_auc(yb, p, w):
    """Weighted Mann-Whitney AUC with ties scored 0.5. With all weights equal
    this is exactly the unweighted Mann-Whitney estimator, so it is used
    unconditionally, matching run_external_validation_ipe.py."""
    ev = yb == 1
    pe, pn = p[ev], p[~ev]
    we, wn = w[ev], w[~ev]
    order = np.argsort(pn)
    pn_s, wn_s = pn[order], wn[order]
    cwn = np.concatenate([[0.0], np.cumsum(wn_s)])
    tot, wn_sum = 0.0, wn_s.sum()
    for s, wi in zip(pe, we):
        lo = np.searchsorted(pn_s, s, side='left')
        hi = np.searchsorted(pn_s, s, side='right')
        tot += wi * (cwn[lo] + 0.5 * (cwn[hi] - cwn[lo]))
    return float(tot / (we.sum() * wn_sum))


def wece(yb, p, w):
    """equal-count decile ECE -- verbatim from run_external_validation_ipe.py"""
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0], bins[-1] = -np.inf, np.inf
    ind = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    tot, wsum = 0.0, w.sum()
    for b in range(10):
        m = ind == b
        if m.sum() == 0:
            continue
        wb, ybb, pb = w[m], yb[m], p[m]
        tot += (wb.sum() / wsum) * abs(np.sum(wb * ybb) / wb.sum()
                                       - np.sum(wb * pb) / wb.sum())
    return float(tot)


def ece_width(yb, p, w, rng=(0.0, 1.0)):
    edges = np.linspace(rng[0], rng[1], N_BINS + 1)
    ind = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    tot, wsum = 0.0, w.sum()
    for b in range(N_BINS):
        m = ind == b
        if m.sum() == 0:
            continue
        wb, ybb, pb = w[m], yb[m], p[m]
        tot += (wb.sum() / wsum) * abs(np.sum(wb * ybb) / wb.sum()
                                       - np.sum(wb * pb) / wb.sum())
    return float(tot)


def wbrier(yb, p, w):
    return float(np.sum(w * (yb - p) ** 2) / w.sum())


def crossfit_isotonic(p, y, w, n_splits=5, seed=SEED):
    pc = np.full(len(p), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(p, y):
        m = IsotonicRegression(out_of_bounds='clip').fit(p[tr], y[tr],
                                                         sample_weight=w[tr])
        pc[te] = m.predict(p[te])
    return pc


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def main():
    t_start = time.time()
    co = pd.read_parquet(f'{BASE}/cohort_ipe.parquet',
                         columns=['vis_id', 'risk_group', 'vte_outcome',
                                  'prior_vte_any', 'prior_vte_this_adm'])
    fe = pd.read_parquet(f'{BASE}/features.parquet').drop(
        columns=['prior_vte_any', 'vte_outcome', 'risk_group'], errors='ignore')
    sc = pd.read_parquet(f'{BASE}/scores.parquet')
    df = co.merge(fe, on='vis_id', how='left', validate='1:1') \
        .merge(sc, on='vis_id', how='left', validate='1:1')
    df['prior_vte'] = df['prior_vte_this_adm']
    df['sw'] = np.where(df['risk_group'] == 'non_high', SW_NONHIGH, SW_HIGH)
    if HIGH_RISK_ONLY:
        df = df[df['risk_group'] == 'high'].copy()
        df['sw'] = 1.0
    df = df.reset_index(drop=True)
    y = df['vte_outcome'].astype(int).values
    w = df['sw'].values.astype(float)
    print(f'cohort n={len(df):,} events={int(y.sum()):,} '
          f'rate={y.mean():.4f} (weighted rate={np.sum(w*y)/w.sum():.4f}) '
          f'high_only={HIGH_RISK_ONLY}', flush=True)

    # ---- feature-coverage audit and prediction --------------------------
    models = {}
    coverage = {}
    dirs = dict(FROZEN)
    dirs.update(ERA_DIRS)
    for key, d in dirs.items():
        fpath = f'{d}/feature_order.json'
        if not os.path.exists(fpath):
            print(f'  skip {key}: {fpath} not found', flush=True)
            continue
        feats = json.load(open(fpath))['features']
        missing_cols = [f for f in feats if f not in df.columns]
        all_nan = [f for f in feats
                   if f in df.columns and df[f].isna().all()]
        usable = [f for f in feats if f in df.columns]
        X = df[usable].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        X = X.clip(-1e30, 1e30)
        if missing_cols:                       # absent -> insert all-NaN cols
            for f in missing_cols:
                X[f] = np.float32(np.nan)
        X = X[feats]                           # restore training column order
        lr = joblib.load(f'{d}/final_lr.joblib')
        imp = joblib.load(f'{d}/final_lr_imputer.joblib')
        xgb = joblib.load(f'{d}/final_xgb.joblib')
        p_lr = lr.predict_proba(imp.transform(X))[:, 1]
        p_xgb = xgb.predict_proba(X)[:, 1]
        models[key] = {'lr': p_lr, 'xgb': p_xgb}
        coverage[key] = {
            'n_features': len(feats),
            'n_present_as_columns': len(usable),
            'n_reconstructable': len(usable) - len(all_nan),
            'features_absent_from_external_table': missing_cols,
            'features_present_but_all_null_externally': all_nan,
            'handling': ('absent / all-null features are passed to the frozen '
                         'model as NaN: XGBoost handles missing values '
                         'natively; the LR imputer (fitted on the MIMIC '
                         'training pool) substitutes the training median, '
                         'i.e. the frozen models are applied UNMODIFIED'),
            'external_missingness': {f: float(df[f].isna().mean())
                                     for f in feats if f in df.columns
                                     and df[f].isna().any()},
        }
        print(f'  {key}: {coverage[key]["n_reconstructable"]}/'
              f'{len(feats)} features reconstructable externally; '
              f'absent={missing_cols} all_null={all_nan}', flush=True)

    pad_raw = df['padua_score'].values.astype(float)
    impr_raw = df['improve_score'].values.astype(float)
    pad_prob = SkLR(C=1.0, max_iter=2000).fit(
        pad_raw.reshape(-1, 1), y, sample_weight=w).predict_proba(
        pad_raw.reshape(-1, 1))[:, 1]
    impr_prob = SkLR(C=1.0, max_iter=2000).fit(
        impr_raw.reshape(-1, 1), y, sample_weight=w).predict_proba(
        impr_raw.reshape(-1, 1))[:, 1]

    # ---- bootstrap: identical RNG stream to run_external_validation_ipe.py
    #      (randint -> optional negative choice -> AUC over all models) -----
    P = {}
    for key, mm in models.items():
        P[f'{key}_lr'] = mm['lr']
        P[f'{key}_xgb'] = mm['xgb']
    P['padua'] = pad_raw
    P['improve'] = impr_raw

    auc_boot = {k: [] for k in P}
    delta_boot = {}
    for key in models:
        for kind in ['lr', 'xgb']:
            delta_boot[f'{key}_{kind}_vs_padua'] = []
            delta_boot[f'{key}_{kind}_vs_improve'] = []
    rng = np.random.RandomState(SEED)
    n = len(y)
    for b in range(N_BOOT):
        idx = rng.randint(0, n, n)
        b_ev = idx[y[idx] == 1]
        b_neg = idx[y[idx] == 0]
        if len(b_neg) > NEG_CAP:
            b_neg = rng.choice(b_neg, NEG_CAP, replace=False)
        rows = np.concatenate([b_ev, b_neg])
        yb = y[rows]; wb = w[rows]
        a = {k: wfast_auc(yb, P[k][rows], wb) for k in P}
        for k in a:
            auc_boot[k].append(a[k])
        for key in models:
            for kind in ['lr', 'xgb']:
                delta_boot[f'{key}_{kind}_vs_padua'].append(
                    a[f'{key}_{kind}'] - a['padua'])
                delta_boot[f'{key}_{kind}_vs_improve'].append(
                    a[f'{key}_{kind}'] - a['improve'])
        if (b + 1) % 500 == 0:
            print(f'  boot {b + 1}/{N_BOOT} ({time.time()-t_start:.0f}s)',
                  flush=True)

    results = {
        'meta': {
            'purpose': 'external validation of the admission-safe (42- and '
                       '40-feature) frozen models, alongside the published '
                       '57-feature frozen models, on the identical external '
                       'rows',
            'analysis_population': ('high_risk_only' if HIGH_RISK_ONLY
                                    else 'full_population_weighted'),
            'protocol': 'identical to run_external_validation_ipe.py: same '
                        'cohort, same outcome, same feature harmonisation, '
                        'same weighted Mann-Whitney AUC with Efron 2,000-'
                        'replicate bootstrap (negatives capped at 50,000)',
            'frozen_model_dirs': dirs,
            'n_boot': N_BOOT, 'neg_cap': NEG_CAP, 'seed': SEED,
        },
        'cohort': {'n': int(len(df)), 'events': int(y.sum()),
                   'event_rate_unweighted': float(y.mean()),
                   'event_rate_weighted': float(np.sum(w * y) / w.sum()),
                   'sampling_weights': {'high': SW_HIGH,
                                        'non_high': round(SW_NONHIGH, 4)},
                   'subgroups': df.groupby('risk_group')['vte_outcome']
                                .agg(['count', 'sum', 'mean'])
                                .rename(columns={'count': 'n', 'sum': 'events',
                                                 'mean': 'event_rate'})
                                .to_dict('index')},
        'feature_coverage': coverage,
        'discrimination': {k: {'auc_est': float(np.mean(v)),
                               'auc_ci': [float(x) for x in
                                          np.percentile(v, [2.5, 97.5])]}
                           for k, v in auc_boot.items()},
        'discrimination_point_estimate': {k: wfast_auc(y.astype(bool), P[k], w)
                                          for k in P},
        'auprc': {k: float(average_precision_score(y, P[k], sample_weight=w))
                  for k in P},
        'delta_auc': {k: ds(v) for k, v in delta_boot.items()},
        'calibration': {},
    }

    # ---- calibration: raw + in-sample isotonic + cross-fitted isotonic ----
    for key in list(models) + ['padua', 'improve']:
        if key in models:
            pl, px = models[key]['lr'], models[key]['xgb']
            block = {'lr': {}, 'xgb': {}}
            for kind, p in [('lr', pl), ('xgb', px)]:
                iso = IsotonicRegression(out_of_bounds='clip').fit(
                    p, y, sample_weight=w)
                pc = iso.predict(p)
                pcf = crossfit_isotonic(p, y, w)
                lo, hi = float(p.min()), float(p.max())
                block[kind] = {
                    'raw': {'brier': wbrier(y, p, w),
                            'ece_equalcount_decile': wece(y, p, w),
                            'ece_equalwidth_01': ece_width(y, p, w),
                            'ece_equalwidth_range': ece_width(y, p, w,
                                                              (lo, hi))},
                    'recal_in_sample': {
                        'brier': wbrier(y, pc, w),
                        'ece_equalcount_decile': wece(y, pc, w),
                        'ece_equalwidth_01': ece_width(y, pc, w),
                        'ece_equalwidth_range': ece_width(y, pc, w, (lo, hi))},
                    'recal_crossfit_5fold': {
                        'brier': wbrier(y, pcf, w),
                        'ece_equalcount_decile': wece(y, pcf, w),
                        'ece_equalwidth_01': ece_width(y, pcf, w),
                        'ece_equalwidth_range': ece_width(y, pcf, w, (lo, hi))},
                }
        else:
            pp = pad_prob if key == 'padua' else impr_prob
            block = {'platt_in_sample': {
                'brier': wbrier(y, pp, w),
                'ece_equalcount_decile': wece(y, pp, w),
                'ece_equalwidth_01': ece_width(y, pp, w)}}
        results['calibration'][key] = block

    # ---- operating characteristics ---------------------------------------
    yb = y.astype(bool)

    def wstat(pred):
        tp = w[pred & yb].sum(); fp = w[pred & ~yb].sum()
        fn = w[~pred & yb].sum(); tn = w[~pred & ~yb].sum()
        return {'sens': float(tp / max(tp + fn, 1e-9)),
                'spec': float(tn / max(tn + fp, 1e-9)),
                'ppv': float(tp / max(tp + fp, 1e-9)),
                'flagged_pct': float(w[pred].sum() / w.sum())}

    oc = {}
    for key in models:
        for kind in ['lr', 'xgb']:
            p = models[key][kind]
            e = {}
            for tgt in [0.05, 0.10, 0.15]:
                k = int(round(tgt * len(y)))
                order = np.lexsort((p, p))      # strict top-k by rank
                pred = np.zeros(len(y), dtype=bool); pred[order[-k:]] = True
                s = wstat(pred); s['threshold'] = float(np.min(p[pred]))
                s['n_flagged'] = int(pred.sum())
                e[f'top_{int(tgt*100)}pct_strict_rank'] = s
            for t in [0.005, 0.01]:
                s = wstat(p >= t); s['threshold'] = t
                e[f'fixed_mimic_{t}'] = s
            oc[f'{key}_{kind}'] = e
    results['operating_characteristics'] = oc

    # ---- sensitivity to the features that are unavailable externally ------
    # adm_elective / adm_emergency are inside the 42- and 40-feature sets but
    # are 100% null in the external feature table. XGBoost sees them as
    # missing; the LR imputer replaces them with the MIMIC training median.
    # Quantify how much that choice moves the external AUC.
    sens = {'unavailable_features': ['adm_elective', 'adm_emergency'],
            'note': ('both columns exist in extval/features.parquet but are '
                     '100% null, so they are unavailable externally; they are '
                     'NOT in the 15-item discharge-code drop list, hence they '
                     'remain inside the 42- and 40-feature sets'),
            'handling_in_primary_run': (
                'passed through as NaN: XGBoost routes missing values natively; '
                'the LR imputer (fitted on the MIMIC training pool) substitutes '
                'the training median, adm_elective=0 and adm_emergency=1. '
                'Either way the feature is CONSTANT across the external cohort, '
                'so it cannot change the LR ranking -- only the LR probability '
                'level -- and any absolute-probability statement that uses an '
                'LR probability inherits that convention.'),
            'variants': {}}
    for key in list(models):
        d = dirs[key]
        feats = json.load(open(f'{d}/feature_order.json'))['features']
        if not all(f in feats for f in sens['unavailable_features']):
            continue
        lr = joblib.load(f'{d}/final_lr.joblib')
        imp = joblib.load(f'{d}/final_lr_imputer.joblib')
        xgb = joblib.load(f'{d}/final_xgb.joblib')
        for vname, fill in [('as_published_nan', (np.nan, np.nan)),
                            ('both_zero', (0.0, 0.0)),
                            ('elective_1_emergency_0', (1.0, 0.0)),
                            ('elective_0_emergency_1', (0.0, 1.0)),
                            ('both_one', (1.0, 1.0))]:
            Xv = df[feats].replace([np.inf, -np.inf], np.nan).astype(
                np.float32).clip(-1e30, 1e30)
            Xv['adm_elective'] = np.float32(fill[0])
            Xv['adm_emergency'] = np.float32(fill[1])
            Xv = Xv[feats]
            p_lr = lr.predict_proba(imp.transform(Xv))[:, 1]
            p_xgb = xgb.predict_proba(Xv)[:, 1]
            sens['variants'][f'{key}__{vname}'] = {
                'lr_auc': wfast_auc(y.astype(bool), p_lr, w),
                'xgb_auc': wfast_auc(y.astype(bool), p_xgb, w),
                'lr_mean_pred': float(np.sum(w * p_lr) / w.sum())}
    results['unavailable_feature_sensitivity'] = sens

    # ---- subgroups (mirrors the main run) --------------------------------
    df['_pad'] = pad_raw
    sub = {}
    for name, idx in [('all', df.index),
                      ('high', df[df['risk_group'] == 'high'].index),
                      ('non_high', df[df['risk_group'] == 'non_high'].index)]:
        g = df.loc[idx]
        yy = g['vte_outcome'].values
        if yy.sum() < 2:
            continue
        row = {'n': int(len(g)), 'events': int(yy.sum()),
               'event_rate': float(yy.mean()),
               'auc_padua': fast_auc(yy, g['_pad'].values)}
        for key in models:
            pos = np.where(df.index.isin(idx))[0]
            for kind in ['lr', 'xgb']:
                row[f'auc_{key}_{kind}'] = fast_auc(yy, models[key][kind][pos])
        sub[name] = row
    results['subgroups'] = sub

    results['runtime_s'] = round(time.time() - t_start, 1)
    out = (f'{BASE}/external_validation_result_highrisk_deploysafe.json'
           if HIGH_RISK_ONLY
           else f'{BASE}/external_validation_result_deploysafe.json')
    json.dump(results, open(out, 'w'), indent=2, default=str)
    print(json.dumps({'cohort': results['cohort'],
                      'feature_coverage': {
                          k: {kk: vv for kk, vv in v.items()
                              if kk != 'external_missingness'}
                          for k, v in coverage.items()},
                      'discrimination': results['discrimination'],
                      'auprc': results['auprc'],
                      'delta_auc': results['delta_auc']}, indent=2),
          flush=True)
    print(f'\nsaved -> {out} ({results["runtime_s"]}s)', flush=True)


if __name__ == '__main__':
    main()
