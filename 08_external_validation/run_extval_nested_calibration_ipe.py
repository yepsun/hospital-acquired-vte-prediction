#!/usr/bin/env python3
"""Held-out (cross-fitted) external calibration for the frozen MIMIC models.

Motivation. run_external_validation_ipe.py (lines 340-346) fits
IsotonicRegression on the FULL external cohort and then evaluates the
calibration error on the SAME admissions (`iso.predict(p)` where `iso` was
fitted on `p, y`). The resulting "expected calibration error < 0.001" therefore
measures how flexibly the map can memorise the cohort, not how well a
calibration map transports to unseen admissions.

This script re-computes the external calibration error with a CROSS-FITTED
(held-out) map: the isotonic / Platt map is fitted on one part of the cohort
and evaluated on the held-out part (k-fold with k = 2, 5, 10, plus a single
fixed 50/50 split). It reports the calibration error under BOTH binning
conventions:
  * equal-count decile bins  -- the convention actually used by the existing
    `wece()` helper (np.quantile bins), i.e. the convention behind the
    currently reported numbers;
  * equal-width bins          -- the convention described in the manuscript
    Methods ("ECE, 10 equal-width bins"), reported both on [0,1] and on the
    observed score range.
so the Methods wording can be aligned with whichever convention is reported.

It also tests, numerically rather than by assumption, whether any rank-based
threshold statement changes (top-X% flagging, Youden-optimal point) and which
absolute-probability-threshold statements do change.

Everything is written to NEW files:
  external_validation_result_highrisk_nested.json   (primary: high-risk only)
  external_validation_result_nested.json            (secondary: weighted full)

Run:  /Users/Yepsun/myenv/bin/python3 run_extval_nested_calibration_ipe.py
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'
SEED = 42
N_BOOT = 2000
NEG_CAP = 50_000
N_BINS = 10
K_FOLDS = [2, 5, 10]
TOP_PCTS = [0.01, 0.02, 0.05, 0.10, 0.15]
FIXED_THRESHOLDS = [0.005, 0.01]

NH_FULL, NH_SAMP = 76_188, 7_072
SW_HIGH, SW_NONHIGH = 1.0, NH_FULL / NH_SAMP

HIGH_RISK_ONLY = os.environ.get('EXTVAL_HIGH_ONLY', '0') == '1'


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wfast_auc(yb, p, w):
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


def _bin_index(p, mode, n_bins, width_range=None):
    if mode == 'count':                      # equal-count (quantile) bins
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        return np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    lo, hi = width_range                 # equal-width bins
    edges = np.linspace(lo, hi, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)


def calibration_error(yb, p, w=None, mode='count', n_bins=N_BINS,
                      width_range=(0.0, 1.0)):
    """Weighted (w given) or unweighted ECE. mode='count' reproduces the
    existing wece(); mode='width' is equal-width binning."""
    if w is None:
        w = np.ones_like(p, dtype=float)
    idx = _bin_index(p, mode, n_bins, width_range)
    tot, wsum = 0.0, w.sum()
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        wb, ybb, pb = w[m], yb[m], p[m]
        obs = float(np.sum(wb * ybb) / wb.sum())
        pred = float(np.sum(wb * pb) / wb.sum())
        tot += (wb.sum() / wsum) * abs(obs - pred)
    return float(tot)


def cal_block(yb, p, w):
    lo, hi = float(np.min(p)), float(np.max(p))
    d = {
        'brier_weighted': float(np.sum(w * (yb - p) ** 2) / w.sum()),
        'brier_unweighted': float(brier_score_loss(yb, p)),
        'ece_weighted_equalcount_decile':
            calibration_error(yb, p, w, mode='count'),
        'ece_weighted_equalwidth_01':
            calibration_error(yb, p, w, mode='width', width_range=(0, 1)),
        'ece_weighted_equalwidth_range':
            calibration_error(yb, p, w, mode='width', width_range=(lo, hi)),
        'ece_unweighted_equalcount_decile':
            calibration_error(yb, p, None, mode='count'),
        'ece_unweighted_equalwidth_01':
            calibration_error(yb, p, None, mode='width', width_range=(0, 1)),
        'ece_unweighted_equalwidth_range':
            calibration_error(yb, p, None, mode='width', width_range=(lo, hi)),
        'prob_range': [lo, hi],
        'mean_pred_weighted': float(np.sum(w * p) / w.sum()),
        'mean_pred_unweighted': float(p.mean()),
    }
    # sklearn calibration_curve(10 'uniform' bins) sanity check, unweighted
    pt, pp = calibration_curve(yb, p, n_bins=N_BINS, strategy='uniform')
    d['sklearn_calibration_curve_uniform_nbins_used'] = int(len(pt))
    return d


def cal_block_with_sklearn_ece(yb, p, w):
    d = cal_block(yb, p, w)
    pt, pp = calibration_curve(yb, p, n_bins=N_BINS, strategy='uniform')
    d['ece_unweighted_sklearn_uniform_meanabs'] = float(
        np.mean(np.abs(pt - pp)))
    pt, pp = calibration_curve(yb, p, n_bins=N_BINS, strategy='quantile')
    d['ece_unweighted_sklearn_quantile_meanabs'] = float(
        np.mean(np.abs(pt - pp)))
    return d


# --------------------------------------------------------------------------
# cross-fitted calibration maps
# --------------------------------------------------------------------------
def _fit_map(kind, p_tr, y_tr, w_tr):
    if kind == 'isotonic':
        return IsotonicRegression(out_of_bounds='clip').fit(
            p_tr, y_tr, sample_weight=w_tr)
    return SkLR(C=1.0, max_iter=2000).fit(
        p_tr.reshape(-1, 1), y_tr, sample_weight=w_tr)


def _predict_map(kind, m, p):
    if kind == 'isotonic':
        return m.predict(p)
    return m.predict_proba(p.reshape(-1, 1))[:, 1]


def crossfit(kind, p, y, w, n_splits, seed=SEED, split=None):
    """Return (cross-fitted predictions, per-fold predictions, fold ids).

    `split` = (train_idx, test_idx) for the single-split variant.
    """
    n = len(p)
    pc = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    if split is not None:
        splits = [split]
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(p, y))
    folds = []
    for k, (tr, te) in enumerate(splits):
        m = _fit_map(kind, p[tr], y[tr], w[tr])
        pte = _predict_map(kind, m, p[te])
        pc[te] = pte
        fold_id[te] = k
        folds.append({
            'fold': k, 'n_test': int(len(te)), 'events_test': int(y[te].sum()),
            'ece_equalcount': calibration_error(y[te], pte, w[te], 'count'),
            'ece_equalwidth_01': calibration_error(
                y[te], pte, w[te], 'width', width_range=(0, 1)),
            'brier': float(np.sum(w[te] * (y[te] - pte) ** 2) / w[te].sum()),
            'auc': wfast_auc(y[te].astype(bool), pte, w[te]),
        })
    return pc, folds, fold_id


def boot_ci(fn, y, w, n_boot=N_BOOT, seed=SEED, neg_cap=NEG_CAP):
    """Efron bootstrap CI of a scalar statistic fn(yb, wb, rows) over rows.

    Uses the same resampling design as run_external_validation_ipe.py
    (admission-level resampling with replacement; negatives capped at 50k for
    rank-type statistics). The capped CI is reported for comparability with the
    main run; the uncapped CI uses the same replicate draw without the cap, so
    the published "0.0000" P-value caveat (no replicate crossed zero) does not
    apply to ECE point estimates.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    vals_capped, vals_full = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        b_ev = idx[y[idx] == 1]
        b_neg = idx[y[idx] == 0]
        if len(b_neg) > neg_cap:
            b_neg = rng.choice(b_neg, neg_cap, replace=False)
        rows_capped = np.concatenate([b_ev, b_neg])
        vals_capped.append(fn(y[rows_capped], w[rows_capped], rows_capped))
        vals_full.append(fn(y[idx], w[idx], idx))
    return ([float(x) for x in np.percentile(vals_capped, [2.5, 97.5])],
            [float(x) for x in np.percentile(vals_full, [2.5, 97.5])])


# --------------------------------------------------------------------------
def _load():
    feat_order = json.load(
        open(f'{BASE}/frozen_models_ipe/feature_order.json'))['features']
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
    return feat_order, df


def main():
    t_start = time.time()
    feat_order, df = _load()
    y = df['vte_outcome'].astype(int).values
    w = df['sw'].values.astype(float)
    print(f'cohort n={len(df):,} events={int(y.sum()):,} '
          f'rate={y.mean():.4f} '
          f'(weighted rate={np.sum(w*y)/w.sum():.4f}) '
          f'high_only={HIGH_RISK_ONLY}', flush=True)

    X = df[feat_order].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X = X.clip(-1e30, 1e30)
    lr = joblib.load(f'{BASE}/frozen_models_ipe/final_lr.joblib')
    imp = joblib.load(f'{BASE}/frozen_models_ipe/final_lr_imputer.joblib')
    xgb = joblib.load(f'{BASE}/frozen_models_ipe/final_xgb.joblib')
    p_lr = lr.predict_proba(imp.transform(X))[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]

    pad_raw = df['padua_score'].values.astype(float)
    impr_raw = df['improve_score'].values.astype(float)

    # model scores to be calibrated: frozen LR, frozen XGB, and the two
    # clinical-score maps (Platt logistic on the raw score, exactly as in
    # run_external_validation_ipe.py lines 228-243)
    scores = {
        'xgb': p_xgb,
        'lr': p_lr,
        'padua_platt': pad_raw,
        'improve_platt': impr_raw,
    }
    kinds = {'xgb': 'isotonic', 'lr': 'isotonic',
             'padua_platt': 'platt', 'improve_platt': 'platt'}

    out = {
        'meta': {
            'purpose': 'held-out (cross-fitted) external calibration; '
                       'replaces the in-sample isotonic recalibration in '
                       'run_external_validation_ipe.py lines 340-346',
            'in_sample_location': 'run_external_validation_ipe.py:340-346 '
                                  '(weighted) and :367-375 (unweighted); '
                                  'wece() at :96-111 uses EQUAL-COUNT '
                                  '(quantile) decile bins; ece() at :114-116 '
                                  'uses sklearn calibration_curve 10 '
                                  'equal-width bins on [0,1]',
            'analysis_population': ('high_risk_only' if HIGH_RISK_ONLY
                                    else 'full_population_weighted'),
            'seed': SEED, 'n_boot': N_BOOT,
            'k_folds': K_FOLDS,
            'binning': {
                'equalcount_decile': 'np.quantile(p, linspace(0,1,11)) edges '
                                     '-- the convention of the existing wece()',
                'equalwidth_01': 'np.linspace(0,1,11) edges (sklearn '
                                 'calibration_curve strategy="uniform")',
                'equalwidth_range': 'np.linspace(min(p),max(p),11) edges',
            },
        },
        'cohort': {'n': int(len(df)), 'events': int(y.sum()),
                   'event_rate_unweighted': float(y.mean()),
                   'event_rate_weighted': float(np.sum(w * y) / w.sum())},
        'calibration': {},
        'operating_characteristics_rank_vs_absolute': {},
    }

    # ------------------------------------------------------------------
    # 1. in-sample (reproduce the currently reported construction)
    # ------------------------------------------------------------------
    print('[1] in-sample (apparent) calibration, as currently reported',
          flush=True)
    cal = out['calibration']
    for name, p in scores.items():
        kind = kinds[name]
        if kind == 'isotonic':
            m = _fit_map('isotonic', p, y, w)
            pc = _predict_map('isotonic', m, p)
            raw_block = cal_block_with_sklearn_ece(y, p, w)
            raw_note = ('raw frozen-model probability (MIMIC-scale; the '
                        'uncalibrated input to the isotonic map)')
        else:
            m = _fit_map('platt', p, y, w)
            pc = _predict_map('platt', m, p)
            raw_block = None
            raw_note = ('not reported: the input is the raw clinical score '
                        '(Padua 0-20 / IMPROVE 0-14), not a probability; the '
                        'manuscript compares these scores after Platt mapping')
        cal[name] = {
            'type': kind,
            'raw': raw_block,
            'raw_note': raw_note,
            'map_in_sample': cal_block_with_sklearn_ece(y, pc, w),
            'spearman_raw_vs_map_in_sample': float(
                spearmanr(p, pc).statistic),
        }
        e = cal[name]['map_in_sample']
        rw = (f'{raw_block["ece_weighted_equalcount_decile"]:.6f}'
              if raw_block else 'n/a')
        print(f'  {name:14s} raw ECE(cw)={rw}'
              f'  in-sample map ECE(cw)='
              f'{e["ece_weighted_equalcount_decile"]:.6f}'
              f'  ECE(ew1)='
              f'{e["ece_weighted_equalwidth_01"]:.6f}',
              flush=True)

    # ------------------------------------------------------------------
    # 2. cross-fitted (held-out) calibration
    # ------------------------------------------------------------------
    print('[2] cross-fitted (held-out) calibration', flush=True)
    for name, p in scores.items():
        kind = kinds[name]
        entry = cal[name]
        entry['crossfit'] = {}
        for k in K_FOLDS:
            pc, folds, fid = crossfit(kind, p, y, w, k)
            entry['crossfit'][f'{k}fold'] = {
                'design': f'StratifiedKFold(n_splits={k}, shuffle=True, '
                          f'random_state={SEED}); map fitted on k-1 folds, '
                          f'evaluated on the held-out fold; predictions pooled',
                'pooled_held_out': cal_block_with_sklearn_ece(y, pc, w),
                'fold_mean_of_fold_ece_equalcount': float(
                    np.mean([f['ece_equalcount'] for f in folds])),
                'fold_mean_of_fold_ece_equalwidth_01': float(
                    np.mean([f['ece_equalwidth_01'] for f in folds])),
                'fold_mean_auc': float(np.mean([f['auc'] for f in folds])),
                'folds': folds,
            }
            e = entry['crossfit'][f'{k}fold']['pooled_held_out']
            print(f'  {name:14s} k={k:2d} ECE(cw)='
                  f'{e["ece_weighted_equalcount_decile"]:.6f} '
                  f'ECE(ew1)={e["ece_weighted_equalwidth_01"]:.6f} '
                  f'ECE(ewrng)={e["ece_weighted_equalwidth_range"]:.6f} '
                  f'uw-cw={e["ece_unweighted_equalcount_decile"]:.6f} '
                  f'brier={e["brier_weighted"]:.6f}', flush=True)
        # single fixed 50/50 split
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(y))
        # stratified 50/50 split
        idx_ev = perm[y[perm] == 1]
        idx_ne = perm[y[perm] == 0]
        tr = np.concatenate([idx_ev[:len(idx_ev) // 2], idx_ne[:len(idx_ne) // 2]])
        te = np.concatenate([idx_ev[len(idx_ev) // 2:], idx_ne[len(idx_ne) // 2:]])
        pc1, folds1, fid1 = crossfit(kind, p, y, w, None, split=(tr, te))
        entry['crossfit']['single_split_50_50'] = {
            'design': 'stratified single 50/50 split (seed 42); map fitted on '
                      'the training half, evaluated on the held-out half',
            'n_train': int(len(tr)), 'n_test': int(len(te)),
            'events_test': int(y[te].sum()),
            'held_out_test_half': cal_block_with_sklearn_ece(y[te], pc1[te], w[te]),
            'folds': folds1,
        }
        e = entry['crossfit']['single_split_50_50']['held_out_test_half']
        print(f'  {name:14s} 50/50 split ECE(cw)='
              f'{e["ece_weighted_equalcount_decile"]:.6f} '
              f'ECE(ew1)={e["ece_weighted_equalwidth_01"]:.6f}', flush=True)

    # ------------------------------------------------------------------
    # 3. rank invariance of threshold strategies
    # ------------------------------------------------------------------
    print('[3] rank-based vs absolute-probability thresholds', flush=True)
    oc = out['operating_characteristics_rank_vs_absolute']
    yb = y.astype(bool)

    def wstat(pred):
        tp = w[pred & yb].sum(); fp = w[pred & ~yb].sum()
        fn = w[~pred & yb].sum(); tn = w[~pred & ~yb].sum()
        return {'sens': float(tp / max(tp + fn, 1e-9)),
                'spec': float(tn / max(tn + fp, 1e-9)),
                'ppv': float(tp / max(tp + fp, 1e-9)),
                'flagged_pct': float(w[pred].sum() / w.sum()),
                'flagged_n': int(pred.sum())}

    for name, p in [('xgb', p_xgb), ('lr', p_lr)]:
        variants = {'raw_frozen_probability': p}
        variants['isotonic_in_sample'] = _predict_map(
            'isotonic', _fit_map('isotonic', p, y, w), p)
        pc_cf, _, _ = crossfit('isotonic', p, y, w, 5)
        variants['isotonic_crossfit_5fold'] = pc_cf
        e = {'spearman': {}, 'top_pct_quantile_threshold': {},
             'top_pct_strict_rank': {}, 'youden': {},
             'fixed_absolute_threshold': {}, 'rank_diagnostics': {}}
        names = list(variants)
        for a in names:
            for b in names:
                if a < b:
                    e['spearman'][f'{a}__vs__{b}'] = float(
                        spearmanr(variants[a], variants[b]).statistic)
        # --- tie/rank diagnostics -----------------------------------------
        # A strictly increasing map preserves the ordering of every pair, so
        # any statistic that depends only on the ordering is invariant.
        # Isotonic regression, however, is only NON-DECREASING: it maps whole
        # intervals of p onto one value. Tied pairs are scored 0.5 in the
        # Mann-Whitney AUC, so tie creation can *change* the reported AUC even
        # though the map is monotone.
        for vname, pv in variants.items():
            ev, ne = pv[yb], pv[~yb]
            ne_s = np.sort(ne)
            lo = np.searchsorted(ne_s, ev, side='left')
            hi = np.searchsorted(ne_s, ev, side='right')
            conc = int(lo.sum()); ties = int((hi - lo).sum())
            e['rank_diagnostics'][vname] = {
                'n_unique_values': int(len(np.unique(pv))),
                'pairwise_concordant': conc,
                'pairwise_tied': ties,
                'pairwise_total': int(len(ev) * len(ne)),
                'tied_fraction': float(ties / (len(ev) * len(ne))),
                'auc_mannwhitney_ties_at_half':
                    float((conc + 0.5 * ties) / (len(ev) * len(ne))),
            }
        # --- (c) rank-based strategy, as implemented in the pipeline ------
        # threshold = np.quantile(p, 1-X); flag p >= threshold. With a
        # tie-creating map this flags MORE than X% of admissions.
        for tgt in TOP_PCTS:
            row = {}
            for vname, pv in variants.items():
                t = np.quantile(pv, 1 - tgt)
                r = wstat(pv >= t)
                r['threshold'] = float(t)
                row[vname] = r
            base = variants['raw_frozen_probability'] >= np.quantile(
                variants['raw_frozen_probability'], 1 - tgt)
            for vname, pv in variants.items():
                s = pv >= np.quantile(pv, 1 - tgt)
                inter = int((s & base).sum()); uni = int((s | base).sum())
                row[vname]['jaccard_vs_raw'] = inter / max(uni, 1)
            e['top_pct_quantile_threshold'][f'top_{int(tgt*100)}pct'] = row
        # --- (c') strict-rank strategy: exactly the top k admissions ------
        # This is the strategy the invariance argument is about. Ties are
        # broken by the raw frozen probability so the comparison is fair.
        for tgt in TOP_PCTS:
            k = int(round(tgt * len(y)))
            row = {}
            for vname, pv in variants.items():
                order = np.lexsort((p, pv))          # ascending, ties -> raw p
                pred = np.zeros(len(y), dtype=bool)
                pred[order[-k:]] = True
                r = wstat(pred)
                r['threshold_implied'] = float(np.min(pv[pred]))
                r['n_flagged'] = int(pred.sum())
                row[vname] = r
            base_order = np.lexsort((p, variants['raw_frozen_probability']))
            base = np.zeros(len(y), dtype=bool); base[base_order[-k:]] = True
            for vname, pv in variants.items():
                order = np.lexsort((p, pv))
                s = np.zeros(len(y), dtype=bool); s[order[-k:]] = True
                inter = int((s & base).sum()); uni = int((s | base).sum())
                row[vname]['jaccard_vs_raw'] = inter / max(uni, 1)
                row[vname]['exact_same_set_as_raw'] = bool(
                    np.array_equal(s, base))
            e['top_pct_strict_rank'][f'top_{int(tgt*100)}pct'] = row
        # Youden-optimal: sens+spec is a rank statistic, so the *operating
        # point* is invariant; the numeric threshold is map-specific.
        for vname, pv in variants.items():
            cand = np.quantile(pv, np.linspace(0.5, 0.999, 200))
            best = None
            for t in cand:
                s = wstat(pv >= t)
                j = s['sens'] + s['spec'] - 1
                if best is None or j > best[0]:
                    best = (j, t, s)
            bs = dict(best[2]); bs['threshold'] = float(best[1])
            bs['youden'] = float(best[0])
            bs['n_flagged'] = int((pv >= best[1]).sum())
            e['youden'][vname] = bs
        # absolute fixed MIMIC thresholds applied on each probability scale
        for t in FIXED_THRESHOLDS:
            row = {}
            for vname, pv in variants.items():
                r = wstat(pv >= t)
                r['threshold'] = t
                row[vname] = r
            e['fixed_absolute_threshold'][f'{t}'] = row
        e['auc_by_variant'] = {v: (wfast_auc(yb, pv, w) if w.min() < w.max()
                                   else fast_auc(yb, pv))
                               for v, pv in variants.items()}
        oc[name] = e
        print(f'  {name}: AUC by variant '
              + ' '.join(f'{v}={e["auc_by_variant"][v]:.6f}' for v in names),
              flush=True)
        print('    unique values / tied pair fraction: ' + '; '.join(
            f'{v}={e["rank_diagnostics"][v]["n_unique_values"]}/'
            f'{e["rank_diagnostics"][v]["tied_fraction"]:.4f}' for v in names),
            flush=True)
        for tgt in TOP_PCTS:
            k = f'top_{int(tgt*100)}pct'
            print(f'    {k} quantile-thr: flagged% '
                  + ' '.join(f'{v}={e["top_pct_quantile_threshold"][k][v]["flagged_pct"]:.4f}'
                             for v in names)
                  + f'; sens ' + ' '.join(
                      f'{v}={e["top_pct_quantile_threshold"][k][v]["sens"]:.4f}'
                      for v in names), flush=True)
            print(f'    {k} strict-rank : jaccard-vs-raw '
                  + ' '.join(f'{v}={e["top_pct_strict_rank"][k][v]["jaccard_vs_raw"]:.6f}'
                             for v in names), flush=True)
        print('    Youden: ' + ' '.join(
            f'{v}: J={e["youden"][v]["youden"]:.4f} '
            f'sens={e["youden"][v]["sens"]:.4f} spec={e["youden"][v]["spec"]:.4f} '
            f'n_flag={e["youden"][v]["n_flagged"]}' for v in names),
            flush=True)

    # ------------------------------------------------------------------
    # 4. bootstrap CIs for the pooled cross-fitted ECE (5-fold), xgb / lr
    # ------------------------------------------------------------------
    print('[4] bootstrap CI for pooled cross-fitted ECE (5-fold, capped '
          'negatives = same design as the main run)', flush=True)
    for name in ['xgb', 'lr']:
        pc, _, _ = crossfit('isotonic', scores[name], y, w, 5)

        def f_cw(yb_, w_, rows):
            return calibration_error(yb_, pc[rows], w_, 'count')

        def f_ew(yb_, w_, rows):
            return calibration_error(yb_, pc[rows], w_, 'width',
                                     width_range=(0, 1))
        t0 = time.time()
        c_cw, f_cw_ = boot_ci(f_cw, y, w, seed=SEED + 1)
        c_ew, f_ew_ = boot_ci(f_ew, y, w, seed=SEED + 2)
        out['calibration'][name]['crossfit']['5fold']['pooled_held_out'][
            'ece_weighted_equalcount_decile_ci_negcap50k'] = c_cw
        out['calibration'][name]['crossfit']['5fold']['pooled_held_out'][
            'ece_weighted_equalcount_decile_ci_uncapped'] = f_cw_
        out['calibration'][name]['crossfit']['5fold']['pooled_held_out'][
            'ece_weighted_equalwidth_01_ci_negcap50k'] = c_ew
        out['calibration'][name]['crossfit']['5fold']['pooled_held_out'][
            'ece_weighted_equalwidth_01_ci_uncapped'] = f_ew_
        print(f'  {name}: ECE(cw) CI {c_cw[0]:.6f}-{c_cw[1]:.6f} '
              f'(uncapped {f_cw_[0]:.6f}-{f_cw_[1]:.6f}) | '
              f'ECE(ew1) CI {c_ew[0]:.6f}-{c_ew[1]:.6f} '
              f'(uncapped {f_ew_[0]:.6f}-{f_ew_[1]:.6f}) '
              f'({time.time()-t0:.0f}s)', flush=True)

    out['runtime_s'] = round(time.time() - t_start, 1)
    sout = (f'{BASE}/external_validation_result_highrisk_nested.json'
            if HIGH_RISK_ONLY
            else f'{BASE}/external_validation_result_nested.json')
    json.dump(out, open(sout, 'w'), indent=2, default=str)
    print(f'\nsaved -> {sout} ({out["runtime_s"]}s)', flush=True)


if __name__ == '__main__':
    main()
