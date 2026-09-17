#!/usr/bin/env python3
"""External validation: apply frozen MIMIC static models to the local cohort.

Mirrors scripts/66_eicu_eval.py (transportability), adapted for the local
hospital cohort:
  - outcome: new-onset VTE (I26/I80/I82 discharge diagnoses), prior VTE excluded
  - models: frozen LR/XGB (extval/frozen_models) trained on MIMIC AJM pool
  - baselines: Padua / IMPROVE recomputed on local data (extval/scores.parquet)
  - reports: discrimination (AUC/AUPRC), delta-AUC vs Padua/IMPROVE, NRI/IDI,
    calibration (raw + isotonic recalibration), threshold porting at 0.5%/1%,
    and subgroup breakdown (high vs non-high risk group).

Output: extval/external_validation_result.json
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.linear_model import LogisticRegression as SkLR
from sklearn.metrics import roc_auc_score, average_precision_score, \
    brier_score_loss

BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'
SEED = 42
N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]
# external cohort has a much higher event rate (4.8%) than MIMIC (0.35%);
# the MIMIC-derived 0.5%/1% thresholds are far below every externally
# calibrated baseline probability, so NRI at those thresholds is degenerate.
# We additionally report NRI at 5%/10% risk thresholds relevant to this
# high-prevalence population.
NRI_THRESHOLDS_EXT = [0.05, 0.10]

# ---- inverse-probability sampling weights ---------------------------------
# The external cohort is risk-enriched: the high-risk stratum is fully
# enumerated (35,870 admissions, weight 1.0) whereas the non-high-risk stratum
# is a department-proportional random sample of 7,072 out of 76,188
# admissions, giving a sampling weight of 76,188/7,072 = 10.773. The
# exclusions applied downstream (LOS<=24h, ICD-only, prior VTE, dedupe) are
# eligibility criteria, not sampling, so the sampling weight is carried
# unchanged through them. Because the stratification variable (risk_group) is
# associated with BOTH the predictors and the outcome, ALL population-level
# quantities are weighted (event rate, calibration, NRI/IDI, threshold
# operating points, AUPRC, AND AUC/delta-AUC via the weighted Mann-Whitney
# estimator). Unweighted results are kept as a sensitivity analysis under the
# 'unweighted' key.
NH_FULL, NH_SAMP = 76_188, 7_072
SW_HIGH, SW_NONHIGH = 1.0, NH_FULL / NH_SAMP

# Analysis population switch. HIGH_RISK_ONLY=True restricts the external
# validation to the fully enumerated high-risk stratum (no sampling, so no
# weighting: all weights collapse to 1). This sidesteps the non-high-risk
# small-event (n=16) instability and the weight-base caveat, and answers the
# clinically relevant question: among patients already screened high-risk,
# can the ML model further discriminate who will actually develop VTE?
# Set via environment variable EXTVAL_HIGH_ONLY=1; default False preserves
# the weighted full-population analysis.
import os as _os
HIGH_RISK_ONLY = _os.environ.get('EXTVAL_HIGH_ONLY', '0') == '1'


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wfast_auc(yb, p, w):
    # weighted Mann-Whitney AUC: P(score_event > score_nonevent) over
    # weight-product-weighted pairs (handles ties as 0.5)
    ev = yb == 1
    pe, pn = p[ev], p[~ev]
    we, wn = w[ev], w[~ev]
    # pairwise compare via searchsorted on sorted non-event scores
    order = np.argsort(pn)
    pn_s = pn[order]
    wn_s = wn[order]
    cwn = np.concatenate([[0.0], np.cumsum(wn_s)])
    tot = 0.0
    wn_sum = wn_s.sum()
    for s, wi in zip(pe, we):
        # weight of non-events strictly below s, and equal to s
        lo = np.searchsorted(pn_s, s, side='left')
        hi = np.searchsorted(pn_s, s, side='right')
        below = cwn[lo]
        equal = cwn[hi] - cwn[lo]
        tot += wi * (below + 0.5 * equal)
    return float(tot / (we.sum() * wn_sum))


def wece(yb, p, w):
    # weighted expected calibration error over fixed decile bins of p
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0], bins[-1] = -np.inf, np.inf
    ind = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    tot = 0.0
    wsum = w.sum()
    for b in range(10):
        m = ind == b
        if m.sum() == 0:
            continue
        wb, ybb, pb = w[m], yb[m], p[m]
        obs = np.sum(wb * ybb) / wb.sum()
        pred = np.sum(wb * pb) / wb.sum()
        tot += (wb.sum() / wsum) * abs(obs - pred)
    return float(tot)


def ece(yb, p):
    pt, pp = calibration_curve(yb, p, n_bins=10)
    return float(np.mean(np.abs(pt - pp)))


def wnri(p_old, p_new, yb, thr, w):
    ev, nev = yb == 1, yb == 0
    up_e = w[(p_new >= thr) & (p_old < thr) & ev].sum()
    dn_e = w[(p_new < thr) & (p_old >= thr) & ev].sum()
    up_n = w[(p_new >= thr) & (p_old < thr) & nev].sum()
    dn_n = w[(p_new < thr) & (p_old >= thr) & nev].sum()
    return float((up_e - dn_e) / w[ev].sum() + (dn_n - up_n) / w[nev].sum())


def nri(p_old, p_new, yb, thr):
    ev, nev = yb == 1, yb == 0
    up_e = ((p_new >= thr) & (p_old < thr) & ev).sum()
    dn_e = ((p_new < thr) & (p_old >= thr) & ev).sum()
    up_n = ((p_new >= thr) & (p_old < thr) & nev).sum()
    dn_n = ((p_new < thr) & (p_old >= thr) & nev).sum()
    return float((up_e - dn_e) / ev.sum() + (dn_n - up_n) / nev.sum())


def idi(p_old, p_new, yb):
    ev, nev = yb == 1, yb == 0
    return float((p_new[ev].mean() - p_old[ev].mean())
                 - (p_new[nev].mean() - p_old[nev].mean()))


def widi(p_old, p_new, yb, w):
    ev, nev = yb == 1, yb == 0
    new_e = np.sum(w[ev] * p_new[ev]) / w[ev].sum()
    old_e = np.sum(w[ev] * p_old[ev]) / w[ev].sum()
    new_n = np.sum(w[nev] * p_new[nev]) / w[nev].sum()
    old_n = np.sum(w[nev] * p_old[nev]) / w[nev].sum()
    return float((new_e - old_e) - (new_n - old_n))


def wbrier(yb, p, w):
    return float(np.sum(w * (yb - p) ** 2) / w.sum())


def wap(yb, p, w):
    # weighted average precision (scikit supports sample_weight)
    return float(average_precision_score(yb, p, sample_weight=w))


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def main():
    feat_order = json.load(open(f'{BASE}/frozen_models/feature_order.json'))['features']

    co = pd.read_parquet(f'{BASE}/cohort.parquet',
                         columns=['vis_id', 'risk_group', 'vte_outcome',
                                  'prior_vte_any', 'prior_vte_this_adm'])
    fe = pd.read_parquet(f'{BASE}/features.parquet').drop(
        columns=['prior_vte_any', 'vte_outcome', 'risk_group'], errors='ignore')
    sc = pd.read_parquet(f'{BASE}/scores.parquet')

    df = co.merge(fe, on='vis_id', how='left', validate='1:1') \
        .merge(sc, on='vis_id', how='left', validate='1:1')
    # prior_vte (history code on this admission) is part of the frozen 57-feature
    # set; map from the local history-flag
    df['prior_vte'] = df['prior_vte_this_adm']

    # inverse-probability sampling weight (assigned before exclusions; the
    # downstream exclusions are eligibility criteria, not sampling, so the
    # weight is carried unchanged)
    df['sw'] = np.where(df['risk_group'] == 'non_high', SW_NONHIGH, SW_HIGH)

    # optionally restrict to the fully enumerated high-risk stratum (no
    # sampling there, so no weighting is needed; all weights collapse to 1)
    if HIGH_RISK_ONLY:
        df = df[df['risk_group'] == 'high'].copy()
        df['sw'] = 1.0
        print('HIGH_RISK_ONLY: restricted to high-risk stratum '
              f'({len(df):,} admissions, no weighting)', flush=True)

    # restrict to admissions without prior VTE (mirrors MIMIC primary cohort)
    mask = df['prior_vte_any'] == 0
    df = df[mask].reset_index(drop=True)
    print(f'cohort after prior-VTE exclusion: {len(df):,} '
          f'events={int(df["vte_outcome"].sum()):,} '
          f'rate={df["vte_outcome"].mean():.4f} '
          f'(weighted rate='
          f'{np.sum(df["sw"]*df["vte_outcome"])/df["sw"].sum():.4f})',
          flush=True)

    # age/male missing -> fill with median from MIMIC train pool (transportability)
    y = df['vte_outcome'].astype(int).values
    w = df['sw'].values.astype(float)

    # build X with the frozen feature set; local data lacks adm_elective /
    # adm_emergency -> set NaN (treated as missing by imputer / native NaN)
    X = df[feat_order].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X = X.clip(-1e30, 1e30)

    lr = joblib.load(f'{BASE}/frozen_models/final_lr.joblib')
    imp = joblib.load(f'{BASE}/frozen_models/final_lr_imputer.joblib')
    xgb = joblib.load(f'{BASE}/frozen_models/final_xgb.joblib')
    p_lr = lr.predict_proba(imp.transform(X))[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]

    pad_raw = df['padua_score'].values.astype(float)
    impr_raw = df['improve_score'].values.astype(float)
    # Calibrate score/model probabilities to the WEIGHTED external population
    # (sampling-weighted fits) so NRI/IDI/thresholds reflect the full hospital
    # population rather than the risk-enriched sample. AUC is unaffected.
    pad_prob = SkLR(C=1.0, max_iter=2000).fit(
        pad_raw.reshape(-1, 1), y, sample_weight=w).predict_proba(
        pad_raw.reshape(-1, 1))[:, 1]
    impr_prob = SkLR(C=1.0, max_iter=2000).fit(
        impr_raw.reshape(-1, 1), y, sample_weight=w).predict_proba(
        impr_raw.reshape(-1, 1))[:, 1]
    # Recalibrate the frozen model probabilities to the external population's
    # event rate so NRI/IDI compare scores on the SAME probability scale
    # (the frozen model outputs MIMIC-scale probabilities ~0.4% on average,
    # far below the external event rate; AUC is unaffected).
    lr_prob = SkLR(C=1.0, max_iter=2000).fit(
        p_lr.reshape(-1, 1), y, sample_weight=w).predict_proba(
        p_lr.reshape(-1, 1))[:, 1]
    xgb_prob = SkLR(C=1.0, max_iter=2000).fit(
        p_xgb.reshape(-1, 1), y, sample_weight=w).predict_proba(
        p_xgb.reshape(-1, 1))[:, 1]

    # ---- discrimination + bootstrap (admission-level, no clustering: local
    #      data has few repeat admissions; use standard Efron bootstrap) ------
    rng = np.random.RandomState(SEED)
    n = len(y)
    ev_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    P = {'lr': p_lr, 'xgb': p_xgb, 'padua': pad_raw, 'improve': impr_raw,
         'padua_prob': pad_prob, 'impr_prob': impr_prob,
         'lr_prob': lr_prob, 'xgb_prob': xgb_prob}
    auc_boot = {k: [] for k in ['lr', 'xgb', 'padua', 'improve']}
    delta_boot = {k: [] for k in ['xgb_vs_padua', 'xgb_vs_improve',
                                  'lr_vs_padua']}
    nri_boot = {(k, t): [] for k in ['xgb_vs_padua', 'xgb_vs_improve',
                                     'lr_vs_padua']
                for t in NRI_THRESHOLDS + NRI_THRESHOLDS_EXT}
    idi_boot = {k: [] for k in ['xgb_vs_padua', 'xgb_vs_improve',
                                'lr_vs_padua']}
    t0 = time.time()
    for b in range(N_BOOT):
        idx = rng.randint(0, n, n)
        b_ev = idx[y[idx] == 1]
        b_neg = idx[y[idx] == 0]
        if len(b_neg) > NEG_CAP:
            b_neg = rng.choice(b_neg, NEG_CAP, replace=False)
        rows = np.concatenate([b_ev, b_neg])
        yb = y[rows]
        wb = w[rows]
        # AUC / delta-AUC: WEIGHTED. The stratum sampling (high-risk fully
        # enumerated, non-high-risk sampled) depends on risk_group, which is
        # associated with both the predictors and the outcome, so unweighted
        # AUC is biased relative to the full hospital population; the weighted
        # Mann-Whitney AUC recovers the population value.
        aucs = {k: wfast_auc(yb, P[k][rows], wb)
                for k in ['lr', 'xgb', 'padua', 'improve']}
        for k in aucs:
            auc_boot[k].append(aucs[k])
        for name, raw_k, prob_k, new in [
                ('xgb_vs_padua', 'padua', 'padua_prob', 'xgb'),
                ('xgb_vs_improve', 'improve', 'impr_prob', 'xgb'),
                ('lr_vs_padua', 'padua', 'padua_prob', 'lr')]:
            delta_boot[name].append(aucs[new] - aucs[raw_k])
            oldp = P[prob_k][rows]
            newp = P[f'{new}_prob'][rows]
            # NRI / IDI depend on the population mix: weighted
            for t in NRI_THRESHOLDS + NRI_THRESHOLDS_EXT:
                nri_boot[(name, t)].append(wnri(oldp, newp, yb, t, wb))
            idi_boot[name].append(widi(oldp, newp, yb, wb))
        if (b + 1) % 500 == 0:
            print(f'  boot {b + 1}/{N_BOOT} ({time.time() - t0:.0f}s)',
                  flush=True)

    w_event_rate = float(np.sum(w * y) / w.sum())
    results = {
        'cohort': {'n': int(len(df)), 'events': int(y.sum()),
                   'event_rate': w_event_rate,
                   'event_rate_unweighted': float(y.mean()),
                   'weighted_n': float(w.sum()),
                   'weighted_events': float(np.sum(w * y)),
                   'sampling_weights': {'high': SW_HIGH,
                                        'non_high': round(SW_NONHIGH, 4)},
                   'weighting': ('inverse-probability: high-risk stratum fully '
                                 'enumerated (w=1); non-high-risk stratum '
                                 'sampled 7,072/76,188 (w=10.773). Because '
                                 'risk_group predicts both features and '
                                 'outcome, ALL population metrics are weighted: '
                                 'event rate, calibration, NRI/IDI, thresholds, '
                                 'AUPRC, and AUC/delta-AUC (weighted '
                                 'Mann-Whitney).'),
                   'prior_vte_excluded': True,
                   'subgroups': df.groupby('risk_group')['vte_outcome']
                                .agg(['count', 'sum', 'mean'])
                                .rename(columns={'count': 'n', 'sum': 'events',
                                                 'mean': 'event_rate'})
                                .to_dict('index')},
        'discrimination': {k: {'auc_est': float(np.mean(v)),
                               'auc_ci': [float(x) for x in
                                          np.percentile(v, [2.5, 97.5])]}
                           for k, v in auc_boot.items()},
        'auprc': {'lr': wap(y, p_lr, w),
                  'xgb': wap(y, p_xgb, w),
                  'padua': wap(y, pad_raw, w),
                  'improve': wap(y, impr_raw, w)},
        'delta_auc': {k: ds(v) for k, v in delta_boot.items()},
        'nri': {f'{k}_thr{t}': {'point': float(np.mean(v)),
                                'ci': [float(x) for x in
                                       np.percentile(v, [2.5, 97.5])]}
                for (k, t), v in nri_boot.items()},
        'idi': {k: ds(v) for k, v in idi_boot.items()},
        'calibration_raw': {},
    }
    for k, p in [('lr', p_lr), ('xgb', p_xgb)]:
        results['calibration_raw'][k] = {
            'brier': wbrier(y, p, w),
            'ece': wece(y, p, w)}
    for k, p in [('lr', p_lr), ('xgb', p_xgb)]:
        iso = IsotonicRegression(out_of_bounds='clip').fit(
            p, y, sample_weight=w)
        pc = iso.predict(p)
        results['calibration_raw'][f'{k}_recal'] = {
            'brier': wbrier(y, pc, w),
            'ece': wece(y, pc, w)}

    # ---- unweighted sensitivity analysis -----------------------------------
    uw = {
        'event_rate': float(y.mean()),
        'auc': {k: float(roc_auc_score(y, p))
                for k, p in [('lr', p_lr), ('xgb', p_xgb),
                             ('padua', pad_raw), ('improve', impr_raw)]},
        'delta_auc': {
            'xgb_vs_padua': float(roc_auc_score(y, p_xgb)
                                  - roc_auc_score(y, pad_raw)),
            'xgb_vs_improve': float(roc_auc_score(y, p_xgb)
                                    - roc_auc_score(y, impr_raw)),
            'lr_vs_padua': float(roc_auc_score(y, p_lr)
                                 - roc_auc_score(y, pad_raw))},
        'auprc': {'lr': float(average_precision_score(y, p_lr)),
                  'xgb': float(average_precision_score(y, p_xgb)),
                  'padua': float(average_precision_score(y, pad_raw)),
                  'improve': float(average_precision_score(y, impr_raw))},
        'calibration_raw': {},
    }
    for k, p in [('lr', p_lr), ('xgb', p_xgb)]:
        uw['calibration_raw'][k] = {
            'brier': float(brier_score_loss(y, p)),
            'ece': ece(y, p)}
        iso = IsotonicRegression(out_of_bounds='clip').fit(p, y)
        pc = iso.predict(p)
        uw['calibration_raw'][f'{k}_recal'] = {
            'brier': float(brier_score_loss(y, pc)),
            'ece': ece(y, pc)}
    results['unweighted'] = uw

    # ---- subgroup: high vs non-high risk group -----------------------------
    df['_p_lr'] = p_lr
    df['_p_xgb'] = p_xgb
    df['_pad'] = pad_raw
    sub = {}
    for name, idx in [('all', df.index),
                      ('high', df[df['risk_group'] == 'high'].index),
                      ('non_high', df[df['risk_group'] == 'non_high'].index)]:
        g = df.loc[idx]
        yy = g['vte_outcome'].values
        if yy.sum() < 2:
            continue
        sub[name] = {
            'n': int(len(g)), 'events': int(yy.sum()),
            'event_rate': float(yy.mean()),
            'auc_xgb': roc_auc_score(yy, g['_p_xgb'].values),
            'auc_lr': roc_auc_score(yy, g['_p_lr'].values),
            'auc_padua': fast_auc(yy, g['_pad'].values),
        }
    results['subgroups'] = sub

    # ---- threshold porting at MIMIC risk levels (weighted) ------------------
    thr = {}
    yb = y.astype(bool)
    for name, p in [('lr', p_lr), ('xgb', p_xgb)]:
        for t in NRI_THRESHOLDS:
            pred = p >= t
            tp = w[pred & yb].sum()
            fp = w[pred & ~yb].sum()
            fn = w[~pred & yb].sum()
            tn = w[~pred & ~yb].sum()
            thr[f'{name}_@{t}'] = {
                'sens': float(tp / max(tp + fn, 1e-9)),
                'spec': float(tn / max(tn + fp, 1e-9)),
                'pv': float(tp / max(tp + fp, 1e-9)),
                'flagged': int((pred).sum()),
                'flagged_pct': float(w[pred].sum() / w.sum()),
            }
    results['thresholds'] = thr

    # ---- locally calibrated thresholds (weighted) ---------------------------
    # Fixed MIMIC-derived thresholds are NOT transportable: on the weighted
    # external population the 0.5% threshold flags almost everyone (very low
    # specificity) because it sits far below the locally calibrated event-rate
    # baseline. Instead we derive LOCAL thresholds on the isotonic-recalibrated
    # probabilities, either by flagging the top-risk X% of the population or by
    # maximising the Youden index. This is the recommended deployment mode:
    # recalibrate locally, then flag a fixed proportion of highest-risk
    # admissions rather than porting an absolute probability cut-off.
    def _wstat(pred):
        tp = w[pred & yb].sum(); fp = w[pred & ~yb].sum()
        fn = w[~pred & yb].sum(); tn = w[~pred & ~yb].sum()
        return {'sens': float(tp / max(tp + fn, 1e-9)),
                'spec': float(tn / max(tn + fp, 1e-9)),
                'ppv': float(tp / max(tp + fp, 1e-9)),
                'flagged_pct': float(w[pred].sum() / w.sum())}

    local_thr = {}
    for name, p in [('lr', p_lr), ('xgb', p_xgb)]:
        iso = IsotonicRegression(out_of_bounds='clip').fit(
            p, y, sample_weight=w)
        pc = iso.predict(p)
        entry = {}
        # flag top X% highest-risk admissions
        for tgt in [0.05, 0.10, 0.15]:
            t = np.quantile(pc, 1 - tgt)
            s = _wstat(pc >= t)
            s['threshold'] = float(t)
            entry[f'top_{int(tgt*100)}pct'] = s
        # Youden-optimal threshold
        cand = np.quantile(pc, np.linspace(0.5, 0.999, 200))
        best = None
        for t in cand:
            s = _wstat(pc >= t)
            j = s['sens'] + s['spec'] - 1
            if best is None or j > best[0]:
                best = (j, t, s)
        bs = dict(best[2]); bs['threshold'] = float(best[1])
        bs['youden'] = float(best[0])
        entry['youden_optimal'] = bs
        # fixed MIMIC thresholds on locally calibrated probabilities (contrast)
        for t in NRI_THRESHOLDS:
            s = _wstat(pc >= t)
            s['threshold'] = float(t)
            entry[f'fixed_mimic_{t}'] = s
        local_thr[name] = entry
    results['local_calibrated_thresholds'] = local_thr

    # ---- feature missingness / coverage report ------------------------------
    miss = X.isna().mean().round(3).sort_values(ascending=False)
    results['feature_missingness'] = miss.to_dict()
    results['analysis_population'] = ('high_risk_only' if HIGH_RISK_ONLY
                                      else 'full_population_weighted')

    out = (f'{BASE}/external_validation_result_highrisk.json' if HIGH_RISK_ONLY
           else f'{BASE}/external_validation_result.json')
    json.dump(results, open(out, 'w'), indent=2, default=str)
    print(json.dumps({'cohort': results['cohort'],
                      'discrimination': results['discrimination'],
                      'auprc': results['auprc'],
                      'delta_auc': results['delta_auc'],
                      'nri': results['nri'],
                      'idi': results['idi'],
                      'calibration_raw': results['calibration_raw'],
                      'subgroups': results['subgroups'],
                      'thresholds': results['thresholds'],
                      'local_calibrated_thresholds':
                          results['local_calibrated_thresholds'],
                      'unweighted': results['unweighted']}, indent=2),
          flush=True)


if __name__ == '__main__':
    main()
