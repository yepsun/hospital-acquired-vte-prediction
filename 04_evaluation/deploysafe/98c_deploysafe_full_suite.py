#!/usr/bin/env python3
"""Script 98c: deployment-safe (look-ahead-free) feature sets with the FULL
inference suite, so the admission-safe model can stand in for (or replace)
the primary 57-feature analysis.

Protocol is scripts_era/98b_caprini_comparison.py verbatim, i.e. the primary
protocol of scripts/98_ajm_sensitivity_cohorts.py extended to three clinical
comparators:
  * LR (L2, C=1.0, per-fold training-median imputation)
  * XGBoost (300 trees, depth 3, lr 0.05, subsample 0.8, colsample 0.8,
    min_child_weight 5, reg_lambda 1, hist, native missing handling),
    random_state FIXED at 42 across repeats (primary convention)
  * 3 repeats x 5-fold StratifiedGroupKFold, group = subject_id, seeds 42/43/44
  * fold-mean AUC with 2,000 fold bootstrap CI
  * seed-42 pooled OOF AUC / AUPRC / Brier / ECE + patient-level cluster
    bootstrap CI (2,000 replicates, negatives capped at 50,000)
  * held-out test-set AUC / AUPRC
  * dAUC / NRI (0.5%, 1%) / IDI of each model against padua, improve, caprini
    (Platt-calibrated score probabilities), plus xgb vs lr

Arms (feature lists are the exact `main` order of feature_sets_v2.json):
  main_57                : the published primary feature set
  strict_42              : main minus STRICT_DROP (15 discharge-ICD columns)
  conservative_40        : strict_42 minus prior_vte / prior_vte_any
  strict44_surgtrauma    : strict_42 plus surgery_flag / trauma_flag kept
                           (i.e. main minus the 13 non-procedural ICD columns)

`--arm subgroups` additionally reports surgical / medical service subgroup
AUCs (first curr_service, same definition as scripts_era/60b) for the 42- and
57-feature XGBoost / LR models.

Never overwrites an existing file: every output path ends in `_deploysafe`.
Run from the repository root that contains results_vte/.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RES = 'results_vte'
D = f'{RES}/ajm/new/data_era'
O = f'{RES}/ajm/new/output_era'
DATASET = f'{D}/primary_inclprior_excl24h_model_dataset_caprini.parquet'
SPLITS = f'{D}/primary_inclprior_excl24h_splits.json'
SEED = 42
N_REPEATS, N_SPLITS = 3, 5
SEEDS = [SEED + r for r in range(N_REPEATS)]
N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]

SCORE_COLS = {'padua': 'padua_score', 'improve': 'improve_score',
              'caprini': 'caprini_score'}
DELTA_PAIRS = [('lr', 'padua'), ('xgb', 'padua'),
               ('lr', 'improve'), ('xgb', 'improve'),
               ('lr', 'caprini'), ('xgb', 'caprini'),
               ('xgb', 'lr')]
AUC_KEYS = ['lr', 'xgb', 'padua', 'improve', 'caprini']

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

# scripts_era/62_review_gaps.py:42-46 -- reused verbatim.
STRICT_DROP = ['cancer_active', 'heart_failure', 'copd', 'chronic_liver',
               'ckd', 'diabetes', 'stroke', 'mi', 'obesity_icd', 'varicose',
               'thrombophilia', 'infection_severe', 'rheumatologic',
               'surgery_flag', 'trauma_flag']
CONSERVATIVE_EXTRA = ['prior_vte', 'prior_vte_any']
# the two flags a clinician could argue are known at admission
SURG_TRAUMA = ['surgery_flag', 'trauma_flag']

MED_SERVICES = {'MED', 'CMED', 'OMED', 'NMED', 'GU', 'GYN', 'PSYCH', 'OBS',
                'EYE', 'DENT', 'ENT', 'NB'}
SURG_SERVICES = {'SURG', 'NSURG', 'PSURG', 'VSURG', 'CSURG', 'TSURG', 'ORTHO',
                 'TRAUM'}
SVC_CACHE = f'{D}/services_primary_caprini.parquet'


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


ARM = argval('--arm', 'main_57')
SUFFIX = '_deploysafe'


# --------------------------------------------------------------------------
# helpers (identical formulas to scripts_era/98b_caprini_comparison.py)
# --------------------------------------------------------------------------
def fit_predict(kind, X_tr, y_tr, X_te):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1], m
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], m


def run_cv(kind, X, y, groups, seed, N):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=seed)
    oof = np.zeros(N)
    fr = []
    for tr, te in sgkf.split(X, y, groups):
        preds, _ = fit_predict(kind, X[tr], y[tr], X[te])
        oof[te] = preds
        fr.append({'auc': float(roc_auc_score(y[te], preds)),
                   'auprc': float(average_precision_score(y[te], preds)),
                   'brier': float(brier_score_loss(y[te], preds))})
    return fr, oof


def fold_boot_ci(fr, key='auc'):
    rng = np.random.RandomState(SEED)
    vals = [np.mean([fr[i][key] for i in rng.randint(0, len(fr), len(fr))])
            for _ in range(N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(yb, p):
    pt, pp = calibration_curve(yb, p, n_bins=10)
    return float(np.mean(np.abs(pt - pp)))


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


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def pkey(k):
    return f'{k}_prob' if k in SCORE_COLS else k


def print_block(name, res, inf):
    print(f'[{name}] LR : fold-mean AUC={res["lr"]["fold_mean_auc"]:.4f} '
          f'[{res["lr"]["fold_mean_auc_ci"][0]:.4f}-'
          f'{res["lr"]["fold_mean_auc_ci"][1]:.4f}] '
          f'OOF={res["lr"]["oof_auc"]:.4f} test={res["test_set"]["lr_auc"]:.4f}',
          flush=True)
    print(f'[{name}] XGB: fold-mean AUC={res["xgb"]["fold_mean_auc"]:.4f} '
          f'[{res["xgb"]["fold_mean_auc_ci"][0]:.4f}-'
          f'{res["xgb"]["fold_mean_auc_ci"][1]:.4f}] '
          f'OOF={res["xgb"]["oof_auc"]:.4f} test={res["test_set"]["xgb_auc"]:.4f}',
          flush=True)
    for k in [f'{a}_vs_{b}' for a, b in DELTA_PAIRS]:
        d = inf['delta_auc'][k]
        print(f"  dAUC {k}: {d['estimate']:+.4f} "
              f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}",
              flush=True)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def build_arms(MAIN):
    return {
        'main_57': list(MAIN),
        'strict_42': [f for f in MAIN if f not in STRICT_DROP],
        'conservative_40': [f for f in MAIN
                            if f not in STRICT_DROP + CONSERVATIVE_EXTRA],
        'strict44_surgtrauma': [f for f in MAIN
                                if f not in STRICT_DROP
                                or f in SURG_TRAUMA],
    }


def run_arm(name, feats, df, splits, MAIN):
    t_start = time.time()
    out_path = f'{O}/deploysafe_full_suite_{name}{SUFFIX}.json'
    print(f'=== arm {name}: {len(feats)} features -> {out_path}', flush=True)

    pool_ids = set(splits['train']) | set(splits['val'])
    pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
    test = df[~df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
    X = pool[feats].values.astype(np.float32)
    y = pool['vte_event'].values.astype(int)
    groups = pool['subject_id'].values
    N = len(y)
    print(f'[{name}] pool N={N} events={y.sum()} ({y.mean()*100:.3f}%) '
          f'subjects={pool["subject_id"].nunique()} | test N={len(test)} '
          f'events={int(test["vte_event"].sum())}', flush=True)

    raw, prob = {}, {}
    for sname, col in SCORE_COLS.items():
        raw[sname] = pool[col].values.astype(float)
        cal = LogisticRegression(C=1.0, max_iter=2000)
        cal.fit(raw[sname].reshape(-1, 1), y)
        prob[sname] = cal.predict_proba(raw[sname].reshape(-1, 1))[:, 1]

    results, oofs = {}, {}
    for kind in ['lr', 'xgb']:
        t0 = time.time()
        all_fr = []
        for seed in SEEDS:
            fr, oof = run_cv(kind, X, y, groups, seed, N)
            all_fr.extend(fr)
            if seed == SEED:
                oofs[kind] = oof
        ci = fold_boot_ci(all_fr)
        results[kind] = {
            'fold_mean_auc': float(np.mean([f['auc'] for f in all_fr])),
            'fold_mean_auc_ci': ci,
            'fold_mean_auprc': float(np.mean([f['auprc'] for f in all_fr])),
            'fold_mean_auprc_ci': fold_boot_ci(all_fr, 'auprc'),
            'fold_mean_brier': float(np.mean([f['brier'] for f in all_fr])),
            'n_folds': len(all_fr),
            'oof_auc': float(roc_auc_score(y, oofs[kind])),
            'oof_auprc': float(average_precision_score(y, oofs[kind])),
            'oof_brier': float(brier_score_loss(y, oofs[kind])),
            'oof_ece': ece(y, oofs[kind]),
        }
        r = results[kind]
        print(f"[{name}] {kind.upper()}: fold-mean AUC={r['fold_mean_auc']:.4f} "
              f"[{ci[0]:.4f}-{ci[1]:.4f}] AUPRC={r['fold_mean_auprc']:.4f} "
              f"| OOF AUC={r['oof_auc']:.4f} AUPRC={r['oof_auprc']:.4f} "
              f"Brier={r['oof_brier']:.5f} ECE={r['oof_ece']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for sname in SCORE_COLS:
        results[f'{sname}_pool'] = {
            'oof_auc': float(roc_auc_score(y, raw[sname])),
            'oof_auprc': float(average_precision_score(y, raw[sname])),
            'oof_brier': float(brier_score_loss(y, prob[sname])),
            'oof_ece': ece(y, prob[sname]),
            'mean_score': float(raw[sname].mean()),
            'sd_score': float(raw[sname].std()),
        }

    # --- patient-level cluster bootstrap ---------------------------------
    t0 = time.time()
    print(f'[{name}] cluster bootstrap ({N_BOOT} replicates) ...', flush=True)
    subj_codes, subj_uniq = pd.factorize(pool['subject_id'])
    n_subj = len(subj_uniq)
    ev_rows = np.where(y == 1)[0]
    neg_rows = np.where(y == 0)[0]
    ev_subj, neg_subj = subj_codes[ev_rows], subj_codes[neg_rows]

    rng = np.random.RandomState(SEED)
    P = {'lr': oofs['lr'], 'xgb': oofs['xgb'], 'padua': raw['padua'],
         'improve': raw['improve'], 'caprini': raw['caprini'],
         'padua_prob': prob['padua'], 'improve_prob': prob['improve'],
         'caprini_prob': prob['caprini']}
    deltas = {f'{a}_vs_{b}': [] for a, b in DELTA_PAIRS}
    auc_boot = {k: [] for k in AUC_KEYS}
    nri_boot = {(f'{a}_vs_{b}', t): [] for a, b in DELTA_PAIRS
                for t in NRI_THRESHOLDS}
    idi_boot = {f'{a}_vs_{b}': [] for a, b in DELTA_PAIRS}

    for b in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        ev_mult = counts[ev_subj]
        take = ev_mult > 0
        if take.sum() < 2:
            continue
        be_rows = np.repeat(ev_rows[take], ev_mult[take])
        bn_rows = neg_rows[counts[neg_subj] > 0]
        if len(bn_rows) > NEG_CAP:
            bn_rows = rng.choice(bn_rows, NEG_CAP, replace=False)
        rows = np.concatenate([be_rows, bn_rows])
        yb = y[rows]
        a = {k: fast_auc(yb, P[k][rows]) for k in AUC_KEYS}
        for k in a:
            auc_boot[k].append(a[k])
        for p1, p2 in DELTA_PAIRS:
            deltas[f'{p1}_vs_{p2}'].append(a[p1] - a[p2])
        for p1, p2 in DELTA_PAIRS:
            nm = f'{p1}_vs_{p2}'
            po, pn = P[pkey(p2)][rows], P[pkey(p1)][rows]
            for t in NRI_THRESHOLDS:
                nri_boot[(nm, t)].append(nri(po, pn, yb, t))
            idi_boot[nm].append(idi(po, pn, yb))
        if (b + 1) % 500 == 0:
            print(f'  boot {b + 1}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)

    def point_auc(k):
        return results[k]['oof_auc'] if k in ('lr', 'xgb') \
            else results[f'{k}_pool']['oof_auc']

    results['oof_inference'] = {
        'method': 'seed-42 5-fold StratifiedGroupKFold pooled OOF; patient-level '
                  'cluster bootstrap (subject_id), 2000 replicates, negatives '
                  'downsampled to 50k per replicate, AUC via Mann-Whitney rank '
                  'statistic; NRI/IDI use score probabilities from univariate '
                  'logistic (Platt) calibration on the pool; NRI is the standard '
                  'Pencina category-based definition',
        'auprc_note': 'AUPRC has NO CI from this bootstrap: capping negatives at '
                      '50k inflates replicate prevalence, which changes the '
                      'average-precision estimand (AUC is rank-based and is '
                      'unaffected). Point AUPRC is reported in fold_mean_auprc '
                      '(fold-bootstrap CI in fold_mean_auprc_ci), oof_auprc and '
                      'test_set; the pooled-OOF AUPRC cluster-bootstrap CI is '
                      'computed without the negative cap by '
                      'scripts_era/98d_deploysafe_auprc_ci.py.',
        'n_replicates_used': len(auc_boot['lr']),
        'oof_auc_ci': {k: {'auc': point_auc(k),
                           'ci': [float(v) for v in np.percentile(v2,
                                                                  [2.5, 97.5])]}
                       for k, v2 in auc_boot.items()},
        'delta_auc': {k: ds(v) for k, v in deltas.items()},
        'nri': {f'{a}_vs_{b}_thr{t}': {
            'point': nri(P[pkey(b)], P[pkey(a)], y, t),
            'ci': [float(v) for v in np.percentile(nri_boot[(f'{a}_vs_{b}', t)],
                                                   [2.5, 97.5])]}
            for a, b in DELTA_PAIRS for t in NRI_THRESHOLDS},
        'idi': {f'{a}_vs_{b}': {
            'point': idi(P[pkey(b)], P[pkey(a)], y),
            'ci': [float(v) for v in np.percentile(idi_boot[f'{a}_vs_{b}'],
                                                   [2.5, 97.5])]}
            for a, b in DELTA_PAIRS},
    }

    # --- held-out test set ------------------------------------------------
    test_X = test[feats].values.astype(np.float32)
    test_y = test['vte_event'].values.astype(int)
    imp = SimpleImputer(strategy='median')
    m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    m.fit(imp.fit_transform(X), y)
    t_lr = m.predict_proba(imp.transform(test_X))[:, 1]
    m = XGBClassifier(**XGB_PARAMS).fit(X, y)
    t_xgb = m.predict_proba(test_X)[:, 1]
    results['test_set'] = {'n': int(len(test)), 'events': int(test_y.sum()),
                           'event_rate': float(test_y.mean())}
    for kind, p in [('lr', t_lr), ('xgb', t_xgb)]:
        results['test_set'][f'{kind}_auc'] = float(roc_auc_score(test_y, p))
        results['test_set'][f'{kind}_auprc'] = float(
            average_precision_score(test_y, p))
    for sname, col in SCORE_COLS.items():
        s = test[col].values.astype(float)
        results['test_set'][f'{sname}_auc'] = float(roc_auc_score(test_y, s))
        results['test_set'][f'{sname}_auprc'] = float(
            average_precision_score(test_y, s))
    results['test_set']['auprc_baseline_prevalence'] = float(test_y.mean())
    print(f'[{name}] test_set: {json.dumps(results["test_set"])}', flush=True)

    results['feature_sets'] = {
        'arm': name,
        'n_features': len(feats),
        'features': list(feats),
        'dropped_from_main_57': [f for f in MAIN if f not in feats],
        'convention': 'XGBoost random_state fixed at 42 across repeats '
                      '(primary pipeline convention; scripts_era/62_review_gaps '
                      'instead passed the repeat seed)',
    }
    results['meta'] = {
        'tag': name, 'dataset': DATASET, 'splits': SPLITS,
        'n_pool': N, 'n_events': int(y.sum()), 'n_subjects': int(n_subj),
        'n_test': int(len(test)), 'n_test_events': int(test_y.sum()),
        'comparators': ['padua_score', 'improve_score', 'caprini_score'],
        'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44), stratify vte_event, '
              'group subject_id; XGB random_state=42 fixed; fold bootstrap '
              '2000 for fold-mean AUC CI',
        'runtime_s': round(time.time() - t_start, 1),
    }
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[{name}] saved -> {out_path} ({time.time()-t_start:.0f}s)', flush=True)
    print_block(name, results, results['oof_inference'])
    return results


def write_summary(MAIN):
    """Compact cross-arm summary built from the per-arm JSONs that exist."""
    summary = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
               'arms': {}, 'feature_definitions': {
                   'strict_drop_15': STRICT_DROP,
                   'conservative_extra_2': CONSERVATIVE_EXTRA,
                   'surg_trauma_2': SURG_TRAUMA}}
    for arm in ['main_57', 'strict_42', 'conservative_40',
                'strict44_surgtrauma']:
        p = f'{O}/deploysafe_full_suite_{arm}{SUFFIX}.json'
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        inf = r['oof_inference']
        ap = {}
        ap_path = f'{O}/deploysafe_auprc_ci_deploysafe.json'
        if os.path.exists(ap_path):
            ap = json.load(open(ap_path)).get('arms', {}).get(arm, {})
        e = {'n_features': r['feature_sets']['n_features'],
             'features': r['feature_sets']['features'],
             'dropped_from_main_57': r['feature_sets']['dropped_from_main_57'],
             'n_pool': r['meta']['n_pool'], 'n_events': r['meta']['n_events'],
             'test_set': r['test_set'],
             'models': {}, 'scores': {},
             'comparators': {
                 k: {'oof_auc': v['oof_auc'], 'oof_auprc': v['oof_auprc']}
                 for k, v in r.items()
                 if k in [f'{s}_pool' for s in SCORE_COLS]}}
        for k in ['lr', 'xgb']:
            e['models'][k] = {
                'fold_mean_auc': r[k]['fold_mean_auc'],
                'fold_mean_auc_ci': r[k]['fold_mean_auc_ci'],
                'fold_mean_auprc': r[k]['fold_mean_auprc'],
                'fold_mean_auprc_ci': r[k].get('fold_mean_auprc_ci'),
                'oof_auc': r[k]['oof_auc'],
                'oof_auc_ci': inf['oof_auc_ci'][k]['ci'],
                'oof_auprc': r[k]['oof_auprc'],
                'oof_auprc_ci_uncapped': ap.get(k, {}).get('ci'),
                'oof_brier': r[k]['oof_brier'], 'oof_ece': r[k]['oof_ece'],
                'delta_auc': {f'{a}_vs_{b}': inf['delta_auc'][f'{a}_vs_{b}']
                              for a, b in DELTA_PAIRS if a == k},
                'nri': {f'vs_{b}_thr{t}': inf['nri'][f'{k}_vs_{b}_thr{t}']
                        for _, b in DELTA_PAIRS if _ == k
                        for t in NRI_THRESHOLDS},
                'idi': {f'vs_{b}': inf['idi'][f'{k}_vs_{b}']
                        for _, b in DELTA_PAIRS if _ == k},
            }
        e['xgb_vs_lr'] = {'delta_auc': inf['delta_auc']['xgb_vs_lr'],
                          'nri_thr0.005': inf['nri']['xgb_vs_lr_thr0.005'],
                          'nri_thr0.01': inf['nri']['xgb_vs_lr_thr0.01'],
                          'idi': inf['idi']['xgb_vs_lr']}
        summary['arms'][arm] = e
    p = f'{O}/deploysafe_full_suite_summary{SUFFIX}.json'
    with open(p, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'summary -> {p} ({len(summary["arms"])} arms)', flush=True)
    return summary


def run_subgroups(df, splits, MAIN, arms):
    """Surgical / medical subgroup AUC under the same protocol as 60b."""
    t_start = time.time()
    out_path = f'{O}/deploysafe_subgroups{SUFFIX}.json'
    if not os.path.exists(SVC_CACHE):
        raise SystemExit(f'missing service cache {SVC_CACHE}; run 60b first')
    svc = pd.read_parquet(SVC_CACHE)
    pool_ids = set(splits['train']) | set(splits['val'])
    pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
    pool = pool.merge(svc, on='hadm_id', how='left')
    print(f'subgroup pool N={len(pool)} events={int(pool["vte_event"].sum())} '
          f'coverage={pool["service_group"].notna().mean():.3f}', flush=True)

    out = {'n_pool': int(len(pool)),
           'service_definition': 'first curr_service in the MIMIC services '
                                 'table (same as scripts_era/60b)',
           'view': '5-fold StratifiedGroupKFold (seed 42) run INSIDE the '
                   'subgroup, group=subject_id; XGB random_state=42',
           'subgroups': {}}
    for arm_name, feats in arms.items():
        out['subgroups'][arm_name] = {'n_features': len(feats), 'rows': {}}
        for grp in ['surgical', 'medical']:
            sub = pool[pool.service_group == grp].reset_index(drop=True)
            X = sub[feats].values.astype(np.float32)
            y = sub['vte_event'].values.astype(int)
            g = sub['subject_id'].values
            oof = {'lr': np.zeros(len(y)), 'xgb': np.zeros(len(y))}
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                        random_state=SEED)
            for tr, te in sgkf.split(X, y, g):
                for kind in ['lr', 'xgb']:
                    p, _ = fit_predict(kind, X[tr], y[tr], X[te])
                    oof[kind][te] = p
            scores = {n: sub[c].values.astype(float)
                      for n, c in SCORE_COLS.items()}
            preds = {**oof, **scores}
            row = {'n': int(len(y)), 'events': int(y.sum()),
                   'event_rate': float(y.mean())}
            codes, uniq = pd.factorize(sub['subject_id'])
            n_subj = len(uniq)
            rng = np.random.RandomState(SEED)
            boot = {k: [] for k in preds}
            for _ in range(N_BOOT):
                counts = np.bincount(rng.randint(0, n_subj, n_subj),
                                     minlength=n_subj)
                rows = np.repeat(np.arange(len(y)), counts[codes])
                yb = y[rows]
                if yb.sum() < 2 or (1 - yb).sum() < 2:
                    continue
                for k, v in preds.items():
                    boot[k].append(fast_auc(yb, v[rows]))
            for k, v in preds.items():
                row[f'{k}_auc'] = float(roc_auc_score(y, v))
                row[f'{k}_auprc'] = float(average_precision_score(y, v))
            row['auc_ci'] = {k: [float(x) for x in np.percentile(v2,
                                                                 [2.5, 97.5])]
                             for k, v2 in boot.items()}
            out['subgroups'][arm_name]['rows'][grp] = row
            print(f"  [{arm_name}] {grp}: n={row['n']} events={row['events']} "
                  f"LR={row['lr_auc']:.4f} XGB={row['xgb_auc']:.4f} "
                  f"Caprini={row['caprini_auc']:.4f} "
                  f"Padua={row['padua_auc']:.4f} "
                  f"IMPROVE={row['improve_auc']:.4f}", flush=True)
    out['runtime_s'] = round(time.time() - t_start, 1)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'subgroups saved -> {out_path} ({time.time()-t_start:.0f}s)',
          flush=True)
    return out


def main():
    fs = json.load(open(f'{RES}/feature_sets_v2.json'))
    MAIN = fs['main']
    splits = json.load(open(SPLITS))
    df = pd.read_parquet(DATASET)
    n_all = len(df)
    df = df.reset_index(drop=True)
    n_excl = int(df['prior_vte_any'].fillna(0).eq(1).sum())
    for c in SCORE_COLS.values():
        if c not in df.columns:
            raise SystemExit(f'missing comparator column {c} in {DATASET}')
    print(f'cohort {n_all:,} rows (prior-VTE-positive {n_excl:,}); '
          f'comparators {list(SCORE_COLS.values())}', flush=True)

    arms = build_arms(MAIN)
    if ARM == 'subgroups':
        sub_arms = {'strict_42': arms['strict_42'], 'main_57': MAIN}
        run_subgroups(df, splits, MAIN, sub_arms)
        return
    if ARM == 'summary':
        write_summary(MAIN)
        return
    if ARM == 'all':
        for name in ['main_57', 'strict_42', 'conservative_40',
                     'strict44_surgtrauma']:
            run_arm(name, arms[name], df, splits, MAIN)
            write_summary(MAIN)
        return
    if ARM not in arms:
        raise SystemExit(f'unknown arm {ARM}; choose from {list(arms)} '
                         f'or all/summary/subgroups')
    run_arm(ARM, arms[ARM], df, splits, MAIN)
    write_summary(MAIN)


if __name__ == '__main__':
    main()
