#!/usr/bin/env python3
"""
Script 62: close review gaps on the inclprior_excl24h primary cohort.

A) Look-ahead sensitivity: static LR/XGB with the discharge-ICD block
   removed (strict drop -> 42 features; conservative drop -> 40 features),
   3 seeds x 5-fold CV fold-mean AUC + Padua on identical rows.
B) Clinical subgroup discrimination on seed-42 pooled out-of-fold
   predictions (one consistent view for all subgroups), patient-level
   cluster-bootstrap CIs.
C) Descriptive quantities quoted in the Discussion: median length of stay,
   median time to VTE diagnosis, heparin exposure, imaging-clean vs rest
   control Padua comparison.

Output: results_vte/ajm/new/output/review_gap_results_inclprior_excl24h.json
"""
import json, warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
MAIN = fs['main']
STRICT_DROP = ['cancer_active', 'heart_failure', 'copd', 'chronic_liver',
               'ckd', 'diabetes', 'stroke', 'mi', 'obesity_icd', 'varicose',
               'thrombophilia', 'infection_severe', 'rheumatologic',
               'surgery_flag', 'trauma_flag']
CONSERVATIVE_EXTRA = ['prior_vte', 'prior_vte_any']

splits = json.load(open(f'{RES}/ajm/new/data/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{RES}/ajm/new/data/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
coh = df.reset_index(drop=True)

X_full = pool[MAIN].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values


def fit_predict(kind, X_tr, y_tr, X_te, seed):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def fast_auc(yb, pb):
    r = rankdata(pb)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ds(vals):
    vals = np.asarray(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(np.mean(vals)), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


out = {}

# ── A) look-ahead sensitivity ────────────────────────────────────────────
print('== A) look-ahead ==', flush=True)
sets = {'main_57': MAIN,
        'strict_42': [f for f in MAIN if f not in STRICT_DROP],
        'conservative_40': [f for f in MAIN
                            if f not in STRICT_DROP + CONSERVATIVE_EXTRA]}
padua_pool = np.nan_to_num(pool['padua_score'].values.astype(float),
                           nan=float(np.nanmedian(pool['padua_score'])))
out['lookahead'] = {'feature_sets': {k: len(v) for k, v in sets.items()},
                    'dropped_strict': STRICT_DROP,
                    'dropped_conservative_extra': CONSERVATIVE_EXTRA,
                    'results': {}}
for name, feats in sets.items():
    X = pool[feats].values.astype(np.float32)
    res = {'fold_mean': {}, 'oof_auc_seed42': {}}
    for kind in ['lr', 'xgb']:
        fms = []
        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=seed)
            aucs = []
            oof = np.zeros(len(y))
            for tr, te in sgkf.split(X, y, groups):
                p = fit_predict(kind, X[tr], y[tr], X[te], seed)
                aucs.append(roc_auc_score(y[te], p))
                if seed == 42:
                    oof[te] = p
            fms.append(float(np.mean(aucs)))
            if seed == 42:
                res['oof_auc_seed42'][kind] = float(roc_auc_score(y, oof))
                if name == 'main_57' and kind == 'lr':
                    globals()['OOF_MAIN_SEED42'] = oof.copy()
        res['fold_mean'][kind] = {
            'per_seed': fms, 'mean': float(np.mean(fms)),
            'range': [float(min(fms)), float(max(fms))]}
    res['fold_mean']['padua'] = float(roc_auc_score(y, padua_pool))
    res['delta_vs_padua_fold_mean'] = {
        k: round(res['fold_mean'][k]['mean'] - res['fold_mean']['padua'], 4)
        for k in ['lr', 'xgb']}
    out['lookahead']['results'][name] = res
    print(name, json.dumps(res['fold_mean'], indent=1)[:300], flush=True)

# ── B) clinical subgroups on seed-42 pooled OOF ──────────────────────────
print('== B) subgroups ==', flush=True)
subj_codes, _ = pd.factorize(groups)
n_subj = len(np.unique(subj_codes))


def boot_ci(mask, preds, rng):
    rows_pos = np.where(y[mask] == 1)[0]
    idx_pos = np.where(mask)[0][rows_pos]
    idx_neg = np.where(mask)[0][y[mask] == 0]
    pos_subj, neg_subj = subj_codes[idx_pos], subj_codes[idx_neg]
    aucs = []
    yy = y[mask]
    for b in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj),
                             minlength=n_subj)
        mult = counts[pos_subj]
        take = mult > 0
        if take.sum() < 2:
            continue
        bp = np.repeat(idx_pos[take], mult[take])
        bn = idx_neg[counts[neg_subj] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        rows = np.concatenate([bp, bn])
        aucs.append(fast_auc(y[rows], preds[rows]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [float(lo), float(hi)]


subgroups = {
    'ICU within 24 h': pool['first_careunit_icu'].values.astype(int) == 1,
    'No ICU': pool['first_careunit_icu'].values.astype(int) == 0,
    'Surgery': pool['surgery_flag'].values.astype(int) == 1,
    'No surgery': pool['surgery_flag'].values.astype(int) == 0,
    'Active cancer': pool['cancer_active'].values.astype(int) == 1,
    'No cancer': pool['cancer_active'].values.astype(int) == 0,
    'Heparin within 24 h': pool['rx_heparin'].values.astype(int) == 1,
    'No heparin': pool['rx_heparin'].values.astype(int) == 0,
}
preds = {'lr': OOF_MAIN_SEED42}
# xgb seed42 OOF on main features: recover from lookahead run? recompute once
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(y))
for tr, te in sgkf.split(X_full, y, groups):
    oof_xgb[te] = fit_predict('xgb', X_full[tr], y[tr], X_full[te], 42)
preds['xgb'] = oof_xgb

out['subgroups'] = {'view': 'seed-42 5-fold pooled out-of-fold predictions '
                            '(train+val pool, n=%d, events=%d)' % (len(y), int(y.sum())),
                    'rows': {}}
for name, mask in subgroups.items():
    row = {'n': int(mask.sum()), 'events': int(y[mask].sum()),
           'event_rate_pct': round(100 * y[mask].mean(), 2)}
    for kind, pr in preds.items():
        auc = fast_auc(y[mask], pr[mask]) if y[mask].sum() > 0 else None
        rng = np.random.RandomState(42 + hash(name) % 1000)
        ci = boot_ci(mask, pr, rng)
        row[kind] = {'auc': auc, 'ci': ci}
    out['subgroups']['rows'][name] = row
    print(name, json.dumps(row)[:200], flush=True)

# ── C) discussion descriptives ───────────────────────────────────────────
print('== C) descriptives ==', flush=True)
adm = pd.to_datetime(coh['admittime_x'])
dis = pd.to_datetime(coh['dischtime_x'])
los_h = (dis - adm).dt.total_seconds() / 3600
cases = coh['vte_event'].values.astype(int) == 1
dx = pd.to_datetime(coh.loc[cases, 'diagnosis_time'])
adm_c = pd.to_datetime(coh.loc[cases, 'admittime_x'])
ttd_h = (dx - adm_c).dt.total_seconds() / 3600

ctrl = coh['vte_event'].values.astype(int) == 0
strict_ids = set(pd.read_parquet(
    f'{RES}/ajm/new/output/strict_control_ids_inclprior_excl24h.parquet')['hadm_id'])
is_strict_ctrl = ctrl & coh['hadm_id'].isin(strict_ids).values
is_rest_ctrl = ctrl & ~coh['hadm_id'].isin(strict_ids).values
ps = np.nan_to_num(coh['padua_score'].values.astype(float),
                   nan=float(np.nanmedian(coh['padua_score'])))
out['descriptives'] = {
    'median_los_days_all': float(los_h.median() / 24),
    'median_los_days_controls': float(los_h[ctrl].median() / 24),
    'median_time_to_vte_dx_days': float(ttd_h.median() / 24),
    'heparin_pct_all': round(100 * coh['rx_heparin'].mean(), 1),
    'heparin_pct_cases': round(100 * coh.loc[cases, 'rx_heparin'].mean(), 1),
    'imaging_clean_vs_rest_controls': {
        'n_strict': int(is_strict_ctrl.sum()), 'n_rest': int(is_rest_ctrl.sum()),
        'padua_mean_strict': round(float(ps[is_strict_ctrl].mean()), 2),
        'padua_mean_rest': round(float(ps[is_rest_ctrl].mean()), 2),
        'padua_highrisk_pct_strict': round(
            100 * (ps[is_strict_ctrl] >= 4).mean(), 1),
        'padua_highrisk_pct_rest': round(
            100 * (ps[is_rest_ctrl] >= 4).mean(), 1)},
}
print(json.dumps(out['descriptives'], indent=1), flush=True)

with open(f'{RES}/ajm/new/output/review_gap_results_inclprior_excl24h.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved -> results_vte/ajm/new/output/review_gap_results_inclprior_excl24h.json',
      flush=True)
