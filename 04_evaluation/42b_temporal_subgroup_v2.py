#!/usr/bin/env python3
"""
Script 42 (VTE M4 / Task 5): temporal validation, subgroups, sensitivity.

  Step 1 temporal: train LR/XGB (fixed params, same as scripts/40) on
    train_pool_temporal (2008-2016), evaluate on test_temporal (2017-2022):
    AUC / AUPRC / Brier / ECE. Compare with random-split OOF AUC
    (static_model_results.json); drop < 0.03 = robust.
  Step 2 subgroups: test-set predictions (models retrained on train+val pool),
    stratified AUC with bootstrap CI (2000): ICU vs non-ICU
    (first_careunit_icu), surgery_flag 1/0, cancer_active 1/0, rx_heparin 1/0.
  Step 3 sensitivity: (a) main + 19 high-missing columns + missing indicators,
    XGB native NaN, seed42 5-fold OOF on pool, vs main OOF AUC;
    (b) ICU subpopulation (first_careunit_icu=1): XGB with main vs
    main+vitals_icu, seed42 5-fold OOF on ICU pool subset, delta AUC [CI]
    (bootstrap 2000 on OOF).

Checks: temporal AUC drop < 0.03; no subgroup AUC < 0.55; one-line sensitivity
conclusion.

Outputs: results_vte/temporal_subgroup.json
"""
import json, time, warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEED = 42
N_BOOT = 2000

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data/primary_inclprior_excl24h_splits.json'))
smr = json.load(open(f'{RES}/ajm/new/output/sensitivity_inclprior_excl24h_result.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data/primary_inclprior_excl24h_model_dataset.parquet')
df = df.replace([np.inf, -np.inf], np.nan)


def fit_predict(kind, X_tr, y_tr, X_te):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1]
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def oof_xgb(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in sgkf.split(X, y, groups):
        oof[te] = fit_predict('xgb', X[tr], y[tr], X[te])
    return oof


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(yb, p):
    pt, pp = calibration_curve(yb, p, n_bins=10)
    return float(np.mean(np.abs(pt - pp)))


def boot_auc_ci(yb, p, delta_ref=None):
    """bootstrap AUC CI; if delta_ref given, also return dAUC CI (paired)."""
    rng = np.random.RandomState(SEED)
    n = len(yb)
    aucs, deltas = [], []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if yb[idx].sum() < 2 or (n - yb[idx].sum()) < 2:
            continue
        aucs.append(fast_auc(yb[idx], p[idx]))
        if delta_ref is not None:
            deltas.append(aucs[-1] - fast_auc(yb[idx], delta_ref[idx]))
    ci = np.percentile(aucs, [2.5, 97.5])
    out = {'auc': float(np.mean(aucs)), 'ci': [float(ci[0]), float(ci[1])]}
    if delta_ref is not None:
        dci = np.percentile(deltas, [2.5, 97.5])
        out['delta_auc'] = float(np.mean(deltas))
        out['delta_ci'] = [float(dci[0]), float(dci[1])]
    return out


results = {}

# ── Step 1: temporal validation ──
print('temporal validation ...', flush=True)
tr_ids, te_ids = set(splits['train_pool_temporal']), set(splits['test_temporal'])
tr = df[df['hadm_id'].isin(tr_ids)].reset_index(drop=True)
te = df[df['hadm_id'].isin(te_ids)].reset_index(drop=True)
Xtr, ytr = tr[FEATS].values.astype(np.float32), tr['vte_event'].values.astype(int)
Xte, yte = te[FEATS].values.astype(np.float32), te['vte_event'].values.astype(int)
print(f'  train_pool N={len(ytr)} events={ytr.sum()} | test_temporal N={len(yte)} '
      f'events={yte.sum()}', flush=True)

results['temporal'] = {}
for k in ['lr', 'xgb']:
    t0 = time.time()
    p = fit_predict(k, Xtr, ytr, Xte)
    auc = roc_auc_score(yte, p)
    ref = smr[k]['oof_auc']
    results['temporal'][k] = {
        'auc': float(auc), 'auprc': float(average_precision_score(yte, p)),
        'brier': float(brier_score_loss(yte, p)), 'ece': ece(yte, p),
        'random_split_oof_auc': ref, 'auc_drop': float(ref - auc)}
    print(f"  {k.upper()}: temporal AUC={auc:.4f} AUPRC="
          f"{average_precision_score(yte, p):.4f} Brier={brier_score_loss(yte, p):.5f} "
          f"ECE={ece(yte, p):.4f} (random OOF {ref:.4f}, drop {ref - auc:+.4f}, "
          f"{time.time() - t0:.0f}s)", flush=True)

# ── Step 2: subgroups on the random test set ──
print('subgroups ...', flush=True)
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[df['hadm_id'].isin(set(splits['test']))].reset_index(drop=True)
Xp, yp = pool[FEATS].values.astype(np.float32), pool['vte_event'].values.astype(int)
Xt, yt = test[FEATS].values.astype(np.float32), test['vte_event'].values.astype(int)
preds = {k: fit_predict(k, Xp, yp, Xt) for k in ['lr', 'xgb']}

SUBGROUPS = {'icu': 'first_careunit_icu', 'surgery': 'surgery_flag',
             'cancer': 'cancer_active', 'rx_heparin': 'rx_heparin'}
results['subgroups'] = {}
for name, col in SUBGROUPS.items():
    for val in [1, 0]:
        m = test[col].values == val
        entry = {'n': int(m.sum()), 'events': int(yt[m].sum()),
                 'event_rate': float(yt[m].mean())}
        for k in ['lr', 'xgb']:
            entry[k] = boot_auc_ci(yt[m], preds[k][m])
        results['subgroups'][f'{name}_{val}'] = entry
        print(f"  {name}={val}: n={entry['n']} events={entry['events']} "
              f"XGB AUC={entry['xgb']['auc']:.4f} "
              f"[{entry['xgb']['ci'][0]:.4f}-{entry['xgb']['ci'][1]:.4f}]", flush=True)

# ── Step 3a: sensitivity — high-missing columns + indicators, XGB ──
print('sensitivity (a): +19 high-missing cols + indicators, XGB ...', flush=True)
t0 = time.time()
hi_miss = ['lab_neutrophils_median', 'lab_lymphocytes_median',
           'lab_ddimer_median', 'lab_fibrinogen_median', 'lab_albumin_median',
           'lab_crp_median', 'derived_nlr'] + fs['vitals_icu']
assert len(hi_miss) == 19 and all(c in df.columns for c in hi_miss)
Xa = pool[FEATS + hi_miss].values.astype(np.float32)
Xa = np.hstack([Xa, np.isnan(pool[hi_miss].values).astype(np.float32)])
oof_a = oof_xgb(Xa, yp, pool['subject_id'].values)
auc_a = roc_auc_score(yp, oof_a)
ref_xgb = smr['xgb']['oof_auc']
results['sensitivity_high_missing'] = {
    'features': f'main({len(FEATS)}) + 19 high-missing + 19 missing indicators '
                f'= {Xa.shape[1]}',
    'oof_auc': float(auc_a), 'main_oof_auc': ref_xgb,
    'delta': float(auc_a - ref_xgb),
    'note': 'XGB native NaN, seed42 5-fold OOF on train+val pool'}
print(f'  AUC {auc_a:.4f} vs main {ref_xgb:.4f} (d={auc_a - ref_xgb:+.4f}, '
      f'{time.time() - t0:.0f}s)', flush=True)

# ── Step 3b: sensitivity — ICU subpopulation, + vitals_icu ──
print('sensitivity (b): ICU subpopulation, + vitals_icu ...', flush=True)
t0 = time.time()
icu = pool[pool['first_careunit_icu'] == 1].reset_index(drop=True)
y_icu = icu['vte_event'].values.astype(int)
g_icu = icu['subject_id'].values
Xb_main = icu[FEATS].values.astype(np.float32)
Xb_vit = icu[FEATS + fs['vitals_icu']].values.astype(np.float32)
oof_main = oof_xgb(Xb_main, y_icu, g_icu)
oof_vit = oof_xgb(Xb_vit, y_icu, g_icu)
b = boot_auc_ci(y_icu, oof_vit, delta_ref=oof_main)
results['sensitivity_icu_vitals'] = {
    'n': len(y_icu), 'events': int(y_icu.sum()),
    'xgb_main_oof_auc': float(roc_auc_score(y_icu, oof_main)),
    'xgb_main_vitals_oof_auc': float(roc_auc_score(y_icu, oof_vit)),
    'vitals_delta_auc': b['delta_auc'], 'vitals_delta_ci': b['delta_ci'],
    'note': 'ICU subpopulation (first_careunit_icu=1) of train+val pool; XGB '
            'native NaN, seed42 5-fold OOF; paired bootstrap 2000'}
print(f"  ICU n={len(y_icu)} events={y_icu.sum()}: main "
      f"{roc_auc_score(y_icu, oof_main):.4f} vs +vitals "
      f"{roc_auc_score(y_icu, oof_vit):.4f} (d={b['delta_auc']:+.4f} "
      f"[{b['delta_ci'][0]:+.4f},{b['delta_ci'][1]:+.4f}], {time.time() - t0:.0f}s)",
      flush=True)

# ── Step 4: checks ──
sg_xgb = {k: v['xgb']['auc'] for k, v in results['subgroups'].items()}
checks = {
    'temporal_drop_lt_0.03': {k: results['temporal'][k]['auc_drop'] < 0.03
                              for k in ['lr', 'xgb']},
    'no_subgroup_below_0.55': all(v >= 0.55 for v in sg_xgb.values()),
    'min_subgroup_auc': float(min(sg_xgb.values())),
    'sensitivity_conclusion': (
        f"加入 19 个高缺失列+缺失指示列后 XGB OOF AUC "
        f"{auc_a:.4f} vs 主结果 {ref_xgb:.4f}（Δ={auc_a - ref_xgb:+.4f}），"
        + ('不改变主结论' if abs(auc_a - ref_xgb) < 0.01 else '差异需关注')
        + f"；ICU 子人群加入 vitals_icu ΔAUC={b['delta_auc']:+.4f} "
          f"[{b['delta_ci'][0]:+.4f},{b['delta_ci'][1]:+.4f}]")}
results['checks'] = checks
print('checks:', json.dumps(checks, indent=1, ensure_ascii=False), flush=True)

results['meta'] = {'features_main_n': len(FEATS),
                   'protocol': 'fixed params (no tuning), same as scripts/40; '
                               'bootstrap 2000 for subgroup/sensitivity CIs'}
with open(f'{RES}/ajm/new/output/temporal_subgroup_inclprior_excl24h.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('Saved -> results_vte/temporal_subgroup.json', flush=True)
