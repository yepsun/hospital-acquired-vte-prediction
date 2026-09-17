#!/usr/bin/env python3
"""Script 55b: ED-triage vitals sensitivity — add MIMIC-IV-ED triage vitals
to the static vitals features and re-estimate XGB discrimination on the pool.

Motivation: continuous vitals in the main set come from ICU chartevents
(+ omr ward supplement), covering only ~15% of admissions. MIMIC-IV-ED
triage vitals (temperature, heart rate, respiratory rate, o2sat, SBP, DBP,
~95% complete) are available for admissions that entered through the ED
(~34%) and can fill the missing vitals, raising coverage to ~43%.

Design (mirrors scripts/42b step 3b):
  - feature set A: main (57) features only
  - feature set B: main + vitals (12 cols), where the vitals columns are
    merged from ICU chartevents/omr first, then ED triage fills remaining
    missing values
  - XGB native NaN, seed42 5-fold OOF on the train+val pool (both A and B
    share the same CV folds so the delta is paired)
  - paired patient-level bootstrap 2000 for delta-AUC CI
  - also report coverage (fraction of pool admissions with any vital) before
    and after the ED fill

Outputs: results_vte/ed_vitals_sensitivity_inclprior_excl24h.json
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RES = 'results_vte'
ED_DIR = 'mimic-iv-ed/2.2/ed'
OUT_JSON = f'{RES}/ajm/new/output_era/ed_vitals_sensitivity_inclprior_excl24h.json'
SEED = 42
N_BOOT = 2000

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

# physical plausibility ranges (same as scripts/33)
CLIP_RANGES = {
    'heart_rate': (20, 300),
    'spo2': (50, 100),
    'sbp': (20, 300),
    'dbp': (10, 200),
    'resp_rate': (2, 80),
    'temperature': (25, 45),  # Celsius after conversion
}

# ED triage column -> main vitals column name
ED2VITAL = {
    'heartrate': 'heart_rate',
    'sbp': 'sbp',
    'dbp': 'dbp',
    'resprate': 'resp_rate',
    'o2sat': 'spo2',
    'temperature': 'temperature',
}
VITAL_COLS = [f'vital_{k}_{s}' for k in
              ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
              for s in ('mean', 'worst')]


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof_xgb(X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in sgkf.split(X, y, groups):
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def boot_paired_delta(yb, p_ref, p_new):
    """paired patient-level bootstrap 2000 for delta AUC [CI]."""
    rng = np.random.RandomState(SEED)
    n = len(yb)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if yb[idx].sum() < 2 or (n - yb[idx].sum()) < 2:
            continue
        deltas.append(fast_auc(yb[idx], p_new[idx]) - fast_auc(yb[idx], p_ref[idx]))
    dci = np.percentile(deltas, [2.5, 97.5])
    return {'delta_auc': float(np.mean(deltas)),
            'delta_ci': [float(dci[0]), float(dci[1])]}


# ---------------- build ED triage vitals (per hadm_id) ----------------
t0 = time.time()
ed = pd.read_csv(f'{ED_DIR}/edstays.csv.gz', usecols=['stay_id', 'hadm_id',
                                                      'intime'])
ed = ed[ed['hadm_id'].notna()].copy()
ed['hadm_id'] = ed['hadm_id'].astype(int)
triage = pd.read_csv(f'{ED_DIR}/triage.csv.gz')
t = ed.merge(triage, on='stay_id')

# keep earliest ED stay per admission (first triage in time order)
t['intime'] = pd.to_datetime(t['intime'])
t = t.sort_values('intime').groupby('hadm_id').first().reset_index()

# temperature: F -> C when > 50 (same convention as scripts/33)
t['temperature'] = np.where(t['temperature'] > 50,
                            (t['temperature'] - 32) * 5 / 9,
                            t['temperature'])
# physical-range clipping
clip_stats = {}
for ed_col, key in ED2VITAL.items():
    lo, hi = CLIP_RANGES[key]
    bad = t[ed_col].notna() & ((t[ed_col] < lo) | (t[ed_col] > hi))
    clip_stats[key] = int(bad.sum())
    t.loc[bad, ed_col] = np.nan
    print(f'clip ed {ed_col}: {clip_stats[key]:,} out of [{lo},{hi}] -> NaN',
          flush=True)

# ED triage provides a single snapshot -> set both mean and worst to the value
ed_feat = t[['hadm_id'] + list(ED2VITAL)].rename(columns=ED2VITAL)
for k in list(ED2VITAL.values()):
    ed_feat[f'vital_{k}_mean'] = ed_feat[k]
    ed_feat[f'vital_{k}_worst'] = ed_feat[k]
ed_feat = ed_feat.drop(columns=list(ED2VITAL.values())).set_index('hadm_id')
print(f'ED triage vitals for {len(ed_feat):,} admissions '
      f'({time.time() - t0:.0f}s)', flush=True)

# ---------------- merge into pool, fill missing vitals ----------------
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
FEATS = fs['main']

df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
df = df.replace([np.inf, -np.inf], np.nan)
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
N = len(y)
print(f'pool N={N} events={y.sum()} ({y.mean() * 100:.3f}%)', flush=True)

# coverage before ED fill
cov_before = {c: float(pool[c].notna().mean()) for c in VITAL_COLS}

# fill missing vitals from ED triage (numpy to avoid index alignment)
ed_map = ed_feat.reindex(pool['hadm_id'])[VITAL_COLS].to_numpy(dtype=float)
filled = pool.copy()
filled[VITAL_COLS] = np.where(pd.isna(filled[VITAL_COLS].to_numpy(dtype=float)),
                              ed_map, filled[VITAL_COLS].to_numpy(dtype=float))
cov_after = {c: float(filled[c].notna().mean()) for c in VITAL_COLS}

any_v_before = float(pool[VITAL_COLS].notna().any(axis=1).mean())
any_v_after = float(filled[VITAL_COLS].notna().any(axis=1).mean())
print(f'any-vital coverage: {any_v_before:.3f} -> {any_v_after:.3f} '
      f'(+{any_v_after - any_v_before:.3f})', flush=True)

XA = pool[FEATS].values.astype(np.float32)
XB = filled[FEATS + fs['vitals_icu']].values.astype(np.float32)

# ---------------- XGB OOF, paired delta ----------------
t0 = time.time()
oof_a = oof_xgb(XA, y, groups)
print(f'  main OOF AUC={roc_auc_score(y, oof_a):.4f} ({time.time() - t0:.0f}s)',
      flush=True)
t0 = time.time()
oof_b = oof_xgb(XB, y, groups)
print(f'  +ED-vitals OOF AUC={roc_auc_score(y, oof_b):.4f} '
      f'({time.time() - t0:.0f}s)', flush=True)
delta = boot_paired_delta(y, oof_a, oof_b)

auc_a = float(roc_auc_score(y, oof_a))
auc_b = float(roc_auc_score(y, oof_b))
out = {
    'protocol': 'v2: static XGB seed42 5-fold OOF (same folds, paired) on the '
                'train+val pool; main(57) vs main+vitals(12) where vitals are '
                'ICU chartevents/omr first and MIMIC-IV-ED triage fills '
                'missing (single snapshot -> mean=worst); patient-level '
                'bootstrap 2000 for delta-AUC CI',
    'n': N,
    'events': int(y.sum()),
    'event_rate': float(y.mean()),
    'xgb_main_oof_auc': auc_a,
    'xgb_main_ed_vitals_oof_auc': auc_b,
    'vitals_delta_auc': delta['delta_auc'],
    'vitals_delta_ci': delta['delta_ci'],
    'any_vital_coverage_before': any_v_before,
    'any_vital_coverage_after': any_v_after,
    'per_vital_coverage_before': cov_before,
    'per_vital_coverage_after': cov_after,
    'n_ed_admissions': int(len(ed_feat)),
    'clip_stats': clip_stats,
    'note': 'ED triage vitals fill missing vitals for admissions that entered '
            'through the ED; elective/direct admissions retain missingness',
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=2)
print('Saved ->', OUT_JSON, flush=True)
print('checks:', json.dumps({k: v for k, v in out.items()
                             if 'delta' in k or 'coverage' in k}, indent=1),
      flush=True)
