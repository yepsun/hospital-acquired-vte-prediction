#!/usr/bin/env python3
"""
Script 99b (external arm of 99): trivial-clinical-baseline comparison in the
external high-risk cohort, added in response to review.

Rules requiring no model (Padua >= 4, IMPROVE >= 4, ICU within first 24 h,
active cancer) are compared with the frozen admission-safe XGBoost model
(strict_42 freeze) restricted to each rule's flag rate (strict-rank rule,
same convention as Table S10 and script 99), on the fully enumerated
high-risk stratum (n = 29,144; 559 events).

Fidelity: the frozen-model prediction path mirrors
run_external_validation_deploysafe_ipe.py exactly (same feature_order.json,
same inf/NaN and clip handling, native XGBoost missing values); the script
asserts the reproduced 42-feature AUC matches Table 4 (0.706) before writing.

Requires the local external export (not redistributable); outputs
trivial_baselines_external_highrisk.json next to the other external outputs.
"""
import json, math, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.environ.get(
    'EXTVAL_BASE', '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval')
# Table 4 uses the era-freeze models (trained on the 2008-2019 development pool)
FREEZE = f'{BASE}/frozen_models_era_deploysafe/strict_42'
FREEZE57 = f'{BASE}/frozen_models_era_deploysafe/main_57'
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    'results_vte/ajm/new/output_era/trivial_baselines_external_highrisk.json'

import joblib
from xgboost import XGBClassifier  # noqa: F401  (loaded via joblib)

co = pd.read_parquet(f'{BASE}/cohort_ipe.parquet',
                     columns=['vis_id', 'risk_group', 'vte_outcome',
                              'prior_vte_any', 'prior_vte_this_adm'])
fe = pd.read_parquet(f'{BASE}/features.parquet').drop(
    columns=['prior_vte_any', 'vte_outcome', 'risk_group'], errors='ignore')
sc = pd.read_parquet(f'{BASE}/scores.parquet')
df = co.merge(fe, on='vis_id', how='left', validate='1:1') \
       .merge(sc, on='vis_id', how='left', validate='1:1')
df['prior_vte'] = df['prior_vte_this_adm']
df = df[df['risk_group'] == 'high'].copy().reset_index(drop=True)
y = df['vte_outcome'].astype(int).values
print(f"high-risk stratum: {len(df)} rows, {int(y.sum())} events")


def frozen_auc(d, feats_file, lr=True):
    feats = json.load(open(f'{feats_file}/feature_order.json'))['features']
    X = d[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X = X.clip(-1e30, 1e30)
    xgb = joblib.load(f'{feats_file}/final_xgb.joblib')
    return xgb.predict_proba(X)[:, 1]


p42 = frozen_auc(df, FREEZE)
auc42 = roc_auc_score(y, p42)
p57 = frozen_auc(df, FREEZE57)
auc57 = roc_auc_score(y, p57)
print(f"reproduced external AUC (era freeze): 42-feat {auc42:.4f} (Table 4: 0.706), "
      f"57-feat {auc57:.4f} (Table 4: 0.678)")
assert abs(auc42 - 0.706) < 3e-3, '42-feature AUC does not reproduce Table 4'
assert abs(auc57 - 0.678) < 3e-3, '57-feature AUC does not reproduce Table 4'


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def opchar(flag):
    tp = int(((flag == 1) & (y == 1)).sum())
    flagged = int(flag.sum())
    sens = tp / int(y.sum())
    lo, hi_ = wilson(tp, int(y.sum()))
    return {'flag_rate': float(flagged / n), 'flagged_n': flagged,
            'sensitivity': sens, 'sensitivity_ci95': [lo, hi_],
            'ppv': (tp / flagged) if flagged else float('nan'), 'tp': tp}


n = len(df)
RULES = {
    'padua_ge4': (df['padua_score'] >= 4).fillna(False).astype(int).values,
    'improve_ge4': (df['improve_score'] >= 4).fillna(False).astype(int).values,
    'icu_first24h': df['first_careunit_icu'].fillna(0).astype(int).values,
    'active_cancer': df['cancer_active'].fillna(0).astype(int).values,
}


def topk(scores, rate):
    k = max(1, int(round(rate * n)))
    thr = np.sort(scores)[::-1][k - 1]
    return (scores >= thr).astype(int)


rows = []
for name, flag in RULES.items():
    r = opchar(flag)
    r.update({'baseline': name, 'type': 'clinical rule'})
    rows.append(r)
    for label, scores in (('xgb42', p42), ('xgb57', p57)):
        m = opchar(topk(scores, r['flag_rate']))
        m.update({'baseline': f'{label} @ top {r["flag_rate"]*100:.1f}%',
                  'type': f'model matched to {name}'})
        rows.append(m)

out = {
    'method': ('external fully enumerated high-risk stratum (29,144 admissions, '
               '559 events); frozen admission-safe XGBoost (strict_42) and '
               '57-feature XGBoost applied as in '
               'run_external_validation_deploysafe_ipe.py; model rows use the '
               'strict-rank rule at each rule\'s flag rate; Wilson 95% CIs.'),
    'n': n, 'n_events': int(y.sum()),
    'reproduced_auc': {'xgb42': auc42, 'xgb57': auc57},
    'rows': rows,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=1)
print(f"\n{'baseline':<34} {'flag%':>6} {'sens':>6} {'ppv':>7}")
for r in rows:
    print(f"{r['baseline']:<34} {r['flag_rate']*100:6.1f} {r['sensitivity']:6.3f} {r['ppv']*100:6.2f}%")
print('saved', OUT)
