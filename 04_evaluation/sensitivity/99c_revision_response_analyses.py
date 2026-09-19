#!/usr/bin/env python3
"""
Script 99c: reviewer-response analyses for the EJIM revision (round 2).

All analyses run on the development pool of the primary arm
(313,064 admissions / 554 events) with the published primary protocol
(seed-42 5-fold StratifiedGroupKFold by patient; XGBoost 300x depth-3 lr .05
with native missing-value routing; LR L2 C=1.0 with per-fold training-median
imputation). Protocol fidelity is asserted against the shipped Table 2 /
Table S3c values before anything is written.

Components
  A. OOF predictions for the 42-feature admission-safe XGBoost (assert
     AUC 0.7864) and for a 23-feature arm that additionally removes all 19
     utilization / utilization-proxy features (15 laboratory measurement
     counts, 3 procedure counts, ICU hours in the first 24 h) -- reviewer
     point M7.
  B. Bedside-computable Padua / IMPROVE variants scored only from items
     knowable at admission (no discharge-coded index-admission ICD items)
     -- reviewer point M8. Declared lower bounds, like the computed Caprini.
  C. Truncation series: model-vs-Padua delta AUC after progressively
     restricting the pool by the same admission-screening rules that define
     the external high-risk stratum (Padua >= 4; Padua >= 4 or Caprini >= 5;
     top-25% and top-15% by Padua score) -- reviewer point M3.
  D. Sex-stratified discrimination (SAGER): xgb42 and Padua AUCs with
     patient-level cluster-bootstrap CIs in the male and female strata.
  E. Strict imaging-clean control pool re-evaluated under the *primary*
     protocol (XGBoost native NaN, 3x5 folds, fixed XGB seed) so that
     Table S3b follows one convention with Table 2 -- reviewer point M10.
  F. Shrinkage-based sample-size numbers (van Smeden / Riley): Cox-Snell
     R-squared of the primary model, minimum N for shrinkage >= 0.9 -- M11.

Outputs (NEW files only):
  output_era/revision_response_analyses.json
  output_era/revision_oof_rows.parquet
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte'
SEED = 42
SEEDS = [42, 43, 44]
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1)

STRICT_DROP = ['cancer_active', 'heart_failure', 'copd', 'chronic_liver',
               'ckd', 'diabetes', 'stroke', 'mi', 'obesity_icd', 'varicose',
               'thrombophilia', 'infection_severe', 'rheumatologic',
               'surgery_flag', 'trauma_flag']
UTIL_FEATURES = ['proc_cvc', 'proc_mechvent', 'proc_transfusion',
                 'icu_los_first24h']

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
MAIN = fs['main']
SAFE42 = [f for f in MAIN if f not in STRICT_DROP]
UTIL = [f for f in SAFE42 if f.endswith('_count')] + UTIL_FEATURES
SAFE23 = [f for f in SAFE42 if f not in UTIL]
print(f'features: main {len(MAIN)}, safe42 {len(SAFE42)}, '
      f'utilization removed {len(UTIL)}, safe23 {len(SAFE23)}', flush=True)
assert len(SAFE42) == 42 and len(SAFE23) == 23

splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
cap = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset_caprini.parquet')
df = df.merge(cap[['hadm_id', 'caprini_score', 'caprini_high']], on='hadm_id',
              how='left', validate='1:1')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
N = len(pool)
n_events = int(y.sum())
print(f'dev pool {N} rows / {n_events} events', flush=True)


def score_col(col):
    s = pool[col].values.astype(float)
    return np.nan_to_num(s, nan=float(np.nanmedian(s)))


padua = score_col('padua_score')
improve = score_col('improve_score')
caprini = score_col('caprini_score')

# B. bedside-computable score variants ---------------------------------------
bmi = pool['bmi'].fillna(0).values
bedside_padua = (3 * pool['prior_vte_any'].fillna(0).values
                 + 3 * pool['first_careunit_icu'].fillna(0).values
                 + 2 * (pool['age'].values >= 70)
                 + 1 * (bmi >= 30)
                 + 1 * pool['rx_hormone'].fillna(0).values).astype(float)
bedside_improve = (3 * pool['prior_vte_any'].fillna(0).values
                   + 3 * pool['first_careunit_icu'].fillna(0).values
                   + 1 * (pool['age'].values > 60)).astype(float)
print(f"screen flag rates: padua>=4 {(padua >= 4).mean():.3f}  "
      f"caprini>=5 {(caprini >= 5).mean():.3f}  "
      f"union {((pool['padua_high'].fillna(0) | pool['caprini_high'].fillna(0)) > 0).mean():.3f}",
      flush=True)


def xgb_oof(feats):
    X = pool[feats].values.astype(np.float32)
    oof = np.zeros(N, dtype=np.float64)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        m = XGBClassifier(**XGB_PARAMS, random_state=SEED)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def lr_oof(feats):
    X = pool[feats].values.astype(np.float32)
    oof = np.zeros(N, dtype=np.float64)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, y, groups):
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(imp.transform(X[te]))[:, 1]
    return oof


print('OOF: xgb42 ...', flush=True)
oof42 = xgb_oof(SAFE42)
auc42 = roc_auc_score(y, oof42)
print(f'  xgb42 OOF AUC {auc42:.4f} (shipped 0.7864)', flush=True)
assert abs(auc42 - 0.7864) < 1e-3

print('OOF: xgb23 (utilization removed) ...', flush=True)
oof23 = xgb_oof(SAFE23)
auc23 = roc_auc_score(y, oof23)
print(f'  xgb23 OOF AUC {auc23:.4f}', flush=True)


def fast_auc(yb, pb):
    r = rankdata(pb)
    n1 = int(yb.sum())
    n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cluster_boot(rows, preds, deltas_pairs, n_boot=N_BOOT, seed=SEED):
    """Patient-level cluster bootstrap on `rows`; returns dict of delta CIs.

    `rows` are positions into the full-length arrays in `preds`; predictions
    are sliced to the subset FIRST so that within-loop positions index the
    subset arrays (a full-length `preds` indexed with subset positions would
    silently score the wrong admissions on truncated subgroups).
    """
    deltas_pairs = dict(deltas_pairs)
    P = {k: np.asarray(v)[rows] for k, v in preds.items()}
    yb0 = y[rows]
    subj_codes, uniq = pd.factorize(groups[rows])
    n_subj = len(uniq)
    ev = np.where(yb0 == 1)[0]
    neg = np.where(yb0 == 0)[0]
    ev_s, neg_s = subj_codes[ev], subj_codes[neg]
    rng = np.random.RandomState(seed)
    acc = {k: [] for k in deltas_pairs}
    for _ in range(n_boot):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        mult = counts[ev_s]
        take = mult > 0
        if take.sum() < 2:
            continue
        be = np.repeat(ev[take], mult[take])
        bn = neg[counts[neg_s] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        idx = np.concatenate([be, bn])
        yy = yb0[idx]
        a = {k: fast_auc(yy, v[idx]) for k, v in P.items()}
        for name, (i, j) in deltas_pairs.items():
            acc[name].append(a[i] - a[j])
    out = {}
    for k, v in acc.items():
        v = np.asarray(v)
        out[k] = {'estimate': float(np.mean(v)),
                  'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  'p_two_sided': float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
    return out


ALL = {'xgb42': oof42, 'xgb23': oof23, 'padua': padua, 'improve': improve,
       'bedside_padua': bedside_padua, 'bedside_improve': bedside_improve,
       'caprini': caprini}
aucs_full = {k: float(roc_auc_score(y, v)) for k, v in ALL.items()}
print('full-pool AUCs:', {k: round(v, 4) for k, v in aucs_full.items()}, flush=True)

print('bootstrap: full pool deltas', flush=True)
boot_full = cluster_boot(np.arange(N), ALL, [
    ('xgb42_vs_padua', ('xgb42', 'padua')),
    ('xgb23_vs_padua', ('xgb23', 'padua')),
    ('xgb42_vs_bedside_padua', ('xgb42', 'bedside_padua')),
    ('xgb42_vs_xgb23', ('xgb42', 'xgb23')),
    ('lr_placeholder', ('padua', 'improve')),  # replaced below
])

# lr42 OOF for the utilization arm (parallel to the 57-feature LR in Table 2)
print('OOF: lr42 ...', flush=True)
ooflr42 = lr_oof(SAFE42)
ooflr23 = lr_oof(SAFE23)
aucs_full['lr42'] = float(roc_auc_score(y, ooflr42))
aucs_full['lr23'] = float(roc_auc_score(y, ooflr23))
boot_lr = cluster_boot(np.arange(N),
                       {'lr42': ooflr42, 'lr23': ooflr23, 'padua': padua},
                       [('lr42_vs_padua', ('lr42', 'padua')),
                        ('lr23_vs_padua', ('lr23', 'padua')),
                        ('lr42_vs_lr23', ('lr42', 'lr23'))])

# C. truncation series --------------------------------------------------------
def trunc_stats(mask, label):
    rows = np.where(mask)[0]
    yy = y[rows]
    res = {'label': label, 'n': int(mask.sum()), 'events': int(yy.sum()),
           'event_rate': float(yy.mean()),
           'auc': {k: float(roc_auc_score(yy, ALL[k][rows]))
                   for k in ('xgb42', 'padua', 'bedside_padua', 'improve')}}
    b = cluster_boot(rows, {k: ALL[k] for k in ('xgb42', 'padua')},
                     [('xgb42_vs_padua', ('xgb42', 'padua'))])
    res['delta_xgb42_vs_padua'] = b['xgb42_vs_padua']
    print(f"  trunc [{label}] n={res['n']} ev={res['events']} "
          f"xgb42 {res['auc']['xgb42']:.4f} padua {res['auc']['padua']:.4f} "
          f"d {res['delta_xgb42_vs_padua']['estimate']:+.4f}", flush=True)
    return res


print('truncation series', flush=True)
padua_score_v = padua
truncations = [
    trunc_stats(np.ones(N, dtype=bool), 'none (full dev pool)'),
    trunc_stats(pool['padua_high'].fillna(0).values.astype(bool), 'Padua >= 4'),
    trunc_stats(((pool['padua_high'].fillna(0) | pool['caprini_high'].fillna(0)) > 0).values,
                'Padua >= 4 or Caprini >= 5 (external screen)'),
    trunc_stats(padua_score_v >= np.quantile(padua_score_v, 0.75), 'top 25% by Padua score'),
    trunc_stats(padua_score_v >= np.quantile(padua_score_v, 0.85), 'top 15% by Padua score'),
]

# D. sex strata ---------------------------------------------------------------
print('sex strata', flush=True)
sex = {}
for name, mask in [('male', pool['male'].values.astype(bool)),
                   ('female', ~pool['male'].values.astype(bool))]:
    rows = np.where(mask)[0]
    sex[name] = {'n': int(mask.sum()), 'events': int(y[mask].sum()),
                 'auc_xgb42': float(roc_auc_score(y[mask], oof42[mask])),
                 'padua_auc': float(roc_auc_score(y[mask], padua[mask]))}
    # per-arm CIs via bootstrap of the AUC itself
    yy = y[rows]
    subj_codes, uniq = pd.factorize(groups[rows])
    n_subj = len(uniq)
    ev = np.where(yy == 1)[0]
    neg = np.where(yy == 0)[0]
    ev_s, neg_s = subj_codes[ev], subj_codes[neg]
    rng = np.random.RandomState(SEED)
    acc = {'xgb42': [], 'padua': []}
    for _ in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        mult = counts[ev_s]
        take = mult > 0
        if take.sum() < 2:
            continue
        be = np.repeat(ev[take], mult[take])
        bn = neg[counts[neg_s] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        idx = np.concatenate([be, bn])
        acc['xgb42'].append(fast_auc(yy[idx], oof42[rows][idx]))
        acc['padua'].append(fast_auc(yy[idx], padua[rows][idx]))
    sex[name]['auc_xgb42_ci'] = [float(np.percentile(acc['xgb42'], 2.5)),
                                 float(np.percentile(acc['xgb42'], 97.5))]
    sex[name]['padua_auc_ci'] = [float(np.percentile(acc['padua'], 2.5)),
                                 float(np.percentile(acc['padua'], 97.5))]
    print(f"  {name}: n={sex[name]['n']} ev={sex[name]['events']} "
          f"xgb42 {sex[name]['auc_xgb42']:.4f} padua {sex[name]['padua_auc']:.4f}", flush=True)

# E. strict pool under the primary protocol -----------------------------------
print('strict pool (primary protocol)', flush=True)
strict_ids = set(pd.read_parquet(
    f'{RES}/ajm/new/output_era/strict_control_ids_inclprior_excl24h.parquet')['hadm_id'])
srows = np.where((y == 1) | (pd.Series(pool['hadm_id']).isin(strict_ids).values & (y == 0)))[0]
sub = pool.iloc[srows]
ys = y[srows]
print(f'  strict pool rows={len(srows)} events={ys.sum()}', flush=True)
Xs = sub[MAIN].values.astype(np.float32)
gs = sub['subject_id'].values
fold_aucs = {'lr': [], 'xgb': []}
for seed in SEEDS:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, te in cv.split(Xs, ys, gs):
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(Xs[tr]), ys[tr])
        fold_aucs['lr'].append(roc_auc_score(ys[te], m.predict_proba(imp.transform(Xs[te]))[:, 1]))
        mx = XGBClassifier(**XGB_PARAMS, random_state=SEED)
        mx.fit(Xs[tr], ys[tr])
        fold_aucs['xgb'].append(roc_auc_score(ys[te], mx.predict_proba(Xs[te])[:, 1]))
strict_primary = {
    'rows': int(len(srows)), 'events': int(ys.sum()), 'n_folds': len(fold_aucs['xgb']),
    'padua_fold_mean': float(np.mean([roc_auc_score(ys, padua[srows])] * len(fold_aucs['xgb']))),
    'padua_auc': float(roc_auc_score(ys, padua[srows])),
    'lr_fold_mean': float(np.mean(fold_aucs['lr'])),
    'xgb_fold_mean': float(np.mean(fold_aucs['xgb'])),
    'protocol': 'primary protocol: LR per-fold median imputation; XGBoost native missing routing, random_state fixed at 42; 3x5 StratifiedGroupKFold seeds 42/43/44',
}
print(f"  strict primary-protocol fold-mean: lr {strict_primary['lr_fold_mean']:.4f} "
      f"xgb {strict_primary['xgb_fold_mean']:.4f}", flush=True)

# F. sample size --------------------------------------------------------------
ll_null = -log_loss(y, np.full(N, y.mean()), labels=[0, 1])
ll_full = -log_loss(y, np.clip(oof42, 1e-12, 1 - 1e-12))
chi2 = 2 * N * (ll_null - ll_full)
r2cs = 1 - np.exp(-chi2 / N)
k_candidates = len(MAIN)
n_min_shrink = 10 * k_candidates / (-np.log(1 - r2cs))
samplesize = {
    'n_pool': N, 'n_events': n_events,
    'cox_snell_r2_oof_xgb42': float(r2cs),
    'lr_chi2_approx': float(chi2),
    'k_candidates': k_candidates,
    'epv_main': n_events / k_candidates,
    'epv_admission_safe': n_events / len(SAFE42),
    'min_n_for_shrinkage_0.9_57feat': float(n_min_shrink),
    'min_n_for_shrinkage_0.9_42feat': float(10 * len(SAFE42) / (-np.log(1 - r2cs))),
    'note': 'van Smeden/Riley shrinkage criterion: chi2_LR ~ -n*log(1-R2cs); '
            'shrinkage S = 1 - k/chi2 >= 0.9 requires n >= 10k / -log(1-R2cs). '
            'R2cs estimated from primary-model OOF log loss.',
}
print(f"  R2cs {r2cs:.5f} -> min N (57 feats, S>=0.9): {n_min_shrink:,.0f}", flush=True)

# save ------------------------------------------------------------------------
out = {
    'method': ('development pool of the primary arm (313,064 / 554); seed-42 '
               '5-fold StratifiedGroupKFold by patient; XGBoost native missing '
               'routing, fixed hyperparameters; OOF AUC asserted against Table S3c '
               '(0.7864) before writing; deltas from patient-level cluster '
               'bootstrap (2,000 replicates, negatives capped at 50,000)'),
    'n_pool': N, 'n_events': n_events,
    'reproduced_auc42': auc42,
    'feature_sets': {'safe42': SAFE42, 'utilization_removed': UTIL, 'safe23': SAFE23},
    'aucs_full_pool': aucs_full,
    'boot_full_pool': {**{k: v for k, v in boot_full.items() if k != 'lr_placeholder'},
                       **boot_lr},
    'bedside_scores': {
        'bedside_padua_items': 'prior VTE history (3) + ICU within first 24 h immobility proxy (3) + age >= 70 (2) + BMI >= 30 (1) + hormonal treatment in first 24 h (1); index-admission ICD items (active cancer, thrombophilia, surgery/trauma, heart/respiratory failure, MI/stroke, infection/rheumatologic) scored 0 because discharge-coded',
        'bedside_padua_high_rate': float((bedside_padua >= 4).mean()),
        'bedside_improve_items': 'prior VTE history (3) + ICU immobility proxy (2) + ICU/CCU stay (1) + age > 60 (1); thrombophilia, cancer, central venous catheter scored 0 (discharge-coded or admission-level)',
        'bedside_improve_high_rate': float((bedside_improve >= 4).mean()),
    },
    'truncation_series': truncations,
    'sex_strata': sex,
    'strict_pool_primary_protocol': strict_primary,
    'sample_size': samplesize,
}
with open(f'{RES}/ajm/new/output_era/revision_response_analyses.json', 'w') as f:
    json.dump(out, f, indent=1)

rows = pd.DataFrame({'hadm_id': pool['hadm_id'].values,
                     'subject_id': groups, 'vte_event': y,
                     'padua_score': padua, 'improve_score': improve,
                     'caprini_score': caprini,
                     'bedside_padua': bedside_padua,
                     'bedside_improve': bedside_improve,
                     'oof_xgb42': oof42, 'oof_xgb23': oof23,
                     'oof_lr42': ooflr42, 'oof_lr23': ooflr23})
rows.to_parquet(f'{RES}/ajm/new/output_era/revision_oof_rows.parquet')
print('saved revision_response_analyses.json + revision_oof_rows.parquet', flush=True)
