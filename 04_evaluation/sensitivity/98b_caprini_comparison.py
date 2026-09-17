#!/usr/bin/env python3
"""Script 98b: incremental value of LR / XGBoost against THREE clinical scores.

Same protocol as scripts/98_ajm_sensitivity_cohorts.py (3 x 5-fold
StratifiedGroupKFold grouped by subject, seeds 42/43/44; fold-mean AUC;
seed-42 OOF AUC; patient-level cluster bootstrap, 2000 replicates, negatives
capped at 50k; test-set evaluation), extended so that Caprini appears as a
third baseline comparator next to Padua and IMPROVE.

Usage:
  python3 98b_caprini_comparison.py --tag <tag> --dataset <parquet> \
      --splits <json> --out <json>
"""
import json, time, warnings, os, sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEED = 42
N_REPEATS, N_SPLITS = 3, 5
SEEDS = [SEED + r for r in range(N_REPEATS)]
N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]
SCORE_COLS = {'padua': 'padua_score', 'improve': 'improve_score',
              'caprini': 'caprini_score'}
NRI_IDI_PAIRS = [('lr', 'padua'), ('xgb', 'padua'),
                 ('lr', 'improve'), ('xgb', 'improve'),
                 ('lr', 'caprini'), ('xgb', 'caprini'),
                 ('xgb', 'lr')]

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TAG = argval('--tag', 'inclprior_excl24h')
DATASET = argval('--dataset',
                 f'{RES}/ajm/new/data_era/'
                 f'primary_inclprior_excl24h_model_dataset_caprini.parquet')
SPLITS = argval('--splits',
                f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json')
OUT = argval('--out', f'{RES}/ajm/new/output_era/'
                      f'sensitivity_{TAG}_caprini_result.json')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(SPLITS))
FEATS = fs['main']

df = pd.read_parquet(DATASET)
n_all = len(df)
n_excl = int(df['prior_vte_any'].fillna(0).eq(1).sum())
df = df.reset_index(drop=True)
n_keep = len(df)

pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[~df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
N = len(y)
print(f'[{TAG}] cohort {n_all:,} rows (prior-VTE-positive {n_excl:,})', flush=True)
print(f'[{TAG}] pool N={N} events={y.sum()} ({y.mean()*100:.3f}%) '
      f'subjects={pool["subject_id"].nunique()}', flush=True)
print(f'[{TAG}] test N={len(test)} events={int(test["vte_event"].sum())}',
      flush=True)

# univariate logistic ("Platt") calibration of each score on the pool:
# raw score for AUC/AUPRC, calibrated probability for Brier/ECE/NRI/IDI.
raw, prob = {}, {}
for name, col in SCORE_COLS.items():
    raw[name] = pool[col].values.astype(float)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(raw[name].reshape(-1, 1), y)
    prob[name] = lr.predict_proba(raw[name].reshape(-1, 1))[:, 1]
    print(f'[{TAG}] {name}: pool mean={raw[name].mean():.2f} '
          f'AUC={roc_auc_score(y, raw[name]):.4f}', flush=True)


def fit_predict(model_kind, X_tr, y_tr, X_te):
    if model_kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1], m
    else:
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)[:, 1], m


def run_cv(kind, seed):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(N)
    fr = []
    for tr, te in sgkf.split(X, y, groups):
        preds, _ = fit_predict(kind, X[tr], y[tr], X[te])
        oof[te] = preds
        fr.append({'auc': roc_auc_score(y[te], preds),
                   'auprc': average_precision_score(y[te], preds),
                   'brier': brier_score_loss(y[te], preds)})
    return fr, oof


def fold_boot_ci(fr):
    rng = np.random.RandomState(SEED)
    vals = [np.mean([fr[i]['auc'] for i in rng.randint(0, len(fr), len(fr))])
            for _ in range(N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
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


results, oofs = {}, {}
def pkey(k):
    """probability vector key: calibrated prob for scores, OOF prob for models"""
    return f'{k}_prob' if k in SCORE_COLS else k


for kind in ['lr', 'xgb']:
    t0 = time.time()
    all_fr = []
    for seed in SEEDS:
        fr, oof = run_cv(kind, seed)
        all_fr.extend(fr)
        if seed == SEED:
            oofs[kind] = oof
    ci = fold_boot_ci(all_fr)
    results[kind] = {
        'fold_mean_auc': float(np.mean([f['auc'] for f in all_fr])),
        'fold_mean_auc_ci': ci,
        'fold_mean_auprc': float(np.mean([f['auprc'] for f in all_fr])),
        'fold_mean_brier': float(np.mean([f['brier'] for f in all_fr])),
        'n_folds': len(all_fr),
        'oof_auc': float(roc_auc_score(y, oofs[kind])),
        'oof_auprc': float(average_precision_score(y, oofs[kind])),
        'oof_brier': float(brier_score_loss(y, oofs[kind])),
        'oof_ece': ece(y, oofs[kind]),
    }
    r = results[kind]
    print(f"[{TAG}] {kind.upper()}: fold-mean AUC={r['fold_mean_auc']:.4f} "
          f"[{ci[0]:.4f}-{ci[1]:.4f}] AUPRC={r['fold_mean_auprc']:.4f} "
          f"| OOF AUC={r['oof_auc']:.4f} Brier={r['oof_brier']:.5f} "
          f"ECE={r['oof_ece']:.4f} ({time.time()-t0:.0f}s)", flush=True)

for name in SCORE_COLS:
    results[f'{name}_pool'] = {
        'oof_auc': float(roc_auc_score(y, raw[name])),
        'oof_auprc': float(average_precision_score(y, raw[name])),
        'oof_brier': float(brier_score_loss(y, prob[name])),
        'oof_ece': ece(y, prob[name]),
        'mean_score': float(raw[name].mean()), 'sd_score': float(raw[name].std()),
        'note': 'raw score for AUC/AUPRC; probability via univariate logistic '
                'calibration fit on the pool for Brier/ECE/NRI/IDI'}
    r = results[f'{name}_pool']
    print(f"[{TAG}] {name.upper()} (pool): AUC={r['oof_auc']:.4f} "
          f"AUPRC={r['oof_auprc']:.4f} Brier={r['oof_brier']:.5f} "
          f"ECE={r['oof_ece']:.4f}", flush=True)

print(f'[{TAG}] cluster bootstrap ...', flush=True)
subj_codes, subj_uniq = pd.factorize(pool['subject_id'])
n_subj = len(subj_uniq)
ev_mask = y == 1
ev_rows = np.where(ev_mask)[0]
neg_rows = np.where(~ev_mask)[0]
ev_subj = subj_codes[ev_rows]
neg_subj = subj_codes[neg_rows]

rng = np.random.RandomState(SEED)
P = {'lr': oofs['lr'], 'xgb': oofs['xgb'],
     'padua': raw['padua'], 'improve': raw['improve'], 'caprini': raw['caprini'],
     'padua_prob': prob['padua'], 'improve_prob': prob['improve'],
     'caprini_prob': prob['caprini']}
DELTA_PAIRS = [('lr', 'padua'), ('xgb', 'padua'),
               ('lr', 'improve'), ('xgb', 'improve'),
               ('lr', 'caprini'), ('xgb', 'caprini'),
               ('xgb', 'lr'),
               ('padua', 'caprini'), ('improve', 'caprini'),
               ('padua', 'improve')]
AUC_KEYS = ['lr', 'xgb', 'padua', 'improve', 'caprini']
deltas = {f'{new}_vs_{old}': [] for new, old in DELTA_PAIRS}
auc_boot = {k: [] for k in AUC_KEYS}
nri_boot = {(f'{new}_vs_{old}', t): [] for new, old in NRI_IDI_PAIRS
            for t in NRI_THRESHOLDS}
idi_boot = {f'{new}_vs_{old}': [] for new, old in NRI_IDI_PAIRS}

for b in range(N_BOOT):
    counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
    ev_mult = counts[ev_subj]
    take = ev_mult > 0
    if take.sum() < 2:
        continue
    be_rows = np.repeat(ev_rows[take], ev_mult[take])
    chosen_neg = counts[neg_subj] > 0
    bn_rows = neg_rows[chosen_neg]
    if len(bn_rows) > NEG_CAP:
        bn_rows = rng.choice(bn_rows, NEG_CAP, replace=False)
    rows = np.concatenate([be_rows, bn_rows])
    yb = y[rows]
    aucs = {k: fast_auc(yb, P[k][rows]) for k in AUC_KEYS}
    for k in aucs:
        auc_boot[k].append(aucs[k])
    for new, old in DELTA_PAIRS:
        deltas[f'{new}_vs_{old}'].append(aucs[new] - aucs[old])
    for new, old in NRI_IDI_PAIRS:
        nm = f'{new}_vs_{old}'
        po, pn = P[pkey(old)][rows], P[pkey(new)][rows]
        for t in NRI_THRESHOLDS:
            nri_boot[(nm, t)].append(nri(po, pn, yb, t))
        idi_boot[nm].append(idi(po, pn, yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT}', flush=True)

results['oof_inference'] = {
    'method': 'patient-level cluster bootstrap (subject_id), 2000 replicates; '
              'negatives downsampled to 50k per replicate, AUC via '
              'Mann-Whitney rank statistic; NRI/IDI use score probabilities '
              'from univariate logistic calibration on the pool',
    'n_replicates_used': len(auc_boot['lr']),
    'oof_auc_ci': {k: {
        'auc': results[k]['oof_auc'] if k in ('lr', 'xgb')
        else results[f'{k}_pool']['oof_auc'],
        'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
        for k, v2 in auc_boot.items()},
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
    'nri': {f'{new}_vs_{old}_thr{t}': {
        'point': nri(P[pkey(old)], P[pkey(new)], y, t),
        'ci': [float(v) for v in np.percentile(
            nri_boot[(f'{new}_vs_{old}', t)], [2.5, 97.5])]}
        for new, old in NRI_IDI_PAIRS
        for t in NRI_THRESHOLDS},
    'idi': {f'{new}_vs_{old}': {
        'point': idi(P[pkey(old)], P[pkey(new)], y),
        'ci': [float(v) for v in np.percentile(
            idi_boot[f'{new}_vs_{old}'], [2.5, 97.5])]}
        for new, old in NRI_IDI_PAIRS},
}
inf = results['oof_inference']
for k, d in inf['oof_auc_ci'].items():
    print(f'  OOF AUC {k}: {d["auc"]:.4f} [{d["ci"][0]:.4f},{d["ci"][1]:.4f}]',
          flush=True)
for k in [f'{a}_vs_{b}' for a, b in DELTA_PAIRS]:
    d = inf['delta_auc'][k]
    print(f"  dAUC {k}: {d['estimate']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}",
          flush=True)
for k, d in inf['nri'].items():
    print(f"  NRI {k}: {d['point']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]",
          flush=True)
for k, d in inf['idi'].items():
    print(f"  IDI {k}: {d['point']:+.5f} [{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]",
          flush=True)

# --- test-set evaluation -------------------------------------------------
test_X = test[FEATS].values.astype(np.float32)
test_y = test['vte_event'].values.astype(int)
imp = SimpleImputer(strategy='median')
m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
m.fit(imp.fit_transform(X), y)
t_lr = m.predict_proba(imp.transform(test_X))[:, 1]
m = XGBClassifier(**XGB_PARAMS).fit(X, y)
t_xgb = m.predict_proba(test_X)[:, 1]
model_test = {'lr': t_lr, 'xgb': t_xgb}
results['test_set'] = {
    'n': int(len(test)), 'events': int(test_y.sum()),
    'event_rate': float(test_y.mean()),
}
for kind in ['lr', 'xgb']:
    results['test_set'][f'{kind}_auc'] = float(roc_auc_score(test_y,
                                                             model_test[kind]))
    results['test_set'][f'{kind}_auprc'] = float(average_precision_score(
        test_y, model_test[kind]))
for name, col in SCORE_COLS.items():
    s = test[col].values.astype(float)
    results['test_set'][f'{name}_auc'] = float(roc_auc_score(test_y, s))
    results['test_set'][f'{name}_auprc'] = float(average_precision_score(test_y, s))
    results['test_set'][f'{name}_high_prop'] = float(
        test[f'{name}_high' if name != 'caprini'
             else 'caprini_high'].mean())
results['test_set']['auprc_baseline_prevalence'] = float(test_y.mean())
print(f"[{TAG}] test_set:", json.dumps(results['test_set']), flush=True)

results['meta'] = {
    'tag': TAG, 'dataset': DATASET, 'splits': SPLITS,
    'n_all': n_all, 'n_kept': n_keep,
    'n_excluded_prior_vte': n_excl, 'features': FEATS,
    'n_pool': N, 'n_events': int(y.sum()),
    'comparators': ['padua_score', 'improve_score (via improve_score column)',
                    'caprini_score (85h, item mapping in '
                    'results_vte/score_mapping_caprini.md)'],
    'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44), stratify vte_event, '
          'group subject_id; fold bootstrap 2000 for fold-mean AUC CI',
}
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved -> {OUT}', flush=True)
