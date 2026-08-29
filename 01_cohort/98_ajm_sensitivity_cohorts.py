#!/usr/bin/env python3
"""
Script 98: AJM primary-cohort sensitivity analyses.

Runs the static-model pipeline (script 97 / 40b protocol) on the three
alternative inclusion-criteria cohorts, with the NDM-aligned cohort as the
primary comparator.

Cohorts:
  sensA (include prior VTE, exclude 24h): 416,638 / 1,623  (original AJM dataset)
  sensB (exclude prior VTE, include 24h): 381,919 / 3,753
  sensC (include both):                   419,301 / 4,070

Outputs: results_vte/ajm_primary/sensitivity_<tag>_result.json
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

XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


TAG = argval('--tag', 'sensB')          # sensA / sensB / sensC
DATASET = argval('--dataset', f'{RES}/ajm_primary/{TAG}_model_dataset.parquet')
SPLITS = argval('--splits', f'{RES}/ajm_primary/{TAG}_splits.json')
OUT = argval('--out', f'{RES}/ajm_primary/sensitivity_{TAG}_result.json')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(SPLITS))
FEATS = fs['main']

df = pd.read_parquet(DATASET)
n_all = len(df)
n_excl = int(df['prior_vte_any'].fillna(0).eq(1).sum())
if TAG == 'sensA':
    # sensA uses the original AJM dataset (no 24h labels, prior retained)
    df = df.reset_index(drop=True)
    n_keep = len(df)
elif TAG == 'sensB':
    df = df[df['prior_vte_any'].fillna(0).eq(0)].reset_index(drop=True)
    n_keep = len(df)
elif TAG == 'sensC':
    df = df.reset_index(drop=True)
    n_keep = len(df)
else:
    # custom pre-filtered dataset (e.g. ajm/new arms): use as-is
    df = df.reset_index(drop=True)
    n_keep = len(df)

pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[~df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
X = pool[FEATS].values.astype(np.float32)
y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
padua_raw = pool['padua_score'].values.astype(float)
N = len(y)
print(f'[{TAG}] cohort: {n_all:,} rows -> kept {n_keep:,} '
      f'(prior excluded {n_excl:,})', flush=True)
print(f'[{TAG}] pool N={N} events={y.sum()} ({y.mean()*100:.3f}%) '
      f'subjects={pool["subject_id"].nunique()}', flush=True)
print(f'[{TAG}] test N={len(test)} events={int(test["vte_event"].sum())}', flush=True)

platt = LogisticRegression(C=1.0, max_iter=2000).fit(padua_raw.reshape(-1, 1), y)
padua_prob = platt.predict_proba(padua_raw.reshape(-1, 1))[:, 1]


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

results['padua_pool'] = {
    'oof_auc': float(roc_auc_score(y, padua_raw)),
    'oof_auprc': float(average_precision_score(y, padua_raw)),
    'oof_brier': float(brier_score_loss(y, padua_prob)),
    'oof_ece': ece(y, padua_prob),
    'note': 'raw score for AUC/AUPRC; probability via univariate logistic '
            'calibration fit on the pool for Brier/ECE/NRI/IDI'}

print(f'[{TAG}] cluster bootstrap ...', flush=True)
subj_codes, subj_uniq = pd.factorize(pool['subject_id'])
n_subj = len(subj_uniq)
ev_mask = y == 1
ev_rows = np.where(ev_mask)[0]
neg_rows = np.where(~ev_mask)[0]
ev_subj = subj_codes[ev_rows]
neg_subj = subj_codes[neg_rows]

rng = np.random.RandomState(SEED)
P = {'lr': oofs['lr'], 'xgb': oofs['xgb'], 'padua': padua_raw,
     'padua_prob': padua_prob}
deltas = {k: [] for k in ['lr_vs_padua', 'xgb_vs_padua', 'xgb_vs_lr']}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua']}
nri_boot = {(a, t): [] for a in ['lr_vs_padua', 'xgb_vs_padua', 'xgb_vs_lr']
            for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in ['lr_vs_padua', 'xgb_vs_padua', 'xgb_vs_lr']}

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
    aucs = {k: fast_auc(yb, P[k][rows]) for k in ['lr', 'xgb', 'padua']}
    for k in aucs:
        auc_boot[k].append(aucs[k])
    deltas['lr_vs_padua'].append(aucs['lr'] - aucs['padua'])
    deltas['xgb_vs_padua'].append(aucs['xgb'] - aucs['padua'])
    deltas['xgb_vs_lr'].append(aucs['xgb'] - aucs['lr'])
    pairs = {'lr_vs_padua': ('padua_prob', 'lr'),
             'xgb_vs_padua': ('padua_prob', 'xgb'),
             'xgb_vs_lr': ('lr', 'xgb')}
    for name, (old, new) in pairs.items():
        for t in NRI_THRESHOLDS:
            nri_boot[(name, t)].append(nri(P[old][rows], P[new][rows], yb, t))
        idi_boot[name].append(idi(P[old][rows], P[new][rows], yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT}', flush=True)

results['oof_inference'] = {
    'method': 'patient-level cluster bootstrap (subject_id), 2000 replicates; '
              'negatives downsampled to 50k per replicate, AUC via Mann-Whitney '
              'rank statistic; NRI/IDI use Padua probability from univariate '
              'logistic calibration on the pool',
    'n_replicates_used': len(auc_boot['lr']),
    'oof_auc_ci': {k: {'auc': results[k if k != 'padua' else 'padua_pool']['oof_auc'],
                       'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
                   for k, v2 in auc_boot.items()},
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
    'nri': {f'{name}_thr{t}': {'point': nri(P[old], P[new], y, t),
                               'ci': [float(v) for v in np.percentile(
                                   nri_boot[(name, t)], [2.5, 97.5])]}
            for name, (old, new) in
            {'lr_vs_padua': ('padua_prob', 'lr'),
             'xgb_vs_padua': ('padua_prob', 'xgb'),
             'xgb_vs_lr': ('lr', 'xgb')}.items()
            for t in NRI_THRESHOLDS},
    'idi': {name: {'point': idi(P[old], P[new], y),
                   'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
            for name, (old, new), v2 in
            [(n, p, idi_boot[n]) for n, p in
             {'lr_vs_padua': ('padua_prob', 'lr'),
              'xgb_vs_padua': ('padua_prob', 'xgb'),
              'xgb_vs_lr': ('lr', 'xgb')}.items()]},
}
inf = results['oof_inference']
for k, d in inf['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}",
          flush=True)
for k, d in inf['nri'].items():
    print(f"  NRI {k}: {d['point']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]",
          flush=True)
for k, d in inf['idi'].items():
    print(f"  IDI {k}: {d['point']:+.5f} [{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]",
          flush=True)

# test-set evaluation (per abstract)
test_X = test[FEATS].values.astype(np.float32)
test_y = test['vte_event'].values.astype(int)
test_padua = test['padua_score'].values.astype(float)
imp = SimpleImputer(strategy='median')
m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
m.fit(imp.fit_transform(X), y)
t_lr = m.predict_proba(imp.transform(test_X))[:, 1]
m = XGBClassifier(**XGB_PARAMS).fit(X, y)
t_xgb = m.predict_proba(test_X)[:, 1]
results['test_set'] = {
    'n': int(len(test)), 'events': int(test_y.sum()),
    'lr_auc': float(roc_auc_score(test_y, t_lr)),
    'xgb_auc': float(roc_auc_score(test_y, t_xgb)),
    'padua_auc': float(roc_auc_score(test_y, test_padua)),
}
print(f"[{TAG}] test_set:", json.dumps(results['test_set']), flush=True)

results['meta'] = {
    'tag': TAG, 'n_all': n_all, 'n_kept': n_keep,
    'n_excluded_prior_vte': n_excl, 'features': FEATS,
    'n_pool': N, 'n_events': int(y.sum()),
    'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44), stratify vte_event, '
          'group subject_id; fold bootstrap 2000 for fold-mean AUC CI',
}
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved -> {OUT}', flush=True)