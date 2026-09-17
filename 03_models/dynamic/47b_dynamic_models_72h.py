#!/usr/bin/env python3
"""
Script 47 (VTE M5 / Task 3 Step 2): dynamic landmark LR / XGBoost evaluation.

Protocol mirrors scripts/40_static_models.py exactly:
  - LR: L2 C=1.0 max_iter=2000, per-fold training-median imputation
  - XGBoost: 300/3/0.05/0.8/0.8/5/1, logloss, native NaN, scale_pos_weight=1
  - CV: StratifiedGroupKFold (stratify label, group subject_id), 3x5 folds
    (seeds 42/43/44) on the train+val pool (undersampled training view);
    fold-mean AUC [CI] (fold bootstrap 2000) + AUPRC
  - seed42 OOF on the FULL evaluation view (all train+val landmark rows,
    not undersampled) + patient-level cluster bootstrap 2000:
    dAUC vs Padua (broadcast per hadm) and vs M4 static XGB (recomputed
    seed42 OOF on admissions, broadcast to landmark rows);
    NRI (0.5%/1%), IDI, Brier, ECE.

Outputs: results_vte/dynamic_model_results.json
"""
import json, sys, time, warnings

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

ID_COLS = ['hadm_id', 'subject_id', 't_lm', 'label', 'split', 'padua_score']


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# M6 sensitivity params (defaults unchanged)
TRAIN_VIEW = argval('--train-view', f'{RES}/landmark_features_train_v2_noprior.parquet')
EVAL_VIEW = argval('--eval-view', f'{RES}/landmark_features_v2_noprior.parquet')
OUT_JSON = argval('--out', f'{RES}/ajm/new/output_era/dynamic_model_results_v2.json')

print('loading landmark features ...', flush=True)
mdf_all = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet',
                          columns=['hadm_id', 'vte_event'])
lab_map = dict(zip(mdf_all['hadm_id'], mdf_all['vte_event'].astype(int)))
tr_df = pd.read_parquet(TRAIN_VIEW)
ev_df = pd.read_parquet(EVAL_VIEW)
pool_ids_all = set(json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))['train']) \
    | set(json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))['val'])
tr_df = tr_df[tr_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
ev_df = ev_df[ev_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
tr_df['label'] = tr_df['hadm_id'].map(lab_map).astype(int)
ev_df['label'] = ev_df['hadm_id'].map(lab_map).astype(int)
FEATS = [c for c in tr_df.columns if c not in ID_COLS]
print(f'train view {tr_df.shape}, eval view {ev_df.shape}, '
      f'features {len(FEATS)}', flush=True)

X = tr_df[FEATS].values.astype(np.float32)
y = tr_df['label'].values.astype(int)
groups = tr_df['subject_id'].values
N = len(y)
print(f'pool N={N} events={y.sum()} ({y.mean() * 100:.3f}%) '
      f'subjects={tr_df["subject_id"].nunique()}', flush=True)

XE = ev_df[FEATS].values.astype(np.float32)
yE = ev_df['label'].values.astype(int)
subjE = ev_df['subject_id'].values
padua_raw_E = ev_df['padua_score'].values.astype(float)
NE = len(yE)
print(f'eval rows {NE} events={yE.sum()} ({yE.mean() * 100:.4f}%)', flush=True)

# map subject_id -> eval row positions (a subject may have >1 hadm)
ev_pos = {}
for i, s in enumerate(subjE):
    ev_pos.setdefault(s, []).append(i)

# Padua calibrated to probability on the eval view (univariate logistic)
padua_raw_E = np.nan_to_num(padua_raw_E, nan=float(np.nanmedian(padua_raw_E)))
platt = LogisticRegression(C=1.0, max_iter=2000).fit(
    padua_raw_E.reshape(-1, 1), yE)
padua_prob_E = platt.predict_proba(padua_raw_E.reshape(-1, 1))[:, 1]


def fit_model(model_kind, X_tr, y_tr):
    if model_kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m, imp
    else:
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr, y_tr)
        return m, None


def predict(model_kind, m, imp, X_te):
    if model_kind == 'lr':
        return m.predict_proba(imp.transform(X_te))[:, 1]
    return m.predict_proba(X_te)[:, 1]


def run_cv(kind, seed, collect_oof=False):
    """one repeat of 5-fold SGKF on undersampled pool; optionally also
    predict the full eval rows of held-out subjects (seed42 OOF)."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=seed)
    oof_pool = np.zeros(N)
    oof_eval = np.zeros(NE) if collect_oof else None
    fr = []
    for tr, te in sgkf.split(X, y, groups):
        m, imp = fit_model(kind, X[tr], y[tr])
        preds = predict(kind, m, imp, X[te])
        oof_pool[te] = preds
        fr.append({'auc': roc_auc_score(y[te], preds),
                   'auprc': average_precision_score(y[te], preds),
                   'brier': brier_score_loss(y[te], preds)})
        if collect_oof:
            te_subj = set(groups[te])
            rows = np.concatenate(
                [ev_pos[s] for s in te_subj if s in ev_pos])
            oof_eval[rows] = predict(kind, m, imp, XE[rows])
    return fr, oof_pool, oof_eval


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


# ── static M4 XGB seed42 OOF (recompute, then broadcast to landmark rows) ──
print('recomputing static M4 XGB seed42 OOF (broadcast source) ...',
      flush=True)
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
STATIC_FEATS = fs['main']
mdf = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
mpool = mdf[mdf['hadm_id'].isin(pool_ids)].reset_index(drop=True)
XS = mpool[STATIC_FEATS].values.astype(np.float32)
yS = mpool['vte_event'].values.astype(int)
gS = mpool['subject_id'].values
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_static = np.zeros(len(yS))
for tr, te in sgkf.split(XS, yS, gS):
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(XS[tr], yS[tr])
    oof_static[te] = m.predict_proba(XS[te])[:, 1]
hadm2static = dict(zip(mpool['hadm_id'].values, oof_static))
static_E = ev_df['hadm_id'].map(hadm2static).values.astype(float)
assert not np.isnan(static_E).any(), 'eval hadm missing static OOF'
print(f'static XGB OOF (admission level, pool) AUC='
      f'{roc_auc_score(yS, oof_static):.4f}', flush=True)

# ── CV: 3x5 folds on undersampled pool ──
results, oofs_eval = {}, {}
for kind in ['lr', 'xgb']:
    t0 = time.time()
    all_fr = []
    for seed in SEEDS:
        fr, _, oof_ev = run_cv(kind, seed, collect_oof=(seed == SEED))
        all_fr.extend(fr)
        if seed == SEED:
            oofs_eval[kind] = oof_ev
    # persist the seed-42 dynamic XGBoost eval OOF for the GRU baseline
    if kind == 'xgb':
        np.save(f'{RES}/ajm/new/output_era/gru_xgb_oof_eval_72h_inclprior_excl24h.npy', oofs_eval['xgb'])
    ci = fold_boot_ci(all_fr)
    oe = oofs_eval[kind]
    results[kind] = {
        'fold_mean_auc': float(np.mean([f['auc'] for f in all_fr])),
        'fold_mean_auc_ci': ci,
        'fold_mean_auprc': float(np.mean([f['auprc'] for f in all_fr])),
        'fold_mean_brier': float(np.mean([f['brier'] for f in all_fr])),
        'n_folds': len(all_fr),
        'oof_auc': float(roc_auc_score(yE, oe)),
        'oof_auprc': float(average_precision_score(yE, oe)),
        'oof_brier': float(brier_score_loss(yE, oe)),
        'oof_ece': ece(yE, oe),
    }
    r = results[kind]
    print(f"{kind.upper()}: fold-mean AUC={r['fold_mean_auc']:.4f} "
          f"[{ci[0]:.4f}-{ci[1]:.4f}] AUPRC={r['fold_mean_auprc']:.4f} | "
          f"OOF(eval) AUC={r['oof_auc']:.4f} Brier={r['oof_brier']:.5f} "
          f"ECE={r['oof_ece']:.4f} ({time.time() - t0:.0f}s)", flush=True)

results['padua_eval'] = {
    'oof_auc': float(roc_auc_score(yE, padua_raw_E)),
    'oof_auprc': float(average_precision_score(yE, padua_raw_E)),
    'oof_brier': float(brier_score_loss(yE, padua_prob_E)),
    'oof_ece': ece(yE, padua_prob_E),
    'note': 'Padua broadcast per hadm; raw score for AUC/AUPRC, probability '
            'via univariate logistic calibration on eval view'}
results['static_xgb_eval'] = {
    'oof_auc': float(roc_auc_score(yE, static_E)),
    'oof_auprc': float(average_precision_score(yE, static_E)),
    'oof_brier': float(brier_score_loss(yE, static_E)),
    'oof_ece': ece(yE, static_E),
    'note': 'M4 static XGB seed42 OOF (admission level, recomputed) '
            'broadcast to landmark rows'}

# ── patient-level cluster bootstrap on eval view (seed42 OOF) ──
# bootstrap is run as a separate lightweight job (47b_bootstrap_only.py) that
# reads dynamic_oof_inputs_72h_inclprior_excl24h.npz and writes dynamic_oof_inference.json
print('cluster bootstrap ... (separate job reads dynamic_oof_inputs_72h_inclprior_excl24h.npz)', flush=True)

rng = np.random.RandomState(SEED)
P = {'lr': oofs_eval['lr'], 'xgb': oofs_eval['xgb'], 'padua': padua_raw_E,
     'padua_prob': padua_prob_E, 'static': static_E}
subj_codes, _ = pd.factorize(ev_df['subject_id'])
# persist OOF inputs so bootstrap inference can be run separately
np.savez(f'{RES}/ajm/new/output_era/dynamic_oof_inputs_72h_inclprior_excl24h.npz',
         yE=yE.astype(np.int8), subj_codes=subj_codes.astype(np.int64),
         padua_raw=padua_raw_E.astype(np.float32),
         padua_prob=padua_prob_E.astype(np.float32),
         lr=oofs_eval['lr'].astype(np.float32), xgb=oofs_eval['xgb'].astype(np.float32),
         static=static_E.astype(np.float32))
print('saved dynamic_oof_inputs_72h_inclprior_excl24h.npz', flush=True)
PAIRS = {'lr_vs_padua': ('padua_prob', 'lr'),
         'xgb_vs_padua': ('padua_prob', 'xgb'),
         'xgb_vs_static': ('static', 'xgb')}
deltas = {k: [] for k in PAIRS}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua', 'static']}
nri_boot = {(a, t): [] for a in PAIRS for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in PAIRS}

# load inference produced separately by 47b_bootstrap_only_72h.py
# (era fix: original arm_ipe script read the MAIN-arm dynamic_oof_inference.json
# here, so dynamic_model_results_72h_*.json embedded main-arm deltas; the true
# 72h deltas lived in dynamic_oof_inference_72h_*.json. Point at the 72h file.)
import os
_INF = f'{RES}/ajm/new/output_era/dynamic_oof_inference_72h_inclprior_excl24h.json'
_t0 = time.time()
while not os.path.exists(_INF):
    assert time.time() - _t0 < 3600, 'bootstrap inference json not produced'
    time.sleep(5)
with open(_INF) as f:
    inf = json.load(f)
inf['method'] = ('patient-level cluster bootstrap (subject_id) on eval '
                 'landmark view (train+val, rebuilt on new t0), 2000 reps; '
                 'negatives downsampled 50k; run as standalone job '
                 '47b_bootstrap_only.py reading dynamic_oof_inputs_72h_inclprior_excl24h.npz')
results['oof_inference'] = inf
for k, d in inf['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} [{d['ci'][0]:+.4f},"
          f"{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}", flush=True)

results['meta'] = {
    'features': FEATS,
    'n_pool_undersampled': N, 'n_events_pool': int(y.sum()),
    'n_eval': NE, 'n_events_eval': int(yE.sum()),
    'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44) on undersampled '
          'train+val view, stratify label, group subject_id; fold bootstrap '
          '2000 for fold-mean AUC CI; fold metrics on undersampled view, '
          'seed42 OOF predicted on full eval view',
    'lr': 'L2 C=1.0 max_iter=2000, no scaling, per-fold training-median '
          'imputation, no class_weight',
    'xgb': {k: v for k, v in XGB_PARAMS.items()
            if isinstance(v, (int, float, str))},
}

with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved -> {OUT_JSON}', flush=True)
