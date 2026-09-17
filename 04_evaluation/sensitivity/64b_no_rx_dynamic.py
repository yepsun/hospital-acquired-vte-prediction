#!/usr/bin/env python3
"""
Script 64b: dynamic landmark variant without the treatment columns.

Mirrors scripts_era/47b_dynamic_models_v2.py exactly (LR / XGBoost, 3x5
StratifiedGroupKFold on the undersampled train+val landmark view, seed-42 OOF
predicted on the full eval view, patient-level cluster bootstrap 2000 reps,
negatives downsampled to 50k) except that the 4 treatment columns of the
dynamic feature set are dropped:

    rx_heparin_last24h, rx_warfarin_last24h, rx_doac_last24h,
    rx_antiplatelet_last24h            (92 -> 88 features)

Comparators on the eval view: raw Padua broadcast per hadm, and the static
seed-42 XGB OOF broadcast per hadm, computed both with the 57-feature main set
(published comparator, xgb_vs_static) and with the 52-feature no-rx set
(consistent treatment policy, xgb_vs_static52).

Outputs (new files; nothing existing is overwritten):
  output_era/dynamic_norx_model_results_inclprior_excl24h.json
  output_era/dynamic_norx_oof_inputs_inclprior_excl24h.npz
"""
import json, time, warnings

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
D = f'{RES}/ajm/new/data_era'
O = f'{RES}/ajm/new/output_era'
TRAIN_VIEW = f'{D}/landmark_features_train_inclprior_excl24h.parquet'
EVAL_VIEW = f'{D}/landmark_features_inclprior_excl24h.parquet'
OUT_JSON = f'{O}/dynamic_norx_model_results_inclprior_excl24h.json'
OUT_NPZ = f'{O}/dynamic_norx_oof_inputs_inclprior_excl24h.npz'

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
RX_DYN_DROP = ['rx_heparin_last24h', 'rx_warfarin_last24h', 'rx_doac_last24h',
               'rx_antiplatelet_last24h']
T0 = time.time()


def el():
    return f'{time.time() - T0:.0f}s'


print('loading landmark features ...', flush=True)
mdf_all = pd.read_parquet(f'{D}/primary_inclprior_excl24h_model_dataset.parquet',
                          columns=['hadm_id', 'vte_event'])
lab_map = dict(zip(mdf_all['hadm_id'], mdf_all['vte_event'].astype(int)))
splits = json.load(open(f'{D}/primary_inclprior_excl24h_splits.json'))
pool_ids_all = set(splits['train']) | set(splits['val'])
tr_df = pd.read_parquet(TRAIN_VIEW)
ev_df = pd.read_parquet(EVAL_VIEW)
tr_df = tr_df[tr_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
ev_df = ev_df[ev_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
tr_df['label'] = tr_df['hadm_id'].map(lab_map).astype(int)
ev_df['label'] = ev_df['hadm_id'].map(lab_map).astype(int)

FEATS_ALL = [c for c in tr_df.columns if c not in ID_COLS]
assert all(c in FEATS_ALL for c in RX_DYN_DROP), 'rx_last24h cols missing'
FEATS = [c for c in FEATS_ALL if c not in RX_DYN_DROP]
print(f'train view {tr_df.shape}, eval view {ev_df.shape}, '
      f'features {len(FEATS)} (dropped {RX_DYN_DROP})', flush=True)

X = tr_df[FEATS].values.astype(np.float32)
y = tr_df['label'].values.astype(int)
groups = tr_df['subject_id'].values
N = len(y)
print(f'pool N={N} events={y.sum()} ({y.mean()*100:.3f}%) '
      f'subjects={tr_df["subject_id"].nunique()}', flush=True)

XE = ev_df[FEATS].values.astype(np.float32)
yE = ev_df['label'].values.astype(int)
subjE = ev_df['subject_id'].values
NE = len(yE)
print(f'eval rows {NE} events={yE.sum()} ({yE.mean()*100:.4f}%) ({el()})',
      flush=True)

ev_pos = {}
for i, s in enumerate(subjE):
    ev_pos.setdefault(s, []).append(i)

padua_raw_E = np.nan_to_num(ev_df['padua_score'].values.astype(float),
                            nan=float(np.nanmedian(
                                ev_df['padua_score'].values.astype(float))))
platt = LogisticRegression(C=1.0, max_iter=2000).fit(
    padua_raw_E.reshape(-1, 1), yE)
padua_prob_E = platt.predict_proba(padua_raw_E.reshape(-1, 1))[:, 1]


def fit_model(kind, X_tr, y_tr):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m, imp
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr)
    return m, None


def predict(kind, m, imp, X_te):
    if kind == 'lr':
        return m.predict_proba(imp.transform(X_te))[:, 1]
    return m.predict_proba(X_te)[:, 1]


def run_cv(kind, seed, collect_oof=False):
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
            rows = np.concatenate([ev_pos[s] for s in te_subj if s in ev_pos])
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
    vals = np.asarray(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


# ── static XGB seed-42 OOF broadcast source (57- and 52-feature) ─────────
print('recomputing static M4 XGB seed42 OOF (broadcast sources) ...', flush=True)
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
MAIN = fs['main']
NORX = [f for f in MAIN if not f.startswith('rx_')]
mdf = pd.read_parquet(f'{D}/primary_inclprior_excl24h_model_dataset.parquet')
mpool = mdf[mdf['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
yS = mpool['vte_event'].values.astype(int)
gS = mpool['subject_id'].values
static_broadcast = {}
for set_name, feats in [('static57', MAIN), ('static52', NORX)]:
    XS = mpool[feats].values.astype(np.float32)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=SEED)
    oof_static = np.zeros(len(yS))
    for tr, te in sgkf.split(XS, yS, gS):
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(XS[tr], yS[tr])
        oof_static[te] = m.predict_proba(XS[te])[:, 1]
    static_broadcast[set_name] = np.nan_to_num(
        ev_df['hadm_id'].map(dict(zip(mpool['hadm_id'].values, oof_static)))
        .values.astype(float))
    print(f'  {set_name} admission-level OOF AUC='
          f'{roc_auc_score(yS, oof_static):.4f} ({el()})', flush=True)

# ── CV: 3x5 folds on the undersampled pool ───────────────────────────────
results, oofs_eval = {}, {}
for kind in ['lr', 'xgb']:
    t0 = time.time()
    all_fr = []
    for seed in SEEDS:
        fr, _, oof_ev = run_cv(kind, seed, collect_oof=(seed == SEED))
        all_fr.extend(fr)
        if seed == SEED:
            oofs_eval[kind] = oof_ev
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
            'via univariate logistic calibration on eval view',
}
for set_name, v in static_broadcast.items():
    results[set_name + '_eval'] = {
        'oof_auc': float(roc_auc_score(yE, v)),
        'oof_auprc': float(average_precision_score(yE, v)),
        'oof_brier': float(brier_score_loss(yE, v)),
        'oof_ece': ece(yE, v),
        'note': 'static seed-42 XGB OOF (admission level) broadcast to '
                'landmark rows; feature set ' +
                ('main 57' if set_name == 'static57' else 'no-rx 52'),
    }

np.savez(OUT_NPZ, yE=yE.astype(np.int8),
         subj_codes=pd.factorize(ev_df['subject_id'])[0].astype(np.int64),
         padua_raw=padua_raw_E.astype(np.float32),
         padua_prob=padua_prob_E.astype(np.float32),
         lr=oofs_eval['lr'].astype(np.float32),
         xgb=oofs_eval['xgb'].astype(np.float32),
         static52=static_broadcast['static52'].astype(np.float32),
         static57=static_broadcast['static57'].astype(np.float32))
print('saved', OUT_NPZ, flush=True)

# ── patient-level cluster bootstrap on the eval view ─────────────────────
print('cluster bootstrap ...', flush=True)
PAIRS = {'lr_vs_padua': ('padua_prob', 'lr'),
         'xgb_vs_padua': ('padua_prob', 'xgb'),
         'xgb_vs_static52': ('static52', 'xgb'),
         'xgb_vs_static57': ('static57', 'xgb')}
# AUC is invariant to the monotone Padua calibration, so bootstrap the AUC
# deltas on the raw score (as 47b does) and NRI/IDI on the probability.
AUC_PAIRS = {k: ('padua' if o == 'padua_prob' else o, n)
             for k, (o, n) in PAIRS.items()}
P = {'lr': oofs_eval['lr'], 'xgb': oofs_eval['xgb'],
     'padua': padua_raw_E, 'padua_prob': padua_prob_E,
     'static52': static_broadcast['static52'],
     'static57': static_broadcast['static57']}
subj_codes, _ = pd.factorize(ev_df['subject_id'])
n_subj = len(np.unique(subj_codes))
ev_rows = np.where(yE == 1)[0]
neg_rows = np.where(yE == 0)[0]
ev_subj = subj_codes[ev_rows]
neg_subj = subj_codes[neg_rows]
rng = np.random.RandomState(SEED)
deltas = {k: [] for k in PAIRS}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua', 'static52', 'static57']}
nri_boot = {(a, t): [] for a in PAIRS for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in PAIRS}
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
    yb = yE[rows]
    aucs = {k: fast_auc(yb, P[k][rows]) for k in auc_boot}
    for k in aucs:
        auc_boot[k].append(aucs[k])
    for name, (old, new) in AUC_PAIRS.items():
        deltas[name].append(aucs[new] - aucs[old])
    for name, (old, new) in PAIRS.items():
        for t in NRI_THRESHOLDS:
            nri_boot[(name, t)].append(nri(P[old][rows], P[new][rows], yb, t))
        idi_boot[name].append(idi(P[old][rows], P[new][rows], yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT} ({el()})', flush=True)

results['oof_inference'] = {
    'method': 'patient-level cluster bootstrap (subject_id) on eval landmark '
              'view (train+val), 2000 reps; negatives downsampled 50k',
    'n_replicates_used': len(auc_boot['lr']),
    'oof_auc_ci': {k: {'auc': float(fast_auc(yE, P[k])),
                       'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
                   for k, v2 in auc_boot.items()},
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
    'nri': {f'{name}_thr{t}': {'point': nri(P[o], P[n], yE, t),
                               'ci': [float(v) for v in np.percentile(
                                   nri_boot[(name, t)], [2.5, 97.5])]}
            for name, (o, n) in PAIRS.items() for t in NRI_THRESHOLDS},
    'idi': {name: {'point': idi(P[o], P[n], yE),
                   'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
            for name, (o, n), v2 in
            [(k, PAIRS[k], idi_boot[k]) for k in PAIRS]},
}
for k, d in results['oof_inference']['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} [{d['ci'][0]:+.4f},"
          f"{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}", flush=True)

results['meta'] = {
    'features': FEATS, 'n_features': len(FEATS),
    'dropped_vs_dynamic_92': RX_DYN_DROP,
    'n_pool_undersampled': N, 'n_events_pool': int(y.sum()),
    'n_eval': NE, 'n_events_eval': int(yE.sum()),
    'cv': 'StratifiedGroupKFold 3x5 (seeds 42/43/44) on undersampled train+val '
          'view, stratify label, group subject_id; fold bootstrap 2000 for '
          'fold-mean AUC CI; fold metrics on undersampled view, seed42 OOF '
          'predicted on full eval view',
    'baseline_dynamic_92': {
        'lr_fold_mean_auc': 0.7604891617928821,
        'lr_oof_auc': 0.8125065767082207,
        'xgb_fold_mean_auc': 0.8067498796328404,
        'xgb_oof_auc': 0.8506208806622986,
        'source': 'output_era/dynamic_model_results_inclprior_excl24h.json',
    },
    'caveat_rx_hormone_padua': 'rx_hormone feeds the Padua "ongoing hormonal '
        'treatment" item; it is not part of the dynamic landmark feature set, '
        'so this arm has no analogous asymmetry for Padua.',
    'runtime_s': round(time.time() - T0, 1),
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved -> {OUT_JSON} ({el()})', flush=True)
