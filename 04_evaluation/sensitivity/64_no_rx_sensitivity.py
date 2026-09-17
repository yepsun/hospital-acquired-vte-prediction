#!/usr/bin/env python3
"""
Script 64: treatment-related-predictor (confounding by indication) sensitivity.

Drops the 5 treatment columns from the 57-feature main set
(rx_heparin, rx_warfarin, rx_doac, rx_antiplatelet, rx_hormone) -> 52 features,
and reruns the full script-98 static protocol on the inclprior_excl24h primary
cohort:
  - LR (L2, C=1.0, per-fold training-median imputation) and XGBoost
    (300/3/0.05/0.8/0.8/5/1, native NaN)
  - 3 seeds x 5-fold StratifiedGroupKFold (stratify vte_event, group subject_id)
  - fold-mean AUC (fold bootstrap 2000) and seed-42/43/44 OOF AUC
  - held-out test AUC (models refit on the full train+val pool)
  - patient-level cluster bootstrap (2000 reps, negatives capped at 50k) for
    dAUC vs Padua and vs IMPROVE with CI and two-sided p, plus NRI (0.5%/1%)
    and IDI

The 57-feature arm is recomputed inside this script on the identical folds so
the two feature sets are compared like-for-like; the paired bootstrap delta of
xgb_52 vs xgb_57 (and lr_52 vs lr_57) is the amount of apparent performance
attributable to the 5 treatment columns.

Also produced: heparin-exposed / heparin-unexposed subgroup OOF AUCs (to
compare with the published 57-feature values 0.7852 / 0.8372) and descriptive
univariate statistics of the 5 dropped columns.

CAVEAT recorded in the output meta: rx_hormone also feeds the Padua score's
"ongoing hormonal treatment" item (scripts/85e_clinical_scores_v3.py:40), so
dropping it from the ML models only - while the score keeps using it - creates
an asymmetry between the arms. IMPROVE uses no treatment column.

Output: results_vte/ajm/new/output_era/no_rx_sensitivity_inclprior_excl24h.json
"""
import json, time, warnings
from collections import OrderedDict

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
DATASET = f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet'
SPLITS = f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'
OUT = f'{RES}/ajm/new/output_era/no_rx_sensitivity_inclprior_excl24h.json'

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

RX_DROP = ['rx_heparin', 'rx_warfarin', 'rx_doac', 'rx_antiplatelet',
           'rx_hormone']

T0 = time.time()
OUTOBJ = {}


def save():
    with open(OUT, 'w') as f:
        json.dump(OUTOBJ, f, indent=2)
    print(f'[saved -> {OUT} ({el()})]', flush=True)


def el():
    return f'{time.time() - T0:.0f}s'


fs = json.load(open(f'{RES}/feature_sets_v2.json'))
MAIN = fs['main']
assert all(c in MAIN for c in RX_DROP), 'rx columns missing from main set'
NORX = [f for f in MAIN if f not in RX_DROP]
FEATURE_SETS = OrderedDict([('main_57', MAIN), ('no_rx_52', NORX)])
print(f'feature sets: main_57={len(MAIN)} no_rx_52={len(NORX)} '
      f'(dropped {RX_DROP})', flush=True)

splits = json.load(open(SPLITS))
df = pd.read_parquet(DATASET)
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
test = df[~df['hadm_id'].isin(pool_ids)].reset_index(drop=True)

y = pool['vte_event'].values.astype(int)
groups = pool['subject_id'].values
N = len(y)
n_subj = pool['subject_id'].nunique()
print(f'pool N={N} events={y.sum()} ({y.mean()*100:.4f}%) subjects={n_subj}',
      flush=True)
print(f'test N={len(test)} events={int(test["vte_event"].sum())}', flush=True)


def score_col(frame, col):
    s = frame[col].values.astype(float)
    return np.nan_to_num(s, nan=float(np.nanmedian(s)))


padua_raw = score_col(pool, 'padua_score')
improve_raw = score_col(pool, 'improve_score')
platt_p = LogisticRegression(C=1.0, max_iter=2000).fit(
    padua_raw.reshape(-1, 1), y)
padua_prob = platt_p.predict_proba(padua_raw.reshape(-1, 1))[:, 1]
platt_i = LogisticRegression(C=1.0, max_iter=2000).fit(
    improve_raw.reshape(-1, 1), y)
improve_prob = platt_i.predict_proba(improve_raw.reshape(-1, 1))[:, 1]


def fit_predict(kind, X_tr, y_tr, X_te, seed):
    if kind == 'lr':
        imp = SimpleImputer(strategy='median')
        m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        m.fit(imp.fit_transform(X_tr), y_tr)
        return m.predict_proba(imp.transform(X_te))[:, 1], m
    m = XGBClassifier(**{**XGB_PARAMS, 'random_state': seed})
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], m


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


def fold_boot_ci(fr):
    rng = np.random.RandomState(SEED)
    vals = [np.mean([fr[i]['auc'] for i in rng.randint(0, len(fr), len(fr))])
            for _ in range(N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


# ── cross-validated runs (identical folds for both feature sets) ──────────
oof = {}          # (set_name, kind, seed) -> OOF vector on the pool
cv_results = {}
for set_name, feats in FEATURE_SETS.items():
    X = pool[feats].values.astype(np.float32)
    cv_results[set_name] = {}
    for kind in ['lr', 'xgb']:
        all_fr, per_seed = [], {}
        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=seed)
            o = np.zeros(N)
            fr = []
            for tr, te in sgkf.split(X, y, groups):
                preds, _ = fit_predict(kind, X[tr], y[tr], X[te], seed)
                o[te] = preds
                fr.append({'auc': roc_auc_score(y[te], preds),
                           'auprc': average_precision_score(y[te], preds),
                           'brier': brier_score_loss(y[te], preds)})
            oof[(set_name, kind, seed)] = o
            all_fr.extend(fr)
            per_seed[str(seed)] = {
                'fold_mean_auc': float(np.mean([f['auc'] for f in fr])),
                'fold_aucs': [float(f['auc']) for f in fr],
                'oof_auc': float(roc_auc_score(y, o)),
                'oof_auprc': float(average_precision_score(y, o)),
                'oof_brier': float(brier_score_loss(y, o)),
                'oof_ece': ece(y, o),
            }
        ci = fold_boot_ci(all_fr)
        cv_results[set_name][kind] = {
            'fold_mean_auc': float(np.mean([f['auc'] for f in all_fr])),
            'fold_mean_auc_ci': ci,
            'fold_mean_auprc': float(np.mean([f['auprc'] for f in all_fr])),
            'fold_mean_brier': float(np.mean([f['brier'] for f in all_fr])),
            'n_folds': len(all_fr),
            'per_seed': per_seed,
            'oof_auc_mean_over_seeds': float(np.mean(
                [per_seed[str(s)]['oof_auc'] for s in SEEDS])),
            'seed42': per_seed[str(SEED)],
        }
        r = cv_results[set_name][kind]
        print(f'[{set_name}] {kind.upper()}: fold-mean AUC={r["fold_mean_auc"]:.4f} '
              f'[{ci[0]:.4f}-{ci[1]:.4f}] | per-seed OOF '
              f'{ {s: round(per_seed[str(s)]["oof_auc"], 4) for s in SEEDS} } '
              f'({el()})', flush=True)

cv_results['padua_pool'] = {
    'auc': float(roc_auc_score(y, padua_raw)),
    'auprc': float(average_precision_score(y, padua_raw)),
    'brier': float(brier_score_loss(y, padua_prob)),
    'ece': ece(y, padua_prob),
}
cv_results['improve_pool'] = {
    'auc': float(roc_auc_score(y, improve_raw)),
    'auprc': float(average_precision_score(y, improve_raw)),
    'brier': float(brier_score_loss(y, improve_prob)),
    'ece': ece(y, improve_prob),
}
print(f'[scores] Padua AUC={cv_results["padua_pool"]["auc"]:.4f} '
      f'IMPROVE AUC={cv_results["improve_pool"]["auc"]:.4f}', flush=True)
OUTOBJ['cv'] = cv_results
save()

# ── patient-level cluster bootstrap, one pass per seed ───────────────────
# prediction vectors: 4 models (52/57-feature LR/XGB on that seed's OOF)
# + Padua and IMPROVE probabilities (fixed across seeds).
AUC_KEYS = ['lr52', 'xgb52', 'lr57', 'xgb57', 'padua', 'improve']
DELTA_PAIRS = OrderedDict([
    ('lr52_vs_padua', ('padua', 'lr52')),
    ('xgb52_vs_padua', ('padua', 'xgb52')),
    ('lr52_vs_improve', ('improve', 'lr52')),
    ('xgb52_vs_improve', ('improve', 'xgb52')),
    ('lr57_vs_padua', ('padua', 'lr57')),
    ('xgb57_vs_padua', ('padua', 'xgb57')),
    ('lr57_vs_improve', ('improve', 'lr57')),
    ('xgb57_vs_improve', ('improve', 'xgb57')),
    ('xgb52_vs_lr52', ('lr52', 'xgb52')),
    ('xgb57_vs_lr57', ('lr57', 'xgb57')),
    ('xgb52_vs_xgb57', ('xgb57', 'xgb52')),
    ('lr52_vs_lr57', ('lr57', 'lr52')),
])
NRI_PAIRS = ['lr52_vs_padua', 'xgb52_vs_padua', 'lr57_vs_padua',
             'xgb57_vs_padua', 'lr52_vs_improve', 'xgb52_vs_improve',
             'lr57_vs_improve', 'xgb57_vs_improve',
             'xgb52_vs_xgb57', 'lr52_vs_lr57']

subj_codes, subj_uniq = pd.factorize(pool['subject_id'])
ev_rows = np.where(y == 1)[0]
neg_rows = np.where(y == 0)[0]
ev_subj = subj_codes[ev_rows]
neg_subj = subj_codes[neg_rows]

bootstrap = {}
for seed in SEEDS:
    P = {'lr52': oof[('no_rx_52', 'lr', seed)],
         'xgb52': oof[('no_rx_52', 'xgb', seed)],
         'lr57': oof[('main_57', 'lr', seed)],
         'xgb57': oof[('main_57', 'xgb', seed)],
         'padua': padua_raw, 'improve': improve_raw}
    PP = {'padua_prob': padua_prob, 'improve_prob': improve_prob}
    rng = np.random.RandomState(seed)
    auc_boot = {k: [] for k in AUC_KEYS}
    deltas = {k: [] for k in DELTA_PAIRS}
    nri_boot = {(a, t): [] for a in NRI_PAIRS for t in NRI_THRESHOLDS}
    idi_boot = {a: [] for a in NRI_PAIRS}
    for b in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        ev_mult = counts[ev_subj]
        take = ev_mult > 0
        if take.sum() < 2:
            continue
        be = np.repeat(ev_rows[take], ev_mult[take])
        bn = neg_rows[counts[neg_subj] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        rows = np.concatenate([be, bn])
        yb = y[rows]
        aucs = {k: fast_auc(yb, P[k][rows]) for k in AUC_KEYS}
        for k in aucs:
            auc_boot[k].append(aucs[k])
        for name, (old, new) in DELTA_PAIRS.items():
            deltas[name].append(aucs[new] - aucs[old])
        for name in NRI_PAIRS:
            old_i, new_i = DELTA_PAIRS[name]
            po = PP['padua_prob'] if old_i == 'padua' else (
                PP['improve_prob'] if old_i == 'improve' else P[old_i])
            pn = PP['padua_prob'] if new_i == 'padua' else (
                PP['improve_prob'] if new_i == 'improve' else P[new_i])
            for t in NRI_THRESHOLDS:
                nri_boot[(name, t)].append(nri(po[rows], pn[rows], yb, t))
            idi_boot[name].append(idi(po[rows], pn[rows], yb))
        if (b + 1) % 500 == 0:
            print(f'  [seed {seed}] boot {b + 1}/{N_BOOT} ({el()})',
                  flush=True)

    def prob_pair(name):
        old_i, new_i = DELTA_PAIRS[name]

        def vec(i):
            return (PP['padua_prob'] if i == 'padua' else
                    PP['improve_prob'] if i == 'improve' else P[i])
        return vec(old_i), vec(new_i)

    bootstrap[str(seed)] = {
        'n_replicates_used': len(auc_boot['lr52']),
        'oof_auc_ci': {k: {'auc': float(fast_auc(y, P[k])),
                           'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
                       for k, v2 in auc_boot.items()},
        'delta_auc': {k: ds(v) for k, v in deltas.items()},
        'nri': {f'{name}_thr{t}': {'point': nri(*prob_pair(name), y, t),
                                   'ci': [float(v) for v in np.percentile(
                                       nri_boot[(name, t)], [2.5, 97.5])]}
                for name in NRI_PAIRS for t in NRI_THRESHOLDS},
        'idi': {name: {'point': idi(*prob_pair(name), y),
                       'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
                for name, v2 in idi_boot.items()},
    }
    for k in ['xgb52_vs_padua', 'xgb52_vs_improve', 'xgb52_vs_xgb57']:
        d = bootstrap[str(seed)]['delta_auc'][k]
        print(f'  [seed {seed}] dAUC {k}: {d["estimate"]:+.4f} '
              f'[{d["ci"][0]:+.4f},{d["ci"][1]:+.4f}] p={d["p_two_sided"]:.4f}',
              flush=True)
OUTOBJ['bootstrap'] = bootstrap
save()

# ── held-out test set ────────────────────────────────────────────────────
test_y = test['vte_event'].values.astype(int)
test_results = {'n': int(len(test)), 'events': int(test_y.sum()),
                'padua_auc': float(roc_auc_score(
                    test_y, score_col(test, 'padua_score'))),
                'improve_auc': float(roc_auc_score(
                    test_y, score_col(test, 'improve_score'))),
                'models': {}, 'importances': {}}
for set_name, feats in FEATURE_SETS.items():
    X = pool[feats].values.astype(np.float32)
    Xt = test[feats].values.astype(np.float32)
    row = {}
    imp = SimpleImputer(strategy='median')
    m = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
    m.fit(imp.fit_transform(X), y)
    row['lr_auc'] = float(roc_auc_score(
        test_y, m.predict_proba(imp.transform(Xt))[:, 1]))
    xm = XGBClassifier(**XGB_PARAMS).fit(X, y)
    row['xgb_auc'] = float(roc_auc_score(test_y, xm.predict_proba(Xt)[:, 1]))
    test_results['models'][set_name] = row
    gain = dict(zip(feats, xm.feature_importances_.tolist()))
    ranked = sorted(gain.items(), key=lambda kv: -kv[1])
    rank_of = {f: i + 1 for i, (f, _) in enumerate(ranked)}
    test_results['importances'][set_name] = {
        'rx_gain': {c: gain[c] for c in RX_DROP if c in gain},
        'rx_gain_rank': {c: rank_of[c] for c in RX_DROP if c in gain},
        'top10': [{'feature': f, 'gain': float(g)} for f, g in ranked[:10]],
        'gain_share_rx': (float(sum(gain[c] for c in RX_DROP if c in gain)
                                / sum(gain.values()))),
    }
    print(f'[test] {set_name}: LR={row["lr_auc"]:.4f} XGB={row["xgb_auc"]:.4f} '
          f'| rx gain share={test_results["importances"][set_name]["gain_share_rx"]:.4f}',
          flush=True)
OUTOBJ['test_set'] = test_results
save()

# ── heparin-exposed / unexposed subgroup OOF AUC (seed 42) ───────────────
heparin = pool['rx_heparin'].values.astype(int)
sub_groups = {'heparin': heparin == 1, 'no_heparin': heparin == 0}


def boot_ci_subgroup(mask, preds, rng):
    idx_pos = np.where(mask)[0][y[mask] == 1]
    idx_neg = np.where(mask)[0][y[mask] == 0]
    pos_subj, neg_subj_s = subj_codes[idx_pos], subj_codes[idx_neg]
    aucs = []
    for _ in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        mult = counts[pos_subj]
        take = mult > 0
        if take.sum() < 2:
            continue
        bp = np.repeat(idx_pos[take], mult[take])
        bn = idx_neg[counts[neg_subj_s] > 0]
        if len(bn) > NEG_CAP:
            bn = rng.choice(bn, NEG_CAP, replace=False)
        aucs.append(fast_auc(y[np.concatenate([bp, bn])],
                             preds[np.concatenate([bp, bn])]))
    return [float(v) for v in np.percentile(aucs, [2.5, 97.5])]


preds = {'lr52': oof[('no_rx_52', 'lr', SEED)],
         'xgb52': oof[('no_rx_52', 'xgb', SEED)],
         'lr57': oof[('main_57', 'lr', SEED)],
         'xgb57': oof[('main_57', 'xgb', SEED)],
         'padua': padua_raw}
subgroups = {'view': 'seed-42 5-fold pooled OOF on the train+val pool '
                     f'(n={N}, events={int(y.sum())})',
             'rows': {}}
for gname, mask in sub_groups.items():
    row = {'n': int(mask.sum()), 'events': int(y[mask].sum()),
           'event_rate_pct': round(100 * y[mask].mean(), 3)}
    for i, mname in enumerate(['padua', 'lr52', 'xgb52', 'lr57', 'xgb57']):
        pr = preds[mname]
        rng = np.random.RandomState(SEED + i)
        row[mname] = {'auc': float(fast_auc(y[mask], pr[mask])),
                      'ci': boot_ci_subgroup(mask, pr, rng)}
    for mname in ['lr52', 'xgb52', 'lr57', 'xgb57']:
        row[mname]['delta_vs_padua'] = row[mname]['auc'] - row['padua']['auc']
    subgroups['rows'][gname] = row
    print(f'[subgroup {gname}] n={row["n"]} ev={row["events"]} | '
          f'xgb52={row["xgb52"]["auc"]:.4f} '
          f'[{row["xgb52"]["ci"][0]:.4f},{row["xgb52"]["ci"][1]:.4f}] | '
          f'xgb57={row["xgb57"]["auc"]:.4f}', flush=True)
OUTOBJ['subgroups'] = subgroups
save()

# ── descriptive univariate association of the dropped columns ────────────
rx_desc = {}
for c in RX_DROP:
    m = pool[c].fillna(0).values.astype(int) == 1
    n1, n0 = int(m.sum()), int((~m).sum())
    e1, e0 = int(y[m].sum()), int(y[~m].sum())
    p1, p0 = e1 / n1 if n1 else np.nan, e0 / n0 if n0 else np.nan
    a_, b_ = e1 + 0.5, n1 - e1 + 0.5
    c_, d_ = e0 + 0.5, n0 - e0 + 0.5
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    or_ = (a_ * d_) / (b_ * c_)
    rx_desc[c] = {
        'n_exposed': n1, 'pct_exposed': round(100 * n1 / N, 2),
        'events_exposed': e1, 'events_unexposed': e0,
        'event_rate_exposed_pct': round(100 * p1, 4),
        'event_rate_unexposed_pct': round(100 * p0, 4),
        'risk_difference_pct': round(100 * (p1 - p0), 4),
        'odds_ratio_haldane': round(float(or_), 3),
        'odds_ratio_ci95': [round(float(np.exp(np.log(or_) - 1.96 * se)), 3),
                            round(float(np.exp(np.log(or_) + 1.96 * se)), 3)],
        'mean_oof_xgb52_exposed': round(float(preds['xgb52'][m].mean()), 5),
        'mean_oof_xgb52_unexposed': round(float(preds['xgb52'][~m].mean()), 5),
        'median_oof_xgb52_exposed': round(float(np.median(preds['xgb52'][m])), 5),
        'median_oof_xgb52_unexposed': round(float(np.median(preds['xgb52'][~m])), 5),
        'mean_oof_xgb57_exposed': round(float(preds['xgb57'][m].mean()), 5),
        'mean_oof_xgb57_unexposed': round(float(preds['xgb57'][~m].mean()), 5),
    }
    print(f'[rx] {c}: exposed {n1} ({100*n1/N:.2f}%), '
          f'event rate {100*p1:.3f}% vs {100*p0:.3f}%, '
          f'OR {or_:.3f} [{rx_desc[c]["odds_ratio_ci95"][0]},'
          f'{rx_desc[c]["odds_ratio_ci95"][1]}]', flush=True)

OUTOBJ['rx_descriptives'] = rx_desc
OUTOBJ['feature_sets'] = {'main_57': MAIN, 'no_rx_52': NORX,
                          'dropped': RX_DROP}
OUTOBJ['meta'] = {
    'cohort': 'inclprior_excl24h primary',
    'dataset': DATASET, 'splits': SPLITS,
    'n_pool': N, 'n_events': int(y.sum()), 'n_subjects': int(n_subj),
    'n_test': int(len(test)), 'n_test_events': int(test_y.sum()),
    'seeds': SEEDS,
    'protocol': 'script-98 static protocol: LR L2 C=1.0 per-fold '
                'training-median imputation; XGB 300/3/0.05/0.8/0.8/5/1 '
                'native NaN; 3x5 StratifiedGroupKFold (stratify '
                'vte_event, group subject_id); fold bootstrap 2000 for '
                'fold-mean AUC CI; seed-42/43/44 OOF; patient-level '
                'cluster bootstrap 2000 reps with negatives downsampled '
                'to 50k for dAUC/NRI/IDI; test-set models refit on the '
                'full train+val pool with seed 42',
    'seed_convention': 'XGB random_state=seed per repeat (the convention of '
                       'the published multi-seed OOF table, script 61); '
                       'script 98 instead fixes random_state=42 for all three '
                       'repeats - the script-98-convention 15-fold means are '
                       'reported separately in '
                       'output_era/no_rx_foldmean_98convention_inclprior_excl24h.json '
                       '(XGB 57-feature 0.8088 reproduces the published '
                       '0.80880 exactly; 52-feature 0.8077)',
    'caveat_rx_hormone_padua': 'rx_hormone is an item of the Padua score '
        '(scripts/85e_clinical_scores_v3.py:40, "ongoing hormonal '
        'treatment", 1 point) and is also one of the 5 columns dropped '
        'here. The Padua comparator therefore retains a treatment-related '
        'predictor that the 52-feature models no longer see; IMPROVE uses '
        'no treatment column. This asymmetry must be stated wherever the '
        '52-feature dAUC vs Padua is reported. Padua/IMPROVE computations '
        'were not modified.',
    'runtime_s': round(time.time() - T0, 1),
}
save()
print(f'Saved -> {OUT} ({el()})', flush=True)
