#!/usr/bin/env python3
"""Standalone patient-level cluster bootstrap for the dynamic-model eval view.

Reads dynamic_oof_inputs_72h_inclprior_excl24h.npz (persisted by 47b) and reproduces the exact
2000-replicate inference in scripts/47b, so a long-running job cannot be
killed mid-loop. Writes the oof_inference block merged into a small json.
"""
import json
import numpy as np
from scipy.stats import rankdata

N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]
SEED = 42
RES = 'results_vte'
NPZ = f'{RES}/ajm/new/output_era/dynamic_oof_inputs_72h_inclprior_excl24h.npz'
OUT = f'{RES}/ajm/new/output_era/dynamic_oof_inference_72h_inclprior_excl24h.json'


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nri(p_old, p_new, yb, thr):
    # SIGN FIX (2026-09-17). The non-event component previously entered with
    # the opposite sign: the code added (up_n - dn_n) instead of (dn_n - up_n).
    # Because the non-event component is large and negative at the sub-1%
    # thresholds used here, that error roughly doubled NRI (e.g. XGBoost vs
    # Padua at 0.5%: 1.3389 instead of 0.3851). The manuscript reports the
    # corrected, standard Pencina (2008) category-based definition
    #   [P(up|event) - P(down|event)] + [P(down|non-event) - P(up|non-event)],
    # matching scripts/98_ajm_sensitivity_cohorts.py. Re-running this script
    # now regenerates the corrected values; the previously published (inflated)
    # JSON is preserved unchanged, and the corrected run is checkpointed in
    # dynamic_oof_inference_72h_inclprior_excl24h_nri_fixed.json.
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


z = np.load(NPZ)
yE = z['yE'].astype(int)
subj_codes = z['subj_codes']
P = {'lr': z['lr'], 'xgb': z['xgb'], 'padua': z['padua_raw'],
     'padua_prob': z['padua_prob'], 'static': z['static']}
print('loaded', {k: v.shape for k, v in P.items()}, 'events', int(yE.sum()), flush=True)

subj_uniq, subj_inv = np.unique(subj_codes, return_inverse=True)
n_subj = len(subj_uniq)
ev_mask = yE == 1
ev_rows = np.where(ev_mask)[0]
neg_rows = np.where(~ev_mask)[0]
ev_subj = subj_inv[ev_mask]
neg_subj = subj_inv[~ev_mask]

PAIRS = {'lr_vs_padua': ('padua_prob', 'lr'),
         'xgb_vs_padua': ('padua_prob', 'xgb'),
         'xgb_vs_static': ('static', 'xgb')}
deltas = {k: [] for k in PAIRS}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua', 'static']}
nri_boot = {(a, t): [] for a in PAIRS for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in PAIRS}

rng = np.random.RandomState(SEED)
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
    yb = yE[rows]
    aucs = {k: fast_auc(yb, P[k][rows]) for k in auc_boot}
    for k in aucs:
        auc_boot[k].append(aucs[k])
    for name in deltas:
        old, new = {'lr_vs_padua': ('padua', 'lr'),
                    'xgb_vs_padua': ('padua', 'xgb'),
                    'xgb_vs_static': ('static', 'xgb')}[name]
        deltas[name].append(aucs[new] - aucs[old])
    for name, (old, new) in PAIRS.items():
        for t in NRI_THRESHOLDS:
            nri_boot[(name, t)].append(nri(P[old][rows], P[new][rows], yb, t))
        idi_boot[name].append(idi(P[old][rows], P[new][rows], yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT}', flush=True)

oof_auc_ci = {k: {'auc': float(fast_auc(yE, P[k])),
                  'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
              for k, v2 in auc_boot.items()}
out = {
    'n_replicates_used': len(auc_boot['lr']),
    'oof_auc_ci': oof_auc_ci,
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
    'nri': {f'{name}_thr{t}': {'point': nri(P[old], P[new], yE, t),
                               'ci': [float(v) for v in np.percentile(
                                   nri_boot[(name, t)], [2.5, 97.5])]}
            for name, (old, new) in PAIRS.items() for t in NRI_THRESHOLDS},
    'idi': {name: {'point': idi(P[old], P[new], yE),
                   'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
            for name, (old, new), v2 in
            [(n, p, idi_boot[n]) for n, p in PAIRS.items()]},
}
json.dump(out, open(OUT, 'w'), indent=2)
print('saved ->', OUT)
for k, d in out['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}")
