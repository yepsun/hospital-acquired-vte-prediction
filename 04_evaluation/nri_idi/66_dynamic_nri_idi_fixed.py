#!/usr/bin/env python3
"""
Script 66 (era / correctness fix): dynamic OOF bootstrap with the CORRECT
category-based NRI sign.

Bug
---
47b_bootstrap_only.py (line 27-32) and its _72h twin compute

    NRI = (up_e - dn_e)/n_events + (up_n - dn_n)/n_non-events

i.e. the non-event component with the sign reversed, whereas the static
script scripts/98_ajm_sensitivity_cohorts.py (and 47b_dynamic_models_v2.py,
which computes the same quantity inline) use the standard Pencina
category-based definition

    NRI = [P(up|event) - P(down|event)] + [P(down|non-event) - P(up|non-event)]

The published dynamic NRI values in
output_era/dynamic_oof_inference_inclprior_excl24h.json are therefore
inflated. IDI is unaffected.

This script reads the SAME persisted inputs
(output_era/dynamic_oof_inputs_inclprior_excl24h.npz, 1,044,833 landmark rows /
2,742 events) and re-runs the identical 2,000-replicate patient-level cluster
bootstrap, emitting the corrected NRI (and the old buggy value in the same run
for an exact side-by-side). The 72h twin is handled with --npz/--out.

Outputs (new filenames only):
  results_vte/ajm/new/output_era/dynamic_oof_inference_inclprior_excl24h_nri_fixed.json
  results_vte/ajm/new/output_era/dynamic_oof_inference_72h_inclprior_excl24h_nri_fixed.json
Run from the repository root that contains results_vte/.
"""
import argparse
import json
import time

import numpy as np
from scipy.stats import rankdata

N_BOOT = 2000
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]
SEED = 42
RES = 'results_vte'

ap = argparse.ArgumentParser()
ap.add_argument('--npz', default=f'{RES}/ajm/new/output_era/dynamic_oof_inputs_inclprior_excl24h.npz')
ap.add_argument('--out', default=f'{RES}/ajm/new/output_era/dynamic_oof_inference_inclprior_excl24h_nri_fixed.json')
ap.add_argument('--arm', default='primary (48h landmark, inclprior_excl24h)')
A = ap.parse_args()


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nri(p_old, p_new, yb, thr):
    """Standard category-based NRI (Pencina 2008): events up - down,
    non-events down - up."""
    ev, nev = yb == 1, yb == 0
    up_e = ((p_new >= thr) & (p_old < thr) & ev).sum() / ev.sum()
    dn_e = ((p_new < thr) & (p_old >= thr) & ev).sum() / ev.sum()
    dn_n = ((p_new < thr) & (p_old >= thr) & nev).sum() / nev.sum()
    up_n = ((p_new >= thr) & (p_old < thr) & nev).sum() / nev.sum()
    return float((up_e - dn_e) + (dn_n - up_n))


def nri_buggy(p_old, p_new, yb, thr):
    """Reproduces 47b_bootstrap_only.py (non-event component sign reversed)."""
    ev, nev = yb == 1, yb == 0
    up_e = ((p_new >= thr) & (p_old < thr) & ev).sum() / ev.sum()
    dn_e = ((p_new < thr) & (p_old >= thr) & ev).sum() / ev.sum()
    up_n = ((p_new >= thr) & (p_old < thr) & nev).sum() / nev.sum()
    dn_n = ((p_new < thr) & (p_old >= thr) & nev).sum() / nev.sum()
    return float((up_e - dn_e) + (up_n - dn_n))


def nri_components(p_old, p_new, yb, thr):
    ev, nev = yb == 1, yb == 0
    return {'events_up': float(((p_new >= thr) & (p_old < thr) & ev).sum() / ev.sum()),
            'events_down': float(((p_new < thr) & (p_old >= thr) & ev).sum() / ev.sum()),
            'nonevents_down': float(((p_new < thr) & (p_old >= thr) & nev).sum() / nev.sum()),
            'nonevents_up': float(((p_new >= thr) & (p_old < thr) & nev).sum() / nev.sum())}


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


z = np.load(A.npz)
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
AUC_PAIRS = {'lr_vs_padua': ('padua', 'lr'),
             'xgb_vs_padua': ('padua', 'xgb'),
             'xgb_vs_static': ('static', 'xgb')}
deltas = {k: [] for k in PAIRS}
auc_boot = {k: [] for k in ['lr', 'xgb', 'padua', 'static']}
nri_boot = {(a, t): [] for a in PAIRS for t in NRI_THRESHOLDS}
idi_boot = {a: [] for a in PAIRS}

t0 = time.time()
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
    for name, (old, new) in PAIRS.items():
        oa, na = AUC_PAIRS[name]
        deltas[name].append(aucs[na] - aucs[oa])
        for t in NRI_THRESHOLDS:
            nri_boot[(name, t)].append(nri(P[old][rows], P[new][rows], yb, t))
        idi_boot[name].append(idi(P[old][rows], P[new][rows], yb))
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)

oof_auc_ci = {k: {'auc': float(fast_auc(yE, P[k])),
                  'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
              for k, v2 in auc_boot.items()}
out = {
    'arm': A.arm,
    'source_npz': A.npz,
    'n_rows': int(len(yE)), 'n_events': int(yE.sum()), 'n_subjects': int(n_subj),
    'bug': "47b_bootstrap_only.py used NRI = (up_e-dn_e) + (up_n-dn_n); "
           "non-event component sign reversed. Corrected here to the standard "
           "Pencina definition (up_e-dn_e) + (dn_n-up_n).",
    'n_replicates_used': len(auc_boot['lr']),
    'oof_auc_ci': oof_auc_ci,
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
    'nri': {f'{name}_thr{t}': {'point': nri(P[old], P[new], yE, t),
                               'point_old_buggy': nri_buggy(P[old], P[new], yE, t),
                               'components': nri_components(P[old], P[new], yE, t),
                               'ci': [float(v) for v in np.percentile(
                                   nri_boot[(name, t)], [2.5, 97.5])]}
            for name, (old, new) in PAIRS.items() for t in NRI_THRESHOLDS},
    'idi': {name: {'point': idi(P[old], P[new], yE),
                   'ci': [float(v) for v in np.percentile(v2, [2.5, 97.5])]}
            for name, (old, new), v2 in
            [(n, p, idi_boot[n]) for n, p in PAIRS.items()]},
}
json.dump(out, open(A.out, 'w'), indent=2)
print('saved ->', A.out, f'({time.time()-t0:.0f}s)', flush=True)
for k, d in out['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}", flush=True)
for k, d in out['nri'].items():
    print(f"  NRI {k}: correct={d['point']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] (old buggy {d['point_old_buggy']:+.4f})",
          flush=True)
for k, d in out['idi'].items():
    print(f"  IDI {k}: {d['point']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]", flush=True)
