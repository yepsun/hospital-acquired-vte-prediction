#!/usr/bin/env python3
"""
Script 68 (era / correctness fix, companion to 65): robustness metrics for the
nested recalibration, computed from the persisted OOF arrays.

The pre-specified ECE uses 10 EQUAL-WIDTH bins. At 0.177% prevalence the
machine-learning probabilities occupy only the first 1-4 bins, so the tail bins
contain a handful of rows and the equal-width ECE is dominated by them. This
companion script reports, for the same OOF arrays:

  - ECE 10 equal-width bins WITH bin counts and observed/predicted per bin
  - ECE 10 equal-count (quantile) bins, both unweighted and count-weighted
  - Brier score
  - reliability deciles formed on the pre-recalibration score (Figure 2 layout)
  - a Figure 2-style reliability plot for uncalibrated / apparent / nested

Reads:  output_era/nested_calib_oof_inputs_inclprior_excl24h.npz
Writes: output_era/nested_calibration_metrics_extra_inclprior_excl24h.json
        output_era/Figure_2_calibration_nested.png
Run from the repository root that contains results_vte/.
"""
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings('ignore')
RES = 'results_vte'
SEEDS = [42, 43, 44]
NPZ = f'{RES}/ajm/new/output_era/nested_calib_oof_inputs_inclprior_excl24h.npz'
OUT = f'{RES}/ajm/new/output_era/nested_calibration_metrics_extra_inclprior_excl24h.json'
FIG = f'{RES}/ajm/new/output_era/Figure_2_calibration_nested.png'

z = np.load(NPZ)
y = z['y'].astype(int)


def uniform_bins(yb, p, bins=10):
    """Binning identical to sklearn.calibration_curve(strategy='uniform'):
    bin k holds p with edges[k-1] < p <= edges[k] (searchsorted, side='left')."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.searchsorted(edges[1:-1], p)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({'lo': float(edges[b]), 'hi': float(edges[b + 1]),
                     'n': int(m.sum()),
                     'mean_predicted': float(p[m].mean()),
                     'observed_fraction': float(yb[m].mean()),
                     'abs_gap': float(abs(p[m].mean() - yb[m].mean()))})
    return rows


def ece_uniform(yb, p, bins=10):
    rows = uniform_bins(yb, p, bins)
    return float(np.mean([r['abs_gap'] for r in rows]))


def quantile_bins(yb, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({'n': int(m.sum()),
                     'mean_predicted': float(p[m].mean()),
                     'observed_fraction': float(yb[m].mean()),
                     'abs_gap': float(abs(p[m].mean() - yb[m].mean()))})
    return rows


def ece_quantile(yb, p, bins=10):
    rows = quantile_bins(yb, p, bins)
    unweighted = float(np.mean([r['abs_gap'] for r in rows]))
    tot = sum(r['n'] for r in rows)
    weighted = float(sum(r['abs_gap'] * r['n'] for r in rows) / tot)
    return unweighted, weighted


def cal_slope_intercept(yb, p):
    """Cox calibration: logistic regression of y on logit(p). Slope 1 / intercept 0
    = perfect calibration. Epsilon-clipping keeps the logit finite."""
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    m = LogisticRegression(C=1e6, max_iter=5000).fit(lp.reshape(-1, 1), yb)
    return {'slope': float(m.coef_.ravel()[0]),
            'intercept': float(m.intercept_[0])}


def decile_curve(yb, p_plot, p_order, bins=10):
    edges = np.quantile(p_order, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(p_order, edges[1:-1], right=False), 0, bins - 1)
    mp, of, nn = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 50:
            mp.append(float(p_plot[m].mean()))
            of.append(float(yb[m].mean()))
            nn.append(int(m.sum()))
    return {'bin_mean_predicted': mp, 'bin_observed_fraction': of, 'bin_n': nn}


out = {'note': 'Companion robustness metrics for 65_nested_calibration_dca_fixed.py; '
               'persisted OOF arrays, no refitting.',
       'n_pool': int(len(y)), 'n_events': int(y.sum()), 'seeds': {}}
for s in SEEDS:
    p_raw = {k: z[f'{k}_seed{s}'] for k in ['lr', 'xgb']}
    p_app = {k: z[f'{k}_apparent_seed{s}'] for k in ['lr', 'xgb']}
    p_nes = {k: z[f'{k}_nested_seed{s}'] for k in ['lr', 'xgb']}
    p_pad_app = z[f'padua_apparent_seed{s}']
    p_pad_nes = z[f'padua_nested_seed{s}']
    blk = {}
    for kind in ['lr', 'xgb']:
        variants = {'uncalibrated': (p_raw[kind], p_raw[kind]),
                    'apparent_isotonic': (p_app[kind], p_raw[kind]),
                    'nested_isotonic': (p_nes[kind], p_raw[kind])}
        blk[kind] = {}
        for tag, (p, order) in variants.items():
            q_un, q_w = ece_quantile(y, p)
            blk[kind][tag] = {
                'brier': float(brier_score_loss(y, p)),
                'auc': float(roc_auc_score(y, p)),
                'ece_uniform10': ece_uniform(y, p),
                'ece_uniform10_bins': uniform_bins(y, p),
                'ece_quantile10_unweighted': q_un,
                'ece_quantile10_weighted': q_w,
                'cox_calibration': cal_slope_intercept(y, p),
                'reliability_deciles': decile_curve(y, p, order)}
        print(f"seed {s} {kind.upper()}: " + ' | '.join(
            f"{t}: Brier={blk[kind][t]['brier']:.5f} "
            f"ECEuw={blk[kind][t]['ece_uniform10']:.4f} "
            f"ECEqu={blk[kind][t]['ece_quantile10_unweighted']:.4f}"
            for t in variants), flush=True)
    blk['padua'] = {}
    for tag, p in [('apparent_platt_pool', p_pad_app),
                   ('nested_platt_crossfitted', p_pad_nes)]:
        q_un, q_w = ece_quantile(y, p)
        blk['padua'][tag] = {
            'brier': float(brier_score_loss(y, p)),
            'auc': float(roc_auc_score(y, p)),
            'ece_uniform10': ece_uniform(y, p),
            'ece_uniform10_bins': uniform_bins(y, p),
            'ece_quantile10_unweighted': q_un,
            'ece_quantile10_weighted': q_w,
            'cox_calibration': cal_slope_intercept(y, p),
            'reliability_deciles': decile_curve(y, p, p_pad_app)}
    out['seeds'][str(s)] = blk

json.dump(out, open(OUT, 'w'), indent=2)
print('saved ->', OUT, flush=True)

# ── Figure 2-style reliability plot (seed 42) ──────────────────────────────
s = 42
blk = out['seeds'][str(s)]
plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#dcdcdc', 'grid.linewidth': 0.55})
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharex=True, sharey=True)
specs = [
    ('Padua (Platt)', [('padua', 'apparent_platt_pool', 'Platt (pool)', '#636363', 's', (0, (3, 2))),
                       ('padua', 'nested_platt_crossfitted', 'Platt (cross-fitted)', '#636363', 's', 'solid')]),
    ('Logistic regression', [('lr', 'uncalibrated', 'Uncalibrated', '#1a6fb5', 'o', (0, (3, 2))),
                             ('lr', 'apparent_isotonic', 'Isotonic (apparent)', '#1a6fb5', 'o', ':'),
                             ('lr', 'nested_isotonic', 'Isotonic (nested)', '#1a6fb5', 'o', 'solid')]),
    ('XGBoost', [('xgb', 'uncalibrated', 'Uncalibrated', '#c22e38', '^', (0, (3, 2))),
                 ('xgb', 'apparent_isotonic', 'Isotonic (apparent)', '#c22e38', '^', ':'),
                 ('xgb', 'nested_isotonic', 'Isotonic (nested)', '#c22e38', '^', 'solid')])]
xmax = 0.9
for j, (ttl, series) in enumerate(specs):
    ax = axes[j]
    ax.plot([0, xmax], [0, xmax], color='#999999', ls='--', lw=1.0, label='Ideal', zorder=1)
    for kind, tag, lab, c, mkr, ls_ in series:
        d = blk[kind][tag]['reliability_deciles']
        ax.plot(np.array(d['bin_mean_predicted']) * 100,
                np.array(d['bin_observed_fraction']) * 100, mkr + '-', color=c,
                ms=4.8, lw=1.6, ls=ls_, mec=c, mew=1.2, label=lab, zorder=3)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-0.04, 0.84)
    ax.set_xlabel('Mean predicted VTE risk per decile (%)')
    if j == 0:
        ax.set_ylabel('Observed VTE fraction (%)')
    ax.legend(frameon=False, fontsize=7.8, loc='upper left', handlelength=2.2)
    ax.set_title(f"({'abc'[j]}) {ttl}", fontsize=9.5, loc='left', pad=5)
fig.tight_layout(w_pad=1.6)
fig.savefig(FIG, dpi=300)
plt.close(fig)
print('saved ->', FIG, flush=True)
