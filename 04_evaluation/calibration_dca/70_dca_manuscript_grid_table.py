#!/usr/bin/env python3
"""
Script 70 (era / correctness fix): manuscript-style DCA table on the
manuscript's own grid, for every row-set x calibration combination.

Reads the persisted arrays only (no refitting):
  output_era/nested_calib_oof_inputs_inclprior_excl24h.npz   (pool OOF, 3 seeds)
  output_era/nested_calib_test_probs_inclprior_excl24h.npz   (test predictions)

Grid/aggregation is 41b's (the source of the manuscript's Table S9 numbers):
  grid = linspace(0.001, 0.05, 100); band = 0.005 <= pt <= 0.02 (30 points);
  band value = unweighted mean; "at 0.5%/1%/2%" = nearest grid points 8/18/38.

Output: results_vte/ajm/new/output_era/dca_manuscript_grid_table_inclprior_excl24h.json
Run from the repository root that contains results_vte/.
"""
import json

import numpy as np

RES = 'results_vte'
SEEDS = [42, 43, 44]
GRID = np.linspace(0.001, 0.05, 100)
BAND = (GRID >= 0.005) & (GRID <= 0.02)
IDX = {'0.5%': 8, '1%': 18, '2%': 38}
OUT = f'{RES}/ajm/new/output_era/dca_manuscript_grid_table_inclprior_excl24h.json'

pool = np.load(f'{RES}/ajm/new/output_era/nested_calib_oof_inputs_inclprior_excl24h.npz')
test = np.load(f'{RES}/ajm/new/output_era/nested_calib_test_probs_inclprior_excl24h.npz')
y = pool['y'].astype(int)
yt = test['y'].astype(int)


def nb(yb, p, pt):
    flag = p >= pt
    tp = (flag & (yb == 1)).sum()
    fp = (flag & (yb == 0)).sum()
    return float(tp / len(yb) - fp / len(yb) * pt / (1 - pt))


def block(yb, probs, label):
    arr = {m: np.array([nb(yb, p, t) for t in GRID]) for m, p in probs.items()}
    prev = yb.mean()
    d = {'label': label, 'n_rows': int(len(yb)), 'n_events': int(yb.sum()),
         'mean_nb_band_per1000': {m: float(v[BAND].mean() * 1e3) for m, v in arr.items()},
         'mean_nb_band_per_admission': {m: float(v[BAND].mean()) for m, v in arr.items()},
         'nb_per1000_at_nearest_grid_point': {k: {m: float(v[i] * 1e3) for m, v in arr.items()}
                                             for k, i in IDX.items()},
         'nb_per_admission_at_nearest_grid_point': {k: {m: float(v[i]) for m, v in arr.items()}
                                                   for k, i in IDX.items()},
         'treat_all_per1000_at_nearest_grid_point': {
             k: float((prev - (1 - prev) * GRID[i] / (1 - GRID[i])) * 1e3) for k, i in IDX.items()}}
    return d


out = {'grid': 'np.linspace(0.001, 0.05, 100); band 0.005-0.02 (30 pts), unweighted mean',
       'blocks': {}}
for s in SEEDS:
    out['blocks'][f'pool_oof_apparent_seed{s}'] = block(
        y, {'padua': pool[f'padua_apparent_seed{s}'],
            'lr': pool[f'lr_apparent_seed{s}'], 'xgb': pool[f'xgb_apparent_seed{s}']},
        f'development pool OOF, isotonic fitted AND evaluated on the same OOF (apparent), seed {s}')
    out['blocks'][f'pool_oof_nested_seed{s}'] = block(
        y, {'padua': pool[f'padua_nested_seed{s}'],
            'lr': pool[f'lr_nested_seed{s}'], 'xgb': pool[f'xgb_nested_seed{s}']},
        f'development pool OOF, cross-fitted isotonic / cross-fitted Platt (nested), seed {s}')
    out['blocks'][f'test_seed{s}'] = block(
        yt, {'padua': test['padua_prob_pool_platt'],
             'lr': test[f'lr_cal_seed{s}'], 'xgb': test[f'xgb_cal_seed{s}']},
        f'held-out test partition, calibrators fitted on the development pool OOF, seed {s} '
        f'(= 41b recipe, the source of the manuscript numbers)')

json.dump(out, open(OUT, 'w'), indent=2)
print('saved ->', OUT, flush=True)

order = [k for s in SEEDS for k in (f'pool_oof_apparent_seed{s}', f'pool_oof_nested_seed{s}',
                                    f'test_seed{s}')]
print('\n%-34s %14s %8s %8s %8s' % ('block', 'rows/events', 'padua', 'lr', 'xgb'))
for k in order:
    b = out['blocks'][k]
    print('%-34s %14s %8.4f %8.4f %8.4f' % (
        k, f"{b['n_rows']}/{b['n_events']}",
        b['mean_nb_band_per1000']['padua'], b['mean_nb_band_per1000']['lr'],
        b['mean_nb_band_per1000']['xgb']))
print('\n(per-admission: divide by 1000)')
print('seed ranges, nested pool OOF band per 1000:',
      {m: [min(out['blocks'][f'pool_oof_nested_seed{s}']['mean_nb_band_per1000'][m] for s in SEEDS),
           max(out['blocks'][f'pool_oof_nested_seed{s}']['mean_nb_band_per1000'][m] for s in SEEDS)]
       for m in ['padua', 'lr', 'xgb']})
print('seed ranges, test band per 1000:',
      {m: [min(out['blocks'][f'test_seed{s}']['mean_nb_band_per1000'][m] for s in SEEDS),
           max(out['blocks'][f'test_seed{s}']['mean_nb_band_per1000'][m] for s in SEEDS)]
       for m in ['padua', 'lr', 'xgb']})
