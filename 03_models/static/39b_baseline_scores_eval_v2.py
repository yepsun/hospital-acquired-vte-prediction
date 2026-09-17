#!/usr/bin/env python3
"""
Script 39 (VTE M4 / Task 2): clinical score baseline evaluation on the test set.

Evaluates Padua and IMPROVE scores as continuous risk scores on the held-out
test split (results_vte/splits.json, 'test'):
  - AUC, AUPRC (raw score as continuous predictor)
  - Brier: score calibrated to probability via univariate logistic regression
    fit ON THE TEST SET (in-sample calibration, noted in output as method;
    optimism is minimal given N~63k)
  - high-risk binary flags (padua_high / improve_high, >=4 threshold):
    sensitivity / specificity / PPV / NPV

Outputs: results_vte/baseline_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RES = 'results_vte'

fs = json.load(open(f'{RES}/feature_sets_v2.json'))
splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')

test = df[df['hadm_id'].isin(set(splits['test']))]
y = test['vte_event'].values.astype(int)
N, events = len(y), int(y.sum())
print(f'test N={N} events={events} ({events / N * 100:.3f}%)', flush=True)


def binary_metrics(yb, flag):
    flag = flag.astype(bool)
    tp = int(((flag) & (yb == 1)).sum()); fp = int(((flag) & (yb == 0)).sum())
    fn = int(((~flag) & (yb == 1)).sum()); tn = int(((~flag) & (yb == 0)).sum())
    return {'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'ppv': tp / (tp + fp) if (tp + fp) else float('nan'),
            'npv': tn / (tn + fn) if (tn + fn) else float('nan'),
            'n_high_risk': int(flag.sum())}


def eval_score(name, score_col, high_col):
    s = test[score_col].values.astype(float)
    auc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    # univariate logistic calibration of the score to probability (fit on test)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(s.reshape(-1, 1), y)
    p = lr.predict_proba(s.reshape(-1, 1))[:, 1]
    brier = brier_score_loss(y, p)
    res = {'test_auc': float(auc), 'test_auprc': float(auprc),
           'brier': float(brier),
           'brier_method': 'score calibrated to probability via univariate logistic '
                           'regression fit on the test set (in-sample)',
           'high_risk': binary_metrics(y, test[high_col].values)}
    print(f"{name}: AUC={auc:.4f} AUPRC={auprc:.4f} Brier={brier:.5f} | "
          f"high-risk sens={res['high_risk']['sensitivity']:.3f} "
          f"spec={res['high_risk']['specificity']:.3f} "
          f"PPV={res['high_risk']['ppv']:.4f} NPV={res['high_risk']['npv']:.4f}", flush=True)
    return res


out = {'n_test': N, 'n_events': events,
       'padua': eval_score('Padua', 'padua_score', 'padua_high'),
       'improve': eval_score('IMPROVE', 'improve_score', 'improve_high')}

# sanity check per plan: Padua AUC expected 0.60-0.70
padua_auc = out['padua']['test_auc']
if padua_auc < 0.55:
    print('WARNING: Padua AUC < 0.55 — re-check score computation', flush=True)

with open(f'{RES}/ajm/new/output_era/baseline_results_inclprior_excl24h.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved -> results_vte/baseline_results.json', flush=True)
