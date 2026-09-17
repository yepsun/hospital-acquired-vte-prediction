#!/usr/bin/env python3
"""Script 39c: test-set baseline evaluation incl. the Caprini score.

Identical protocol to scripts_era/39b_baseline_scores_eval_v2.py (Padua /
IMPROVE on the era primary-cohort test split), extended with Caprini and with
IMPROVE-DD added for completeness of the JSON.

Reads:  results_vte/ajm/new/data_era/primary_inclprior_excl24h_model_dataset_caprini.parquet
        results_vte/ajm/new/data_era/primary_inclprior_excl24h_splits.json
Writes: results_vte/ajm/new/output_era/baseline_results_caprini_inclprior_excl24h.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

D = 'results_vte/ajm/new/data_era'
OUT = 'results_vte/ajm/new/output_era/baseline_results_caprini_inclprior_excl24h.json'

splits = json.load(open(f'{D}/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(
    f'{D}/primary_inclprior_excl24h_model_dataset_caprini.parquet')

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
            'n_high_risk': int(flag.sum()), 'pct_high_risk': float(flag.mean())}


def eval_score(name, score_col, high_col):
    s = test[score_col].values.astype(float)
    auc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(s.reshape(-1, 1), y)
    p = lr.predict_proba(s.reshape(-1, 1))[:, 1]
    brier = brier_score_loss(y, p)
    res = {'test_auc': float(auc), 'test_auprc': float(auprc),
           'brier': float(brier),
           'brier_method': 'score calibrated to probability via univariate '
                           'logistic regression fit on the test set (in-sample)',
           'auprc_baseline_prevalence': float(y.mean()),
           'high_risk': binary_metrics(y, test[high_col].values)}
    print(f"{name}: AUC={auc:.4f} AUPRC={auprc:.4f} Brier={brier:.5f} | "
          f"high-risk sens={res['high_risk']['sensitivity']:.3f} "
          f"spec={res['high_risk']['specificity']:.3f} "
          f"PPV={res['high_risk']['ppv']:.4f} NPV={res['high_risk']['npv']:.4f}",
          flush=True)
    return res


out = {'n_test': N, 'n_events': events,
       'test_event_rate': float(y.mean()),
       'cohort': 'era primary inclprior_excl24h (2008-2019)',
       'padua': eval_score('Padua', 'padua_score', 'padua_high'),
       'improve': eval_score('IMPROVE', 'improve_score', 'improve_high'),
       'improve_dd': eval_score('IMPROVE-DD', 'improve_dd', 'improve_dd_high'),
       'caprini': eval_score('Caprini', 'caprini_score', 'caprini_high')}

with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f'Saved -> {OUT}', flush=True)
