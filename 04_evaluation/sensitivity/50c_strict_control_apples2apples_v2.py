#!/usr/bin/env python3
"""Decisive apples-to-apples test for the control-definition concern.

Re-runs the s4 strict-control subset construction (same definition as
50b_sensitivity_v2.py s4): controls = v2 non-events with NO active VTE ICD,
no positive/uncertain VTE mention, AND VTE imaging (CTPA / lower-extremity
duplex/CTV) performed with a clean report. Events unchanged (1,623).

Then evaluates Padua, IMPROVE, LR and XGBoost AUCs on the SAME strict
control subset (same rows, same StratifiedGroupKFold), so the score-vs-ML
comparison is like-for-like. Also reports the same four models on the MAIN
control pool on the same rows/folds for reference.
"""
import csv
import gzip
import json
import re
import sqlite3
import sys

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

RES = 'results_vte'
SEED = 42
N_SPLITS = 5
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1, eval_metric='logloss', scale_pos_weight=1,
                  tree_method='hist', n_jobs=-1, random_state=SEED)

# ICD sets
ICD9_PE = json.load(open(f'{RES}/pe_dynamic_config_icd9.json')) if False else None


def load_icd_vte():
    from pe_dynamic.config import ICD9_PE, ICD10_PE
    con = sqlite3.connect('mimic4.db')
    dx = pd.read_sql('SELECT hadm_id, icd_code, icd_version FROM diagnoses_icd', con)
    con.close()
    code = dx.icd_code.str.replace('.', '', regex=False).str.upper()
    is_pe = ((dx.icd_version == 9) & code.isin(ICD9_PE)) | \
            ((dx.icd_version == 10) & code.isin(ICD10_PE))
    dvt10 = code.str.startswith('I824') | code.isin({'I801', 'I802', 'I803'})
    is_dvt = ((dx.icd_version == 9) & code.str.startswith('4534')) | \
             ((dx.icd_version == 10) & dvt10)
    return set(dx.loc[is_pe | is_dvt, 'hadm_id'])


def find_imaged(cand_set):
    names = json.load(open(f'{RES}/dvt_exam_names.json'))
    DUPLEX = {n.upper() for n in names['duplex']}
    CTV = {n.upper() for n in names['ctv']}
    EXAM_RE = re.compile(r'(?i)examination\s*:?\s*(.{3,120}?)(?:\n|$)')
    PE_KEY = re.compile(r'(?i)\bpulmonary\s+embol(?:ism|us|i)\b')
    imaged = set()
    with gzip.open('physionet.org/files/mimiciv/mimic-iv-note/2.2/note/radiology.csv.gz', 'rt',
                   errors='replace') as f:
        for row in csv.DictReader(f):
            hid = row.get('hadm_id', '')
            if not hid or not hid.isdigit() or int(hid) not in cand_set:
                continue
            text = row.get('text', '') or ''
            exam = ''
            mm = EXAM_RE.search(text[:300])
            if mm:
                exam = mm.group(1).strip().upper()
            if PE_KEY.search(text) or exam in DUPLEX or exam in CTV:
                imaged.add(int(hid))
    return imaged


def eval_models(padua, improve, Xf, y, groups):
    """Return per-fold AUC for score-based and trained models on same folds.

    padua/improve: np arrays (n,) of raw scores on the evaluated rows.
    Xf: feature DataFrame (main 57 features).
    """
    out = {'padua': [], 'improve': [], 'lr': [], 'xgb': []}
    imp = SimpleImputer(strategy='median')
    Ximp = imp.fit_transform(Xf)
    for seed in (42, 43, 44):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(Xf, y, groups):
            out['padua'].append(roc_auc_score(y[te], padua[te]))
            out['improve'].append(roc_auc_score(y[te], improve[te]))
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(Ximp[tr], y[tr])
            out['lr'].append(roc_auc_score(y[te], lr.predict_proba(Ximp[te])[:, 1]))
            xgb = XGBClassifier(**XGB_PARAMS)
            xgb.fit(Ximp[tr], y[tr])
            out['xgb'].append(roc_auc_score(y[te], xgb.predict_proba(Ximp[te])[:, 1]))
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in out.items()}


def main():
    lab = pd.read_parquet(f'{RES}/ajm/new/data_era/vte_labels_inclprior_excl24h.parquet')
    keep = set(json.load(open(f'{RES}/ajm/new/data_era/inclprior_excl24h_kept_ids.json')))
    lab = lab[lab['hadm_id'].isin(keep)]
    m = pd.read_parquet(f'{RES}/radiology_vte_mentions.parquet')
    m['hadm_id'] = pd.to_numeric(m.hadm_id).astype('int64')
    m = m[m['hadm_id'].isin(keep)]
    mdf = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
    feat = json.load(open(f'{RES}/feature_sets_v2.json'))['main']

    icd_vte = load_icd_vte()
    bad = set(m.loc[m.label.isin(['positive', 'uncertain']), 'hadm_id'])
    non = lab[lab.vte_event == 0]
    cand = non[~non.hadm_id.isin(icd_vte | bad)]

    # BUG-FREE re-implementation vs paper: paper used cand=non minus icd/bad too.
    # n_cands printed for audit.
    print(f'strict candidates (no ICD, no pos/uncertain) = {len(cand)}', flush=True)
    imaged = find_imaged(set(cand.hadm_id))
    strict = cand[cand.hadm_id.isin(imaged)]
    pd.DataFrame({'hadm_id': strict.hadm_id}).to_parquet(
        f'{RES}/ajm/new/output_era/strict_control_ids_inclprior_excl24h.parquet')
    print(f'strict controls (imaged, clean) = {len(strict)}', flush=True)

    ev_ids = set(lab.loc[lab.vte_event == 1, 'hadm_id'])

    def run_subset(controls, tag):
        use = mdf[mdf['hadm_id'].isin(ev_ids | set(controls.hadm_id))].reset_index(drop=True)
        y = use['vte_event'].values.astype(int)
        groups = use['subject_id'].values
        X = use[feat].values.astype(np.float32)
        padua = use['padua_score'].values.astype(np.float64)
        improve = use['improve_score'].values.astype(np.float64)
        res = eval_models(padua, improve, pd.DataFrame(X, columns=feat), y, groups)
        n_ev, n_ct = int(y.sum()), len(y) - int(y.sum())
        print(f'[{tag}] rows={len(y)} events={n_ev} controls={n_ct}')
        for k, (a, se) in res.items():
            print(f'    {k:8s} AUC={a:.4f} SE={se:.4f} CI=({a-1.96*se:.4f}-{a+1.96*se:.4f})')
        return {'model': tag, 'rows': int(len(y)), 'events': n_ev, 'controls': n_ct,
                'auc': {k: {'mean': v[0], 'se': v[1],
                            'ci': [v[0]-1.96*v[1], v[0]+1.96*v[1]]} for k, v in res.items()}}

    out = {}
    out['strict_control'] = run_subset(strict, 'strict-imaged-clean')
    out['main_control'] = run_subset(non, 'main-all-non-vte')

    path = f'{RES}/ajm/new/output_era/strict_control_apples2apples_inclprior_excl24h.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved -> {path}')


if __name__ == '__main__':
    main()