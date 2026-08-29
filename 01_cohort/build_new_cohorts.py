#!/usr/bin/env python3
"""
Build the two study arms from final_cohort_vte.pkl (new PE/DVT ascertainment):

  primary_keep24h : cases = (PE | DVT) - prior VTE  (24h-window cases KEPT) -> 1,915
  sens_excl24h    : cases = same minus 24h-window cases (original definition) -> 627

Controls (no VTE ICD anywhere, per final_cohort flow) shared by both arms.
Rows restricted to model_dataset_v3_fixed.parquet hadm_ids; feature columns kept
identical so script 98 protocol runs unchanged. Splits: subject-grouped
GroupShuffleSplit 70/15/15, seed 42 (+ chronological temporal test = last 15%).
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RES = 'results_vte'
SEED = 42

cohort = pickle.load(open(f'{RES}/ajm/new/data/final_cohort_vte.pkl', 'rb'))
controls = set(cohort['controls'])
union = cohort['pe_cases'] | cohort['dvt_cases']
cases_keep24 = union - cohort['excluded_prior']
cases_excl24 = cases_keep24 - cohort['excluded_24h']
print(f'cases keep24h={len(cases_keep24)}  excl24h={len(cases_excl24)}  '
      f'controls={len(controls)}')

df = pd.read_parquet(f'{RES}/model_dataset_v3_fixed.parquet')
adm = pd.read_parquet(f'{RES}/admissions_base.parquet')[['hadm_id', 'admittime']]
df = df.merge(adm, on='hadm_id', how='left')
ids = set(df['hadm_id'])
assert cases_keep24 <= ids and controls <= ids


def make_arm(case_set, tag):
    d = df[df['hadm_id'].isin(controls | case_set)].copy()
    d['vte_event'] = d['hadm_id'].isin(case_set).astype(int)
    dx = pd.Series(cohort['dx_time'])
    d['diagnosis_time'] = d['hadm_id'].map(dx)
    n_dup = d['hadm_id'].duplicated().sum()
    assert n_dup == 0, f'{n_dup} duplicated hadm_ids'
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70 / 0.85, random_state=SEED)
    pool_idx, test_idx = next(gss1.split(d, groups=d['subject_id']))
    pool, test = d.iloc[pool_idx], d.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.70 / 0.85, random_state=SEED + 1)
    tr_idx, va_idx = next(gss2.split(pool, groups=pool['subject_id']))
    train, val = pool.iloc[tr_idx], pool.iloc[va_idx]
    order = pool.sort_values('admittime')
    cut = int(len(order) * 0.85)
    tr_pool_t, te_t = order.iloc[:cut], order.iloc[cut:]
    splits = {
        'train': sorted(train['hadm_id'].tolist()),
        'val': sorted(val['hadm_id'].tolist()),
        'test': sorted(test['hadm_id'].tolist()),
        'train_pool_temporal': sorted(tr_pool_t['hadm_id'].tolist()),
        'test_temporal': sorted(te_t['hadm_id'].tolist()),
    }
    out = f'{RES}/ajm/new/data/{tag}'
    d.drop(columns=['admittime']).to_parquet(out + '_model_dataset.parquet')
    with open(out + '_splits.json', 'w') as f:
        json.dump(splits, f)
    ev_pool = int(pool['vte_event'].sum())
    print(f'[{tag}] rows={len(d):,} events={int(d["vte_event"].sum()):,} '
          f'train={len(train):,} val={len(val):,} test={len(test):,} '
          f'pool_events={ev_pool:,} test_events={int(test["vte_event"].sum()):,}')


make_arm(cases_keep24, 'primary_keep24h')
make_arm(cases_excl24, 'sens_excl24h')
print('done')
