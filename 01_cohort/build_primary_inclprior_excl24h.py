#!/usr/bin/env python3
"""
Build the new primary arm: include prior VTE, exclude 24h-window cases.

  inclprior_excl24h : cases = (PE | DVT) - excluded_24h  -> 674
  controls          : no-VTE-ICD controls (397,709) PLUS prior-VTE admissions
                      without a new in-hospital event and without ICD-only
                      exclusion (excluded_prior - union - excluded_icd)

Same feature columns, split protocol, and output layout as
build_new_cohorts.py so script 98 and downstream scripts run unchanged.
"""
import json
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RES = 'results_vte'
SEED = 42

cohort = pickle.load(open(f'{RES}/ajm/new/data/final_cohort_vte.pkl', 'rb'))
union = cohort['pe_cases'] | cohort['dvt_cases']
case_set = union - cohort['excluded_24h']
prior_noncases = cohort['excluded_prior'] - union - cohort['excluded_icd']
control_set = cohort['controls'] | prior_noncases
print(f'cases={len(case_set)}  prior_noncases={len(prior_noncases)}  '
      f'controls={len(control_set)}')

df = pd.read_parquet(f'{RES}/model_dataset_v3_fixed.parquet')
adm = pd.read_parquet(f'{RES}/admissions_base.parquet')[['hadm_id', 'admittime']]
df = df.merge(adm, on='hadm_id', how='left')
ids = set(df['hadm_id'])
assert case_set <= ids and control_set <= ids

tag = 'inclprior_excl24h'
d = df[df['hadm_id'].isin(control_set | case_set)].copy()
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
print(f'[{tag}] rows={len(d):,} events={int(d["vte_event"].sum()):,} '
      f'train={len(train):,} val={len(val):,} test={len(test):,} '
      f'pool_events={int(pool["vte_event"].sum()):,} '
      f'test_events={int(test["vte_event"].sum()):,}')
print('done')
