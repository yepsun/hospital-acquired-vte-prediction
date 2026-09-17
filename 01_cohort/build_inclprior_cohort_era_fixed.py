#!/usr/bin/env python3
"""Rebuild the broken "include prior VTE, retain 24-h-window events" arm.

The published arm (data_era/sens_inclprior_model_dataset.parquet, 366,917
rows / 2,145 events) drew its controls from the pure no-VTE-ICD pool only:
build_new_cohorts.py:24 sets `controls = set(cohort['controls'])` and :39
builds `d = df[df['hadm_id'].isin(controls | case_set)]`, so the prior-VTE
admissions that had no new in-hospital event and no ICD-only exclusion
(`excluded_prior - union - excluded_icd`, 14,274 rows in the era) were never
added as controls.

This script uses the correct recipe, taken verbatim from
build_primary_inclprior_excl24h.py:25-26:

    cases    = union (= pe_cases | dvt_cases)                  -> 2,145
    controls = controls | (excluded_prior - union - excluded_icd)
                                                            -> 364,772 + 14,274

Era restriction follows the convention used for every other file in
data_era/: the cohort is built on the full MIMIC-IV pool with the
build_new_cohorts.py split recipe (subject-grouped GroupShuffleSplit
70/15/15, seeds 42/43, plus a chronological 85/15 temporal cut), after which
both the rows and every split key are intersected with the 2008-2019 anchor
years carried in data_era/era_map.parquet; only train/val/test are kept, as
in the published data_era/*_splits.json files.

Writes NEW files only; nothing under data/, data_era/ or output_era/ is
overwritten.
"""
import json
import pickle

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RES = 'results_vte/ajm/new'
SEED = 42
TAG = 'sens_inclprior_fixed'
ERA_GROUPS = {'2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019'}

cohort = pickle.load(open(f'{RES}/data/final_cohort_vte.pkl', 'rb'))
union = cohort['pe_cases'] | cohort['dvt_cases']
controls = set(cohort['controls'])
prior_noncases = cohort['excluded_prior'] - union - cohort['excluded_icd']
case_set = set(union)
control_set = controls | prior_noncases
print(f'cases={len(case_set)}  prior_noncases={len(prior_noncases)}  '
      f'pure_controls={len(controls)}  controls={len(control_set)}  '
      f'total={len(case_set) + len(control_set)}')

df = pd.read_parquet('results_vte/model_dataset_v3_fixed.parquet')
adm = pd.read_parquet('results_vte/admissions_base.parquet')[['hadm_id', 'admittime']]
df = df.merge(adm, on='hadm_id', how='left')
ids = set(df['hadm_id'])
assert case_set <= ids and control_set <= ids

d = df[df['hadm_id'].isin(control_set | case_set)].copy()
d['vte_event'] = d['hadm_id'].isin(case_set).astype(int)
dx = pd.Series(cohort['dx_time'])
d['diagnosis_time'] = d['hadm_id'].map(dx)
n_dup = d['hadm_id'].duplicated().sum()
assert n_dup == 0, f'{n_dup} duplicated hadm_ids'

ref_cols = list(pd.read_parquet(f'{RES}/data_era/sens_inclprior_model_dataset.parquet').columns)
assert [c for c in d.columns if c != 'admittime'] == ref_cols, \
    'column order/layout drifted from published arms'

gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70 / 0.85, random_state=SEED)
pool_idx, test_idx = next(gss1.split(d, groups=d['subject_id']))
pool, test = d.iloc[pool_idx], d.iloc[test_idx]
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.70 / 0.85, random_state=SEED + 1)
tr_idx, va_idx = next(gss2.split(pool, groups=pool['subject_id']))
train, val = pool.iloc[tr_idx], pool.iloc[va_idx]
order = pool.sort_values('admittime')
cut = int(len(order) * 0.85)
tr_pool_t, te_t = order.iloc[:cut], order.iloc[cut:]
splits_full = {
    'train': sorted(train['hadm_id'].tolist()),
    'val': sorted(val['hadm_id'].tolist()),
    'test': sorted(test['hadm_id'].tolist()),
    'train_pool_temporal': sorted(tr_pool_t['hadm_id'].tolist()),
    'test_temporal': sorted(te_t['hadm_id'].tolist()),
}
d = d.drop(columns=['admittime'])
print(f'[full pool] rows={len(d):,} events={int(d["vte_event"].sum()):,} '
      f'train={len(train):,} val={len(val):,} test={len(test):,} '
      f'pool_events={int(pool["vte_event"].sum()):,} '
      f'test_events={int(test["vte_event"].sum()):,}')

era_map = pd.read_parquet(f'{RES}/data_era/era_map.parquet')
era_ids = set(era_map.loc[era_map['anchor_year_group'].isin(ERA_GROUPS), 'hadm_id'])
in_era = d['hadm_id'].isin(era_ids)
d_era = d[in_era].reset_index(drop=True)
assert (d_era['anchor_year_group'].isin(ERA_GROUPS)).all()
assert d['hadm_id'].isin(era_ids).sum() == len(d_era)

splits = {k: sorted(set(v) & era_ids) for k, v in splits_full.items()
          if k in ('train', 'val', 'test')}
covered = set(splits['train']) | set(splits['val']) | set(splits['test'])
assert covered == set(d_era['hadm_id']), 'era splits do not tile the era rows'

out = f'{RES}/data_era/{TAG}'
d_era.to_parquet(out + '_model_dataset.parquet')
with open(out + '_splits.json', 'w') as f:
    json.dump(splits, f)

ev = int(d_era['vte_event'].sum())
print(f'[{TAG}] rows={len(d_era):,} events={ev:,} controls={len(d_era) - ev:,}')
for k, v in splits.items():
    print(f'  {k}: {len(v):,}  events={int(d_era["hadm_id"].isin(v).mul(d_era["vte_event"]).sum()):,}')
print(f'wrote {out}_model_dataset.parquet and {out}_splits.json')
