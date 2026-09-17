#!/usr/bin/env python3
"""External cohort for the new primary definition: exclude VTE imaged within
24 h of admission; prior-VTE admissions RETAINED (mirrors the new MIMIC
primary arm inclprior_excl24h).

Derived from the existing extval/cohort.parquet (all rows, prior VTE included)
plus extval/radiology_llm.parquet timing:
  - new cases: LLM-'y' acute PE / lower-extremity DVT with imaging time
    >= admit + 24 h
  - old cases whose only radiological event is within 24 h: excluded from the
    cohort (mirrors MIMIC removal of 24h-window cases; NOT turned into controls)
  - ICD-only (VTE ICD but no >24h radiological confirmation): excluded, as before
  - prior-VTE flags (this-admission history codes / earlier-admission events)
    unchanged from cohort.parquet

Output: extval/cohort_ipe.parquet
"""
import re

import pandas as pd

BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'

co = pd.read_parquet(f'{BASE}/cohort.parquet')

rad = pd.read_parquet(f'{BASE}/radiology_llm.parquet')
rad['time'] = pd.to_datetime(rad['检查时间'], errors='coerce') \
    .fillna(pd.to_datetime(rad['出报告时间'], errors='coerce'))
rad = rad[rad['llm_verdict'] == 'y'].copy()
PE_RE = re.compile(r'肺|肺动脉|肺栓塞')
LEG_DVT_RE = re.compile(r'股|腘|胫|腓静脉|小腿肌间|深静脉|髂外|髂总|髂静脉|下肢')
txt = rad['检查诊断'].fillna('').astype(str) + ' ' + \
    rad['检查所见'].fillna('').astype(str)
rad = rad[txt.apply(lambda t: bool(PE_RE.search(t) or LEG_DVT_RE.search(t)))].copy()
rad = rad.merge(co[['vis_id', 'admit']], left_on='就诊号', right_on='vis_id',
                how='inner')
rad = rad.dropna(subset=['admit'])
case_vis_new = set(rad.loc[(rad['time'].notna()) &
                           (rad['time'] >= rad['admit'] + pd.Timedelta(hours=24)),
                           'vis_id'])

old_cases = set(co.loc[co['vte_outcome'] == 1, 'vis_id'])
cases_24h_only = old_cases - case_vis_new
print(f'old cases: {len(old_cases)}, new (>24h) cases: {len(case_vis_new)}, '
      f'24h-only cases removed: {len(cases_24h_only)}')

co['vte_outcome'] = co['vis_id'].isin(case_vis_new).astype(int)
co['vte_radiology'] = co['vte_outcome']

# remove 24h-only old cases entirely
co = co[~co['vis_id'].isin(cases_24h_only)].copy()
# ICD-only under the new outcome: VTE ICD but no >24h radiological event
icd_only = (co['vte_icd'] == 1) & (co['vte_outcome'] == 0)
print(f'ICD-only excluded: {int(icd_only.sum())}')
co = co[~icd_only].copy()

co.to_parquet(f'{BASE}/cohort_ipe.parquet', index=False)
print('cohort_ipe rows:', len(co))
print(co.groupby('risk_group').agg(
    n=('vis_id', 'count'), events=('vte_outcome', 'sum'),
    event_rate=('vte_outcome', 'mean'), prior_vte=('prior_vte_any', 'sum')))
