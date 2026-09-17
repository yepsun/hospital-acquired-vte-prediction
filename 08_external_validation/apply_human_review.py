#!/usr/bin/env python3
"""Apply human review of DeepSeek/GLM discordant reports to the external
radiology adjudication, and record post-review agreement statistics.

Input:  extval/llm_discordant_for_review.csv  (human_verdict filled)
Action: overrides llm_verdict in extval/radiology_llm.parquet for the 153
        discordant report_ids (human verdict is final; original DeepSeek
        parquet backed up as radiology_llm_deepseek_orig.parquet).
Output: updates extval/llm_agreement.json with human-review statistics.
Idempotent: re-running re-applies the same overrides.
"""
import json
import os
import shutil

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = f'{BASE}/llm_discordant_for_review.csv'
PARQUET = f'{BASE}/radiology_llm.parquet'
BACKUP = f'{BASE}/radiology_llm_deepseek_orig.parquet'
AGREE = f'{BASE}/llm_agreement.json'

rev = pd.read_csv(CSV, dtype={'human_verdict': str})
assert rev['human_verdict'].isin(['y', 'n']).all(), 'unfilled/invalid human_verdict'
assert len(rev) == 153

if not os.path.exists(BACKUP):
    shutil.copy(PARQUET, BACKUP)
    print('backup ->', BACKUP)

df = pd.read_parquet(BACKUP)  # always start from pristine DeepSeek labels
ids = set(rev['report_id'])
m = df['report_id'].isin(ids)
# report_id is not unique in the merged parquet (identical 检查所见 shared by
# sibling rows of one admission); the LLM merge already mapped verdicts by id,
# so overrides follow the same semantics. All duplicated-id siblings belong to
# the same admission; per-report timing effects are negligible.
print(f'unique review ids: {len(ids)}; parquet rows matched: {int(m.sum())}')
ov = rev.drop_duplicates('report_id').set_index('report_id')
df.loc[m, 'llm_reason'] = df.loc[m, 'report_id'].map(
    lambda r: f"human review override (DS={ov.loc[r, 'deepseek_verdict']}, "
              f"GLM={ov.loc[r, 'glm_verdict']})")
df.loc[m, 'llm_verdict'] = df.loc[m, 'report_id'].map(ov['human_verdict'])
df.to_parquet(PARQUET, index=False)
print('overrides applied:', int(m.sum()))
print('final verdict distribution:',
      df['llm_verdict'].value_counts().to_dict())

rev['ds_y'] = rev['deepseek_verdict'] == 'y'
rev['glm_y'] = rev['glm_verdict'] == 'y'
rev['h_y'] = rev['human_verdict'] == 'y'
human_stats = {
    'n_human_reviewed': int(len(rev)),
    'human_y': int(rev['h_y'].sum()),
    'human_n': int((~rev['h_y']).sum()),
    'deepseek_y_status_agrees_with_human': int((rev['ds_y'] == rev['h_y']).sum()),
    'glm_y_status_agrees_with_human': int((rev['glm_y'] == rev['h_y']).sum()),
    'human_overturns_deepseek_y_status': int((rev['ds_y'] != rev['h_y']).sum()),
    'human_overturns_glm_y_status': int((rev['glm_y'] != rev['h_y']).sum()),
    'glm_only_positive_confirmed_by_human': int(
        ((~rev['ds_y']) & rev['glm_y'] & rev['h_y']).sum()),
    'deepseek_only_positive_overturned_by_human': int(
        (rev['ds_y'] & (~rev['glm_y']) & (~rev['h_y'])).sum()),
}
ag = json.load(open(AGREE))
ag['human_review'] = human_stats
json.dump(ag, open(AGREE, 'w'), ensure_ascii=False, indent=2)
print(json.dumps(human_stats, ensure_ascii=False, indent=2))
