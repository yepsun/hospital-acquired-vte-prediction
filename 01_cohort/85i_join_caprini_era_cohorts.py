#!/usr/bin/env python3
"""Script 85i: join the Caprini score into the era cohort datasets.

Reads results_vte/clinical_scores_v3_caprini.parquet (425,090 admissions) and
adds caprini_score / caprini_high / caprini_modplus / caprini_band to each era
cohort dataset. Writes NEW files with suffix *_caprini.parquet; the published
era datasets are never modified.

Outputs: results_vte/ajm/new/data_era/<tag>_model_dataset_caprini.parquet
"""
import os

import pandas as pd

D = 'results_vte/ajm/new/data_era'
CAP = 'results_vte/clinical_scores_v3_caprini.parquet'
COLS = ['hadm_id', 'caprini_score', 'caprini_high', 'caprini_modplus',
        'caprini_band']

DATASETS = [
    'primary_inclprior_excl24h',
    'primary_keep24h',
    'sens_excl24h',
    'sens_inclprior',
    'sens_inclprior_fixed',
]

cap = pd.read_parquet(CAP, columns=COLS)
assert cap['hadm_id'].is_unique

for tag in DATASETS:
    src = f'{D}/{tag}_model_dataset.parquet'
    dst = f'{D}/{tag}_model_dataset_caprini.parquet'
    if not os.path.exists(src):
        print(f'skip (missing): {src}')
        continue
    if os.path.exists(dst):
        print(f'skip (exists, not overwriting): {dst}')
        continue
    df = pd.read_parquet(src)
    n = len(df)
    assert 'caprini_score' not in df.columns, f'{tag} already has caprini'
    # drop any stale helper columns from an earlier partial run
    df = df[[c for c in df.columns if not c.startswith('caprini')]]
    m = df.merge(cap, on='hadm_id', how='left', validate='1:1')
    assert len(m) == n, (len(m), n)
    assert m['caprini_score'].notna().all(), \
        f'{tag}: {int(m["caprini_score"].isna().sum())} unmatched hadm_id'
    m.to_parquet(dst, index=False)
    print(f'{dst}: rows={len(m):,} events={int(m["vte_event"].sum()):,} '
          f'caprini mean={m["caprini_score"].mean():.2f} '
          f'high>=5={m["caprini_high"].mean()*100:.1f}%')
