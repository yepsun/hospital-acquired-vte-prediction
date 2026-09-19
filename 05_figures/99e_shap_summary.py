#!/usr/bin/env python3
"""99e: SHAP summary for the deployment-ready frozen 42-feature XGBoost model.

Reviewer point (interpretability / M7 cross-check): the main text and
supplement report only aggregate gain shares. This produces a SHAP beeswarm
for the frozen era-freeze strict_42 model (the recommended deployment arm)
on development-pool admissions (all 554 events plus a 60,000-row random
non-event sample), saved as the supplement's Figure S1.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import load

RES = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte'
EJIM = '/Users/Yepsun/Mywork/Vscodeprojects/mimic/results_vte/JAMIA/ejim'
FRZ = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval/frozen_models_era_deploysafe/strict_42'

fmeta = json.load(open(f'{FRZ}/feature_order.json'))
feats = fmeta['features'] if isinstance(fmeta, dict) else fmeta
model = load(f'{FRZ}/final_xgb.joblib')
print(f'frozen strict_42: {len(feats)} features', flush=True)

splits = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
df = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet')
pool_ids = set(splits['train']) | set(splits['val'])
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)

rng = np.random.RandomState(42)
ev = pool[pool['vte_event'] == 1]
non = pool[pool['vte_event'] == 0].sample(60_000, random_state=42)
sample = pd.concat([ev, non]).reset_index(drop=True)
X = sample[feats].values.astype(np.float32)
print(f'SHAP sample: {len(X)} rows ({len(ev)} events)', flush=True)

expl = shap.TreeExplainer(model)
sv = expl.shap_values(X)
if isinstance(sv, list):
    sv = sv[1] if len(sv) > 1 else sv[0]

mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)
share_util = mean_abs[[feats.index(f) for f in feats
                       if f.endswith('_count') or f in
                       ('proc_cvc', 'proc_mechvent', 'proc_transfusion',
                        'icu_los_first24h')]].sum() / mean_abs.sum()
print('top-10 mean|SHAP|:', [(feats[i], round(float(mean_abs[i]), 5)) for i in order[:10]],
      flush=True)
print(f'utilization-proxy share of mean|SHAP|: {share_util:.3f}', flush=True)

plt.figure()
shap.summary_plot(sv, X, feature_names=feats, show=False, max_display=20)
plt.gcf().set_size_inches(7.2, 6.4)
plt.tight_layout()
plt.savefig(f'{EJIM}/supplemental_figure_s1_shap.png', dpi=300, bbox_inches='tight')
plt.close()

from PIL import Image
img = Image.open(f'{EJIM}/supplemental_figure_s1_shap.png').convert('RGB')
w, h = img.size
img.resize((w * 2, h * 2), Image.LANCZOS).save(
    f'{EJIM}/supplemental_figure_s1_shap.tif', compression='tiff_lzw', dpi=(600, 600))

out = {'model': 'frozen era-freeze strict_42 XGBoost (42 admission-safe features)',
       'background': 'development pool; all 554 events + 60,000 random non-events',
       'mean_abs_shap': {feats[i]: float(mean_abs[i]) for i in order},
       'utilization_proxy_share_mean_abs_shap': float(share_util)}
json.dump(out, open(f'{RES}/ajm/new/output_era/shap_summary_strict42.json', 'w'), indent=1)
print('saved supplemental_figure_s1_shap.png/.tif + shap_summary_strict42.json', flush=True)
