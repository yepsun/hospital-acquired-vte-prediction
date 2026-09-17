#!/usr/bin/env python3
"""Script 98d: valid AUPRC uncertainty for the deployment-safe arms.

Why this exists
---------------
The primary protocol's patient-level cluster bootstrap caps negatives at
50,000 rows per replicate.  That is harmless for AUC (a rank statistic on the
retained rows) but it inflates replicate prevalence from 0.177% to ~1.1%, which
changes the average-precision *estimand* - a capped replicate's AP (~0.07)
is not comparable to the full-pool AP (~0.0125).  AUPRC confidence intervals
therefore have to come from an UNCAPPED patient-level cluster bootstrap.

This script
  * rebuilds the seed-42 pooled-OOF probabilities for LR and XGBoost for every
    arm by re-running exactly the 98c/98b protocol, and asserts that the
    resulting OOF AUC equals the value stored in the arm JSON (so the vectors
    used here are provably the ones reported);
  * runs a patient-level cluster bootstrap over all rows of the development
    pool (subjects resampled with replacement, all of a sampled subject's rows
    kept, no negative cap), 2000 replicates, reporting the 2.5/97.5 percentile
    interval of pooled-OOF average precision;
  * writes the corrected intervals to a NEW json
    (deploysafe_auprc_ci_deploysafe.json);
  * rewrites the `oof_inference.oof_auprc_ci` field of the *_deploysafe arm
    jsons with the corrected interval and records the provenance, replacing
    the invalid negative-capped interval emitted by the first 98c revision.
    Only *_deploysafe files created by this task are touched.

Run from the repository root that contains results_vte/.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

RES = 'results_vte'
HERE = os.path.dirname(os.path.abspath(__file__))
D = f'{RES}/ajm/new/data_era'
O = f'{RES}/ajm/new/output_era'
SUFFIX = '_deploysafe'
ARMS = ['main_57', 'strict_42', 'conservative_40', 'strict44_surgtrauma']
SEED = 42
N_SPLITS = 5
N_BOOT = 2000

_argv = sys.argv
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location(
    'ds98c', os.path.join(HERE, '98c_deploysafe_full_suite.py'))
ds98c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds98c)
sys.argv = _argv

MAIN = json.load(open(f'{RES}/feature_sets_v2.json'))['main']
ARMS_FEATS = ds98c.build_arms(MAIN)
splits = json.load(open(f'{D}/primary_inclprior_excl24h_splits.json'))
pool_ids = set(splits['train']) | set(splits['val'])
df = pd.read_parquet(
    f'{D}/primary_inclprior_excl24h_model_dataset_caprini.parquet')
pool = df[df['hadm_id'].isin(pool_ids)].reset_index(drop=True)
y = pool['vte_event'].values.astype(int)
N = len(y)
codes, uniq = pd.factorize(pool['subject_id'])
n_subj = len(uniq)
print(f'pool N={N} events={y.sum()} subjects={n_subj}', flush=True)

out = {'method': 'patient-level cluster bootstrap, subject_id resampled with '
                 'replacement, ALL rows of sampled subjects retained (no '
                 'negative cap), 2000 replicates; pooled-OOF average precision '
                 f'of the seed-42 5-fold StratifiedGroupKFold predictions; '
                 f'rng seed {SEED}. Difference from the capped AUC bootstrap: '
                 'the 50k negative cap inflates replicate prevalence and is '
                 'valid for rank-based AUC only, not for average precision.',
       'n_replicates': N_BOOT, 'n_pool': int(N), 'n_events': int(y.sum()),
       'arms': {}}

t_all = time.time()
for arm in ARMS:
    feats = ARMS_FEATS[arm]
    ap_path = f'{O}/deploysafe_full_suite_{arm}{SUFFIX}.json'
    ref = json.load(open(ap_path))
    X = pool[feats].values.astype(np.float32)
    oof = {}
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=SEED)
    for tr, te in sgkf.split(X, y, codes):
        for kind in ['lr', 'xgb']:
            p, _ = ds98c.fit_predict(kind, X[tr], y[tr], X[te])
            oof.setdefault(kind, np.zeros(N))[te] = p
    for kind in ['lr', 'xgb']:
        auc = float(roc_auc_score(y, oof[kind]))
        ref_auc = ref[kind]['oof_auc']
        assert abs(auc - ref_auc) < 1e-9, \
            f'{arm}/{kind}: recomputed OOF AUC {auc} != stored {ref_auc}'
        print(f'[{arm}] {kind}: OOF vector verified against arm json '
              f'(AUC={auc:.10f})', flush=True)

    t0 = time.time()
    rng = np.random.RandomState(SEED)
    boot = {'lr': [], 'xgb': []}
    point = {k: float(average_precision_score(y, v)) for k, v in oof.items()}
    assert all(abs(point[k] - ref[k]['oof_auprc']) < 1e-9 for k in oof)
    for b in range(N_BOOT):
        counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
        rows = np.repeat(np.arange(N), counts[codes])
        yb = y[rows]
        for k, v in oof.items():
            boot[k].append(float(average_precision_score(yb, v[rows])))
        if (b + 1) % 500 == 0:
            print(f'  [{arm}] bootstrap {b + 1}/{N_BOOT} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    arm_out = {'n_features': len(feats)}
    for k in ['lr', 'xgb']:
        arm_out[k] = {
            'auprc': point[k],
            'ci': [float(v) for v in np.percentile(boot[k], [2.5, 97.5])],
            'n_replicates_used': len(boot[k])}
    out['arms'][arm] = arm_out
    for k in ['lr', 'xgb']:
        r = out['arms'][arm][k]
        print(f"[{arm}] {k.upper()} OOF AUPRC={r['auprc']:.5f} "
              f"[{r['ci'][0]:.5f},{r['ci'][1]:.5f}]", flush=True)

    # --- patch the arm json (own *_deploysafe artefact) -------------------
    ref['oof_inference'].pop('oof_auprc_ci', None)
    ref['oof_inference']['oof_auprc_ci_uncapped'] = out['arms'][arm]
    ref['oof_inference']['oof_auprc_ci_provenance'] = (
        'computed by scripts_era/98d_deploysafe_auprc_ci.py: uncapped '
        'patient-level cluster bootstrap (2000 replicates). The negative-capped '
        'primary bootstrap cannot be used for AP because it inflates replicate '
        'prevalence; the capped AP interval written by the first 98c revision '
        'was removed. Pooled-OOF AP point estimates match oof_auprc exactly.')
    with open(ap_path, 'w') as f:
        json.dump(ref, f, indent=2)
    print(f'[{arm}] patched -> {ap_path}', flush=True)

out['runtime_s'] = round(time.time() - t_all, 1)
p = f'{O}/deploysafe_auprc_ci{SUFFIX}.json'
with open(p, 'w') as f:
    json.dump(out, f, indent=2)
print(f'saved -> {p} ({out["runtime_s"]:.0f}s)', flush=True)
