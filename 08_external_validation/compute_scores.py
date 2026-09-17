#!/usr/bin/env python3
"""Compute Padua / IMPROVE scores on the external-validation cohort.

Same item mapping as scripts/85e_clinical_scores_v3.py:
  - reduced mobility / paralysis-immobility proxy: ICU stay within first 24h
    (here: ICU evidence flag -> icu_los_first24h>0 proxy)
  - recent surgery/trauma (Padua): surgery_flag | trauma_flag
  - obesity: bmi>=30 | obesity_icd
  - hormonal: rx_hormone

Output: extval/scores.parquet (vis_id, padua_score, improve_score)
"""
import pandas as pd

co = pd.read_parquet('extval/cohort.parquet',
                     columns=['vis_id', 'age', 'prior_vte_any'])
fe = pd.read_parquet('extval/features.parquet')

df = co.merge(fe[['vis_id', 'cancer_active', 'thrombophilia', 'heart_failure',
                  'copd', 'mi', 'stroke', 'infection_severe', 'rheumatologic',
                  'bmi', 'obesity_icd', 'rx_hormone', 'surgery_flag',
                  'trauma_flag', 'proc_cvc', 'icu_los_first24h']],
              on='vis_id', how='left', validate='1:1')

immob = (df['icu_los_first24h'] > 0).fillna(False)

# --- Padua ---
padua_items = {
    'active_cancer':        (3, df['cancer_active'] == 1),
    'prior_vte':            (3, df['prior_vte_any'] == 1),
    'reduced_mobility':     (3, immob),
    'thrombophilia':        (3, df['thrombophilia'] == 1),
    'recent_surg_trauma':   (2, (df['surgery_flag'] == 1) |
                             (df['trauma_flag'] == 1)),
    'age_ge70':             (2, df['age'] >= 70),
    'hf_resp_failure':      (1, (df['heart_failure'] == 1) |
                             (df['copd'] == 1)),
    'acute_mi_stroke':      (1, (df['mi'] == 1) | (df['stroke'] == 1)),
    'acute_infection_rheum': (1, (df['infection_severe'] == 1) |
                              (df['rheumatologic'] == 1)),
    'obesity':              (1, (df['bmi'] >= 30) | (df['obesity_icd'] == 1)),
    'hormonal_treatment':   (1, df['rx_hormone'] == 1),
}
df['padua_score'] = sum(pts * cond.fillna(False).astype(int)
                        for pts, cond in padua_items.values())

# --- IMPROVE ---
improve_items = {
    'prior_vte':      (3, df['prior_vte_any'] == 1),
    'thrombophilia':  (2, df['thrombophilia'] == 1),
    'paralysis_immob': (2, immob),
    'cancer':         (2, df['cancer_active'] == 1),
    'icu_ccu':        (1, df['icu_los_first24h'] > 0),
    'cvc':            (1, df['proc_cvc'] == 1),
    'age_gt60':       (1, df['age'] > 60),
}
df['improve_score'] = sum(pts * cond.fillna(False).astype(int)
                          for pts, cond in improve_items.values())

out = df[['vis_id', 'padua_score', 'improve_score']].copy()
out.to_parquet('extval/scores.parquet', index=False)
print('saved extval/scores.parquet', out.shape)
print('Padua mean %.2f  high(>=4) %.2f%%' %
      (out['padua_score'].mean(), (out['padua_score'] >= 4).mean() * 100))
print('IMPROVE mean %.2f  high(>=4) %.2f%%' %
      (out['improve_score'].mean(), (out['improve_score'] >= 4).mean() * 100))
