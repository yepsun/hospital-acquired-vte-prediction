"""Script 36: Padua / IMPROVE clinical risk scores.

Padua Prediction Score (11 items, >=4 high risk) and IMPROVE VTE RAM
(>=4 high risk), plus an IMPROVE-DD variant adding D-dimer >=2xULN (2 pts).
Item-to-data mapping documented in results_vte/score_mapping.md.
"""
import pandas as pd

N_EXPECT = None  # dynamic from labels

co = pd.read_parquet('results_vte/features_comorbid_v3.parquet')
vi = pd.read_parquet('results_vte/features_vitals_v3.parquet', columns=['hadm_id', 'bmi'])
la = pd.read_parquet('results_vte/features_labs_v3.parquet', columns=['hadm_id', 'lab_ddimer_median'])
tr = pd.read_parquet('results_vte/features_treatments_v3.parquet',
                     columns=['hadm_id', 'rx_hormone', 'proc_cvc', 'icu_los_first24h'])
lb = pd.read_parquet('results_vte/vte_labels_v3.parquet', columns=['hadm_id', 'vte_event'])

df = co.merge(vi, on='hadm_id', validate='1:1') \
       .merge(la, on='hadm_id', validate='1:1') \
       .merge(tr, on='hadm_id', validate='1:1') \
       .merge(lb, on='hadm_id', validate='1:1')
assert N_EXPECT is None or len(df) == N_EXPECT, len(df)

# Shared proxy: reduced mobility / lower-limb immobilization.
# Proxy = ICU stay within first 24h of admission (no mobility orders in MIMIC).
immob = (df['icu_los_first24h'] > 0).fillna(False)

# --- Padua ---
padua_items = {
    'active_cancer':        (3, df['cancer_active'] == 1),
    'prior_vte':            (3, df['prior_vte_any'] == 1),
    'reduced_mobility':     (3, immob),
    'thrombophilia':        (3, df['thrombophilia'] == 1),
    'recent_surg_trauma':   (2, (df['surgery_flag'] == 1) | (df['trauma_flag'] == 1)),
    'age_ge70':             (2, df['age'] >= 70),
    'hf_resp_failure':      (1, (df['heart_failure'] == 1) | (df['copd'] == 1)),
    'acute_mi_stroke':      (1, (df['mi'] == 1) | (df['stroke'] == 1)),
    'acute_infection_rheum':(1, (df['infection_severe'] == 1) | (df['rheumatologic'] == 1)),
    'obesity':              (1, (df['bmi'] >= 30) | (df['obesity_icd'] == 1)),
    'hormonal_treatment':   (1, df['rx_hormone'] == 1),
}
df['padua_score'] = sum(pts * cond.fillna(False).astype(int)
                        for pts, cond in padua_items.values())
df['padua_high'] = (df['padua_score'] >= 4).astype(int)

# --- IMPROVE VTE RAM ---
improve_items = {
    'prior_vte':      (3, df['prior_vte_any'] == 1),
    'thrombophilia':  (2, df['thrombophilia'] == 1),
    'paralysis_immob':(2, immob),
    'cancer':         (2, df['cancer_active'] == 1),
    'icu_ccu':        (1, df['icu_los_first24h'] > 0),
    'cvc':            (1, df['proc_cvc'] == 1),
    'age_gt60':       (1, df['age'] > 60),
}
df['improve_score'] = sum(pts * cond.fillna(False).astype(int)
                          for pts, cond in improve_items.values())
df['improve_high'] = (df['improve_score'] >= 4).astype(int)

# --- IMPROVE-DD (D-dimer variant) ---
# D-dimer >= 2xULN, ULN = 500 ng/mL FEU -> threshold 1000 ng/mL.
# Missing ddimer (99.3% of cohort) counts as 0: NOT generalizable, see score_mapping.md.
dd_high = df['lab_ddimer_median'] >= 1000
df['improve_dd'] = df['improve_score'] + 2 * dd_high.fillna(False).astype(int)
df['improve_dd_high'] = (df['improve_dd'] >= 4).astype(int)

out = df[['hadm_id', 'padua_score', 'padua_high',
          'improve_score', 'improve_high',
          'improve_dd', 'improve_dd_high']].copy()
for c in out.columns[1:]:
    out[c] = out[c].astype(int)
assert out['hadm_id'].is_unique
out.to_parquet('results_vte/clinical_scores_v3.parquet', index=False)
print('saved clinical_scores_v3.parquet', out.shape)

# --- QC ---
def report(score, high, name):
    s = df[score]
    print(f'\n== {name} ==')
    print(f'mean={s.mean():.2f} median={s.median():.0f} '
          f'high_risk_prop={df[high].mean():.4f} (n={int(df[high].sum())})')
    ct = pd.crosstab(df[high], df['vte_event'], normalize='index')
    print(ct.round(4))
    rate_hi = df.loc[df[high] == 1, 'vte_event'].mean()
    rate_lo = df.loc[df[high] == 0, 'vte_event'].mean()
    print(f'event rate high={rate_hi:.4f} vs low={rate_lo:.4f} ratio={rate_hi/rate_lo:.2f}x')

report('padua_score', 'padua_high', 'Padua')
report('improve_score', 'improve_high', 'IMPROVE')
report('improve_dd', 'improve_dd_high', 'IMPROVE-DD')
print('\nPadua score distribution:')
print(df['padua_score'].value_counts().sort_index())
print('\nIMPROVE score distribution:')
print(df['improve_score'].value_counts().sort_index())
print('\nItem prevalence:')
for name, (pts, cond) in {**{f'padua.{k}': v for k, v in padua_items.items()},
                          **{f'improve.{k}': v for k, v in improve_items.items()}}.items():
    print(f'  {name} ({pts}pt): {cond.fillna(False).mean():.4f}')
print(f'  ddimer>=1000 (2pt, improve_dd): {dd_high.fillna(False).mean():.5f}')
