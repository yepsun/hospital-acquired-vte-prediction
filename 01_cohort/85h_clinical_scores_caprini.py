"""Script 85h: Caprini VTE risk assessment model (third clinical comparator).

Implements the Caprini RAM (Caprini JA. Thrombosis risk assessment as a guide
to quality patient care. Dis Mon 2005;51:70-8; item set as restated in the
2010 update, Caprini JA. Clin Appl Thromb Hemost 2010;16:24S-37S) on the same
structured MIMIC-IV sources used for Padua / IMPROVE in `85e`.

Version implemented: **Caprini 2005 (2010 restatement)**, 1-2-3-5 point item
set, risk categories 0 = lowest, 1-2 = low, 3-4 = moderate, >= 5 = high.

Data sources (identical to 85e):
  features_comorbid_v3.parquet   age, cancer_active, copd, heart_failure,
                                 varicose, thrombophilia, infection_severe,
                                 mi, stroke, prior_vte_any, surgery_flag,
                                 obesity_icd
  features_vitals_v3.parquet     bmi
  features_treatments_v3.parquet proc_cvc, rx_hormone, icu_los_first24h
  vte_labels_v3.parquet          vte_event (QC only)

DELIBERATE DESIGN CHOICES (see results_vte/score_mapping_caprini.md):
  * Only pre-extracted v3 feature columns are used; no new DB extraction.
  * The single immobility proxy (icu_los_first24h > 0) is used ONCE, for the
    1-point "medical patient at bed rest" item. The 2-point "confined to bed
    > 72 h" item is left unmapped rather than counting the same 24 h proxy a
    second time at a higher weight.
  * surgery_flag (ICD-10-PCS section 0, any duration) is used as a declared
    proxy for "major surgery > 45 min" (2 pts). Operative duration and
    approach (open vs laparoscopic vs arthroscopic) are not recoverable.
  * All other items with no source in the extracted structured data are
    declared unmappable and scored 0 (conservative, never imputed).

Output: results_vte/clinical_scores_v3_caprini.parquet
  hadm_id, caprini_score, caprini_high (>=5), caprini_modplus (>=3),
  caprini_band (0 = 0 pts, 1 = 1-2, 2 = 3-4, 3 = >=5)
Also prints QC and writes results_vte/ajm/new/output_era/caprini_distribution.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

VERSION = 'Caprini 2005 RAM (2010 restatement)'
OUT_PARQUET = 'results_vte/clinical_scores_v3_caprini.parquet'
OUT_JSON = 'results_vte/ajm/new/output_era/caprini_distribution.json'

co = pd.read_parquet('results_vte/features_comorbid_v3.parquet', columns=[
    'hadm_id', 'age', 'cancer_active', 'copd', 'heart_failure', 'varicose',
    'thrombophilia', 'infection_severe', 'mi', 'stroke', 'prior_vte_any',
    'surgery_flag', 'obesity_icd'])
vi = pd.read_parquet('results_vte/features_vitals_v3.parquet',
                     columns=['hadm_id', 'bmi'])
tr = pd.read_parquet('results_vte/features_treatments_v3.parquet',
                     columns=['hadm_id', 'proc_cvc', 'rx_hormone',
                              'icu_los_first24h'])
lb = pd.read_parquet('results_vte/vte_labels_v3.parquet',
                     columns=['hadm_id', 'vte_event'])
ps = pd.read_parquet('results_vte/clinical_scores_v3.parquet')

df = (co.merge(vi, on='hadm_id', validate='1:1')
        .merge(tr, on='hadm_id', validate='1:1')
        .merge(lb, on='hadm_id', validate='1:1'))
N = len(df)
print(f'cohort rows: {N:,}')

# --- shared proxy, identical convention to 85e (Padua / IMPROVE) ---
immob = (df['icu_los_first24h'] > 0).fillna(False)

# --- Caprini item -> points -> data source --------------------------------
# (id, points, item label, mapping type, boolean condition, source column).
# mapping type / limitation text are mirrored in score_mapping_caprini.md.
CAPRINI_ITEMS = [
    ('age_41_60', 1, 'Age 41-60 y', 'direct',
     (df['age'] >= 41) & (df['age'] <= 60), 'age'),
    ('minor_surgery', 1, 'Minor surgery', 'unmappable', None, '-'),
    ('bmi_gt25', 1, 'BMI > 25 kg/m2', 'direct + fallback',
     (df['bmi'] > 25) | (df['obesity_icd'] == 1), 'bmi, obesity_icd'),
    ('swollen_legs', 1, 'Swollen legs (current)', 'unmappable', None, '-'),
    ('varicose_veins', 1, 'Varicose veins', 'direct',
     (df['varicose'] == 1), 'varicose'),
    ('pregnancy_postpartum', 1, 'Pregnancy or postpartum', 'unmappable',
     None, '-'),
    ('recurrent_abortion', 1, 'History of unexplained/recurrent abortion',
     'unmappable', None, '-'),
    ('oc_hrt', 1, 'Oral contraceptives or hormone replacement', 'direct',
     (df['rx_hormone'] == 1), 'rx_hormone'),
    ('sepsis_lt1mo', 1, 'Sepsis (< 1 month)', 'proxy',
     (df['infection_severe'] == 1), 'infection_severe'),
    ('serious_lung_disease', 1, 'Serious lung disease incl. pneumonia (< 1 mo)',
     'unmappable', None, '-'),
    ('abnormal_pulm_func', 1, 'Abnormal pulmonary function (COPD)', 'direct',
     (df['copd'] == 1), 'copd'),
    ('acute_mi_lt1mo', 1, 'Acute MI (< 1 month)', 'proxy',
     (df['mi'] == 1), 'mi'),
    ('chf_lt1mo', 1, 'Congestive heart failure (< 1 month)', 'proxy',
     (df['heart_failure'] == 1), 'heart_failure'),
    ('ibd', 1, 'Inflammatory bowel disease', 'unmappable', None, '-'),
    ('bed_rest_medical', 1, 'Medical patient currently at bed rest', 'PROXY',
     immob, 'icu_los_first24h > 0'),
    ('age_61_74', 2, 'Age 61-74 y', 'direct',
     (df['age'] >= 61) & (df['age'] <= 74), 'age'),
    ('arthroscopic_surgery', 2, 'Arthroscopic surgery', 'unmappable',
     None, '-'),
    ('major_surgery_gt45min', 2, 'Major surgery (> 45 min)', 'proxy',
     (df['surgery_flag'] == 1), 'surgery_flag'),
    ('laparoscopic_surgery', 2, 'Laparoscopic surgery (> 45 min)',
     'unmappable', None, '-'),
    ('malignancy', 2, 'Malignancy (present or previous)', 'proxy',
     (df['cancer_active'] == 1), 'cancer_active'),
    ('confined_bed_gt72h', 2, 'Confined to bed (> 72 h)', 'unmappable',
     None, '-'),
    ('immobilizing_cast', 2, 'Immobilizing cast', 'unmappable', None, '-'),
    ('central_venous_access', 2, 'Central venous access', 'direct',
     (df['proc_cvc'] == 1), 'proc_cvc'),
    ('age_ge75', 3, 'Age >= 75 y', 'direct', (df['age'] >= 75), 'age'),
    ('prior_dvt_pe', 3, 'History of DVT/PE', 'direct',
     (df['prior_vte_any'] == 1), 'prior_vte_any'),
    ('family_history', 3, 'Family history of thrombosis', 'unmappable',
     None, '-'),
    ('factor_v_leiden', 3, 'Factor V Leiden', 'unmappable', None, '-'),
    ('prothrombin_20210a', 3, 'Prothrombin 20210A mutation', 'unmappable',
     None, '-'),
    ('lupus_anticoagulant', 3, 'Lupus anticoagulant', 'unmappable', None, '-'),
    ('anticardiolipin', 3, 'Anticardiolipin antibodies', 'unmappable',
     None, '-'),
    ('homocysteine', 3, 'Elevated serum homocysteine', 'unmappable', None, '-'),
    ('hit', 3, 'Heparin-induced thrombocytopenia', 'unmappable', None, '-'),
    ('other_thrombophilia', 3, 'Other congenital/acquired thrombophilia',
     'proxy', (df['thrombophilia'] == 1),
     'thrombophilia (D68.5/D68.6, 289.8)'),
    ('stroke_lt1mo', 5, 'Stroke (< 1 month)', 'proxy',
     (df['stroke'] == 1), 'stroke'),
    ('elective_arthroplasty', 5,
     'Elective major lower-extremity arthroplasty', 'unmappable', None, '-'),
    ('hip_pelvis_leg_fracture', 5,
     'Hip, pelvis or leg fracture (< 1 month)', 'unmappable', None, '-'),
    ('multiple_trauma', 5, 'Multiple trauma (< 1 month)', 'unmappable',
     None, '-'),
    ('acute_spinal_cord_injury', 5,
     'Acute spinal cord injury / paralysis (< 1 month)', 'unmappable',
     None, '-'),
]

score = np.zeros(N, dtype=int)
prev, mapped = {}, {}
for key, pts, label, mtype, cond, src in CAPRINI_ITEMS:
    if cond is None:
        prev[key] = 0.0
        continue
    hit = cond.fillna(False).astype(bool).values
    prev[key] = float(hit.mean())
    mapped[key] = prev[key]
    score += pts * hit.astype(int)

df['caprini_score'] = score
df['caprini_high'] = (df['caprini_score'] >= 5).astype(int)
df['caprini_modplus'] = (df['caprini_score'] >= 3).astype(int)
df['caprini_band'] = pd.cut(df['caprini_score'], bins=[-1, 0, 2, 4, 10 ** 9],
                            labels=[0, 1, 2, 3]).astype(int)

out = df[['hadm_id', 'caprini_score', 'caprini_high', 'caprini_modplus',
          'caprini_band']].copy()
for c in out.columns[1:]:
    out[c] = out[c].astype(int)
assert out['hadm_id'].is_unique and len(out) == N
out.to_parquet(OUT_PARQUET, index=False)
print(f'saved {OUT_PARQUET}', out.shape)

# --- sensitivity: what if trauma_flag were used as a 5-pt proxy? ----------
trauma = (df['trauma_flag'] == 1).fillna(False).astype(int).values \
    if 'trauma_flag' in df else None
if trauma is None:
    trauma = (pd.read_parquet('results_vte/features_comorbid_v3.parquet',
                              columns=['hadm_id', 'trauma_flag'])
              .set_index('hadm_id').loc[df['hadm_id'], 'trauma_flag']
              .values == 1).astype(int)
alt = df['caprini_score'].values + 5 * trauma
print(f'[sensitivity] had trauma_flag been mapped to "multiple trauma" (5 pt): '
      f'mean={alt.mean():.2f} vs primary {df["caprini_score"].mean():.2f}; '
      f'high-risk (>=5) {100 * (alt >= 5).mean():.1f}% vs '
      f'{100 * df["caprini_high"].mean():.1f}%')

# --- QC / face validity ---------------------------------------------------
y = df['vte_event'].values.astype(int)
dist = {
    'version': VERSION,
    'n': int(N),
    'events': int(y.sum()),
    'event_rate': float(y.mean()),
    'mean': float(df['caprini_score'].mean()),
    'sd': float(df['caprini_score'].std()),
    'median': float(df['caprini_score'].median()),
    'q1': float(df['caprini_score'].quantile(0.25)),
    'q3': float(df['caprini_score'].quantile(0.75)),
    'min': int(df['caprini_score'].min()),
    'max': int(df['caprini_score'].max()),
    'high_risk_threshold': 5,
    'high_risk_prop': float(df['caprini_high'].mean()),
    'high_risk_n': int(df['caprini_high'].sum()),
    'moderate_plus_prop': float(df['caprini_modplus'].mean()),
    'score_counts': {int(k): int(v) for k, v in
                     df['caprini_score'].value_counts().sort_index().items()},
    'item_prevalence': prev,
    'mapped_items': sorted(mapped),
    'unmappable_items': [k for k, _p, _l, mt, _c, _s in CAPRINI_ITEMS
                         if mt == 'unmappable'],
    'sensitivity_trauma_flag_as_5pt': {
        'mean': float(alt.mean()), 'high_risk_prop': float((alt >= 5).mean())},
}

BANDS = [('lowest (0)', 0, 0), ('low (1-2)', 1, 2),
         ('moderate (3-4)', 3, 4), ('high (>=5)', 5, 99)]
dist['event_rate_by_band'] = {}
for name, lo, hi in BANDS:
    m = df['caprini_score'].between(lo, hi).values
    dist['event_rate_by_band'][name] = {
        'n': int(m.sum()), 'events': int(y[m].sum()),
        'event_rate': float(y[m].mean()),
        'share_of_cohort': float(m.mean())}

dist['event_rate_by_score'] = {
    str(int(s)): {'n': int((df['caprini_score'] == s).sum()),
                  'events': int(y[(df['caprini_score'] == s).values].sum()),
                  'event_rate': float(y[(df['caprini_score'] == s).values].mean())}
    for s in sorted(df['caprini_score'].unique())}

for label, hi_mask in [('high>=5', df['caprini_high'] == 1),
                       ('mod>=3', df['caprini_modplus'] == 1)]:
    hm = hi_mask.values
    a, b = y[hm].mean(), y[~hm].mean()
    dist[f'crude_{label}'] = {'event_rate_high': float(a),
                              'event_rate_low': float(b),
                              'risk_ratio': float(a / b),
                              'n_high': int(hm.sum()),
                              'n_low': int((~hm).sum())}

cmpdf = df[['hadm_id', 'vte_event', 'caprini_score', 'caprini_high']].merge(
    ps[['hadm_id', 'padua_score', 'padua_high', 'improve_score',
        'improve_high']], on='hadm_id', validate='1:1')
assert (cmpdf['vte_event'].values == y).all()
dist['comparison_on_same_rows'] = {}
for name, sc, hi in [('caprini', 'caprini_score', 'caprini_high'),
                     ('padua', 'padua_score', 'padua_high'),
                     ('improve', 'improve_score', 'improve_high')]:
    v = cmpdf[sc].values.astype(float)
    h = (cmpdf[hi] == 1).values
    r = rankdata(v)
    n1, n0 = int(y.sum()), len(v) - int(y.sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    dist['comparison_on_same_rows'][name] = {
        'mean': float(v.mean()), 'sd': float(v.std()),
        'high_risk_prop': float(h.mean()),
        'event_rate_high': float(y[h].mean()),
        'event_rate_low': float(y[~h].mean()),
        'risk_ratio': float(y[h].mean() / y[~h].mean()),
        'auc': float(auc)}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(dist, f, indent=2)
print(f'saved {OUT_JSON}')

print(f'\n== Caprini ({VERSION}) ==')
print(f'mean={dist["mean"]:.2f} SD={dist["sd"]:.2f} median={dist["median"]:.0f} '
      f'[IQR {dist["q1"]:.0f}-{dist["q3"]:.0f}] range {dist["min"]}-{dist["max"]}')
print(f'high-risk (>=5): {dist["high_risk_prop"]*100:.1f}% '
      f'(n={dist["high_risk_n"]:,}); moderate+ (>=3): '
      f'{dist["moderate_plus_prop"]*100:.1f}%')
print('score distribution (score: n, share):')
for s, c in dist['score_counts'].items():
    print(f'  {s:>3}: {c:>9,} ({c / N * 100:5.2f}%)')
print('event rate by band:')
for k, v in dist['event_rate_by_band'].items():
    print(f'  {k:<16} n={v["n"]:>9,} events={v["events"]:>5} '
          f'rate={v["event_rate"]*100:.3f}%')
for label in ['high>=5', 'mod>=3']:
    c = dist[f'crude_{label}']
    print(f'crude {label}: high={c["event_rate_high"]*100:.3f}% '
          f'low={c["event_rate_low"]*100:.3f}% RR={c["risk_ratio"]:.2f}x')
print('same-row comparison:')
for k, v in dist['comparison_on_same_rows'].items():
    print(f'  {k:<8} mean={v["mean"]:5.2f} SD={v["sd"]:4.2f} '
          f'high={v["high_risk_prop"]*100:5.1f}% RR={v["risk_ratio"]:.2f}x '
          f'AUC={v["auc"]:.4f}')
print('\nitem prevalence (mapped items only):')
for k, v in mapped.items():
    print(f'  {k:<26} {v*100:6.2f}%')
print(f'\nunmappable items ({len(dist["unmappable_items"])}): '
      + ', '.join(dist['unmappable_items']))
