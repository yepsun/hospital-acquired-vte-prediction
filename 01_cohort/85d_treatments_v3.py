"""Script 35: Admission-24h medication & procedure/treatment features.

Consumes: results_vte/vte_labels_v3.parquet, mimic4.db
  (pharmacy, procedures_icd, d_icd_procedures, procedureevents, inputevents,
   d_items, transfers)
Produces: results_vte/features_treatments_v3.parquet
  columns: hadm_id, rx_heparin, rx_warfarin, rx_doac, rx_antiplatelet,
           rx_hormone (all 0/1), proc_cvc, proc_mechvent, proc_transfusion
           (all 0/1), icu_los_first24h (float hours)

Source choices (from availability exploration, see m2_qc_notes.md):
  - Medications: pharmacy.medication (primary; prescriptions.drug gives nearly
    identical cohort coverage -- spot-checked heparin 54.54% vs 54.54%).
    Fuzzy match on lower(medication). Non-systemic heparin uses are excluded
    (flush/dwell/lock/priming/CRRT/IABP/hemodialysis circuit). LMWH
    (enoxaparin/lovenox) folded into rx_heparin.
    Window: pharmacy.starttime in [admittime, +24h] (start-only, anti-leakage;
    rows with NULL starttime excluded, counted).
  - proc_cvc: procedures_icd (ICD-10 02HV*/05HM*/06HM*, ICD-9 3893*/3895*)
    UNION procedureevents line-placement itemids (PICC 224264, dialysis
    catheter 224270, Presep 224273, tunneled Hickman 225315, tunneled access
    229531, bedside line placement 229580). chartevents 'Central Line ordered
    on POE' itemids (225707/228036) had 0 rows in cohort window.
  - proc_mechvent: procedures_icd (ICD-10 5A19*, ICD-9 967*) UNION
    procedureevents ventilation itemids (225792 invasive, 225794 non-invasive,
    226260 mechanically ventilated).
  - proc_transfusion: procedures_icd (ICD-10 3023*, ICD-9 990*) UNION
    inputevents blood-product itemids (FFP/plasma/PRBC/platelets incl.
    OR/PACU intake).
  - ICD procedures have date-granular chartdate: window = chartdate between
    date(admittime) and date(admittime+24h).
  - icu_los_first24h: transfers rows whose careunit is an ICU-type unit,
    overlap hours of [intime, outtime] with [admittime, +24h], summed.
"""
import sqlite3

import numpy as np
import pandas as pd

RX_PATTERNS = {
    # LMWH (enoxaparin/lovenox) intentionally folded into rx_heparin
    'rx_heparin': ['heparin', 'enoxaparin', 'lovenox'],
    'rx_warfarin': ['warfarin', 'coumadin'],
    'rx_doac': ['apixaban', 'rivaroxaban', 'dabigatran', 'edoxaban'],
    'rx_antiplatelet': ['aspirin', 'clopidogrel', 'prasugrel', 'ticagrelor',
                        'dipyridamole'],
    'rx_hormone': ['estrogen', 'estradiol', 'tamoxifen', 'medroxyprogesterone',
                   'contracept', 'letrozole', 'anastrozole', 'leuprolide'],
}
# non-systemic heparin uses excluded from rx_heparin
HEPARIN_EXCLUDE = ['flush', 'dwell', 'lock', 'priming', 'crrt', 'iabp',
                   'hemodialysis']

CVC_PE_ITEMIDS = [224264, 224270, 224273, 225315, 229531, 229580]
VENT_PE_ITEMIDS = [225792, 225794, 226260]
BLOOD_IE_ITEMIDS = [220970, 220971, 220996, 225168, 225170, 226367, 226368,
                    226369, 227070, 227071, 227072]

base = pd.read_parquet('results_vte/vte_labels_v3.parquet')
base['admittime'] = pd.to_datetime(base['admittime'])

con = sqlite3.connect('mimic4.db')
upload = base[['hadm_id', 'admittime']].rename(columns={'admittime': 'admit'})
upload.to_sql('tmp_labels', con, if_exists='replace', index=False)
con.execute('CREATE INDEX idx_tmp_labels_hadm ON tmp_labels(hadm_id)')

# ---------------- medications (pharmacy, 24h window, start-only) ----------------
null_start, = con.execute(
    "SELECT count(*) FROM pharmacy p JOIN tmp_labels t ON p.hadm_id=t.hadm_id "
    "WHERE p.starttime IS NULL").fetchone()
print(f'pharmacy rows for cohort with NULL starttime (excluded): {null_start:,}')

med = pd.read_sql(
    "SELECT DISTINCT p.hadm_id, lower(p.medication) AS med FROM pharmacy p "
    "JOIN tmp_labels t ON p.hadm_id = t.hadm_id "
    "WHERE p.medication IS NOT NULL AND p.starttime IS NOT NULL "
    "AND p.starttime >= t.admit "
    "AND p.starttime <= datetime(t.admit, '+24 hours')",
    con)
print(f'windowed distinct (hadm, medication) pharmacy pairs: {len(med):,}')

rx = pd.DataFrame({'hadm_id': base.hadm_id})
for col, pats in RX_PATTERNS.items():
    mask = med.med.str.contains('|'.join(pats), regex=True)
    if col == 'rx_heparin':
        mask &= ~med.med.str.contains('|'.join(HEPARIN_EXCLUDE), regex=True)
    hits = med.loc[mask, 'hadm_id'].unique()
    rx[col] = rx.hadm_id.isin(hits).astype(int)
    print(f'{col}: {rx[col].mean():.2%}')

# ---------------- ICD procedures (date-granular window) ----------------
PROC_ICD = {
    'proc_cvc': ("(p.icd_version=10 AND (p.icd_code GLOB '02HV*' OR "
                 "p.icd_code GLOB '05HM*' OR p.icd_code GLOB '06HM*')) OR "
                 "(p.icd_version=9 AND (p.icd_code GLOB '3893*' OR "
                 "p.icd_code GLOB '3895*'))"),
    'proc_mechvent': ("(p.icd_version=10 AND p.icd_code GLOB '5A19*') OR "
                      "(p.icd_version=9 AND p.icd_code GLOB '967*')"),
    'proc_transfusion': ("(p.icd_version=10 AND p.icd_code GLOB '3023*') OR "
                         "(p.icd_version=9 AND p.icd_code GLOB '990*')"),
}
proc_hits = {c: set() for c in PROC_ICD}
for col, cond in PROC_ICD.items():
    ids = pd.read_sql(
        f"SELECT DISTINCT p.hadm_id FROM procedures_icd p "
        f"JOIN tmp_labels t ON p.hadm_id = t.hadm_id "
        f"WHERE ({cond}) AND date(p.chartdate) BETWEEN date(t.admit) "
        f"AND date(t.admit, '+24 hours')", con)
    proc_hits[col] |= set(ids.hadm_id)
    print(f'{col} procedures_icd: {len(ids)/len(base):.2%}')

# ---------------- ICU event-table supplements ----------------
for col, tbl, ids, tcol in [
        ('proc_cvc', 'procedureevents', CVC_PE_ITEMIDS, 'starttime'),
        ('proc_mechvent', 'procedureevents', VENT_PE_ITEMIDS, 'starttime'),
        ('proc_transfusion', 'inputevents', BLOOD_IE_ITEMIDS, 'starttime')]:
    df = pd.read_sql(
        f"SELECT DISTINCT e.hadm_id FROM {tbl} e "
        f"JOIN tmp_labels t ON e.hadm_id = t.hadm_id "
        f"WHERE e.itemid IN ({','.join(map(str, ids))}) "
        f"AND e.{tcol} >= t.admit "
        f"AND e.{tcol} <= datetime(t.admit, '+24 hours')", con)
    proc_hits[col] |= set(df.hadm_id)
    print(f'{col} {tbl}: {len(df)/len(base):.2%}')

for col in proc_hits:
    rx[col] = rx.hadm_id.isin(proc_hits[col]).astype(int)
    print(f'{col} combined: {rx[col].mean():.2%}')

# ---------------- icu_los_first24h (transfers overlap) ----------------
tr = pd.read_sql(
    "SELECT tr.hadm_id, tr.careunit, tr.intime, tr.outtime, t.admit "
    "FROM transfers tr JOIN tmp_labels t ON tr.hadm_id = t.hadm_id "
    "WHERE tr.intime IS NOT NULL AND tr.outtime IS NOT NULL "
    "AND tr.intime < datetime(t.admit, '+24 hours') "
    "AND tr.outtime > t.admit",
    con)
con.execute('DROP TABLE tmp_labels')
con.close()

icu_unit = tr.careunit.str.contains(
    'Intensive Care|ICU|CCU|CSRU', case=False, regex=True)
print('ICU-type careunits used:',
      sorted(tr.loc[icu_unit, 'careunit'].unique()))
tr = tr[icu_unit].copy()
for c in ['intime', 'outtime', 'admit']:
    tr[c] = pd.to_datetime(tr[c])
win_end = tr.admit + pd.Timedelta(hours=24)
lo = tr.intime.clip(lower=tr.admit)
hi = tr.outtime.clip(upper=win_end)
tr['overlap_h'] = (hi - lo).dt.total_seconds() / 3600
tr = tr[tr.overlap_h > 0]
icu_los = tr.groupby('hadm_id').overlap_h.sum()
print(f'icu_los_first24h>0: {(icu_los > 0).sum()/len(base):.2%} of cohort')

# ---------------- assemble ----------------
out = rx.merge(icu_los.rename('icu_los_first24h'), on='hadm_id', how='left')
out['icu_los_first24h'] = out['icu_los_first24h'].fillna(0.0)

n_expect = len(base)
assert len(out) == n_expect, f'row count {len(out)} != {n_expect}'
assert out.hadm_id.is_unique

out.to_parquet('results_vte/features_treatments_v3.parquet', index=False)
print(out.shape)

FLAG_COLS = list(RX_PATTERNS) + ['proc_cvc', 'proc_mechvent',
                                 'proc_transfusion']
print('\nflag rates:')
rates = {}
for c in FLAG_COLS:
    rates[c] = float(out[c].mean())
    print(f'  {c}: {rates[c]:.2%}')
    assert out[c].sum() > 0, f'{c} all zero -- join failure, recheck'

s = out.icu_los_first24h
print(f'\nicu_los_first24h: mean={s.mean():.2f}h median={s.median():.2f}h '
      f'p90={s.quantile(0.9):.2f}h max={s.max():.2f}h')
assert s.max() <= 24.0 + 1e-6, 'icu_los exceeds 24h window'

qc = f"""

## Task 5: 用药与操作特征（features_treatments_v3.parquet）

- 行数 {len(out):,}（= vte_labels，断言通过），列 {out.shape[1]}
- **用药主源 = pharmacy.medication**（lower contains 模糊匹配，starttime 在
  [admittime, +24h] 的起始记录，防泄漏；prescriptions.drug 抽查覆盖率几乎
  相同（heparin 54.54% vs 54.54%），选 pharmacy 因与计划一致且列更规整）。
  cohort 内 starttime 为 NULL 的 pharmacy 行 {null_start:,} 条（行级，已排除）
- rx_heparin **含 LMWH**（enoxaparin/lovenox 并入本列）；排除非全身用药：
  flush/dwell/lock/priming/CRRT/IABP/hemodialysis 环路用肝素
- 各列发生率：
"""
for c in FLAG_COLS:
    qc += f'  - {c}: {rates[c]:.2%}\n'
qc += f"""- **proc_cvc 源 = procedures_icd（ICD-10 02HV*/05HM*/06HM*，ICD-9
  3893*/3895*）∪ procedureevents 置管 itemid（PICC 224264、dialysis catheter
  224270、Presep 224273、tunneled Hickman 225315、tunneled access 229531、
  bedside line placement 229580）**；chartevents 'Central Line ordered on POE'
  itemid 225707/228036 在 cohort 24h 窗内 0 行，不可用
- **proc_mechvent 源 = procedures_icd（ICD-10 5A19*，ICD-9 967*）∪
  procedureevents（225792/225794/226260）**
- **proc_transfusion 源 = procedures_icd（ICD-10 3023*，ICD-9 990*）∪
  inputevents 血制品 itemid（FFP/plasma/PRBC/platelets，含 OR/PACU intake）**
- ICD 操作 chartdate 为日期粒度：窗 = chartdate ∈ [date(admittime),
  date(admittime+24h)]，可能多纳入入院当日早于 admittime 的操作（粒度限制）
- icu_los_first24h：transfers 中 ICU 类 careunit（匹配 'Intensive Care|ICU|
  CCU|CSRU'）区间与 [admittime, +24h] 交叠小时求和；mean={s.mean():.2f}h
  median={s.median():.2f}h p90={s.quantile(0.9):.2f}h（>0 占比
  {(s > 0).mean():.2%}）
- **核对偏差说明**：rx_heparin {rates['rx_heparin']:.1%} 高于计划预期带
  （10–30%）——MIMIC-IV 住院患者 VTE 预防性皮下肝素/LMWH 在入院 24h 内
  极普遍，排除 flush/dwell 等非全身用途后仍 >50%，为真实发生率而非连接
  错误（各源行数均已核查）；proc_cvc {rates['proc_cvc']:.1%}、
  proc_transfusion {rates['proc_transfusion']:.1%} 略低于预期带下限——本
  cohort 为全院入院（非 ICU 限定），24h 内置管/输血比例天然较低，已穷尽
  可用来源（procedures_icd + procedureevents/inputevents，chartevents 源
  覆盖为 0）
"""
with open('results_vte/m2_qc_notes.md', 'a') as f:
    f.write(qc)
print('QC notes appended.')
