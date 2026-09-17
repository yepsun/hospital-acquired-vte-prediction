#!/usr/bin/env python3
"""Build the external-validation cohort from the local hospital VTE dataset.

Cohort = all admissions with full clinical data:
  - 高危 (high-risk) group: 35,870 admissions (complete data)
  - 非高危 (non-high-risk) group: 7,072 stratified-sample admissions

VTE outcome (aligned to the MIMIC medRxiv primary definition):
  - Cases: radiology-confirmed acute PE / DVT, judged by a locally deployed
    DeepSeek model over imaging reports (extval/radiology_llm.parquet,
    llm_verdict == 'y'). Following the MIMIC primary cohort, VTE imaged within
    the first 24 h of admission (incl. ED imaging) is retained as events; only
    VTE diagnosed strictly before the index admission is excluded.
  - Controls: all remaining admissions without any VTE ICD code (I26*/I80*/I82*).
    Admissions with a VTE ICD code but no radiological confirmation are excluded
    (neither cases nor controls), mirroring the MIMIC ICD-only exclusion.
  - Base filter (mirrors MIMIC): length of stay > 24 h.
  - Prior VTE excluded (history codes + prior admissions with VTE).

Output: extval/cohort.parquet
  columns: vis_id, pat_id, risk_group, admit, disch, vte_outcome,
           prior_vte_any, prior_admissions_n, age, male
"""
import re

import pandas as pd
import numpy as np

# MIMIC-consistent VTE ICD-10 subset for CONTROL EXCLUSION only (cases are
# radiology-confirmed): PE I26*, DVT I801/I802/I803/I824*. Used to exclude
# ICD-only admissions from the control group, matching the MIMIC ICD-only
# exclusion. Note: NOT the broad I80*/I82* prefix — I84 (haemorrhoids),
# I85 (oesophageal varices), I87 (venous insufficiency), I83 (varicose veins),
# I81 (portal/other-vein thrombosis) are not lower-extremity DVT.
PE_ICD = ('I26',)
DVT_ICD = ('I801', 'I802', 'I803', 'I824')
PRIOR_HISTORY_TERMS = ('肺栓塞个人史', '肺栓塞史', '急性肺栓塞史', '深静脉血栓史',
                       '下肢深静脉血栓史', '血栓史', '静脉血栓史')
# any 编目诊断名称 containing these word-pairs signals a history diagnosis
HIST_PAIR = ('史')
VTE_WORDS = ('栓', '血栓', '静脉')


def load_dx(path):
    dx = pd.read_csv(path, encoding='gb18030',
                     usecols=['就诊号', '编目诊断编码', '编目诊断名称',
                              '编目诊断类型描述'])
    dx['就诊号'] = dx['就诊号'].astype(str)
    return dx


def main():
    hi = pd.read_csv('高危VTE患者/就诊信息.csv', encoding='gb18030')
    nh = pd.read_csv('非高危VTE患者/分层抽样患者就诊信息.csv', encoding='gb18030')
    hi['risk_group'] = 'high'
    nh['risk_group'] = 'non_high'
    co = pd.concat([hi, nh], ignore_index=True)
    co['vis_id'] = co['就诊号'].astype(str)
    co['pat_id'] = co['患者号'].astype(str)
    co['admit'] = pd.to_datetime(co['入院时间'])
    co['disch'] = pd.to_datetime(co['出院时间'])
    co = co[['vis_id', 'pat_id', 'risk_group', 'admit', 'disch']].copy()

    # ---- base filter: length of stay > 24 h (mirrors MIMIC) -----------------
    co['los_h'] = (co['disch'] - co['admit']).dt.total_seconds() / 3600
    n_los = int((co['los_h'] <= 24).sum())
    co = co[co['los_h'] > 24].copy()
    print(f'LOS<=24h admissions excluded: {n_los}')

    # ---- VTE ICD codes (discharge diagnoses) for CONTROL EXCLUSION only ----
    def vte_icd_visits(path):
        dx = load_dx(path)
        disc = dx[dx['编目诊断类型描述'].astype(str).str.contains('出院诊断', na=False)]
        code = disc['编目诊断编码'].fillna('').astype(str)
        pe = code.str.match('I26')
        dvt = code.str.match('|'.join(DVT_ICD))
        return set(disc.loc[pe | dvt, '就诊号'])

    icd_vis = (vte_icd_visits('高危VTE患者/首页诊断.csv')
               | vte_icd_visits('非高危VTE患者/分层抽样患者首页诊断.csv'))
    co['vte_icd'] = co['vis_id'].isin(icd_vis).astype(int)

    # ---- radiology-confirmed VTE (cases) via local DeepSeek adjudication ----
    # MIMIC VTE = acute PE OR lower-extremity DVT. Restrict LLM-'y' reports to
    # these two types; exclude upper-extremity (颈/锁骨下/腋/肱/尺/桡), central
    # (下腔/上腔/髂), and visceral (门/脾/肠系膜/肝) thrombosis, which are not
    # PE or lower-extremity DVT.
    PE_RE = re.compile(r'肺|肺动脉|肺栓塞')
    LEG_DVT_RE = re.compile(r'股|腘|胫|腓静脉|小腿肌间|深静脉|髂外|髂总|髂静脉|下肢')
    rad = pd.read_parquet('extval/radiology_llm.parquet')
    rad['time'] = pd.to_datetime(rad['检查时间'], errors='coerce') \
        .fillna(pd.to_datetime(rad['出报告时间'], errors='coerce'))
    rad = rad[rad['llm_verdict'] == 'y'].copy()
    txt = rad['检查诊断'].fillna('').astype(str) + ' ' + \
        rad['检查所见'].fillna('').astype(str)
    keep = txt.apply(lambda t: bool(PE_RE.search(t) or LEG_DVT_RE.search(t)))
    rad = rad[keep].copy()
    # retains within-24h imaging events (MIMIC primary); excludes strictly
    # pre-admission diagnosis (time < admit)
    rad = rad.merge(co[['vis_id', 'admit']], left_on='就诊号', right_on='vis_id',
                    how='inner')
    rad = rad.dropna(subset=['admit'])
    rad_after = rad[(rad['time'].notna()) & (rad['time'] >= rad['admit'])]
    case_vis = set(rad_after['vis_id'])

    co['vte_radiology'] = co['vis_id'].isin(case_vis).astype(int)

    # ---- assemble cases / controls -----------------------------------------
    # cases = radiology-confirmed; controls = no VTE ICD and no radiology case;
    # ICD-only admissions (VTE ICD but no radiology confirmation) are excluded
    co['vte_outcome'] = co['vte_radiology']
    icd_only = (co['vte_icd'] == 1) & (co['vte_radiology'] == 0)
    print('radiology-confirmed cases:', int(co['vte_radiology'].sum()))
    print('ICD-only (excluded as neither case nor control):', int(icd_only.sum()))
    co = co[~icd_only].copy()
    print('cohort after ICD-only exclusion:', len(co))
    print('vte event rate (of cases+controls):', co['vte_outcome'].mean())

    # ---- prior VTE from THIS admission's history codes / text --------------
    def history_visits(path):
        dx = load_dx(path)
        name = dx['编目诊断名称'].fillna('').astype(str)
        # Z86.706 PE personal history, or name contains 史 AND (栓|血栓|静脉)
        mask = (dx['编目诊断编码'].fillna('').astype(str) == 'Z86.706')
        mask |= name.str.contains(HIST_PAIR, na=False) & \
            name.str.contains('|'.join(VTE_WORDS), na=False)
        return set(dx.loc[mask, '就诊号'])

    hist_vis = (history_visits('高危VTE患者/首页诊断.csv')
                | history_visits('非高危VTE患者/分层抽样患者首页诊断.csv'))
    co['prior_vte_this_adm'] = co['vis_id'].isin(hist_vis).astype(int)

    # ---- prior VTE from EARLIER admission of same patient ------------------
    co = co.sort_values(['pat_id', 'admit', 'vis_id'])
    co['prior_vte_adm'] = (
        co.groupby('pat_id')['vte_outcome'].cumsum().shift(1).fillna(0) > 0
    ).astype(int)

    co['prior_vte_any'] = ((co['prior_vte_this_adm'] == 1)
                           | (co['prior_vte_adm'] == 1)).astype(int)

    # ---- prior admissions count --------------------------------------------
    co['prior_admissions_n'] = co.groupby('pat_id').cumcount()

    # ---- dedupe visits appearing in both high & non-high files ------------
    # (same visit, same admit time) -> keep the high-risk assignment
    co = co.sort_values(['vis_id', 'risk_group'])
    co = co.drop_duplicates('vis_id', keep='first')
    print('after dedupe cohort rows:', len(co))

    # ---- demographics from EMR (parsed across all record types) -------------
    demo = pd.read_parquet(
        '/var/folders/dp/xkwn23650r3g8fml_ldjh_jr0000gp/T/opencode/demo_all.parquet')
    demo['vis_id'] = demo['vis'].astype(str)
    demo = demo.drop_duplicates('vis_id', keep='first')
    demo['male'] = (demo['sex'] == '男').astype(int)
    co = co.merge(demo[['vis_id', 'age', 'male']], on='vis_id', how='left')

    co.to_parquet('extval/cohort.parquet', index=False)
    print('cohort rows:', len(co))
    print('with age:', co['age'].notna().sum(),
          'with male:', co['male'].notna().sum())
    print(co.groupby('risk_group').agg(
        n=('vis_id', 'count'),
        events=('vte_outcome', 'sum'),
        event_rate=('vte_outcome', 'mean'),
        prior_vte=('prior_vte_any', 'mean'),
        age_med=('age', 'median'),
        male=('male', 'mean')))


if __name__ == '__main__':
    main()
