#!/usr/bin/env python3
"""Build MIMIC-aligned static features for the external-validation cohort.

Maps the 57 MIMIC main static features to the local hospital data using the
same item definitions as scripts/85a-85d (labs = 24h median+count; comorbid =
index-admission ICD prefixes with VTE-blacklist stripped; treatments =
medications/procedures within 24h; surgery = surgery records; ICU = ICU
evidence proxy).

Output: extval/features.parquet  (one row per cohort vis_id)
"""
import re

import numpy as np
import pandas as pd

CO = 'extval/cohort.parquet'
OUT = 'extval/features.parquet'

# ---------------- lab item -> MIMIC key mapping (Chinese names) --------------
LAB_MAP = {
    'wbc': ['白细胞总数', '△白细胞', '白细胞'],
    'neutrophils': ['中性粒细胞绝对值'],
    'lymphocytes': ['淋巴细胞绝对值'],
    'platelet': ['△血小板', '血小板'],
    'rdw': ['红细胞体积分布宽度'],
    'hemoglobin': ['△血红蛋白', '血红蛋白'],
    'ddimer': ['D-二聚体'],
    'inr': ['国际标准化比值'],
    'pt': ['凝血酶原时间'],
    'aptt': ['活化部分凝血活酶时间'],
    'fibrinogen': ['纤维蛋白原'],
    'creatinine': ['肌酐(酶法)'],
    'bun': ['尿素'],
    'albumin': ['白蛋白'],
    'crp': ['C反应蛋白'],
}
# exact-ish exclusion substrings to avoid mis-mapping (e.g., 尿素酶, 尿肌酐)
LAB_EXCLUDE = ['尿素酶', '百分比', '平均', '压积', '大血小板', '低值',
               '计算', '介素', '抗体', '蛋白电泳', '糖化', '24小时尿',
               '尿肌酐', '尿尿素', '尿微量', '脑脊液', '血清']

# Unit conversion to MIMIC-IV units (only for analytes whose MEDIAN enters the
# 57-feature set). MIMIC units: wbc/neut/lymph/platelet = K/uL (==x10^9/L),
# hemoglobin = g/dL, creatinine = mg/dL, BUN = mg/dL, RDW = %, INR = ratio,
# PT/APTT = s.
#   allowed unit -> factor to multiply local value to get MIMIC units
LAB_UNIT = {
    'wbc': {'×10^9/L': 1.0, '10^9/L': 1.0, 'x10^9/L': 1.0},
    'neutrophils': {'×10^9/L': 1.0, '10^9/L': 1.0},
    'lymphocytes': {'×10^9/L': 1.0, '10^9/L': 1.0},
    'platelet': {'×10^9/L': 1.0, '10^9/L': 1.0},
    'rdw': {'%': 1.0},
    'hemoglobin': {'g/L': 0.1},      # g/L -> g/dL
    'inr': {'': 1.0},
    'pt': {'s': 1.0, 'sec': 1.0},
    'aptt': {'s': 1.0, 'sec': 1.0},
    'creatinine': {'μmol/L': 1.0 / 88.4, 'umol/L': 1.0 / 88.4},  # -> mg/dL
    'bun': {'mmol/L': 2.8},          # mmol/L urea-N -> mg/dL BUN
    'albumin': {'g/L': 0.1},         # g/L -> g/dL
    'crp': {'mg/L': 1.0},
}
# counts-only analytes: unit irrelevant (presence count), keep all
COUNT_ONLY = {'ddimer', 'fibrinogen'}


def map_lab_name(name):
    """Return MIMIC key for a Chinese lab item name, or None."""
    name = str(name)
    for exc in LAB_EXCLUDE:
        if exc in name:
            return None
    for key, aliases in LAB_MAP.items():
        for a in aliases:
            if a in name:
                return key
    return None


def map_lab_unit(key, unit):
    """Return conversion factor for a lab key + unit, or None if unmapped."""
    if key in COUNT_ONLY:
        return 1.0
    u = str(unit).strip() if unit is not None and str(unit) != 'nan' else ''
    fac = LAB_UNIT.get(key)
    if fac is None:
        return None
    for k, v in fac.items():
        if u == k:
            return v
    return None  # unmapped unit -> drop


def extract_labs():
    """Median + count per analyte within [admit, admit+24h]."""
    co = pd.read_parquet(CO)[['vis_id', 'admit']]
    agg = {f'{k}_vals': [] for k in LAB_MAP}
    agg_vis = {f'{k}_vis': set() for k in LAB_MAP}
    files = ['高危VTE患者/检验结果1.csv', '高危VTE患者/检验结果2.csv',
             '高危VTE患者/检验结果3.csv',
             '非高危VTE患者/分层抽样患者检验结果.csv']
    for f in files:
        for chunk in pd.read_csv(f, encoding='gb18030', chunksize=1_000_000,
                                 usecols=['就诊号', '项目中文名称', '定量结果',
                                          '结果单位', '标本采样时间']):
            chunk['就诊号'] = chunk['就诊号'].astype(str)
            chunk['key'] = chunk['项目中文名称'].map(map_lab_name)
            chunk = chunk[chunk['key'].notna()]
            if chunk.empty:
                continue
            chunk['st'] = pd.to_datetime(chunk['标本采样时间'], errors='coerce')
            chunk['val'] = pd.to_numeric(chunk['定量结果'], errors='coerce')
            chunk = chunk[chunk['val'].notna() & chunk['st'].notna()]
            # unit conversion to MIMIC units; drop unmapped-unit rows
            chunk['fac'] = [map_lab_unit(k, u) for k, u in
                            zip(chunk['key'], chunk['结果单位'])]
            chunk = chunk[chunk['fac'].notna()]
            chunk['val'] = chunk['val'] * chunk['fac']
            chunk = chunk.merge(co, left_on='就诊号', right_on='vis_id',
                                how='inner')
            win = chunk[(chunk['st'] >= chunk['admit']) &
                        (chunk['st'] <= chunk['admit'] + pd.Timedelta(hours=24))]
            for key in LAB_MAP:
                sub = win[win['key'] == key]
                if len(sub):
                    agg[f'{key}_vals'].append(
                        sub[['vis_id', 'val']].groupby('vis_id')['val'].median())
                    agg_vis[f'{key}_vis'] |= set(sub['vis_id'])
            del chunk, win
    # assemble
    med = pd.DataFrame({'vis_id': co['vis_id']})
    for key in LAB_MAP:
        col = f'lab_{key}_median'
        if len(agg[f'{key}_vals']):
            m = pd.concat(agg[f'{key}_vals']).groupby(level=0).first()
            med = med.merge(m.rename(col), left_on='vis_id', right_index=True,
                            how='left')
        else:
            med[col] = np.nan
        med[f'lab_{key}_count'] = med['vis_id'].isin(
            agg_vis[f'{key}_vis']).astype(int)
    # physical clipping (same ranges as MIMIC)
    clip = {'wbc': (0.1, 500), 'platelet': (1, 2000), 'creatinine': (0.1, 30),
            'hemoglobin': (1, 25), 'inr': (0.5, 20), 'pt': (5, 150),
            'aptt': (10, 300), 'albumin': (0.5, 7), 'bun': (1, 300)}
    for k, (lo, hi) in clip.items():
        c = f'lab_{k}_median'
        if c in med:
            bad = med[c].notna() & ((med[c] < lo) | (med[c] > hi))
            med.loc[bad, c] = np.nan
            print(f'clip {c}: {int(bad.sum())} -> NaN')
    return med


def extract_bmi():
    """BMI from vitals height/weight, falling back to surgery records."""
    co = pd.read_parquet(CO)[['vis_id']]
    rows = []
    # vitals
    for f, s3 in [('高危VTE患者/生命体征.csv', False),
                  ('非高危VTE患者/分层抽样患者生命体征.csv', False)]:
        v = pd.read_csv(f, encoding='gb18030', usecols=['就诊号', '体征类型名称',
                                                        '体征数值'])
        v['就诊号'] = v['就诊号'].astype(str)
        ht = v[v['体征类型名称'].str.contains('身高', na=False)]
        wt = v[v['体征类型名称'].str.contains('体重', na=False)]
        ht = ht.groupby('就诊号')['体征数值'].median()
        wt = wt.groupby('就诊号')['体征数值'].median()
        rows.append((ht, wt))
    ht_all = pd.concat([r[0] for r in rows]).groupby(level=0).first()
    wt_all = pd.concat([r[1] for r in rows]).groupby(level=0).first()
    bmi = pd.DataFrame({'vis_id': ht_all.index,
                        'height_m': ht_all.values / 100.0,
                        'weight_kg': wt_all.reindex(ht_all.index).values})
    bmi['bmi'] = bmi['weight_kg'] / (bmi['height_m'] ** 2)
    bmi = bmi[bmi['bmi'].between(10, 80)]
    # surgery fallback
    surg_h = []
    for f in ['高危VTE患者/手术麻醉记录.csv',
              '非高危VTE患者/分层抽样患者手术麻醉记录.csv']:
        s = pd.read_csv(f, encoding='gb18030', usecols=['就诊号', '手术时身高',
                                                        '手术时体重'])
        s['就诊号'] = s['就诊号'].astype(str)
        s['手术时身高'] = pd.to_numeric(s['手术时身高'], errors='coerce')
        s['手术时体重'] = pd.to_numeric(s['手术时体重'], errors='coerce')
        surg_h.append(s)
    s = pd.concat(surg_h, ignore_index=True)
    s = s.dropna(subset=['手术时身高', '手术时体重'])
    s = s[(s['手术时身高'] > 120) & (s['手术时身高'] < 220) &
          (s['手术时体重'] > 25) & (s['手术时体重'] < 300)]
    s['bmi'] = s['手术时体重'] / (s['手术时身高'] / 100) ** 2
    sb = s.groupby('就诊号')['bmi'].median()
    out = co.merge(bmi[['vis_id', 'bmi']], on='vis_id', how='left')
    out = out.merge(sb.rename('bmi_surg'), left_on='vis_id', right_index=True,
                    how='left')
    out['bmi'] = out['bmi'].fillna(out['bmi_surg'])
    out['bmi'] = out['bmi'].clip(10, 80)
    return out[['vis_id', 'bmi']]


def extract_comorbid():
    """Comorbidity flags from index-admission ICD codes (same prefixes as MIMIC)."""
    co = pd.read_parquet(CO)[['vis_id']]
    COMORB = {
        'cancer_active': ['C'],
        'heart_failure': ['I50'],
        'copd': ['J40', 'J41', 'J42', 'J43', 'J44'],
        'chronic_liver': ['K70', 'K71', 'K72', 'K73', 'K74'],
        'ckd': ['N18'],
        'diabetes': ['E10', 'E11', 'E12', 'E13', 'E14'],
        'stroke': ['I63', 'I64'],
        'mi': ['I21', 'I22'],
        'obesity_icd': ['E66'],
        'varicose': ['I83'],
        'thrombophilia': ['D685', 'D686'],
        'infection_severe': ['A40', 'A41'],
        'rheumatologic': ['M05', 'M06', 'M32', 'M33', 'M34'],
    }
    VTE_BLACKLIST = ('I26', 'I80', 'I82')
    flags = {c: set() for c in COMORB}
    trauma = set()
    for f in ['高危VTE患者/首页诊断.csv',
              '非高危VTE患者/分层抽样患者首页诊断.csv']:
        dx = pd.read_csv(f, encoding='gb18030', usecols=['就诊号', '编目诊断编码'])
        dx['就诊号'] = dx['就诊号'].astype(str)
        dx['code'] = dx['编目诊断编码'].fillna('').astype(str).str.upper() \
            .str.replace('.', '', regex=False)
        # strip VTE blacklist (outcome leakage guard, as MIMIC)
        bl = dx['code'].str.startswith(VTE_BLACKLIST)
        print(f'{f}: blacklist rows stripped {int(bl.sum())}')
        dx = dx[~bl]
        for c, prefs in COMORB.items():
            hit = dx['code'].str.startswith(tuple(prefs))
            flags[c] |= set(dx.loc[hit, '就诊号'])
        trauma |= set(dx[dx['code'].str.startswith(('S', 'T'))]['就诊号'])
    out = pd.DataFrame({'vis_id': co['vis_id']})
    for c, s in flags.items():
        out[c] = out['vis_id'].isin(s).astype(int)
    out['trauma_flag'] = out['vis_id'].isin(trauma).astype(int)
    return out


def extract_treatments():
    """Medication & procedure flags within [admit, admit+24h] + ICU proxy.

    MIMIC-aligned: only verified/executed orders (医嘱状态说明 in {核实, 执行})
    are counted; non-systemic heparin uses (flush/dwell/lock/CRRT/hemodialysis
    via route or text) are excluded from rx_heparin, mirroring the pharmacy
    starttime-24h semantics of the MIMIC AJM model.
    """
    co = pd.read_parquet(CO)[['vis_id', 'admit']]
    STATUS_KEEP = ['核实', '执行']
    RX = {
        'rx_heparin': ['肝素', '依诺肝素', '低分子肝素', '那屈肝素', '达肝素',
                       '克赛', '普洛静'],
        'rx_warfarin': ['华法林'],
        'rx_doac': ['利伐沙班', '达比加群', '阿哌沙班', '依度沙班', '艾多沙班'],
        'rx_antiplatelet': ['阿司匹林', '氯吡格雷', '替格瑞洛', '普拉格雷',
                            '双嘧达莫'],
        'rx_hormone': ['雌激素', '雌二醇', '他莫昔芬', '甲羟孕酮', '避孕',
                       '来曲唑', '阿那曲唑', '亮丙瑞林'],
    }
    HEP_EXCLUDE = ['肝素帽', '肝素封管', '肝素冲洗', '肝素锂', '肝素钠封管',
                   '采血管', '肝素化管路', '抗凝管']
    # non-systemic administration routes for anticoagulation (MIMIC excludes
    # flush/dwell/lock/CRRT/hemodialysis circuits)
    ROUTE_EXCLUDE = ['封管', '冲洗', '血滤', '透析', '造影', '手术用', '肝素帽',
                     '抗凝管', '采血管']
    PROC = {
        'proc_cvc': ['深静脉', '中心静脉', 'PICC', 'picc', '锁骨下静脉置管',
                     '股静脉置管', '颈内静脉置管', '静脉置管'],
        'proc_mechvent': ['呼吸机', '机械通气', '无创通气', '气管插管'],
        'proc_transfusion': ['输血', '红细胞', '血浆', '血小板', '冷沉淀'],
    }
    hits = {c: set() for c in list(RX) + list(PROC)}
    icu_vis = set()
    files = ['高危VTE患者/住院医嘱1.csv', '高危VTE患者/住院医嘱2.csv',
             '高危VTE患者/住院医嘱3.csv',
             '非高危VTE患者/分层抽样患者住院医嘱.csv']
    for f in files:
        for chunk in pd.read_csv(f, encoding='gb18030', chunksize=500_000,
                                 usecols=['就诊号', '医嘱正文', '医嘱类别描述',
                                          '医嘱开立时间', '医嘱状态说明',
                                          '给药途径和方法']):
            chunk['就诊号'] = chunk['就诊号'].astype(str)
            chunk = chunk[chunk['医嘱状态说明'].isin(STATUS_KEEP)]
            chunk['txt'] = chunk['医嘱正文'].fillna('').astype(str)
            chunk['route'] = chunk['给药途径和方法'].fillna('').astype(str)
            chunk['ot'] = pd.to_datetime(chunk['医嘱开立时间'], errors='coerce')
            chunk = chunk.merge(co, left_on='就诊号', right_on='vis_id',
                                how='inner')
            win = chunk[(chunk['ot'] >= chunk['admit']) &
                        (chunk['ot'] <= chunk['admit'] + pd.Timedelta(hours=24))]
            for c, pats in RX.items():
                m = win['txt'].str.contains('|'.join(pats), regex=True)
                if c == 'rx_heparin':
                    m &= ~win['txt'].str.contains('|'.join(HEP_EXCLUDE),
                                                  regex=True)
                    m &= ~win['route'].str.contains('|'.join(ROUTE_EXCLUDE),
                                                    regex=True)
                hits[c] |= set(win.loc[m, 'vis_id'])
            for c, pats in PROC.items():
                m = win['txt'].str.contains('|'.join(pats), regex=True)
                hits[c] |= set(win.loc[m, 'vis_id'])
            # ICU evidence anywhere in stay (bed fee / ICU care)
            ic = chunk[chunk['txt'].str.contains(
                '重症监护病房床位费|重症监护护理|ICU床费', regex=True)]
            icu_vis |= set(ic['vis_id'])
            del chunk, win
    out = pd.DataFrame({'vis_id': co['vis_id']})
    for c in list(RX) + list(PROC):
        out[c] = out['vis_id'].isin(hits[c]).astype(int)
    out['icu_flag'] = out['vis_id'].isin(icu_vis).astype(int)
    return out


def extract_surgery():
    """surgery_flag from surgery/anesthesia records (any surgery this admission)."""
    co = pd.read_parquet(CO)[['vis_id']]
    surg = set()
    for f in ['高危VTE患者/手术麻醉记录.csv',
              '非高危VTE患者/分层抽样患者手术麻醉记录.csv']:
        s = pd.read_csv(f, encoding='gb18030', usecols=['就诊号', '手术名称'])
        s['就诊号'] = s['就诊号'].astype(str)
        surg |= set(s.dropna(subset=['手术名称'])['就诊号'])
    out = pd.DataFrame({'vis_id': co['vis_id']})
    out['surgery_flag'] = out['vis_id'].isin(surg).astype(int)
    return out


def main():
    co = pd.read_parquet(CO)[['vis_id', 'age', 'male', 'vte_outcome',
                              'prior_vte_any', 'prior_admissions_n',
                              'risk_group']]
    print('== labs ==', flush=True)
    labs = extract_labs()
    print('== bmi ==', flush=True)
    bmi = extract_bmi()
    print('== comorbid ==', flush=True)
    com = extract_comorbid()
    print('== treatments ==', flush=True)
    tr = extract_treatments()
    print('== surgery ==', flush=True)
    su = extract_surgery()

    df = co.merge(labs, on='vis_id', how='left') \
        .merge(bmi, on='vis_id', how='left') \
        .merge(com, on='vis_id', how='left') \
        .merge(tr, on='vis_id', how='left') \
        .merge(su, on='vis_id', how='left')

    # administrative features not directly available locally:
    df['adm_elective'] = np.nan
    df['adm_emergency'] = np.nan
    df['first_careunit_icu'] = df['icu_flag']
    # icu_los_first24h: 24h if ICU evidence present, else 0 (proxy; MIMIC used
    # exact ICU transfer hours). Local ICU timing unavailable -> conservative 0/24.
    df['icu_los_first24h'] = (df['icu_flag'] * 24.0).astype(float)

    df.to_parquet(OUT, index=False)
    print('\nsaved', OUT, df.shape)
    print('\nfeature coverage (non-null %):')
    print(df.notna().mean().round(3).to_string())
    print('\nflag rates:')
    flags = ['cancer_active', 'heart_failure', 'copd', 'chronic_liver', 'ckd',
             'diabetes', 'stroke', 'mi', 'obesity_icd', 'varicose',
             'thrombophilia', 'infection_severe', 'rheumatologic',
             'trauma_flag', 'surgery_flag', 'rx_heparin', 'rx_warfarin',
             'rx_doac', 'rx_antiplatelet', 'rx_hormone', 'proc_cvc',
             'proc_mechvent', 'proc_transfusion', 'first_careunit_icu']
    for c in flags:
        print(f'  {c:22s} {df[c].mean():.4f}')


if __name__ == '__main__':
    main()
