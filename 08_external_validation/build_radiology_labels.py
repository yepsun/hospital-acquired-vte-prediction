#!/usr/bin/env python3
"""Radiology VTE labeling for the local hospital cohort.

Ports the MIMIC pe_vte logic (text_rules.classify_dvt_report /
pe_screen.screen_pe_report) to Chinese report text, scanning 检查诊断/检查所见.
Label semantics mirror MIMIC: positive / negative / uncertain / no_mention.
DVT excludes chronic & superficial (great/lesser saphenous) thrombus; calf
muscular (肌间) veins are counted as deep-vein positive (MIMIC-consistent).

Output: extval/radiology_labels.parquet
  columns: vis_id, vte_type(pe|dvt), label, reason, time, admit
"""
import re

import pandas as pd

# ---- DVT: deep-vein thrombosis keywords (Chinese) ----
DVT_KW = re.compile(
    r'(?:深静脉|股(?:总|浅|深)?静脉|腘静脉|胫(?:前|后)?静脉|'
    r'腓静脉|小腿肌间静脉|肌间静脉|髂(?:外|内|总)?静脉|'
    r'小腿.*静脉|上肢深静脉|上肢静脉|锁骨下静脉|腋静脉|头臂静脉|'
    r'静脉).{0,8}(?:血栓|栓塞)|血栓|附壁血栓|瘤栓|栓子')
# ---- superficial veins (excluded from DVT) ----
SUPERFICIAL = re.compile(r'(?:大隐静脉|小隐静脉|隐静脉|表浅静脉)')
# ---- negation (must be found in the SAME sentence as the thrombus keyword) ----
NEG = re.compile(
    r'(?:未见|未见明显|未见确切|未发现|未见明确|未见异常|'
    r'排除|未见血栓|未见确切血栓|阴性|未见明显血栓|'
    r'无(?:明显)?血栓|未探及|未见血流信号|管腔通畅|'
    r'血流充盈好|未见明确血栓|未见确切充盈缺损|未见充盈缺损)')
# ---- chronic / old (same-sentence) ----
CHRONIC = re.compile(
    r'(?:陈旧|陈旧性|慢性|机化|机化性|已机化|慢性血栓|'
    r'陈旧血栓|长期|稳定|较前无变化|无变化|无进展|未变化|'
    r'恢复期|钙化|管腔再通|附壁钙化|陈旧性血栓|陈旧血栓)')
# ---- acute override (new event) ----
ACUTE = re.compile(
    r'(?:新发|急性|新出现|近期|新鲜|急性血栓|新发血栓|'
    r'较前增宽|较前明显|新出现的|新近)')
# ---- hedge / uncertain (same-sentence) ----
UNCERTAIN = re.compile(
    r'(?:可能|考虑|不除外|可疑|不确定|疑似|待复查|'
    r'请结合临床|难以鉴别|不能排除|待定|'
    r'性质待定|待排除|较少见|难以除外|建议|除外|与.*鉴别)')
# ---- PE keywords ----
PE_KW = re.compile(
    r'(?:肺栓塞|肺动脉栓塞|肺.*血栓栓塞|肺动脉.*充盈缺损|'
    r'肺(?:动脉)?栓塞征象|肺栓塞形成|充盈缺损.*肺动脉|'
    r'肺动脉分支.*栓|肺灌注.*缺损|肺栓塞|肺栓塞可能)')
DEEP_DVT = re.compile(r'(?:深静脉|股静脉|腘静脉|胫|腓静脉|肌间静脉)')
# sentence splitter (Chinese sentence punctuation + newlines)
_SENT_SPLIT = re.compile(r'[。；;！!？?\n]+')


def _dvt_sentence(sent: str) -> str:
    """Return the judgement reason for one sentence containing a thrombus kw."""
    if SUPERFICIAL.search(sent) and not DEEP_DVT.search(sent):
        return 'dvt_superficial_only'
    if NEG.search(sent) and not ACUTE.search(sent):
        return 'dvt_negated'
    if CHRONIC.search(sent) and not ACUTE.search(sent):
        return 'dvt_chronic'
    if UNCERTAIN.search(sent):
        return 'dvt_hedged'
    return 'dvt_positive'


def _pe_sentence(sent: str) -> str:
    if NEG.search(sent) and not ACUTE.search(sent):
        return 'pe_negated_or_absent'
    if CHRONIC.search(sent) and not ACUTE.search(sent):
        return 'pe_chronic'
    if UNCERTAIN.search(sent):
        return 'pe_hedged'
    return 'pe_positive'


def classify_radiology(text: str) -> tuple[str, str]:
    """Return (label, reason) for a radiology report's 诊断/所见 text.

    Sentence-level: negation / chronic / hedge are evaluated within the same
    sentence as the thrombus keyword, avoiding cross-sentence misjudgment
    (e.g. '血栓形成' wrongly negated by '不除外' in a different clause).
    """
    t = re.sub(r'\s+', '', str(text or ''))
    if not t:
        return 'no_mention', 'empty'
    sents = _SENT_SPLIT.split(t)
    # process all thrombus-bearing sentences; any positive wins, else the
    # most informative (uncertain > negative) label
    pe_seen = dvt_seen = False
    any_pos = False
    labels = []
    for s in sents:
        if not s:
            continue
        if PE_KW.search(s):
            pe_seen = True
            r = _pe_sentence(s)
            labels.append(r)
            if r == 'pe_positive':
                any_pos = True
        elif DVT_KW.search(s):
            dvt_seen = True
            r = _dvt_sentence(s)
            labels.append(r)
            if r == 'dvt_positive':
                any_pos = True
    if any_pos:
        return 'positive', ('pe_positive' if pe_seen else 'dvt_positive')
    if not (pe_seen or dvt_seen):
        return 'no_mention', 'no_keyword'
    # no positive; prefer uncertain over negative
    if any('hedged' in r or r == 'pe_hedged' or r == 'dvt_hedged'
           for r in labels):
        return 'uncertain', next(r for r in labels if 'hedged' in r
                                  or r == 'pe_hedged' or r == 'dvt_hedged')
    return 'negative', labels[0]


def build_radiology_labels():
    co = pd.read_parquet('extval/cohort.parquet', columns=['vis_id', 'admit'])
    rows = []
    files = ['高危VTE患者/检查结果.csv',
             '非高危VTE患者/分层抽样患者检查结果.csv']
    for f in files:
        df = pd.read_csv(f, encoding='gb18030',
                         usecols=['就诊号', '检查所见', '检查诊断', '检查时间',
                                  '出报告时间'])
        df['就诊号'] = df['就诊号'].astype(str)
        for col in ['检查所见', '检查诊断']:
            df[col] = df[col].fillna('').astype(str)
        text = df['检查诊断'] + '\n' + df['检查所见']
        df['label'], df['reason'] = zip(*text.map(classify_radiology))
        df['time'] = pd.to_datetime(df['检查时间'], errors='coerce') \
            .fillna(pd.to_datetime(df['出报告时间'], errors='coerce'))
        df = df[df['label'].isin(['positive', 'uncertain'])]
        rows.append(df)
    allr = pd.concat(rows, ignore_index=True)
    allr = allr.rename(columns={'就诊号': 'vis_id'})
    allr = allr.merge(co, on='vis_id', how='inner')
    # PE vs DVT type inference from report text
    pe_mask = allr['检查诊断'].str.contains('肺栓塞|肺动脉|肺', na=False) | \
        allr['检查所见'].str.contains('肺栓塞|肺动脉|肺', na=False)
    allr['vte_type'] = 'dvt'
    allr.loc[pe_mask, 'vte_type'] = 'pe'
    allr = allr[['vis_id', 'vte_type', 'label', 'reason', 'time', 'admit']].copy()
    allr.to_parquet('extval/radiology_labels.parquet', index=False)
    print('radiology labels:', allr.shape)
    print(allr.groupby(['vte_type', 'label']).size())
    return allr


if __name__ == '__main__':
    build_radiology_labels()
