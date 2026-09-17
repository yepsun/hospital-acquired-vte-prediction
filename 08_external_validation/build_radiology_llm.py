#!/usr/bin/env python3
"""LLM adjudication of local hospital radiology reports for VTE.

Ports the MIMIC human-review adjudication approach (scripts/60) to the local
hospital imaging reports. Every report containing a VTE keyword (血栓/栓塞/
肺栓塞/深静脉/充盈缺损) is judged by a locally deployed DeepSeek model for
whether it documents acute PE or acute DVT.

Judgement semantics mirror MIMIC (SYSTEM prompt in scripts/60):
  - y  = acute PE or acute DVT (intraluminal thrombus/embolus)
  - n  = negative / rules out / only chronic-old / prophylaxis / indication
  - uncertain = genuinely ambiguous
Excludes chronic/old and superficial (saphenous) thrombus; calf muscular
(肌间) veins count as DVT.

Output:
  extval/radiology_llm.jsonl         (resumable per-report judgements)
  extval/radiology_llm.parquet       (merged: vis_id, report, verdict, reason)

Usage:
  python3 extval/build_radiology_llm.py            # run + merge
  python3 extval/build_radiology_llm.py --merge    # merge only
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from build_radiology_labels import classify_radiology

BASE = 'https://api.deepseek.com/v1'
MODEL = 'deepseek-v4-flash'
API_KEY = os.environ.get('DEEPSEEK_API_KEY')
MAX_TOKENS = 1500
INPUT_CAP = 6000
WORKERS = int(os.environ.get('LLM_WORKERS', '8'))
OUT_JSONL = 'extval/radiology_llm.jsonl'
OUT_PARQUET = 'extval/radiology_llm.parquet'

# VTE keyword candidates for LLM adjudication
VTE_KW = re.compile(
    r'血栓|栓塞|肺栓塞|深静脉|充盈缺损|血栓栓塞|静脉血栓|'
    r'瘤栓|栓子|血栓栓塞症|静脉内低回声|不能压瘪|不可压瘪|'
    r'血流信号充盈缺损|管腔扩张.*栓|栓.*充盈缺损')

# Only reports the rule marks positive OR uncertain (ambiguous) go to the LLM;
# clear negatives (未见/除外/陈旧/表面静脉) are skipped to limit LLM volume.
ADJUDICATE = {'positive', 'uncertain'}

SYSTEM = (
    "You are a senior radiologist adjudicating whether a radiology report "
    "documents a diagnosis of acute venous thromboembolism (VTE): acute "
    "pulmonary embolism (PE) or acute deep vein thrombosis (DVT).\n"
    "Rules:\n"
    "- Answer 'y' ONLY if the report documents POSITIVE findings of acute PE "
    "or acute DVT (e.g. intraluminal thrombus/embolus in pulmonary arteries "
    "or lower-extremity deep veins, including calf muscular veins).\n"
    "- Answer 'n' if the report is negative (未见/无/未见明显), rules out VTE, "
    "mentions only chronic/old/unchanged findings (陈旧/慢性/机化), describes "
    "only superficial-vein thrombus (大隐静脉/小隐静脉), or only states "
    "prophylaxis/risk factors/clinical indication without positive findings.\n"
    "- If genuinely ambiguous, answer 'uncertain'.\n"
    "Respond ONLY with JSON: {\"verdict\": \"y\"|\"n\"|\"uncertain\", "
    "\"reason\": \"concise reason citing the key finding (in Chinese)\"}")

REPORT_COLS = ['就诊号', '检查诊断', '检查所见', '检查项目名称', '检查时间', '出报告时间']


def load_candidates():
    rows = []
    files = ['高危VTE患者/检查结果.csv',
             '非高危VTE患者/分层抽样患者检查结果.csv']
    for f in files:
        df = pd.read_csv(f, encoding='gb18030', usecols=REPORT_COLS)
        df['就诊号'] = df['就诊号'].astype(str)
        text = df['检查诊断'].fillna('').astype(str) + '\n' + \
            df['检查所见'].fillna('').astype(str)
        df = df[text.str.contains(VTE_KW, regex=True)].copy()
        rows.append(df)
    allr = pd.concat(rows, ignore_index=True)
    allr = allr.drop_duplicates(subset=['就诊号', '检查诊断', '检查所见']).reset_index(
        drop=True)
    allr['report_id'] = [
        hashlib.sha256((str(v) + str(t)).encode()).hexdigest()[:16]
        for v, t in zip(allr['就诊号'], allr['检查诊断'] + allr['检查所见'])]
    allr['report_text'] = (
        '检查项目: ' + allr['检查项目名称'].fillna('').astype(str) + '\n'
        '检查诊断: ' + allr['检查诊断'].fillna('').astype(str) + '\n'
        '检查所见: ' + allr['检查所见'].fillna('').astype(str))
    # rule pre-filter: only positive/uncertain reports reach the LLM; clear
    # negatives are skipped (kept in the parquet with llm_verdict='n').
    allr['rule_label'] = allr['report_text'].map(
        lambda t: classify_radiology(t)[0])
    return allr


def parse_verdict(raw: str) -> dict:
    m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if d.get('verdict') in ('y', 'n', 'uncertain'):
                return {'verdict': d['verdict'],
                        'reason': str(d.get('reason', ''))[:400]}
        except (json.JSONDecodeError, ValueError):
            pass
    return {'verdict': 'uncertain', 'reason': 'unparseable model response'}


def judge_one(row):
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user',
             'content': f"RADIOLOGY REPORT:\n{row['report_text'][:INPUT_CAP]}"}]
    ph = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:12]
    budget = MAX_TOKENS
    for attempt in range(3):
        try:
            r = requests.post(f'{BASE}/chat/completions',
                              headers={'Authorization': f'Bearer {API_KEY}'},
                              json={'model': MODEL, 'messages': msgs,
                                    'temperature': 0, 'max_tokens': budget},
                              timeout=180)
            r.raise_for_status()
            choice = r.json()['choices'][0]
            raw = choice['message'].get('content') or ''
            if not raw.strip():
                if choice.get('finish_reason') == 'length' and budget < 6000:
                    budget *= 2
                    continue
                raise ValueError("empty content")
            out = parse_verdict(raw)
            out.update(report_id=row['report_id'], model=MODEL, prompt_hash=ph)
            return out
        except Exception as e:
            if attempt == 2:
                return {'report_id': row['report_id'], 'model': MODEL,
                        'verdict': 'uncertain',
                        'reason': f'ERROR: {e}'}
            time.sleep(2 ** attempt)


def run():
    df = load_candidates()
    adjudicate = df[df['rule_label'].isin(ADJUDICATE)]
    print(f'{len(df)} candidate reports; {len(adjudicate)} (positive/uncertain) '
          f'need LLM adjudication', flush=True)
    done = set()
    if os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL):
            done.add(json.loads(line)['report_id'])
    todo = [r for _, r in adjudicate.iterrows() if r['report_id'] not in done]
    print(f'{len(done)} done, {len(todo)} to judge (model={MODEL}, '
          f'workers={WORKERS})', flush=True)
    if not todo:
        print('all done')
        return
    with open(OUT_JSONL, 'a') as out, ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(judge_one, r): r['report_id'] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            out.write(json.dumps(res, ensure_ascii=False) + '\n')
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)}', flush=True)
    print('wrote', OUT_JSONL)


def merge():
    df = load_candidates()
    verdicts, reasons = {}, {}
    for line in open(OUT_JSONL):
        d = json.loads(line)
        verdicts[d['report_id']] = d['verdict']
        reasons[d['report_id']] = d.get('reason', '')
    # clear rule-negatives never went to the LLM; mark them 'n' directly
    df['llm_verdict'] = df['report_id'].map(verdicts)
    rule_neg = df['rule_label'].isin(['negative', 'no_mention'])
    df.loc[rule_neg & df['llm_verdict'].isna(), 'llm_verdict'] = 'n'
    df['llm_reason'] = df['report_id'].map(reasons)
    df.loc[rule_neg, 'llm_reason'] = df.loc[rule_neg, 'rule_label']
    n_miss = df['llm_verdict'].isna().sum()
    if n_miss:
        print(f'[warn] {n_miss} reports not yet judged; rerun without --merge')
    cols = ['report_id', '就诊号', '检查项目名称', '检查诊断', '检查所见',
            '检查时间', '出报告时间', 'rule_label', 'llm_verdict', 'llm_reason']
    df[cols].to_parquet(OUT_PARQUET, index=False)
    print('merged ->', OUT_PARQUET)
    print('verdict distribution:', df['llm_verdict'].value_counts(dropna=False)
          .to_dict())


if __name__ == '__main__':
    if '--merge' in sys.argv:
        merge()
    else:
        if not API_KEY:
            sys.exit('DEEPSEEK_API_KEY not set')
        run()
        merge()
