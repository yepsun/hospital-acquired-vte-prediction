#!/usr/bin/env python3
"""Second-model (GLM) adjudication of local hospital radiology reports.

Independent cross-check of the DeepSeek adjudication (build_radiology_llm.py):
identical candidate set, prompt, verdict semantics, and pre-filter; different
vendor model. Adjudicates only the reports that went to the LLM stage
(rule_label positive/uncertain). glm-5.3-flash always reasons, so the token
budget must cover reasoning tokens; the budget doubles on length-truncated
empty content.

Usage:
  ZHIPU_API_KEY=... python3 extval/build_radiology_llm_glm.py
  python3 extval/build_radiology_llm_glm.py --merge
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from build_radiology_llm import (ADJUDICATE, INPUT_CAP, SYSTEM,
                                 load_candidates, parse_verdict)

BASE = os.environ.get('GLM_BASE', 'https://open.bigmodel.cn/api/paas/v4')
MODEL = os.environ.get('GLM_MODEL', 'glm-5.3-flash')
API_KEY = os.environ.get('ZHIPU_API_KEY')
MAX_TOKENS = 1500
MAX_BUDGET = 8000
WORKERS = int(os.environ.get('LLM_WORKERS', '8'))
OUT_JSONL = 'extval/radiology_llm_glm.jsonl'
OUT_PARQUET = 'extval/radiology_llm_glm.parquet'


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
                              timeout=300)
            r.raise_for_status()
            choice = r.json()['choices'][0]
            raw = choice['message'].get('content') or ''
            if not raw.strip():
                if choice.get('finish_reason') == 'length' and budget < MAX_BUDGET:
                    budget *= 2
                    continue
                raise ValueError('empty content')
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
    if not API_KEY:
        sys.exit('ZHIPU_API_KEY not set')
    df = load_candidates()
    adjudicate = df[df['rule_label'].isin(ADJUDICATE)]
    print(f'{len(df)} candidate reports; {len(adjudicate)} need GLM '
          f'adjudication', flush=True)
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
            if i % 50 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)}', flush=True)
    print('wrote', OUT_JSONL)


def merge():
    df = load_candidates()
    verdicts, reasons = {}, {}
    for line in open(OUT_JSONL):
        d = json.loads(line)
        verdicts[d['report_id']] = d['verdict']
        reasons[d['report_id']] = d.get('reason', '')
    adj = df[df['rule_label'].isin(ADJUDICATE)].copy()
    adj['glm_verdict'] = adj['report_id'].map(verdicts)
    adj['glm_reason'] = adj['report_id'].map(reasons)
    n_miss = adj['glm_verdict'].isna().sum()
    if n_miss:
        print(f'[warn] {n_miss} reports not yet judged; rerun without --merge')
    cols = ['report_id', '就诊号', '检查项目名称', '检查诊断', '检查所见',
            '检查时间', 'rule_label', 'glm_verdict', 'glm_reason']
    adj[cols].to_parquet(OUT_PARQUET, index=False)
    print('merged ->', OUT_PARQUET)
    print('verdict distribution:',
          adj['glm_verdict'].value_counts(dropna=False).to_dict())


if __name__ == '__main__':
    if '--merge' in sys.argv:
        merge()
    else:
        run()
        merge()
