#!/usr/bin/env python3
"""Agreement between DeepSeek and GLM radiology-report adjudications.

Compares per-report verdicts on the LLM-adjudicated subset (rule
positive/uncertain reports), computes overall and binary (y vs not-y)
agreement with Cohen's kappa, and exports discordant reports for human
review.

Outputs:
  extval/llm_agreement.json              statistics
  extval/llm_discordant_for_review.csv   reports with mismatched verdicts
"""
import json
from collections import Counter

import pandas as pd

from build_radiology_llm import ADJUDICATE, load_candidates

DS_JSONL = 'extval/radiology_llm.jsonl'
GLM_JSONL = 'extval/radiology_llm_glm.jsonl'
OUT_JSON = 'extval/llm_agreement.json'
OUT_CSV = 'extval/llm_discordant_for_review.csv'
LABELS = ['y', 'n', 'uncertain']


def load_verdicts(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d['report_id']] = d
    return out


def kappa(pairs, labels):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[l] / n * cb[l] / n for l in labels)
    return po, (po - pe) / (1 - pe) if pe < 1 else float('nan')


def main():
    cand = load_candidates()
    adj = cand[cand['rule_label'].isin(ADJUDICATE)].set_index('report_id')
    ds, glm = load_verdicts(DS_JSONL), load_verdicts(GLM_JSONL)
    common = [r for r in adj.index if r in ds and r in glm]
    print(f'llm-adjudicated: {len(adj)}; deepseek: {len(ds)}; glm: {len(glm)}; '
          f'paired: {len(common)}')

    pairs3 = [(ds[r]['verdict'], glm[r]['verdict']) for r in common]
    pairs2 = [('y' if a == 'y' else 'not-y', 'y' if b == 'y' else 'not-y')
              for a, b in pairs3]
    po3, k3 = kappa(pairs3, LABELS)
    po2, k2 = kappa(pairs2, ['y', 'not-y'])

    conf = pd.crosstab([ds[r]['verdict'] for r in common],
                       [glm[r]['verdict'] for r in common],
                       rownames=['deepseek'], colnames=['glm'],
                       dropna=False).reindex(index=LABELS, columns=LABELS,
                                             fill_value=0)
    print(conf)

    discordant = [r for r in common if ds[r]['verdict'] != glm[r]['verdict']]
    flips = [r for r in discordant
             if (ds[r]['verdict'] == 'y') != (glm[r]['verdict'] == 'y')]
    rows = []
    for r in discordant:
        c = adj.loc[r]
        rows.append({
            'report_id': r, '就诊号': c['就诊号'],
            '检查项目名称': c['检查项目名称'], '检查时间': c['检查时间'],
            'rule_label': c['rule_label'],
            'deepseek_verdict': ds[r]['verdict'],
            'deepseek_reason': ds[r].get('reason', ''),
            'glm_verdict': glm[r]['verdict'],
            'glm_reason': glm[r].get('reason', ''),
            'y_status_flips': r in flips,
            'report_text': c['report_text'],
            'human_verdict': ''})
    pd.DataFrame(rows).sort_values(
        ['y_status_flips', 'report_id'], ascending=[False, True]
    ).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    stats = {
        'n_llm_adjudicated': len(common),
        'labels': LABELS,
        'overall_agreement': round(po3, 4), 'kappa_3class': round(k3, 4),
        'binary_agreement_y_vs_noty': round(po2, 4),
        'kappa_binary': round(k2, 4),
        'confusion_deepseek_rows_glm_cols': conf.to_dict(),
        'n_discordant_any': len(discordant),
        'n_discordant_y_status': len(flips),
        'deepseek_y': sum(ds[r]['verdict'] == 'y' for r in common),
        'glm_y': sum(glm[r]['verdict'] == 'y' for r in common),
        'concordant_y': sum(a == 'y' and b == 'y' for a, b in pairs3),
    }
    json.dump(stats, open(OUT_JSON, 'w'), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'discordant reports -> {OUT_CSV} (human_verdict column to fill)')


if __name__ == '__main__':
    main()
