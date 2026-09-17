#!/usr/bin/env python3
"""Parse age/sex from ALL EMR record types (best coverage) for both cohorts."""
import csv
import json
import re
import sys

import pandas as pd

AGE_RE = re.compile(r'年龄[：:]\s*(\d{1,3})\s*岁')
SEX_RE = re.compile(r'性别[：:]\s*(男|女)')

# fallback: 年龄 patterns without 岁
AGE2_RE = re.compile(r'年龄[：:]\s*(\d{1,3})\b')


def parse_file(path):
    rows = []
    with open(path, encoding='gb18030') as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        vis_i, name_i, cont_i = idx['his_vis_id'], idx['mr_name'], idx['file_content']
        for row in r:
            try:
                vis = row[vis_i]
                d = json.loads(row[cont_i])
                s = json.dumps(d, ensure_ascii=False)
            except Exception:
                continue
            m_age = AGE_RE.search(s)
            m_sex = SEX_RE.search(s)
            if m_age and m_sex:
                age = int(m_age.group(1))
                if 18 <= age <= 110:
                    rows.append((vis, age, m_sex.group(1)))
    return rows


def main():
    high = parse_file('高危VTE患者/病历结果.csv')
    print('high parsed rows:', len(high), flush=True)
    nonh = parse_file('非高危VTE患者/分层抽样患者病历结果.csv')
    print('non-high parsed rows:', len(nonh), flush=True)
    pdf = pd.DataFrame(high + nonh, columns=['vis', 'age', 'sex'])
    pdf = pdf.drop_duplicates('vis', keep='first')
    pdf.to_parquet('/var/folders/dp/xkwn23650r3g8fml_ldjh_jr0000gp/T/opencode/demo_all.parquet',
                   index=False)
    print('unique visits with demographics:', len(pdf))
    print(pdf.groupby('sex').size().to_dict())


if __name__ == '__main__':
    main()
