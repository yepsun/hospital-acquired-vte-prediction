"""M4 v2: build model dataset + feature sets + splits on the v2 cohort.

Same logic as scripts/38_build_model_dataset.py but:
  - joins the v1 static features (features_static_v1.parquet) to the v2
    labels (vte_labels_v2.parquet) instead of the v1 labels;
  - keeps only v2 cohort admissions;
  - rebuilds train/val/test splits (subject-grouped 70/15/15, seed 42)
    and temporal splits by anchor_year_group on the v2 cohort.

Outputs:
  results_vte/model_dataset_v2.parquet
  results_vte/feature_sets_v2.json
  results_vte/splits_v3.json
"""
import json
import sqlite3

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURES = "results_vte/features_static_v3.parquet"
LABELS_V2 = "results_vte/vte_labels_v3.parquet"
ADMISSIONS = "results_vte/admissions_base.parquet"
DB = "mimic4.db"

OUT_DATASET = "results_vte/model_dataset_v3.parquet"
OUT_FEATURE_SETS = "results_vte/feature_sets_v3.json"
OUT_SPLITS = "results_vte/splits_v3.json"

MISSING_THRESHOLD = 0.60
SEED = 42

ID_COLS = ["hadm_id", "subject_id", "anchor_year_group"]
OUTCOME = "vte_event"
TIME_COLS = ["admittime", "dischtime", "diagnosis_time"]
META_COLS = ["vte_type", "t0_source"]
BASELINE_SCORE_COLS = [
    "padua_score", "padua_high",
    "improve_score", "improve_high",
    "improve_dd", "improve_dd_high",
]

TEMPORAL_TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
TEMPORAL_TEST_GROUPS = ["2017 - 2019", "2020 - 2022"]


def main():
    # ---------- 1. 载入 v3 标签 + v1 特征 ----------
    labels = pd.read_parquet(LABELS_V2)
    feats = pd.read_parquet(FEATURES)
    # features_static_v1 内含 v1 的标签列 (vte_event/vte_type/t0_source)，以 v3 标签为准剔除
    feats = feats.drop(columns=['vte_event', 'vte_type', 't0_source'],
                       errors='ignore')
    adm = pd.read_parquet(ADMISSIONS)[["hadm_id", "subject_id"]]
    with sqlite3.connect(DB) as conn:
        patients = pd.read_sql("SELECT subject_id, anchor_year_group FROM patients", conn)

    df = labels.merge(feats, on="hadm_id", how="left", validate="1:1")
    df = df.merge(adm, on="hadm_id", how="left", validate="1:1")
    df = df.merge(patients[["subject_id", "anchor_year_group"]].drop_duplicates("subject_id"),
                  on="subject_id", how="left", validate="m:1")
    assert df["subject_id"].notna().all(), "存在未匹配 subject_id 的入院"
    assert len(df) == len(labels), f"行数异常: {len(df)} != {len(labels)}"
    assert len(df) == df["hadm_id"].nunique(), "hadm_id 重复"
    assert len(labels) == labels["hadm_id"].nunique(), "标签 hadm_id 重复"

    # ---------- 2. 特征集 ----------
    excluded_fixed = set(ID_COLS + [OUTCOME] + TIME_COLS + META_COLS + BASELINE_SCORE_COLS)
    # merge 产生的 *_x/*_y 后缀时间列也排除
    excluded_fixed |= {c for c in df.columns if c.startswith(('admittime', 'dischtime', 'diagnosis_time'))}
    vitals_icu = [c for c in df.columns if c.startswith("vital_") and not c.endswith("_source")]
    source_cols = [c for c in df.columns if c.endswith("_source")]

    candidates = [c for c in df.columns if c not in excluded_fixed and c not in source_cols]
    missing_rate = df[candidates].isna().mean()
    high_missing = missing_rate[missing_rate > MISSING_THRESHOLD].sort_values(ascending=False)
    main_feats = missing_rate[missing_rate <= MISSING_THRESHOLD].index.tolist()

    print(f"缺失率 >{MISSING_THRESHOLD:.0%} 排除列 ({len(high_missing)}):")
    for col, r in high_missing.items():
        print(f"  {col}: {r:.1%}")

    feature_sets = {
        "main": main_feats,
        "vitals_icu": vitals_icu,
        "baseline_scores": ["padua_score", "improve_score"],
        "id_cols": ID_COLS,
        "outcome": OUTCOME,
    }

    # ---------- 3. 划分 (subject_id 分组 70/15/15) ----------
    y = df[OUTCOME].astype(int)
    groups = df["subject_id"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, rest_idx = next(gss1.split(df, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_sub, test_sub = next(gss2.split(df.iloc[rest_idx], y.iloc[rest_idx], groups.iloc[rest_idx]))
    val_idx, test_idx = rest_idx[val_sub], rest_idx[test_sub]

    s_train, s_val, s_test = (set(groups.iloc[i]) for i in (train_idx, val_idx, test_idx))
    assert not (s_train & s_val) and not (s_train & s_test) and not (s_val & s_test), \
        "存在 subject 跨集合!"

    print(f"\n总行数: {len(df)}; 总事件数: {int(y.sum())} ({y.mean():.3%})")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        n_ev = int(y.iloc[idx].sum())
        print(f"  {name}: {len(idx)} 行 ({len(idx)/len(df):.1%}), 事件 {n_ev} ({y.iloc[idx].mean():.3%})")

    # ---------- 4. 时间外切分 ----------
    train_pool_mask = df["anchor_year_group"].isin(TEMPORAL_TRAIN_GROUPS)
    test_temporal_mask = df["anchor_year_group"].isin(TEMPORAL_TEST_GROUPS)
    assert not (train_pool_mask & test_temporal_mask).any()
    print("\ntemporal 切分:")
    print(f"  train_pool: {int(train_pool_mask.sum())} 行, 事件 {int(y[train_pool_mask].sum())}")
    print(f"  test_temporal: {int(test_temporal_mask.sum())} 行, 事件 {int(y[test_temporal_mask].sum())}")

    splits = {
        "train": df.iloc[train_idx]["hadm_id"].tolist(),
        "val": df.iloc[val_idx]["hadm_id"].tolist(),
        "test": df.iloc[test_idx]["hadm_id"].tolist(),
        "train_pool_temporal": df.loc[train_pool_mask, "hadm_id"].tolist(),
        "test_temporal": df.loc[test_temporal_mask, "hadm_id"].tolist(),
    }

    # ---------- 5. 写出 ----------
    df.to_parquet(OUT_DATASET, index=False)
    with open(OUT_FEATURE_SETS, "w") as f:
        json.dump(feature_sets, f, indent=2)
    with open(OUT_SPLITS, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\n已写出: {OUT_DATASET}, {OUT_FEATURE_SETS}, {OUT_SPLITS}")


if __name__ == "__main__":
    main()
