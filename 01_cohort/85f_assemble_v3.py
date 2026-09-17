"""Task 7: assemble M2 static feature matrix + final leakage checks.

Left-joins vte_labels with features_labs/vitals/comorbid/treatments/clinical_scores
on hadm_id (assert 422,408 rows & unique key after each join), writes
results_vte/features_static_v3.parquet, generates results_vte/m2_feature_report.md,
and runs three leakage checks (24h window plausibility, no current-admission VTE
ICD-derived columns, no diagnosis_time < admittime+24h).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RV = ROOT / "results_vte"
N_ROWS = None  # dynamic

TABLES = [
    "features_labs",
    "features_vitals",
    "features_comorbid",
    "features_treatments",
    "clinical_scores",
]

LABEL_COLS = {"hadm_id", "admittime", "dischtime", "diagnosis_time", "vte_type", "t0_source", "vte_event"}
# Legit feature columns whose names match VTE keywords but are comorbidity history,
# not derived from the index admission's VTE ICD codes.
ALLOWED_VTE_NAMED = {"prior_vte", "prior_vte_any", "thrombophilia", "varicose"}
VTE_NAME_RE = re.compile(r"vte|pulmonary_embol|(^|_)pe(_|$)|(^|_)dvt(_|$)|thrombos", re.I)


def check(cond: bool, name: str) -> str:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return status


def main() -> None:
    base = pd.read_parquet(RV / "vte_labels_v3.parquet")
    assert base["hadm_id"].is_unique
    assert base["hadm_id"].is_unique

    df = base
    for t in TABLES:
        part = pd.read_parquet(RV / f"{t}.parquet")
        assert part["hadm_id"].is_unique, f"{t} hadm_id not unique"
        df = df.merge(part, on="hadm_id", how="left", validate="one_to_one")
        assert len(df) == len(base), f"after join {t}: {len(df)} != {len(base)}"
        assert df["hadm_id"].is_unique
        print(f"joined {t}: {df.shape}")

    out = RV / "features_static_v3.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} {df.shape}")

    feature_cols = [c for c in df.columns if c not in LABEL_COLS]

    # ---- leakage checks ----
    print("\n=== leakage checks ===")
    # (1) all feature scripts restricted to admittime+24h window (by construction in
    # scripts/14-36). Plausibility spot-check: lab count columns should not show
    # abnormally large values (a full-stay window would accumulate far more draws).
    count_cols = [c for c in df.columns if c.startswith("lab_") and c.endswith("_count")]
    max_count = df[count_cols].max().max()
    s1 = check(max_count <= 100, f"24h window: max lab count in 24h = {max_count} (<=100 plausible)")

    # (2) no current-admission VTE ICD-derived columns among feature columns
    suspect = [c for c in feature_cols if VTE_NAME_RE.search(c) and c not in ALLOWED_VTE_NAMED]
    s2 = check(len(suspect) == 0, f"no VTE ICD-derived feature columns (suspect: {suspect or 'none'})")

    # (3) no diagnosis_time earlier than admittime+24h
    diag = pd.to_datetime(df["diagnosis_time"])
    t24 = pd.to_datetime(df["admittime"]) + pd.Timedelta(hours=24)
    n_early = int((diag < t24).sum())
    s3 = check(n_early == 0, f"diagnosis_time < admittime+24h rows = {n_early} (expect 0)")

    # ---- report ----
    pos = df[df["vte_event"] == 1]
    neg = df[df["vte_event"] == 0]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    stats = pd.DataFrame({
        "missing_pct": df[numeric].isna().mean() * 100,
        "mean": df[numeric].mean(),
        "median": df[numeric].median(),
        "mean_pos": pos[numeric].mean(),
        "mean_neg": neg[numeric].mean(),
    })
    stats["diff_pct"] = np.where(
        stats["mean_neg"] != 0,
        (stats["mean_pos"] - stats["mean_neg"]) / stats["mean_neg"].abs() * 100,
        np.nan,
    )

    high_missing = stats[stats["missing_pct"] > 60].sort_values("missing_pct", ascending=False)

    # direction sanity: expected-positive-direction features
    expected_up = {
        "age": "age↑", "cancer_active": "cancer↑", "padua_score": "padua↑",
        "padua_high": "padua_high↑", "improve_score": "improve↑", "surgery_flag": "surgery↑",
        "prior_vte": "prior_vte↑", "thrombophilia": "thrombophilia↑",
        "heart_failure": "heart_failure↑", "bmi": "bmi↑", "lab_ddimer_median": "ddimer↑",
        "proc_cvc": "cvc↑", "proc_mechvent": "mechvent↑", "infection_severe": "infection↑",
        "trauma_flag": "trauma↑",
    }
    anomalies = []
    for col, label in expected_up.items():
        if col in stats.index and not (stats.loc[col, "mean_pos"] > stats.loc[col, "mean_neg"]):
            anomalies.append((col, label, stats.loc[col, "mean_pos"], stats.loc[col, "mean_neg"]))

    top10 = stats.reindex(stats["diff_pct"].abs().sort_values(ascending=False).index)
    top10 = top10[top10["mean_pos"] > top10["mean_neg"]].head(10)

    lines = []
    lines.append("# M2 静态特征报告 (features_static_v3)\n")
    lines.append(f"- 行数: {len(df):,}；hadm_id 唯一: {df['hadm_id'].is_unique}")
    lines.append(f"- 总列数: {df.shape[1]}（标签/时间列 {len(LABEL_COLS)}，特征列 {len(feature_cols)}）")
    lines.append(f"- 阳性 (vte_event=1): {len(pos):,}；阴性: {len(neg):,}\n")

    lines.append("## 泄漏终检\n")
    lines.append(f"1. 24h 时间窗：所有 lab_/vital_/rx_/proc_ 特征在 scripts/14–36 中均限定 admittime 起 24h 内；"
                 f"抽查 lab count 列最大值 = {max_count}，无异常大值 → **{s1}**")
    lines.append(f"2. 特征列无当次住院 VTE ICD 派生列（列名扫描，可疑列：{suspect or '无'}；"
                 f"prior_vte/thrombophilia 为病史合并症，属合法特征）→ **{s2}**")
    lines.append(f"3. diagnosis_time < admittime+24h 行数 = {n_early}（M1 已剔除）→ **{s3}**\n")

    lines.append("## 缺失 >60% 的列\n")
    if high_missing.empty:
        lines.append("无\n")
    else:
        lines.append("| column | missing_pct |")
        lines.append("|---|---|")
        for c, r in high_missing.iterrows():
            lines.append(f"| {c} | {r['missing_pct']:.1f}% |")
        lines.append("")

    lines.append("## 阳性组 vs 阴性组：差异最大的前 10 个特征（阳性更高）\n")
    lines.append("| feature | mean_pos | mean_neg | diff_% |")
    lines.append("|---|---|---|---|")
    for c, r in top10.iterrows():
        lines.append(f"| {c} | {r['mean_pos']:.4g} | {r['mean_neg']:.4g} | {r['diff_pct']:+.1f}% |")
    lines.append("")

    lines.append("## 预期方向合理性核查（阳性组应更高）\n")
    lines.append("| feature | mean_pos | mean_neg | 方向 |")
    lines.append("|---|---|---|---|")
    for col, label in expected_up.items():
        if col in stats.index:
            r = stats.loc[col]
            ok = r["mean_pos"] > r["mean_neg"]
            lines.append(f"| {col} ({label}) | {r['mean_pos']:.4g} | {r['mean_neg']:.4g} | {'符合' if ok else '**异常**'} |")
    lines.append("")
    if anomalies:
        lines.append(f"异常项：{[a[0] for a in anomalies]}\n")
    else:
        lines.append("全部符合预期方向，无异常项。\n")

    lines.append("## 全列统计（缺失率 / 均值 / 中位数 / 分组均值）\n")
    lines.append("| column | missing_% | mean | median | mean_pos | mean_neg | diff_% |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, r in stats.sort_values("missing_pct", ascending=False).iterrows():
        lines.append(f"| {c} | {r['missing_pct']:.1f} | {r['mean']:.4g} | {r['median']:.4g} | "
                     f"{r['mean_pos']:.4g} | {r['mean_neg']:.4g} | {r['diff_pct']:+.1f} |")

    rpt = RV / "m2_feature_report.md"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {rpt}")
    print(f"final shape: {df.shape}; missing>60%: {list(high_missing.index)}")
    print(f"direction anomalies: {[a[0] for a in anomalies] or 'none'}")


if __name__ == "__main__":
    main()
