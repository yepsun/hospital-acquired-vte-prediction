"""Task 1 (M5): build landmark skeleton table.

Landmark rule (Global Constraints):
  t_lm = admittime + 48h, 72h, 96h, ...
  require t_lm < min(dischtime, diagnosis_time if vte_event==1) - 24h
  event admissions: only landmarks before diagnosis_time, label=1
  non-event admissions: label=0

Output: results_vte/landmark_skeleton_v2.parquet (v2 labels/splits)
  hadm_id, subject_id, t_lm, hours_since_adm, label, split

M6 sensitivity params (defaults unchanged):
  --start H   first landmark offset in hours (default 48)
  --out PATH  output parquet (default results_vte/landmark_skeleton.parquet)
"""
import json
import sys
import numpy as np
import pandas as pd

SEED = 42
VTE = "results_vte"


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


START_H = float(argval('--start', 48))
OUT_PATH = argval('--out', f"{VTE}/landmark_skeleton_v2.parquet")


def main():
    labels = pd.read_parquet(f"{VTE}/ajm/new/data/vte_labels_inclprior_excl24h.parquet")
    adm = pd.read_parquet(f"{VTE}/admissions_base.parquet",
                          columns=["hadm_id", "subject_id", "admittime"])
    with open(f"{VTE}/ajm/new/data/primary_inclprior_excl24h_splits.json") as f:
        splits = json.load(f)

    df = labels.merge(adm, on="hadm_id", how="left")
    assert df["subject_id"].notna().all(), "hadm_id missing subject_id"

    # endpoint: min(dischtime, diagnosis_time if event) - 24h
    end = df["dischtime"].copy()
    is_ev = df["vte_event"] == 1
    end[is_ev] = np.minimum(df.loc[is_ev, "dischtime"],
                            df.loc[is_ev, "diagnosis_time"])
    df["cutoff"] = end - pd.Timedelta(hours=24)

    hours = np.arange(START_H, 24 * 365, 24, dtype=float)  # START_H, +24h, ...
    adm_t = df["admittime"].to_numpy()
    los_h = (df["cutoff"] - df["admittime"]).dt.total_seconds().to_numpy() / 3600.0
    # number of valid landmarks per admission: hours_since_adm h valid iff h < los_h
    n_lm = np.clip(np.searchsorted(hours, los_h, side="left"), 0, len(hours))

    row_idx = np.repeat(np.arange(len(df)), n_lm)
    lm_h = np.concatenate([hours[:n] for n in n_lm])

    out = pd.DataFrame({
        "hadm_id": df["hadm_id"].to_numpy()[row_idx],
        "subject_id": df["subject_id"].to_numpy()[row_idx],
        "t_lm": adm_t[row_idx] + pd.to_timedelta(lm_h, unit="h"),
        "hours_since_adm": lm_h,
        "label": df["vte_event"].to_numpy()[row_idx],
    })

    hadm2split = {}
    for name in ("train", "val", "test"):
        hadm2split.update({h: name for h in splits[name]})
    out["split"] = out["hadm_id"].map(hadm2split)
    assert out["split"].notna().all(), "hadm_id not covered by train/val/test"

    out.to_parquet(OUT_PATH, index=False)

    # --- checks ---
    ev = df.set_index("hadm_id")
    m_ev = out["label"] == 1
    diag = ev.loc[out.loc[m_ev, "hadm_id"], "diagnosis_time"]
    assert (out.loc[m_ev, "t_lm"].to_numpy() < diag.to_numpy()).all(), \
        "t_lm >= diagnosis_time for an event landmark"
    disch = ev.loc[out["hadm_id"], "dischtime"]
    assert (out["t_lm"].to_numpy() < disch.to_numpy()).all(), "t_lm >= dischtime"
    pos_hadm = set(out.loc[m_ev, "hadm_id"])
    assert all(ev.loc[h, "vte_event"] == 1 for h in pos_hadm), \
        "label=1 hadm_id with vte_event != 1"

    # --- report ---
    per_adm = out.groupby("hadm_id").size()
    n_pos, n_neg = int(m_ev.sum()), int((~m_ev).sum())
    q = per_adm.quantile([0.25, 0.5, 0.75])
    print(f"total landmark rows: {len(out):,}")
    print(f"admissions with >=1 landmark: {per_adm.size:,} / {len(df):,}")
    print(f"landmarks per admission: median={q[0.5]:.0f} "
          f"IQR=[{q[0.25]:.0f}, {q[0.75]:.0f}] max={per_adm.max()}")
    print(f"label=1: {n_pos:,} ({n_pos / len(out) * 100:.2f}%)  "
          f"(expected 4-8k, ratio 0.3-1%)")
    print(f"label=0: {n_neg:,}")
    print(f"event admissions: {int(is_ev.sum()):,}")
    for name in ("train", "val", "test"):
        sub = out[out["split"] == name]
        print(f"split={name}: rows={len(sub):,}, "
              f"event landmarks={int(sub['label'].sum()):,}")
    assert 0.002 <= n_pos / len(out) <= 0.03, "label=1 ratio out of expected range"
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
