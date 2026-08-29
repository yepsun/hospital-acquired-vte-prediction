#!/usr/bin/env python3
"""
Script 46 (VTE M5 / Task 3 Step 1): landmark feature matrix.

For each landmark row in landmark_skeleton.parquet (hadm_id, t_lm,
hours_since_adm, label, split) build features using ONLY buckets with
bucket_end <= t_lm (bucket k covers [12k, 12k+12) h since admission;
usable iff 12*(k+1) <= hours_since_adm):

  static broadcast (features_static_v1): age, male, bmi, 14 comorbidities,
    adm_elective, adm_emergency, first_careunit_icu, prior_admissions_n
  labs (9 analytes): last_<k> (median of most recent bucket with a value),
    last_<k>_age_h (t_lm - bucket_end of that bucket), cum_<k>_median,
    cum_<k>_count, trend_<k> (last value - previous valid value)
  vitals (6): last_<v>_mean, last_<v>_worst, last_<v>_age_h (one per vital)
  rx: rx_<drug>_last24h (any dose in the 2 buckets before t_lm),
    proc_<p>_cum (latest cumulative flag, ffilled over sparse buckets)
  hours_since_adm (already in skeleton)

Leakage assertion: last_<k>_age_h >= 0 on a sample of rows.

Outputs:
  results_vte/landmark_features.parquet       full evaluation view (float32)
  results_vte/landmark_features_train.parquet train+val rows, negatives
                                              undersampled 10% (seed 42)
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RES = 'results_vte'
SEED = 42
NEG_KEEP = 0.10
BUCKET_H = 12.0


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# M6 sensitivity params (defaults unchanged)
SKELETON_PATH = argval('--skeleton', f'{RES}/landmark_skeleton_v2.parquet')
OUT_FULL = argval('--out-full', f'{RES}/landmark_features_v2.parquet')
OUT_TRAIN = argval('--out-train', f'{RES}/landmark_features_train_v2.parquet')

LAB_KEYS = ['wbc', 'platelet', 'rdw', 'hemoglobin', 'creatinine', 'bun',
            'inr', 'pt', 'aptt']
VIT_KEYS = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
RX_KEYS = ['heparin', 'warfarin', 'doac', 'antiplatelet']
PROC_KEYS = ['cvc', 'mechvent', 'transfusion']

STATIC_COLS = ['age', 'male', 'bmi', 'cancer_active', 'heart_failure',
               'copd', 'chronic_liver', 'ckd', 'diabetes', 'stroke', 'mi',
               'obesity_icd', 'varicose', 'thrombophilia', 'infection_severe',
               'rheumatologic', 'prior_vte', 'adm_elective', 'adm_emergency',
               'first_careunit_icu', 'prior_admissions_n']

# feature column order


def build_feats(vit_keys):
    return (STATIC_COLS + ['hours_since_adm']
            + [x for k in LAB_KEYS for x in
               [f'last_{k}', f'last_{k}_age_h', f'cum_{k}_median',
                f'cum_{k}_count', f'trend_{k}']]
            + [f'last_{v}_{s}' for v in vit_keys for s in ['mean', 'worst']]
            + [f'last_{v}_age_h' for v in vit_keys]
            + [f'rx_{d}_last24h' for d in RX_KEYS]
            + [f'proc_{p}_cum' for p in PROC_KEYS])


FEATS = build_feats(VIT_KEYS)


def group_arrays(df, cols):
    """dict hadm_id -> (bucket_idx array, values matrix) from sorted long table."""
    df = df.sort_values(['hadm_id', 'bucket_idx'], kind='mergesort')
    h = df['hadm_id'].values
    b = df['bucket_idx'].values.astype(np.int64)
    V = df[cols].values.astype(np.float32)
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    ends = np.r_[starts[1:], len(h)]
    return {h[s]: (b[s:e], V[s:e]) for s, e in zip(starts, ends)}


print('loading ...', flush=True)
sk = pd.read_parquet(SKELETON_PATH)
labs = pd.read_parquet(f'{RES}/buckets_labs_v2.parquet')
vitals = pd.read_parquet(f'{RES}/buckets_vitals_v2.parquet')
rx = pd.read_parquet(f'{RES}/buckets_rx_v2.parquet')
static = pd.read_parquet(f'{RES}/features_static_v1.parquet',
                         columns=['hadm_id', 'padua_score'] + STATIC_COLS)

# vitals temperature not in 9-analyte lab buckets; check availability
vit_cols = [c for c in vitals.columns if c not in ('hadm_id', 'bucket_idx')]
have_vit = sorted({c[len('vital_'):-len('_mean')] for c in vit_cols
                   if c.endswith('_mean')})
VIT_KEYS = [v for v in VIT_KEYS if v in have_vit]
FEATS = build_feats(VIT_KEYS)

L = group_arrays(labs, [f'lab_{k}_median' for k in LAB_KEYS]
                 + [f'lab_{k}_count' for k in LAB_KEYS])
V = group_arrays(vitals, [f'vital_{v}_{s}' for v in VIT_KEYS
                          for s in ['mean', 'worst']])
R = group_arrays(rx, [f'rx_{d}' for d in RX_KEYS]
                 + [f'proc_{p}_cum' for p in PROC_KEYS])

sk = sk.merge(static, on='hadm_id', how='left', validate='m:1')
print(f'skeleton {len(sk)} rows; features: {len(FEATS)}', flush=True)

n_lab = len(LAB_KEYS)
n_vit = len(VIT_KEYS)
NLABF = 5 * n_lab
NVITF = 3 * n_vit
NRXF = len(RX_KEYS) + len(PROC_KEYS)
n_feat = len(FEATS)

sk = sk.sort_values('hadm_id', kind='mergesort').reset_index(drop=True)
hours = sk['hours_since_adm'].values
kmax_all = np.ceil(hours / BUCKET_H).astype(np.int64) - 1  # bucket_end<=t_lm

X = np.full((len(sk), n_feat), np.nan, dtype=np.float32)
# static block + hours_since_adm
X[:, :len(STATIC_COLS)] = sk[STATIC_COLS].values.astype(np.float32)
X[:, len(STATIC_COLS)] = hours.astype(np.float32)

hadm_vals = sk['hadm_id'].values
starts = np.flatnonzero(np.r_[True, hadm_vals[1:] != hadm_vals[:-1]])
ends = np.r_[starts[1:], len(hadm_vals)]

c0 = len(STATIC_COLS) + 1  # first dynamic column
print('building dynamic features per hadm ...', flush=True)
for gi, (s, e) in enumerate(zip(starts, ends)):
    hadm = hadm_vals[s]
    kms = kmax_all[s:e]
    hrs = hours[s:e]
    out = X[s:e]

    if hadm in L:
        b, M = L[hadm]
        med = M[:, :n_lab]
        cnt = M[:, n_lab:]
        pos = np.searchsorted(b, kms, side='right') - 1
        # last valid index per bucket per key
        valid = ~np.isnan(med)
        lv = np.maximum.accumulate(
            np.where(valid, np.arange(len(b))[:, None], -1), axis=0)
        cum_cnt = np.nancumsum(cnt, axis=0)
        # cumulative median per bucket per key (few buckets per hadm)
        cm = np.full_like(med, np.nan)
        for i in range(len(b)):
            with np.errstate(all='ignore'):
                cm[i] = np.nanmedian(med[:i + 1], axis=0)
        ok = pos >= 0
        pclip = np.clip(pos, 0, None)
        lv_p = lv[pclip]                       # rows x keys, -1 if none
        has = ok[:, None] & (lv_p >= 0)
        lv_c = np.clip(lv_p, 0, None)
        kcol = np.arange(n_lab)[None, :]
        last_val = med[lv_c, kcol]
        prev_lv = lv[np.clip(lv_c - 1, 0, None), kcol]
        has_prev = has & (prev_lv >= 0) & (prev_lv < lv_p)
        prev_val = med[np.clip(prev_lv, 0, None), kcol]
        base = c0
        blk = out[:, base:base + NLABF].reshape(e - s, n_lab, 5)
        blk[:, :, 0] = np.where(has, last_val, np.nan)
        blk[:, :, 1] = np.where(
            has, hrs[:, None] - BUCKET_H * (b[lv_c] + 1)[None, :], np.nan)
        blk[:, :, 2] = np.where(has, cm[lv_c, kcol], np.nan)
        blk[:, :, 3] = np.where(ok[:, None], cum_cnt[pclip], np.nan)
        blk[:, :, 4] = np.where(has_prev, last_val - prev_val, np.nan)

    if hadm in V:
        b, M = V[hadm]  # cols: per vital [mean, worst] interleaved
        pos = np.searchsorted(b, kms, side='right') - 1
        mean_m = M[:, 0::2]
        valid = ~np.isnan(mean_m)
        lv = np.maximum.accumulate(
            np.where(valid, np.arange(len(b))[:, None], -1), axis=0)
        ok = pos >= 0
        pclip = np.clip(pos, 0, None)
        lv_p = lv[pclip]
        has = ok[:, None] & (lv_p >= 0)
        lv_c = np.clip(lv_p, 0, None)
        kcol = np.arange(n_vit)[None, :]
        base = c0 + NLABF
        blk = out[:, base:base + 2 * n_vit].reshape(e - s, n_vit, 2)
        blk[:, :, 0] = np.where(has, M[lv_c, 2 * kcol], np.nan)
        blk[:, :, 1] = np.where(has, M[lv_c, 2 * kcol + 1], np.nan)
        out[:, base + 2 * n_vit: base + 3 * n_vit] = np.where(
            has, hrs[:, None] - BUCKET_H * (b[lv_c] + 1)[None, :], np.nan)

    if hadm in R:
        b, M = R[hadm]
        pos = np.searchsorted(b, kms, side='right') - 1
        base = c0 + NLABF + NVITF
        # rx last 24h: dose in bucket kmax-1 or kmax (vectorized via cumsum)
        cs = np.vstack([np.zeros((1, len(RX_KEYS)), dtype=np.float32),
                        np.cumsum(M[:, :len(RX_KEYS)], axis=0)])
        hi = np.searchsorted(b, kms, side='right')          # exclusive end
        lo = np.searchsorted(b, kms - 1, side='left')       # inclusive start
        r24 = (cs[hi] - cs[lo] > 0).astype(np.float32)
        out[:, base:base + len(RX_KEYS)] = r24
        ok = pos >= 0
        out[:, base + len(RX_KEYS):] = np.where(
            ok[:, None], M[np.clip(pos, 0, None), len(RX_KEYS):], 0.0)
    else:
        base = c0 + NLABF + NVITF
        out[:, base:base + NRXF] = 0.0

    if (gi + 1) % 50000 == 0:
        print(f'  {gi + 1}/{len(starts)} hadms', flush=True)

# proc/rx for hadms never in rx table handled above; also proc cols NaN->0
base = c0 + NLABF + NVITF
X[:, base:] = np.nan_to_num(X[:, base:], nan=0.0)

# ── leakage assertion: age_h >= 0 on sample ──
rng = np.random.RandomState(SEED)
age_cols = [FEATS.index(f'last_{k}_age_h') for k in LAB_KEYS]
samp = rng.choice(len(sk), 10, replace=False)
ages = X[samp][:, age_cols]
assert np.all(np.isnan(ages) | (ages >= 0)), 'leakage: negative age_h'
print('leakage assertion OK: sampled last_<key>_age_h all >= 0', flush=True)

out_df = pd.DataFrame(X, columns=FEATS)
for c in ['hadm_id', 'subject_id', 't_lm', 'label', 'split', 'padua_score']:
    out_df[c] = sk[c].values
out_df.to_parquet(OUT_FULL, index=False)
print(f'saved full eval view {out_df.shape} -> {OUT_FULL}',
      flush=True)

# ── training view: train+val rows, negatives undersampled 10% ──
pool_mask = sk['split'].isin(['train', 'val']).values
pos = pool_mask & (sk['label'].values == 1)
neg = pool_mask & (sk['label'].values == 0)
keep_neg = neg & (rng.random(len(sk)) < NEG_KEEP)
train_df = out_df[pos | keep_neg].reset_index(drop=True)
train_df.to_parquet(OUT_TRAIN, index=False)
print(f'saved train view {train_df.shape} '
      f'(pos {int(pos.sum())}, neg kept {int(keep_neg.sum())}/{int(neg.sum())}) '
      f'-> {OUT_TRAIN}', flush=True)

miss = out_df[FEATS].isna().mean().sort_values(ascending=False)
print(f'n features: {len(FEATS)}')
print('missing rate top 10:')
for n, v in miss.head(10).items():
    print(f'  {n:25s} {v * 100:6.2f}%')
