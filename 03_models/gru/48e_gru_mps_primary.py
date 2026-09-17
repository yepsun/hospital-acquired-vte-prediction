#!/usr/bin/env python3
"""
Script 48d (VTE M5 / Task 4): GRU sequence model on bucket time series --
Apple Silicon GPU (MPS) version of 48c_gru_multiseed.py.

Identical to 48c (seeds, folds, hyperparameters, undersampling, evaluation
protocol) except all torch compute runs on the MPS device:
  device = mps (fallback cpu if unavailable), torch.mps.manual_seed(SEED),
  model/batches moved to device, predictions returned via .cpu().numpy().
Weight init happens on CPU before .to(device), so init draws are identical
to the CPU run; only float op ordering on MPS may differ slightly.
Outputs: gru_results_inclprior_excl24h_mps_seed{SEED}.json (does not overwrite CPU results).
"""
import json, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
RES = 'results_vte'
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_SPLITS = 5
N_BOOT = 2000
NEG_CAP = 50_000
MAX_LEN = 42           # buckets (21 days)
BUCKET_H = 12
HIDDEN = 64
BATCH = 512
EVAL_BATCH = 4096
MAX_EPOCHS = 30
PATIENCE = 3
LR, WD = 1e-3, 1e-2

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device = {DEVICE}', flush=True)

torch.manual_seed(SEED)
if DEVICE.type == 'mps':
    torch.mps.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(8)

LABS = ['wbc', 'platelet', 'rdw', 'hemoglobin', 'creatinine', 'bun',
        'inr', 'pt', 'aptt']
VITALS = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
SEQ_COLS = ([f'lab_{k}_median' for k in LABS]
            + [f'lab_{k}_count' for k in LABS]
            + [f'vital_{k}_mean' for k in VITALS]
            + [f'vital_{k}_worst' for k in VITALS]
            + ['rx_heparin', 'rx_warfarin', 'rx_doac', 'rx_antiplatelet',
               'proc_cvc_cum', 'proc_mechvent_cum', 'proc_transfusion_cum'])
N_CH = len(SEQ_COLS)          # 37
N_IN = 2 * N_CH               # values + mask = 74
assert N_CH == 37

# ── load views ──
print('loading views ...', flush=True)
mdf_all = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet',
                          columns=['hadm_id', 'vte_event'])
lab_map = dict(zip(mdf_all['hadm_id'], mdf_all['vte_event'].astype(int)))
_sp = json.load(open(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_splits.json'))
pool_ids_all = set(_sp['train']) | set(_sp['val'])
tr_df = pd.read_parquet(f'{RES}/ajm/new/data_era/landmark_features_train_inclprior_excl24h.parquet')
ev_df = pd.read_parquet(f'{RES}/ajm/new/data_era/landmark_features_inclprior_excl24h.parquet')
tr_df = tr_df[tr_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
ev_df = ev_df[ev_df['hadm_id'].isin(pool_ids_all)].reset_index(drop=True)
tr_df['label'] = tr_df['hadm_id'].map(lab_map).astype(int)
ev_df['label'] = ev_df['hadm_id'].map(lab_map).astype(int)
N, NE = len(tr_df), len(ev_df)
y = tr_df['label'].values.astype(int)
groups = tr_df['subject_id'].values
yE = ev_df['label'].values.astype(int)
padua_raw_E = ev_df['padua_score'].values.astype(float)
print(f'train view {tr_df.shape} events={y.sum()} | eval view {ev_df.shape} '
      f'events={yE.sum()}', flush=True)

# ── static features (57 main cols) ──
fs = json.load(open(f'{RES}/feature_sets_v2.json'))
STATIC_FEATS = fs['main']
sdf = pd.read_parquet(f'{RES}/ajm/new/data_era/primary_inclprior_excl24h_model_dataset.parquet',
                      columns=['hadm_id'] + STATIC_FEATS)
hadm2srow = {h: i for i, h in enumerate(sdf['hadm_id'].values)}
S_all = sdf[STATIC_FEATS].values.astype(np.float32)
S_tr = S_all[tr_df['hadm_id'].map(hadm2srow).values]          # (N, 57)
S_ev = S_all[ev_df['hadm_id'].map(hadm2srow).values]          # (NE, 57)
assert not np.isnan(S_tr).all(axis=0).any(), 'static col all-NaN in pool'

# ── bucket merge -> per-hadm dense arrays ──
print('merging bucket tables ...', flush=True)
lb = pd.read_parquet(f'{RES}/buckets_labs_v2.parquet')
vb = pd.read_parquet(f'{RES}/buckets_vitals_v2.parquet')
rb = pd.read_parquet(f'{RES}/buckets_rx_v2.parquet')
pool_hadms = set(ev_df['hadm_id'].unique())
lb = lb[lb['hadm_id'].isin(pool_hadms)]
vb = vb[vb['hadm_id'].isin(pool_hadms)]
rb = rb[rb['hadm_id'].isin(pool_hadms)]
m = lb.merge(vb, on=['hadm_id', 'bucket_idx'], how='outer') \
      .merge(rb, on=['hadm_id', 'bucket_idx'], how='outer')
del lb, vb, rb
m = m.sort_values(['hadm_id', 'bucket_idx']).reset_index(drop=True)
print(f'merged bucket rows {len(m)}', flush=True)

bidx = m['bucket_idx'].values.astype(np.int64)
V = m[SEQ_COLS].values.astype(np.float32)
M = (~np.isnan(V)).astype(np.float32)
V = np.nan_to_num(V, nan=0.0)
hadm_arr = m['hadm_id'].values
del m

# per-hadm dense arrays covering [min_start, max_n_b) for its landmark rows
def nbuckets(hours):
    return (np.asarray(hours, dtype=np.float64) // BUCKET_H).astype(np.int64)

n_b_tr = nbuckets(tr_df['hours_since_adm'].values)
n_b_ev = nbuckets(ev_df['hours_since_adm'].values)
need = {}
for h, nb in zip(tr_df['hadm_id'].values, n_b_tr):
    lo = max(0, nb - MAX_LEN)
    cur = need.get(h)
    need[h] = (min(cur[0], lo), max(cur[1], nb)) if cur else (lo, nb)
for h, nb in zip(ev_df['hadm_id'].values, n_b_ev):
    lo = max(0, nb - MAX_LEN)
    cur = need.get(h)
    need[h] = (min(cur[0], lo), max(cur[1], nb)) if cur else (lo, nb)

print('building per-hadm dense arrays ...', flush=True)
t0 = time.time()
hadm_dense = {}   # hadm -> (offset, V_dense float16 (L,37), M_dense uint8)
tot = 0
# boundaries of hadm groups in sorted flat arrays
chg = np.flatnonzero(np.r_[True, hadm_arr[1:] != hadm_arr[:-1], True])
for i in range(len(chg) - 1):
    s, e = chg[i], chg[i + 1]
    h = hadm_arr[s]
    if h not in need:
        continue
    off, hi = need[h]
    L = hi - off
    vd = np.zeros((L, N_CH), dtype=np.float16)
    md = np.zeros((L, N_CH), dtype=np.uint8)
    rows = bidx[s:e] - off
    keep = (rows >= 0) & (rows < L)
    r = rows[keep]
    vd[r] = V[s:e][keep].astype(np.float16)
    md[r] = M[s:e][keep].astype(np.uint8)
    hadm_dense[h] = (off, vd, md)
    tot += L
del V, M, bidx, hadm_arr
print(f'dense arrays: {len(hadm_dense)} hadms, {tot} rows total, '
      f'~{tot * N_CH * 3 / 1e9:.2f} GB ({time.time() - t0:.0f}s)', flush=True)

# row -> (hadm, n_b) lookup arrays
hadm_tr = tr_df['hadm_id'].values
hadm_ev = ev_df['hadm_id'].values


def build_batch(hadms, n_bs, mu, sd):
    """left-padded (B, MAX_LEN, 74) float32 CPU tensor, normalized values."""
    B = len(hadms)
    x = np.zeros((B, MAX_LEN, N_IN), dtype=np.float32)
    for i in range(B):
        d = hadm_dense.get(hadms[i])
        if d is None:
            continue  # hadm has no bucket rows at all -> all-missing seq
        off, vd, md = d
        nb = int(n_bs[i])
        L = min(nb, MAX_LEN)
        if L <= 0:
            continue
        s = nb - L - off
        v = vd[s:s + L].astype(np.float32)
        mk = md[s:s + L].astype(np.float32)
        pos = MAX_LEN - L
        x[i, pos:, :N_CH] = (v - mu) / sd * mk
        x[i, pos:, N_CH:] = mk
    return torch.from_numpy(x)


# ── model ──
class GRUModel(nn.Module):
    def __init__(self, n_in=N_IN, n_static=len(STATIC_FEATS), hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64),
                                  nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, x_static):
        h, _ = self.gru(x_seq)
        z = torch.cat([h[:, -1], x_static], dim=1)
        return torch.sigmoid(self.head(z)).squeeze(-1)


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true):
        loss = nn.functional.binary_cross_entropy(y_pred, y_true,
                                                  reduction='none')
        w = torch.where(y_true == 1, self.pos_weight, 1.0)
        return (loss * w).mean()


def train_gru(tr_idx, va_idx, mu, sd, s_mu, s_sd, s_med):
    """train on tr_idx, early stop on va_idx (undersampled view rows)."""
    def stat(idx):
        xs = S_tr[idx].copy()
        for j in range(xs.shape[1]):
            xs[np.isnan(xs[:, j]), j] = s_med[j]
        return torch.from_numpy((xs - s_mu) / s_sd)

    Xst_tr, Xst_va = stat(tr_idx).to(DEVICE), stat(va_idx).to(DEVICE)
    ytr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(DEVICE)
    yva = y[va_idx]
    h_tr, nb_tr = hadm_tr[tr_idx], n_b_tr[tr_idx]
    h_va, nb_va = hadm_tr[va_idx], n_b_tr[va_idx]

    pos_weight = float((y[tr_idx] == 0).sum() / max(y[tr_idx].sum(), 1))
    model = GRUModel().to(DEVICE)
    crit = WeightedBCE(pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.RandomState(SEED)

    best_auc, best_state, bad, hist = 0.0, None, 0, []
    for ep in range(MAX_EPOCHS):
        te = time.time()
        model.train()
        order = rng.permutation(len(tr_idx))
        tot_loss = 0.0
        n_batches = 0
        for s in range(0, len(order), BATCH):
            bi = order[s:s + BATCH]
            xb = build_batch(h_tr[bi], nb_tr[bi], mu, sd).to(DEVICE)
            opt.zero_grad()
            pred = model(xb, Xst_tr[bi])
            loss = crit(pred, ytr[bi])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            n_batches += 1
        # val
        model.eval()
        vp = []
        with torch.no_grad():
            for s in range(0, len(va_idx), EVAL_BATCH):
                sl = slice(s, s + EVAL_BATCH)
                xb = build_batch(h_va[sl], nb_va[sl], mu, sd).to(DEVICE)
                vp.append(model(xb, Xst_va[sl]).cpu().numpy())
        vp = np.concatenate(vp)
        va_auc = roc_auc_score(yva, vp) if len(np.unique(yva)) > 1 else 0.5
        hist.append({'epoch': ep, 'train_loss': tot_loss / max(n_batches, 1),
                     'val_auc': float(va_auc)})
        print(f'    ep{ep:02d} loss={hist[-1]["train_loss"]:.4f} '
              f'val_auc={va_auc:.4f} ({time.time() - te:.0f}s)', flush=True)
        if va_auc > best_auc:
            best_auc, bad = va_auc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, hist, best_auc


@torch.no_grad()
def predict_rows(model, hadms, n_bs, S_rows, mu, sd, s_mu, s_sd, s_med):
    model.eval()
    xs = S_rows.copy()
    for j in range(xs.shape[1]):
        xs[np.isnan(xs[:, j]), j] = s_med[j]
    xs = torch.from_numpy((xs - s_mu) / s_sd).to(DEVICE)
    out = np.zeros(len(hadms), dtype=np.float32)
    for s in range(0, len(hadms), EVAL_BATCH):
        sl = slice(s, s + EVAL_BATCH)
        xb = build_batch(hadms[sl], n_bs[sl], mu, sd).to(DEVICE)
        out[sl] = model(xb, xs[sl]).cpu().numpy()
    return out


# ── helpers (same as scripts/47) ──
def fold_boot_ci(fr):
    rng = np.random.RandomState(SEED)
    vals = [np.mean([fr[i]['auc'] for i in rng.randint(0, len(fr), len(fr))])
            for _ in range(N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def fast_auc(yb, p):
    r = rankdata(p)
    n1 = int(yb.sum()); n0 = len(yb) - n1
    return float((r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(yb, p):
    pt, pp = calibration_curve(yb, p, n_bins=10)
    return float(np.mean(np.abs(pt - pp)))


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'estimate': float(vals.mean()), 'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


# ── dynamic XGB seed42 OOF on eval view (delta comparator), precomputed
# by scripts/48b_xgb_oof.py in a separate process (torch + XGBoost in one
# process segfaults on macOS: duplicate OpenMP runtimes) ──
oof_xgb_ev = np.load(f'{RES}/ajm/new/output_era/gru_xgb_oof_eval_inclprior_excl24h.npy')
print(f'XGB OOF eval AUC={roc_auc_score(yE, oof_xgb_ev):.4f} '
      '(precomputed by 48b)', flush=True)
ev_pos = {}
subjE = ev_df['subject_id'].values
for i, s in enumerate(subjE):
    ev_pos.setdefault(s, []).append(i)
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(sgkf.split(np.zeros((N, 1)), y, groups))

# ── 5-fold GRU CV ──
print(f'GRU 5-fold CV (seed {SEED}, device {DEVICE}) ...', flush=True)
oof_gru_ev = np.zeros(NE)
fold_rows, histories = [], []
for fi, (tr, te) in enumerate(FOLDS):
    t0 = time.time()
    # channel normalization from fold-train hadms' buckets (all buckets of
    # fold-train subjects, training window only is approximated by all rows)
    tr_hadms = set(hadm_tr[tr])
    vs, ms_ = [], []
    for h in tr_hadms:
        d = hadm_dense.get(h)
        if d is None:
            continue
        off, vd, md = d
        vs.append(vd.astype(np.float32)); ms_.append(md.astype(np.float32))
    vv = np.concatenate(vs); mm = np.concatenate(ms_)
    cnt = mm.sum(axis=0)
    mu = np.where(cnt > 0, (vv * mm).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0,
                   ((vv - mu) ** 2 * mm).sum(axis=0) / np.maximum(cnt, 1), 1.0)
    sd = np.sqrt(np.maximum(var, 1e-6))
    sd[sd == 0] = 1.0
    # static imputation/standardization from fold-train rows
    xs = S_tr[tr]
    s_med = np.nanmedian(xs, axis=0)
    xs2 = np.where(np.isnan(xs), s_med, xs)
    s_mu, s_sd = xs2.mean(axis=0), xs2.std(axis=0)
    s_sd[s_sd == 0] = 1.0

    model, hist, best = train_gru(tr, te, mu, sd, s_mu, s_sd, s_med)
    histories.append(hist)
    vp = predict_rows(model, hadm_tr[te], n_b_tr[te], S_tr[te],
                      mu, sd, s_mu, s_sd, s_med)
    fold_rows.append({'fold': fi, 'auc': float(roc_auc_score(y[te], vp)),
                      'auprc': float(average_precision_score(y[te], vp)),
                      'brier': float(brier_score_loss(y[te], vp)),
                      'epochs': len(hist), 'best_val_auc': float(best)})
    print(f'  fold {fi}: AUC={fold_rows[-1]["auc"]:.4f} '
          f'AUPRC={fold_rows[-1]["auprc"]:.4f} epochs={len(hist)} '
          f'({time.time() - t0:.0f}s)', flush=True)
    # OOF on full eval rows of held-out subjects
    te_subj = set(groups[te])
    rows = np.concatenate([ev_pos[s] for s in te_subj if s in ev_pos])
    oof_gru_ev[rows] = predict_rows(model, hadm_ev[rows], n_b_ev[rows],
                                    S_ev[rows], mu, sd, s_mu, s_sd, s_med)

ci = fold_boot_ci(fold_rows)
results = {
    'fold_mean_auc': float(np.mean([f['auc'] for f in fold_rows])),
    'fold_mean_auc_ci': ci,
    'fold_mean_auprc': float(np.mean([f['auprc'] for f in fold_rows])),
    'fold_mean_brier': float(np.mean([f['brier'] for f in fold_rows])),
    'n_folds': len(fold_rows),
    'folds': fold_rows,
    'oof_auc': float(roc_auc_score(yE, oof_gru_ev)),
    'oof_auprc': float(average_precision_score(yE, oof_gru_ev)),
    'oof_brier': float(brier_score_loss(yE, oof_gru_ev)),
    'oof_ece': ece(yE, oof_gru_ev),
}
print(f"GRU: fold-mean AUC={results['fold_mean_auc']:.4f} "
      f"[{ci[0]:.4f}-{ci[1]:.4f}] | OOF(eval) AUC={results['oof_auc']:.4f} "
      f"AUPRC={results['oof_auprc']:.4f} Brier={results['oof_brier']:.5f} "
      f"ECE={results['oof_ece']:.4f}", flush=True)

# ── patient-level cluster bootstrap on eval view ──
print('cluster bootstrap ...', flush=True)
subj_codes, subj_uniq = pd.factorize(ev_df['subject_id'])
n_subj = len(subj_uniq)
ev_rows = np.where(yE == 1)[0]
neg_rows = np.where(yE == 0)[0]
ev_subj, neg_subj = subj_codes[ev_rows], subj_codes[neg_rows]
rng = np.random.RandomState(SEED)
P = {'gru': oof_gru_ev, 'xgb': oof_xgb_ev, 'padua': padua_raw_E}
auc_boot = {k: [] for k in P}
deltas = {'gru_vs_xgb': [], 'gru_vs_padua': []}
for b in range(N_BOOT):
    counts = np.bincount(rng.randint(0, n_subj, n_subj), minlength=n_subj)
    ev_mult = counts[ev_subj]
    take = ev_mult > 0
    if take.sum() < 2:
        continue
    be = np.repeat(ev_rows[take], ev_mult[take])
    bn = neg_rows[counts[neg_subj] > 0]
    if len(bn) > NEG_CAP:
        bn = rng.choice(bn, NEG_CAP, replace=False)
    rows = np.concatenate([be, bn])
    yb = yE[rows]
    a = {k: fast_auc(yb, P[k][rows]) for k in P}
    for k in a:
        auc_boot[k].append(a[k])
    deltas['gru_vs_xgb'].append(a['gru'] - a['xgb'])
    deltas['gru_vs_padua'].append(a['gru'] - a['padua'])
    if (b + 1) % 500 == 0:
        print(f'  boot {b + 1}/{N_BOOT}', flush=True)

results['oof_inference'] = {
    'method': 'patient-level cluster bootstrap (subject_id) on full eval '
              'view (train+val landmark rows), 2000 replicates; negatives '
              'downsampled to 50k per replicate, AUC via Mann-Whitney rank '
              'statistic (same as scripts/47)',
    'n_replicates_used': len(auc_boot['gru']),
    'oof_auc_ci': {k: {'auc': float(fast_auc(yE, P[k])),
                       'ci': [float(v) for v in
                              np.percentile(v2, [2.5, 97.5])]}
                   for k, v2 in auc_boot.items()},
    'delta_auc': {k: ds(v) for k, v in deltas.items()},
}
for k, d in results['oof_inference']['delta_auc'].items():
    print(f"  dAUC {k}: {d['estimate']:+.4f} [{d['ci'][0]:+.4f},"
          f"{d['ci'][1]:+.4f}] p={d['p_two_sided']:.4f}", flush=True)

results['training'] = {
    'histories': histories,
    'note': 'early stop on fold-test val AUC, patience 3; loss printed '
            'per epoch (no divergence observed if train_loss decreases)',
}
results['meta'] = {
    'device': str(DEVICE),
    'seq_channels': SEQ_COLS,
    'n_in_channels': N_IN,
    'max_len_buckets': MAX_LEN,
    'static_features': STATIC_FEATS,
    'n_pool_undersampled': N, 'n_events_pool': int(y.sum()),
    'n_eval': NE, 'n_events_eval': int(yE.sum()),
    'cv': f'StratifiedGroupKFold 5 folds (seed {SEED}) on undersampled '
          'train+val view, '
          'stratify label, group subject_id; fold bootstrap 2000 for '
          f'fold-mean AUC CI; seed{SEED} OOF on full eval view (batch 4096)',
    'model': 'GRU(74->64, 1 layer) -> last hidden -> concat static 57 -> '
             'MLP(121->64->1) -> sigmoid; left-padded sequences, last-42 '
             'buckets truncation; per-fold channel standardization on '
             'observed values; static per-fold median imputation + '
             'standardization',
    'training': f'WeightedBCE pos_weight=fold neg/pos, AdamW lr 1e-3 '
                f'wd 1e-2, batch 512, <=30 epochs, early stop val AUC '
                f'patience 3, grad clip 1.0, seed {SEED}, device {DEVICE} '
                f'(MPS re-run of 48c CPU protocol)',
}

outp = f'{RES}/ajm/new/output_era/gru_results_inclprior_excl24h_mps_seed{SEED}.json'
with open(outp, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved -> {outp}', flush=True)
