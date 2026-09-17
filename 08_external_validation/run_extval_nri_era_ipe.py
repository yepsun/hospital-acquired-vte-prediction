#!/usr/bin/env python3
"""External NRI / IDI for the era-freeze arms (57 / 42 / 40 features).

The main external-validation runs report NRI/IDI only for the data-freeze
57-feature models (run_external_validation_ipe.py). This script re-runs the
same NRI/IDI machinery for the era-pool frozen models
(frozen_models_era_deploysafe/*), i.e. the models trained on the 313,064-
admission / 554-event MIMIC development pool of the primary analysis, on:

  * the fully enumerated high-risk cohort (no weighting)   -> ..._highrisk.json
  * the weighted full population (inverse-probability sampling weight on the
    non-high-risk stratum)                                 -> ..._weighted.json

Comparisons (each arm, both model families), at the imported MIMIC-derived
absolute thresholds 0.5% and 1%:
    57 / 42 / 40 features vs. Padua
    57 / 42 / 40 features vs. IMPROVE

NRI uses the corrected category-based Pencina (2008) definition, i.e.
    [P(up|event) - P(down|event)] + [P(down|non-event) - P(up|non-event)]
identical to wnri()/nri() in run_external_validation_ipe.py; IDI is
widi()/idi() from the same file. Probabilities are put on the external
probability scale exactly as there: Platt (univariate logistic) maps are
fitted with the sampling weights for Padua, for IMPROVE and for each frozen
model's probability, so all comparisons are on one scale.

Bootstrap: subject-level (patient-level, cluster) resampling of the local
patients with replacement, 2,000 replicates, 95% percentile CIs. The legacy
admission-level Efron bootstrap of run_external_validation_ipe.py is run as a
sensitivity (it reproduces the published data-freeze numbers from the
main57_legacy arm, which is included here purely as a regression check).

Outputs (new files; nothing is overwritten):
    extval/external_validation_nri_era_highrisk.json
    extval/external_validation_nri_era_weighted.json

Run:  /Users/Yepsun/myenv/bin/python3 run_extval_nri_era_ipe.py
      EXTVAL_HIGH_ONLY=1 /Users/Yepsun/myenv/bin/python3 run_extval_nri_era_ipe.py
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLR

BASE = '/Users/Yepsun/Mywork/Vscodeprojects/VTE/extval'
SEED = 42
N_BOOT = int(os.environ.get('EXTVAL_N_BOOT', '2000'))
NEG_CAP = 50_000
NRI_THRESHOLDS = [0.005, 0.01]

NH_FULL, NH_SAMP = 76_188, 7_072
SW_HIGH, SW_NONHIGH = 1.0, NH_FULL / NH_SAMP

HIGH_RISK_ONLY = os.environ.get('EXTVAL_HIGH_ONLY', '0') == '1'

ARMS = {
    # data-freeze 57-feature models: regression check against the published
    # high-risk NRI/IDI values (XGBoost vs Padua 0.000 (0.5%) / +0.039 (1%),
    # IDI -0.0057; vs IMPROVE 0.000 / -0.067, IDI -0.0095)
    'main57_legacy': f'{BASE}/frozen_models_ipe',
    # era-pool refits: the models the manuscript describes
    'main57_era': f'{BASE}/frozen_models_era_deploysafe/main_57',
    'strict42_era': f'{BASE}/frozen_models_era_deploysafe/strict_42',
    'cons40_era': f'{BASE}/frozen_models_era_deploysafe/conservative_40',
}

ARM_LABEL = {'main57_legacy': '57 features (data-freeze, regression check)',
             'main57_era': '57 features',
             'strict42_era': '42 features (admission-safe)',
             'cons40_era': '40 features (conservative)'}


def wnri(p_old, p_new, yb, thr, w):
    """Corrected category-based NRI, weighted. Verbatim from
    run_external_validation_ipe.py:119-125."""
    ev, nev = yb == 1, yb == 0
    up_e = w[(p_new >= thr) & (p_old < thr) & ev].sum()
    dn_e = w[(p_new < thr) & (p_old >= thr) & ev].sum()
    up_n = w[(p_new >= thr) & (p_old < thr) & nev].sum()
    dn_n = w[(p_new < thr) & (p_old >= thr) & nev].sum()
    return float((up_e - dn_e) / w[ev].sum() + (dn_n - up_n) / w[nev].sum())


def widi(p_old, p_new, yb, w):
    """Weighted IDI. Verbatim from run_external_validation_ipe.py:143-149."""
    ev, nev = yb == 1, yb == 0
    new_e = np.sum(w[ev] * p_new[ev]) / w[ev].sum()
    old_e = np.sum(w[ev] * p_old[ev]) / w[ev].sum()
    new_n = np.sum(w[nev] * p_new[nev]) / w[nev].sum()
    old_n = np.sum(w[nev] * p_old[nev]) / w[nev].sum()
    return float((new_e - old_e) - (new_n - old_n))


def ds(vals):
    vals = np.array(vals)
    ci = np.percentile(vals, [2.5, 97.5])
    p = min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {'boot_mean': float(vals.mean()),
            'ci': [float(ci[0]), float(ci[1])],
            'p_two_sided': float(p)}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def platt(x_raw, y, w):
    """Univariate logistic (Platt) map, fitted with the sampling weights."""
    return SkLR(C=1.0, max_iter=2000).fit(
        np.asarray(x_raw, dtype=float).reshape(-1, 1), y,
        sample_weight=w).predict_proba(
        np.asarray(x_raw, dtype=float).reshape(-1, 1))[:, 1]


def main():
    t0 = time.time()
    co = pd.read_parquet(f'{BASE}/cohort_ipe.parquet',
                         columns=['vis_id', 'pat_id', 'risk_group',
                                  'vte_outcome', 'prior_vte_any',
                                  'prior_vte_this_adm'])
    fe = pd.read_parquet(f'{BASE}/features.parquet').drop(
        columns=['prior_vte_any', 'vte_outcome', 'risk_group'], errors='ignore')
    sc = pd.read_parquet(f'{BASE}/scores.parquet')
    df = co.merge(fe, on='vis_id', how='left', validate='1:1') \
           .merge(sc, on='vis_id', how='left', validate='1:1')
    df['prior_vte'] = df['prior_vte_this_adm']
    df['sw'] = np.where(df['risk_group'] == 'non_high', SW_NONHIGH, SW_HIGH)
    if HIGH_RISK_ONLY:
        df = df[df['risk_group'] == 'high'].copy()
        df['sw'] = 1.0
    df = df.reset_index(drop=True)
    y = df['vte_outcome'].astype(int).values
    w = df['sw'].values.astype(float)
    pat = df['pat_id'].values
    print(f'cohort n={len(df):,} patients={len(np.unique(pat)):,} '
          f'events={int(y.sum()):,} rate={y.mean():.4f} '
          f'(weighted rate={np.sum(w * y) / w.sum():.4f}) '
          f'high_only={HIGH_RISK_ONLY}', flush=True)

    # ---- frozen-model probabilities per arm ------------------------------
    probs = {}
    for key, d in ARMS.items():
        fpath = f'{d}/feature_order.json'
        if not os.path.exists(fpath):
            print(f'  skip {key}: {fpath} not found', flush=True)
            continue
        feats = json.load(open(fpath))['features']
        usable = [f for f in feats if f in df.columns]
        missing = [f for f in feats if f not in df.columns]
        X = df[usable].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        X = X.clip(-1e30, 1e30)
        for f in missing:
            X[f] = np.float32(np.nan)
        X = X[feats]
        lr = joblib.load(f'{d}/final_lr.joblib')
        imp = joblib.load(f'{d}/final_lr_imputer.joblib')
        xgb = joblib.load(f'{d}/final_xgb.joblib')
        probs[key] = {'lr': lr.predict_proba(imp.transform(X))[:, 1],
                      'xgb': xgb.predict_proba(X)[:, 1]}
        print(f'  {key}: {len(feats)} features, '
              f'{len(usable) - sum(df[f].isna().all() for f in usable)} '
              f'reconstructable', flush=True)

    # ---- common probability scale (Platt maps, weighted) ------------------
    # Scores are mapped from the raw integer score (their natural linear
    # predictor). Frozen model probabilities are mapped from their LOG-ODDS:
    # fitting Platt on the raw probability value fails here because the value
    # spans only ~0.0002-0.09, so the L2-regularised coefficient collapses to
    # ~0 and every admission is mapped to the base rate (checked: the
    # high-risk cohort maps to 0.019183 +/- 0.000004, the weighted population
    # to 0.011997 +/- 0.000001, identically for LR and XGBoost and for every
    # feature set). That degenerate map is reproduced below as
    # 'platt_on_probability__defective' for the record; all reported NRI/IDI
    # use the log-odds map, which is the same estimator on the correct scale.
    pad_prob = platt(df['padua_score'].values.astype(float), y, w)
    impr_prob = platt(df['improve_score'].values.astype(float), y, w)
    P = {'padua': pad_prob, 'improve': impr_prob}
    P_defect = {'padua': pad_prob, 'improve': impr_prob}
    map_diag = {}
    for key, mm in probs.items():
        for fam in ['lr', 'xgb']:
            P[f'{key}__{fam}'] = platt(logit(mm[fam]), y, w)
            P_defect[f'{key}__{fam}'] = platt(mm[fam], y, w)
            map_diag[f'{key}__{fam}'] = {
                'raw_prob_min': float(mm[fam].min()),
                'raw_prob_max': float(mm[fam].max()),
                'logodds_recal_min': float(P[f'{key}__{fam}'].min()),
                'logodds_recal_max': float(P[f'{key}__{fam}'].max()),
                'probability_recal_min': float(P_defect[f'{key}__{fam}'].min()),
                'probability_recal_max': float(P_defect[f'{key}__{fam}'].max()),
                'probability_recal_n_unique_9dp': int(len(
                    np.unique(np.round(P_defect[f'{key}__{fam}'], 9)))),
            }

    comparisons = []
    for key in probs:
        for fam in ['lr', 'xgb']:
            for comp in ['padua', 'improve']:
                comparisons.append((f'{key}__{fam}', comp))

    # ---- full-sample estimates -------------------------------------------
    full = {}
    for new, comp in comparisons:
        full[f'{new}__vs_{comp}__nri'] = {
            f'thr{t}': wnri(P[comp], P[new], y, t, w) for t in NRI_THRESHOLDS}
        full[f'{new}__vs_{comp}__idi'] = widi(P[comp], P[new], y, w)

    # ---- full-sample estimate under the defective legacy map (record only) --
    legacy_map = {}
    for new, comp in comparisons:
        legacy_map[f'{new}__vs_{comp}__nri'] = {
            f'thr{t}': wnri(P_defect[comp], P_defect[new], y, t, w)
            for t in NRI_THRESHOLDS}
        legacy_map[f'{new}__vs_{comp}__idi'] = widi(
            P_defect[comp], P_defect[new], y, w)

    # ---- bootstrap (log-odds map = primary) --------------------------------
    uniq_pat, inv = np.unique(pat, return_inverse=True)
    rows_by_pat = [np.where(inv == i)[0] for i in range(len(uniq_pat))]
    boot = {}
    for mode in ['patient', 'admission']:
        rng = np.random.RandomState(SEED)
        acc = {(new, comp, kind, t): [] for new, comp in comparisons
               for kind in ['nri', 'idi'] for t in NRI_THRESHOLDS}
        n = len(y)
        for b in range(N_BOOT):
            if mode == 'patient':
                samp = rng.choice(len(uniq_pat), len(uniq_pat), replace=True)
                rows = np.concatenate([rows_by_pat[i] for i in samp])
            else:
                idx = rng.randint(0, n, n)
                b_ev = idx[y[idx] == 1]
                b_neg = idx[y[idx] == 0]
                if len(b_neg) > NEG_CAP:
                    b_neg = rng.choice(b_neg, NEG_CAP, replace=False)
                rows = np.concatenate([b_ev, b_neg])
            yb = y[rows]
            wb = w[rows]
            for new, comp in comparisons:
                po, pn = P[comp][rows], P[new][rows]
                for t in NRI_THRESHOLDS:
                    acc[(new, comp, 'nri', t)].append(
                        wnri(po, pn, yb, t, wb))
                acc[(new, comp, 'idi', NRI_THRESHOLDS[0])].append(
                    widi(po, pn, yb, wb))
            if (b + 1) % 500 == 0:
                print(f'  [{mode}] boot {b + 1}/{N_BOOT} '
                      f'({time.time() - t0:.0f}s)', flush=True)
        for (new, comp, kind, t), v in acc.items():
            if kind == 'nri':
                boot[f'{mode}__{new}__vs_{comp}__nri_thr{t}'] = ds(v)
            elif t == NRI_THRESHOLDS[0]:
                boot[f'{mode}__{new}__vs_{comp}__idi'] = ds(v)

    out = {
        'meta': {
            'purpose': 'external NRI/IDI for the era-freeze arms, replacing '
                       'the data-freeze provenance caveat in the supplement',
            'analysis_population': ('high_risk_only' if HIGH_RISK_ONLY
                                    else 'full_population_weighted'),
            'nri_definition': 'corrected category-based Pencina (2008) NRI, '
                              '[P(up|event)-P(down|event)] + '
                              '[P(down|non-event)-P(up|non-event)]; verbatim '
                              'wnri()/widi() from run_external_validation_ipe.py',
            'thresholds': NRI_THRESHOLDS,
            'probability_scale': 'Platt (univariate logistic) maps fitted with '
                                 'the sampling weights on the external cohort. '
                                 'Padua and IMPROVE are mapped from the raw '
                                 'integer score; frozen model probabilities are '
                                 'mapped from their LOG-ODDS. Mapping the model '
                                 'probability from the probability value itself '
                                 '(the legacy code path) collapses: the '
                                 'L2-regularised coefficient goes to ~0 and '
                                 'every admission is assigned the base rate, so '
                                 'that map is kept only as '
                                 'platt_on_probability__defective.',
            'bootstrap_primary': 'subject-level (patient-level cluster) '
                                 'resampling of local patients, 2,000 '
                                 'replicates, percentile 95% CI',
            'bootstrap_sensitivity': 'admission-level Efron bootstrap using the '
                                     'legacy resampling design and negative cap',
            'frozen_model_dirs': ARMS,
            'arm_labels': ARM_LABEL,
            'n_boot': N_BOOT, 'seed': SEED,
        },
        'cohort': {'n': int(len(df)), 'patients': int(len(uniq_pat)),
                   'events': int(y.sum()),
                   'event_rate_unweighted': float(y.mean()),
                   'event_rate_weighted': float(np.sum(w * y) / w.sum())},
        'probability_maps': map_diag,
        'full_sample': full,
        'full_sample_platt_on_probability__defective': legacy_map,
        'bootstrap': boot,
        'runtime_s': round(time.time() - t0, 1),
    }
    out_path = (f'{BASE}/external_validation_nri_era_highrisk.json'
                if HIGH_RISK_ONLY
                else f'{BASE}/external_validation_nri_era_weighted.json')
    json.dump(out, open(out_path, 'w'), indent=2, default=str)
    print(f'\nsaved -> {out_path} ({out["runtime_s"]}s)', flush=True)

    for new, comp in comparisons:
        print(f'{new} vs {comp}: full-sample NRI '
              f'{full[f"{new}__vs_{comp}__nri"]}, IDI '
              f'{full[f"{new}__vs_{comp}__idi"]:+.4f}')
        for t in NRI_THRESHOLDS:
            k = f'patient__{new}__vs_{comp}__nri_thr{t}'
            print(f'   patient-boot thr{t}: {boot[k]}')
        print(f'   patient-boot IDI: '
              f'{boot[f"patient__{new}__vs_{comp}__idi"]}')
        for t in NRI_THRESHOLDS:
            k = f'admission__{new}__vs_{comp}__nri_thr{t}'
            print(f'   admission-boot thr{t}: {boot[k]}')
        print(f'   admission-boot IDI: '
              f'{boot[f"admission__{new}__vs_{comp}__idi"]}')


if __name__ == '__main__':
    main()
