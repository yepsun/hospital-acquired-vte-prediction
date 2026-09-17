# Analysis code: machine learning versus Padua/IMPROVE/Caprini for in-hospital VTE prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22168152.svg)](https://doi.org/10.5281/zenodo.22168152)
<!-- DOI and version to be updated for the next release -->

The complete analysis code for:

> Feng J, Li Y, Yu S, Shi J, Sun X. *Machine learning prediction of radiologically
> confirmed hospital-acquired venous thromboembolism: comparison with the Padua and
> IMPROVE scores and external validation.*

Everything reported in the manuscript and its supplement is produced by the scripts
in this repository, with two exceptions that are stated where they occur: the
diagnostic-imaging extraction and outcome-adjudication pipeline (Step 2 below) and
the external hospital's locally held data (Step 9).

## Cohort, in one paragraph

MIMIC-IV v3.1 linked to MIMIC-IV-Note v2.2 and MIMIC-IV-ED v2.2. From 425,090
adult admissions (age ≥18 y, length of stay >24 h), the 35,552 admissions anchored
to 2020–2022 were excluded a priori because MIMIC-IV-Note v2.2 contains no
radiology reports for that era; of the remaining 389,538, a further 9,818 were
excluded (VTE ICD code with no radiological confirmation, 9,758; imaging-confirmed
VTE diagnosed within 24 h of admission, 1,471; categories overlapping). The
**primary analytic cohort is 379,720 admissions (674 new in-hospital VTE:
PE 507, DVT 183, both 16) in 159,397 patients**, restricted to anchor years
2008–2019. Admissions with prior VTE are **retained** (34,109 admissions; 96
events). The model-development pool (train + validation) is **313,064 admissions,
554 events**; the held-out test partition is 66,656 admissions, 120 events.
Event rate 0.178% in the full cohort, 0.177% in the development pool.

## Pipeline

Run every script from the repository root of your local working copy, i.e. from a
directory that contains a `results_vte/` tree (see *Data requirements*). All scripts
resolve inputs and outputs as `results_vte/...`; the era-restricted pipeline reads
`results_vte/ajm/new/data_era/` and writes `results_vte/ajm/new/output_era/`, and
the shipped `07_results_summary/` is a copy of that output directory.

| Step | Directory | Scripts | What it does |
|---|---|---|---|
| 1. Feature extraction (upstream) | `01_cohort/` | `85d_treatments_v3.py`, `85e_clinical_scores_v3.py`, `85f_assemble_v3.py`, `86_build_model_dataset_v3.py`, `85h_clinical_scores_caprini.py` | `85d` extracts the admission-24-h medication and procedure features (`features_treatments_v3.parquet`: heparin/warfarin/DOAC/antiplatelet/hormone, CVC, mechanical ventilation, transfusion, ICU hours); `85e` computes the Padua and IMPROVE scores with their declared proxies (`clinical_scores_v3.parquet`); `85f` assembles the static feature matrix with leakage checks (`features_static_v3.parquet`); `86` builds the model dataset, the feature-set definition and the splits (`model_dataset_v2.parquet`, `feature_sets_v2.json`, `splits_v3.json`); `85h` computes the Caprini approximation (`clinical_scores_v3_caprini.parquet`). Item-level mappings are documented in `score_mapping.md` and `score_mapping_caprini.md`. |
| 2. Outcome ascertainment | *not included* | — | Radiology reporting → PE/DVT event labelling, with the expert-adjudicated MIMIC-IV-Ext-PE reconciliation, the rule-based DVT classifier, duplicate human adjudication and third-party resolution described in the supplement (Table S6). These scripts operate on a credentialed local MIMIC-IV-Note copy and are not redistributed; their outputs are `results_vte/ajm/new/data/vte_labels_*.parquet`. |
| 3. Cohort and splits | `01_cohort/` | `build_new_cohorts.py`, `build_primary_inclprior_excl24h.py`, `build_inclprior_cohort_era_fixed.py`, `85i_join_caprini_era_cohorts.py` | Build the primary arm and the alternative inclusion-criteria arms, with patient-grouped 70/15/15 train/val/test splits and a chronological 85/15 temporal cut; rebuild the corrected "include prior VTE, retain 24-h-window events" arm; join the Caprini score into each arm. The era restriction itself is a row/key filter on `anchor_year_group` (2008–2010, 2011–2013, 2014–2016, 2017–2019), applied through `data_era/era_map.parquet`; the reference implementation of that filter is the `era_map` block at the end of `build_inclprior_cohort_era_fixed.py`. |
| 4. Features | `02_features/` | `44b_build_landmarks_v2.py`, `46b_landmark_features_v2.py`, `feature_sets_v2.json` | Build the 48-hour landmark skeleton and the 92-column time-updated landmark feature matrices (evaluation view and negatives-undersampled training view). `feature_sets_v2.json` defines the 57-feature main set and the 12 ED-triage vital columns; the 42-feature admission-safe, 44-feature surgery/trauma upper bound and 40-feature conservative sets are defined inside `04_evaluation/deploysafe/98c_deploysafe_full_suite.py`. Inputs are the 12-hour bucket tables (`results_vte/buckets_*_v2.parquet`) from the bucket-aggregation stage of the extraction pipeline. |
| 5. Models — static | `03_models/static/` | `39b_baseline_scores_eval_v2.py`, `39c_baseline_scores_eval_caprini.py`, `61_static_oof_multiseed.py`, `61b_static_oof_multiseed_era_fixed.py` | EHR-computed Padua/IMPROVE (and Caprini) test-set baselines; static LR/XGBoost out-of-fold evaluation with per-seed cluster-bootstrap ΔAUC for seeds 42/43/44, on the primary arm and on the corrected sensitivity arm. |
| 6. Models — dynamic and sequence | `03_models/dynamic/`, `03_models/gru/` | `47b_dynamic_models_v2.py`, `47b_dynamic_models_72h.py`, `47b_bootstrap_only.py`, `47b_bootstrap_only_72h.py`, `48e_gru_mps_primary.py` | Dynamic landmark LR/XGBoost at the 48-h and 72-h starting landmarks, with the cluster-bootstrap inference step split into its own script because it consumes the OOF arrays written by the model run; GRU sequence model (three 5-fold repeats, seeds 42/43/44) on Apple-silicon MPS. |
| 7. Evaluation | `04_evaluation/` | see the subdirectories below | Calibration and decision curves, nested (cross-fitted) calibration, DCA estimand reconciliation and the manuscript grid table, temporal/ICU/surgery/cancer/heparin/prior-VTE/service subgroups, strict-control, ED-vitals, treatment-free and inclusion-criteria sensitivities, NRI/IDI, the Caprini benchmark comparison, and the deployment-safe (42-/40-/44-feature) suite. |
| 8. Figures | `05_figures/` | `58_publication_figures_v3.py`, `Figure_1_cohort_flow.mmd` | Figure 2 (calibration of predicted risk) and the decision-curve panels used for the calibration/decision-curve numbers; `Figure_1_cohort_flow.mmd` is the Mermaid source of Figure 1 (cohort flow). |
| 9. External validation | `08_external_validation/` | 19 scripts | Cohort build, featurization, score computation, LLM-assisted report pre-screening, human review, model freezing, external validation, nested calibration and NRI re-runs for the independent Beijing cohort. **The external patient data and the frozen model binaries are not redistributable and are not included** — see that directory's section below. |

`run_scripts/` holds the three drivers used in the authoring environment
(`run_all_era.sh`, `run_era.sh`, `run_era_resume.sh`). They contain absolute paths
from that environment; use them as a record of the run order rather than as
drop-in launchers.

## Results → file map

`07_results_summary/` is a verbatim copy of `results_vte/ajm/new/output_era/`
(the era-restricted run). Every number in the manuscript and supplement can be
traced to one of these files.

### Main text

| Manuscript element | File | Produced by |
|---|---|---|
| Table 1 — baseline characteristics by VTE status | *no shipped results file*: derived from the cohort dataset built in Step 3. The descriptive quantities quoted in the text (median length of stay, median time from admission to the diagnostic report 4.4 d, heparin exposure 57.1%, strict-control descriptives) are in `review_gap_results_inclprior_excl24h.json` → `descriptives` | 03 (cohort), `04_evaluation/sensitivity/62_review_gaps.py` |
| Table 2 — static model performance (fold-mean, OOF, ΔAUC, Caprini row, test set) | `sensitivity_inclprior_excl24h_result.json` (fold-mean AUC, fold-mean CI), `sensitivity_inclprior_excl24h_caprini_result.json` (Caprini/IMPROVE pooled OOF rows), `static_oof_multiseed_inclprior_excl24h.json` (OOF AUC and ΔAUC per seed), `baseline_results_caprini_inclprior_excl24h.json` (test-set AUCs, high-risk sensitivity/specificity/PPV/NPV) | `04_evaluation/sensitivity/98_ajm_sensitivity_cohorts.py`, `04_evaluation/sensitivity/98b_caprini_comparison.py`, `03_models/static/61_static_oof_multiseed.py`, `03_models/static/39c_baseline_scores_eval_caprini.py` |
| Table 2 footnote — NRI / IDI vs Padua | `static_oof_nri_idi_fixed_inclprior_excl24h.json` | `04_evaluation/nri_idi/67_static_nri_idi_fixed.py` |
| Table 3 — dynamic landmark models | `dynamic_model_results_inclprior_excl24h.json` (OOF/fold-mean, broadcast static comparison, Padua broadcast), `dynamic_oof_inference_inclprior_excl24h_nri_fixed.json` (ΔAUC and corrected NRI/IDI), `gru_results_inclprior_excl24h_mps_seed42.json` (also `_seed43`, `_seed44`) | `03_models/dynamic/47b_dynamic_models_v2.py`, `03_models/dynamic/47b_bootstrap_only.py`, `04_evaluation/nri_idi/66_dynamic_nri_idi_fixed.py`, `03_models/gru/48e_gru_mps_primary.py` |
| Table 4 — external validation | *not shipped* (external data are not redistributable) | `08_external_validation/run_external_validation_deploysafe_ipe.py` |
| Figure 1 — cohort flow | `05_figures/Figure_1_cohort_flow.mmd` | Mermaid source; counts from Step 3 |
| Figure 2 — calibration | `calibration_dca_inclprior_excl24h.json`, `nested_calibration_dca_inclprior_excl24h.json` | `04_evaluation/calibration_dca/41b_calibration_dca_v2.py`; figure drawn by `05_figures/58_publication_figures_v3.py` |
| Decision-curve panels | `calibration_dca_inclprior_excl24h.json`, `nested_calibration_dca_inclprior_excl24h.json`, `dca_reconciliation_inclprior_excl24h.json`, `dca_manuscript_grid_table_inclprior_excl24h.json` | `04_evaluation/calibration_dca/41b_…`, `65_…`, `69_…`, `70_…`; figure drawn by `05_figures/58_publication_figures_v3.py` |

### Supplement

| Manuscript element | File | Produced by |
|---|---|---|
| Table S1 — Padua / IMPROVE item mapping | `01_cohort/score_mapping.md` | narrative document accompanying `85e_clinical_scores_v3.py` |
| Table S2 — subgroup discrimination | `review_gap_results_inclprior_excl24h.json` → `subgroups` (ICU, surgery, cancer, heparin), `service_subgroups_inclprior_excl24h.json` and `service_subgroups_caprini_inclprior_excl24h.json` (medical/surgical service), `prior_vte_stratified_auc.json` | `04_evaluation/sensitivity/62_review_gaps.py`, `04_evaluation/subgroups/60_service_subgroups_v2.py`, `04_evaluation/subgroups/60b_service_subgroups_caprini.py`, `04_evaluation/subgroups/63_prior_stratified_auc.py` |
| Table S3a — outcome/control sensitivity | `ed_vitals_sensitivity_inclprior_excl24h.json` (ED triage vitals row), `strict_control_apples2apples_era_fixed.json` → `dev_main` / `dev_strict` (strict-control row of the main-cohort column) | `04_evaluation/sensitivity/55b_ed_vitals_sensitivity_v2.py`, `04_evaluation/sensitivity/50c_strict_control_apples2apples_era_fixed.py` |
| Table S3b — like-for-like control definition comparison | `strict_control_apples2apples_era_fixed.json` → `dev_main` / `dev_strict`, `protocols.fixed` | `04_evaluation/sensitivity/50c_strict_control_apples2apples_era_fixed.py` |
| Table S3c — look-ahead (admission-safe) feature sets | `deploysafe_full_suite_main_57_deploysafe.json`, `deploysafe_full_suite_strict_42_deploysafe.json`, `deploysafe_full_suite_strict44_surgtrauma_deploysafe.json`, `deploysafe_full_suite_conservative_40_deploysafe.json` | `04_evaluation/deploysafe/98c_deploysafe_full_suite.py` |
| Table S4 — landmark-start sensitivity (72 h) | `dynamic_model_results_72h_inclprior_excl24h.json`, `dynamic_oof_inference_72h_inclprior_excl24h_nri_fixed.json` | `03_models/dynamic/47b_dynamic_models_72h.py`, `03_models/dynamic/47b_bootstrap_only_72h.py`, `04_evaluation/nri_idi/66_dynamic_nri_idi_fixed.py` |
| Table S5 — threshold operating characteristics | `nested_calibration_dca_inclprior_excl24h.json`, `nested_calibration_metrics_extra_inclprior_excl24h.json` (the three per-seed partial files are the intermediate dumps that feed the combined file) | `04_evaluation/calibration_dca/65_nested_calibration_dca_fixed.py`, `04_evaluation/calibration_dca/68_nested_calibration_metrics_extra.py` |
| Table S6 — outcome-ascertainment validation | *no shipped results file*: adjudication counts are recorded in the ascertainment pipeline (Step 2) | — |
| Table S7 — inclusion-criteria 2×2 | `sensitivity_inclprior_excl24h_result.json` (primary), `sensitivity_keep24h_result.json`, `sensitivity_excl24h_result.json`, `sensitivity_inclprior_fixed_result.json` (corrected final cell) | `04_evaluation/sensitivity/98_ajm_sensitivity_cohorts.py`, `01_cohort/build_inclprior_cohort_era_fixed.py` + `04_evaluation/sensitivity/98_ajm_sensitivity_cohorts.py` |
| Table S8 / S10 — external validation | *not shipped* | `08_external_validation/run_external_validation_deploysafe_ipe.py`, `08_external_validation/run_extval_nested_calibration_ipe.py` |
| Table S9 — calibration and decision-curve summary | `nested_calibration_dca_inclprior_excl24h.json` (cross-fitted calibration and OOF DCA), `dca_reconciliation_inclprior_excl24h.json` (estimand/provenance reconciliation), `dca_manuscript_grid_table_inclprior_excl24h.json` (the exact 0.5–2% band means and per-threshold grid values quoted in the table), `calibration_dca_inclprior_excl24h.json` (test-partition sensitivity) | `04_evaluation/calibration_dca/65_…`, `69_…`, `70_…`, `41b_…` |
| Table S11 — Caprini item mapping | `01_cohort/score_mapping_caprini.md`, `caprini_distribution.json` (score distribution, item prevalence, high-risk proportions, mapping-error bounds) | `01_cohort/85h_clinical_scores_caprini.py` writes `caprini_distribution.json`; the mapping is documented in `01_cohort/score_mapping_caprini.md` |
| Table S12 — treatment-predictor sensitivity | `no_rx_sensitivity_inclprior_excl24h.json` (52-feature static arms, subgroups by heparin exposure, ΔAUC/NRI/IDI), `no_rx_foldmean_98convention_inclprior_excl24h.json` (52-feature fold-mean under the Table 2 fixed-seed convention), `dynamic_norx_model_results_inclprior_excl24h.json` (88-feature dynamic arms) | `04_evaluation/sensitivity/64_no_rx_sensitivity.py`, `64c_no_rx_foldmean_98conv.py`, `64b_no_rx_dynamic.py` |
| Deploy-safe suite (preferred deployment feature set) | `deploysafe_full_suite_summary_deploysafe.json` (all four arms in one file), `deploysafe_subgroups_deploysafe.json` (medical/surgical subgroups for every arm), `deploysafe_auprc_ci_deploysafe.json` (AUPRC with bootstrap CIs) | `04_evaluation/deploysafe/98c_deploysafe_full_suite.py`, `98d_deploysafe_auprc_ci.py` |

Two files in `07_results_summary/` are retained only as the documented earlier
version of a published result and are **not** the numbers to quote:
`strict_control_apples2apples_era_fixed_partial_devonly.json` (development-pool
blocks only; the complete file supersedes it) and the `whole_main` / `whole_strict`
blocks inside `strict_control_apples2apples_era_fixed.json` (the full-cohort,
all-rows-imputed version described as superseded in the Table S3b footnote).

## Corrections since v1.0.0

Three defects in the v1.0.0 code affected shipped numbers. All three are corrected
in this release.

1. **Net-reclassification non-event sign.** `47b_bootstrap_only.py` and
   `47b_bootstrap_only_72h.py` computed the non-event component of the category-based
   NRI with the opposite sign, which roughly doubles NRI when that component is
   large and negative — as it is on the landmark view. The corrected scripts are
   shipped here. The legacy JSON each script produced is retained alongside the
   corrected one (`dynamic_oof_inference_inclprior_excl24h.json` and
   `dynamic_oof_inference_72h_inclprior_excl24h.json` next to the
   `..._nri_fixed.json` files) so the change is auditable. ΔAUC and IDI are
   unaffected by the sign convention.
2. **Estimand / pool mislabelling.** One table reported a whole-cohort fold-mean
   AUC while another reported the development-pool fold-mean AUC for the same
   model, and the two were compared across tables. The whole-cohort figures also
   trained and scored on rows the primary protocol holds out. The corrected
   comparison is the development-pool, like-for-like version in
   `strict_control_apples2apples_era_fixed.json` (`dev_main` / `dev_strict`,
   `protocols.fixed`); the superseded full-cohort values are retained only inside
   the `whole_main` / `whole_strict` blocks. Fold-mean values should be compared
   within a table, not across tables.
3. **Feature-set seed convention.** The 42- and 40-feature arms were originally
   run with the repeat seed passed through to XGBoost rather than held fixed, which
   moved the fourth decimal of the fold-mean AUC. The shipped 57-feature
   pipeline, and the 42-/40-/44-feature arms in `deploysafe_full_suite_*`, all use
   the fixed-seed convention of Table 2. The superseded 42- and 40-feature
   fold-mean values are noted in the Table S3c footnote.

Also corrected for this release, without changing any result: the era-restricted
re-run replaces the pre-era pipeline entirely (the superseded admission counts and
development pool of v1.0.0 no longer appear anywhere in this release), and the
superseded "include prior VTE, retain 24-h-window events" cohort cell was withdrawn
rather than shipped, because its cohort builder omitted 14,274 event-free prior-VTE
admissions from the control set; the corrected cell is
`sensitivity_inclprior_fixed_result.json`.

## Data requirements (not included)

- **MIMIC-IV v3.1, MIMIC-IV-Note v2.2, MIMIC-IV-ED v2.2** — available from
  [PhysioNet](https://physionet.org/) under the PhysioNet Data Use Agreement
  (credentialed access; CITI human-subjects training required). MIMIC-IV is
  de-identified; use of MIMIC-IV was approved by the Institutional Review Boards of
  the Massachusetts Institute of Technology (Protocol 0403000206) and Beth Israel
  Deaconess Medical Center (Protocol 2001P001710), and the external validation by
  the IRB of Peking Union Medical College Hospital (Approval I-26PJ1451) with a
  waiver of informed consent for retrospective de-identified data.
  **No patient-level data are redistributed in this repository.** The extraction,
  labelling and feature-building outputs (`results_vte/*.parquet`,
  `results_vte/ajm/new/data*`) are patient-level derivatives of MIMIC-IV and are
  likewise not redistributed; regenerate them from your own credentialed copy.
- **Expected local layout** (relative to the directory the scripts are run from):
  - `results_vte/` — extracted feature tables, bucket tables, `feature_sets_v2.json`,
    `model_dataset_v3_fixed.parquet`, `admissions_base.parquet`, `clinical_scores_v3_caprini.parquet`
  - `results_vte/ajm/new/data/` — inclusion-criteria cohort datasets, splits and
    landmark feature matrices for the pre-era pipeline
  - `results_vte/ajm/new/data_era/` — the same artefacts restricted to anchor years
    2008–2019, plus `era_map.parquet` (hadm_id → anchor_year_group), which is what
    the shipped scripts read
  - `results_vte/ajm/new/output_era/` — script outputs; `07_results_summary/` is a
    copy of this directory
- **Not produced by any script in this release:** the raw MIMIC extraction
  (`admissions_base.parquet`, `buckets_labs_v2.parquet`, `buckets_vitals_v2.parquet`,
  `buckets_rx_v2.parquet`, `features_static_v1.parquet`), the PE/DVT outcome
  ascertainment (`results_vte/ajm/new/data/vte_labels_*.parquet`), and the
  "fixed" re-issues of the v3 label rebuild that the era cohort builders actually
  consume (`results_vte/features_static_v3_fixed.parquet`,
  `results_vte/model_dataset_v3_fixed.parquet`).
  Steps 1 and 4 build the model dataset and the landmark matrices from those
  artefacts; the scripts that produce the artefacts themselves are not part of this
  release.
- **External validation data** — the independent tertiary-care hospital cohort in
  Beijing contains protected health information and is **not available, not
  redistributable, and not included**. The external-validation *code* is included
  (Step 9) so the analysis is auditable; it cannot be executed without a local
  export of that hospital's data.

## Environment

The era-restricted pipeline and every shipped results file were produced with:

```
interpreter: /Users/Yepsun/myenv/bin/python3   (CPython 3.14.6, macOS arm64)
```

Installed package versions used for the manuscript (`requirements.txt` pins these):

| Package | Version |
|---|---|
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |
| xgboost | 3.3.0 |
| scipy | 1.18.0 |
| torch | 2.13.0 |
| joblib | 1.5.3 |
| matplotlib | 3.11.1 |
| pyarrow | 25.0.0 |

GRU training runs on CPU or on Apple-silicon MPS. The shipped
`48e_gru_mps_primary.py` is the MPS device port of the seed protocol and
reproduced the seed-42 CPU results exactly; the CPU predecessors (48b, 48c) are
superseded by it and are not shipped. The external-validation LLM pre-screening
scripts additionally need `requests` and an API key supplied through the
environment (`DEEPSEEK_API_KEY`; the Zhipu variant reads `ZHIPU_API_KEY`).

## Reproduction order and approximate wall-clock

Approximate single-machine wall-clock times, taken from the run logs in
`results_vte/ajm/new/output_era/logs/`. Steps that were not logged are marked "—".

Prerequisites, run once before the table below: the Step 1 scripts in order
(`85d_treatments_v3.py` → `85e_clinical_scores_v3.py` → `85f_assemble_v3.py` →
`86_build_model_dataset_v3.py`, plus `85h_clinical_scores_caprini.py`) and the
Step 4 scripts (`44b_build_landmarks_v2.py`, then `46b_landmark_features_v2.py`
with `--skeleton`, `--out-full` and `--out-train` pointed at the era paths and
`--start 72` for the 72-hour views). The era restriction is applied to the cohort
datasets and split keys through `data_era/era_map.parquet` before any of the
commands below, and `85i_join_caprini_era_cohorts.py` must run after
`85h` and before the Caprini comparisons.

| # | Command | Approx. time |
|---|---|---|
| 1 | `python3 04_evaluation/sensitivity/98_ajm_sensitivity_cohorts.py --tag inclprior_excl24h …` (one call per cell of the 2×2 matrix) | ~3 min per cell (~12 min for four) |
| 2 | `python3 01_cohort/build_inclprior_cohort_era_fixed.py` | — |
| 3 | `python3 01_cohort/85i_join_caprini_era_cohorts.py` | — |
| 4 | `python3 03_models/static/39b_baseline_scores_eval_v2.py` | < 5 s |
| 5 | `python3 03_models/static/39c_baseline_scores_eval_caprini.py` | — |
| 6 | `python3 03_models/static/61_static_oof_multiseed.py` | ~3.5 min |
| 7 | `python3 03_models/static/61b_static_oof_multiseed_era_fixed.py` | ~5 min |
| 8 | `python3 04_evaluation/calibration_dca/41b_calibration_dca_v2.py` | ~1 min |
| 9 | `python3 04_evaluation/subgroups/42b_temporal_subgroup_v2.py` | ~2 min |
| 10 | `python3 04_evaluation/subgroups/60_service_subgroups_v2.py` | ~40 s |
| 11 | `python3 04_evaluation/subgroups/60b_service_subgroups_caprini.py` | ~2.5 min |
| 12 | `python3 04_evaluation/sensitivity/55b_ed_vitals_sensitivity_v2.py` | ~2.5 min |
| 13 | `python3 04_evaluation/sensitivity/50c_strict_control_apples2apples_v2.py` | ~3 min |
| 14 | `python3 04_evaluation/sensitivity/50c_strict_control_apples2apples_era_fixed.py` | ~82 min |
| 15 | `python3 04_evaluation/sensitivity/62_review_gaps.py` | ~8.5 min |
| 16 | `python3 04_evaluation/calibration_dca/57_threshold_calibration_figures_v2.py` | ~40 s |
| 17 | `python3 05_figures/58_publication_figures_v3.py` | — |
| 18 | `python3 03_models/dynamic/47b_dynamic_models_v2.py --train-view … --eval-view … --out …` | ~2 min |
| 19 | `python3 03_models/dynamic/47b_bootstrap_only.py` | ~40 s |
| 20 | `python3 03_models/dynamic/47b_dynamic_models_72h.py …` then `47b_bootstrap_only_72h.py` | ~50 s combined |
| 21 | `python3 03_models/gru/48e_gru_mps_primary.py 42` (then `43`, `44`) | ~3.3 min per seed (~10 min) |
| 22 | `python3 04_evaluation/subgroups/63_prior_stratified_auc.py` | < 1 min |
| 23 | `python3 04_evaluation/sensitivity/64_no_rx_sensitivity.py` | ~22 min |
| 24 | `python3 04_evaluation/sensitivity/64b_no_rx_dynamic.py` | ~4 min |
| 25 | `python3 04_evaluation/sensitivity/64c_no_rx_foldmean_98conv.py` | ~2.5 min |
| 26 | `python3 04_evaluation/calibration_dca/65_nested_calibration_dca_fixed.py` | ~3 min |
| 27 | `python3 04_evaluation/calibration_dca/68_nested_calibration_metrics_extra.py` | — |
| 28 | `python3 04_evaluation/nri_idi/66_dynamic_nri_idi_fixed.py` | ~1 min |
| 29 | `python3 04_evaluation/nri_idi/67_static_nri_idi_fixed.py` | ~1.5 min |
| 30 | `python3 04_evaluation/calibration_dca/69_dca_reconciliation_fixed.py` | < 1 min |
| 31 | `python3 04_evaluation/calibration_dca/70_dca_manuscript_grid_table.py` | < 1 min |
| 32 | `python3 04_evaluation/sensitivity/98b_caprini_comparison.py --tag inclprior_excl24h` (and `--tag inclprior_fixed`) | ~3.5 min per call |
| 33 | `python3 04_evaluation/deploysafe/98c_deploysafe_full_suite.py` | ~15 min (four arms + subgroups) |
| 34 | `python3 04_evaluation/deploysafe/98d_deploysafe_auprc_ci.py` | ~8.5 min |
| 35 | `python3 08_external_validation/…` | requires the external data |

`run_scripts/run_all_era.sh` (the four 2×2 cells, `39b` through `57`, and the
dynamic pair) took ~38 minutes end to end; `run_scripts/run_era.sh` (dynamic pair
+ three GRU seeds) took ~19 minutes. A full reproduction from a populated
`data_era/` is roughly 2.5–3 hours.

## Step 9 — external validation code (`08_external_validation/`)

Nineteen scripts: external cohort construction (`build_cohort.py`,
`build_cohort_ipe.py`), demographics and featurization (`parse_demographics.py`,
`build_features.py`), score computation (`compute_scores.py`), outcome
ascertainment with LLM-assisted pre-screening and human review
(`build_radiology_labels.py`, `build_radiology_llm.py`,
`build_radiology_llm_glm.py`, `llm_agreement.py`, `review_entries.py`,
`apply_human_review.py`), model freezing (`freeze_mimic_models.py`,
`freeze_mimic_models_ipe.py`, `freeze_mimic_models_deploysafe_ipe.py`), external
validation and re-runs (`run_external_validation.py`,
`run_external_validation_ipe.py`, `run_external_validation_deploysafe_ipe.py`,
`run_extval_nested_calibration_ipe.py`, `run_extval_nri_era_ipe.py`).

Notes for anyone reading or adapting this code:

- **The external patient data are absent and not redistributable.** No `.parquet`,
  `.csv`, `.jsonl`, model binary (`.joblib`) or log from that directory is
  included, and neither are the external-validation result JSONs.
- **Sanitization applied.** The published copy of `build_radiology_llm.py` reads
  its API key from the environment (`DEEPSEEK_API_KEY`) instead of the
  hard-coded key present in the working copy, matching the pattern already used in
  `build_radiology_llm_glm.py`. No other credential, password, connection string
  or database configuration was found in any of the 19 scripts.
- **Environment-specific paths remain.** These scripts carry absolute paths from
  the authoring machine (`/Users/Yepsun/…`) and one scratch path
  (`/var/folders/…/opencode/demo_all.parquet`, written by
  `parse_demographics.py` and read by `build_cohort.py`). They contain no hospital
  name, no person's name, no credential and no patient identifier; they must be
  pointed at your own working directories before the code will run. The Chinese
  column and file names (`检查诊断`, `就诊号`, `高危VTE患者/…`) are the external
  site's export schema and are required by the code.
- `apply_human_review.py` expects the reviewer-completed
  `llm_discordant_for_review.csv`; without the external export that file does not
  exist and the script cannot run.

## License

Code: MIT (see `LICENSE`). MIMIC-IV and any derivatives remain governed by the
PhysioNet Data Use Agreement and may not be redistributed; the external validation
data are not available and are not included.

## Citation

If you use this code, please cite the accompanying manuscript and the code release:

> Feng J, Li Y, Yu S, Shi J, Sun X. Analysis code for machine learning versus the
> Padua and IMPROVE scores for prediction of radiologically confirmed in-hospital
> venous thromboembolism. Zenodo, 2026. doi: [10.5281/zenodo.22168152](https://doi.org/10.5281/zenodo.22168152)

## Contact

Corresponding authors: Xuefeng Sun (sunxfer@sina.com), Juhong Shi (shijh@pumch.cn).
